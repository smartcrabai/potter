// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/bmesh/tools/bmesh_bevel.cc, source/blender/blenlib/intern/math_geom.cc, SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.

//! Blender-derived vertex meshes and output reconstruction for EDGES and VERTICES bevels.
//!
//! ADJ corners use a two-segment control mesh, cubic subdivision, and Blender's odd-segment
//! interpolation. EDGES also rebuilds manifold selected-edge strips and cutoff/terminal corner
//! faces; VERTICES rebuilds rings around closed or boundary fans.

use glam::{DMat3, DVec3, Vec3};
use serde_json::Value;
use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};

use super::bevel_boundary::{
    AngleKind, BoundProfile, Front, PRO_CIRCLE_R, PRO_LINE_R, PRO_SQUARE_IN_R, PRO_SQUARE_R,
    internal_invariant, isect_line_line,
};
use super::bevel_shading::Kind;
use super::{Mesh, Polygon, build_mesh_with_edge_identity};
use crate::error::Result;
use crate::geom::clnor::{self, NormalCodeOptions};

use crate::geom::uv_layers;
use crate::geom::{
    attribute_keys, edge_key,
    provenance::{
        CornerFloat2LayerKey, CornerFloat2Sources, Origin, PointInterpolationEvent,
        mean_value_corners,
    },
};

const BEVEL_EPSILON: f64 = 1.0e-6;
const BEVEL_EPSILON_SQ: f64 = 1.0e-12;
const BEVEL_EPSILON_ANG: f64 = 2.0 * PI / 180.0;
const CIRCLE_FULLNESS: [f64; 11] = [
    0.0, 0.559, 0.642, 0.551, 0.646, 0.624, 0.646, 0.619, 0.647, 0.639, 0.647,
];

/// Blender's `find_profile_fullness` for a non-custom profile.
pub(super) fn profile_fullness(seg: usize, profile: f64, super_r: f64) -> f64 {
    if super_r == PRO_LINE_R {
        0.0
    } else if super_r == PRO_CIRCLE_R && seg > 0 && seg <= CIRCLE_FULLNESS.len() {
        CIRCLE_FULLNESS[seg - 1]
    } else if seg.is_multiple_of(2) {
        2.4506 * profile - 0.000_003 * seg as f64 - 0.6266
    } else {
        2.3635 * profile + 0.000_152 * seg as f64 - 0.6060
    }
}

pub(super) fn power_of_two_at_least(value: usize) -> usize {
    let mut power = 1;
    while power < value {
        power *= 2;
    }
    power
}

fn superellipse_co(x: f64, r: f64, rbig: bool) -> f64 {
    if rbig {
        (1.0 - x.powf(r)).powf(1.0 / r)
    } else {
        1.0 - (1.0 - (1.0 - x).powf(r)).powf(1.0 / r)
    }
}

/// Blender's `find_superellipse_chord_endpoint`: the `x` whose superellipse point is `dtarget`
/// away from `(x0, y0)`, found by the Illinois false-position method.
fn superellipse_chord_endpoint(x0: f64, dtarget: f64, r: f64, rbig: bool) -> f64 {
    let y0 = superellipse_co(x0, r, rbig);
    let tol = 1.0e-13;
    let maxiter = 10;
    let mut xmin = (x0 + SQRT_2 / 2.0 * dtarget).min(1.0);
    let mut xmax = (x0 + dtarget).min(1.0);
    let mut ymin = superellipse_co(xmin, r, rbig);
    let ymax = superellipse_co(xmax, r, rbig);
    let mut dmaxerr = ((xmax - x0).powi(2) + (ymax - y0).powi(2)).sqrt() - dtarget;
    let mut dminerr = ((xmin - x0).powi(2) + (ymin - y0).powi(2)).sqrt() - dtarget;
    let mut xnew = xmax - dmaxerr * (xmax - xmin) / (dmaxerr - dminerr);
    let mut lastupdated_upper = true;
    for _ in 0..maxiter {
        let ynew = superellipse_co(xnew, r, rbig);
        let dnewerr = ((xnew - x0).powi(2) + (ynew - y0).powi(2)).sqrt() - dtarget;
        if dnewerr.abs() < tol {
            break;
        }
        if dnewerr < 0.0 {
            xmin = xnew;
            ymin = ynew;
            dminerr = dnewerr;
            xnew = if lastupdated_upper {
                xmax - dmaxerr * (xmax - xmin) / (dmaxerr - dminerr)
            } else {
                (dmaxerr / 2.0 * xmin - dminerr * xmax) / (dmaxerr / 2.0 - dminerr)
            };
            lastupdated_upper = false;
        } else {
            xmax = xnew;
            dmaxerr = dnewerr;
            xnew = if lastupdated_upper {
                (dmaxerr * xmin - dminerr / 2.0 * xmax) / (dmaxerr - dminerr / 2.0)
            } else {
                xmax - dmaxerr * (xmax - xmin) / (dmaxerr - dminerr)
            };
            lastupdated_upper = true;
        }
    }
    let _ = ymin;
    xnew
}

/// Blender's `find_even_superellipse_chords_general`: `seg + 1` samples of the superellipse with
/// equal chords, found by iterating the chord lengths to their average.
fn even_chords_general(seg: usize, r: f64) -> (Vec<f64>, Vec<f64>) {
    let smoothitermax = 10;
    let error_tol = 1.0e-7;
    let mut xvals = vec![0.0; seg + 1];
    let mut yvals = vec![0.0; seg + 1];
    let imax = seg.div_ceil(2) - 1;
    let seg_odd = seg % 2 == 1;
    let rbig = r > 1.0;
    let mx = if rbig {
        0.5_f64.powf(1.0 / r)
    } else {
        1.0 - 0.5_f64.powf(1.0 / r)
    };
    for i in 0..=imax {
        xvals[i] = i as f64 * mx / seg as f64 * 2.0;
        yvals[i] = superellipse_co(xvals[i], r, rbig);
    }
    yvals[0] = 1.0;
    for _ in 0..smoothitermax {
        let mut sum = 0.0;
        let mut dmin = 2.0_f64;
        let mut dmax = 0.0_f64;
        for i in 0..imax {
            let d = ((xvals[i + 1] - xvals[i]).powi(2) + (yvals[i + 1] - yvals[i]).powi(2)).sqrt();
            sum += d;
            dmax = dmax.max(d);
            dmin = dmin.min(d);
        }
        let davg = if seg_odd {
            sum += SQRT_2 / 2.0 * (yvals[imax] - xvals[imax]);
            sum / (imax as f64 + 0.5)
        } else {
            sum += ((xvals[imax] - mx).powi(2) + (yvals[imax] - mx).powi(2)).sqrt();
            sum / (imax as f64 + 1.0)
        };
        let precision_reached = (dmax - davg).partial_cmp(&error_tol)
            != Some(std::cmp::Ordering::Greater)
            && (dmin - davg).partial_cmp(&error_tol) != Some(std::cmp::Ordering::Less);
        if precision_reached {
            break;
        }
        for i in 1..=imax {
            xvals[i] = superellipse_chord_endpoint(xvals[i - 1], davg, r, rbig);
            yvals[i] = superellipse_co(xvals[i], r, rbig);
        }
    }
    if !seg_odd {
        xvals[imax + 1] = mx;
        yvals[imax + 1] = mx;
    }
    for i in imax + 1..=seg {
        yvals[i] = xvals[seg - i];
        xvals[i] = yvals[seg - i];
    }
    if !rbig {
        for i in 0..=seg {
            let temp = xvals[i];
            xvals[i] = 1.0 - yvals[i];
            yvals[i] = 1.0 - temp;
        }
    }
    (xvals, yvals)
}

/// Blender's `find_even_superellipse_chords`: the 2D profile samples for `n` segments.
pub(super) fn even_chords(n: usize, r: f64) -> (Vec<f64>, Vec<f64>) {
    let mut xvals = vec![0.0; n + 1];
    let mut yvals = vec![0.0; n + 1];
    let seg_odd = n % 2 == 1;
    let n2 = n / 2;
    if r == PRO_LINE_R {
        for i in 0..=n {
            xvals[i] = i as f64 / n as f64;
            yvals[i] = 1.0 - i as f64 / n as f64;
        }
        return (xvals, yvals);
    }
    if r == PRO_CIRCLE_R {
        let temp = FRAC_PI_2 / n as f64;
        for i in 0..=n {
            xvals[i] = (i as f64 * temp).sin();
            yvals[i] = (i as f64 * temp).cos();
        }
        return (xvals, yvals);
    }
    if r == PRO_SQUARE_IN_R {
        if seg_odd {
            let temp = 1.0 / (n2 as f64 + SQRT_2 / 2.0);
            for i in 0..=n2 {
                xvals[i] = 0.0;
                yvals[i] = 1.0 - i as f64 * temp;
                xvals[n - i] = yvals[i];
                yvals[n - i] = xvals[i];
            }
        } else {
            for i in 0..=n2 {
                xvals[i] = 0.0;
                yvals[i] = 1.0 - i as f64 / n2 as f64;
                xvals[n - i] = yvals[i];
                yvals[n - i] = xvals[i];
            }
        }
        return (xvals, yvals);
    }
    if r == PRO_SQUARE_R {
        if seg_odd {
            let temp = 1.0 / (n2 as f64 + SQRT_2 / 2.0);
            for i in 0..=n2 {
                xvals[i] = i as f64 * temp;
                yvals[i] = 1.0;
                xvals[n - i] = yvals[i];
                yvals[n - i] = xvals[i];
            }
        } else {
            for i in 0..=n2 {
                xvals[i] = i as f64 / n2 as f64;
                yvals[i] = 1.0;
                xvals[n - i] = yvals[i];
                yvals[n - i] = xvals[i];
            }
        }
        return (xvals, yvals);
    }
    even_chords_general(n, r)
}

/// Profile sampling of one bevel: Blender's `ProfileSpacing` for `seg` segments, and for the power
/// of two `seg_2 >= seg` (at least 4) that the subdivision uses.
#[derive(Clone, Debug)]
pub(super) struct ProfileSpacing {
    seg: usize,
    seg_2: usize,
    super_r: f64,
    xvals: Vec<f64>,
    yvals: Vec<f64>,
    xvals_2: Vec<f64>,
    yvals_2: Vec<f64>,
}

impl ProfileSpacing {
    pub(super) fn new(seg: usize, super_r: f64) -> Self {
        let seg_2 = power_of_two_at_least(seg).max(4);
        let (xvals, yvals) = even_chords(seg, super_r);
        let (xvals_2, yvals_2) = if seg_2 == seg {
            (xvals.clone(), yvals.clone())
        } else {
            even_chords(seg_2, super_r)
        };
        Self {
            seg,
            seg_2,
            super_r,
            xvals,
            yvals,
            xvals_2,
            yvals_2,
        }
    }
}

/// Blender's `make_unit_square_map`, as the columns applied to a planar profile point `(x, y, 0)`:
/// `c0 * x + c1 * y + c3`. `None` when the profile's control points are degenerate.
fn unit_square_map(va: DVec3, vmid: DVec3, vb: DVec3) -> Option<[DVec3; 3]> {
    let va_vmid = vmid - va;
    let vb_vmid = vmid - vb;
    if va_vmid == DVec3::ZERO || vb_vmid == DVec3::ZERO {
        return None;
    }
    let angle = va_vmid.cross(vb_vmid).length().atan2(va_vmid.dot(vb_vmid));
    if (angle - PI).abs() <= BEVEL_EPSILON_ANG {
        return None;
    }
    Some([vmid - va, vmid - vb, va + vb - vmid])
}

/// Blender's `calculate_profile_segments` without projection. The control points of a planar
/// profile lie in its plane, so Blender's projection onto the profile plane is the identity.
fn profile_segments(
    seg: usize,
    xvals: &[f64],
    yvals: &[f64],
    map: Option<[DVec3; 3]>,
    start: DVec3,
    end: DVec3,
) -> Vec<DVec3> {
    (0..=seg)
        .map(|k| {
            if k == 0 {
                start
            } else if k == seg {
                end
            } else if let Some([c0, c1, c3]) = map {
                c0 * xvals[k] + c1 * yvals[k] + c3
            } else {
                let t = k as f64 / seg as f64;
                start * (1.0 - t) + end * t
            }
        })
        .collect()
}

/// One edge profile of a bevel: the arc from `start` to `end` through `middle`.
#[derive(Clone, Debug)]
pub(super) struct Profile {
    start: DVec3,
    seg: usize,
    seg_2: usize,
    prof: Vec<DVec3>,
    prof_2: Vec<DVec3>,
}

impl Profile {
    fn new(spacing: &ProfileSpacing, start: DVec3, end: DVec3, middle: DVec3) -> Self {
        let map = if spacing.super_r == PRO_LINE_R {
            None
        } else {
            unit_square_map(start, middle, end)
        };
        let prof = profile_segments(spacing.seg, &spacing.xvals, &spacing.yvals, map, start, end);
        let prof_2 = if spacing.seg_2 == spacing.seg {
            prof.clone()
        } else {
            profile_segments(
                spacing.seg_2,
                &spacing.xvals_2,
                &spacing.yvals_2,
                map,
                start,
                end,
            )
        };
        Self {
            start,
            seg: spacing.seg,
            seg_2: spacing.seg_2,
            prof,
            prof_2,
        }
    }

    /// Blender's `get_profile_point`: point `index` of an `nseg`-segment sampling, where `nseg` is
    /// the bevel's segment count or a power of two no larger than `seg_2`.
    fn point(&self, index: usize, nseg: usize) -> DVec3 {
        if nseg == self.seg {
            self.prof[index]
        } else {
            self.prof_2[index * (self.seg_2 / nseg)]
        }
    }
}

impl Profile {
    /// The samples the front end computed for a boundary vertex's profile.
    pub(super) fn from_bound(profile: &BoundProfile, seg: usize, seg_2: usize) -> Self {
        Self {
            start: profile.start,
            seg,
            seg_2,
            prof: profile.prof_co.clone(),
            prof_2: profile.prof_co_2.clone(),
        }
    }
}

/// Blender's `VMesh`: `count` boundary vertices, each with `(seg / 2 + 1) * (seg + 1)` points.
#[derive(Clone, Debug)]
pub(super) struct VMesh {
    count: usize,
    seg: usize,
    pub(super) co: Vec<DVec3>,
}

impl VMesh {
    pub(super) fn new(count: usize, seg: usize) -> Self {
        Self {
            count,
            seg,
            co: vec![DVec3::ZERO; count * (seg / 2 + 1) * (seg + 1)],
        }
    }

    /// Blender's `mesh_vert` storage index.
    pub(super) fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i * (self.seg + 1) * (self.seg / 2 + 1) + j * (self.seg + 1) + k
    }

    fn get(&self, i: usize, j: usize, k: usize) -> DVec3 {
        self.co[self.index(i, j, k)]
    }

    fn put(&mut self, i: usize, j: usize, k: usize, co: DVec3) {
        let index = self.index(i, j, k);
        // Blender stores every VMesh coordinate in `NewVert::co`, a float[3].
        self.co[index] = co.as_vec3().as_dvec3();
    }

    /// Blender's `mesh_vert_canon`.
    pub(super) fn canon(&self, i: usize, j: usize, k: usize) -> (usize, usize, usize) {
        let n = self.count;
        let ns = self.seg;
        let ns2 = ns / 2;
        let odd = ns % 2;
        if odd == 0 && j == ns2 && k == ns2 {
            return (0, j, k);
        }
        if j < ns2 + odd && k <= ns2 {
            return (i, j, k);
        }
        if k <= ns2 {
            return ((i + n - 1) % n, k, ns - j);
        }
        ((i + 1) % n, ns - k, j)
    }

    /// The storage index of the cell each storage index shares its `BMVert` with in an ADJ mesh:
    /// its canonical cell (`vmesh_copy_equiv_verts`). Cells with `j` past the stored rows are unused.
    pub(super) fn canonical_representatives(&self) -> Vec<usize> {
        let mut representative: Vec<usize> = (0..self.co.len()).collect();
        for i in 0..self.count {
            for j in 0..=self.seg / 2 {
                for k in 0..=self.seg {
                    let (a, b, c) = self.canon(i, j, k);
                    representative[self.index(i, j, k)] = self.index(a, b, c);
                }
            }
        }
        representative
    }

    fn canon_get(&self, i: usize, j: usize, k: usize) -> DVec3 {
        let (a, b, c) = self.canon(i, j, k);
        self.get(a, b, c)
    }

    fn canon_put(&mut self, i: usize, j: usize, k: usize, co: DVec3) {
        let (a, b, c) = self.canon(i, j, k);
        self.put(a, b, c, co);
    }

    /// Blender's `is_canon`.
    fn is_canon(&self, i: usize, j: usize, k: usize) -> bool {
        let ns2 = self.seg / 2;
        if self.seg % 2 == 1 {
            j <= ns2 && k <= ns2
        } else {
            (j < ns2 && k <= ns2) || (j == ns2 && k == ns2 && i == 0)
        }
    }

    /// Blender's `vmesh_copy_equiv_verts`.
    fn copy_equiv(&mut self) {
        let ns2 = self.seg / 2;
        for i in 0..self.count {
            for j in 0..=ns2 {
                for k in 0..=self.seg {
                    if self.is_canon(i, j, k) {
                        continue;
                    }
                    let (a, b, c) = self.canon(i, j, k);
                    let co = self.get(a, b, c);
                    self.put(i, j, k, co);
                }
            }
        }
    }

    /// Blender's `vmesh_center`.
    fn center(&self) -> DVec3 {
        let ns2 = self.seg / 2;
        if self.seg % 2 == 1 {
            let mut sum = DVec3::ZERO;
            for i in 0..self.count {
                sum += self.get(i, ns2, ns2);
            }
            sum * (1.0 / self.count as f64)
        } else {
            self.get(0, ns2, ns2)
        }
    }
}

fn avg4(v0: DVec3, v1: DVec3, v2: DVec3, v3: DVec3) -> DVec3 {
    (v0 + v1 + v2 + v3) * 0.25
}

/// Blender's `sabin_gamma`: the Sabin correction for the center of an `n`-valent subdivision.
fn sabin_gamma(n: usize) -> f64 {
    if n < 3 {
        return 0.0;
    }
    match n {
        3 => return 0.065_247_584,
        4 => return 0.25,
        5 => return 0.401_983_447,
        6 => return 0.523_423_277,
        _ => {}
    }
    let k = (PI / n as f64).cos();
    let k2 = k * k;
    let k4 = k2 * k2;
    let k6 = k4 * k2;
    let y = (3.0_f64.sqrt() * (64.0 * k6 - 144.0 * k4 + 135.0 * k2 - 27.0).sqrt() + 9.0 * k)
        .powf(1.0 / 3.0);
    let x = 0.480_749_856_769_136 * y - (0.231_120_424_783_545 * (12.0 * k2 - 9.0)) / y;
    (k * x + 2.0 * k2 - 1.0) / (x * x * (k * x + 1.0))
}

/// Blender's `fill_vmesh_fracs`: cumulative fractions of ring 0 of boundary vertex `i`.
fn vmesh_fracs(vm: &VMesh, i: usize) -> Vec<f64> {
    let ns = vm.seg;
    let mut frac = vec![0.0; ns + 1];
    let mut total = 0.0_f32;
    for k in 0..ns {
        let co0 = vm.get(i, 0, k).as_vec3();
        let co1 = vm.get(i, 0, k + 1).as_vec3();
        total += (co0 - co1).length();
        frac[k + 1] = f64::from(total);
    }
    if total > 0.0 {
        for value in frac.iter_mut().skip(1) {
            *value = f64::from(*value as f32 / total);
        }
    } else {
        frac[ns] = 1.0;
    }
    frac
}

/// Blender's `fill_profile_fracs`: cumulative fractions of a profile sampled with `ns` segments.
fn profile_fracs(profile: &Profile, ns: usize) -> Vec<f64> {
    let mut frac = vec![0.0; ns + 1];
    let mut total = 0.0_f32;
    let mut co = profile.start.as_vec3();
    for k in 0..ns {
        let next = profile.point(k + 1, ns).as_vec3();
        total += (co - next).length();
        frac[k + 1] = f64::from(total);
        co = next;
    }
    if total > 0.0 {
        for value in frac.iter_mut().skip(1) {
            *value = f64::from(*value as f32 / total);
        }
    } else {
        frac[ns] = 1.0;
    }
    frac
}

/// Blender's single-precision `==` (`bmesh_bevel.cc` `interp_range`: `rest == 0` at line 4573 and
/// `*r_rest == 1.0f` at line 4580). `partial_cmp` gives C's `==` on `float`: +0 and -0 are equal,
/// and NaN is unequal to everything.
fn is_exactly(value: f32, target: f32) -> bool {
    value.partial_cmp(&target) == Some(std::cmp::Ordering::Equal)
}

/// Blender's `interp_range`: the `i` with `frac[i] <= f <= frac[i + 1]`, and the fraction of the
/// way from `frac[i]` to `frac[i + 1]`. Returns `n` when `f` lies beyond the last entry.
fn interp_range(frac: &[f64], n: usize, f: f64) -> (usize, f32) {
    let f = f as f32;
    for i in 0..n {
        let next = frac[i + 1] as f32;
        if f <= next {
            let rest = f - frac[i] as f32;
            let r = if is_exactly(rest, 0.0) {
                0.0
            } else {
                rest / (next - frac[i] as f32)
            };
            if i == n - 1 && is_exactly(r, 1.0) {
                return (n, 0.0);
            }
            return (i, r);
        }
    }
    (n, 0.0)
}

/// Blender's `interp_bilinear_quad_v3`.
fn interp_bilinear_quad(quad: [DVec3; 4], u: f32, v: f32) -> DVec3 {
    let q0 = quad[0].as_vec3();
    let q1 = quad[1].as_vec3();
    let q2 = quad[2].as_vec3();
    let q3 = quad[3].as_vec3();
    let mut result = q0 * ((1.0 - u) * (1.0 - v));
    result += q1 * (u * (1.0 - v));
    result += q2 * (u * v);
    result += q3 * ((1.0 - u) * v);
    result.as_dvec3()
}

/// Blender's `interp_vmesh`: resamples `vm_in` to `nseg` segments, matching each new point to the
/// input mesh through the profile fractions of the two adjacent boundaries.
fn interp_vmesh(vm_in: &VMesh, profiles: &[Profile], nseg: usize) -> VMesh {
    let n_bndv = vm_in.count;
    let ns_in = vm_in.seg;
    let nseg2 = nseg / 2;
    let odd = nseg % 2;
    let mut vm_out = VMesh::new(n_bndv, nseg);
    let mut prev_frac = vmesh_fracs(vm_in, n_bndv - 1);
    let mut prev_new_frac = profile_fracs(&profiles[n_bndv - 1], nseg);
    for (i, profile) in profiles[..n_bndv].iter().enumerate() {
        let frac = vmesh_fracs(vm_in, i);
        let new_frac = profile_fracs(profile, nseg);
        for j in 0..(nseg2 + odd) {
            for (k, &new_k) in new_frac[..=nseg2].iter().enumerate() {
                let (k_in, restk) = interp_range(&frac, ns_in, new_k);
                let (k_in_prev, restkprev) =
                    interp_range(&prev_frac, ns_in, prev_new_frac[nseg - j]);
                let mut j_in = ns_in - k_in_prev;
                let mut restj = -restkprev;
                if restj > -1.0e-6_f32 {
                    restj = 0.0;
                } else {
                    j_in -= 1;
                    restj += 1.0_f32;
                }
                let co = if restj < 1.0e-6_f32 && restk < 1.0e-6_f32 {
                    vm_in.canon_get(i, j_in, k_in)
                } else {
                    let j0inc = usize::from(!(restj < 1.0e-6_f32 || j_in == ns_in));
                    let k0inc = usize::from(!(restk < 1.0e-6_f32 || k_in == ns_in));
                    let quad = [
                        vm_in.canon_get(i, j_in, k_in),
                        vm_in.canon_get(i, j_in, k_in + k0inc),
                        vm_in.canon_get(i, j_in + j0inc, k_in + k0inc),
                        vm_in.canon_get(i, j_in + j0inc, k_in),
                    ];
                    interp_bilinear_quad(quad, restk, restj)
                };
                vm_out.put(i, j, k, co);
            }
        }
        prev_frac = frac;
        prev_new_frac = new_frac;
    }
    if nseg.is_multiple_of(2) {
        vm_out.put(0, nseg2, nseg2, vm_in.center());
    }
    vm_out.copy_equiv();
    vm_out
}

/// Blender's `cubic_subdiv`: one Catmull-Clark step on the vertex mesh with the boundary rules of
/// Levin (1999). Like Blender, it writes the smoothed boundary back into `vm_in`.
fn cubic_subdiv(vm_in: &mut VMesh, profiles: &[Profile], custom: bool) -> VMesh {
    let n_boundary = vm_in.count;
    let ns_in = vm_in.seg;
    let ns_in2 = ns_in / 2;
    let ns_out = 2 * ns_in;
    let mut vm_out = VMesh::new(n_boundary, ns_out);

    for i in 0..n_boundary {
        vm_out.put(i, 0, 0, vm_in.get(i, 0, 0));
        for k in 1..ns_in {
            let co = vm_in.get(i, 0, k);
            let co = if custom {
                co
            } else {
                let acc = (vm_in.get(i, 0, k - 1) + vm_in.get(i, 0, k + 1)) + co * -2.0;
                co + acc * (-1.0 / 6.0)
            };
            vm_out.canon_put(i, 0, 2 * k, co);
        }
    }
    for (i, profile) in profiles.iter().enumerate().take(n_boundary) {
        for k in (1..ns_out).step_by(2) {
            let co = profile.point(k, ns_out);
            let co = if custom {
                co
            } else {
                let acc =
                    (vm_out.canon_get(i, 0, k - 1) + vm_out.canon_get(i, 0, k + 1)) + co * -2.0;
                co + acc * (-1.0 / 6.0)
            };
            vm_out.canon_put(i, 0, k, co);
        }
    }
    vm_out.copy_equiv();

    for i in 0..n_boundary {
        for k in 0..ns_in {
            let co = vm_out.get(i, 0, 2 * k);
            vm_in.put(i, 0, k, co);
        }
    }
    vm_in.copy_equiv();

    for i in 0..n_boundary {
        for j in 0..ns_in2 {
            for k in 0..ns_in2 {
                let co = avg4(
                    vm_in.get(i, j, k),
                    vm_in.get(i, j, k + 1),
                    vm_in.get(i, j + 1, k),
                    vm_in.get(i, j + 1, k + 1),
                );
                vm_out.put(i, 2 * j + 1, 2 * k + 1, co);
            }
        }
    }
    for i in 0..n_boundary {
        for j in 0..ns_in2 {
            for k in 1..=ns_in2 {
                let co = avg4(
                    vm_in.get(i, j, k),
                    vm_in.get(i, j + 1, k),
                    vm_out.canon_get(i, 2 * j + 1, 2 * k - 1),
                    vm_out.canon_get(i, 2 * j + 1, 2 * k + 1),
                );
                vm_out.put(i, 2 * j + 1, 2 * k, co);
            }
        }
    }
    for i in 0..n_boundary {
        for j in 1..ns_in2 {
            for k in 0..ns_in2 {
                let co = avg4(
                    vm_in.get(i, j, k),
                    vm_in.get(i, j, k + 1),
                    vm_out.canon_get(i, 2 * j - 1, 2 * k + 1),
                    vm_out.canon_get(i, 2 * j + 1, 2 * k + 1),
                );
                vm_out.put(i, 2 * j, 2 * k + 1, co);
            }
        }
    }

    let gamma = 0.25;
    let beta = -gamma;
    for i in 0..n_boundary {
        for j in 1..ns_in2 {
            for k in 1..=ns_in2 {
                let co1 = avg4(
                    vm_out.canon_get(i, 2 * j, 2 * k - 1),
                    vm_out.canon_get(i, 2 * j, 2 * k + 1),
                    vm_out.canon_get(i, 2 * j - 1, 2 * k),
                    vm_out.canon_get(i, 2 * j + 1, 2 * k),
                );
                let co2 = avg4(
                    vm_out.canon_get(i, 2 * j - 1, 2 * k - 1),
                    vm_out.canon_get(i, 2 * j + 1, 2 * k - 1),
                    vm_out.canon_get(i, 2 * j - 1, 2 * k + 1),
                    vm_out.canon_get(i, 2 * j + 1, 2 * k + 1),
                );
                let co = co1 + co2 * beta + vm_in.get(i, j, k) * gamma;
                vm_out.put(i, 2 * j, 2 * k, co);
            }
        }
    }
    vm_out.copy_equiv();

    let gamma = sabin_gamma(n_boundary);
    let beta = -gamma;
    let mut co1 = DVec3::ZERO;
    let mut co2 = DVec3::ZERO;
    for i in 0..n_boundary {
        co1 += vm_out.get(i, ns_in, ns_in - 1);
        co2 += vm_out.get(i, ns_in - 1, ns_in - 1);
        co2 += vm_out.get(i, ns_in - 1, ns_in + 1);
    }
    let co = co1 * (1.0 / n_boundary as f64)
        + co2 * (beta / (2.0 * n_boundary as f64))
        + vm_in.get(0, ns_in2, ns_in2) * gamma;
    for i in 0..n_boundary {
        vm_out.put(i, ns_in, ns_in, co);
    }

    for (i, profile) in profiles.iter().enumerate().take(n_boundary) {
        let inext = (i + 1) % n_boundary;
        for k in 0..=ns_out {
            let co = profile.point(k, ns_out);
            vm_out.put(i, 0, k, co);
            if k >= ns_in && k < ns_out {
                vm_out.put(inext, ns_out - k, 0, co);
            }
        }
    }
    vm_out
}

/// Blender's `snap_to_superellipsoid`, including its square-profile midline cases.
fn snap_to_superellipsoid(mut co: DVec3, r: f64, midline: bool) -> DVec3 {
    if r == PRO_CIRCLE_R {
        return co.normalize_or_zero();
    }
    let a = co.x.max(0.0);
    let b = co.y.max(0.0);
    let c = co.z.max(0.0);
    if r == PRO_SQUARE_R || r == PRO_SQUARE_IN_R {
        let mut x = a.min(1.0);
        let mut y = b.min(1.0);
        if r == PRO_SQUARE_R {
            if 1.0 - x < 1.0 - y {
                x = 1.0;
                if midline {
                    y = 1.0;
                }
            } else {
                y = 1.0;
                if midline {
                    x = 1.0;
                }
            }
        } else if x < y {
            x = 0.0;
            if midline {
                y = 0.0;
            }
        } else {
            y = 0.0;
            if midline {
                x = 0.0;
            }
        }
        co.x = x;
        co.y = y;
        co.z = 0.0;
        return co;
    }
    let rinv = 1.0 / r;
    if a == 0.0 {
        if b == 0.0 {
            DVec3::new(0.0, 0.0, c.powf(rinv))
        } else {
            let y = (1.0 / (1.0 + (c / b).powf(r))).powf(rinv);
            DVec3::new(0.0, y, c * y / b)
        }
    } else {
        let x = (1.0 / (1.0 + (b / a).powf(r) + (c / a).powf(r))).powf(rinv);
        DVec3::new(x, b * x / a, c * x / a)
    }
}

/// Blender's `make_cube_corner_square`: the octant of a square profile, which needs no subdivision.
fn cube_corner_square(seg: usize) -> VMesh {
    let ns2 = seg / 2;
    let mut vm = VMesh::new(3, seg);
    for i in 0..3 {
        for j in 0..=ns2 {
            for k in 0..=ns2 {
                if !vm.is_canon(i, j, k) {
                    continue;
                }
                let mut co = DVec3::ZERO;
                co[i] = 1.0;
                co[(i + 1) % 3] = k as f64 * 2.0 / seg as f64;
                co[(i + 2) % 3] = j as f64 * 2.0 / seg as f64;
                vm.put(i, j, k, co);
            }
        }
    }
    vm.copy_equiv();
    vm
}

/// Blender's `make_cube_corner_square_in`: inward-square corner control mesh.
fn cube_corner_square_in_vmesh(seg: usize) -> VMesh {
    let ns2 = seg / 2;
    let step = if seg.is_multiple_of(2) {
        2.0 / seg as f64
    } else {
        2.0 / (2.0 * ns2 as f64 + SQRT_2)
    };
    let mut vm = VMesh::new(3, seg);
    for i in 0..3 {
        for k in 0..=ns2 {
            let mut first = DVec3::ZERO;
            first[i] = 1.0 - k as f64 * step;
            vm.put(i, 0, k, first);
            let mut second = DVec3::ZERO;
            second[(i + 1) % 3] = 1.0 - k as f64 * step;
            vm.put(i, 0, seg - k, second);
        }
    }
    vm
}

/// Blender's `make_cube_corner_adj_vmesh`: the unit octant used by tri-corner `VMeshes`.
fn cube_corner_adj_vmesh(spacing: &ProfileSpacing) -> VMesh {
    let nseg = spacing.seg;
    let r = spacing.super_r;
    if r == PRO_SQUARE_R {
        return cube_corner_square(nseg);
    }
    if r == PRO_SQUARE_IN_R {
        return cube_corner_square_in_vmesh(nseg);
    }
    let mut vm0 = VMesh::new(3, 2);
    let mut profiles = Vec::with_capacity(3);
    for i in 0..3 {
        let mut start = DVec3::ZERO;
        start[i] = 1.0;
        let mut end = DVec3::ZERO;
        end[(i + 1) % 3] = 1.0;
        let mut middle = DVec3::ZERO;
        middle[i] = 1.0;
        middle[(i + 1) % 3] = 1.0;
        let profile = Profile::new(spacing, start, end, middle);
        vm0.put(i, 0, 0, start);
        vm0.put(i, 0, 1, profile.point(1, 2));
        profiles.push(profile);
    }
    let mut center = DVec3::splat(1.0 / 3.0_f64.sqrt());
    if nseg > 2 {
        if r > 1.5 {
            center *= 1.4;
        } else if r < 0.75 {
            center *= 0.6;
        }
    }
    vm0.put(0, 1, 1, center);
    vm0.copy_equiv();

    let mut vm1 = vm0;
    while vm1.seg < nseg {
        vm1 = cubic_subdiv(&mut vm1, &profiles, false);
    }
    if vm1.seg != nseg {
        vm1 = interp_vmesh(&vm1, &profiles, nseg);
    }
    let ns2 = nseg / 2;
    for i in 0..3 {
        for j in 0..=ns2 {
            for k in 0..=nseg {
                let co = snap_to_superellipsoid(vm1.get(i, j, k), r, false);
                vm1.put(i, j, k, co);
            }
        }
    }
    vm1
}

/// Blender's `tri_corner_adj_vmesh`: the cube-corner octant mapped onto the corner.
pub(super) fn cube_corner_vmesh(
    spacing: &ProfileSpacing,
    bounds: [DVec3; 3],
    vertex: DVec3,
) -> VMesh {
    let (va, vb, vc, vd) = (bounds[0], bounds[1], bounds[2], vertex);
    let cols = [
        (va - vb - vc + vd) * 0.5,
        (vb - va - vc + vd) * 0.5,
        (vc - va - vb + vd) * 0.5,
        (va + vb + vc - vd) * 0.5,
    ];
    let mut vm = cube_corner_adj_vmesh(spacing);
    let ns = spacing.seg;
    let ns2 = ns / 2;
    for i in 0..3 {
        for j in 0..=ns2 {
            for k in 0..=ns {
                let p = vm.get(i, j, k);
                let co = cols[0] * p.x + cols[1] * p.y + cols[2] * p.z + cols[3];
                vm.put(i, j, k, co);
            }
        }
    }
    vm
}

/// Blender's `adj_vmesh` for the general case: a two-segment control mesh subdivided to `seg`.
pub(super) fn general_vmesh(
    seg: usize,
    fullness: f64,
    custom: bool,
    profiles: &[Profile],
    vertex: DVec3,
) -> VMesh {
    let n_bndv = profiles.len();
    let mut vm0 = VMesh::new(n_bndv, 2);
    let mut boundverts_center = DVec3::ZERO;
    for (i, profile) in profiles.iter().enumerate() {
        vm0.put(i, 0, 0, profile.start);
        vm0.put(i, 0, 1, profile.point(1, 2));
        boundverts_center += profile.start;
    }
    boundverts_center *= 1.0 / n_bndv as f64;
    let center_direction = vertex - boundverts_center;
    if center_direction.length_squared() > BEVEL_EPSILON_SQ {
        if custom {
            let negative_fullest = boundverts_center * 2.0 - vertex;
            vm0.put(
                0,
                1,
                1,
                negative_fullest + center_direction * (2.0 * fullness),
            );
        } else {
            vm0.put(0, 1, 1, boundverts_center + center_direction * fullness);
        }
    } else {
        vm0.put(0, 1, 1, boundverts_center);
    }
    vm0.copy_equiv();

    let mut vm1 = vm0;
    loop {
        vm1 = cubic_subdiv(&mut vm1, profiles, custom);
        if vm1.seg >= seg {
            break;
        }
    }
    if vm1.seg != seg {
        vm1 = interp_vmesh(&vm1, profiles, seg);
    }
    vm1
}

type WeldProfileSamples = (usize, usize, Vec<DVec3>, Vec<DVec3>, f64, f64);

/// Blender's weld case recalculates the two beveled boundary profiles after moving their planes.
pub(super) fn weld_profile_samples(front: &Front, vertex: usize) -> Option<WeldProfileSamples> {
    let bev = front.verts[vertex].as_ref()?;
    if bev.selcount != 2 || bev.ring.len() != 2 {
        return None;
    }
    let mut weld_bounds = bev
        .ring
        .iter()
        .copied()
        .filter_map(|bound| front.bounds[bound].ebev.map(|edge| (bound, edge)));
    let (weld1, edge1) = weld_bounds.next()?;
    let (weld2, edge2) = weld_bounds.next()?;
    if weld_bounds.next().is_some() {
        return None;
    }
    let mut pro1 = weld_profile_params(front, vertex, weld1, edge1);
    let mut pro2 = weld_profile_params(front, vertex, weld2, edge2);
    move_weld_profile_planes(front, vertex, &mut pro1, &mut pro2);
    let super_r1 = pro1.super_r;
    let super_r2 = pro2.super_r;
    let samples1 = recalculated_profile_samples(front, weld1, &pro1);
    let samples2 = recalculated_profile_samples(front, weld2, &pro2);
    Some((weld1, weld2, samples1, samples2, super_r1, super_r2))
}
/// Blender's `bevel_vert_two_edges` uses the first boundary's projection-free profile.
pub(super) fn two_edge_vertex_samples(front: &Front, vertex: usize) -> Option<Vec<DVec3>> {
    if !front.p.affect_vertices {
        return None;
    }
    let bev = front.verts[vertex].as_ref()?;
    if bev.ring.len() != 2 {
        return None;
    }
    let first = bev.ring[0];
    let second = bev.ring[1];
    let mut profile = front.bounds[first].profile.clone();
    profile.super_r = front.p.pro_super_r;
    profile.start = front.bounds[first].co;
    profile.end = front.bounds[second].co;
    profile.middle = front.geo.points[vertex];
    profile.plane_co = DVec3::ZERO;
    profile.plane_no = DVec3::ZERO;
    profile.proj_dir = DVec3::ZERO;
    Some(recalculated_profile_samples(front, first, &profile))
}

fn vmesh_f32_add(a: DVec3, b: DVec3) -> DVec3 {
    (a.as_vec3() + b.as_vec3()).as_dvec3()
}

fn vmesh_f32_sub(a: DVec3, b: DVec3) -> DVec3 {
    (a.as_vec3() - b.as_vec3()).as_dvec3()
}

fn vmesh_f32_mul(a: DVec3, scale: f64) -> DVec3 {
    (a.as_vec3() * scale as f32).as_dvec3()
}

fn vmesh_f32_dot(a: DVec3, b: DVec3) -> f64 {
    f64::from(a.as_vec3().dot(b.as_vec3()))
}

fn vmesh_f32_cross(a: DVec3, b: DVec3) -> DVec3 {
    a.as_vec3().cross(b.as_vec3()).as_dvec3()
}

fn vmesh_f32_length(a: DVec3) -> f64 {
    f64::from(a.as_vec3().length())
}

fn vmesh_f32_normalize(a: DVec3) -> DVec3 {
    a.as_vec3().normalize_or_zero().as_dvec3()
}

fn vmesh_nearly_parallel(a: DVec3, b: DVec3) -> bool {
    let a = vmesh_f32_normalize(a);
    let b = vmesh_f32_normalize(b);
    let dot = vmesh_f32_dot(a, b);
    let angle = if dot >= 0.0 {
        let difference = vmesh_f32_length(vmesh_f32_sub(a, b)) * 0.5;
        f64::from(2.0_f32 * (difference as f32).clamp(-1.0, 1.0).asin())
    } else {
        let opposite = vmesh_f32_mul(b, -1.0);
        let difference = vmesh_f32_length(vmesh_f32_sub(a, opposite)) * 0.5;
        f64::from(std::f32::consts::PI - 2.0_f32 * (difference as f32).clamp(-1.0, 1.0).asin())
    };
    let epsilon = f64::from(2.0_f32 * std::f32::consts::PI / 180.0_f32);
    angle.abs() < epsilon || (f64::from(std::f32::consts::PI) - angle).abs() < epsilon
}

fn source_edge_line(front: &Front, edge: usize) -> (DVec3, DVec3) {
    let edge = &front.edges[edge];
    let (a, b) = front.geo.edge_order[edge.edge_id];
    (front.geo.points[a], front.geo.points[b])
}

/// Blender's `set_profile_params`, applied at the late weld-profile construction point.
fn weld_profile_params(front: &Front, vertex: usize, bound_id: usize, edge: usize) -> BoundProfile {
    let bound = &front.bounds[bound_id];
    let start = bound.co;
    let end = front.bounds[bound.next].co;
    let mut profile = bound.profile.clone();
    let vertex_co = front.geo.points[vertex];
    profile.super_r = front.p.pro_super_r;
    let (edge_v1, edge_v2) = source_edge_line(front, edge);
    profile.proj_dir = vmesh_f32_normalize(vmesh_f32_sub(edge_v1, edge_v2));
    if front.edges[edge].is_rev {
        profile.proj_dir = vmesh_f32_mul(profile.proj_dir, -1.0);
    }
    let (kind, middle, _) = isect_line_line(edge_v1, edge_v2, start, end);
    profile.middle = if kind == 0 { edge_v1 } else { middle };
    profile.start = start;
    profile.end = end;

    let d1 = vmesh_f32_normalize(vmesh_f32_sub(profile.middle, start));
    let d2 = vmesh_f32_normalize(vmesh_f32_sub(profile.middle, end));
    profile.plane_no = vmesh_f32_normalize(vmesh_f32_cross(d1, d2));
    let mut do_linear_interp = false;
    if vmesh_nearly_parallel(d1, d2) {
        profile.middle = vertex_co;
        let previous = front.edges[edge].prev;
        let next = front.edges[edge].next;
        if front.edges[previous].is_bev
            && front.edges[next].is_bev
            && front.verts[vertex]
                .as_ref()
                .is_some_and(|bev| bev.selcount >= 3)
        {
            let (prev_v1, prev_v2) = source_edge_line(front, previous);
            let (next_v1, next_v2) = source_edge_line(front, next);
            let d3 = vmesh_f32_normalize(vmesh_f32_sub(prev_v1, prev_v2));
            let d4 = vmesh_f32_normalize(vmesh_f32_sub(next_v1, next_v2));
            if vmesh_nearly_parallel(d3, d4) {
                profile.middle = vmesh_f32_mul(vmesh_f32_add(start, end), 0.5);
                do_linear_interp = true;
            } else {
                let (kind, meet, _) =
                    isect_line_line(start, vmesh_f32_add(start, d3), end, vmesh_f32_add(end, d4));
                if kind != 0 {
                    profile.middle = meet;
                } else {
                    profile.middle = vmesh_f32_mul(vmesh_f32_add(start, end), 0.5);
                    do_linear_interp = true;
                }
            }
        }
        profile.end = end;
        let d1 = vmesh_f32_normalize(vmesh_f32_sub(profile.middle, start));
        let d2 = vmesh_f32_normalize(vmesh_f32_sub(profile.middle, end));
        profile.plane_no = vmesh_f32_normalize(vmesh_f32_cross(d1, d2));
        if vmesh_nearly_parallel(d1, d2) {
            do_linear_interp = true;
        } else {
            profile.proj_dir = profile.plane_no;
        }
    }
    profile.plane_co = start;
    if do_linear_interp {
        profile.super_r = PRO_LINE_R;
        profile.start = start;
        profile.end = end;
        profile.middle = vmesh_f32_mul(vmesh_f32_add(start, end), 0.5);
        profile.plane_co = DVec3::ZERO;
        profile.plane_no = DVec3::ZERO;
        profile.proj_dir = DVec3::ZERO;
    }
    profile
}

/// Blender's `move_weld_profile_planes` for the two recalculated weld profiles.
fn move_weld_profile_planes(
    front: &Front,
    vertex: usize,
    first: &mut BoundProfile,
    second: &mut BoundProfile,
) {
    if first.proj_dir == DVec3::ZERO || second.proj_dir == DVec3::ZERO {
        return;
    }
    let vertex = front.geo.points[vertex];
    let d1 = vmesh_f32_sub(vertex, first.start);
    let d2 = vmesh_f32_sub(vertex, second.start);
    let normal = vmesh_f32_cross(d1, d2);
    let length = vmesh_f32_length(normal);
    let normal = vmesh_f32_normalize(normal);
    let raw_normal1 = vmesh_f32_cross(d1, first.proj_dir);
    let raw_normal2 = vmesh_f32_cross(d2, second.proj_dir);
    let length1 = vmesh_f32_length(raw_normal1);
    let length2 = vmesh_f32_length(raw_normal2);
    let normal1 = vmesh_f32_normalize(raw_normal1);
    let normal2 = vmesh_f32_normalize(raw_normal2);
    if length != 0.0 && (length1 != 0.0 || length2 != 0.0) {
        let dot1 = vmesh_f32_dot(normal, normal1).abs();
        let dot2 = vmesh_f32_dot(normal, normal2).abs();
        if (dot1 - 1.0).abs() > BEVEL_EPSILON {
            first.plane_no = normal;
        }
        if (dot2 - 1.0).abs() > BEVEL_EPSILON {
            second.plane_no = normal;
        }
    }
    first.special_params = true;
    second.special_params = true;
}

fn recalculated_profile_samples(
    front: &Front,
    bound_id: usize,
    profile: &BoundProfile,
) -> Vec<DVec3> {
    let seg = front.p.seg;
    if seg == 1 {
        return Vec::new();
    }
    let bound = &front.bounds[bound_id];
    let miter = front.p.custom.is_some() && (bound.is_arc_start || bound.is_patch_start);
    let reversed = front.p.custom.is_some() && !bound.is_profile_start && !miter;
    let spacing = if miter {
        front
            .pro_spacing_miter
            .as_ref()
            .unwrap_or(&front.pro_spacing)
    } else {
        &front.pro_spacing
    };
    let map = if front.p.custom.is_none() && profile.super_r == PRO_LINE_R {
        None
    } else {
        unit_square_map(profile.start, profile.middle, profile.end)
    };
    let sample = |nseg: usize, xvals: &[f64], yvals: &[f64]| -> Vec<DVec3> {
        (0..=nseg)
            .map(|k| {
                let mut co = if k == 0 {
                    profile.start
                } else if k == nseg {
                    profile.end
                } else if let Some([c0, c1, c3]) = map {
                    let (x, y) = if reversed {
                        (yvals[nseg - k], xvals[nseg - k])
                    } else {
                        (xvals[k], yvals[k])
                    };
                    vmesh_f32_add(
                        vmesh_f32_add(vmesh_f32_mul(c0, x), vmesh_f32_mul(c1, y)),
                        c3,
                    )
                } else {
                    let t = k as f64 / nseg as f64;
                    vmesh_f32_add(
                        vmesh_f32_mul(profile.start, 1.0 - t),
                        vmesh_f32_mul(profile.end, t),
                    )
                };
                if profile.proj_dir != DVec3::ZERO {
                    let direction = profile.proj_dir;
                    let denominator = vmesh_f32_dot(profile.plane_no, direction);
                    if denominator.abs() >= f64::from(f32::EPSILON) {
                        let numerator =
                            vmesh_f32_dot(profile.plane_no, vmesh_f32_sub(profile.plane_co, co));
                        let t = f64::from(numerator as f32 / denominator as f32);
                        co = vmesh_f32_add(co, vmesh_f32_mul(direction, t));
                    }
                }
                co
            })
            .collect()
    };
    sample(seg, &spacing.xvals, &spacing.yvals)
}

/// Closest point of the segment `a`-`b` to `point` (Blender's `closest_to_line_segment_v3`).
pub(super) fn closest_on_segment(point: DVec3, a: DVec3, b: DVec3) -> DVec3 {
    let direction = b - a;
    let length_squared = direction.length_squared();
    if length_squared == 0.0 {
        return a;
    }
    let t = ((point - a).dot(direction) / length_squared).clamp(0.0, 1.0);
    a + direction * t
}
/// Full Blender `make_unit_square_map` frame, including its normal column for 3D projection.
fn unit_square_frame(va: DVec3, vmid: DVec3, vb: DVec3) -> Option<[DVec3; 4]> {
    let va_vmid = vmid - va;
    let vb_vmid = vmid - vb;
    if va_vmid == DVec3::ZERO || vb_vmid == DVec3::ZERO {
        return None;
    }
    let angle = va_vmid.cross(vb_vmid).length().atan2(va_vmid.dot(vb_vmid));
    if (angle - PI).abs() <= BEVEL_EPSILON_ANG {
        return None;
    }
    let normal = vb_vmid.cross(va_vmid).normalize_or_zero();
    let vd = va - vb_vmid + normal;
    Some([vmid - va, vmid - vb, vmid + vd - va - vb, va + vb - vmid])
}

fn lerp3(a: DVec3, b: DVec3, t: f64) -> DVec3 {
    a * (1.0 - t) + b * t
}

fn edge_line(front: &Front, edge: usize) -> (DVec3, DVec3) {
    let edge = &front.edges[edge];
    (front.geo.points[edge.vert], front.geo.points[edge.other])
}

/// Blender's `square_out_adj_vmesh` for the even-segment square-profile path.
pub(super) fn square_out_adj_vmesh(front: &Front, vertex: usize) -> Result<VMesh> {
    let bev = front.verts[vertex]
        .as_ref()
        .ok_or_else(|| internal_invariant("square-out vertex not beveled"))?;
    let ring = &bev.ring;
    let count = ring.len();
    let seg = front.p.seg;
    let ns2 = seg / 2;
    let ns2inv = 1.0 / ns2 as f64;
    let mut vm = VMesh::new(count, seg);
    let mut centerline = vec![DVec3::ZERO; count * (ns2 + 1)];
    let mut cset = vec![false; count];
    let center_index = |i: usize, j: usize| i * (ns2 + 1) + j;

    let mut i = 0;
    while i < count {
        let bound = &front.bounds[ring[i]];
        let e1 = bound.efirst;
        let e2 = bound.elast;
        let angle_kind = match (e1, e2) {
            (Some(e1), Some(e2)) => front.edges_angle_kind(e1, e2, vertex),
            _ => AngleKind::Straight,
        };
        if bound.is_patch_start {
            let next = &front.bounds[bound.next];
            centerline[center_index(i, 0)] = (bound.co + next.co) * 0.5;
            cset[i] = true;
            let after = &front.bounds[next.next];
            centerline[center_index(i + 1, 0)] = (next.co + after.co) * 0.5;
            cset[i + 1] = true;
            i += 3;
            continue;
        }
        if bound.is_arc_start {
            centerline[center_index(i, 0)] = bound.profile.middle;
            cset[i] = true;
            i += 2;
            continue;
        }
        if angle_kind == AngleKind::Smaller
            && let (Some(e1), Some(e2)) = (e1, e2)
        {
            let (e1a, e1b) = edge_line(front, e1);
            let (e2a, e2b) = edge_line(front, e2);
            let co1 = bound.co + (e1a - e1b);
            let co2 = bound.co + (e2a - e2b);
            let (kind1, v1co, _) = isect_line_line(e1a, e1b, bound.co, co2);
            let (kind2, v2co, _) = isect_line_line(e2a, e2b, bound.co, co1);
            let iprev = (i + count - 1) % count;
            let center = front.geo.points[vertex];
            if kind2 != 0 {
                let slot = center_index(i, 0);
                if cset[i] {
                    if centerline[slot].distance_squared(center) > v2co.distance_squared(center) {
                        centerline[slot] = v2co;
                    }
                } else {
                    centerline[slot] = v2co;
                    cset[i] = true;
                }
            }
            if kind1 != 0 {
                let slot = center_index(iprev, 0);
                if cset[iprev] {
                    if centerline[slot].distance_squared(center) > v1co.distance_squared(center) {
                        centerline[slot] = v1co;
                    }
                } else {
                    centerline[slot] = v1co;
                    cset[iprev] = true;
                }
            }
        }
        i += 1;
    }

    for i in 0..count {
        if cset[i] {
            continue;
        }
        let bound = &front.bounds[ring[i]];
        let prev = &front.bounds[bound.prev];
        let next = &front.bounds[bound.next];
        let co1 = bound.co;
        let co2 = next.co;
        if let Some(edge) = next.efirst {
            let (e1, e2) = edge_line(front, edge);
            if prev.is_arc_start && next.is_arc_start {
                let (kind, meet, _) = isect_line_line(e1, e2, co1, co2);
                if kind != 0 {
                    centerline[center_index(i, 0)] = meet;
                    cset[i] = true;
                }
            } else {
                let point = if prev.is_arc_start { co1 } else { co2 };
                centerline[center_index(i, 0)] = closest_on_segment(point, e1, e2);
                cset[i] = true;
            }
        }
        if !cset[i] {
            centerline[center_index(i, 0)] = (co1 + co2) * 0.5;
            cset[i] = true;
        }
    }

    let center = front.geo.points[vertex];
    for i in 0..count {
        let start = centerline[center_index(i, 0)];
        for j in 1..=ns2 {
            centerline[center_index(i, j)] = lerp3(start, center, j as f64 * ns2inv);
        }
    }
    for i in 0..count {
        let bound = &front.bounds[ring[i]];
        let co = bound.co;
        let previous = (i + count - 1) % count;
        let previous_center = centerline[center_index(previous, 0)];
        for j in 0..ns2 {
            vm.put(i, j, 0, lerp3(co, previous_center, j as f64 * ns2inv));
        }
        let current_center = centerline[center_index(i, 0)];
        for k in 1..=ns2 {
            vm.put(i, 0, k, lerp3(co, current_center, k as f64 * ns2inv));
        }
    }
    vm.put(0, ns2, ns2, center);
    vm.copy_equiv();

    for i in 0..count {
        let previous = (i + count - 1) % count;
        for j in 1..ns2 {
            for k in 1..=ns2 {
                let (kind, meet1, meet2) = isect_line_line(
                    vm.get(i, 0, k),
                    centerline[center_index(previous, k)],
                    vm.get(i, j, 0),
                    centerline[center_index(i, j)],
                );
                let co = match kind {
                    0 => lerp3(
                        vm.get(i, 0, k),
                        centerline[center_index(previous, k)],
                        j as f64 * ns2inv,
                    ),
                    1 => meet1,
                    _ => (meet1 + meet2) * 0.5,
                };
                vm.put(i, j, k, co);
            }
        }
    }
    vm.copy_equiv();
    Ok(vm)
}

/// Blender's `snap_to_pipe_profile`, including the square-profile midline rule.
fn snap_to_pipe_profile(
    co: DVec3,
    profile: &BoundProfile,
    pipe_edge: usize,
    front: &Front,
    midline: bool,
) -> DVec3 {
    if profile.start.abs_diff_eq(profile.end, 1.0e-6) {
        return profile.start;
    }
    let (edge_v1, edge_v2) = edge_line(front, pipe_edge);
    let normal = edge_v1 - edge_v2;
    let projected = |point: DVec3| {
        let t = (point - co).dot(normal) / normal.length_squared();
        point - normal * t
    };
    let start = projected(profile.start);
    let end = projected(profile.end);
    let middle = projected(profile.middle);
    if let Some([c0, c1, c2, c3]) = unit_square_frame(start, middle, end) {
        let basis = DMat3::from_cols(c0, c1, c2);
        if basis.determinant() != 0.0 {
            let mut point = basis.inverse() * (co - c3);
            point = snap_to_superellipsoid(point, profile.super_r, midline);
            return c0 * point.x + c1 * point.y + c2 * point.z + c3;
        }
    }
    closest_on_segment(co, start, end)
}

/// Blender's `pipe_adj_vmesh`: snap the ADJ grid to the pipe profile.
pub(super) fn pipe_adj_vmesh(
    mut vm: VMesh,
    front: &Front,
    vertex: usize,
    vpipe_bound_index: usize,
) -> Result<VMesh> {
    let bev = front.verts[vertex]
        .as_ref()
        .ok_or_else(|| internal_invariant("pipe vertex not beveled"))?;
    let ring = &bev.ring;
    let n_bndv = ring.len();
    let ns = vm.seg;
    let half_ns = ns / 2;
    let vpipe = &front.bounds[vpipe_bound_index];
    let ipipe1 = ring
        .iter()
        .position(|bound| *bound == vpipe_bound_index)
        .ok_or_else(|| internal_invariant("pipe not in ring"))?;
    let second_pipe = front.bounds[vpipe.next].next;
    let ipipe2 = ring
        .iter()
        .position(|bound| *bound == second_pipe)
        .ok_or_else(|| internal_invariant("second pipe not in ring"))?;
    let pipe_edge = vpipe
        .ebev
        .ok_or_else(|| internal_invariant("pipe boundary has no edge"))?;
    let custom = front.p.custom.is_some();

    for i in 0..n_bndv {
        for j in 1..=half_ns {
            for k in 0..=half_ns {
                if !vm.is_canon(i, j, k) {
                    continue;
                }
                let co = if custom {
                    let (profile1, profile2, factor) = if i == ipipe1 || i == ipipe2 {
                        if n_bndv == 3 && i == ipipe1 {
                            let ring = j.max(k);
                            let first = vm.get(i, ring, 0);
                            let second = vm.get(i, 0, ring);
                            let factor = if k < j {
                                j.min(k) as f64 / (2 * ring) as f64
                            } else {
                                (2 * ring - j) as f64 / (2 * ring) as f64
                            };
                            (first, second, factor)
                        } else {
                            let other_pipe = if i == ipipe1 { ipipe2 } else { ipipe1 };
                            (
                                vm.get(i, 0, k),
                                vm.get(other_pipe, 0, ns - k),
                                j as f64 / ns as f64,
                            )
                        }
                    } else {
                        (vm.get(i, j, 0), vm.get(i, j, ns), k as f64 / ns as f64)
                    };
                    lerp3(profile1, profile2, factor)
                } else {
                    let midline = ns.is_multiple_of(2)
                        && k == half_ns
                        && ((i == 0 && j == half_ns) || i == ipipe1 || i == ipipe2);
                    snap_to_pipe_profile(vm.get(i, j, k), &vpipe.profile, pipe_edge, front, midline)
                };
                vm.put(i, j, k, co);
            }
        }
    }
    Ok(vm)
}

/// Output points, origins, polygons, and Blender edge/face classifications of a rebuilt bevel.
pub(super) struct Emission {
    points: Vec<DVec3>,
    origins: Vec<usize>,
    polygons: Vec<OutPolygon>,
    seam_edges: Vec<(usize, usize)>,
    sharp_edges: Vec<(usize, usize)>,
    edge_flag_suppressions: Vec<(usize, usize)>,
}

/// An output polygon: per corner, the output point, its UV query and source face. `face` is the
/// representative face when the first loop has a source; otherwise it is only a fallback.
struct OutPolygon {
    vertices: Vec<usize>,
    queries: Vec<DVec3>,
    sources: Vec<Option<usize>>,
    face: usize,
    kind: Kind,
    material: Option<u32>,
    /// Optional per-corner registration mask for Blender's TRIFAN split; `None` registers all loops.
    uv_registration_mask: Option<u8>,
}

impl Emission {
    /// Records seam and sharp marks for an emitted edge.
    pub(super) fn record_edge_flags(
        &mut self,
        first: usize,
        second: usize,
        sharp: bool,
        seam: bool,
    ) {
        if seam {
            self.seam_edges.push((first, second));
        }
        if sharp {
            self.sharp_edges.push((first, second));
        }
    }

    /// Prevents inherited edge flags on interior bevel-strip rails.
    pub(super) fn suppress_edge_flags(&mut self, first: usize, second: usize) {
        self.edge_flag_suppressions.push((first, second));
    }

    /// Starts a rebuild with copied source points and identity point origins.
    pub(super) fn new(source_points: &[DVec3]) -> Self {
        Self {
            points: source_points.to_vec(),
            origins: (0..source_points.len()).collect(),
            polygons: Vec::new(),
            seam_edges: Vec::new(),
            sharp_edges: Vec::new(),
            edge_flag_suppressions: Vec::new(),
        }
    }

    pub(super) fn push_point(&mut self, co: DVec3, origin: usize) -> usize {
        self.points.push(co);
        self.origins.push(origin);
        self.points.len() - 1
    }

    /// Appends a polygon. `snapped` holds, per corner, the segment that its UV query is projected
    /// onto (Blender's `bev_create_ngon` snap edges), if any.
    pub(super) fn push_polygon(
        &mut self,
        vertices: Vec<usize>,
        snapped: impl AsRef<[Option<(DVec3, DVec3)>]>,
        sources: Vec<Option<usize>>,
        face: usize,
        kind: Kind,
        material: Option<u32>,
    ) {
        let queries = vertices
            .iter()
            .enumerate()
            .map(|(corner, id)| {
                let point = self.points[*id];
                match snapped.as_ref().get(corner).copied().flatten() {
                    Some((a, b)) => closest_on_segment(point, a, b),
                    None => point,
                }
            })
            .collect();
        self.polygons.push(OutPolygon {
            vertices,
            queries,
            sources,
            face,
            kind,
            material,
            uv_registration_mask: None,
        });
    }
    /// Restricts the next emitted polygon's Blender TRIFAN loop registration mask.
    pub(super) fn set_last_uv_registration(&mut self, mask: u8) -> Result<()> {
        let polygon = self
            .polygons
            .last_mut()
            .ok_or_else(|| internal_invariant("emitted polygon missing"))?;
        debug_assert!(polygon.vertices.len() <= u8::BITS as usize);
        polygon.uv_registration_mask = Some(mask);
        Ok(())
    }
}

/// Blender's `contig_ldata_around_vert`, computed once per input vertex.
pub(super) fn loop_vertices_contiguous(
    layers: &SourceLoopMathLayers<'_>,
    source_polygons: &[Polygon],
    vertex_count: usize,
    topology: &SourceBMeshLoops,
) -> Vec<bool> {
    let mut contiguous = vec![true; vertex_count];
    let mut face_order = Vec::new();
    for (vertex, contiguous_vertex) in contiguous.iter_mut().enumerate() {
        face_order.clear();
        append_source_faces_around_vertex(
            vertex,
            &topology.disk_edges,
            &topology.edge_loop_first,
            &topology.loops,
            &mut face_order,
        );
        let mut reference = None;
        for &face_index in &face_order {
            let Some(corner) = corner_of(source_polygons, face_index, vertex) else {
                continue;
            };
            if let Some(reference_loop) = reference {
                if !loop_math_equal(layers, reference_loop, (face_index, corner)) {
                    *contiguous_vertex = false;
                    break;
                }
            } else {
                reference = Some((face_index, corner));
            }
        }
    }
    contiguous
}

/// Blender's `EdgeHalf::is_seam`: requires opposite loop directions and equal math-layer values.
pub(super) fn edge_is_seam(
    layers: &SourceLoopMathLayers<'_>,
    polygons: &[Polygon],
    sides: &[(usize, usize, usize, u32, usize)],
    a: usize,
    b: usize,
) -> bool {
    if !layers.has_loop_layers {
        return false;
    }
    let [first, second] = sides else {
        return true;
    };
    let (first_face, first_start, first_end, _, first_corner) = *first;
    let (second_face, second_start, second_end, _, second_corner) = *second;
    if first_start == second_start && first_end == second_end {
        return true;
    }
    let first_next = (first_corner + 1) % polygons[first_face].0.len();
    let second_next = (second_corner + 1) % polygons[second_face].0.len();
    let first_a = if first_start == a {
        first_corner
    } else {
        first_next
    };
    let first_b = if first_start == b {
        first_corner
    } else {
        first_next
    };
    let second_a = if second_start == a {
        second_corner
    } else {
        second_next
    };
    let second_b = if second_start == b {
        second_corner
    } else {
        second_next
    };
    !loop_math_equal(layers, (first_face, first_a), (second_face, second_a))
        || !loop_math_equal(layers, (first_face, first_b), (second_face, second_b))
}

/// Blender's `math_layer_info_init` components: faces join only across oppositely wound edges
/// whose math loop layers agree at both ends. Labels follow face order; top/bottom become 0 and 1.
pub(super) fn face_components(
    points: &[DVec3],
    polygons: &[Polygon],
    layers: &SourceLoopMathLayers<'_>,
) -> Result<Vec<usize>> {
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (index, (face, _)) in polygons.iter().enumerate() {
        let count = face.len();
        for position in 0..count {
            edge_faces
                .entry(edge_key(face[position], face[(position + 1) % count]))
                .or_default()
                .push(index);
        }
    }
    let mut label = vec![usize::MAX; polygons.len()];
    let mut components = 0;
    for start in 0..polygons.len() {
        if label[start] != usize::MAX {
            continue;
        }
        let current = components;
        components += 1;
        label[start] = current;
        let mut stack = vec![start];
        while let Some(face_index) = stack.pop() {
            let face = &polygons[face_index].0;
            let count = face.len();
            for position in 0..count {
                let (a, b) = (face[position], face[(position + 1) % count]);
                let incident_faces = &edge_faces[&edge_key(a, b)];
                // Blender's BM_edge_loop_pair only joins manifold edges.
                if incident_faces.len() != 2 {
                    continue;
                }
                let other = incident_faces
                    .iter()
                    .copied()
                    .find(|other| *other != face_index)
                    .ok_or_else(|| internal_invariant("edge has no neighbouring face"))?;
                if label[other] != usize::MAX {
                    continue;
                }
                let (Some(a_here), Some(a_there), Some(b_here), Some(b_there)) = (
                    corner_of(polygons, face_index, a),
                    corner_of(polygons, other, a),
                    corner_of(polygons, face_index, b),
                    corner_of(polygons, other, b),
                ) else {
                    continue;
                };
                let other_face = &polygons[other].0;
                if b_there == (a_there + 1) % other_face.len() {
                    continue;
                }
                if loop_math_equal(layers, (face_index, a_here), (other, a_there))
                    && loop_math_equal(layers, (face_index, b_here), (other, b_there))
                {
                    label[other] = current;
                    stack.push(other);
                }
            }
        }
    }
    if components <= 1 {
        return Ok(label);
    }
    let mut top = 0;
    let mut top_z = -1.0e30_f64;
    let mut bottom = 0;
    let mut bottom_z = 1.0e30_f64;
    for (face, polygon) in polygons.iter().enumerate() {
        let z = bounds_center(points, &polygon.0).z;
        if z > top_z {
            top_z = z;
            top = face;
        }
        if z < bottom_z {
            bottom_z = z;
            bottom = face;
        }
    }
    let top_component = label[top];
    let mut bottom_component = label[bottom];
    let first = label[0];
    swap_labels(&mut label, first, top_component);
    if bottom_component != top_component {
        if bottom_component == 0 {
            bottom_component = top_component;
        }
        let second = label[1];
        swap_labels(&mut label, second, bottom_component);
    }
    Ok(label)
}

fn swap_labels(labels: &mut [usize], first: usize, second: usize) {
    if first == second {
        return;
    }
    for label in labels.iter_mut() {
        if *label == first {
            *label = second;
        } else if *label == second {
            *label = first;
        }
    }
}

/// Blender's `BM_face_calc_center_bounds`: the centre of the face's bounding box.
pub(super) fn bounds_center(points: &[DVec3], face: &[usize]) -> DVec3 {
    let mut low = DVec3::splat(f64::INFINITY);
    let mut high = DVec3::splat(f64::NEG_INFINITY);
    for vertex in face {
        low = low.min(points[*vertex]);
        high = high.max(points[*vertex]);
    }
    (low + high) * 0.5
}

/// Blender's `ortho_basis_v3v3_v3`: two unit vectors orthogonal to the unit vector `n`.
fn ortho_basis(n: DVec3) -> (DVec3, DVec3) {
    let f = n.x * n.x + n.y * n.y;
    if f > 1.192_092_9e-7 {
        let d = 1.0 / f.sqrt();
        let n1 = DVec3::new(n.y * d, -n.x * d, 0.0);
        let n2 = DVec3::new(-n.z * n1.y, n.z * n1.x, n.x * n1.y - n.y * n1.x);
        (n1, n2)
    } else {
        (
            DVec3::new(if n.z < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        )
    }
}

/// The 2D coordinates of `point` in the plane of a face with unit normal `normal` (Blender's
/// `axis_dominant_v3_to_m3` followed by `mul_v2_m3v3`).
pub(super) fn project_to_face(normal: DVec3, point: DVec3) -> (f64, f64) {
    let (n1, n2) = ortho_basis(normal);
    (n1.dot(point), n2.dot(point))
}

/// Blender's `BM_face_point_inside_test`: `isect_point_poly_v2` on the face projected along its
/// normal. It is a half-open even-odd crossing test, so boundary points can count as inside.
pub(super) fn point_strictly_inside(
    points: &[DVec3],
    face: &[usize],
    normal: DVec3,
    point: DVec3,
) -> bool {
    let p = project_to_face(normal, point);
    let polygon: Vec<(f64, f64)> = face
        .iter()
        .map(|index| project_to_face(normal, points[*index]))
        .collect();
    let count = polygon.len();
    let mut inside = false;
    let mut j = count - 1;
    for i in 0..count {
        let (vi, vj) = (polygon[i], polygon[j]);
        if (vi.1 > p.1) != (vj.1 > p.1) && p.0 < (vj.0 - vi.0) * (p.1 - vi.1) / (vj.1 - vi.1) + vi.0
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Blender's `STD_UV_CONNECT_LIMIT`: UV coordinates closer than this are one UV vertex.
const STD_UV_CONNECT_LIMIT: f32 = 1.0e-4;

/// One source FLOAT2 corner layer, indexed by face then loop.
pub(super) struct SourceFloat2Layer {
    key: CornerFloat2LayerKey,
    faces: Vec<Option<Vec<[f64; 2]>>>,
}

/// Canonical UV maps and generic FLOAT2 corner attributes used by Bevel.
pub(super) fn source_float2_layers(mesh: &Mesh) -> Vec<SourceFloat2Layer> {
    let mut layers: Vec<SourceFloat2Layer> = uv_layers::read_layers(mesh)
        .into_iter()
        .map(|layer| SourceFloat2Layer {
            key: CornerFloat2LayerKey::Uv(layer.name),
            faces: complete_float2_faces(mesh, layer.faces),
        })
        .collect();
    for (name, attribute) in &mesh.attributes {
        if name == uv_layers::ATTRIBUTE
            || name == "blender_attributes"
            || !is_corner_float2(attribute)
        {
            continue;
        }
        let Some(values) = attribute.get("values").and_then(Value::as_object) else {
            continue;
        };
        let faces = mesh
            .faces
            .iter()
            .map(|face| values.get(&format!("f{}", face.id)).and_then(float2_row))
            .collect();
        let faces = complete_float2_faces(mesh, faces);
        layers.push(SourceFloat2Layer {
            key: CornerFloat2LayerKey::Attribute(name.clone()),
            faces,
        });
    }
    if let Some(attributes) = mesh
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
    {
        for attribute in attributes {
            if !is_corner_float2(attribute) {
                continue;
            }
            let Some(name) = attribute["name"].as_str() else {
                continue;
            };
            if layers.iter().any(|layer| {
                matches!(&layer.key, CornerFloat2LayerKey::Uv(layer_name) if layer_name == name)
            }) {
                continue;
            }
            let Some(values) = attribute["values"].as_array() else {
                continue;
            };
            let mut offset = 0;
            let faces = mesh
                .faces
                .iter()
                .map(|face| {
                    let end = offset + face.vertices.len();
                    let row = values.get(offset..end).and_then(|values| {
                        values
                            .iter()
                            .map(float2_value)
                            .collect::<Option<Vec<[f64; 2]>>>()
                    });
                    offset = end;
                    row
                })
                .collect();
            let faces = complete_float2_faces(mesh, faces);
            layers.push(SourceFloat2Layer {
                key: CornerFloat2LayerKey::BlenderAttribute(name.to_owned()),
                faces,
            });
        }
    }
    layers
}

fn is_corner_float2(attribute: &Value) -> bool {
    attribute["domain"]
        .as_str()
        .is_some_and(|domain| domain.eq_ignore_ascii_case("corner"))
        && attribute["type"]
            .as_str()
            .or_else(|| attribute["data_type"].as_str())
            .is_some_and(|data_type| data_type.eq_ignore_ascii_case("float2"))
}

fn float2_row(value: &Value) -> Option<Vec<[f64; 2]>> {
    value.as_array()?.iter().map(float2_value).collect()
}

fn float2_value(value: &Value) -> Option<[f64; 2]> {
    let components = value.as_array()?;
    Some([components.first()?.as_f64()?, components.get(1)?.as_f64()?])
}

fn complete_float2_faces(
    mesh: &Mesh,
    faces: Vec<Option<Vec<[f64; 2]>>>,
) -> Vec<Option<Vec<[f64; 2]>>> {
    mesh.faces
        .iter()
        .zip(faces)
        .map(|(face, values)| {
            let mut values = values.unwrap_or_default();
            values.resize(face.vertices.len(), [0.0, 0.0]);
            values.truncate(face.vertices.len());
            Some(values)
        })
        .collect()
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorStorage {
    Float,
    Byte,
}

enum SourceLoopMathLayer<'a> {
    Float2(&'a [Option<Vec<[f64; 2]>>]),
    FloatColor(Vec<Vec<[f64; 4]>>),
    ByteColor(Vec<Vec<[f64; 4]>>),
}

/// Loop-data layers that Blender compares for edge/vertex continuity.
pub(super) struct SourceLoopMathLayers<'a> {
    layers: Vec<SourceLoopMathLayer<'a>>,
    /// Whether any loop-data layer exists, including non-math layers.
    pub has_loop_layers: bool,
    /// Whether at least one loop-data layer participates in math continuity.
    pub has_math_layers: bool,
}

/// Build the loop layers used by continuity checks; FLOAT2 values borrow the per-layer source data.
pub(super) fn source_loop_math_layers<'a>(
    mesh: &Mesh,
    float2_layers: &'a [SourceFloat2Layer],
) -> SourceLoopMathLayers<'a> {
    let mut layers: Vec<SourceLoopMathLayer<'a>> = float2_layers
        .iter()
        .map(|layer| SourceLoopMathLayer::Float2(&layer.faces))
        .collect();
    for (key, attribute) in &mesh.attributes {
        if !is_corner_domain(attribute) {
            continue;
        }
        let Some(data_type) = attribute["type"].as_str() else {
            continue;
        };
        let Some(storage) = color_storage(mesh, key, data_type) else {
            continue;
        };
        let Some(values) = attribute["values"].as_object() else {
            continue;
        };
        let faces = mesh
            .faces
            .iter()
            .map(|face| {
                let row = values.get(&format!("f{}", face.id)).and_then(float4_row);
                complete_color_face(row, face.vertices.len(), storage)
            })
            .collect();
        layers.push(match storage {
            ColorStorage::Float => SourceLoopMathLayer::FloatColor(faces),
            ColorStorage::Byte => SourceLoopMathLayer::ByteColor(faces),
        });
    }
    if let Some(attributes) = mesh
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
    {
        for attribute in attributes {
            if !is_corner_domain(attribute) {
                continue;
            }
            let Some(name) = attribute["name"].as_str() else {
                continue;
            };
            let Some(data_type) = attribute["data_type"].as_str() else {
                continue;
            };
            let Some(storage) = color_storage(mesh, name, data_type) else {
                continue;
            };
            let Some(values) = attribute["values"].as_array() else {
                continue;
            };
            let mut offset = 0;
            let faces = mesh
                .faces
                .iter()
                .map(|face| {
                    let end = offset + face.vertices.len();
                    let row = values.get(offset..end).and_then(|values| {
                        values
                            .iter()
                            .map(float4_value)
                            .collect::<Option<Vec<[f64; 4]>>>()
                    });
                    offset = end;
                    complete_color_face(row, face.vertices.len(), storage)
                })
                .collect();
            layers.push(match storage {
                ColorStorage::Float => SourceLoopMathLayer::FloatColor(faces),
                ColorStorage::Byte => SourceLoopMathLayer::ByteColor(faces),
            });
        }
    }
    SourceLoopMathLayers {
        has_loop_layers: source_has_loop_layers(mesh),
        has_math_layers: !layers.is_empty(),
        layers,
    }
}

fn is_corner_domain(attribute: &Value) -> bool {
    attribute["domain"]
        .as_str()
        .is_some_and(|domain| domain.eq_ignore_ascii_case("corner"))
}

fn color_storage(mesh: &Mesh, key: &str, data_type: &str) -> Option<ColorStorage> {
    if data_type.eq_ignore_ascii_case("BYTE_COLOR") {
        return Some(ColorStorage::Byte);
    }
    if data_type.eq_ignore_ascii_case("FLOAT_COLOR") {
        return Some(ColorStorage::Float);
    }
    if !data_type.eq_ignore_ascii_case("color") {
        return None;
    }
    let name = attribute_keys::user_name(key).unwrap_or(key);
    let source_type = mesh.attributes["blender_metadata"]["color_data_types"][name]
        .as_str()
        .unwrap_or("FLOAT_COLOR");
    Some(if source_type.eq_ignore_ascii_case("BYTE_COLOR") {
        ColorStorage::Byte
    } else {
        ColorStorage::Float
    })
}

fn complete_color_face(
    values: Option<Vec<[f64; 4]>>,
    corner_count: usize,
    storage: ColorStorage,
) -> Vec<[f64; 4]> {
    let mut values = values.unwrap_or_default();
    values.resize(corner_count, [0.0, 0.0, 0.0, 1.0]);
    values.truncate(corner_count);
    if storage == ColorStorage::Byte {
        for color in &mut values {
            *color = byte_color_value(*color);
        }
    }
    values
}

fn byte_color_value(color: [f64; 4]) -> [f64; 4] {
    [
        unit_float_to_byte(linearrgb_to_srgb(color[0] as f32)),
        unit_float_to_byte(linearrgb_to_srgb(color[1] as f32)),
        unit_float_to_byte(linearrgb_to_srgb(color[2] as f32)),
        unit_float_to_byte(color[3] as f32),
    ]
}

fn linearrgb_to_srgb(value: f32) -> f32 {
    if value < 0.003_130_8 {
        if value < 0.0 { 0.0 } else { value * 12.92 }
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn unit_float_to_byte(value: f32) -> f64 {
    if value <= 0.0 {
        0.0
    } else if value > 1.0 - 0.5 / 255.0 {
        255.0
    } else {
        f64::from((255.0 * value + 0.5).floor())
    }
}

fn float4_row(value: &Value) -> Option<Vec<[f64; 4]>> {
    value.as_array()?.iter().map(float4_value).collect()
}

fn float4_value(value: &Value) -> Option<[f64; 4]> {
    let components = value.as_array()?;
    Some([
        components.first()?.as_f64()?,
        components.get(1)?.as_f64()?,
        components.get(2)?.as_f64()?,
        components.get(3)?.as_f64()?,
    ])
}

fn source_has_loop_layers(mesh: &Mesh) -> bool {
    mesh.attributes.contains_key("custom_normal")
        || mesh
            .attributes
            .get("uv_layers")
            .and_then(|store| store["layers"].as_array())
            .is_some_and(|layers| !layers.is_empty())
        || mesh.attributes.values().any(|attribute| {
            is_corner_domain(attribute)
                || attribute
                    .as_array()
                    .is_some_and(|entries| entries.iter().any(is_corner_domain))
        })
}

fn loop_math_equal(
    layers: &SourceLoopMathLayers<'_>,
    first: (usize, usize),
    second: (usize, usize),
) -> bool {
    layers.layers.iter().all(|layer| match layer {
        SourceLoopMathLayer::Float2(faces) => {
            let a = faces
                .get(first.0)
                .and_then(Option::as_ref)
                .and_then(|face| face.get(first.1));
            let b = faces
                .get(second.0)
                .and_then(Option::as_ref)
                .and_then(|face| face.get(second.1));
            matches!((a, b), (Some(a), Some(b)) if float2_loop_data_equal(a, b))
        }
        SourceLoopMathLayer::FloatColor(faces) => {
            let a = faces.get(first.0).and_then(|face| face.get(first.1));
            let b = faces.get(second.0).and_then(|face| face.get(second.1));
            matches!((a, b), (Some(a), Some(b)) if color_distance_squared(a, b) < 0.001)
        }
        SourceLoopMathLayer::ByteColor(faces) => {
            let a = faces.get(first.0).and_then(|face| face.get(first.1));
            let b = faces.get(second.0).and_then(|face| face.get(second.1));
            matches!((a, b), (Some(a), Some(b)) if color_distance_squared(a, b) < 0.001)
        }
    })
}

fn color_distance_squared(first: &[f64; 4], second: &[f64; 4]) -> f32 {
    let mut distance = 0.0_f32;
    for channel in 0..4 {
        let difference = first[channel] as f32 - second[channel] as f32;
        distance += difference * difference;
    }
    distance
}

fn float2_loop_data_equal(first: &[f64; 2], second: &[f64; 2]) -> bool {
    let x = first[0] as f32 - second[0] as f32;
    let y = first[1] as f32 - second[1] as f32;
    x * x + y * y < 1.0e-5_f32
}

fn float2_uv_match(first: &[f64; 2], second: &[f64; 2]) -> bool {
    (first[0] as f32 - second[0] as f32).abs() <= STD_UV_CONNECT_LIMIT
        && (first[1] as f32 - second[1] as f32).abs() <= STD_UV_CONNECT_LIMIT
}

/// Blender's per-layer `determine_uv_vert_connectivity`: two input loops share a FLOAT2 vertex
/// when their coordinates match within `STD_UV_CONNECT_LIMIT`.
fn uv_connected(
    uvs: &[Vec<Option<Vec<[f64; 2]>>>],
    first: (usize, usize),
    second: (usize, usize),
) -> bool {
    !uvs.is_empty()
        && uvs.iter().all(|layer| {
            let a = layer
                .get(first.0)
                .and_then(Option::as_ref)
                .and_then(|face| face.get(first.1));
            let b = layer
                .get(second.0)
                .and_then(Option::as_ref)
                .and_then(|face| face.get(second.1));
            match (a, b) {
                (Some(a), Some(b)) => float2_uv_match(a, b),
                _ => false,
            }
        })
}

/// Blender's `determine_uv_vert_connectivity`: source loops follow `BM_FACES_OF_VERT` radial order.
fn source_vertex_buckets(
    uvs: &[Vec<Option<Vec<[f64; 2]>>>],
    source_polygons: &[Polygon],
    vertex: usize,
    topology: &SourceBMeshLoops,
    face_order: &mut Vec<usize>,
) -> Vec<Vec<(usize, usize)>> {
    face_order.clear();
    append_source_faces_around_vertex(
        vertex,
        &topology.disk_edges,
        &topology.edge_loop_first,
        &topology.loops,
        face_order,
    );
    let mut buckets: Vec<Vec<(usize, usize)>> = Vec::new();
    for &face_index in face_order.iter() {
        let Some(corner) = corner_of(source_polygons, face_index, vertex) else {
            continue;
        };
        let loop_id = (face_index, corner);
        match buckets.iter_mut().find(|bucket| {
            bucket
                .iter()
                .any(|other| uv_connected(uvs, *other, loop_id))
        }) {
            Some(bucket) => bucket.push(loop_id),
            None => buckets.push(vec![loop_id]),
        }
    }
    buckets
}

fn corner_of(source_polygons: &[Polygon], face: usize, vertex: usize) -> Option<usize> {
    source_polygons
        .get(face)?
        .0
        .iter()
        .position(|value| *value == vertex)
}

/// Blender's `update_uv_vert_map` and `bevel_merge_uvs` for one FLOAT2 layer. Loops of one new
/// vertex whose representative faces are connected for this layer are grouped and averaged.
/// Averaging source weights matches averaging the interpolated values because interpolation is
/// linear. `loops` holds `(output vertex, representative face)` in output loop order.
fn merge_connected_loops(
    corners: &mut [Vec<(usize, f64)>],
    loops: &[(usize, Option<usize>)],
    origins: &[usize],
    source_polygons: &[Polygon],
    topology: &SourceBMeshLoops,
    uvs: &[Vec<Option<Vec<[f64; 2]>>>],
) {
    if uvs.is_empty() {
        return;
    }
    let mut at_vertex: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut buckets: HashMap<usize, Vec<Vec<usize>>> = HashMap::new();
    let mut source_buckets: HashMap<usize, Vec<Vec<(usize, usize)>>> = HashMap::new();
    let mut source_face_order = Vec::new();
    for (index, &(vertex, attached)) in loops.iter().enumerate() {
        let Some(frep) = attached else {
            continue;
        };
        let vertex_buckets = buckets.entry(vertex).or_default();
        let existing = at_vertex.entry(vertex).or_default();
        if vertex_buckets.is_empty() {
            vertex_buckets.push(vec![index]);
            existing.push(index);
            continue;
        }
        let origin = origins[vertex];
        let frep_corner = corner_of(source_polygons, frep, origin);
        let mut found = false;
        for &other in existing.iter() {
            let Some(other_frep) = loops[other].1 else {
                continue;
            };
            let connected = other_frep == frep
                || match (frep_corner, corner_of(source_polygons, other_frep, origin)) {
                    (Some(a), Some(b)) => {
                        let vertex_source = source_buckets.entry(origin).or_insert_with(|| {
                            source_vertex_buckets(
                                uvs,
                                source_polygons,
                                origin,
                                topology,
                                &mut source_face_order,
                            )
                        });
                        vertex_source.iter().any(|bucket| {
                            bucket.contains(&(frep, a)) && bucket.contains(&(other_frep, b))
                        })
                    }
                    _ => false,
                };
            if connected
                && let Some(bucket) = vertex_buckets
                    .iter_mut()
                    .find(|bucket| bucket.contains(&other))
            {
                bucket.push(index);
                found = true;
                break;
            }
        }
        if !found {
            vertex_buckets.push(vec![index]);
        }
        existing.push(index);
    }
    for vertex_buckets in buckets.into_values() {
        for bucket in vertex_buckets {
            if bucket.len() < 2 {
                continue;
            }
            let weight = 1.0 / bucket.len() as f64;
            let mut merged: Vec<(usize, f64)> = bucket
                .iter()
                .flat_map(|&index| {
                    corners[index]
                        .iter()
                        .map(move |(source, w)| (*source, w * weight))
                })
                .collect();
            merged.sort_by_key(|(source, _)| *source);
            let mut collapsed: Vec<(usize, f64)> = Vec::with_capacity(merged.len());
            for (source, w) in merged {
                match collapsed.last_mut() {
                    Some((last, total)) if *last == source => *total += w,
                    _ => collapsed.push((source, w)),
                }
            }
            for &index in &bucket {
                corners[index].clone_from(&collapsed);
            }
        }
    }
}

fn mark_output_edge_flag(mesh: &mut Mesh, edge_index: usize, name: &str) {
    if name == "sharp" {
        let Some(edge_key) = mesh
            .edges
            .get(edge_index)
            .map(|edge| format!("e{}", edge.id))
        else {
            return;
        };
        if let Some(values) = mesh
            .attributes
            .get_mut("sharp")
            .and_then(|attribute| attribute.get_mut("values"))
            .and_then(Value::as_object_mut)
        {
            values.insert(edge_key, Value::Bool(true));
        }
    }
    if let Some(flag) = mesh
        .attributes
        .get_mut("blender_edge_flags")
        .and_then(Value::as_array_mut)
        .and_then(|flags| flags.get_mut(edge_index))
        .and_then(Value::as_object_mut)
    {
        flag.insert(name.to_owned(), Value::Bool(true));
    }
}
fn clear_output_edge_flag(mesh: &mut Mesh, edge_index: usize, name: &str) {
    if name == "sharp"
        && let Some(edge_key) = mesh
            .edges
            .get(edge_index)
            .map(|edge| format!("e{}", edge.id))
        && let Some(values) = mesh
            .attributes
            .get_mut("sharp")
            .and_then(|attribute| attribute.get_mut("values"))
            .and_then(Value::as_object_mut)
    {
        values.remove(&edge_key);
    }
    if let Some(flag) = mesh
        .attributes
        .get_mut("blender_edge_flags")
        .and_then(Value::as_array_mut)
        .and_then(|flags| flags.get_mut(edge_index))
        .and_then(Value::as_object_mut)
    {
        flag.remove(name);
    }
}

/// Blender's face projection for `BM_loop_interp_from_face`.
struct BevelFaceProjection {
    basis: [Vec3; 2],
    vertices: Vec<usize>,
    corners: Vec<[f32; 2]>,
}

impl BevelFaceProjection {
    fn new(mesh: &Mesh, face: &[usize]) -> Self {
        let vertices = face.to_vec();
        let mut normal = match vertices.len() {
            3 => {
                let first = mesh.vertices[vertices[0]].co.as_vec3();
                let second = mesh.vertices[vertices[1]].co.as_vec3();
                let third = mesh.vertices[vertices[2]].co.as_vec3();
                (first - second).cross(second - third)
            }
            4 => {
                let first = mesh.vertices[vertices[0]].co.as_vec3();
                let second = mesh.vertices[vertices[1]].co.as_vec3();
                let third = mesh.vertices[vertices[2]].co.as_vec3();
                let fourth = mesh.vertices[vertices[3]].co.as_vec3();
                (first - third).cross(second - fourth)
            }
            _ => {
                let mut normal = Vec3::ZERO;
                for index in 0..vertices.len() {
                    let previous = mesh.vertices[vertices[index]].co.as_vec3();
                    let current = mesh.vertices[vertices[(index + 1) % vertices.len()]]
                        .co
                        .as_vec3();
                    normal.x += (previous.y - current.y) * (previous.z + current.z);
                    normal.y += (previous.z - current.z) * (previous.x + current.x);
                    normal.z += (previous.x - current.x) * (previous.y + current.y);
                }
                normal
            }
        };
        normal = normal.normalize_or_zero();
        if normal == Vec3::ZERO {
            normal = bevel_face_fallback_normal(bevel_face_fallback_tangent(mesh, &vertices));
        }
        let basis = bevel_ortho_basis(normal);
        let corners = vertices
            .iter()
            .map(|index| {
                let point = mesh.vertices[*index].co.as_vec3();
                [basis[0].dot(point), basis[1].dot(point)]
            })
            .collect();
        Self {
            basis,
            vertices,
            corners,
        }
    }

    fn project(&self, point: DVec3) -> [f32; 2] {
        let point = point.as_vec3();
        [self.basis[0].dot(point), self.basis[1].dot(point)]
    }
}

fn bevel_face_fallback_tangent(mesh: &Mesh, face: &[usize]) -> Vec3 {
    if face.len() == 4 {
        let first = mesh.vertices[face[0]].co.as_vec3();
        let second = mesh.vertices[face[1]].co.as_vec3();
        let third = mesh.vertices[face[2]].co.as_vec3();
        let fourth = mesh.vertices[face[3]].co.as_vec3();
        let first_pair = (fourth - third) + (first - second);
        let second_pair = (first - fourth) + (second - third);
        if first_pair.length_squared() < second_pair.length_squared() {
            second_pair.normalize_or_zero()
        } else {
            first_pair.normalize_or_zero()
        }
    } else {
        let mut tangent = Vec3::ZERO;
        let mut longest_squared = 0.0;
        for index in 0..face.len() {
            let first = mesh.vertices[face[index]].co.as_vec3();
            let second = mesh.vertices[face[(index + 1) % face.len()]].co.as_vec3();
            let edge = first - second;
            let length_squared = edge.length_squared();
            if length_squared > longest_squared {
                tangent = edge;
                longest_squared = length_squared;
            }
        }
        tangent.normalize_or_zero()
    }
}

fn bevel_face_fallback_normal(tangent: Vec3) -> Vec3 {
    let absolute = tangent.abs();
    let axis = if absolute.x > absolute.y {
        if absolute.x > absolute.z { 0 } else { 2 }
    } else if absolute.y > absolute.z {
        1
    } else {
        2
    };
    match axis {
        0 => Vec3::new(-tangent.y - tangent.z, tangent.x, tangent.x),
        1 => Vec3::new(tangent.y, -tangent.x - tangent.z, tangent.y),
        _ => Vec3::new(tangent.z, tangent.z, -tangent.x - tangent.y),
    }
    .normalize_or_zero()
}

/// Blender's `ortho_basis_v3v3_v3`, used by `axis_dominant_v3_to_m3`.
fn bevel_ortho_basis(normal: Vec3) -> [Vec3; 2] {
    let xy_squared = normal.x * normal.x + normal.y * normal.y;
    if xy_squared > f32::EPSILON {
        let inverse_length = 1.0 / xy_squared.sqrt();
        let first = Vec3::new(normal.y * inverse_length, -normal.x * inverse_length, 0.0);
        let second = Vec3::new(
            -normal.z * first.y,
            normal.z * first.x,
            normal.x * first.y - normal.y * first.x,
        );
        [first, second]
    } else {
        [
            Vec3::new(if normal.z < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
            Vec3::Y,
        ]
    }
}

#[derive(Clone, Copy)]
struct Double2Length {
    direction: [f64; 2],
    length: f64,
}

impl Double2Length {
    fn from_points(point: [f32; 2], origin: [f32; 2]) -> Self {
        let direction = [
            f64::from(point[0]) - f64::from(origin[0]),
            f64::from(point[1]) - f64::from(origin[1]),
        ];
        let length = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        Self { direction, length }
    }
}

fn mean_value_half_tangent(first: Double2Length, second: Double2Length) -> f64 {
    let area = first.direction[0] * second.direction[1] - first.direction[1] * second.direction[0];
    if area == 0.0 {
        return 0.0;
    }
    let dot = first.direction[0] * second.direction[0] + first.direction[1] * second.direction[1];
    let value = (first.length * second.length - dot) / area;
    if value.is_finite() { value } else { 0.0 }
}

fn segment_distance_squared_v2(point: [f32; 2], first: [f32; 2], second: [f32; 2]) -> f32 {
    let direction = [second[0] - first[0], second[1] - first[1]];
    let offset = [point[0] - first[0], point[1] - first[1]];
    let denominator = direction[0] * direction[0] + direction[1] * direction[1];
    let factor = if denominator == 0.0 {
        0.0
    } else {
        ((direction[0] * offset[0] + direction[1] * offset[1]) / denominator).clamp(0.0, 1.0)
    };
    let delta = [
        point[0] - (first[0] + direction[0] * factor),
        point[1] - (first[1] + direction[1] * factor),
    ];
    delta[0] * delta[0] + delta[1] * delta[1]
}

/// Blender's `interp_weights_poly_v2`, including its adaptive point/edge cutoff and double tangents.
fn bmesh_interp_weights_poly_v2(points: &[[f32; 2]], point: [f32; 2], weights: &mut [f32]) {
    let count = points.len();
    weights[..count].fill(0.0);
    let mut max_value = 0.0_f32;
    for vertex in points {
        max_value = max_value.max((vertex[0] - point[0]).abs());
        max_value = max_value.max((vertex[1] - point[1]).abs());
    }
    let epsilon = 16.0 * f32::EPSILON * max_value;
    let epsilon_squared = epsilon * epsilon;
    let mut current = count - 1;
    let mut next = 0;
    let mut current_vertex = points[current];
    let mut next_vertex = points[next];
    let mut direction_current = Double2Length::from_points(points[count - 2], point);
    let mut direction_next = Double2Length::from_points(current_vertex, point);
    let mut tangent_previous = mean_value_half_tangent(direction_current, direction_next);
    let mut total_weight = 0.0_f32;
    let mut point_or_segment = 0_u8;

    while next < count {
        if direction_next.length < f64::from(epsilon) {
            point_or_segment = 1;
            break;
        }
        if segment_distance_squared_v2(point, current_vertex, next_vertex) < epsilon_squared {
            point_or_segment = 2;
            break;
        }

        direction_current = direction_next;
        direction_next = Double2Length::from_points(next_vertex, point);
        let tangent = mean_value_half_tangent(direction_current, direction_next);
        weights[current] = if direction_current.length == 0.0 {
            0.0
        } else {
            ((tangent_previous + tangent) / direction_current.length) as f32
        };
        total_weight += weights[current];
        current = next;
        next += 1;
        if next == count {
            break;
        }
        current_vertex = next_vertex;
        next_vertex = points[next];
        tangent_previous = tangent;
    }

    match point_or_segment {
        1 => {
            weights[..count].fill(0.0);
            weights[current] = 1.0;
        }
        2 => {
            weights[..count].fill(0.0);
            let direction = [
                next_vertex[0] - current_vertex[0],
                next_vertex[1] - current_vertex[1],
            ];
            let offset = [point[0] - current_vertex[0], point[1] - current_vertex[1]];
            let denominator = direction[0] * direction[0] + direction[1] * direction[1];
            let factor = if denominator > 0.0 {
                ((direction[0] * offset[0] + direction[1] * offset[1]) / denominator)
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
            weights[current] = 1.0 - factor;
            weights[next] = factor;
        }
        _ if total_weight != 0.0 => {
            for weight in &mut weights[..count] {
                *weight /= total_weight;
            }
        }
        _ => {}
    }
}

#[derive(Clone, Copy)]
struct SourceBMeshLoop {
    face: usize,
    vertex: usize,
    edge: usize,
    next: usize,
    radial_next: usize,
}
/// Input `BMesh` topology in source face-table order with preserved disk/radial adjacency.
pub(super) struct SourceBMeshLoops {
    source_corner_offsets: Vec<usize>,
    source_corner_order: Vec<usize>,
    disk_edges: Vec<Vec<usize>>,
    edge_loop_first: Vec<Option<usize>>,
    loops: Vec<SourceBMeshLoop>,
}

/// Input `BMesh` disk and radial loops in source face-table order.
pub(super) fn source_bmesh_loops(
    mesh: &Mesh,
    source_polygons: &[Polygon],
) -> Result<SourceBMeshLoops> {
    let mut source_corner_offsets = Vec::with_capacity(source_polygons.len());
    let mut source_corner_offset = 0;
    for (face, _) in source_polygons {
        source_corner_offsets.push(source_corner_offset);
        source_corner_offset += face.len();
    }
    let source_corner_order =
        clnor::corner_order(mesh)?.unwrap_or_else(|| (0..source_corner_offset).collect());
    let source_vertex_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut edge_by_vertices = HashMap::with_capacity(mesh.edges.len());
    let mut disk_edges = vec![Vec::new(); mesh.vertices.len()];
    for (edge_index, edge) in mesh.edges.iter().enumerate() {
        let first = source_vertex_index[&edge.vertices[0]];
        let second = source_vertex_index[&edge.vertices[1]];
        edge_by_vertices.insert(edge_key(first, second), edge_index);
        disk_edges[first].push(edge_index);
        if second != first {
            disk_edges[second].push(edge_index);
        }
    }

    let source_corner_count: usize = source_polygons.iter().map(|(face, _)| face.len()).sum();
    let mut loops: Vec<SourceBMeshLoop> = Vec::with_capacity(source_corner_count);
    let mut edge_loop_first: Vec<Option<usize>> = vec![None; mesh.edges.len()];
    let mut source_face_order: Vec<usize> = (0..source_polygons.len()).collect();
    source_face_order.sort_unstable_by_key(|&face_index| {
        let offset = source_corner_offsets[face_index];
        let face = &source_polygons[face_index].0;
        source_corner_order[offset..offset + face.len()]
            .iter()
            .copied()
            .min()
            .unwrap_or(usize::MAX)
    });
    for face_index in source_face_order {
        let face = &source_polygons[face_index].0;
        let face_start = loops.len();
        for corner in 0..face.len() {
            let next_corner = (corner + 1) % face.len();
            let edge = edge_by_vertices[&edge_key(face[corner], face[next_corner])];
            let loop_id = loops.len();
            loops.push(SourceBMeshLoop {
                face: face_index,
                vertex: face[corner],
                edge,
                next: face_start + next_corner,
                radial_next: loop_id,
            });
            if let Some(anchor) = edge_loop_first[edge] {
                let old_next = loops[anchor].radial_next;
                loops[loop_id].radial_next = old_next;
                loops[anchor].radial_next = loop_id;
            }
            edge_loop_first[edge] = Some(loop_id);
        }
    }
    Ok(SourceBMeshLoops {
        source_corner_offsets,
        source_corner_order,
        disk_edges,
        edge_loop_first,
        loops,
    })
}

fn radial_face_loop_count(first: Option<usize>, vertex: usize, loops: &[SourceBMeshLoop]) -> usize {
    let Some(first) = first else {
        return 0;
    };
    let mut current = first;
    let mut count = 0;
    loop {
        if loops[current].vertex == vertex {
            count += 1;
        }
        current = loops[current].radial_next;
        if current == first {
            return count;
        }
    }
}

fn radial_face_loop_first(
    first: Option<usize>,
    vertex: usize,
    loops: &[SourceBMeshLoop],
) -> Option<usize> {
    let first = first?;
    let mut current = first;
    loop {
        if loops[current].vertex == vertex {
            return Some(current);
        }
        current = loops[current].radial_next;
        if current == first {
            return None;
        }
    }
}

/// `BM_FACES_OF_VERT` on the input loops before `bevel_rebuild_existing_polygons`.
fn append_source_faces_around_vertex(
    vertex: usize,
    disk_edges: &[Vec<usize>],
    edge_loop_first: &[Option<usize>],
    loops: &[SourceBMeshLoop],
    faces: &mut Vec<usize>,
) {
    let disk = &disk_edges[vertex];
    let face_loop_count: usize = disk
        .iter()
        .map(|edge| radial_face_loop_count(edge_loop_first[*edge], vertex, loops))
        .sum();
    if face_loop_count == 0 {
        return;
    }
    let mut first_loop = None;
    for edge in disk {
        let Some(edge_loop) = edge_loop_first[*edge] else {
            continue;
        };
        first_loop = Some(if loops[edge_loop].vertex == vertex {
            edge_loop
        } else {
            loops[edge_loop].next
        });
        break;
    }
    let Some(mut first_loop) = first_loop else {
        return;
    };
    let mut current_loop = first_loop;
    let mut current_edge = loops[current_loop].edge;
    for ordinal in 0..face_loop_count {
        faces.push(loops[current_loop].face);
        if ordinal + 1 == face_loop_count {
            break;
        }
        let mut next_loop = loops[current_loop].radial_next;
        while next_loop != current_loop && loops[next_loop].vertex != vertex {
            next_loop = loops[next_loop].radial_next;
        }
        if next_loop != first_loop {
            current_loop = next_loop;
            continue;
        }

        let Some(edge_position) = disk.iter().position(|edge| *edge == current_edge) else {
            return;
        };
        let mut next_edge_and_loop = None;
        for step in 1..=disk.len() {
            let edge = disk[(edge_position + step) % disk.len()];
            if let Some(loop_id) = radial_face_loop_first(edge_loop_first[edge], vertex, loops) {
                next_edge_and_loop = Some((edge, loop_id));
                break;
            }
        }
        let Some((edge, loop_id)) = next_edge_and_loop else {
            return;
        };
        current_edge = edge;
        first_loop = loop_id;
        current_loop = first_loop;
    }
}

/// `BMesh` face-table order after bevel: retained input faces, VMesh/edge faces, then rebuilt source
/// faces first encountered by the selected vertices' radial face walk.
fn bmesh_emission_face_order(
    mesh: &Mesh,
    source_polygons: &[Polygon],
    emitted_polygons: &[OutPolygon],
    emitted_origins: &[usize],
    topology: &SourceBMeshLoops,
) -> Vec<usize> {
    let mut original_faces = Vec::new();
    let mut rebuilt_faces = vec![None; source_polygons.len()];
    let mut created_faces = Vec::new();
    for (emitted, polygon) in emitted_polygons.iter().enumerate() {
        match polygon.kind {
            Kind::Original => {
                let offset = topology.source_corner_offsets[polygon.face];
                let source_face = &source_polygons[polygon.face].0;
                let first_loop = topology.source_corner_order[offset..offset + source_face.len()]
                    .iter()
                    .copied()
                    .min()
                    .unwrap_or(usize::MAX);
                original_faces.push((first_loop, emitted));
            }
            Kind::Source => rebuilt_faces[polygon.face] = Some(emitted),
            Kind::Corner | Kind::Strip => created_faces.push(emitted),
        }
    }
    original_faces.sort_unstable_by_key(|(first_loop, _)| *first_loop);
    let mut order = Vec::with_capacity(emitted_polygons.len());
    order.extend(original_faces.into_iter().map(|(_, emitted)| emitted));
    order.extend(created_faces);
    if rebuilt_faces.iter().all(Option::is_none) {
        return order;
    }

    let mut rebuilt_vertex = vec![false; mesh.vertices.len()];
    for &vertex in emitted_origins.iter().skip(mesh.vertices.len()) {
        rebuilt_vertex[vertex] = true;
    }
    let mut seen_rebuilt = vec![false; source_polygons.len()];
    let mut source_face_order = Vec::new();
    for (vertex, &is_beveled) in rebuilt_vertex.iter().enumerate() {
        if is_beveled {
            append_source_faces_around_vertex(
                vertex,
                &topology.disk_edges,
                &topology.edge_loop_first,
                &topology.loops,
                &mut source_face_order,
            );
        }
    }
    for face in source_face_order {
        if !seen_rebuilt[face]
            && let Some(emitted) = rebuilt_faces[face]
        {
            seen_rebuilt[face] = true;
            order.push(emitted);
        }
    }
    order
}

fn has_quaternion_attribute(mesh: &Mesh, domain: &str) -> bool {
    mesh.attributes.values().any(|attribute| {
        (attribute["domain"]
            .as_str()
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(domain))
            && attribute["type"]
                .as_str()
                .is_some_and(|kind| kind.eq_ignore_ascii_case("quaternion")))
            || attribute.as_array().is_some_and(|entries| {
                entries.iter().any(|entry| {
                    entry["domain"]
                        .as_str()
                        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(domain))
                        && entry["data_type"]
                            .as_str()
                            .is_some_and(|kind| kind.eq_ignore_ascii_case("quaternion"))
                })
            })
    })
}

/// Builds the output mesh and its Blender face kinds from an emission. UVs use mean-value
/// interpolation in each corner's source face and Blender's connected-loop merge.
pub(super) fn finish(
    mesh: &Mesh,
    source_polygons: &[Polygon],
    emission: Emission,
    loose: Vec<[usize; 2]>,
    float2_layers: &[SourceFloat2Layer],
    source_loop_topology: Option<&SourceBMeshLoops>,
) -> Result<(Mesh, Vec<Kind>)> {
    let Emission {
        points: emitted_points,
        origins: emitted_origins,
        polygons: emitted_polygons,
        seam_edges: emitted_seam_edges,
        sharp_edges: emitted_sharp_edges,
        edge_flag_suppressions: emitted_edge_flag_suppressions,
    } = emission;
    let mut used = vec![false; emitted_points.len()];
    for polygon in &emitted_polygons {
        for vertex in &polygon.vertices {
            used[*vertex] = true;
        }
    }
    for [first, second] in &loose {
        used[*first] = true;
        used[*second] = true;
    }
    let mut remap = vec![0; emitted_points.len()];
    let mut points = Vec::new();
    let mut origins = Vec::new();
    let mut identities = Vec::new();
    for (index, point) in emitted_points.iter().enumerate() {
        if used[index] {
            remap[index] = points.len();
            points.push(*point);
            identities.push(emitted_origins[index]);
            // Blender's `create_mesh_bmvert` copies point data from the bevelled source vertex.
            origins.push(vec![(emitted_origins[index], 1.0)]);
        }
    }
    let max_face_corners = source_polygons
        .iter()
        .map(|(face, _)| face.len())
        .max()
        .unwrap_or(0);
    let mut interpolation_weights = vec![0.0_f32; max_face_corners];
    let owned_source_loop_topology;
    let source_loop_topology = if let Some(topology) = source_loop_topology {
        topology
    } else {
        owned_source_loop_topology = source_bmesh_loops(mesh, source_polygons)?;
        &owned_source_loop_topology
    };
    let source_corner_offsets = &source_loop_topology.source_corner_offsets;
    let source_corner_order = &source_loop_topology.source_corner_order;
    let face_projections: Vec<BevelFaceProjection> = source_polygons
        .iter()
        .map(|(face, _)| BevelFaceProjection::new(mesh, face))
        .collect();
    let output_corner_count: usize = emitted_polygons
        .iter()
        .map(|polygon| polygon.vertices.len())
        .sum();
    let mut point_interpolation_events = has_quaternion_attribute(mesh, "point").then(Vec::new);
    let mut corner_interpolated =
        has_quaternion_attribute(mesh, "corner").then(|| vec![false; output_corner_count]);
    let bmesh_face_order = bmesh_emission_face_order(
        mesh,
        source_polygons,
        &emitted_polygons,
        &emitted_origins,
        source_loop_topology,
    );
    let mut emitted_corner_offsets = Vec::with_capacity(emitted_polygons.len());
    let mut emitted_corner_offset = 0;
    let mut retained_original_loops = Vec::new();
    for polygon in &emitted_polygons {
        emitted_corner_offsets.push(emitted_corner_offset);
        if polygon.kind == Kind::Original {
            for (corner, &vertex) in polygon.vertices.iter().enumerate() {
                let source_corner =
                    corner_of(source_polygons, polygon.face, emitted_origins[vertex])
                        .ok_or_else(|| internal_invariant("retained loop has no corner"))?;
                let source_loop = source_corner_offsets[polygon.face] + source_corner;
                retained_original_loops.push((
                    source_corner_order[source_loop],
                    emitted_corner_offset + corner,
                ));
            }
        }
        emitted_corner_offset += polygon.vertices.len();
    }
    retained_original_loops.sort_unstable_by_key(|(source_loop, _)| *source_loop);
    let mut output_corner_order = vec![0; output_corner_count];
    let mut bmesh_corner_order = 0;
    for (_, output_loop) in retained_original_loops {
        output_corner_order[output_loop] = bmesh_corner_order;
        bmesh_corner_order += 1;
    }
    for &polygon_index in &bmesh_face_order {
        let polygon = &emitted_polygons[polygon_index];
        if polygon.kind == Kind::Original {
            continue;
        }
        for corner in 0..polygon.vertices.len() {
            output_corner_order[emitted_corner_offsets[polygon_index] + corner] =
                bmesh_corner_order;
            bmesh_corner_order += 1;
        }
    }
    debug_assert_eq!(bmesh_corner_order, output_corner_count);
    let mut normal_sources = Vec::with_capacity(output_corner_count);
    let mut polygons: Vec<Polygon> = Vec::with_capacity(emitted_polygons.len());
    let mut faces = Vec::with_capacity(emitted_polygons.len());
    let mut kinds = Vec::with_capacity(emitted_polygons.len());
    let mut queries = Vec::new();
    let mut corner_sources = Vec::new();
    let mut loops = Vec::new();
    for polygon in &emitted_polygons {
        let retained_original_face = polygon.kind == Kind::Original;
        // Source-free generated faces have no Blender face exemplar and keep default face data.
        let face_source = polygon
            .sources
            .first()
            .copied()
            .flatten()
            .map(|_| polygon.face);
        let material = if matches!(polygon.kind, Kind::Original | Kind::Source) {
            source_polygons[polygon.face].1
        } else {
            polygon
                .material
                .unwrap_or_else(|| face_source.map_or(0, |face| source_polygons[face].1))
        };
        polygons.push((
            polygon
                .vertices
                .iter()
                .map(|vertex| remap[*vertex])
                .collect(),
            material,
        ));
        faces.push(face_source);
        kinds.push(polygon.kind);
        let attached = polygon.sources.first().copied().flatten();
        for (corner, (query, source)) in polygon.queries.iter().zip(&polygon.sources).enumerate() {
            corner_sources.push((queries.len(), *source));
            queries.push(*query);
            let emitted_vertex = polygon.vertices[corner];
            let uv_registered = polygon
                .uv_registration_mask
                .is_none_or(|mask| mask & (1_u8 << corner) != 0);
            let attached =
                attached.filter(|_| uv_registered && emitted_vertex >= mesh.vertices.len());
            loops.push((emitted_vertex, attached));
            let normal_source = if retained_original_face {
                source.as_ref().and_then(|face| {
                    let source_vertex = emitted_origins[polygon.vertices[corner]];
                    let face_vertices = &source_polygons[*face].0;
                    let source_corner = face_vertices
                        .get(corner)
                        .filter(|&&vertex| vertex == source_vertex)
                        .map(|_| corner)
                        .or_else(|| corner_of(source_polygons, *face, source_vertex))?;
                    Some(source_corner_offsets[*face] + source_corner)
                })
            } else {
                None
            };
            normal_sources.push(normal_source);
        }
    }
    let mut corners = mean_value_corners(
        mesh,
        &queries,
        corner_sources
            .iter()
            .map(|(query, source)| (std::slice::from_ref(query), *source)),
    );
    for &polygon_index in &bmesh_face_order {
        let polygon = &emitted_polygons[polygon_index];
        if polygon.kind == Kind::Original || polygon.sources.first().copied().flatten().is_none() {
            continue;
        }
        for (corner, source) in polygon.sources.iter().enumerate() {
            let Some(source_face) = *source else {
                continue;
            };
            let face_projection = &face_projections[source_face];
            let weights = &mut interpolation_weights[..face_projection.vertices.len()];
            bmesh_interp_weights_poly_v2(
                &face_projection.corners,
                face_projection.project(polygon.queries[corner]),
                weights,
            );
            let emitted_vertex = polygon.vertices[corner];
            let output_vertex = remap[emitted_vertex];
            let output_corner = emitted_corner_offsets[polygon_index] + corner;
            let corner_origin = &mut corners[output_corner];
            corner_origin.clear();
            corner_origin.extend(weights.iter().enumerate().filter_map(
                |(source_corner, &weight)| {
                    (weight != 0.0).then_some((
                        source_corner_offsets[source_face] + source_corner,
                        f64::from(weight),
                    ))
                },
            ));
            if let Some(corner_interpolated) = &mut corner_interpolated {
                corner_interpolated[output_corner] = true;
            }
            if let Some(events) = &mut point_interpolation_events {
                let sources: Vec<(usize, f64)> = face_projection
                    .vertices
                    .iter()
                    .zip(weights.iter())
                    .filter_map(|(&source_vertex, &weight)| {
                        (weight != 0.0).then_some((source_vertex, f64::from(weight)))
                    })
                    .collect();
                let origin = &mut origins[output_vertex];
                origin.clear();
                origin.extend(sources.iter().copied());
                events.push(PointInterpolationEvent {
                    output_vertex,
                    mutated_source_vertex: (emitted_vertex < mesh.vertices.len())
                        .then_some(emitted_origins[emitted_vertex]),
                    sources,
                });
            } else {
                let origin = &mut origins[output_vertex];
                origin.clear();
                origin.extend(
                    face_projection
                        .vertices
                        .iter()
                        .zip(weights.iter())
                        .filter_map(|(&source_vertex, &weight)| {
                            (weight != 0.0).then_some((source_vertex, f64::from(weight)))
                        }),
                );
            }
        }
    }
    let corner_float2_sources = if float2_layers.is_empty() {
        None
    } else {
        Some(
            float2_layers
                .iter()
                .map(|layer| {
                    let mut merged = corners.clone();
                    merge_connected_loops(
                        &mut merged,
                        &loops,
                        &emitted_origins,
                        source_polygons,
                        source_loop_topology,
                        std::slice::from_ref(&layer.faces),
                    );
                    CornerFloat2Sources {
                        key: layer.key.clone(),
                        corners: merged,
                    }
                })
                .collect(),
        )
    };
    let origin = Origin {
        vertices: origins,
        faces,
        point_interpolation_events,
        corner_interpolated,
        corner_float2_sources,
        corners: Some(corners),
    };
    let loose = loose
        .into_iter()
        .map(|[first, second]| [remap[first], remap[second]])
        .collect();
    let mut result =
        build_mesh_with_edge_identity(points, polygons, loose, &origin, Some(&identities), mesh)?;
    if !emitted_edge_flag_suppressions.is_empty()
        || !emitted_seam_edges.is_empty()
        || !emitted_sharp_edges.is_empty()
    {
        let edge_index: HashMap<(u32, u32), usize> = result
            .edges
            .iter()
            .enumerate()
            .map(|(index, edge)| (edge_key(edge.vertices[0], edge.vertices[1]), index))
            .collect();
        for (first, second) in emitted_edge_flag_suppressions {
            let first = result.vertices[remap[first]].id;
            let second = result.vertices[remap[second]].id;
            let Some(index) = edge_index.get(&edge_key(first, second)) else {
                continue;
            };
            clear_output_edge_flag(&mut result, *index, "sharp");
            clear_output_edge_flag(&mut result, *index, "seam");
        }
        for (edges, name) in [(emitted_seam_edges, "seam"), (emitted_sharp_edges, "sharp")] {
            for (first, second) in edges {
                let first = result.vertices[remap[first]].id;
                let second = result.vertices[remap[second]].id;
                let Some(index) = edge_index.get(&edge_key(first, second)) else {
                    continue;
                };
                mark_output_edge_flag(&mut result, *index, name);
            }
        }
    }
    // `CD_PROP_INT16_2D` has no interpolation callback, so bevel-created BMesh loops stay automatic.
    clnor::transfer_custom_normals(
        mesh,
        &mut result,
        &normal_sources,
        NormalCodeOptions {
            corner_order: Some(&output_corner_order),
            ..NormalCodeOptions::default()
        },
    )?;
    Ok((result, kinds))
}
