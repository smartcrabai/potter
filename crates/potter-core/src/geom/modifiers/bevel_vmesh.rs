// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/bmesh/tools/bmesh_bevel.cc, source/blender/blenlib/intern/math_geom.cc, SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.

//! Blender's ADJ ("adjacent edges") vertex mesh for bevel corners, for `segments > 2`.
//!
//! A corner with `n` boundary vertices is replaced by a mesh of `(seg / 2 + 1) * (seg + 1)` points
//! per boundary vertex, indexed `(i, j, k)` as in Blender's `mesh_vert`: boundary vertex `i`, ring
//! `j`, and segment `k`. Points that Blender shares between neighbouring boundary vertices are
//! reached through `mesh_vert_canon`. The mesh comes from cubic subdivision of a two-segment control
//! mesh (`cubic_subdiv`), followed by interpolation for odd segment counts (`interp_vmesh`).
//!
//! Covered: EDGES bevels whose beveled corners are all three-edge cube-like corners
//! (`tri_corner_adj_vmesh`, or `square_out_adj_vmesh` for the square profile with an even segment
//! count), and VERTICES bevels (`adj_vmesh`).

use std::collections::{HashMap, HashSet};
use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};

use glam::DVec3;

use super::{Mesh, Polygon, build_mesh};
use crate::error::Result;
use crate::geom::uv_layers;
use crate::geom::{
    edge_key,
    provenance::{Origin, mean_value_corners},
};

/// Superellipse exponent of a square profile (Blender `PRO_SQUARE_R`).
pub(super) const PRO_SQUARE_R: f64 = 1.0e4;
/// Superellipse exponent of a circular profile (Blender `PRO_CIRCLE_R`).
pub(super) const PRO_CIRCLE_R: f64 = 2.0;
/// Superellipse exponent of a linear profile (Blender `PRO_LINE_R`).
pub(super) const PRO_LINE_R: f64 = 1.0;
/// Superellipse exponent of an inward square profile (Blender `PRO_SQUARE_IN_R`).
pub(super) const PRO_SQUARE_IN_R: f64 = 0.0;

const BEVEL_EPSILON: f64 = 1.0e-6;
const BEVEL_EPSILON_SQ: f64 = 1.0e-12;
const BEVEL_EPSILON_ANG: f64 = 2.0 * PI / 180.0;
const CIRCLE_FULLNESS: [f64; 11] = [
    0.0, 0.559, 0.642, 0.551, 0.646, 0.624, 0.646, 0.619, 0.647, 0.639, 0.647,
];

/// Superellipse exponent for a `profile` setting, computed in single precision as Blender does and
/// snapped to the special cases.
pub(super) fn profile_super_r(profile: f64) -> f64 {
    let profile = profile as f32;
    if profile >= 0.950 {
        return PRO_SQUARE_R;
    }
    let r = f64::from(-(2.0_f32).ln() / profile.sqrt().ln());
    if (r - PRO_CIRCLE_R).abs() < 1.0e-4 {
        PRO_CIRCLE_R
    } else if (r - PRO_LINE_R).abs() < 1.0e-4 {
        PRO_LINE_R
    } else if r < 1.0e-4 {
        PRO_SQUARE_IN_R
    } else {
        r
    }
}

/// Blender's `find_profile_fullness` for a non-custom profile.
fn profile_fullness(seg: usize, profile: f64, super_r: f64) -> f64 {
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

fn power_of_two_at_least(value: usize) -> usize {
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
fn even_chords(n: usize, r: f64) -> (Vec<f64>, Vec<f64>) {
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
struct ProfileSpacing {
    seg: usize,
    seg_2: usize,
    super_r: f64,
    xvals: Vec<f64>,
    yvals: Vec<f64>,
    xvals_2: Vec<f64>,
    yvals_2: Vec<f64>,
}

impl ProfileSpacing {
    fn new(seg: usize, super_r: f64) -> Self {
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
struct Profile {
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

/// Blender's `VMesh`: `count` boundary vertices, each with `(seg / 2 + 1) * (seg + 1)` points.
#[derive(Clone, Debug)]
struct VMesh {
    count: usize,
    seg: usize,
    co: Vec<DVec3>,
}

impl VMesh {
    fn new(count: usize, seg: usize) -> Self {
        Self {
            count,
            seg,
            co: vec![DVec3::ZERO; count * (seg / 2 + 1) * (seg + 1)],
        }
    }

    /// Blender's `mesh_vert` storage index.
    fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i * (self.seg + 1) * (self.seg / 2 + 1) + j * (self.seg + 1) + k
    }

    fn get(&self, i: usize, j: usize, k: usize) -> DVec3 {
        self.co[self.index(i, j, k)]
    }

    fn put(&mut self, i: usize, j: usize, k: usize, co: DVec3) {
        let index = self.index(i, j, k);
        self.co[index] = co;
    }

    /// Blender's `mesh_vert_canon`.
    fn canon(&self, i: usize, j: usize, k: usize) -> (usize, usize, usize) {
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
    let mut total = 0.0;
    for k in 0..ns {
        total += (vm.get(i, 0, k) - vm.get(i, 0, k + 1)).length();
        frac[k + 1] = total;
    }
    if total > 0.0 {
        for value in frac.iter_mut().skip(1) {
            *value /= total;
        }
    } else {
        frac[ns] = 1.0;
    }
    frac
}

/// Blender's `fill_profile_fracs`: cumulative fractions of a profile sampled with `ns` segments.
fn profile_fracs(profile: &Profile, ns: usize) -> Vec<f64> {
    let mut frac = vec![0.0; ns + 1];
    let mut total = 0.0;
    let mut co = profile.start;
    for k in 0..ns {
        let next = profile.point(k + 1, ns);
        total += (co - next).length();
        frac[k + 1] = total;
        co = next;
    }
    if total > 0.0 {
        for value in frac.iter_mut().skip(1) {
            *value /= total;
        }
    } else {
        frac[ns] = 1.0;
    }
    frac
}

/// Blender's `interp_range`: the `i` with `frac[i] <= f <= frac[i + 1]`, and the fraction of the
/// way from `frac[i]` to `frac[i + 1]`. Returns `n` when `f` lies beyond the last entry.
#[expect(
    clippy::float_cmp,
    reason = "mirrors Blender's exact `*r_rest == 1.0f` test in interp_range"
)]
fn interp_range(frac: &[f64], n: usize, f: f64) -> (usize, f64) {
    for i in 0..n {
        if f <= frac[i + 1] {
            let rest = f - frac[i];
            let r = if rest == 0.0 {
                0.0
            } else {
                rest / (frac[i + 1] - frac[i])
            };
            if i == n - 1 && r == 1.0 {
                return (n, 0.0);
            }
            return (i, r);
        }
    }
    (n, 0.0)
}

/// Blender's `interp_bilinear_quad_v3`.
fn interp_bilinear_quad(quad: [DVec3; 4], u: f64, v: f64) -> DVec3 {
    quad[0] * ((1.0 - u) * (1.0 - v))
        + quad[1] * (u * (1.0 - v))
        + quad[2] * (u * v)
        + quad[3] * ((1.0 - u) * v)
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
                if restj > -BEVEL_EPSILON {
                    restj = 0.0;
                } else {
                    j_in -= 1;
                    restj += 1.0;
                }
                let co = if restj < BEVEL_EPSILON && restk < BEVEL_EPSILON {
                    vm_in.canon_get(i, j_in, k_in)
                } else {
                    let j0inc = usize::from(!(restj < BEVEL_EPSILON || j_in == ns_in));
                    let k0inc = usize::from(!(restk < BEVEL_EPSILON || k_in == ns_in));
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
fn cubic_subdiv(vm_in: &mut VMesh, profiles: &[Profile]) -> VMesh {
    let n_boundary = vm_in.count;
    let ns_in = vm_in.seg;
    let ns_in2 = ns_in / 2;
    let ns_out = 2 * ns_in;
    let mut vm_out = VMesh::new(n_boundary, ns_out);

    for i in 0..n_boundary {
        vm_out.put(i, 0, 0, vm_in.get(i, 0, 0));
        for k in 1..ns_in {
            let co = vm_in.get(i, 0, k);
            let acc = (vm_in.get(i, 0, k - 1) + vm_in.get(i, 0, k + 1)) + co * -2.0;
            vm_out.canon_put(i, 0, 2 * k, co + acc * (-1.0 / 6.0));
        }
    }
    for (i, profile) in profiles.iter().enumerate().take(n_boundary) {
        for k in (1..ns_out).step_by(2) {
            let co = profile.point(k, ns_out);
            let acc = (vm_out.canon_get(i, 0, k - 1) + vm_out.canon_get(i, 0, k + 1)) + co * -2.0;
            vm_out.canon_put(i, 0, k, co + acc * (-1.0 / 6.0));
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

/// Blender's `snap_to_superellipsoid` (no midline): projects an octant point onto the
/// superellipsoid of exponent `r`.
fn snap_to_superellipsoid(co: DVec3, r: f64) -> DVec3 {
    if r == PRO_CIRCLE_R {
        return co.normalize_or_zero();
    }
    let a = co.x.max(0.0);
    let b = co.y.max(0.0);
    let c = co.z.max(0.0);
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

/// Blender's `make_cube_corner_adj_vmesh`: the octant of a cube corner, with boundary vertices on
/// the unit axes and profiles on the octant's faces, snapped to the superellipsoid.
fn cube_corner_adj_vmesh(spacing: &ProfileSpacing) -> VMesh {
    let nseg = spacing.seg;
    let r = spacing.super_r;
    if r == PRO_SQUARE_R {
        return cube_corner_square(nseg);
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
        vm1 = cubic_subdiv(&mut vm1, &profiles);
    }
    if vm1.seg != nseg {
        vm1 = interp_vmesh(&vm1, &profiles, nseg);
    }
    let ns2 = nseg / 2;
    for i in 0..3 {
        for j in 0..=ns2 {
            for k in 0..=nseg {
                let co = snap_to_superellipsoid(vm1.get(i, j, k), r);
                vm1.put(i, j, k, co);
            }
        }
    }
    vm1
}

/// Blender's `tri_corner_adj_vmesh`: the cube-corner octant mapped onto the corner. The map sends
/// the unit axes to `bounds` and the unit cube's far corner to `vertex` (`make_unit_cube_map`).
fn cube_corner_vmesh(spacing: &ProfileSpacing, bounds: [DVec3; 3], vertex: DVec3) -> VMesh {
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

/// Blender's `adj_vmesh` for the general case: a two-segment control mesh whose boundary vertices
/// are `bounds` and whose profiles pass through `vertex`, subdivided up to `seg` segments.
fn general_vmesh(
    spacing: &ProfileSpacing,
    fullness: f64,
    bounds: &[DVec3],
    vertex: DVec3,
) -> VMesh {
    let n_bndv = bounds.len();
    let mut profiles = Vec::with_capacity(n_bndv);
    let mut vm0 = VMesh::new(n_bndv, 2);
    let mut boundverts_center = DVec3::ZERO;
    for (i, &start) in bounds.iter().enumerate() {
        let end = bounds[(i + 1) % n_bndv];
        let profile = Profile::new(spacing, start, end, vertex);
        vm0.put(i, 0, 0, start);
        vm0.put(i, 0, 1, profile.point(1, 2));
        boundverts_center += start;
        profiles.push(profile);
    }
    boundverts_center *= 1.0 / n_bndv as f64;
    let center_direction = vertex - boundverts_center;
    if center_direction.length_squared() > BEVEL_EPSILON_SQ {
        vm0.put(0, 1, 1, boundverts_center + center_direction * fullness);
    } else {
        vm0.put(0, 1, 1, boundverts_center);
    }
    vm0.copy_equiv();

    let mut vm1 = vm0;
    loop {
        vm1 = cubic_subdiv(&mut vm1, &profiles);
        if vm1.seg >= spacing.seg {
            break;
        }
    }
    if vm1.seg != spacing.seg {
        vm1 = interp_vmesh(&vm1, &profiles, spacing.seg);
    }
    vm1
}

/// Closest point of the segment `a`-`b` to `point` (Blender's `closest_to_line_segment_v3`).
fn closest_on_segment(point: DVec3, a: DVec3, b: DVec3) -> DVec3 {
    let direction = b - a;
    let length_squared = direction.length_squared();
    if length_squared == 0.0 {
        return a;
    }
    let t = ((point - a).dot(direction) / length_squared).clamp(0.0, 1.0);
    a + direction * t
}

/// Blender's `isect_line_line_epsilon_v3`: `(kind, r_i1, r_i2)` for the lines `v1`-`v2` and `v3`-`v4`.
/// `kind` is 0 for parallel lines, 1 for coplanar lines meeting at `r_i1 == r_i2`, and 2 for skew
/// lines, whose nearest points are `r_i1` on the first line and `r_i2` on the second.
fn isect_line_line(v1: DVec3, v2: DVec3, v3: DVec3, v4: DVec3) -> (u8, DVec3, DVec3) {
    let a = v2 - v1;
    let b = v4 - v3;
    let ab = a.cross(b);
    let div = ab.dot(ab);
    if div == 0.0 {
        return (0, DVec3::ZERO, DVec3::ZERO);
    }
    let c = v3 - v1;
    if c.dot(ab).abs() <= BEVEL_EPSILON {
        let point = v1 + a * (c.cross(b).dot(ab) / div);
        return (1, point, point);
    }
    // Skew: slide the second line along the common normal into the plane of the first line.
    let offset = (v1 - v3).project_onto(ab);
    let c = c + offset;
    let r_i1 = v1 + a * (c.cross(b).dot(ab) / div);
    (2, r_i1, r_i1 - offset)
}

/// Blender's `edges_angle_kind` test for the corner edges `vertex`-`first` and `vertex`-`second`:
/// whether the angle between them, measured through the face with `normal`, is the smaller one.
/// Straight edges are not smaller.
fn angle_is_smaller(vertex: DVec3, first: DVec3, second: DVec3, normal: DVec3) -> bool {
    let cross = (vertex - first)
        .normalize_or_zero()
        .cross((vertex - second).normalize_or_zero());
    cross.length_squared() > BEVEL_EPSILON_SQ && cross.normalize().dot(normal) >= 0.0
}

/// Blender's `square_out_adj_vmesh`: the vertex mesh of a three-edge corner for the square profile
/// with an even segment count. Each boundary vertex moves onto the centerlines of the beveled edges,
/// where its offset lines meet them, and the centerlines run from the edges to the vertex. Corner
/// edge `i` joins the vertex to `neighbors[i]`, and centerline `i` lies on that edge.
fn square_out_vmesh(points: &[DVec3], polygons: &[Polygon], corner: &Corner, seg: usize) -> VMesh {
    let count = corner.count();
    let ns2 = seg / 2;
    let ns2_inv = 1.0 / ns2 as f64;
    let vertex = points[corner.vertex];
    let neighbor = |edge: usize| points[corner.neighbors[edge % count]];
    // Blender's closer_v3_v3v3v3: keep the candidate that is nearer the vertex.
    let nearer = |current: DVec3, candidate: DVec3| {
        if (current - vertex).length_squared() <= (candidate - vertex).length_squared() {
            current
        } else {
            candidate
        }
    };
    // centerline[i][j]: j = 0 on edge i, j = ns2 at the vertex.
    let mut centerline = vec![vec![DVec3::ZERO; ns2 + 1]; count];
    let mut set = vec![false; count];
    for i in 0..count {
        let previous = (i + count - 1) % count;
        let bound = corner.bounds[i];
        let first = neighbor(previous);
        let second = neighbor(i);
        let normal = face_normal(points, &polygons[corner.bound_face(i)].0);
        if !angle_is_smaller(vertex, first, second, normal) {
            continue;
        }
        let (kind, on_second, _) = isect_line_line(vertex, second, bound, bound + (first - vertex));
        if kind != 0 {
            centerline[i][0] = if set[i] {
                nearer(centerline[i][0], on_second)
            } else {
                on_second
            };
            set[i] = true;
        }
        let (kind, on_first, _) = isect_line_line(vertex, first, bound, bound + (second - vertex));
        if kind != 0 {
            centerline[previous][0] = if set[previous] {
                nearer(centerline[previous][0], on_first)
            } else {
                on_first
            };
            set[previous] = true;
        }
    }
    for i in 0..count {
        if !set[i] {
            // Blender's fallback: the point of edge `i` nearest to the next boundary vertex.
            centerline[i][0] =
                closest_on_segment(corner.bounds[(i + 1) % count], vertex, neighbor(i));
        }
    }
    for line in &mut centerline {
        let start = line[0];
        for (j, point) in line.iter_mut().enumerate().skip(1) {
            *point = start + (vertex - start) * (j as f64 * ns2_inv);
        }
    }

    let mut vm = VMesh::new(count, seg);
    for i in 0..count {
        let bound = corner.bounds[i];
        let before = centerline[(i + count - 1) % count][0];
        for j in 0..ns2 {
            vm.put(i, j, 0, bound + (before - bound) * (j as f64 * ns2_inv));
        }
        let after = centerline[i][0];
        for k in 1..=ns2 {
            vm.put(i, 0, k, bound + (after - bound) * (k as f64 * ns2_inv));
        }
    }
    vm.put(0, ns2, ns2, vertex);
    vm.copy_equiv();
    for i in 0..count {
        let before = (i + count - 1) % count;
        for j in 1..ns2 {
            for k in 1..=ns2 {
                let (kind, meet1, meet2) = isect_line_line(
                    vm.get(i, 0, k),
                    centerline[before][k],
                    vm.get(i, j, 0),
                    centerline[i][j],
                );
                let point = match kind {
                    0 => {
                        let start = vm.get(i, 0, k);
                        start + (centerline[before][k] - start) * (j as f64 * ns2_inv)
                    }
                    1 => meet1,
                    _ => (meet1 + meet2) * 0.5,
                };
                vm.put(i, j, k, point);
            }
        }
    }
    vm.copy_equiv();
    vm
}

/// The vertex mesh of a three-edge corner for `spacing`: Blender's `square_out_adj_vmesh` for the
/// square profile with an even segment count (`bevel_build_rings`), otherwise the cube-corner map.
fn corner_vmesh(
    points: &[DVec3],
    polygons: &[Polygon],
    corner: &Corner,
    spacing: &ProfileSpacing,
) -> VMesh {
    if spacing.super_r == PRO_SQUARE_R && spacing.seg.is_multiple_of(2) {
        return square_out_vmesh(points, polygons, corner, spacing.seg);
    }
    cube_corner_vmesh(
        spacing,
        [corner.bounds[0], corner.bounds[1], corner.bounds[2]],
        points[corner.vertex],
    )
}

/// A beveled corner: its vertex, its edges and faces in Blender's CCW order, and its boundary
/// vertices. Edge `i` joins the corner to `neighbors[i]`; `faces[i]` lies between edge `i` and
/// edge `i + 1`; boundary vertex `i` lies on face `faces[i - 1]`. The ring from boundary vertex `i`
/// to `i + 1` runs along edge `i` (EDGES) or through face `faces[i]` (VERTICES).
#[derive(Clone, Debug)]
struct Corner {
    vertex: usize,
    neighbors: Vec<usize>,
    faces: Vec<usize>,
    bounds: Vec<DVec3>,
}

impl Corner {
    fn count(&self) -> usize {
        self.neighbors.len()
    }

    fn edge_index(&self, neighbor: usize) -> Option<usize> {
        self.neighbors.iter().position(|other| *other == neighbor)
    }

    /// The face that boundary vertex `i` lies on.
    fn bound_face(&self, i: usize) -> usize {
        self.faces[(i + self.count() - 1) % self.count()]
    }
}

/// Edge and face cycles around `vertex`, from the faces that contain it. `None` unless those faces
/// form one closed fan, which holds only at an interior manifold vertex.
fn corner_topology(vertex: usize, polygons: &[Polygon]) -> Option<(Vec<usize>, Vec<usize>)> {
    // Keyed by the neighbour `next` that follows the vertex in a face: the face's previous neighbour
    // `previous` and the face index. Stepping `next -> previous` walks the fan in Blender's CCW order.
    let mut successor: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut first = None;
    for (face_index, (face, _)) in polygons.iter().enumerate() {
        let Some(position) = face.iter().position(|value| *value == vertex) else {
            continue;
        };
        let count = face.len();
        let previous = face[(position + count - 1) % count];
        let next = face[(position + 1) % count];
        if successor.insert(next, (previous, face_index)).is_some() {
            return None;
        }
        first.get_or_insert(next);
    }
    let start = first?;
    let mut neighbors = Vec::with_capacity(successor.len());
    let mut faces = Vec::with_capacity(successor.len());
    let mut current = start;
    loop {
        neighbors.push(current);
        let (previous, face) = *successor.get(&current)?;
        faces.push(face);
        current = previous;
        if current == start {
            break;
        }
        if neighbors.len() > successor.len() {
            return None;
        }
    }
    (neighbors.len() == successor.len()).then_some((neighbors, faces))
}

/// Number of distinct edges that meet `vertex` in the faces of `polygons`.
fn valence(vertex: usize, polygons: &[Polygon]) -> usize {
    let mut neighbors = HashSet::new();
    for (face, _) in polygons {
        let count = face.len();
        for (position, value) in face.iter().enumerate() {
            if *value == vertex {
                neighbors.insert(face[(position + count - 1) % count]);
                neighbors.insert(face[(position + 1) % count]);
            }
        }
    }
    neighbors.len()
}

/// Newell normal of the face `face` of `points`, normalized.
fn face_normal(points: &[DVec3], face: &[usize]) -> DVec3 {
    let mut normal = DVec3::ZERO;
    for (index, vertex) in face.iter().enumerate() {
        normal += points[*vertex].cross(points[face[(index + 1) % face.len()]]);
    }
    normal.normalize_or_zero()
}

fn face_centroid(points: &[DVec3], face: &[usize]) -> DVec3 {
    face.iter()
        .fold(DVec3::ZERO, |sum, vertex| sum + points[*vertex])
        / face.len() as f64
}

/// Blender's `tri_corner_test` for a three-edge corner whose edges are all beveled: the three
/// dihedral angles are near right angles, and the offsets are equal.
fn is_cube_corner(corner: &Corner, points: &[DVec3], polygons: &[Polygon], super_r: f64) -> bool {
    let vertex = points[corner.vertex];
    let offset = (corner.bounds[0] - vertex).length();
    if corner
        .bounds
        .iter()
        .any(|bound| ((*bound - vertex).length() - offset).abs() > BEVEL_EPSILON)
    {
        return false;
    }
    let normals: HashMap<usize, DVec3> = corner
        .faces
        .iter()
        .map(|face| (*face, face_normal(points, &polygons[*face].0)))
        .collect();
    let mut in_plane = 0;
    let mut total_angle = 0.0;
    for i in 0..corner.count() {
        let before = corner.bound_face(i);
        let after = corner.faces[i];
        let (normal_before, normal_after) = (normals[&before], normals[&after]);
        let angle = PI - normal_before.dot(normal_after).clamp(-1.0, 1.0).acos();
        let midpoint = (vertex + points[corner.neighbors[i]]) * 0.5;
        let convex =
            (face_centroid(points, &polygons[before].0) - midpoint).dot(normal_after) < 0.0;
        let signed = if convex { angle } else { -angle };
        let absolute = signed.abs();
        if absolute <= PI / 4.0 {
            in_plane += 1;
        } else if absolute >= 3.0 * PI / 4.0 {
            return false;
        }
        total_angle += signed;
    }
    if in_plane != 0 {
        return false;
    }
    let angdiff = (total_angle.abs() - 3.0 * FRAC_PI_2).abs();
    !((super_r == PRO_SQUARE_R && angdiff > PI / 16.0) || angdiff > PI / 4.0)
}

/// Output points, origins, and polygons of a rebuilt bevel. The first points are the input points,
/// and [`finish`] drops the ones that no polygon or loose edge uses.
pub(super) struct Emission {
    points: Vec<DVec3>,
    origins: Vec<usize>,
    polygons: Vec<OutPolygon>,
}

/// An output polygon: per corner, the output point, the point its UV is interpolated at, and the
/// face whose UVs are interpolated there. `face` supplies the polygon's material.
struct OutPolygon {
    vertices: Vec<usize>,
    queries: Vec<DVec3>,
    sources: Vec<Option<usize>>,
    face: usize,
}

impl Emission {
    fn new(source_points: &[DVec3]) -> Self {
        Self {
            points: source_points.to_vec(),
            origins: (0..source_points.len()).collect(),
            polygons: Vec::new(),
        }
    }

    fn point(&self, id: usize) -> DVec3 {
        self.points[id]
    }

    fn push_point(&mut self, co: DVec3, origin: usize) -> usize {
        self.points.push(co);
        self.origins.push(origin);
        self.points.len() - 1
    }

    /// Appends a polygon. `snapped` holds, per corner, the segment that the corner's UV query is
    /// projected onto (Blender's `bev_create_ngon` snap edges), if any.
    fn push_polygon(
        &mut self,
        vertices: Vec<usize>,
        snapped: impl AsRef<[Option<(DVec3, DVec3)>]>,
        sources: Vec<Option<usize>>,
        face: usize,
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
        });
    }
}

/// Output ids of one corner's vertex mesh; interior points are created on first use.
struct CornerIds {
    vertex: usize,
    vm: VMesh,
    bound_ids: Vec<usize>,
    interior: HashMap<(usize, usize, usize), usize>,
}

impl CornerIds {
    fn new(corner: &Corner, vm: VMesh, emission: &mut Emission) -> Self {
        let bound_ids = corner
            .bounds
            .iter()
            .map(|co| emission.push_point(*co, corner.vertex))
            .collect();
        Self {
            vertex: corner.vertex,
            vm,
            bound_ids,
            interior: HashMap::new(),
        }
    }

    fn id(&mut self, emission: &mut Emission, i: usize, j: usize, k: usize) -> usize {
        let (a, b, c) = self.vm.canon(i, j, k);
        if b == 0 && c == 0 {
            return self.bound_ids[a];
        }
        if let Some(&id) = self.interior.get(&(a, b, c)) {
            return id;
        }
        let id = emission.push_point(self.vm.get(a, b, c), self.vertex);
        self.interior.insert((a, b, c), id);
        id
    }
}

/// A beveled vertex that the bevel's vertex mesh does not cover.
#[derive(Clone, Debug)]
pub(super) struct CornerRejection {
    /// Input point index of the vertex.
    pub(super) vertex: usize,
    /// Number of distinct edges at the vertex.
    pub(super) valence: usize,
    /// Number of selected edges at the vertex.
    pub(super) selected: usize,
    pub(super) reason: &'static str,
}

/// The three-edge cube-like corners of an EDGES bevel with the selected edges, in vertex order, or
/// the corners that the vertex-mesh construction does not cover. `bound_of(face, vertex)` is the
/// bevel's boundary point of a face corner.
fn gather_corners(
    points: &[DVec3],
    polygons: &[Polygon],
    selected_edges: &HashSet<(usize, usize)>,
    bound_of: &dyn Fn(usize, usize) -> DVec3,
    super_r: f64,
) -> std::result::Result<Vec<Corner>, Vec<CornerRejection>> {
    let mut edges: Vec<(usize, usize)> = selected_edges.iter().copied().collect();
    edges.sort_unstable();
    let mut vertices: Vec<usize> = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
    vertices.sort_unstable();
    vertices.dedup();

    let mut rejections = Vec::new();
    let mut corners: Vec<Corner> = Vec::new();
    for &vertex in &vertices {
        let selected = edges
            .iter()
            .filter(|(a, b)| *a == vertex || *b == vertex)
            .count();
        let Some((neighbors, faces)) = corner_topology(vertex, polygons) else {
            rejections.push(CornerRejection {
                vertex,
                valence: valence(vertex, polygons),
                selected,
                reason: "the faces around the vertex do not form one closed fan",
            });
            continue;
        };
        if neighbors.len() != 3 {
            rejections.push(CornerRejection {
                vertex,
                valence: neighbors.len(),
                selected,
                reason: "only three-edge corners are supported",
            });
            continue;
        }
        if selected != 3 {
            rejections.push(CornerRejection {
                vertex,
                valence: 3,
                selected,
                reason: "all three edges at the corner must be beveled",
            });
            continue;
        }
        let bounds = (0..3)
            .map(|i| bound_of(faces[(i + 2) % 3], vertex))
            .collect();
        let corner = Corner {
            vertex,
            neighbors,
            faces,
            bounds,
        };
        if !is_cube_corner(&corner, points, polygons, super_r) {
            rejections.push(CornerRejection {
                vertex,
                valence: 3,
                selected,
                reason: "the corner is not cube-like (dihedral angles or offsets differ)",
            });
            continue;
        }
        corners.push(corner);
    }
    if super_r == PRO_SQUARE_IN_R {
        // Blender's build_square_in_vmesh merges boundary vertices across corners: not ported.
        rejections.extend(corners.iter().map(|corner| CornerRejection {
            vertex: corner.vertex,
            valence: corner.count(),
            selected: corner.count(),
            reason: "profile 0 (square inward) merges boundary vertices across corners, which is not supported",
        }));
    }
    if !rejections.is_empty() {
        return Err(rejections);
    }
    Ok(corners)
}

/// The beveled corners that the EDGES vertex-mesh construction does not cover (empty when all
/// corners are three-edge cube-like corners). Used to guard UV layers, whose rules are verified
/// only for those corners.
pub(super) fn cube_corner_rejections(
    points: &[DVec3],
    polygons: &[Polygon],
    selected_edges: &HashSet<(usize, usize)>,
    bound_of: &dyn Fn(usize, usize) -> DVec3,
    super_r: f64,
) -> Vec<CornerRejection> {
    gather_corners(points, polygons, selected_edges, bound_of, super_r)
        .err()
        .unwrap_or_default()
}

/// Rebuilds an EDGES bevel whose beveled corners are all three-edge cube-like corners: each beveled
/// edge becomes a strip of `seg` quads between its corners' rings, and each corner becomes a vertex
/// mesh. `bound_of(face, vertex)` is the bevel's boundary point of a face corner. Otherwise returns
/// the corners that are not covered, which is never empty.
pub(super) fn bevel_edges(
    mesh: &Mesh,
    seg: usize,
    super_r: f64,
    points: &[DVec3],
    polygons: &[Polygon],
    selected_edges: &HashSet<(usize, usize)>,
    bound_of: &dyn Fn(usize, usize) -> DVec3,
) -> std::result::Result<Emission, Vec<CornerRejection>> {
    let mut edges: Vec<(usize, usize)> = selected_edges.iter().copied().collect();
    edges.sort_unstable();
    let corners = gather_corners(points, polygons, selected_edges, bound_of, super_r)?;
    let corner_of: HashMap<usize, usize> = corners
        .iter()
        .enumerate()
        .map(|(index, corner)| (corner.vertex, index))
        .collect();

    let spacing = ProfileSpacing::new(seg, super_r);
    let mut emission = Emission::new(points);
    let mut ids: Vec<CornerIds> = Vec::with_capacity(corners.len());
    for corner in &corners {
        let vm = corner_vmesh(points, polygons, corner, &spacing);
        ids.push(CornerIds::new(corner, vm, &mut emission));
    }

    // Original faces: each beveled corner becomes its boundary vertex on that face.
    for (face_index, (face, _)) in polygons.iter().enumerate() {
        let mut vertices = Vec::with_capacity(face.len());
        for &vertex in face {
            match corner_of.get(&vertex) {
                Some(&index) => {
                    let corner = &corners[index];
                    let Some(i) = (0..corner.count()).find(|i| corner.bound_face(*i) == face_index)
                    else {
                        return Err(vec![CornerRejection {
                            vertex: corner.vertex,
                            valence: corner.count(),
                            selected: corner.count(),
                            reason: "the bevel boundary point is missing from a face around the corner",
                        }]);
                    };
                    vertices.push(ids[index].bound_ids[i]);
                }
                None => vertices.push(vertex),
            }
        }
        emission.push_polygon(
            vertices,
            [None; 4],
            vec![Some(face_index); face.len()],
            face_index,
        );
    }

    // UV seams (`EdgeHalf::is_seam`) select seam-specific branches for odd segment counts. Blender's
    // face components decide the representative faces for those branches.
    let uvs = source_uvs(mesh);
    let components = if seg % 2 == 1 && !uvs.is_empty() {
        face_components(points, polygons, &uvs)
    } else {
        Vec::new()
    };
    let mid = seg / 2;
    let ns2 = seg / 2;
    let odd = seg % 2;

    // Strips along each beveled edge (`bevel_build_edge_polygons`). `face_a` is the face the ring at
    // `a` starts on (Blender's `e1->fprev`), `face_b` the other face of the edge.
    for &(a, b) in &edges {
        let index_a = corner_of[&a];
        let index_b = corner_of[&b];
        let ring_a = corners[index_a].edge_index(b).unwrap_or(0);
        let face_a = corners[index_a].bound_face(ring_a);
        let face_b = corners[index_a].faces[ring_a];
        let ring_b = corners[index_b].edge_index(a).unwrap_or(0);
        let forward_b = corners[index_b].bound_face(ring_b) == face_a;
        let side_a: Vec<usize> = (0..=seg)
            .map(|s| ids[index_a].id(&mut emission, ring_a, 0, s))
            .collect();
        let side_b: Vec<usize> = (0..=seg)
            .map(|s| {
                let k = if forward_b { s } else { seg - s };
                ids[index_b].id(&mut emission, ring_b, 0, k)
            })
            .collect();
        let edge = Some((points[a], points[b]));
        let seam = edge_is_seam(&uvs, polygons, a, b);
        // Odd seams: the straddling strip interpolates in one representative face, and the loops in
        // the other face snap to the edge. `center_adj` is the neighbouring strip that does the same.
        let straddle_face = if odd == 1 && seam {
            choose_representative(&[face_a, face_b], points, polygons, &components)
        } else {
            face_a
        };
        let center_adj = if straddle_face == face_a {
            mid + 2
        } else {
            mid
        };
        for s in 0..seg {
            let k = s + 1;
            let vertices = vec![side_a[s], side_b[s], side_b[s + 1], side_a[s + 1]];
            let (sources, snapped) = if odd == 1 {
                if k == mid + 1 {
                    if seam {
                        let snapped = if straddle_face == face_a {
                            [None, None, edge, edge]
                        } else {
                            [edge, edge, None, None]
                        };
                        ([straddle_face; 4], snapped)
                    } else {
                        ([face_a, face_a, face_b, face_b], [None; 4])
                    }
                } else if seam && k == center_adj {
                    if k == mid {
                        ([face_a; 4], [None, None, edge, edge])
                    } else {
                        ([face_b; 4], [edge, edge, None, None])
                    }
                } else if k <= mid {
                    ([face_a; 4], [None; 4])
                } else {
                    ([face_b; 4], [None; 4])
                }
            } else if k <= mid {
                let snapped = if k == mid {
                    [None, None, edge, edge]
                } else {
                    [None; 4]
                };
                ([face_a; 4], snapped)
            } else {
                let snapped = if k == mid + 1 {
                    [edge, edge, None, None]
                } else {
                    [None; 4]
                };
                ([face_b; 4], snapped)
            };
            emission.push_polygon(vertices, snapped, sources.map(Some).to_vec(), face_a);
        }
    }

    // Vertex mesh rings and centers of each corner (`bevel_build_rings`).
    for (index, corner) in corners.iter().enumerate() {
        let n = corner.count();
        let vertex_point = points[corner.vertex];
        let rep: Vec<usize> = (0..n).map(|i| corner.bound_face(i)).collect();
        // Blender's `EdgeHalf::is_seam` for each beveled edge of the corner.
        let seam: Vec<bool> = (0..n)
            .map(|i| edge_is_seam(&uvs, polygons, corner.vertex, corner.neighbors[i]))
            .collect();
        let any_seam = seam.iter().any(|flag| *flag);
        // Odd segments: `frep_beats_next` (which face of each boundary pair wins the tie) and the
        // center's representative face `center_frep`.
        let beats: Vec<bool> = if odd == 1 {
            (0..n)
                .map(|i| {
                    choose_representative(
                        &[rep[i], rep[(i + 1) % n]],
                        points,
                        polygons,
                        &components,
                    ) == rep[i]
                })
                .collect()
        } else {
            Vec::new()
        };
        let center_frep = if odd == 1 && any_seam {
            center_representative(corner, points, polygons, &components)
        } else {
            rep[0]
        };
        // Blender's `snap_edges_for_vmesh_vert` for odd segments: a ring point snaps to the seam edge
        // that Blender's tie-break assigns to it. Edge `i` is `corner.neighbors[i]` when a seam.
        let edge_of = |i: usize| seam[i].then(|| corner.neighbors[i]);
        let snap_center = |idx: usize, near: Option<usize>, far: Option<usize>| {
            let prev = (idx + n - 1) % n;
            let next = (idx + 1) % n;
            if beats[prev] && rep[prev] == center_frep {
                near
            } else if !beats[idx] && rep[next] == center_frep {
                far
            } else {
                None
            }
        };
        let snap_for = |i: usize, jj: usize, kk: usize| -> Option<usize> {
            let previ = (i + n - 1) % n;
            let (before, own, after) = (edge_of(previ), edge_of(i), edge_of((i + 1) % n));
            if jj < ns2 && kk < ns2 {
                None
            } else if jj < ns2 && kk == ns2 {
                if beats[i] { None } else { own }
            } else if jj < ns2 && kk == ns2 + 1 {
                if beats[i] { own } else { None }
            } else if jj == ns2 && kk < ns2 {
                if beats[previ] { before } else { None }
            } else if jj == ns2 && kk == ns2 {
                snap_center(i, before, own)
            } else if jj == ns2 && kk == ns2 + 1 {
                snap_center((i + 1) % n, own, after)
            } else {
                None
            }
        };
        let seg_of = |neighbor: Option<usize>| neighbor.map(|other| (vertex_point, points[other]));
        for i in 0..n {
            let f = rep[i];
            let f2 = rep[(i + 1) % n];
            let along_here = Some((vertex_point, points[corner.neighbors[i]]));
            let along_prev = Some((vertex_point, points[corner.neighbors[(i + n - 1) % n]]));
            for j in 0..ns2 {
                for k in 0..(ns2 + odd) {
                    let vertices = vec![
                        ids[index].id(&mut emission, i, j, k),
                        ids[index].id(&mut emission, i, j, k + 1),
                        ids[index].id(&mut emission, i, j + 1, k + 1),
                        ids[index].id(&mut emission, i, j + 1, k),
                    ];
                    let (sources, snapped) = if odd == 0 {
                        let c1 = if k == ns2 - 1 { along_here } else { None };
                        let c3 = if j == ns2 - 1 { along_prev } else { None };
                        let c2 = if k == ns2 - 1 {
                            along_here
                        } else if j == ns2 - 1 {
                            along_prev
                        } else {
                            None
                        };
                        ([f; 4], [None, c1, c2, c3])
                    } else {
                        // Blender's column `k == ns2` takes the seam's `frep_beats_next` face, or the
                        // split between `f` and `f2` when the edge is not a seam.
                        let sources = if k == ns2 && seam[i] {
                            let fc = if beats[i] { f } else { f2 };
                            [fc; 4]
                        } else if k == ns2 {
                            [f, f2, f2, f]
                        } else {
                            [f; 4]
                        };
                        let snapped = [
                            seg_of(snap_for(i, j, k)),
                            seg_of(snap_for(i, j, k + 1)),
                            seg_of(snap_for(i, j + 1, k + 1)),
                            seg_of(snap_for(i, j + 1, k)),
                        ];
                        (sources, snapped)
                    };
                    emission.push_polygon(vertices, snapped, sources.map(Some).to_vec(), f);
                }
            }
        }
        if odd == 1 {
            let vertices: Vec<usize> = (0..n)
                .map(|i| ids[index].id(&mut emission, i, ns2, ns2))
                .collect();
            // Blender's `center_face_interps`: the seam's representative face, or each boundary's own.
            let sources: Vec<Option<usize>> = (0..n)
                .map(|i| Some(if any_seam { center_frep } else { rep[i] }))
                .collect();
            let snapped: Vec<Option<(DVec3, DVec3)>> =
                (0..n).map(|i| seg_of(snap_for(i, ns2, ns2))).collect();
            let face = sources[0].unwrap_or(rep[0]);
            emission.push_polygon(vertices, snapped, sources, face);
        }
    }
    Ok(emission)
}

const BEVEL_EPSILON_BIG: f64 = 1.0e-4;

/// Blender's `CustomData_data_equals` over every UV layer, for two input loops `(face, corner)`.
fn uv_equal(uvs: &SourceUvs, first: (usize, usize), second: (usize, usize)) -> bool {
    uvs.iter().all(|layer| {
        let a = layer.get(first.0).and_then(Option::as_ref).and_then(|face| face.get(first.1));
        let b = layer.get(second.0).and_then(Option::as_ref).and_then(|face| face.get(second.1));
        // `layerEqual_propfloat2`: squared distance below 1e-5.
        matches!((a, b), (Some(a), Some(b)) if (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) < 1.0e-5)
    })
}

/// Blender's `contig_ldata_around_vert`: every input loop around `vertex` has the same UVs.
fn uv_contiguous_around(uvs: &SourceUvs, source_polygons: &[Polygon], vertex: usize) -> bool {
    let mut first = None;
    for (face_index, (face, _)) in source_polygons.iter().enumerate() {
        for (corner, value) in face.iter().enumerate() {
            if *value != vertex {
                continue;
            }
            match first {
                None => first = Some((face_index, corner)),
                Some(reference) => {
                    if !uv_equal(uvs, reference, (face_index, corner)) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Blender's `EdgeHalf::is_seam`: `!contig_ldata_across_edge`. The edge is a UV seam when its two
/// faces disagree in UVs at either end, or when it is not shared by exactly two faces.
fn edge_is_seam(uvs: &SourceUvs, polygons: &[Polygon], a: usize, b: usize) -> bool {
    if uvs.is_empty() {
        return false;
    }
    let key = edge_key(a, b);
    let faces: Vec<usize> = polygons
        .iter()
        .enumerate()
        .filter(|(_, (face, _))| {
            let count = face.len();
            (0..count).any(|k| edge_key(face[k], face[(k + 1) % count]) == key)
        })
        .map(|(index, _)| index)
        .collect();
    match faces.as_slice() {
        &[first, second] => [a, b].into_iter().any(|vertex| {
            match (
                corner_of(polygons, first, vertex),
                corner_of(polygons, second, vertex),
            ) {
                (Some(x), Some(y)) => !uv_equal(uvs, (first, x), (second, y)),
                _ => true,
            }
        }),
        _ => true,
    }
}

/// Blender's `math_layer_info_init` components: faces joined across edges whose UVs agree at both
/// ends. Labels follow face order; the components of the top and bottom faces are then swapped to
/// labels 0 and 1.
fn face_components(points: &[DVec3], polygons: &[Polygon], uvs: &SourceUvs) -> Vec<usize> {
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
                for &other in &edge_faces[&edge_key(a, b)] {
                    if other == face_index || label[other] != usize::MAX {
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
                    if uv_equal(uvs, (face_index, a_here), (other, a_there))
                        && uv_equal(uvs, (face_index, b_here), (other, b_there))
                    {
                        label[other] = current;
                        stack.push(other);
                    }
                }
            }
        }
    }
    if components <= 1 {
        return label;
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
    label
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
fn bounds_center(points: &[DVec3], face: &[usize]) -> DVec3 {
    let mut low = DVec3::splat(f64::INFINITY);
    let mut high = DVec3::splat(f64::NEG_INFINITY);
    for vertex in face {
        low = low.min(points[*vertex]);
        high = high.max(points[*vertex]);
    }
    (low + high) * 0.5
}

/// Blender's `choose_rep_face` among `faces`: the lowest component, then the lowest material index,
/// then the lowest bounds centre in z, x, and y. Faces are unselected, so selection does not decide.
#[expect(
    clippy::float_cmp,
    reason = "Blender's choose_rep_face breaks ties on exact equality"
)]
fn choose_representative(
    faces: &[usize],
    points: &[DVec3],
    polygons: &[Polygon],
    components: &[usize],
) -> usize {
    let key = |face: usize| {
        let centre = bounds_center(points, &polygons[face].0);
        (
            components.get(face).copied().unwrap_or(0),
            polygons[face].1,
            centre.z,
            centre.x,
            centre.y,
        )
    };
    let mut best = faces[0];
    let mut best_key = key(best);
    for &face in &faces[1..] {
        let candidate = key(face);
        let less = candidate.0 < best_key.0
            || (candidate.0 == best_key.0
                && (candidate.1 < best_key.1
                    || (candidate.1 == best_key.1
                        && (candidate.2 < best_key.2
                            || (candidate.2 == best_key.2
                                && (candidate.3 < best_key.3
                                    || (candidate.3 == best_key.3
                                        && candidate.4 < best_key.4)))))));
        if less {
            best = face;
            best_key = candidate;
        }
    }
    best
}

/// The two neighbours of `vertex` along the edges of `face`, in the order Blender's
/// `get_incident_edges` finds them: the edge that starts the face's loop order comes first.
fn incident_neighbors(face: &[usize], vertex: usize) -> (usize, usize) {
    let count = face.len();
    let position = face.iter().position(|value| *value == vertex).unwrap_or(0);
    if position == 0 {
        (face[1 % count], face[count - 1])
    } else {
        (face[position - 1], face[(position + 1) % count])
    }
}

/// Squared distance from `p` to the segment `a`-`b`, evaluated in single precision as Blender's
/// `dist_squared_to_line_segment_v3` is, so that equal-distance ties break the same way.
fn dist_sq_to_segment_f32(p: DVec3, a: DVec3, b: DVec3) -> f32 {
    let (p, a, b) = (p.as_vec3(), a.as_vec3(), b.as_vec3());
    let direction = b - a;
    let length_squared = direction.dot(direction);
    let t = if length_squared == 0.0 {
        0.0
    } else {
        ((p - a).dot(direction) / length_squared).clamp(0.0, 1.0)
    };
    (a + direction * t - p).length_squared()
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
fn project_to_face(normal: DVec3, point: DVec3) -> (f64, f64) {
    let (n1, n2) = ortho_basis(normal);
    (n1.dot(point), n2.dot(point))
}

/// Blender's `BM_face_point_inside_test`: `isect_point_poly_v2` on the face projected along its
/// normal. It is a half-open even-odd crossing test, so boundary points can count as inside.
fn point_strictly_inside(points: &[DVec3], face: &[usize], normal: DVec3, point: DVec3) -> bool {
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

/// Blender's `projected_boundary_area`: the area of the corner's boundary vertices, each snapped to
/// the closer of the face's two edges at the vertex unless it lies inside the face, projected onto
/// the face's plane.
fn projected_boundary_area(
    corner: &Corner,
    face_index: usize,
    points: &[DVec3],
    polygons: &[Polygon],
) -> f64 {
    let face = &polygons[face_index].0;
    let normal = face_normal(points, face);
    let (near, far) = incident_neighbors(face, corner.vertex);
    let vertex_point = points[corner.vertex];
    let snapped: Vec<DVec3> = corner
        .bounds
        .iter()
        .map(|co| {
            if point_strictly_inside(points, face, normal, *co) {
                return *co;
            }
            let near_d = dist_sq_to_segment_f32(*co, vertex_point, points[near]);
            let far_d = dist_sq_to_segment_f32(*co, vertex_point, points[far]);
            let (near_point, far_point) = (
                closest_on_segment(*co, vertex_point, points[near]),
                closest_on_segment(*co, vertex_point, points[far]),
            );
            if near_d <= far_d {
                near_point
            } else {
                far_point
            }
        })
        .collect();
    let projected: Vec<(f64, f64)> = snapped
        .iter()
        .map(|point| project_to_face(normal, *point))
        .collect();
    let count = projected.len();
    let area: f64 = (0..count)
        .map(|k| {
            let (a, b) = (projected[k], projected[(k + 1) % count]);
            a.0 * b.1 - b.0 * a.1
        })
        .sum();
    (0.5 * area).abs()
}

/// Blender's `frep_for_center_poly` for VERTICES with an odd segment count: the representative face
/// of a corner's center, chosen among the faces at each edge and rejected when its boundary projects
/// to a degenerate area.
fn center_representative(
    corner: &Corner,
    points: &[DVec3],
    polygons: &[Polygon],
    components: &[usize],
) -> usize {
    let has_uv = !components.is_empty();
    let mut choices: Vec<usize> = Vec::new();
    let mut first_choice = None;
    for i in 0..corner.count() {
        let pair = [corner.bound_face(i), corner.faces[i]];
        let best = choose_representative(&pair, points, polygons, components);
        first_choice.get_or_insert(best);
        if choices.contains(&best) {
            continue;
        }
        if has_uv && projected_boundary_area(corner, best, points, polygons) < BEVEL_EPSILON_BIG {
            continue;
        }
        choices.push(best);
    }
    if choices.is_empty() {
        first_choice.unwrap_or(corner.faces[0])
    } else {
        choose_representative(&choices, points, polygons, components)
    }
}

/// Rebuilds a VERTICES bevel of `selected` vertices with `seg > 2`: each selected vertex becomes a
/// general vertex mesh whose boundary runs along its edges, and each face corner at a selected
/// vertex is replaced by the mesh ring through that face. `edge_point(vertex, previous, next,
/// neighbor)` is the boundary point on the edge to `neighbor` for the face corner `previous,
/// vertex, next`. `None` when a selected vertex is not a closed fan with three or more edges.
pub(super) fn bevel_vertices(
    mesh: &Mesh,
    seg: usize,
    super_r: f64,
    profile: f64,
    points: &[DVec3],
    polygons: &[Polygon],
    selected: &HashSet<usize>,
    edge_point: &mut dyn FnMut(usize, usize, usize, usize) -> DVec3,
) -> Option<Emission> {
    if super_r == PRO_SQUARE_IN_R || super_r == PRO_SQUARE_R {
        return None;
    }
    let spacing = ProfileSpacing::new(seg, super_r);
    let fullness = profile_fullness(seg, profile, super_r);

    // Boundary points, one per (vertex, neighbour), taken from the first face that meets them.
    let mut order: Vec<usize> = Vec::new();
    let mut bounds_of: HashMap<(usize, usize), DVec3> = HashMap::new();
    for (face, _) in polygons {
        let count = face.len();
        for (position, &vertex) in face.iter().enumerate() {
            if !selected.contains(&vertex) {
                continue;
            }
            if !order.contains(&vertex) {
                order.push(vertex);
            }
            let previous = face[(position + count - 1) % count];
            let next = face[(position + 1) % count];
            for neighbor in [previous, next] {
                bounds_of
                    .entry((vertex, neighbor))
                    .or_insert_with(|| edge_point(vertex, previous, next, neighbor));
            }
        }
    }

    let uvs = source_uvs(mesh);
    let components = if seg % 2 == 1 && !uvs.is_empty() {
        face_components(points, polygons, &uvs)
    } else {
        Vec::new()
    };
    let mut emission = Emission::new(points);
    let mut corners: Vec<Corner> = Vec::with_capacity(order.len());
    let mut corner_of: HashMap<usize, usize> = HashMap::new();
    for &vertex in &order {
        let (neighbors, faces) = corner_topology(vertex, polygons)?;
        if neighbors.len() < 3 {
            return None;
        }
        let bounds: Vec<DVec3> = neighbors
            .iter()
            .map(|neighbor| bounds_of.get(&(vertex, *neighbor)).copied())
            .collect::<Option<_>>()?;
        corner_of.insert(vertex, corners.len());
        corners.push(Corner {
            vertex,
            neighbors,
            faces,
            bounds,
        });
    }
    // Blender's `bv->any_seam` for odd segment counts: the UVs around the vertex are not contiguous.
    let any_seam: Vec<bool> = corners
        .iter()
        .map(|corner| seg % 2 == 1 && !uv_contiguous_around(&uvs, polygons, corner.vertex))
        .collect();
    let mut ids: Vec<CornerIds> = Vec::with_capacity(corners.len());
    for corner in &corners {
        let vm = general_vmesh(&spacing, fullness, &corner.bounds, points[corner.vertex]);
        ids.push(CornerIds::new(corner, vm, &mut emission));
    }

    // Faces: each selected corner is replaced by its ring through that face, in polygon order.
    for (face_index, (face, _)) in polygons.iter().enumerate() {
        let count = face.len();
        let mut vertices = Vec::with_capacity(face.len());
        for (position, &vertex) in face.iter().enumerate() {
            let Some(&index) = corner_of.get(&vertex) else {
                vertices.push(vertex);
                continue;
            };
            let corner = &corners[index];
            let ring = (0..corner.count()).find(|i| corner.faces[*i] == face_index)?;
            let previous = face[(position + count - 1) % count];
            let arc: Vec<usize> = if corner.neighbors[ring] == previous {
                (0..=seg).collect()
            } else if corner.neighbors[(ring + 1) % corner.count()] == previous {
                (0..=seg).rev().collect()
            } else {
                return None;
            };
            for k in arc {
                vertices.push(ids[index].id(&mut emission, ring, 0, k));
            }
        }
        let len = vertices.len();
        emission.push_polygon(vertices, [None; 4], vec![Some(face_index); len], face_index);
    }

    // Quads of each ring, and the center polygon.
    let ns2 = seg / 2;
    let odd = seg % 2;
    for (index, corner) in corners.iter().enumerate() {
        let n = corner.count();
        let vertex_point = points[corner.vertex];
        for i in 0..n {
            let face_before = corner.bound_face(i);
            let face_after = corner.faces[i];
            let along_here = Some((vertex_point, points[corner.neighbors[i]]));
            let along_next = Some((vertex_point, points[corner.neighbors[(i + 1) % n]]));
            let seam_here = edge_is_seam(&uvs, polygons, corner.vertex, corner.neighbors[i]);
            for j in 0..ns2 {
                for k in 0..(ns2 + odd) {
                    let vertices = vec![
                        ids[index].id(&mut emission, i, j, k),
                        ids[index].id(&mut emission, i, j, k + 1),
                        ids[index].id(&mut emission, i, j + 1, k + 1),
                        ids[index].id(&mut emission, i, j + 1, k),
                    ];
                    let mut sources = [face_after; 4];
                    let mut snapped = [None; 4];
                    if j < k {
                        if j + 1 == ns2 && k == ns2 {
                            snapped[2] = along_next;
                            snapped[3] = along_here;
                        }
                    } else if j == k {
                        snapped[0] = along_here;
                        snapped[2] = along_here;
                        if !seam_here {
                            sources[3] = face_before;
                        }
                    }
                    emission.push_polygon(
                        vertices,
                        snapped,
                        sources.map(Some).to_vec(),
                        face_after,
                    );
                }
            }
        }
        if odd == 1 {
            let vertices: Vec<usize> = (0..n)
                .map(|i| ids[index].id(&mut emission, i, ns2, ns2))
                .collect();
            if any_seam[index] {
                // Blender's `build_center_ngon` with `bv->any_seam`: one representative face for the
                // whole center. Each center point snaps to the closer of that face's two edges at the
                // vertex, unless its boundary vertex lies inside the face.
                let frep = center_representative(corner, points, polygons, &components);
                let face = &polygons[frep].0;
                let normal = face_normal(points, face);
                let (near, far) = incident_neighbors(face, corner.vertex);
                let vertex_point = points[corner.vertex];
                let snapped: Vec<Option<(DVec3, DVec3)>> = (0..n)
                    .map(|i| {
                        if point_strictly_inside(points, face, normal, corner.bounds[i]) {
                            return None;
                        }
                        let center_point = emission.point(vertices[i]);
                        let near_distance =
                            dist_sq_to_segment_f32(center_point, vertex_point, points[near]);
                        let far_distance =
                            dist_sq_to_segment_f32(center_point, vertex_point, points[far]);
                        let other = if near_distance < far_distance {
                            near
                        } else {
                            far
                        };
                        Some((vertex_point, points[other]))
                    })
                    .collect();
                emission.push_polygon(vertices, snapped, vec![Some(frep); n], frep);
            } else {
                let sources = (0..n).map(|i| Some(corner.bound_face(i))).collect();
                let face = corner.bound_face(0);
                emission.push_polygon(vertices, [None; 4], sources, face);
            }
        }
    }
    Some(emission)
}

/// Blender's `STD_UV_CONNECT_LIMIT`: UV coordinates closer than this are one UV vertex.
const STD_UV_CONNECT_LIMIT: f64 = 1.0e-4;

/// Input UV coordinates per layer, face, and corner (`None` where a face has no data).
type SourceUvs = Vec<Vec<Option<Vec<[f64; 2]>>>>;

fn source_uvs(mesh: &Mesh) -> SourceUvs {
    uv_layers::read_layers(mesh)
        .into_iter()
        .map(|layer| layer.faces)
        .collect()
}

/// Whether two input loops have the same UV vertex in every layer (Blender's
/// `contig_ldata_across_loops` for math layers, with `STD_UV_CONNECT_LIMIT`).
fn uv_connected(uvs: &SourceUvs, first: (usize, usize), second: (usize, usize)) -> bool {
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
                (Some(a), Some(b)) => {
                    // `compare_v2v2` with `compare_ff`, which tests `fabs(a - b) <= limit` per component.
                    (a[0] - b[0]).abs() <= STD_UV_CONNECT_LIMIT
                        && (a[1] - b[1]).abs() <= STD_UV_CONNECT_LIMIT
                }
                _ => false,
            }
        })
}

/// Blender's `determine_uv_vert_connectivity`: the input loops around `vertex`, grouped into UV
/// vertices.
fn source_vertex_buckets(
    uvs: &SourceUvs,
    source_polygons: &[Polygon],
    vertex: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut buckets: Vec<Vec<(usize, usize)>> = Vec::new();
    for (face_index, (face, _)) in source_polygons.iter().enumerate() {
        for (corner, value) in face.iter().enumerate() {
            if *value != vertex {
                continue;
            }
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

/// Blender's `update_uv_vert_map` and `bevel_merge_uvs`. Loops of one new vertex that share a
/// representative face, or whose representative faces are UV-connected at the input vertex, end up
/// in one bucket, and each bucket takes the average of its loops' UVs. Blender averages the
/// interpolated values; averaging the source weights gives the same result because interpolation is
/// linear. `loops` holds `(output vertex, representative face)` in output loop order.
fn merge_connected_loops(
    corners: &mut [Vec<(usize, f64)>],
    loops: &[(usize, Option<usize>)],
    origins: &[usize],
    source_polygons: &[Polygon],
    uvs: &SourceUvs,
) {
    if uvs.is_empty() {
        return;
    }
    let mut at_vertex: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut buckets: HashMap<usize, Vec<Vec<usize>>> = HashMap::new();
    let mut source_buckets: HashMap<usize, Vec<Vec<(usize, usize)>>> = HashMap::new();
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
                        let vertex_source = source_buckets
                            .entry(origin)
                            .or_insert_with(|| source_vertex_buckets(uvs, source_polygons, origin));
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

/// Builds the output mesh from an emission: drops unused points, gives each polygon corner its
/// UVs by Blender's mean-value interpolation in the corner's source face at its query point, and
/// averages UVs across the loops that Blender merges.
pub(super) fn finish(
    mesh: &Mesh,
    source_polygons: &[Polygon],
    emission: Emission,
    loose: Vec<[usize; 2]>,
) -> Result<Mesh> {
    let Emission {
        points: emitted_points,
        origins: emitted_origins,
        polygons: emitted_polygons,
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
    for (index, point) in emitted_points.iter().enumerate() {
        if used[index] {
            remap[index] = points.len();
            points.push(*point);
            origins.push(vec![(emitted_origins[index], 1.0)]);
        }
    }
    let mut polygons: Vec<Polygon> = Vec::with_capacity(emitted_polygons.len());
    let mut faces = Vec::with_capacity(emitted_polygons.len());
    let mut queries = Vec::new();
    let mut corner_sources = Vec::new();
    let mut loops = Vec::new();
    for polygon in &emitted_polygons {
        polygons.push((
            polygon
                .vertices
                .iter()
                .map(|vertex| remap[*vertex])
                .collect(),
            source_polygons[polygon.face].1,
        ));
        faces.push(Some(polygon.face));
        let attached = polygon.sources.first().copied().flatten();
        for (corner, (query, source)) in polygon.queries.iter().zip(&polygon.sources).enumerate() {
            corner_sources.push((queries.len(), *source));
            queries.push(*query);
            loops.push((polygon.vertices[corner], attached));
        }
    }
    let mut corners = mean_value_corners(
        mesh,
        &queries,
        corner_sources
            .iter()
            .map(|(query, source)| (std::slice::from_ref(query), *source)),
    );
    merge_connected_loops(
        &mut corners,
        &loops,
        &emitted_origins,
        source_polygons,
        &source_uvs(mesh),
    );
    let origin = Origin {
        vertices: origins,
        faces,
        corners: Some(corners),
        raw_normals: false,
    };
    let loose = loose
        .into_iter()
        .map(|[first, second]| [remap[first], remap[second]])
        .collect();
    build_mesh(points, polygons, loose, &origin, mesh)
}
