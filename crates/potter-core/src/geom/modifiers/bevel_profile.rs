// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/curveprofile.cc (BKE_curveprofile_init, BKE_curveprofile_update,
//   create_samples, point_calculate_handle), source/blender/blenkernel/intern/curve.cc
//   (BKE_curve_forward_diff_bezier), source/blender/blenlib/intern/math_vector.cc (angle_v2v2),
//   SPDX-FileCopyrightText: Blender Authors, GPL-2.0-or-later.
//
// The sample ordering uses the macOS libc `qsort`, `_isort` and `heapsort` (the implementation
// Blender calls on macOS, reproduced from libSystem). Blender's comparator is not a total order,
// so the exact algorithm decides which samples receive the remainder counts.

//! Blender 5.2.2 `CurveProfile` for the bevel custom profile.
//!
//! `BKE_curveprofile_init(profile, totsegments)` only runs `BKE_curveprofile_update(profile, 0)`:
//! the handles of every point are recomputed (AUTO and VECTOR sides, FREE and ALIGN sides are left
//! alone), the path table is built, and then the `totsegments + 1` samples are written. The
//! clipping, duplicate removal and point insertion of `update` are edits, not part of the
//! initialization, so the stored locations are sampled as they are.
//!
//! Sample flags: `use_sample_straight_edges` (bit 0x4) gives every segment the same number of
//! samples, `use_sample_even_lengths` (bit 0x8) resamples the path table by arc length, and
//! `use_clip` (bit 0x1) has no effect on the samples.
//!
//! [`CurveProfile::sample`] returns Blender's `segments[0..=totsegments]` as `(x, y)` pairs, the
//! order of `CurveProfilePoint.location`. `set_profile_spacing` in `bmesh_bevel.cc` reads
//! `xvals[i] = segments[i].y` and `yvals[i] = segments[i].x`, so consumers swap the pair. With
//! `use_sample_even_lengths`, Blender never writes the last sample, so it is `(0, 0)`.
//!
//! FREE and ALIGN handle sides preserve their stored `handle_1` / `handle_2` coordinates during
//! `point_calculate_handle`. Those coordinates are captured from the Blender profile struct and
//! are part of each custom point input.

use serde_json::{Value, json};

use crate::error::{ErrorCode, PotError, Result};

/// `0x4023EDFA` in Blender's `point_calculate_handle`: the AUTO handle length factor.
const AUTO_HANDLE_FACTOR: f32 = f32::from_bits(0x4023_EDFA);
/// `0x40490FDB` in `angle_v2v2`.
const PI_F32: f32 = f32::from_bits(0x4049_0FDB);
/// `0x3EAAAAAB` in `point_calculate_handle`: one third, the VECTOR handle fraction.
const ONE_THIRD: f32 = f32::from_bits(0x3EAA_AAAB);
/// `0x0554AD2E` in `angle_v2v2`: squared length below which a vector is treated as zero.
const ANGLE_EPSILON: f32 = f32::from_bits(0x0554_AD2E);
/// The path table holds `min(16 * points, 527) - 15` entries (`BKE_curveprofile_update`).
const TABLE_CAP: usize = 527;
const TABLE_OFFSET: usize = 15;
/// The constant `BKE_curveprofile_update` writes over the last table entry.
const TABLE_SENTINEL: [f32; 2] = [0.0, 1.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HandleType {
    Free,
    Auto,
    Vector,
    Align,
}

#[derive(Clone, Debug, PartialEq)]
struct ProfilePoint {
    location: [f32; 2],
    handle_types: [HandleType; 2],
    handles: Pair,
}

/// Blender's `CurveProfile` as the bevel custom profile uses it.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CurveProfile {
    points: Vec<ProfilePoint>,
    straight_edges: bool,
    even_lengths: bool,
}

impl CurveProfile {
    /// The `custom_profile` parameter. An absent parameter is Blender's LINE preset: points (1, 0)
    /// and (0, 1) with AUTO handles, `use_clip` true (Blender's default; it does not change the
    /// samples), and both sampling flags off.
    pub(super) fn from_param(value: Option<&Value>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::line());
        };
        let object = value
            .as_object()
            .ok_or_else(|| invalid_profile(EXPECTED_PROFILE))?;
        if let Some(key) = object
            .keys()
            .find(|key| !PROFILE_KEYS.contains(&key.as_str()))
        {
            return Err(invalid_profile(&format!(
                "{EXPECTED_PROFILE}; unknown key `{key}`"
            )));
        }
        let points = object
            .get("points")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_profile(EXPECTED_PROFILE))?;
        if points.len() < 2 {
            return Err(invalid_profile("at least two points"));
        }
        let points = points.iter().map(parse_point).collect::<Result<Vec<_>>>()?;
        // `use_clip` only matters for edits; it is validated so that a malformed profile fails.
        flag(object, "use_clip")?;
        let straight_edges = flag(object, "use_sample_straight_edges")?;
        let even_lengths = flag(object, "use_sample_even_lengths")?;
        Ok(Self {
            points,
            straight_edges,
            even_lengths,
        })
    }

    fn line() -> Self {
        Self {
            points: vec![
                ProfilePoint {
                    location: [1.0, 0.0],
                    handle_types: [HandleType::Auto, HandleType::Auto],
                    handles: [[1.0, 0.0], [1.0, 0.0]],
                },
                ProfilePoint {
                    location: [0.0, 1.0],
                    handle_types: [HandleType::Auto, HandleType::Auto],
                    handles: [[0.0, 1.0], [0.0, 1.0]],
                },
            ],
            straight_edges: false,
            even_lengths: false,
        }
    }

    /// Blender's `segments` after `BKE_curveprofile_init(profile, totsegments)`: `totsegments + 1`
    /// samples as `(x, y)`, empty for `totsegments == 0`.
    pub(super) fn sample(&self, totsegments: usize) -> Vec<(f64, f64)> {
        if totsegments == 0 {
            return Vec::new();
        }
        let handles = handle_state(&self.points);
        let samples = if self.even_lengths {
            arc_length_samples(&self.points, &handles, totsegments)
        } else {
            create_samples(&self.points, &handles, totsegments, self.straight_edges)
        };
        samples
            .into_iter()
            .map(|[x, y]| (f64::from(x), f64::from(y)))
            .collect()
    }
}

const EXPECTED_PROFILE: &str = "an object with points, use_clip, use_sample_straight_edges and \
    use_sample_even_lengths";
const PROFILE_KEYS: [&str; 4] = [
    "points",
    "use_clip",
    "use_sample_straight_edges",
    "use_sample_even_lengths",
];
const POINT_KEYS: [&str; 5] = [
    "location",
    "handle_1",
    "handle_2",
    "handle_type_1",
    "handle_type_2",
];

fn invalid_profile(expected: &str) -> PotError {
    PotError::with_details(
        ErrorCode::InvalidArgument,
        format!("invalid `custom_profile` parameter for modifier `bevel`; expected {expected}"),
        json!({
            "parameter": "custom_profile",
            "pointer": "/params/custom_profile"
        }),
    )
}

fn flag(object: &serde_json::Map<String, Value>, key: &str) -> Result<bool> {
    object
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid_profile(&format!("{EXPECTED_PROFILE}; `{key}` as a boolean")))
}

fn parse_point(value: &Value) -> Result<ProfilePoint> {
    const EXPECTED_POINT: &str = "points with location, handle_1 and handle_2 coordinates [x, y], \
        and handle types AUTO, VECTOR, FREE or ALIGNED";
    let object = value
        .as_object()
        .ok_or_else(|| invalid_profile(EXPECTED_POINT))?;
    if let Some(key) = object
        .keys()
        .find(|key| !POINT_KEYS.contains(&key.as_str()))
    {
        return Err(invalid_profile(&format!(
            "{EXPECTED_POINT}; unknown key `{key}`"
        )));
    }
    Ok(ProfilePoint {
        location: parse_coordinates(object.get("location"), EXPECTED_POINT)?,
        handles: [
            parse_coordinates(object.get("handle_1"), EXPECTED_POINT)?,
            parse_coordinates(object.get("handle_2"), EXPECTED_POINT)?,
        ],
        handle_types: [
            handle_type(object, "handle_type_1", EXPECTED_POINT)?,
            handle_type(object, "handle_type_2", EXPECTED_POINT)?,
        ],
    })
}

fn parse_coordinates(value: Option<&Value>, expected: &str) -> Result<[f32; 2]> {
    let coordinates = value
        .and_then(Value::as_array)
        .filter(|coordinates| coordinates.len() == 2)
        .ok_or_else(|| invalid_profile(expected))?;
    let mut parsed = [0.0_f32; 2];
    for (slot, value) in parsed.iter_mut().zip(coordinates) {
        let number = value
            .as_f64()
            .map(|number| number as f32)
            .filter(|number| number.is_finite())
            .ok_or_else(|| invalid_profile(expected))?;
        *slot = number;
    }
    Ok(parsed)
}

fn handle_type(
    object: &serde_json::Map<String, Value>,
    key: &str,
    expected: &str,
) -> Result<HandleType> {
    match object.get(key).and_then(Value::as_str) {
        Some("AUTO") => Ok(HandleType::Auto),
        Some("VECTOR") => Ok(HandleType::Vector),
        Some("FREE") => Ok(HandleType::Free),
        Some("ALIGNED") => Ok(HandleType::Align),
        _ => Err(invalid_profile(expected)),
    }
}

type Pair = [[f32; 2]; 2];

/// Blender's `point_calculate_handle` for every point. AUTO and VECTOR sides are recomputed;
/// FREE and ALIGN sides retain the stored handle coordinates from the profile input.
fn handle_state(points: &[ProfilePoint]) -> Vec<Pair> {
    let locations: Vec<[f32; 2]> = points.iter().map(|point| point.location).collect();
    let mut state: Vec<Pair> = points.iter().map(|point| point.handles).collect();
    for (index, pair) in state.iter_mut().enumerate() {
        apply_point_handles(points[index].handle_types, &locations, index, pair);
    }
    state
}

/// Blender's `point_calculate_handle`. The operations are in the order of the arm64 build so that
/// the f32 results are the same bits.
fn apply_point_handles(
    types: [HandleType; 2],
    locations: &[[f32; 2]],
    index: usize,
    pair: &mut Pair,
) {
    let [type_1, type_2] = types;
    if type_1 == HandleType::Free && type_2 == HandleType::Free {
        return;
    }
    let p = locations[index];
    let previous = index.checked_sub(1).map(|i| locations[i]);
    let next = locations.get(index + 1).copied();
    // `a` points from the previous point to `p`; a missing previous point mirrors the next one.
    let (ax, ay) = if let Some(q) = previous {
        (p[0] - q[0], p[1] - q[1])
    } else {
        let n = next.unwrap_or(p);
        let vx = p[0] + p[0] - n[0];
        let vy = p[1] + p[1] - n[1];
        (p[0] - vx, p[1] - vy)
    };
    // `b` points from `p` to the next point; a missing next point mirrors the previous one.
    let (bx, by) = if let Some(n) = next {
        (n[0] - p[0], n[1] - p[1])
    } else {
        let q = previous.unwrap_or(p);
        let vx = p[0] + p[0] - q[0];
        let vy = p[1] + p[1] - q[1];
        (vx - p[0], vy - p[1])
    };
    let la = (ax * ax + ay * ay).sqrt();
    let lb = (bx * bx + by * by).sqrt();
    let la_safe = if la == 0.0 { 1.0 } else { la };
    let lb_safe = if lb == 0.0 { 1.0 } else { lb };
    let tx = ax / la_safe + bx / lb_safe;
    let ty = ay / la_safe + by / lb_safe;
    let scale = (tx * tx + ty * ty).sqrt() * AUTO_HANDLE_FACTOR;
    if (type_1 == HandleType::Auto || type_2 == HandleType::Auto) && scale != 0.0 {
        if type_1 == HandleType::Auto {
            let q = -la_safe / scale;
            pair[0] = [p[0] + tx * q, p[1] + ty * q];
        }
        if type_2 == HandleType::Auto {
            let q = lb_safe / scale;
            pair[1] = [p[0] + tx * q, p[1] + ty * q];
        }
    }
    if type_1 == HandleType::Vector {
        pair[0] = [p[0] - ax * ONE_THIRD, p[1] - ay * ONE_THIRD];
    }
    if type_2 == HandleType::Vector {
        pair[1] = [p[0] + bx * ONE_THIRD, p[1] + by * ONE_THIRD];
    }
}

/// Blender's `angle_v2v2`: the angle between two 2D vectors in [0, pi].
fn angle_v2v2(a: [f32; 2], b: [f32; 2]) -> f32 {
    fn unit(v: [f32; 2]) -> [f32; 2] {
        let length_squared = v[0] * v[0] + v[1] * v[1];
        if length_squared > ANGLE_EPSILON {
            let inverse = 1.0 / length_squared.sqrt();
            [v[0] * inverse, v[1] * inverse]
        } else {
            [0.0, 0.0]
        }
    }
    let v1 = unit(a);
    let v2 = unit(b);
    let dot = v1[0] * v2[0] + v1[1] * v2[1];
    let opposed = dot < 0.0;
    let (vx, vy) = if opposed {
        (v1[0] + v2[0], v1[1] + v2[1])
    } else {
        (v1[0] - v2[0], v1[1] - v2[1])
    };
    let half = ((vx * vx + vy * vy).sqrt() * 0.5).clamp(-1.0, 1.0);
    let angle = half.asin() + half.asin();
    if opposed { PI_F32 - angle } else { angle }
}

/// Blender's `BKE_curve_forward_diff_bezier`: `it + 1` samples of a cubic with the given control
/// values, at t = k / it, by forward differencing.
fn forward_diff(q0: f32, q1: f32, q2: f32, q3: f32, it: usize) -> Vec<f32> {
    let steps = it as f32;
    let steps_squared = steps * steps;
    let steps_cubed = steps_squared * steps;
    let cubic = ((q3 - q0) + (q1 - q2) * 3.0) / steps_cubed;
    let third = cubic * 6.0;
    let quadratic = ((q0 - (q1 + q1)) + q2) * 3.0 / steps_squared;
    let mut second = quadratic + quadratic;
    let mut first = (q1 - q0) * 3.0 / steps + quadratic + cubic;
    let mut value = q0;
    let mut out = Vec::with_capacity(it + 1);
    for _ in 0..=it {
        second += third;
        out.push(value);
        value += first;
        first += second;
    }
    out
}

/// Blender's `create_samples`: `n + 1` samples along the curve. Segments get sample counts from
/// their handle types (or evenly with `straight_edges`), and the segments with the largest
/// curvature angles get the remainder.
fn create_samples(
    points: &[ProfilePoint],
    handles: &[Pair],
    n: usize,
    straight_edges: bool,
) -> Vec<[f32; 2]> {
    let locations: Vec<[f32; 2]> = points.iter().map(|point| point.location).collect();
    let npts = points.len();
    let nseg = npts - 1;
    let mut angles: Vec<(usize, f32)> = (0..nseg)
        .map(|k| {
            let a = [
                handles[k][1][0] - locations[k][0],
                handles[k][1][1] - locations[k][1],
            ];
            let b = [
                locations[k + 1][0] - handles[k + 1][0][0],
                locations[k + 1][1] - handles[k + 1][0][1],
            ];
            (k, angle_v2v2(a, b))
        })
        .collect();
    libc_qsort(&mut angles, &|x: &(usize, f32), y: &(usize, f32)| {
        i32::from(x.1 <= y.1)
    });

    let mut counts = vec![0usize; nseg];
    if nseg <= n {
        let remainder = if straight_edges {
            let base = n / nseg;
            if base >= 1 {
                counts.fill(base);
            }
            n % nseg
        } else {
            let fixed = |k: usize| {
                points[k].handle_types[1] == HandleType::Vector
                    && points[k + 1].handle_types[0] == HandleType::Vector
            };
            let free = (0..nseg).filter(|&k| !fixed(k)).count();
            let free_count = if free == 0 { nseg } else { free };
            let extra = n - nseg + free_count;
            let base = extra / free_count;
            if base >= 1 {
                for (k, count) in counts.iter_mut().enumerate() {
                    *count = if !fixed(k) || free_count == nseg {
                        base
                    } else {
                        1
                    };
                }
                extra - base * free_count
            } else {
                extra
            }
        };
        for &(k, _) in angles.iter().take(remainder) {
            counts[k] += 1;
        }
    } else {
        for &(k, _) in angles.iter().take(n) {
            counts[k] += 1;
        }
    }

    let mut out = vec![[0.0_f32; 2]; n + 1];
    let mut written = 0;
    for (k, &count) in counts.iter().enumerate() {
        if count >= 1 {
            let p0 = locations[k];
            let p1 = handles[k][1];
            let p2 = handles[k + 1][0];
            let p3 = locations[k + 1];
            let xs = forward_diff(p0[0], p1[0], p2[0], p3[0], count);
            let ys = forward_diff(p0[1], p1[1], p2[1], p3[1], count);
            for (offset, (x, y)) in xs.into_iter().zip(ys).enumerate() {
                out[written + offset] = [x, y];
            }
        }
        written += count;
    }
    out
}

/// The path table of `BKE_curveprofile_update`: `create_samples` with the non-even split and a
/// sentinel written over the last entry.
fn path_table(points: &[ProfilePoint], handles: &[Pair]) -> Vec<[f32; 2]> {
    let table_len = points.len().wrapping_mul(16).min(TABLE_CAP) - TABLE_OFFSET;
    let mut table = create_samples(points, handles, table_len - 1, false);
    table[table_len - 1] = TABLE_SENTINEL;
    table
}

/// The `use_sample_even_lengths` path of `BKE_curveprofile_update`: `n` samples spaced by equal
/// arc length along the path table. The last sample is never written and stays `(0, 0)`.
fn arc_length_samples(points: &[ProfilePoint], handles: &[Pair], n: usize) -> Vec<[f32; 2]> {
    let table = path_table(points, handles);
    let chords: Vec<f32> = table
        .windows(2)
        .map(|pair| {
            let dx = pair[0][0] - pair[1][0];
            let dy = pair[0][1] - pair[1][1];
            (dx * dx + dy * dy).sqrt()
        })
        .collect();
    let mut total = 0.0_f32;
    for chord in &chords {
        total += chord;
    }
    let step = total / n as f32;

    let mut out = vec![[0.0_f32; 2]; n + 1];
    out[0] = table[0];
    let mut chord_index = 0;
    let mut chord_remaining = chords[0];
    let mut travelled = 0.0_f32;
    for sample in out.iter_mut().take(n).skip(1) {
        let (advance, fraction) = if chord_remaining >= step {
            let span = travelled + chord_remaining;
            travelled += step;
            (step, travelled / span)
        } else {
            let mut needed = step - chord_remaining;
            chord_index += 1;
            chord_remaining = chords[chord_index];
            while chord_remaining < needed {
                needed -= chord_remaining;
                chord_index += 1;
                chord_remaining = chords[chord_index];
            }
            travelled = 0.0 + needed;
            (needed, travelled / (0.0 + chord_remaining))
        };
        let one_minus = 1.0 - fraction;
        let a = table[chord_index];
        let b = table[chord_index + 1];
        *sample = [
            b[0] * fraction + a[0] * one_minus,
            b[1] * fraction + a[1] * one_minus,
        ];
        chord_remaining -= advance;
    }
    out
}

/// macOS libc `qsort` (`_qsort`, `_isort`, `heapsort`), with the depth limit `2 * floor(log2 n)`.
fn libc_qsort<T: Copy>(items: &mut [T], cmp: &dyn Fn(&T, &T) -> i32) {
    if items.len() < 2 {
        return;
    }
    let depth = 2 * (usize::BITS - 1 - items.len().leading_zeros()) as i32;
    qsort_slice(items, depth, cmp);
}

fn qsort_slice<T: Copy>(mut v: &mut [T], mut depth: i32, cmp: &dyn Fn(&T, &T) -> i32) {
    loop {
        let n = v.len();
        if n <= 7 {
            insertion_sort(v, 0, cmp);
            return;
        }
        if depth < 1 {
            heapsort(v, cmp);
            return;
        }
        let middle = n / 2;
        let pivot = if n >= 41 {
            let d = n / 8;
            let low = median_of_three(v, 0, d, 2 * d, cmp);
            let mid = median_of_three(v, middle - d, middle, middle + d, cmp);
            let high = median_of_three(v, n - 1 - 2 * d, n - 1 - d, n - 1, cmp);
            median_of_three(v, low, mid, high, cmp)
        } else {
            median_of_three(v, 0, middle, n - 1, cmp)
        };
        v.swap(0, pivot);

        let (mut pa, mut pb) = (1_isize, 1_isize);
        let (mut pc, mut pd) = (n as isize - 1, n as isize - 1);
        let mut swapped = false;
        loop {
            while pb <= pc {
                let order = cmp(&v[pb as usize], &v[0]);
                if order > 0 {
                    break;
                }
                if order == 0 {
                    swapped = true;
                    v.swap(pa as usize, pb as usize);
                    pa += 1;
                }
                pb += 1;
            }
            while pb <= pc {
                let order = cmp(&v[pc as usize], &v[0]);
                if order < 0 {
                    break;
                }
                if order == 0 {
                    swapped = true;
                    v.swap(pc as usize, pd as usize);
                    pd -= 1;
                }
                pc -= 1;
            }
            if pb > pc {
                break;
            }
            v.swap(pb as usize, pc as usize);
            swapped = true;
            pb += 1;
            pc -= 1;
        }

        let len = n as isize;
        let swap_left = (pa).min(pb - pa);
        vecswap(v, 0, pb - swap_left, swap_left);
        let swap_right = (pd - pc).min(len - pd - 1);
        vecswap(v, pb as usize, len - swap_right, swap_right);

        let left = pb - pa;
        let right = pd - pc;
        if !swapped && insertion_sort(v, (n >> 2) + 1, cmp) {
            return;
        }
        if left <= right {
            if left > 1 {
                qsort_slice(&mut v[..left as usize], depth - 1, cmp);
            }
            if right <= 1 {
                return;
            }
            let rest = std::mem::take(&mut v);
            v = &mut rest[(len - right) as usize..];
        } else {
            if right > 1 {
                qsort_slice(&mut v[(len - right) as usize..], depth - 1, cmp);
            }
            if left <= 1 {
                return;
            }
            let rest = std::mem::take(&mut v);
            v = &mut rest[..left as usize];
        }
        depth -= 1;
        if depth < 1 {
            heapsort(v, cmp);
            return;
        }
    }
}

fn median_of_three<T: Copy>(
    v: &[T],
    a: usize,
    b: usize,
    c: usize,
    cmp: &dyn Fn(&T, &T) -> i32,
) -> usize {
    if cmp(&v[a], &v[b]) < 0 {
        if cmp(&v[b], &v[c]) < 0 {
            return b;
        }
        return if cmp(&v[a], &v[c]) < 0 { c } else { a };
    }
    if cmp(&v[b], &v[c]) > 0 {
        return b;
    }
    if cmp(&v[a], &v[c]) < 0 { a } else { c }
}

fn vecswap<T: Copy>(v: &mut [T], i: usize, j: isize, count: isize) {
    for k in 0..count.max(0) as usize {
        v.swap(i + k, j as usize + k);
    }
}

/// `_isort`: insertion sort that stops once `budget` swaps are exceeded (0 means no limit).
/// Returns whether the slice is sorted.
fn insertion_sort<T: Copy>(v: &mut [T], budget: usize, cmp: &dyn Fn(&T, &T) -> i32) -> bool {
    let mut moves = 0;
    for i in 1..v.len() {
        let mut cur = i;
        while cur > 0 && cmp(&v[cur - 1], &v[cur]) > 0 {
            v.swap(cur - 1, cur);
            if budget != 0 {
                if moves >= budget {
                    return false;
                }
                moves += 1;
            }
            cur -= 1;
        }
    }
    true
}

/// libSystem `heapsort`: a bottom-up heap sort over 1-based indices.
fn heapsort<T: Copy>(v: &mut [T], cmp: &dyn Fn(&T, &T) -> i32) {
    let n = v.len();
    if n < 2 {
        return;
    }
    let mut k = n / 2;
    while k > 0 {
        let mut i = k;
        let mut l = 2 * i;
        if l <= n {
            loop {
                let j = if l < n {
                    l + usize::from(cmp(&v[l - 1], &v[l]) < 0)
                } else {
                    l
                };
                if cmp(&v[j - 1], &v[i - 1]) < 1 {
                    break;
                }
                v.swap(j - 1, i - 1);
                i = j;
                l = 2 * j;
                if l > n {
                    break;
                }
            }
        }
        k -= 1;
    }

    let mut size = n;
    loop {
        let tmp = v[size - 1];
        v[size - 1] = v[0];
        let h = size - 1;
        let mut hole = 1;
        if size >= 3 {
            let mut l = 2;
            loop {
                let j = if l < h {
                    l + usize::from(cmp(&v[l - 1], &v[l]) < 0)
                } else {
                    l
                };
                v[hole - 1] = v[j - 1];
                hole = j;
                l = 2 * j;
                if l > h {
                    break;
                }
            }
        }
        while hole != 1 {
            let parent = hole >> 1;
            if cmp(&tmp, &v[parent - 1]) < 0 {
                break;
            }
            v[hole - 1] = v[parent - 1];
            hole = parent;
        }
        v[hole - 1] = tmp;
        size = h;
        if h <= 1 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "oracle fixtures and profile fixtures")]

    use serde_json::Value;

    use super::*;

    /// `tests/fixtures/curve_profile_oracle.py` run with Blender 5.2.2.
    const ORACLE: &str = include_str!("../../../../../tests/fixtures/curve_profile_oracle.json");

    fn oracle() -> Value {
        serde_json::from_str(ORACLE).expect("curve profile oracle JSON")
    }

    fn param_from_case(case: &Value) -> Value {
        let points: Vec<Value> = case["points"]
            .as_array()
            .expect("points")
            .iter()
            .map(|point| {
                json!({
                    "location": point["location"],
                    "handle_1": point["handle_1"],
                    "handle_2": point["handle_2"],
                    "handle_type_1": point["handle_type_1"],
                    "handle_type_2": point["handle_type_2"],
                })
            })
            .collect();
        json!({
            "points": points,
            "use_clip": case["use_clip"],
            "use_sample_straight_edges": case["use_sample_straight_edges"],
            "use_sample_even_lengths": case["use_sample_even_lengths"],
        })
    }

    fn assert_matches(name: &str, count: usize, actual: &[(f64, f64)], expected: &Value) {
        let expected = expected.as_array().expect("samples");
        assert_eq!(
            actual.len(),
            count + 1,
            "{name}: sample count for totsegments {count}"
        );
        assert_eq!(actual.len(), expected.len(), "{name}: fixture length");
        for (index, (&(x, y), value)) in actual.iter().zip(expected).enumerate() {
            // Blender stores f32 samples; serde_json's default float parsing may be one double ulp
            // off, which the f32 cast absorbs. The bits compared are Blender's f32 bits.
            let ex = value[0].as_f64().expect("x") as f32;
            let ey = value[1].as_f64().expect("y") as f32;
            let (x, y) = (x as f32, y as f32);
            assert!(
                x.to_bits() == ex.to_bits() && y.to_bits() == ey.to_bits(),
                "{name}: totsegments {count} sample {index}: got ({x:?}, {y:?}), Blender ({ex:?}, {ey:?})"
            );
        }
    }

    #[test]
    fn samples_match_blender_bit_for_bit() {
        let oracle = oracle();
        let cases = oracle["cases"].as_array().expect("cases");
        assert!(cases.len() >= 6, "oracle must cover several curves");
        for case in cases {
            let name = case["name"].as_str().expect("name");
            let profile = CurveProfile::from_param(Some(&param_from_case(case)))
                .unwrap_or_else(|error| panic!("{name}: {}", error.message));
            for (count, expected) in case["samples"].as_object().expect("samples") {
                let count: usize = count.parse().expect("segment count");
                assert_matches(name, count, &profile.sample(count), expected);
            }
        }
    }

    #[test]
    fn absent_parameter_is_the_line_preset() {
        let oracle = oracle();
        let case = oracle["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| case["name"] == "line_auto_default")
            .expect("line_auto_default case");
        let profile = CurveProfile::from_param(None).expect("LINE preset");
        for (count, expected) in case["samples"].as_object().expect("samples") {
            let count: usize = count.parse().expect("segment count");
            assert_matches("line_auto_default", count, &profile.sample(count), expected);
        }
    }

    #[test]
    fn zero_segments_yield_no_samples() {
        assert_eq!(
            CurveProfile::from_param(None).expect("LINE").sample(0),
            Vec::<(f64, f64)>::new()
        );
    }

    #[test]
    fn use_clip_does_not_clamp_initialization_samples() {
        let point = |location| {
            json!({
                "location": location,
                "handle_1": location,
                "handle_2": location,
                "handle_type_1": "FREE",
                "handle_type_2": "FREE"
            })
        };
        let value = json!({
            "points": [point([-0.25, 0.0]), point([1.25, 1.0])],
            "use_clip": true,
            "use_sample_straight_edges": false,
            "use_sample_even_lengths": false
        });
        let profile = CurveProfile::from_param(Some(&value)).expect("valid profile");

        assert_eq!(profile.sample(1)[0], (-0.25, 0.0));
    }

    #[test]
    fn malformed_profiles_are_rejected() {
        let valid_point = json!({
            "location": [0.0, 0.0],
            "handle_1": [0.0, 0.0],
            "handle_2": [0.0, 0.0],
            "handle_type_1": "AUTO",
            "handle_type_2": "AUTO"
        });
        let valid_end = json!({
            "location": [1.0, 1.0],
            "handle_1": [1.0, 1.0],
            "handle_2": [1.0, 1.0],
            "handle_type_1": "AUTO",
            "handle_type_2": "AUTO"
        });
        let flags = json!({
            "use_clip": true,
            "use_sample_straight_edges": false,
            "use_sample_even_lengths": false,
        });
        let with = |points: Value| {
            let mut object = flags.as_object().expect("flags").clone();
            object.insert("points".to_owned(), points);
            Value::Object(object)
        };
        let cases = [
            json!(3),
            with(json!([valid_point.clone()])),
            with(json!([valid_point.clone(), {
                "location": [1.0, 1.0],
                "handle_1": [1.0, 1.0],
                "handle_type_1": "AUTO",
                "handle_type_2": "AUTO"
            }])),
            with(json!([valid_point.clone(), {
                "location": [1.0],
                "handle_1": [1.0, 1.0],
                "handle_2": [1.0, 1.0],
                "handle_type_1": "AUTO",
                "handle_type_2": "AUTO"
            }])),
            with(json!([valid_point.clone(), {
                "location": [1.0, 1.0],
                "handle_1": [1.0, 1.0],
                "handle_2": [1.0, 1.0],
                "handle_type_1": "ALIGN",
                "handle_type_2": "AUTO"
            }])),
            with(json!([valid_point.clone(), {
                "location": [1.0, 1.0],
                "handle_1": [1.0, 1.0],
                "handle_2": [1.0, 1.0],
                "handle_type_1": "AUTO",
                "handle_type_2": "AUTO",
                "select": true
            }])),
            with(json!([valid_point.clone(), {
                "location": [1e39, 1.0],
                "handle_1": [1.0, 1.0],
                "handle_2": [1.0, 1.0],
                "handle_type_1": "AUTO",
                "handle_type_2": "AUTO"
            }])),
            json!({"points": [valid_point.clone(), valid_end.clone()], "use_clip": true}),
            json!({
                "points": [valid_point.clone(), valid_end.clone()],
                "use_clip": true,
                "use_sample_straight_edges": false,
                "use_sample_even_lengths": false,
                "use_sample_superellipse": false,
            }),
        ];
        for case in &cases {
            let error = CurveProfile::from_param(Some(case)).expect_err("malformed profile");
            assert_eq!(error.code, ErrorCode::InvalidArgument);
        }
        let valid = with(json!([valid_point, {
            "location": [1.0, 1.0],
            "handle_1": [1.0, 1.0],
            "handle_2": [1.0, 1.0],
            "handle_type_1": "ALIGNED",
            "handle_type_2": "FREE"
        }]));
        CurveProfile::from_param(Some(&valid)).expect("ALIGNED and FREE are valid handle types");
    }
}
