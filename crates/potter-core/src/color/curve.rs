//! Blender 5.2.2's view `CurveMapping`: the colour-management curve that `view_settings.curve_mapping`
//! applies before the view transform when `use_curve_mapping` is set.
//!
//! This is a float32 port of the table path that Blender's CPU display processor runs:
//! `calchandle_curvemap`, `curvemap_make_table`, `BKE_curvemapping_premultiply`,
//! `BKE_curvemap_evaluateF`, and `evaluate_film_like` (`colortools.cc`), and
//! `BKE_curve_correct_bezpart` and `BKE_curve_forward_diff_bezier` (`curve.cc`). Every arithmetic
//! step is a separate `f32` operation in the order of the Blender source, with no fused
//! multiply-add. Single-curve evaluation matched Blender 5.2.2 bit-for-bit on the probed cases in
//! the tests below.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{PotError, Result};

/// `CM_RESOL`: samples per Bézier segment.
const CM_RESOL: usize = 32;
/// `CM_TABLE`: intervals in the evaluation table, which has `CM_TABLE + 1` samples.
const CM_TABLE: usize = 256;
/// `CM_TABLEDIV`: `1.0f / 256.0f`, exact.
const CM_TABLEDIV: f32 = 1.0 / 256.0;
/// `FLT_EPSILON`.
const FLT_EPSILON: f32 = f32::EPSILON;

/// Blender's `eCurveMappingTone`. `Standard` applies each channel's curve to that channel after the
/// combined curve. `FilmLike` evaluates the combined curve only and keeps each pixel's hue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurveTone {
    #[default]
    Standard,
    FilmLike,
}

/// Blender's control-point handle type: `Auto` (`HD_AUTO`), `AutoClamped` (`HD_AUTO_ANIM`), or
/// `Vector` (`HD_VECT`). `Vector` takes precedence over `AutoClamped`, as in Blender.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandleType {
    #[default]
    Auto,
    AutoClamped,
    Vector,
}

/// One control point of a view curve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurvePoint {
    /// `(x, y)` in the curve's input and output space.
    pub location: [f32; 2],
    pub handle: HandleType,
}

/// One curve (`CurveMap`): control points sorted by strictly increasing `x`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurveMap {
    pub points: Vec<CurvePoint>,
}

/// Blender's view `CurveMapping`. `curves` are `[R, G, B, combined]`, matching `cm[0..4]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurveMapping {
    pub tone: CurveTone,
    /// Blender's `CUMA_EXTEND_EXTRAPOLATE`: extend beyond the end points along the end tangents
    /// (`true`) or horizontally (`false`).
    pub extend_extrapolate: bool,
    /// Blender's `CUMA_DO_CLIP`. Blender applies it to output `y` only in `evaluateF`
    /// (`BKE_curvemapping_evaluateF`) and when editing points. The display path never clips, so
    /// [`PreparedCurve::evaluate_rgb`] ignores it. It is kept so the setting round-trips.
    pub use_clip: bool,
    /// Clip rectangle `[min_x, min_y, max_x, max_y]`. Only `min_x` and `max_x` shape the table
    /// domain (Blender's `clipr`): the table spans the union of this range and the points' `x`
    /// range. `min_y` and `max_y` are kept for round-trips; the display path never reads them.
    pub clip: [f64; 4],
    /// Black and white levels per RGB channel. The display path maps `(x - black) * bwmul`, where
    /// `bwmul = 1 / max(white - black, 1e-5)`.
    pub black: [f32; 3],
    pub white: [f32; 3],
    pub curves: [CurveMap; 4],
}

impl Default for CurveMapping {
    /// Blender's `BKE_curvemapping_add(4, 0, 0, 1, 1)` state: standard tone, extrapolated, clip on
    /// over `[0, 1] x [0, 1]`, black 0, white 1, and four identity curves through `(0, 0)` and
    /// `(1, 1)` with AUTO handles.
    fn default() -> Self {
        Self {
            tone: CurveTone::Standard,
            extend_extrapolate: true,
            use_clip: true,
            clip: [0.0, 0.0, 1.0, 1.0],
            black: [0.0; 3],
            white: [1.0; 3],
            curves: std::array::from_fn(|_| identity_curve()),
        }
    }
}

fn identity_curve() -> CurveMap {
    CurveMap {
        points: vec![
            CurvePoint {
                location: [0.0, 0.0],
                handle: HandleType::Auto,
            },
            CurvePoint {
                location: [1.0, 1.0],
                handle: HandleType::Auto,
            },
        ],
    }
}

impl CurveMapping {
    /// Checks the input that table construction needs.
    ///
    /// - Each curve has at least two points.
    /// - Every point's `x` and `y` are finite, and the `x` values are strictly increasing.
    /// - The clip rectangle is finite, with `min < max` on both axes.
    /// - Black and white levels are finite.
    ///
    /// Points outside the clip rectangle are allowed. Blender widens the table domain to include
    /// them.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorCode::InvalidArgument`](crate::error::ErrorCode::InvalidArgument) with a
    /// message naming the first violated rule.
    pub fn validate(&self) -> Result<()> {
        self.check().map_err(PotError::invalid_argument)
    }

    fn check(&self) -> std::result::Result<(), String> {
        for (index, curve) in self.curves.iter().enumerate() {
            if curve.points.len() < 2 {
                return Err(format!("curve {index} needs at least two points"));
            }
            let mut previous: Option<f32> = None;
            for (point_index, point) in curve.points.iter().enumerate() {
                let [x, y] = point.location;
                if !x.is_finite() || !y.is_finite() {
                    return Err(format!("curve {index} point {point_index} must be finite"));
                }
                if previous.is_some_and(|last| x <= last) {
                    return Err(format!(
                        "curve {index} point {point_index} x must be strictly increasing"
                    ));
                }
                previous = Some(x);
            }
        }
        if !self.clip.iter().all(|value| finite_as_f32(*value)) {
            return Err("clip values must be finite".to_owned());
        }
        if self.clip[0] >= self.clip[2] || self.clip[1] >= self.clip[3] {
            return Err("clip minimum must be below its maximum on both axes".to_owned());
        }
        if !self
            .black
            .iter()
            .chain(&self.white)
            .all(|value| value.is_finite())
        {
            return Err("black and white levels must be finite".to_owned());
        }
        Ok(())
    }

    /// Parses the JSON form used by the `color.update` operation (`set.curve`) and by the Blender
    /// bridge. Every field is required and unknown fields are rejected:
    ///
    /// ```json
    /// {
    ///   "tone": "standard" | "film_like",
    ///   "extend": "extrapolated" | "horizontal",
    ///   "use_clip": true,
    ///   "clip": [min_x, min_y, max_x, max_y],
    ///   "black": [r, g, b],
    ///   "white": [r, g, b],
    ///   "curves": [ {"points": [{"location": [x, y], "handle": "auto" | "auto_clamped" | "vector"}]} ] x4
    /// }
    /// ```
    ///
    /// The result is checked with [`CurveMapping::validate`].
    ///
    /// # Errors
    ///
    /// Returns a message that names the failing field, for the caller to report.
    pub fn from_json(value: &Value) -> std::result::Result<Self, String> {
        let root = object(
            value,
            "curve",
            &[
                "tone", "extend", "use_clip", "clip", "black", "white", "curves",
            ],
        )?;
        let tone = match required(root, "curve", "tone")?.as_str() {
            Some("standard") => CurveTone::Standard,
            Some("film_like") => CurveTone::FilmLike,
            _ => return Err("curve/tone must be `standard` or `film_like`".to_owned()),
        };
        let extend_extrapolate = match required(root, "curve", "extend")?.as_str() {
            Some("extrapolated") => true,
            Some("horizontal") => false,
            _ => return Err("curve/extend must be `extrapolated` or `horizontal`".to_owned()),
        };
        let use_clip = required(root, "curve", "use_clip")?
            .as_bool()
            .ok_or_else(|| "curve/use_clip must be a boolean".to_owned())?;
        let clip = numbers::<4>(required(root, "curve", "clip")?, "curve/clip")?;
        let black = numbers::<3>(required(root, "curve", "black")?, "curve/black")?
            .map(|value| value as f32);
        let white = numbers::<3>(required(root, "curve", "white")?, "curve/white")?
            .map(|value| value as f32);
        let curve_values = required(root, "curve", "curves")?
            .as_array()
            .filter(|curves| curves.len() == 4)
            .ok_or_else(|| "curve/curves must be an array of four curves".to_owned())?;
        let curves = curve_values
            .iter()
            .enumerate()
            .map(|(index, curve)| parse_curve(curve, &format!("curve/curves/{index}")))
            .collect::<std::result::Result<Vec<_>, String>>()?;
        let curves: [CurveMap; 4] = curves
            .try_into()
            .map_err(|_| "curve/curves must be an array of four curves".to_owned())?;
        let mapping = Self {
            tone,
            extend_extrapolate,
            use_clip,
            clip,
            black,
            white,
            curves,
        };
        mapping.check()?;
        Ok(mapping)
    }

    /// Builds the evaluation tables that Blender's display processor builds
    /// (`ColormanageProcessor`): each R/G/B table is premultiplied by the combined curve, and the
    /// film-like tone reads the combined table.
    ///
    /// Call [`CurveMapping::validate`] first.
    ///
    /// # Panics
    ///
    /// Panics if a curve has fewer than two points, which `validate` rejects.
    #[must_use]
    pub fn prepare(&self) -> PreparedCurve {
        let clip_min_x = self.clip[0] as f32;
        let clip_max_x = self.clip[2] as f32;
        let extrapolate = self.extend_extrapolate;
        let raw: [Table; 4] = std::array::from_fn(|index| {
            build_table(
                clip_min_x,
                clip_max_x,
                &self.curves[index].points,
                extrapolate,
            )
        });
        let premul: [Table; 3] =
            std::array::from_fn(|index| premultiply(&raw[index], &raw[3], extrapolate));
        let bwmul: [f32; 3] = std::array::from_fn(|channel| {
            let span = max_ff(self.white[channel] - self.black[channel], 1e-5);
            1.0 / span
        });
        PreparedCurve {
            tone: self.tone,
            extrapolate,
            black: self.black,
            bwmul,
            raw,
            premul,
        }
    }
}

/// Evaluation tables for one [`CurveMapping`], immutable once built.
#[derive(Clone, Debug)]
pub struct PreparedCurve {
    tone: CurveTone,
    extrapolate: bool,
    black: [f32; 3],
    bwmul: [f32; 3],
    /// `cm[0..4]` as built from the curves: used by [`PreparedCurve::evaluate`] and by the film-like
    /// tone.
    raw: [Table; 4],
    /// `cm[c]` premultiplied by `cm[3]` (Blender's `CUMA_PREMULLED` tables): used by the standard
    /// tone.
    premul: [Table; 3],
}

impl PreparedCurve {
    /// Applies the display path to one scene-linear RGB pixel: `(x - black) * bwmul`, then the
    /// standard tone (each channel through its premultiplied table) or the film-like tone
    /// (hue-preserving, from the combined table). The result is not clipped, and alpha is not
    /// touched.
    ///
    /// A NaN input channel gives NaN output. Blender leaves `int(NaN)` undefined here. This port
    /// returns NaN rather than choosing a table index.
    #[must_use]
    pub fn evaluate_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let balanced: [f32; 3] = std::array::from_fn(|channel| {
            (rgb[channel] - self.black[channel]) * self.bwmul[channel]
        });
        match self.tone {
            CurveTone::Standard => std::array::from_fn(|channel| {
                evaluate_table(&self.premul[channel], self.extrapolate, balanced[channel])
            }),
            CurveTone::FilmLike => film_like(&self.raw[3], self.extrapolate, balanced),
        }
    }

    /// Evaluates curve `curve` (`0` = R, `1` = G, `2` = B, `3` = combined) at `x`, as Blender's
    /// `CurveMapping.evaluate` does. There is no premultiply, black/white, or `y` clip. Used for
    /// oracle tests against Blender.
    ///
    /// A NaN input gives NaN output, as in [`PreparedCurve::evaluate_rgb`].
    ///
    /// # Panics
    ///
    /// Panics if `curve` is 4 or more.
    #[must_use]
    pub fn evaluate(&self, curve: usize, x: f32) -> f32 {
        evaluate_table(&self.raw[curve], self.extrapolate, x)
    }
}

/// One evaluation table (`CurveMap.table` with its domain and extension slopes).
#[derive(Clone, Debug)]
struct Table {
    /// `mintable`: the `x` of the first sample.
    mintable: f32,
    /// `range`: the stored reciprocal `1 / (CM_TABLEDIV * table_range)`.
    range: f32,
    /// `CM_TABLE + 1` samples as `(x, y)`.
    samples: Vec<[f32; 2]>,
    /// Unit vector from the first point's left handle toward the curve interior.
    ext_in: [f32; 2],
    /// Unit vector from the last point's right handle toward the curve interior.
    ext_out: [f32; 2],
}

fn finite_as_f32(value: f64) -> bool {
    value.is_finite() && (value as f32).is_finite()
}

fn object<'a>(
    value: &'a Value,
    path: &str,
    allowed: &[&str],
) -> std::result::Result<&'a Map<String, Value>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{path} must be an object"))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("{path} has unknown field `{key}`"));
    }
    Ok(object)
}

fn required<'a>(
    object: &'a Map<String, Value>,
    path: &str,
    key: &str,
) -> std::result::Result<&'a Value, String> {
    object
        .get(key)
        .ok_or_else(|| format!("{path}/{key} is required"))
}

fn number(value: &Value, path: &str) -> std::result::Result<f64, String> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("{path} must contain finite numbers"))
}

fn numbers<const N: usize>(value: &Value, path: &str) -> std::result::Result<[f64; N], String> {
    let items = value
        .as_array()
        .filter(|items| items.len() == N)
        .ok_or_else(|| format!("{path} must be an array of {N} numbers"))?;
    let mut out = [0.0; N];
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = number(item, path)?;
    }
    Ok(out)
}

fn parse_curve(value: &Value, path: &str) -> std::result::Result<CurveMap, String> {
    let object = object(value, path, &["points"])?;
    let points = required(object, path, "points")?
        .as_array()
        .ok_or_else(|| format!("{path}/points must be an array"))?;
    let points = points
        .iter()
        .enumerate()
        .map(|(index, point)| parse_point(point, &format!("{path}/points/{index}")))
        .collect::<std::result::Result<Vec<_>, String>>()?;
    Ok(CurveMap { points })
}

fn parse_point(value: &Value, path: &str) -> std::result::Result<CurvePoint, String> {
    let object = object(value, path, &["location", "handle"])?;
    let [x, y] = numbers::<2>(
        required(object, path, "location")?,
        &format!("{path}/location"),
    )?;
    let handle = match required(object, path, "handle")?.as_str() {
        Some("auto") => HandleType::Auto,
        Some("auto_clamped") => HandleType::AutoClamped,
        Some("vector") => HandleType::Vector,
        _ => {
            return Err(format!(
                "{path}/handle must be `auto`, `auto_clamped`, or `vector`"
            ));
        }
    };
    Ok(CurvePoint {
        location: [x as f32, y as f32],
        handle,
    })
}

/// Blender's `min_ff`: `a < b ? a : b`.
fn min_ff(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// Blender's `max_ff`: `a > b ? a : b`.
fn max_ff(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `std::min(a, b)`: `b < a ? b : a`.
fn std_min(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

/// `std::max(a, b)`: `a < b ? b : a`.
fn std_max(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn scale(v: [f32; 2], factor: f32) -> [f32; 2] {
    [v[0] * factor, v[1] * factor]
}

/// Blender's `madd_v2_v2v2fl`: `a + b * factor`, with the multiply and add rounded separately.
fn madd(a: [f32; 2], b: [f32; 2], factor: f32) -> [f32; 2] {
    [a[0] + b[0] * factor, a[1] + b[1] * factor]
}

/// `len_v2`. The `z` term of `len_v3` is zero throughout the curve code, so `len_v3` gives the same
/// bits.
fn len2(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// `2 * p2 - p`, the mirrored virtual neighbour at a curve end.
fn mirror(p2: [f32; 2], p: [f32; 2]) -> [f32; 2] {
    [2.0 * p2[0] - p[0], 2.0 * p2[1] - p[1]]
}

fn is_auto(handle: HandleType) -> bool {
    matches!(handle, HandleType::Auto | HandleType::AutoClamped)
}

/// `calchandle_curvemap`: the left and right handle of one point, from its neighbours' positions.
/// `prev` and `next` are `None` at the curve ends, where the missing neighbour is mirrored.
fn calc_handle(
    p2: [f32; 2],
    prev: Option<[f32; 2]>,
    next: Option<[f32; 2]>,
    handle: HandleType,
) -> ([f32; 2], [f32; 2]) {
    let (p1, p3) = match (prev, next) {
        (Some(p1), Some(p3)) => (p1, p3),
        (None, Some(p3)) => (mirror(p2, p3), p3),
        (Some(p1), None) => (p1, mirror(p2, p1)),
        // A curve has at least two points, so one neighbour exists.
        (None, None) => return ([0.0; 2], [0.0; 2]),
    };
    let dvec_a = sub(p2, p1);
    let dvec_b = sub(p3, p2);
    let mut len_a = len2(dvec_a);
    let mut len_b = len2(dvec_b);
    if len_a == 0.0 {
        len_a = 1.0;
    }
    if len_b == 0.0 {
        len_b = 1.0;
    }

    let mut left = [0.0; 2];
    let mut right = [0.0; 2];
    if is_auto(handle) {
        let tvec = [
            dvec_b[0] / len_b + dvec_a[0] / len_a,
            dvec_b[1] / len_b + dvec_a[1] / len_a,
        ];
        let len = len2(tvec) * 2.5614;
        if len != 0.0 {
            len_a /= len;
            left = madd(p2, tvec, -len_a);
            if let (HandleType::AutoClamped, Some(prev), Some(next)) = (handle, prev, next) {
                // Keep the handle horizontal at a local extremum, and otherwise keep it on its
                // side of the neighbour's height.
                let ydiff1 = prev[1] - p2[1];
                let ydiff2 = next[1] - p2[1];
                if (ydiff1 <= 0.0 && ydiff2 <= 0.0) || (ydiff1 >= 0.0 && ydiff2 >= 0.0) {
                    left[1] = p2[1];
                } else if ydiff1 <= 0.0 {
                    left[1] = std_max(prev[1], left[1]);
                } else {
                    left[1] = std_min(prev[1], left[1]);
                }
            }

            len_b /= len;
            right = madd(p2, tvec, len_b);
            if let (HandleType::AutoClamped, Some(prev), Some(next)) = (handle, prev, next) {
                let ydiff1 = prev[1] - p2[1];
                let ydiff2 = next[1] - p2[1];
                if (ydiff1 <= 0.0 && ydiff2 <= 0.0) || (ydiff1 >= 0.0 && ydiff2 >= 0.0) {
                    right[1] = p2[1];
                } else if ydiff1 <= 0.0 {
                    right[1] = std_min(next[1], right[1]);
                } else {
                    right[1] = std_max(next[1], right[1]);
                }
            }
        }
    }
    if handle == HandleType::Vector {
        left = madd(p2, dvec_a, -1.0 / 3.0);
        right = madd(p2, dvec_b, 1.0 / 3.0);
    }
    (left, right)
}

/// `BKE_curve_correct_bezpart`: pulls the two inner handles of one segment in when they overlap in
/// `x`. `v2` and `v3` are the start's right handle and the end's left handle.
fn correct_bezpart(v1: [f32; 2], v2: &mut [f32; 2], v3: &mut [f32; 2], v4: [f32; 2]) {
    let h1 = sub(v1, *v2);
    let h2 = sub(v4, *v3);
    let span = v4[0] - v1[0];
    let handle1 = h1[0].abs();
    let handle2 = h2[0].abs();
    if handle1 + handle2 <= 0.0 {
        return;
    }
    if handle1 + handle2 > span {
        let fac = span / (handle1 + handle2);
        *v2 = [v1[0] - fac * h1[0], v1[1] - fac * h1[1]];
        *v3 = [v4[0] - fac * h2[0], v4[1] - fac * h2[1]];
    }
}

/// `BKE_curve_forward_diff_bezier` with `it = CM_RESOL - 1`: `CM_RESOL` values of one Bézier
/// coordinate at uniform `t`, starting at `q0`, by forward differencing.
fn forward_diff(q0: f32, q1: f32, q2: f32, q3: f32) -> [f32; CM_RESOL] {
    let mut f = 31.0_f32;
    let rt0 = q0;
    let rt1 = 3.0 * (q1 - q0) / f;
    f *= f;
    let rt2 = 3.0 * (q0 - 2.0 * q1 + q2) / f;
    f *= 31.0;
    let rt3 = (q3 - q0 + 3.0 * (q1 - q2)) / f;

    let mut value = rt0;
    let mut first = rt1 + rt2 + rt3;
    let mut second = 2.0 * rt2 + 6.0 * rt3;
    let third = 6.0 * rt3;
    let mut out = [0.0; CM_RESOL];
    for slot in &mut out {
        *slot = value;
        value += first;
        first += second;
        second += third;
    }
    out
}

/// `curvemap_calc_extend`: the value beyond `first` (left of it) or `last` (right of it).
fn calc_extend(
    extrapolate: bool,
    ext_in: [f32; 2],
    ext_out: [f32; 2],
    x: f32,
    first: [f32; 2],
    last: [f32; 2],
) -> f32 {
    if x <= first[0] {
        if !extrapolate {
            return first[1];
        }
        if ext_in[0] == 0.0 {
            return first[1] + ext_in[1] * 10000.0;
        }
        return first[1] + ext_in[1] * (x - first[0]) / ext_in[0];
    }
    if x >= last[0] {
        if !extrapolate {
            return last[1];
        }
        if ext_out[0] == 0.0 {
            return last[1] - ext_out[1] * 10000.0;
        }
        return last[1] + ext_out[1] * (x - last[0]) / ext_out[0];
    }
    // Blender returns 0 here. The callers pass only `x` outside the samples, so this is reached
    // only in the sliver where the table index rounds past `CM_TABLE` before `x` reaches the last
    // sample.
    0.0
}

/// `curvemap_make_table`: samples each segment, then resamples onto `CM_TABLE + 1` uniform `x`
/// steps across the table domain.
///
/// # Panics
///
/// Panics if `points` has fewer than two entries.
fn build_table(
    clip_min_x: f32,
    clip_max_x: f32,
    points: &[CurvePoint],
    extrapolate: bool,
) -> Table {
    assert!(points.len() >= 2, "a view curve needs two control points");
    let count = points.len();
    let positions: Vec<[f32; 2]> = points.iter().map(|point| point.location).collect();

    let mut mintable = clip_min_x;
    let mut maxtable = clip_max_x;
    for position in &positions {
        mintable = min_ff(mintable, position[0]);
        maxtable = max_ff(maxtable, position[0]);
    }
    let table_range = maxtable - mintable;

    let mut left = vec![[0.0_f32; 2]; count];
    let mut right = vec![[0.0_f32; 2]; count];
    for index in 0..count {
        let prev = index.checked_sub(1).map(|prev| positions[prev]);
        let next = positions.get(index + 1).copied();
        (left[index], right[index]) =
            calc_handle(positions[index], prev, next, points[index].handle);
    }

    // The first and last handles of a plain AUTO end point aim at the nearest handle of the
    // neighbour, keeping their original length. Curves of two points are not adjusted.
    if count > 2 {
        if points[0].handle == HandleType::Auto {
            let p0 = positions[0];
            let hlen = len2(sub(p0, right[0]));
            let mut vec = [std_max(left[1][0], p0[0]), left[1][1]];
            vec = sub(vec, p0);
            let nlen = len2(vec);
            if nlen > FLT_EPSILON {
                vec = scale(vec, hlen / nlen);
                right[0] = add(vec, p0);
                left[0] = sub(p0, vec);
            }
        }
        let last = count - 1;
        if points[last].handle == HandleType::Auto {
            let p = positions[last];
            let hlen = len2(sub(p, left[last]));
            let mut vec = [std_min(right[last - 1][0], p[0]), right[last - 1][1]];
            vec = sub(vec, p);
            let nlen = len2(vec);
            if nlen > FLT_EPSILON {
                vec = scale(vec, hlen / nlen);
                left[last] = add(vec, p);
                right[last] = sub(p, vec);
            }
        }
    }

    // Sample every segment: `CM_RESOL` points each, at uniform `t`. Sampling pulls in overlapping
    // inner handles, so the extension slopes below use the corrected handles.
    let mut samples = vec![[0.0_f32; 2]; (count - 1) * CM_RESOL];
    for segment in 0..count - 1 {
        let v1 = positions[segment];
        let mut v2 = right[segment];
        let mut v3 = left[segment + 1];
        let v4 = positions[segment + 1];
        correct_bezpart(v1, &mut v2, &mut v3, v4);
        right[segment] = v2;
        left[segment + 1] = v3;
        let xs = forward_diff(v1[0], v2[0], v3[0], v4[0]);
        let ys = forward_diff(v1[1], v2[1], v3[1], v4[1]);
        for (offset, (x, y)) in xs.into_iter().zip(ys).enumerate() {
            samples[segment * CM_RESOL + offset] = [x, y];
        }
    }

    let mut ext_in = sub(left[0], positions[0]);
    let ext_in_range = len2(ext_in);
    ext_in = [ext_in[0] / ext_in_range, ext_in[1] / ext_in_range];
    let mut ext_out = sub(positions[count - 1], right[count - 1]);
    let ext_out_range = len2(ext_out);
    ext_out = [ext_out[0] / ext_out_range, ext_out[1] / ext_out_range];

    let range = CM_TABLEDIV * table_range;
    let reciprocal = 1.0 / range;
    let first = 0;
    let last = samples.len() - 1;
    let mut point = 0;
    let mut table = Vec::with_capacity(CM_TABLE + 1);
    for step in 0..=CM_TABLE {
        let cur_x = mintable + range * (step as f32);
        while cur_x >= samples[point][0] && point != last {
            point += 1;
        }
        let y = if point == first || (point == last && cur_x >= samples[point][0]) {
            if (cur_x - samples[point][0]).abs() <= 1e-6 {
                samples[point][1]
            } else {
                calc_extend(
                    extrapolate,
                    ext_in,
                    ext_out,
                    cur_x,
                    samples[first],
                    samples[last],
                )
            }
        } else {
            let mut fac1 = samples[point][0] - samples[point - 1][0];
            let fac2 = samples[point][0] - cur_x;
            fac1 = if fac1 > FLT_EPSILON { fac2 / fac1 } else { 0.0 };
            fac1 * samples[point - 1][1] + (1.0 - fac1) * samples[point][1]
        };
        table.push([cur_x, y]);
    }

    Table {
        mintable,
        range: reciprocal,
        samples: table,
        ext_in,
        ext_out,
    }
}

/// `BKE_curvemapping_premultiply` for one channel: `cm[channel]` followed by the combined curve,
/// with the extension slopes multiplied component-wise.
fn premultiply(channel: &Table, combined: &Table, extrapolate: bool) -> Table {
    let samples = channel
        .samples
        .iter()
        .map(|&[x, y]| [x, evaluate_table(combined, extrapolate, y)])
        .collect();
    Table {
        mintable: channel.mintable,
        range: channel.range,
        samples,
        ext_in: [
            channel.ext_in[0] * combined.ext_in[0],
            channel.ext_in[1] * combined.ext_in[1],
        ],
        ext_out: [
            channel.ext_out[0] * combined.ext_out[0],
            channel.ext_out[1] * combined.ext_out[1],
        ],
    }
}

/// `BKE_curvemap_evaluateF`: linear interpolation in the table, or the extension beyond it.
fn evaluate_table(table: &Table, extrapolate: bool, value: f32) -> f32 {
    if value.is_nan() {
        return f32::NAN;
    }
    let fi = (value - table.mintable) * table.range;
    let last = CM_TABLE as f32;
    if !(0.0..=last).contains(&fi) {
        return calc_extend(
            extrapolate,
            table.ext_in,
            table.ext_out,
            value,
            table.samples[0],
            table.samples[CM_TABLE],
        );
    }
    let index = fi as usize;
    if index >= CM_TABLE {
        return table.samples[CM_TABLE][1];
    }
    let fraction = fi - index as f32;
    (1.0 - fraction) * table.samples[index][1] + fraction * table.samples[index + 1][1]
}

/// `evaluate_film_like`: the combined curve on the minimum and maximum channels, and the median
/// placed to keep the pixel's hue.
#[expect(
    clippy::float_cmp,
    reason = "Blender assigns channels by exact equality with the extremes"
)]
fn film_like(table: &Table, extrapolate: bool, input: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = input;
    let minimum = std_min(std_min(x, y), z);
    let maximum = std_max(std_max(x, y), z);
    let median = std_max(std_min(x, y), std_min(z, std_max(x, y)));

    let new_min = evaluate_table(table, extrapolate, minimum);
    let new_max = evaluate_table(table, extrapolate, maximum);
    let scaling_ratio = (new_max - new_min) / (maximum - minimum);
    let new_median = new_min + (median - minimum) * scaling_ratio;

    let channel = |value: f32| {
        let base = if value == minimum {
            new_min
        } else {
            new_median
        };
        if value == maximum { new_max } else { base }
    };
    [channel(x), channel(y), channel(z)]
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Blender 5.2.2 `CurveMapping.evaluate` values (`/tmp/looks-probe/curve_probe.json`, probe
    /// cases named in each test). Every expected value is a Blender output, not an emulation.
    fn assert_close(actual: f32, expected: f32) {
        let tolerance = 1e-6_f32.max(1e-6 * expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "evaluate gave {actual}, Blender gave {expected}"
        );
    }

    fn curve_of(points: &[(f32, f32, HandleType)]) -> CurveMap {
        CurveMap {
            points: points
                .iter()
                .map(|&(x, y, handle)| CurvePoint {
                    location: [x, y],
                    handle,
                })
                .collect(),
        }
    }

    fn combined(points: &[(f32, f32, HandleType)]) -> CurveMapping {
        let mut mapping = CurveMapping::default();
        mapping.curves[3] = curve_of(points);
        mapping
    }

    const AUTO: HandleType = HandleType::Auto;
    const CLAMPED: HandleType = HandleType::AutoClamped;
    const VECTOR: HandleType = HandleType::Vector;

    /// Probe case `b_auto5`: five AUTO points on the combined curve.
    fn auto5() -> Vec<(f32, f32, HandleType)> {
        vec![
            (0.0, 0.0, AUTO),
            (0.15, 0.4, AUTO),
            (0.4, 0.3, AUTO),
            (0.7, 0.85, AUTO),
            (1.0, 1.0, AUTO),
        ]
    }

    #[test]
    fn default_mapping_is_identity_with_extrapolation() {
        // Probe case `a_default`.
        let prepared = CurveMapping::default().prepare();
        assert_close(prepared.evaluate(3, 2.0), 2.0);
        assert_close(prepared.evaluate(3, -1.0), -1.0);
        assert_close(prepared.evaluate(3, 0.5), 0.5);
        assert_close(prepared.evaluate(3, 3.0), 3.0);
        assert_close(prepared.evaluate(0, 2.0), 2.0);
    }

    #[test]
    fn auto_combined_curve_matches_blender() {
        // Probe case `b_auto5`. The end handles are vertical here (the left handle of point 1 lies
        // left of point 0), so the extension goes through the 10000 branch of Blender's
        // `curvemap_calc_extend`.
        let prepared = combined(&auto5()).prepare();
        assert_close(prepared.evaluate(3, -0.5), -10000.0);
        assert_close(prepared.evaluate(3, -0.25), -10000.0);
        assert_close(prepared.evaluate(3, 0.0), 0.0);
        assert_close(prepared.evaluate(3, 0.25), 0.373_742_25);
        assert_close(prepared.evaluate(3, 0.5), 0.393_788_04);
        assert_close(prepared.evaluate(3, 0.75), 0.893_045_66);
        assert_close(prepared.evaluate(3, 1.5), 1.143_543_6);
        assert_close(prepared.evaluate(3, 2.0), 1.287_087_1);
    }

    #[test]
    fn auto_clamped_curve_matches_blender() {
        // Probe case `c_clamped5`: the same points with AUTO_CLAMPED handles.
        let points: Vec<_> = auto5()
            .into_iter()
            .map(|(x, y, _)| (x, y, CLAMPED))
            .collect();
        let prepared = combined(&points).prepare();
        assert_close(prepared.evaluate(3, -0.5), -1.333_333_3);
        assert_close(prepared.evaluate(3, 0.25), 0.365_444_8);
        assert_close(prepared.evaluate(3, 0.5), 0.368_053_14);
        assert_close(prepared.evaluate(3, 1.5), 1.249_999_8);
    }

    #[test]
    fn vector_handles_and_clamped_end_match_blender() {
        // Probe case `d_vector_mix`: a VECTOR point at 0.25, a CLAMPED point at 0.75.
        let prepared = combined(&[
            (0.0, 0.0, AUTO),
            (0.25, 0.5, VECTOR),
            (0.5, 0.3, AUTO),
            (0.75, 0.8, CLAMPED),
            (1.0, 1.0, AUTO),
        ])
        .prepare();
        assert_close(prepared.evaluate(3, 0.25), 0.5);
        assert_close(prepared.evaluate(3, 0.5), 0.3);
        assert_close(prepared.evaluate(3, 0.75), 0.8);
        assert_close(prepared.evaluate(3, 1.5), 1.299_841_2);
    }

    #[test]
    fn horizontal_extension_matches_blender() {
        // Probe case `e_horizontal`: the same points as `b_auto5` with `extend = Horizontal`.
        let mut mapping = combined(&auto5());
        mapping.extend_extrapolate = false;
        let prepared = mapping.prepare();
        assert_close(prepared.evaluate(3, -0.5), 0.0);
        assert_close(prepared.evaluate(3, 0.5), 0.393_788_04);
        assert_close(prepared.evaluate(3, 1.5), 1.000_000_1);
    }

    #[test]
    fn clip_rectangle_widens_the_table_domain() {
        // Probe case `f_clip_wide`: clip x in [-0.5, 2.0] moves the table grid, so values change.
        let mut mapping = combined(&auto5());
        mapping.clip = [-0.5, -0.25, 2.0, 1.5];
        let prepared = mapping.prepare();
        assert_close(prepared.evaluate(3, -0.5), -10000.0);
        assert_close(prepared.evaluate(3, -0.25), -10000.0);
        assert_close(prepared.evaluate(3, 0.25), 0.373_683_8);
        assert_close(prepared.evaluate(3, 1.5), 1.143_543_6);
    }

    #[test]
    fn narrow_clip_does_not_shrink_the_domain_below_the_points() {
        // Probe case `f2_clip_narrow`: clip x in [0.1, 0.9], but the points at 0 and 1 widen it
        // back to [0, 1]. Values match the default clip.
        let mut mapping = combined(&auto5());
        mapping.clip = [0.1, 0.0, 0.9, 1.0];
        let prepared = mapping.prepare();
        assert_close(prepared.evaluate(3, 0.25), 0.373_742_25);
        assert_close(prepared.evaluate(3, 0.5), 0.393_788_04);
    }

    #[test]
    fn use_clip_does_not_change_evaluation() {
        // Probe case `n_noclip`: the display path never clips.
        let mut mapping = combined(&auto5());
        mapping.use_clip = false;
        assert_close(mapping.prepare().evaluate(3, 0.25), 0.373_742_25);
    }

    #[test]
    fn per_channel_curves_match_blender() {
        // Probe case `h_rgb_curves`: R is three points (AUTO), G is two points (AUTO).
        let mut mapping = combined(&auto5());
        mapping.curves[0] = curve_of(&[(0.0, 0.0, AUTO), (0.5, 0.7, AUTO), (1.0, 1.0, AUTO)]);
        mapping.curves[1] = curve_of(&[(0.0, 0.1, AUTO), (1.0, 0.8, AUTO)]);
        let prepared = mapping.prepare();
        assert_close(prepared.evaluate(0, -0.5), -0.932_312);
        assert_close(prepared.evaluate(0, 0.25), 0.422_546_2);
        assert_close(prepared.evaluate(0, 0.5), 0.7);
        assert_close(prepared.evaluate(0, 1.5), 1.218_764_9);
        assert_close(prepared.evaluate(1, -0.5), -0.250_000_03);
        assert_close(prepared.evaluate(1, 0.5), 0.450_000_05);
        assert_close(prepared.evaluate(1, 1.5), 1.150_000_1);
    }

    #[test]
    fn clamped_local_extrema_match_blender() {
        // Probe case `j_local_extrema`: a peak at 0.5 and a falling end.
        let mut mapping = combined(&[
            (0.0, 0.2, CLAMPED),
            (0.5, 1.0, CLAMPED),
            (1.0, 0.1, CLAMPED),
        ]);
        mapping.curves[0] = curve_of(&[(0.0, 0.2, AUTO), (0.5, 1.0, AUTO), (1.0, 0.1, AUTO)]);
        let prepared = mapping.prepare();
        assert_close(prepared.evaluate(3, -1.0), -1.4);
        assert_close(prepared.evaluate(3, 0.25), 0.830_868_84);
        assert_close(prepared.evaluate(3, 0.5), 1.0);
        assert_close(prepared.evaluate(3, 0.75), 0.818_523_5);
        assert_close(prepared.evaluate(3, 1.5), -0.799_999_95);
        assert_close(prepared.evaluate(0, 0.25), 0.908_760_55);
        assert_close(prepared.evaluate(0, 0.75), 0.903_418_1);
    }

    #[test]
    fn stored_points_after_update_are_evaluated() {
        // Probe case `k_update_clamp`: Blender's `update()` clamped the selected point's y from 1.3
        // to the clip maximum 1.0. The port starts from the stored points.
        let prepared = combined(&[(0.0, 0.0, AUTO), (0.5, 1.0, AUTO), (1.0, 1.0, AUTO)]).prepare();
        assert_close(prepared.evaluate(3, 0.25), 0.775_167_1);
        assert_close(prepared.evaluate(3, 0.5), 0.999_999_9);
        assert_close(prepared.evaluate(3, 1.5), 0.846_344_65);
    }

    #[test]
    fn film_like_tone_uses_the_combined_curve_for_evaluate() {
        // Probe case `i_filmlike`: Blender records the combined curve only for film-like tone.
        let mut mapping = combined(&auto5());
        mapping.tone = CurveTone::FilmLike;
        let prepared = mapping.prepare();
        assert_close(prepared.evaluate(3, 0.25), 0.373_742_25);
        assert_close(prepared.evaluate(3, 0.5), 0.393_788_04);
    }

    #[test]
    fn nan_inputs_return_nan_without_panicking() {
        let prepared = combined(&auto5()).prepare();
        assert!(prepared.evaluate(3, f32::NAN).is_nan());
        assert!(prepared.evaluate_rgb([f32::NAN, 0.5, 0.5])[0].is_nan());
        assert!(
            prepared
                .evaluate_rgb([0.5, 0.5, 0.5])
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn default_mapping_validates_and_round_trips_through_serde() -> std::result::Result<(), String>
    {
        let default = CurveMapping::default();
        default.validate().map_err(|error| error.to_string())?;
        let value = serde_json::to_value(&default).map_err(|error| error.to_string())?;
        let back: CurveMapping =
            serde_json::from_value(value).map_err(|error| error.to_string())?;
        assert_eq!(back, default);
        Ok(())
    }

    #[test]
    fn validate_rejects_invalid_curves() {
        let mut one_point = CurveMapping::default();
        one_point.curves[2].points.truncate(1);
        assert!(one_point.validate().is_err());

        let unsorted = combined(&[(0.5, 0.5, AUTO), (0.25, 0.2, AUTO)]);
        assert!(unsorted.validate().is_err());

        let inverted_clip = CurveMapping {
            clip: [1.0, 0.0, 0.0, 1.0],
            ..CurveMapping::default()
        };
        assert!(inverted_clip.validate().is_err());

        let non_finite = combined(&[(0.0, f32::INFINITY, AUTO), (1.0, 1.0, AUTO)]);
        assert!(non_finite.validate().is_err());
    }

    /// The operation form of [`auto5`] on the combined curve, with identity R, G, and B curves.
    fn operation_json() -> Value {
        let identity = json!({"points": [
            {"location": [0.0, 0.0], "handle": "auto"},
            {"location": [1.0, 1.0], "handle": "auto"}
        ]});
        json!({
            "tone": "standard",
            "extend": "extrapolated",
            "use_clip": true,
            "clip": [0.0, 0.0, 1.0, 1.0],
            "black": [0.0, 0.0, 0.0],
            "white": [1.0, 1.0, 1.0],
            "curves": [
                identity,
                identity,
                identity,
                {"points": [
                    {"location": [0.0, 0.0], "handle": "auto"},
                    {"location": [0.15, 0.4], "handle": "auto"},
                    {"location": [0.4, 0.3], "handle": "auto"},
                    {"location": [0.7, 0.85], "handle": "auto"},
                    {"location": [1.0, 1.0], "handle": "auto"}
                ]}
            ]
        })
    }

    #[test]
    fn from_json_reads_the_operation_form() -> std::result::Result<(), String> {
        let parsed = CurveMapping::from_json(&operation_json())?;
        assert_eq!(parsed, combined(&auto5()));
        Ok(())
    }

    #[test]
    fn from_json_rejects_unknown_fields_and_bad_handles() {
        let mut value = operation_json();
        value["extra"] = json!(1);
        let error = CurveMapping::from_json(&value).err().unwrap_or_default();
        assert!(error.contains("unknown field `extra`"), "{error}");

        let mut value = operation_json();
        value["curves"][0]["points"][0]["handle"] = json!("free");
        let error = CurveMapping::from_json(&value).err().unwrap_or_default();
        assert!(error.contains("curve/curves/0/points/0/handle"), "{error}");
    }
}
