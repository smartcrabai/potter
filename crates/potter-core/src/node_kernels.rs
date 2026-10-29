use serde_json::Value as JsonValue;

use crate::color::curve::{CurveMapping, PreparedCurve};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MathSemantics {
    Cycles,
    Geometry,
    Eevee,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EeveeModuloError {
    QuotientOutOfRange,
    PrecisionBoundary,
}

/// The 256 intervals plus endpoint used by Blender's ramp lookup tables.
pub(crate) const COLOR_RAMP_TABLE_SIZE: usize = 257;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum KernelValue {
    Scalar(f32),
    Vector([f32; 3]),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CompareValue<'a> {
    Scalar(f32),
    Integer(i32),
    Vector([f32; 3]),
    Color([f32; 4]),
    Boolean(bool),
    String(&'a str),
}

pub(crate) fn math(
    operation: &str,
    a: f32,
    b: f32,
    c: f32,
    clamp: bool,
    semantics: MathSemantics,
) -> Option<f32> {
    let value = match operation {
        "ADD" => a + b,
        "SUBTRACT" => a - b,
        "MULTIPLY" => a * b,
        "DIVIDE" => safe_divide(a, b),
        "MULTIPLY_ADD" => a * b + c,
        "POWER" => safe_power(a, b, semantics),
        "LOGARITHM" => match semantics {
            MathSemantics::Eevee if a > 0.0 && b > 0.0 => a.log2() / b.log2(),
            _ => safe_log(a, b),
        },
        "SQRT" => safe_sqrt(a),
        "INV_SQRT" | "INVERSE_SQRT" => match semantics {
            MathSemantics::Cycles | MathSemantics::Geometry => safe_inverse_sqrt(a),
            MathSemantics::Eevee => 1.0 / a.sqrt(),
        },
        "ABSOLUTE" => a.abs(),
        "EXPONENT" => a.exp(),
        "MINIMUM" => min_float(a, b, semantics),
        "MAXIMUM" => max_float(a, b, semantics),
        "LESS_THAN" => bool_float(a < b),
        "GREATER_THAN" => bool_float(a > b),
        "SIGN" => sign(a, semantics),
        "COMPARE" => match semantics {
            MathSemantics::Cycles | MathSemantics::Geometry => {
                bool_float(equal_f32(a, b) || (a - b).abs() <= c.max(f32::EPSILON))
            }
            MathSemantics::Eevee => bool_float((a - b).abs() <= c.max(1.0e-5)),
        },
        "SMOOTH_MIN" => smooth_min(a, b, c, semantics),
        "SMOOTH_MAX" => -smooth_min(-a, -b, c, semantics),
        "ROUND" => (a + 0.5).floor(),
        "FLOOR" => a.floor(),
        "CEIL" => a.ceil(),
        "TRUNC" => a.trunc(),
        "FRACT" | "FRACTION" => a - a.floor(),
        "MODULO" => match semantics {
            MathSemantics::Eevee => eevee_compatible_mod(a, b).ok()?,
            MathSemantics::Cycles | MathSemantics::Geometry => safe_modulo(a, b),
        },
        "FLOORED_MODULO" => safe_floored_modulo(a, b),
        "WRAP" => wrap(a, b, c),
        "SNAP" => safe_divide(a, b).floor() * b,
        "PINGPONG" => pingpong(a, b, semantics),
        "SINE" => a.sin(),
        "COSINE" => a.cos(),
        "TANGENT" => a.tan(),
        "ARCSINE" | "ASIN" => match semantics {
            MathSemantics::Eevee if !(-1.0..=1.0).contains(&a) => 0.0,
            _ => a.clamp(-1.0, 1.0).asin(),
        },
        "ARCCOSINE" | "ACOS" => match semantics {
            MathSemantics::Eevee if !(-1.0..=1.0).contains(&a) => 0.0,
            _ => a.clamp(-1.0, 1.0).acos(),
        },
        "ARCTANGENT" | "ATAN" => a.atan(),
        "ARCTAN2" | "ARCTAN_2" => match semantics {
            MathSemantics::Cycles | MathSemantics::Eevee if a == 0.0 && b == 0.0 => 0.0,
            MathSemantics::Cycles | MathSemantics::Eevee | MathSemantics::Geometry => a.atan2(b),
        },
        "SINH" => a.sinh(),
        "COSH" => a.cosh(),
        "TANH" => a.tanh(),
        "DEGREES" => a * (180.0 / std::f32::consts::PI),
        "RADIANS" => a * (std::f32::consts::PI / 180.0),
        _ => return None,
    };
    Some(if clamp { value.clamp(0.0, 1.0) } else { value })
}

pub(crate) fn vector_math(
    operation: &str,
    output: &str,
    first: [f32; 3],
    second: [f32; 3],
    third: [f32; 3],
    scale: f32,
    semantics: MathSemantics,
) -> Option<KernelValue> {
    let scalar = match operation {
        "DOT_PRODUCT" => Some(dot3(first, second)),
        "DISTANCE" => Some(length3(sub3(first, second))),
        "LENGTH" => Some(length3(first)),
        _ => None,
    };
    if let Some(value) = scalar {
        return (output == "Value").then_some(KernelValue::Scalar(value));
    }
    if output != "Vector" {
        return None;
    }
    let value = match operation {
        "ADD" => add3(first, second),
        "SUBTRACT" => sub3(first, second),
        "MULTIPLY" => mul3(first, second),
        "DIVIDE" => zip3(first, second, safe_divide),
        "CROSS_PRODUCT" => cross3(first, second),
        "PROJECT" => project3(first, second),
        "REFLECT" => {
            let normal = vector_normal(second, semantics, true);
            sub3(first, scale3(normal, 2.0 * dot3(first, normal)))
        }
        "REFRACT" => refract3(first, second, scale, semantics),
        "FACEFORWARD" => {
            if dot3(third, second) < 0.0 {
                first
            } else {
                scale3(first, -1.0)
            }
        }
        "MULTIPLY_ADD" => add3(mul3(first, second), third),
        "SCALE" => scale3(first, scale),
        "NORMALIZE" => match semantics {
            MathSemantics::Eevee => eevee_normalize(first)?,
            MathSemantics::Cycles | MathSemantics::Geometry => {
                vector_normal(first, semantics, false)
            }
        },
        "SNAP" => zip3(first, second, |a, b| safe_divide(a, b).floor() * b),
        "ROUND" => first.map(|component| (component + 0.5).floor()),
        "FLOOR" => first.map(f32::floor),
        "CEIL" => first.map(f32::ceil),
        "MODULO" => match semantics {
            MathSemantics::Eevee => eevee_vector_modulo(first, second).ok()?,
            MathSemantics::Cycles | MathSemantics::Geometry => zip3(first, second, safe_modulo),
        },
        "FRACTION" | "FRACT" => first.map(|component| component - component.floor()),
        "ABSOLUTE" => first.map(f32::abs),
        "MINIMUM" => zip3(first, second, |a, b| min_float(a, b, semantics)),
        "MAXIMUM" => zip3(first, second, |a, b| max_float(a, b, semantics)),
        "WRAP" => std::array::from_fn(|index| wrap(first[index], second[index], third[index])),
        "SINE" => first.map(f32::sin),
        "COSINE" => first.map(f32::cos),
        "TANGENT" => first.map(f32::tan),
        "POWER" => zip3(first, second, |a, b| match semantics {
            MathSemantics::Eevee => eevee_compatible_pow(a, b),
            MathSemantics::Cycles | MathSemantics::Geometry => safe_power(a, b, semantics),
        }),
        "SIGN" => first.map(|value| sign(value, semantics)),
        _ => return None,
    };
    Some(KernelValue::Vector(value))
}

fn safe_divide(a: f32, b: f32) -> f32 {
    if b == 0.0 { 0.0 } else { a / b }
}

fn safe_modulo(a: f32, b: f32) -> f32 {
    if b == 0.0 { 0.0 } else { a % b }
}

pub(crate) fn eevee_compatible_mod(a: f32, b: f32) -> Result<f32, EeveeModuloError> {
    if b == 0.0 {
        return Ok(0.0);
    }
    // GLSL float-to-int conversion is undefined outside the signed 32-bit range.
    let quotient = a / b;
    if !quotient.is_finite() || quotient < i32::MIN as f32 || quotient >= 2_147_483_648.0 {
        return Err(EeveeModuloError::QuotientOutOfRange);
    }
    let truncated = quotient as i32;
    let result = a - truncated as f32 * b;
    let tolerance = f32::EPSILON * quotient.abs().max(1.0);
    if (quotient - quotient.round()).abs() <= tolerance && !equal_f32(result, safe_modulo(a, b)) {
        return Err(EeveeModuloError::PrecisionBoundary);
    }
    Ok(result)
}

pub(crate) fn eevee_vector_modulo(a: [f32; 3], b: [f32; 3]) -> Result<[f32; 3], EeveeModuloError> {
    Ok([
        eevee_compatible_mod(a[0], b[0])?,
        eevee_compatible_mod(a[1], b[1])?,
        eevee_compatible_mod(a[2], b[2])?,
    ])
}

pub(crate) fn eevee_normalize(value: [f32; 3]) -> Option<[f32; 3]> {
    // EEVEE's handling of subnormal length-squared depends on the GPU's float mode.
    let length_squared = dot3(value, value);
    if (length_squared > 0.0 && length_squared < f32::MIN_POSITIVE)
        || value
            .iter()
            .any(|component| *component != 0.0 && component.abs() < f32::MIN_POSITIVE)
    {
        return None;
    }
    Some(vector_normal(value, MathSemantics::Eevee, false))
}

fn eevee_compatible_pow(base: f32, exponent: f32) -> f32 {
    if exponent == 0.0 {
        1.0
    } else if base < 0.0 {
        let magnitude = -base;
        let parity = -exponent - 2.0 * (-exponent / 2.0).floor();
        if parity == 0.0 {
            magnitude.powf(exponent)
        } else {
            -magnitude.powf(exponent)
        }
    } else if base == 0.0 {
        0.0
    } else {
        base.powf(exponent)
    }
}

fn eevee_math_power(base: f32, exponent: f32) -> f32 {
    if base >= 0.0 {
        eevee_compatible_pow(base, exponent)
    } else {
        let fractional = exponent.abs() - exponent.abs().floor();
        if fractional > 0.999 || fractional < 0.001 {
            eevee_compatible_pow(base, (exponent + 0.5).floor())
        } else {
            0.0
        }
    }
}

fn safe_floored_modulo(a: f32, b: f32) -> f32 {
    if b == 0.0 {
        0.0
    } else {
        a - (a / b).floor() * b
    }
}

fn safe_power(base: f32, exponent: f32, semantics: MathSemantics) -> f32 {
    match semantics {
        MathSemantics::Eevee => eevee_math_power(base, exponent),
        MathSemantics::Cycles => {
            let integral_exponent = exponent >= i32::MIN as f32
                && exponent < 2_147_483_648.0
                && equal_f32((exponent as i32) as f32, exponent);
            if base == 0.0 && exponent < 0.0 {
                0.0
            } else if base >= 0.0 || integral_exponent {
                base.powf(exponent)
            } else {
                0.0
            }
        }
        MathSemantics::Geometry => {
            if base >= 0.0 || exponent.fract() == 0.0 {
                base.powf(exponent)
            } else {
                0.0
            }
        }
    }
}

fn safe_log(value: f32, base: f32) -> f32 {
    if value > 0.0 && base > 0.0 {
        safe_divide(value.ln(), base.ln())
    } else {
        0.0
    }
}

fn safe_sqrt(value: f32) -> f32 {
    value.max(0.0).sqrt()
}

fn safe_inverse_sqrt(value: f32) -> f32 {
    if value > 0.0 { 1.0 / value.sqrt() } else { 0.0 }
}

fn min_float(a: f32, b: f32, semantics: MathSemantics) -> f32 {
    match semantics {
        MathSemantics::Cycles => a.min(b),
        MathSemantics::Geometry | MathSemantics::Eevee => {
            if b < a {
                b
            } else {
                a
            }
        }
    }
}

fn max_float(a: f32, b: f32, semantics: MathSemantics) -> f32 {
    match semantics {
        MathSemantics::Cycles => a.max(b),
        MathSemantics::Geometry | MathSemantics::Eevee => {
            if a < b {
                b
            } else {
                a
            }
        }
    }
}

fn equal_f32(first: f32, second: f32) -> bool {
    first.partial_cmp(&second) == Some(std::cmp::Ordering::Equal)
}

fn sign(value: f32, semantics: MathSemantics) -> f32 {
    match semantics {
        MathSemantics::Cycles if value == 0.0 => 0.0,
        MathSemantics::Cycles if value < 0.0 => -1.0,
        MathSemantics::Cycles => 1.0,
        MathSemantics::Geometry | MathSemantics::Eevee if value > 0.0 => 1.0,
        MathSemantics::Geometry | MathSemantics::Eevee if value < 0.0 => -1.0,
        MathSemantics::Geometry | MathSemantics::Eevee => 0.0,
    }
}

fn bool_float(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

fn smooth_min(a: f32, b: f32, width: f32, semantics: MathSemantics) -> f32 {
    if width == 0.0 {
        return min_float(a, b, semantics);
    }
    let h = (width - (a - b).abs()).max(0.0) / width;
    min_float(a, b, semantics) - h * h * h * width * (1.0 / 6.0)
}

fn wrap(value: f32, maximum: f32, minimum: f32) -> f32 {
    let range = maximum - minimum;
    if range == 0.0 {
        minimum
    } else {
        value - range * ((value - minimum) / range).floor()
    }
}

fn pingpong(value: f32, scale: f32, semantics: MathSemantics) -> f32 {
    match semantics {
        MathSemantics::Cycles => {
            if scale == 0.0 {
                0.0
            } else {
                let period = scale * 2.0;
                let fraction = (value - scale) / period;
                ((fraction - fraction.floor()) * period - scale).abs()
            }
        }
        MathSemantics::Geometry => {
            let length = scale.abs();
            if length == 0.0 {
                0.0
            } else {
                length - ((value.rem_euclid(2.0 * length)) - length).abs()
            }
        }
        MathSemantics::Eevee => {
            if scale == 0.0 {
                0.0
            } else {
                let fraction = (value - scale) / (scale * 2.0);
                ((fraction - fraction.floor()) * scale * 2.0 - scale).abs()
            }
        }
    }
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|index| a[index] + b[index])
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|index| a[index] - b[index])
}

fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|index| a[index] * b[index])
}

fn scale3(value: [f32; 3], scale: f32) -> [f32; 3] {
    value.map(|component| component * scale)
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length3(value: [f32; 3]) -> f32 {
    dot3(value, value).sqrt()
}

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let length = length3(value);
    if length == 0.0 {
        value
    } else {
        scale3(value, 1.0 / length)
    }
}

fn normalize3_with_threshold(value: [f32; 3], threshold: f32) -> [f32; 3] {
    let length_squared = dot3(value, value);
    if length_squared > threshold {
        scale3(value, 1.0 / length_squared.sqrt())
    } else {
        [0.0; 3]
    }
}
fn normalize3_geometry(value: [f32; 3]) -> [f32; 3] {
    let length_squared = dot3(value, value);
    if length_squared > 1.0e-35 {
        let length = length_squared.sqrt();
        value.map(|component| component / length)
    } else {
        [0.0; 3]
    }
}

fn vector_normal(value: [f32; 3], semantics: MathSemantics, safe: bool) -> [f32; 3] {
    match semantics {
        MathSemantics::Cycles => normalize3(value),
        MathSemantics::Geometry => normalize3_geometry(value),
        MathSemantics::Eevee if safe => normalize3_with_threshold(value, 1.0e-35),
        MathSemantics::Eevee => {
            let length_squared = dot3(value, value);
            if length_squared > 0.0 {
                scale3(value, 1.0 / length_squared.sqrt())
            } else {
                value
            }
        }
    }
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn zip3(a: [f32; 3], b: [f32; 3], f: impl Fn(f32, f32) -> f32) -> [f32; 3] {
    std::array::from_fn(|index| f(a[index], b[index]))
}

fn project3(value: [f32; 3], onto: [f32; 3]) -> [f32; 3] {
    let length_squared = dot3(onto, onto);
    if length_squared == 0.0 {
        [0.0; 3]
    } else {
        scale3(onto, dot3(value, onto) / length_squared)
    }
}

fn refract3(incident: [f32; 3], normal: [f32; 3], eta: f32, semantics: MathSemantics) -> [f32; 3] {
    let normal = vector_normal(normal, semantics, true);
    let dot = dot3(normal, incident);
    let discriminant = 1.0 - eta * eta * (1.0 - dot * dot);
    if discriminant < 0.0 {
        [0.0; 3]
    } else {
        sub3(
            scale3(incident, eta),
            scale3(normal, eta * dot + discriminant.sqrt()),
        )
    }
}

pub(crate) fn map_range(
    value: KernelValue,
    from_min: KernelValue,
    from_max: KernelValue,
    to_min: KernelValue,
    to_max: KernelValue,
    steps: KernelValue,
    interpolation: &str,
    clamp: bool,
) -> Option<KernelValue> {
    map_range_with_semantics(
        value,
        from_min,
        from_max,
        to_min,
        to_max,
        steps,
        interpolation,
        clamp,
        MathSemantics::Geometry,
    )
}

pub(crate) fn shader_map_range(
    value: KernelValue,
    from_min: KernelValue,
    from_max: KernelValue,
    to_min: KernelValue,
    to_max: KernelValue,
    steps: KernelValue,
    interpolation: &str,
    clamp: bool,
    semantics: MathSemantics,
) -> Option<KernelValue> {
    map_range_with_semantics(
        value,
        from_min,
        from_max,
        to_min,
        to_max,
        steps,
        interpolation,
        clamp,
        semantics,
    )
}

fn map_range_with_semantics(
    value: KernelValue,
    from_min: KernelValue,
    from_max: KernelValue,
    to_min: KernelValue,
    to_max: KernelValue,
    steps: KernelValue,
    interpolation: &str,
    clamp: bool,
    semantics: MathSemantics,
) -> Option<KernelValue> {
    match (value, from_min, from_max, to_min, to_max, steps) {
        (
            KernelValue::Scalar(value),
            KernelValue::Scalar(from_min),
            KernelValue::Scalar(from_max),
            KernelValue::Scalar(to_min),
            KernelValue::Scalar(to_max),
            KernelValue::Scalar(steps),
        ) => {
            // Shader scalar Map Range returns zero for degenerate inputs; GN keeps safe_divide behavior.
            if semantics != MathSemantics::Geometry && equal_f32(from_max, from_min) {
                let apply_clamp = clamp
                    && (semantics == MathSemantics::Cycles
                        || matches!(interpolation, "LINEAR" | "STEPPED"));
                Some(KernelValue::Scalar(if apply_clamp {
                    clamp_range(0.0, to_min, to_max)
                } else {
                    0.0
                }))
            } else {
                map_range_scalar(
                    value,
                    from_min,
                    from_max,
                    to_min,
                    to_max,
                    steps,
                    interpolation,
                    clamp,
                    semantics,
                )
                .map(KernelValue::Scalar)
            }
        }
        (
            KernelValue::Vector(value),
            KernelValue::Vector(from_min),
            KernelValue::Vector(from_max),
            KernelValue::Vector(to_min),
            KernelValue::Vector(to_max),
            KernelValue::Vector(steps),
        ) => {
            let vector_semantics = if semantics == MathSemantics::Cycles {
                MathSemantics::Cycles
            } else {
                MathSemantics::Geometry
            };
            let x = map_range_scalar(
                value[0],
                from_min[0],
                from_max[0],
                to_min[0],
                to_max[0],
                steps[0],
                interpolation,
                clamp,
                vector_semantics,
            )?;
            let y = map_range_scalar(
                value[1],
                from_min[1],
                from_max[1],
                to_min[1],
                to_max[1],
                steps[1],
                interpolation,
                clamp,
                vector_semantics,
            )?;
            let z = map_range_scalar(
                value[2],
                from_min[2],
                from_max[2],
                to_min[2],
                to_max[2],
                steps[2],
                interpolation,
                clamp,
                vector_semantics,
            )?;
            Some(KernelValue::Vector([x, y, z]))
        }
        _ => None,
    }
}

fn map_range_scalar(
    value: f32,
    from_min: f32,
    from_max: f32,
    to_min: f32,
    to_max: f32,
    steps: f32,
    interpolation: &str,
    clamp: bool,
    semantics: MathSemantics,
) -> Option<f32> {
    let mut factor = safe_divide(value - from_min, from_max - from_min);
    let result = match interpolation {
        "LINEAR" => {
            let result = to_min + factor * (to_max - to_min);
            if clamp {
                clamp_range(result, to_min, to_max)
            } else {
                result
            }
        }
        "STEPPED" => {
            factor = if semantics != MathSemantics::Geometry
                && steps.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
            {
                0.0
            } else {
                safe_divide((factor * (steps + 1.0)).floor(), steps)
            };
            let result = to_min + factor * (to_max - to_min);
            if clamp {
                clamp_range(result, to_min, to_max)
            } else {
                result
            }
        }
        "SMOOTHSTEP" => {
            factor = factor.clamp(0.0, 1.0);
            factor = (3.0 - 2.0 * factor) * (factor * factor);
            to_min + factor * (to_max - to_min)
        }
        "SMOOTHERSTEP" => {
            factor = factor.clamp(0.0, 1.0);
            factor = factor * factor * factor * (factor * (factor * 6.0 - 15.0) + 10.0);
            to_min + factor * (to_max - to_min)
        }
        _ => return None,
    };
    Some(result)
}

fn clamp_minmax(value: f32, minimum: f32, maximum: f32) -> f32 {
    let value = if value < minimum { minimum } else { value };
    if value > maximum { maximum } else { value }
}

fn clamp_range(value: f32, first: f32, second: f32) -> f32 {
    if first > second {
        clamp_minmax(value, second, first)
    } else {
        clamp_minmax(value, first, second)
    }
}

pub(crate) fn clamp(
    value: KernelValue,
    minimum: KernelValue,
    maximum: KernelValue,
    mode: &str,
) -> Option<KernelValue> {
    let clamp_component = |value, minimum, maximum| match mode {
        "MINMAX" => Some(clamp_minmax(value, minimum, maximum)),
        "RANGE" => Some(clamp_range(value, minimum, maximum)),
        _ => None,
    };
    match (value, minimum, maximum) {
        (
            KernelValue::Scalar(value),
            KernelValue::Scalar(minimum),
            KernelValue::Scalar(maximum),
        ) => clamp_component(value, minimum, maximum).map(KernelValue::Scalar),
        (
            KernelValue::Vector(value),
            KernelValue::Vector(minimum),
            KernelValue::Vector(maximum),
        ) => {
            let x = clamp_component(value[0], minimum[0], maximum[0])?;
            let y = clamp_component(value[1], minimum[1], maximum[1])?;
            let z = clamp_component(value[2], minimum[2], maximum[2])?;
            Some(KernelValue::Vector([x, y, z]))
        }
        _ => None,
    }
}

pub(crate) fn mix_float(
    mode: &str,
    factor: f32,
    a: f32,
    b: f32,
    clamp_factor: bool,
    _clamp_result: bool,
) -> Option<f32> {
    if !valid_blend_mode(mode) {
        return None;
    }
    let factor = if clamp_factor {
        factor.clamp(0.0, 1.0)
    } else {
        factor
    };
    Some((1.0 - factor) * a + factor * b)
}

pub(crate) fn mix_vector(
    mode: &str,
    factor: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    clamp_factor: bool,
    _clamp_result: bool,
) -> Option<[f32; 3]> {
    if !valid_blend_mode(mode) {
        return None;
    }
    Some(std::array::from_fn(|index| {
        let factor = if clamp_factor {
            factor[index].clamp(0.0, 1.0)
        } else {
            factor[index]
        };
        (1.0 - factor) * a[index] + factor * b[index]
    }))
}

pub(crate) fn mix_rotation(
    factor: f32,
    a_xyzw: [f32; 4],
    b_xyzw: [f32; 4],
    clamp_factor: bool,
) -> Option<[f32; 4]> {
    // Match BLI's unit-scale assertion, which also permits near-zero quaternions.
    const UNIT_SCALE_EPSILON: f32 = 0.002;
    let unit_scale = |quaternion: [f32; 4]| {
        let norm_squared = quaternion[0] * quaternion[0]
            + quaternion[1] * quaternion[1]
            + quaternion[2] * quaternion[2]
            + quaternion[3] * quaternion[3];
        let below_or_unordered = |value: f32| {
            matches!(
                value.partial_cmp(&UNIT_SCALE_EPSILON),
                None | Some(std::cmp::Ordering::Less)
            )
        };
        below_or_unordered((norm_squared - 1.0).abs()) || below_or_unordered(norm_squared.abs())
    };
    if !unit_scale(a_xyzw) || !unit_scale(b_xyzw) {
        return None;
    }

    let factor = if clamp_factor {
        factor.clamp(0.0, 1.0)
    } else {
        factor
    };
    let a = [a_xyzw[3], a_xyzw[0], a_xyzw[1], a_xyzw[2]];
    let b = [b_xyzw[3], b_xyzw[0], b_xyzw[1], b_xyzw[2]];
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let conjugate_a = [a[0], -a[1], -a[2], -a[3]];
    let relative_a = if dot < 0.0 {
        [
            -conjugate_a[0],
            -conjugate_a[1],
            -conjugate_a[2],
            -conjugate_a[3],
        ]
    } else {
        conjugate_a
    };
    let relative = [
        relative_a[0] * b[0] - relative_a[1] * b[1] - relative_a[2] * b[2] - relative_a[3] * b[3],
        relative_a[0] * b[1] + relative_a[1] * b[0] + relative_a[2] * b[3] - relative_a[3] * b[2],
        relative_a[0] * b[2] + relative_a[2] * b[0] + relative_a[3] * b[1] - relative_a[1] * b[3],
        relative_a[0] * b[3] + relative_a[3] * b[0] + relative_a[1] * b[2] - relative_a[2] * b[1],
    ];
    let safe_acos = if relative[0] <= -1.0 {
        std::f32::consts::PI
    } else if relative[0] >= 1.0 {
        0.0
    } else {
        relative[0].acos()
    };
    let half_angle = factor * safe_acos;
    let imaginary_length_squared =
        relative[1] * relative[1] + relative[2] * relative[2] + relative[3] * relative[3];
    let normalized_imaginary = if imaginary_length_squared > 1.0e-35 {
        let length = imaginary_length_squared.sqrt();
        [
            relative[1] / length,
            relative[2] / length,
            relative[3] / length,
        ]
    } else {
        [0.0; 3]
    };
    let sine = half_angle.sin();
    let power = [
        half_angle.cos(),
        sine * normalized_imaginary[0],
        sine * normalized_imaginary[1],
        sine * normalized_imaginary[2],
    ];
    let result = [
        a[0] * power[0] - a[1] * power[1] - a[2] * power[2] - a[3] * power[3],
        a[0] * power[1] + a[1] * power[0] + a[2] * power[3] - a[3] * power[2],
        a[0] * power[2] + a[2] * power[0] + a[3] * power[1] - a[1] * power[3],
        a[0] * power[3] + a[3] * power[0] + a[1] * power[2] - a[2] * power[1],
    ];
    Some([result[1], result[2], result[3], result[0]])
}

pub(crate) fn mix_color(
    mode: &str,
    factor: f32,
    a: [f32; 4],
    b: [f32; 4],
    clamp_factor: bool,
    clamp_result: bool,
) -> Option<[f32; 4]> {
    if !valid_blend_mode(mode) {
        return None;
    }
    let factor = if clamp_factor {
        factor.clamp(0.0, 1.0)
    } else {
        factor
    };
    let mut result = blend_rgba(mode, factor, a, b)?;
    if clamp_result {
        result = result.map(|component| component.clamp(0.0, 1.0));
    }
    Some(result)
}

fn valid_blend_mode(mode: &str) -> bool {
    matches!(
        mode,
        "MIX"
            | "DARKEN"
            | "MULTIPLY"
            | "BURN"
            | "LIGHTEN"
            | "SCREEN"
            | "DODGE"
            | "ADD"
            | "OVERLAY"
            | "SOFT_LIGHT"
            | "LINEAR_LIGHT"
            | "DIFFERENCE"
            | "EXCLUSION"
            | "SUBTRACT"
            | "DIVIDE"
            | "HUE"
            | "SATURATION"
            | "VALUE"
            | "COLOR"
    )
}

fn blend_rgba(mode: &str, factor: f32, a: [f32; 4], b: [f32; 4]) -> Option<[f32; 4]> {
    let inverse = 1.0 - factor;
    let mut result = a;
    match mode {
        "MIX" => return Some(std::array::from_fn(|i| inverse * a[i] + factor * b[i])),
        "ADD" => {
            for channel in 0..3 {
                result[channel] += factor * b[channel];
            }
        }
        "MULTIPLY" => {
            for channel in 0..3 {
                result[channel] *= inverse + factor * b[channel];
            }
        }
        "SCREEN" => {
            for channel in 0..3 {
                result[channel] =
                    1.0 - (inverse + factor * (1.0 - b[channel])) * (1.0 - a[channel]);
            }
        }
        "OVERLAY" => {
            for channel in 0..3 {
                result[channel] = if a[channel] < 0.5 {
                    a[channel] * (inverse + 2.0 * factor * b[channel])
                } else {
                    1.0 - (inverse + 2.0 * factor * (1.0 - b[channel])) * (1.0 - a[channel])
                };
            }
        }
        "SUBTRACT" => {
            for channel in 0..3 {
                result[channel] -= factor * b[channel];
            }
        }
        "DIVIDE" => {
            for channel in 0..3 {
                if b[channel] != 0.0 {
                    result[channel] = inverse * a[channel] + factor * a[channel] / b[channel];
                }
            }
        }
        "DIFFERENCE" => {
            for channel in 0..3 {
                result[channel] = inverse * a[channel] + factor * (a[channel] - b[channel]).abs();
            }
        }
        "EXCLUSION" => {
            for channel in 0..3 {
                result[channel] = (inverse * a[channel]
                    + factor * (a[channel] + b[channel] - 2.0 * a[channel] * b[channel]))
                    .max(0.0);
            }
        }
        "DARKEN" | "LIGHTEN" => {
            for channel in 0..3 {
                let blend = if mode == "DARKEN" {
                    a[channel].min(b[channel])
                } else {
                    a[channel].max(b[channel])
                };
                result[channel] = blend * factor + a[channel] * inverse;
            }
        }
        "DODGE" => {
            for channel in 0..3 {
                if a[channel] != 0.0 {
                    let denominator = 1.0 - factor * b[channel];
                    result[channel] = if denominator <= 0.0 {
                        1.0
                    } else {
                        let ratio = a[channel] / denominator;
                        if ratio > 1.0 { 1.0 } else { ratio }
                    };
                }
            }
        }
        "BURN" => {
            for channel in 0..3 {
                let denominator = inverse + factor * b[channel];
                result[channel] = if denominator <= 0.0 {
                    0.0
                } else {
                    (1.0 - (1.0 - a[channel]) / denominator).clamp(0.0, 1.0)
                };
            }
        }
        "HUE" => {
            let second = rgb_to_hsv([b[0], b[1], b[2]]);
            if second[1] != 0.0 {
                let first = rgb_to_hsv([a[0], a[1], a[2]]);
                let blended = hsv_to_rgb([second[0], first[1], first[2]]);
                for channel in 0..3 {
                    result[channel] = inverse * a[channel] + factor * blended[channel];
                }
            }
        }
        "SATURATION" => {
            let first = rgb_to_hsv([a[0], a[1], a[2]]);
            if first[1] != 0.0 {
                let second = rgb_to_hsv([b[0], b[1], b[2]]);
                let blended =
                    hsv_to_rgb([first[0], inverse * first[1] + factor * second[1], first[2]]);
                result[..3].copy_from_slice(&blended);
            }
        }
        "VALUE" => {
            let first = rgb_to_hsv([a[0], a[1], a[2]]);
            let second = rgb_to_hsv([b[0], b[1], b[2]]);
            let blended = hsv_to_rgb([first[0], first[1], inverse * first[2] + factor * second[2]]);
            result[..3].copy_from_slice(&blended);
        }
        "COLOR" => {
            let second = rgb_to_hsv([b[0], b[1], b[2]]);
            if second[1] != 0.0 {
                let first = rgb_to_hsv([a[0], a[1], a[2]]);
                let blended = hsv_to_rgb([second[0], second[1], first[2]]);
                for channel in 0..3 {
                    result[channel] = inverse * a[channel] + factor * blended[channel];
                }
            }
        }
        "SOFT_LIGHT" => {
            for channel in 0..3 {
                let screen = 1.0 - (1.0 - b[channel]) * (1.0 - a[channel]);
                result[channel] = inverse * a[channel]
                    + factor * ((1.0 - a[channel]) * b[channel] * a[channel] + a[channel] * screen);
            }
        }
        "LINEAR_LIGHT" => {
            for channel in 0..3 {
                result[channel] = if b[channel] > 0.5 {
                    a[channel] + factor * (2.0 * (b[channel] - 0.5))
                } else {
                    a[channel] + factor * (2.0 * b[channel] - 1.0)
                };
            }
        }
        _ => return None,
    }
    Some(result)
}

fn rgb_to_hsv([mut red, mut green, mut blue]: [f32; 3]) -> [f32; 3] {
    let mut offset = 0.0;
    let mut minimum_green_blue;
    if green < blue {
        std::mem::swap(&mut green, &mut blue);
        offset = -1.0;
    }
    minimum_green_blue = blue;
    if red < green {
        std::mem::swap(&mut red, &mut green);
        offset = -2.0 / 6.0 - offset;
        minimum_green_blue = green.min(blue);
    }
    let chroma = red - minimum_green_blue;
    let hue = (offset + (green - blue) / (6.0 * chroma + 1e-20)).abs();
    let saturation = chroma / (red + 1e-20);
    [hue, saturation, red]
}

fn hsv_to_rgb([hue, saturation, value]: [f32; 3]) -> [f32; 3] {
    let red = ((hue * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0);
    let green = (2.0 - (hue * 6.0 - 2.0).abs()).clamp(0.0, 1.0);
    let blue = (2.0 - (hue * 6.0 - 4.0).abs()).clamp(0.0, 1.0);
    [
        ((red - 1.0) * saturation + 1.0) * value,
        ((green - 1.0) * saturation + 1.0) * value,
        ((blue - 1.0) * saturation + 1.0) * value,
    ]
}

pub(crate) fn color_ramp(
    elements: &[(f32, [f32; 4])],
    factor: f32,
    interpolation: &str,
    color_mode: &str,
    hue_interpolation: &str,
) -> Option<[f32; 4]> {
    if elements.is_empty() {
        return None;
    }
    if !matches!(color_mode, "RGB" | "HSV" | "HSL") {
        return None;
    }
    if !matches!(
        interpolation,
        "LINEAR" | "EASE" | "CONSTANT" | "B_SPLINE" | "CARDINAL"
    ) {
        return None;
    }
    if !matches!(hue_interpolation, "NEAR" | "FAR" | "CW" | "CCW") {
        return None;
    }
    if elements.len() == 1 {
        return Some(elements[0].1);
    }

    let interpolation = if color_mode == "RGB" {
        interpolation
    } else {
        "LINEAR"
    };
    let first = elements[0];
    if factor <= first.0 && matches!(interpolation, "LINEAR" | "EASE" | "CONSTANT") {
        return Some(first.1);
    }
    let index = elements
        .iter()
        .position(|element| element.0 > factor)
        .unwrap_or(elements.len());
    let (lower, upper) = if index == elements.len() {
        if matches!(interpolation, "LINEAR" | "EASE" | "CONSTANT") {
            return Some(elements[elements.len() - 1].1);
        }
        let last = elements[elements.len() - 1];
        (last, (1.0, last.1))
    } else if index == 0 {
        let first = elements[0];
        ((0.0, first.1), first)
    } else {
        (elements[index - 1], elements[index])
    };
    if interpolation == "CONSTANT" {
        return Some(lower.1);
    }
    let mut amount = if equal_f32(lower.0, upper.0) {
        if index == elements.len() { 1.0 } else { 0.0 }
    } else {
        (factor - upper.0) / (lower.0 - upper.0)
    };

    if interpolation == "B_SPLINE" || interpolation == "CARDINAL" {
        amount = amount.clamp(0.0, 1.0);
        let p0 = if index >= elements.len() - 1 {
            upper.1
        } else {
            elements[index + 1].1
        };
        let p3 = if index < 2 {
            lower.1
        } else {
            elements[index - 2].1
        };
        let weights = spline_weights(amount, interpolation == "CARDINAL");
        return Some(std::array::from_fn(|channel| {
            let value = weights[3] * p3[channel]
                + weights[2] * lower.1[channel]
                + weights[1] * upper.1[channel]
                + weights[0] * p0[channel];
            value.clamp(0.0, 1.0)
        }));
    }

    if interpolation == "EASE" {
        let squared = amount * amount;
        amount = 3.0 * squared - 2.0 * squared * amount;
    }
    let inverse = 1.0 - amount;
    if color_mode == "RGB" {
        return Some(std::array::from_fn(|channel| {
            inverse * upper.1[channel] + amount * lower.1[channel]
        }));
    }

    let upper_rgb = [upper.1[0], upper.1[1], upper.1[2]];
    let lower_rgb = [lower.1[0], lower.1[1], lower.1[2]];
    let upper_color = if color_mode == "HSV" {
        rgb_to_hsv(upper_rgb)
    } else {
        rgb_to_hsl(upper_rgb)
    };
    let lower_color = if color_mode == "HSV" {
        rgb_to_hsv(lower_rgb)
    } else {
        rgb_to_hsl(lower_rgb)
    };
    let hue = interpolate_hue(upper_color[0], lower_color[0], amount, hue_interpolation)?;
    let color = [
        hue,
        inverse * upper_color[1] + amount * lower_color[1],
        inverse * upper_color[2] + amount * lower_color[2],
    ];
    let rgb = if color_mode == "HSV" {
        hsv_to_rgb(color)
    } else {
        hsl_to_rgb(color)
    };
    Some([
        rgb[0],
        rgb[1],
        rgb[2],
        inverse * upper.1[3] + amount * lower.1[3],
    ])
}

pub(crate) fn shader_color_ramp(
    elements: &[(f32, [f32; 4])],
    table: &[[f32; 4]],
    factor: f32,
    interpolation: &str,
    color_mode: &str,
    semantics: MathSemantics,
) -> Option<[f32; 4]> {
    if elements.is_empty() || table.len() != COLOR_RAMP_TABLE_SIZE || factor.is_nan() {
        return None;
    }
    if semantics == MathSemantics::Eevee && elements.len() == 1 {
        return Some(elements[0].1);
    }
    if semantics == MathSemantics::Eevee && elements.len() == 2 && color_mode == "RGB" {
        let first = elements[0];
        let second = elements[1];
        match interpolation {
            "CONSTANT" => {
                return Some(if factor > first.0.max(second.0) {
                    second.1
                } else {
                    first.1
                });
            }
            "LINEAR" | "EASE" => {
                let scale = 1.0 / (second.0 - first.0);
                let bias = -scale * first.0;
                let mut amount = (factor * scale + bias).clamp(0.0, 1.0);
                if interpolation == "EASE" {
                    amount = amount * amount * (3.0 - 2.0 * amount);
                }
                return Some(std::array::from_fn(|channel| {
                    first.1[channel] * (1.0 - amount) + second.1[channel] * amount
                }));
            }
            _ => {}
        }
    }

    let factor = factor.clamp(0.0, 1.0);
    let scaled = if semantics == MathSemantics::Eevee && interpolation != "CONSTANT" {
        let sampler_offset = 0.5 / COLOR_RAMP_TABLE_SIZE as f32;
        let sampler_scale = 1.0 - (1.0 / COLOR_RAMP_TABLE_SIZE as f32);
        let coordinate = factor * sampler_scale + sampler_offset;
        coordinate * COLOR_RAMP_TABLE_SIZE as f32 - 0.5
    } else {
        factor * (COLOR_RAMP_TABLE_SIZE - 1) as f32
    };
    let scaled = scaled.clamp(0.0, (COLOR_RAMP_TABLE_SIZE - 1) as f32);
    let index = scaled.floor() as usize;
    let amount = scaled - index as f32;
    let lower = table[index];
    if amount == 0.0 || interpolation == "CONSTANT" {
        return Some(lower);
    }
    let upper = *table.get(index + 1)?;
    let sampled =
        std::array::from_fn(|channel| (1.0 - amount) * lower[channel] + amount * upper[channel]);
    if semantics == MathSemantics::Eevee {
        // EEVEE's SFLOAT16 ColorBand sampler returns half-precision filtered samples.
        Some(sampled.map(|value| exr::prelude::f16::from_f32(value).to_f32()))
    } else {
        Some(sampled)
    }
}

fn spline_weights(t: f32, cardinal: bool) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    if cardinal {
        let tension = 0.71;
        [
            -tension * t3 + 2.0 * tension * t2 - tension * t,
            (2.0 - tension) * t3 + (tension - 3.0) * t2 + 1.0,
            (tension - 2.0) * t3 + (3.0 - 2.0 * tension) * t2 + tension * t,
            tension * t3 - tension * t2,
        ]
    } else {
        [
            -0.166_666_66 * t3 + 0.5 * t2 - 0.5 * t + 0.166_666_66,
            0.5 * t3 - t2 + 0.666_666_7,
            -0.5 * t3 + 0.5 * t2 + 0.5 * t + 0.166_666_66,
            0.166_666_66 * t3,
        ]
    }
}

fn interpolate_hue(first: f32, second: f32, factor: f32, mode: &str) -> Option<f32> {
    let first = first.rem_euclid(1.0);
    let second = second.rem_euclid(1.0);
    let mut from = first;
    let mut to = second;
    match mode {
        "NEAR" if first < second && second - first > 0.5 => from += 1.0,
        "NEAR" if first > second && second - first < -0.5 => to += 1.0,
        "FAR" if equal_f32(first, second) => from += 1.0,
        "FAR" if first < second && second - first < 0.5 => from += 1.0,
        "FAR" if first > second && second - first > -0.5 => to += 1.0,
        "CCW" if first > second => to += 1.0,
        "CW" if first < second => from += 1.0,
        "NEAR" | "FAR" | "CW" | "CCW" => {}
        _ => return None,
    }
    Some((from * (1.0 - factor) + to * factor).rem_euclid(1.0))
}

fn rgb_to_hsl([red, green, blue]: [f32; 3]) -> [f32; 3] {
    let maximum = red.max(green).max(blue);
    let minimum = red.min(green).min(blue);
    let delta = maximum - minimum;
    let lightness = f32::midpoint(maximum, minimum).min(1.0);
    if equal_f32(maximum, minimum) {
        return [0.0, 0.0, lightness];
    }
    let saturation = if lightness > 0.5 {
        delta / (2.0 - maximum - minimum)
    } else {
        delta / (maximum + minimum)
    };
    let mut hue = if equal_f32(maximum, red) {
        (green - blue) / delta + if green < blue { 6.0 } else { 0.0 }
    } else if equal_f32(maximum, green) {
        (blue - red) / delta + 2.0
    } else {
        (red - green) / delta + 4.0
    };
    hue /= 6.0;
    [hue, saturation, lightness]
}

fn hsl_to_rgb([hue, saturation, lightness]: [f32; 3]) -> [f32; 3] {
    let red = ((hue * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0);
    let green = (2.0 - (hue * 6.0 - 2.0).abs()).clamp(0.0, 1.0);
    let blue = (2.0 - (hue * 6.0 - 4.0).abs()).clamp(0.0, 1.0);
    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    [
        (red - 0.5) * chroma + lightness,
        (green - 0.5) * chroma + lightness,
        (blue - 0.5) * chroma + lightness,
    ]
}

pub(crate) fn curve_mapping_from_value(value: &JsonValue) -> Option<CurveMapping> {
    CurveMapping::from_node_json(value).ok()
}

pub(crate) fn curve_mapping_prepare(mapping: &CurveMapping) -> PreparedCurve {
    mapping.prepare()
}

pub(crate) fn curve_mapping_value(
    mapping: &CurveMapping,
    prepared: &PreparedCurve,
    curve: usize,
    value: f32,
) -> Option<f32> {
    if curve >= 4 {
        return None;
    }
    let value = prepared.evaluate(curve, value);
    Some(if mapping.use_clip {
        value.clamp(mapping.clip[1] as f32, mapping.clip[3] as f32)
    } else {
        value
    })
}
pub(crate) fn shader_curve_mapping_value(
    prepared: &PreparedCurve,
    curve: usize,
    value: f32,
) -> Option<f32> {
    (curve < 4).then(|| prepared.evaluate(curve, value))
}

pub(crate) fn curve_mapping_vector(prepared: &PreparedCurve, value: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|channel| prepared.evaluate(channel, value[channel]))
}

pub(crate) fn curve_mapping_rgb(prepared: &PreparedCurve, value: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|channel| {
        let combined = prepared.evaluate(3, value[channel]);
        prepared.evaluate(channel, combined)
    })
}

pub(crate) fn integer_math(operation: &str, a: i32, b: i32, c: i32) -> Option<i32> {
    Some(match operation {
        "ADD" => a.wrapping_add(b),
        "SUBTRACT" => a.wrapping_sub(b),
        "MULTIPLY" => a.wrapping_mul(b),
        "DIVIDE" => int_divide(a, b),
        "DIVIDE_FLOOR" => int_floor_divide(a, b),
        "DIVIDE_CEIL" => int_ceil_divide(a, b),
        "DIVIDE_ROUND" => int_round_divide(a, b),
        "POWER" if b >= 0 => int_power(a, b),
        "MULTIPLY_ADD" => a.wrapping_mul(b).wrapping_add(c),
        "FLOORED_MODULO" => int_floored_modulo(a, b),
        "MODULO" => {
            if b == 0 || (a == i32::MIN && b == -1) {
                0
            } else {
                a % b
            }
        }
        "ABSOLUTE" => a.wrapping_abs(),
        "SIGN" => a.signum(),
        "MINIMUM" => a.min(b),
        "MAXIMUM" => a.max(b),
        "GCD" => int_gcd(a, b),
        "LCM" => int_lcm(a, b),
        "NEGATE" => a.wrapping_neg(),
        _ => return None,
    })
}

fn int_divide(a: i32, b: i32) -> i32 {
    if b == 0 {
        0
    } else if a == i32::MIN && b == -1 {
        i32::MIN
    } else {
        a / b
    }
}

fn int_floor_divide(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    let quotient = i64::from(a) / i64::from(b);
    let remainder = i64::from(a) % i64::from(b);
    let floor = if remainder != 0 && (remainder < 0) != (b < 0) {
        quotient - 1
    } else {
        quotient
    };
    floor as i32
}

fn int_ceil_divide(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    let quotient = i64::from(a) / i64::from(b);
    let remainder = i64::from(a) % i64::from(b);
    let ceil = if remainder != 0 && (remainder < 0) == (b < 0) {
        quotient + 1
    } else {
        quotient
    };
    ceil as i32
}

fn int_round_divide(a: i32, b: i32) -> i32 {
    let divisor = i64::from(b).abs();
    if divisor == 0 {
        return 0;
    }
    let numerator = i64::from(a).abs() * 2 + divisor;
    let rounded = numerator / (divisor * 2);
    let sign = if (a < 0) ^ (b < 0) { -1 } else { 1 };
    (rounded * sign) as i32
}

fn int_power(base: i32, exponent: i32) -> i32 {
    let mut exponent = exponent as u32;
    let mut base = base;
    let mut result = 1_i32;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result.wrapping_mul(base);
        }
        exponent >>= 1;
        base = base.wrapping_mul(base);
    }
    result
}

fn int_floored_modulo(a: i32, b: i32) -> i32 {
    if b == 0 {
        return 0;
    }
    let modulus = i64::from(b);
    let remainder = i64::from(a) % modulus;
    ((remainder + modulus) % modulus) as i32
}

fn int_gcd(a: i32, b: i32) -> i32 {
    let mut first = i64::from(a).unsigned_abs();
    let mut second = i64::from(b).unsigned_abs();
    while second != 0 {
        let remainder = first % second;
        first = second;
        second = remainder;
    }
    first as i32
}

fn int_lcm(a: i32, b: i32) -> i32 {
    if a == 0 || b == 0 {
        return 0;
    }
    let gcd = i64::from(int_gcd(a, b)).abs();
    ((i64::from(a).abs() / gcd) * i64::from(b).abs()) as i32
}

pub(crate) fn boolean_math(operation: &str, a: bool, b: bool, _c: bool) -> Option<bool> {
    Some(match operation {
        "AND" => a && b,
        "OR" => a || b,
        "NOT" => !a,
        "NAND" => !(a && b),
        "NOR" => !(a || b),
        "XNOR" => a == b,
        "XOR" => a != b,
        "IMPLY" => !a || b,
        "NIMPLY" => a && !b,
        _ => return None,
    })
}

pub(crate) fn compare(
    operation: &str,
    mode: &str,
    a: CompareValue<'_>,
    b: CompareValue<'_>,
    parameter: f32,
    epsilon: f32,
) -> Option<bool> {
    match (a, b) {
        (CompareValue::Scalar(a), CompareValue::Scalar(b)) => {
            compare_float(operation, a, b, epsilon)
        }
        (CompareValue::Integer(a), CompareValue::Integer(b)) => compare_integer(operation, a, b),
        (CompareValue::Vector(a), CompareValue::Vector(b)) => {
            compare_vector(operation, mode, a, b, parameter, epsilon)
        }
        (CompareValue::Color(a), CompareValue::Color(b)) => compare_color(operation, a, b, epsilon),
        (CompareValue::Boolean(a), CompareValue::Boolean(b)) => match operation {
            "EQUAL" => Some(a == b),
            "NOT_EQUAL" => Some(a != b),
            _ => None,
        },
        (CompareValue::String(a), CompareValue::String(b)) => match operation {
            "EQUAL" => Some(a == b),
            "NOT_EQUAL" => Some(a != b),
            _ => None,
        },
        _ => None,
    }
}

fn compare_float(operation: &str, a: f32, b: f32, epsilon: f32) -> Option<bool> {
    Some(match operation {
        "LESS_THAN" => a < b,
        "LESS_EQUAL" => a <= b,
        "GREATER_THAN" => a > b,
        "GREATER_EQUAL" => a >= b,
        "EQUAL" => (a - b).abs() <= epsilon,
        "NOT_EQUAL" => (a - b).abs() > epsilon,
        _ => return None,
    })
}

fn compare_integer(operation: &str, a: i32, b: i32) -> Option<bool> {
    Some(match operation {
        "LESS_THAN" => a < b,
        "LESS_EQUAL" => a <= b,
        "GREATER_THAN" => a > b,
        "GREATER_EQUAL" => a >= b,
        "EQUAL" => a == b,
        "NOT_EQUAL" => a != b,
        _ => return None,
    })
}

fn compare_vector(
    operation: &str,
    mode: &str,
    a: [f32; 3],
    b: [f32; 3],
    parameter: f32,
    epsilon: f32,
) -> Option<bool> {
    if mode == "ELEMENT" {
        return Some(match operation {
            "LESS_THAN" => a[0] < b[0] && a[1] < b[1] && a[2] < b[2],
            "LESS_EQUAL" => a[0] <= b[0] && a[1] <= b[1] && a[2] <= b[2],
            "GREATER_THAN" => a[0] > b[0] && a[1] > b[1] && a[2] > b[2],
            "GREATER_EQUAL" => a[0] >= b[0] && a[1] >= b[1] && a[2] >= b[2],
            "EQUAL" => {
                (a[0] - b[0]).abs() <= epsilon
                    && (a[1] - b[1]).abs() <= epsilon
                    && (a[2] - b[2]).abs() <= epsilon
            }
            "NOT_EQUAL" => {
                (a[0] - b[0]).abs() > epsilon
                    || (a[1] - b[1]).abs() > epsilon
                    || (a[2] - b[2]).abs() > epsilon
            }
            _ => return None,
        });
    }
    let (a, b) = match mode {
        "AVERAGE" => ((a[0] + a[1] + a[2]) / 3.0, (b[0] + b[1] + b[2]) / 3.0),
        "DOT_PRODUCT" => (dot3(a, b), parameter),
        "DIRECTION" => (vector_angle(a, b), parameter),
        "LENGTH" => (length3(a), length3(b)),
        _ => return None,
    };
    if operation == "NOT_EQUAL" && mode == "DOT_PRODUCT" {
        return Some((a - b).abs() >= epsilon);
    }
    compare_float(operation, a, b, epsilon)
}

fn vector_angle(a: [f32; 3], b: [f32; 3]) -> f32 {
    let a = normalize3(a);
    let b = normalize3(b);
    if dot3(a, b) >= 0.0 {
        2.0 * (length3(sub3(a, b)) / 2.0).clamp(-1.0, 1.0).asin()
    } else {
        std::f32::consts::PI - 2.0 * (length3(add3(a, b)) / 2.0).clamp(-1.0, 1.0).asin()
    }
}

fn compare_color(operation: &str, a: [f32; 4], b: [f32; 4], epsilon: f32) -> Option<bool> {
    Some(match operation {
        "EQUAL" => {
            (a[0] - b[0]).abs() <= epsilon
                && (a[1] - b[1]).abs() <= epsilon
                && (a[2] - b[2]).abs() <= epsilon
        }
        "NOT_EQUAL" => {
            (a[0] - b[0]).abs() > epsilon
                || (a[1] - b[1]).abs() > epsilon
                || (a[2] - b[2]).abs() > epsilon
        }
        "BRIGHTER" | "COLOR_BRIGHTER" => luminance(a) > luminance(b),
        "DARKER" | "COLOR_DARKER" => luminance(a) < luminance(b),
        _ => return None,
    })
}

fn luminance(color: [f32; 4]) -> f32 {
    color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722
}
