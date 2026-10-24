// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender 5.2.2 (https://www.blender.org):
//   source/blender/blenkernel/intern/fmodifier.cc, GPL-2.0-or-later.
//   source/blender/blenlib/intern/noise.cc, GPL-2.0-or-later AND BSD-3-Clause.
//   source/blender/blenlib/intern/noise_c.cc, GPL-2.0-or-later.

use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{FCurve, FCurveModifier},
};

/// Applies Blender's time-modifier pass in reverse modifier order.
///
/// # Errors
///
/// Returns a scene error for malformed or non-finite modifier data or times.
pub(crate) fn evaluate_time(curve: &FCurve, frame: f32) -> Result<(f32, Vec<f32>)> {
    if !frame.is_finite() {
        return Err(invalid_modifier(
            curve,
            "modifier evaluation time is non-finite",
        ));
    }
    let mut evaltime = frame;
    let mut cycle_offsets = Vec::new();
    for (index, modifier) in curve.modifiers.iter().enumerate().rev() {
        let is_cycle = modifier.modifier_type == "CYCLES";
        if !is_supported(modifier)
            || muted(modifier)
            || (is_cycle && index > 0)
            || !in_restricted_range(modifier, evaltime)
        {
            if is_cycle {
                cycle_offsets.push(0.0);
            }
            continue;
        }
        let Some((new_time, offset)) = evaluate_time_modifier(curve, modifier, evaltime)? else {
            if is_cycle {
                cycle_offsets.push(0.0);
            }
            continue;
        };
        evaltime = blend(new_time, evaltime, influence(modifier, evaltime));
        if is_cycle {
            cycle_offsets.push(offset);
        }
    }
    if !evaltime.is_finite() || cycle_offsets.iter().any(|offset| !offset.is_finite()) {
        return Err(invalid_modifier(
            curve,
            "time modifier produced a non-finite value",
        ));
    }
    Ok((evaltime, cycle_offsets))
}

/// Applies Blender's value-modifier pass in modifier stack order.
///
/// # Errors
///
/// Returns a scene error for malformed or non-finite modifier data or results.
pub(crate) fn evaluate_value(
    curve: &FCurve,
    evaltime: f32,
    value: f32,
    cycle_offsets: &[f32],
) -> Result<f32> {
    let mut result = value;
    let mut cycle_offsets = cycle_offsets.iter().rev();
    for (index, modifier) in curve.modifiers.iter().enumerate() {
        let cycle_offset = if modifier.modifier_type == "CYCLES" {
            cycle_offsets.next().copied().unwrap_or(0.0)
        } else {
            0.0
        };
        if !is_supported(modifier)
            || muted(modifier)
            || (modifier.modifier_type == "CYCLES" && index > 0)
            || !in_restricted_range(modifier, evaltime)
        {
            continue;
        }
        let Some(new_value) =
            evaluate_value_modifier(curve, modifier, result, evaltime, cycle_offset)?
        else {
            continue;
        };
        result = blend(new_value, result, influence(modifier, evaltime));
    }
    if !result.is_finite() {
        return Err(invalid_modifier(
            curve,
            "modifier evaluation produced a non-finite value",
        ));
    }
    Ok(result)
}

fn is_supported(modifier: &FCurveModifier) -> bool {
    modifier.is_supported()
}

fn property<'a>(modifier: &'a FCurveModifier, name: &str) -> Option<&'a Value> {
    modifier.properties.get(name)
}

fn number(modifier: &FCurveModifier, name: &str, default: f32) -> f32 {
    property(modifier, name)
        .and_then(Value::as_f64)
        .map_or(default, |value| value as f32)
}

fn integer(modifier: &FCurveModifier, name: &str, default: i64) -> i64 {
    property(modifier, name)
        .and_then(Value::as_i64)
        .unwrap_or(default)
}

fn boolean(modifier: &FCurveModifier, name: &str, default: bool) -> bool {
    property(modifier, name)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

fn text<'a>(modifier: &'a FCurveModifier, name: &str, default: &'a str) -> &'a str {
    property(modifier, name)
        .and_then(Value::as_str)
        .unwrap_or(default)
}

fn muted(modifier: &FCurveModifier) -> bool {
    boolean(modifier, "mute", false)
}

fn in_restricted_range(modifier: &FCurveModifier, time: f32) -> bool {
    !boolean(modifier, "use_restricted_range", false)
        || (time >= number(modifier, "frame_start", 0.0)
            && time <= number(modifier, "frame_end", 0.0))
}

fn influence(modifier: &FCurveModifier, time: f32) -> f32 {
    let mut amount = if boolean(modifier, "use_influence", false) {
        number(modifier, "influence", 1.0)
    } else {
        1.0
    };
    if boolean(modifier, "use_restricted_range", false) {
        let start = number(modifier, "frame_start", 0.0);
        let end = number(modifier, "frame_end", 0.0);
        let blend_in = number(modifier, "blend_in", 0.0);
        let blend_out = number(modifier, "blend_out", 0.0);
        if time < start || time > end {
            return 0.0;
        }
        if blend_in != 0.0 && time >= start && time <= start + blend_in {
            amount *= (time - start) / blend_in;
            return amount;
        }
        if blend_out != 0.0 && time <= end && time >= end - blend_out {
            let a = end;
            let b = end - blend_out;
            amount *= (time - a) / (b - a);
        }
    }
    amount
}

fn blend(target: f32, origin: f32, amount: f32) -> f32 {
    amount * target + (1.0 - amount) * origin
}

fn evaluate_time_modifier(
    curve: &FCurve,
    modifier: &FCurveModifier,
    time: f32,
) -> Result<Option<(f32, f32)>> {
    let (result, offset) = match modifier.modifier_type.as_str() {
        "CYCLES" => {
            let Some((time, offset)) = cycles_time(curve, modifier, time) else {
                return Ok(None);
            };
            (time, offset)
        }
        "LIMITS" => {
            let result = if boolean(modifier, "use_min_x", false)
                && time < number(modifier, "min_x", 0.0)
            {
                number(modifier, "min_x", 0.0)
            } else if boolean(modifier, "use_max_x", false) && time > number(modifier, "max_x", 0.0)
            {
                number(modifier, "max_x", 0.0)
            } else {
                time
            };
            (result, 0.0)
        }
        "STEPPED" => {
            let start = number(modifier, "frame_start", 0.0);
            let end = number(modifier, "frame_end", 0.0);
            if (boolean(modifier, "use_frame_start", false) && time < start)
                || (boolean(modifier, "use_frame_end", false) && time > end)
            {
                (time, 0.0)
            } else {
                let step = number(modifier, "frame_step", 2.0);
                let offset = number(modifier, "frame_offset", 0.0);
                if !step.is_finite() || step == 0.0 {
                    return Err(invalid_modifier(
                        curve,
                        "stepped frame step must be nonzero",
                    ));
                }
                (
                    (((time - offset) / step) as i32) as f32 * step + offset,
                    0.0,
                )
            }
        }
        _ => return Ok(None),
    };
    if result.is_finite() && offset.is_finite() {
        Ok(Some((result, offset)))
    } else {
        Err(invalid_modifier(
            curve,
            "time modifier produced a non-finite value",
        ))
    }
}

fn cycles_time(curve: &FCurve, modifier: &FCurveModifier, time: f32) -> Option<(f32, f32)> {
    let first = curve.keyframes.first()?;
    let last = curve.keyframes.last()?;
    if time >= first.frame as f32 && time <= last.frame as f32 {
        return None;
    }
    let before = time < first.frame as f32;
    let mode = text(
        modifier,
        if before { "mode_before" } else { "mode_after" },
        "REPEAT",
    );
    if mode == "NONE" {
        return None;
    }
    let cycles = integer(
        modifier,
        if before {
            "cycles_before"
        } else {
            "cycles_after"
        },
        0,
    );
    let first_time = first.frame as f32;
    let last_time = last.frame as f32;
    let first_value = first.value as f32;
    let last_value = last.value as f32;
    let period = last_time - first_time;
    if period == 0.0 {
        return Some((time, 0.0));
    }
    let side = if before { -1.0_f64 } else { 1.0_f64 };
    let origin = if before { first_time } else { last_time };
    let cycle = side * (f64::from(time) - f64::from(origin)) / f64::from(period);
    if cycles != 0 && cycle > cycles as f64 {
        return Some((time, 0.0));
    }
    let cycle_time = (time - origin) % period;
    let mut offset = 0.0_f32;
    if mode == "REPEAT_OFFSET" {
        offset = if before {
            ((time - origin) / period).floor()
        } else {
            ((time - origin) / period).ceil()
        } * (last_value - first_value);
    }
    let mapped = if cycle_time == 0.0 {
        if mode == "MIRROR" && (cycle as i32) % 2 != 0 {
            if before { last_time } else { first_time }
        } else if before {
            first_time
        } else {
            last_time
        }
    } else if mode == "MIRROR" && ((cycle + 1.0) as i32) % 2 != 0 {
        if before {
            first_time - cycle_time
        } else {
            last_time - cycle_time
        }
    } else {
        first_time + cycle_time
    };
    Some((
        if mapped < first_time {
            mapped + period
        } else {
            mapped
        },
        offset,
    ))
}

fn evaluate_value_modifier(
    curve: &FCurve,
    modifier: &FCurveModifier,
    value: f32,
    time: f32,
    cycle_offset: f32,
) -> Result<Option<f32>> {
    let result = match modifier.modifier_type.as_str() {
        "GENERATOR" => generator(curve, modifier, value, time)?,
        "FNGENERATOR" => function_generator(curve, modifier, value, time)?,
        "ENVELOPE" => envelope(curve, modifier, value, time)?,
        "CYCLES" => value + cycle_offset,
        "NOISE" => noise(curve, modifier, value, time)?,
        "LIMITS" => {
            let mut result = value;
            if boolean(modifier, "use_min_y", false) && result < number(modifier, "min_y", 0.0) {
                result = number(modifier, "min_y", 0.0);
            }
            if boolean(modifier, "use_max_y", false) && result > number(modifier, "max_y", 0.0) {
                result = number(modifier, "max_y", 0.0);
            }
            result
        }
        _ => return Ok(None),
    };
    if result.is_finite() {
        Ok(Some(result))
    } else {
        Err(invalid_modifier(
            curve,
            "value modifier produced a non-finite value",
        ))
    }
}

fn generator(curve: &FCurve, modifier: &FCurveModifier, value: f32, time: f32) -> Result<f32> {
    let order = integer(modifier, "poly_order", 1);
    if order <= 0 {
        return Ok(value);
    }
    let coefficients = property(modifier, "coefficients").and_then(Value::as_array);
    let default_coefficients = [0.0_f32, 1.0_f32];
    let mut result = match text(modifier, "mode", "POLYNOMIAL") {
        "POLYNOMIAL" => {
            let mut power = 1.0_f32;
            let mut sum = 0.0_f32;
            for index in 0..=order as usize {
                let coefficient = coefficients
                    .and_then(|items| items.get(index))
                    .and_then(Value::as_f64)
                    .map_or_else(
                        || default_coefficients.get(index).copied().unwrap_or(0.0),
                        |entry| entry as f32,
                    );
                sum += coefficient * power;
                power *= time;
            }
            sum
        }
        "POLYNOMIAL_FACTORISED" => {
            let mut product = 1.0_f32;
            for index in 0..order as usize {
                let slope = coefficients
                    .and_then(|items| items.get(index * 2))
                    .and_then(Value::as_f64)
                    .map_or(0.0, |entry| entry as f32);
                let intercept = coefficients
                    .and_then(|items| items.get(index * 2 + 1))
                    .and_then(Value::as_f64)
                    .map_or(if index == 0 { 1.0 } else { 0.0 }, |entry| entry as f32);
                product *= slope * time + intercept;
            }
            product
        }
        _ => return Err(invalid_modifier(curve, "generator mode is invalid")),
    };
    if boolean(modifier, "use_additive", false) {
        result += value;
    }
    Ok(result)
}

fn function_generator(
    curve: &FCurve,
    modifier: &FCurveModifier,
    value: f32,
    time: f32,
) -> Result<f32> {
    let additive = boolean(modifier, "use_additive", false);
    let argument = f64::from(
        number(modifier, "phase_multiplier", 1.0) * time + number(modifier, "phase_offset", 0.0),
    );
    let function = text(modifier, "function_type", "SIN");
    let generated = match function {
        "SIN" => Some(argument.sin()),
        "COS" => Some(argument.cos()),
        "SINC" => Some(if argument.abs() < 0.0001 {
            1.0
        } else {
            (std::f64::consts::PI * argument).sin() / (std::f64::consts::PI * argument)
        }),
        "TAN" => {
            if ((argument - std::f64::consts::FRAC_PI_2) % std::f64::consts::PI).abs()
                < f64::from(f32::EPSILON)
            {
                None
            } else {
                Some(argument.tan())
            }
        }
        "LN" if argument > 1.0 => Some(argument.ln()),
        "SQRT" if argument > 0.0 => Some(argument.sqrt()),
        "LN" | "SQRT" => None,
        _ => {
            return Err(invalid_modifier(
                curve,
                "function-generator type is invalid",
            ));
        }
    };
    let Some(generated) = generated else {
        return Ok(if additive { value } else { 0.0 });
    };
    let output = (number(modifier, "amplitude", 1.0) * generated as f32)
        + number(modifier, "value_offset", 0.0);
    Ok(if additive { value + output } else { output })
}

fn envelope(curve: &FCurve, modifier: &FCurveModifier, value: f32, time: f32) -> Result<f32> {
    let Some(points) = property(modifier, "control_points").and_then(Value::as_array) else {
        return Ok(value);
    };
    let Some(first) = points.first() else {
        return Ok(value);
    };
    let last = points.last().unwrap_or(first);
    let mut minimum = point_value(first, "min", -1.0);
    let mut maximum = point_value(first, "max", 1.0);
    if time >= point_value(last, "frame", 0.0) {
        minimum = point_value(last, "min", -1.0);
        maximum = point_value(last, "max", 1.0);
    } else if time > point_value(first, "frame", 0.0) {
        for pair in points.windows(2) {
            let left = &pair[0];
            let right = &pair[1];
            let left_time = point_value(left, "frame", 0.0);
            let right_time = point_value(right, "frame", 0.0);
            if left_time <= time && right_time >= time {
                let difference = right_time - left_time;
                let factor = (time - left_time) / difference;
                let inverse = (right_time - time) / difference;
                minimum = inverse * point_value(left, "min", -1.0)
                    + factor * point_value(right, "min", -1.0);
                maximum = inverse * point_value(left, "max", 1.0)
                    + factor * point_value(right, "max", 1.0);
                break;
            }
        }
    }
    let default_minimum = number(modifier, "default_min", -1.0);
    let default_maximum = number(modifier, "default_max", 1.0);
    let reference = number(modifier, "reference_value", 0.0);
    let factor = (value - (reference + default_minimum)) / (default_maximum - default_minimum);
    let result = minimum + factor * (maximum - minimum);
    if result.is_finite() {
        Ok(result)
    } else {
        Err(invalid_modifier(
            curve,
            "envelope evaluation produced a non-finite value",
        ))
    }
}

fn point_value(point: &Value, name: &str, default: f32) -> f32 {
    point
        .get(name)
        .and_then(Value::as_f64)
        .map_or(default, |value| value as f32)
}

fn noise(curve: &FCurve, modifier: &FCurveModifier, value: f32, time: f32) -> Result<f32> {
    let scale = number(modifier, "scale", 1.0);
    let offset = number(modifier, "offset", 0.0);
    let phase = number(modifier, "phase", 1.0);
    let depth = integer(modifier, "depth", 0);
    if depth < 0 {
        return Err(invalid_modifier(curve, "noise depth must be non-negative"));
    }
    let noise = if boolean(modifier, "use_legacy_noise", false) {
        legacy_turbulence(scale, time - offset, phase, 0.1, depth)
    } else {
        let scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        perlin_fbm(
            (time - offset) * scale + 0.618_034,
            phase,
            depth,
            number(modifier, "roughness", 0.5),
            number(modifier, "lacunarity", 2.0),
        )
    };
    let strength = number(modifier, "strength", 1.0);
    Ok(match text(modifier, "blend_type", "REPLACE") {
        "ADD" => value + noise * strength,
        "SUBTRACT" => value - noise * strength,
        "MULTIPLY" => value * noise * strength,
        "REPLACE" => value + (noise - 0.5) * strength,
        _ => return Err(invalid_modifier(curve, "noise blend type is invalid")),
    })
}

fn legacy_turbulence(size: f32, x: f32, y: f32, z: f32, depth: i64) -> f32 {
    let mut sum = crate::geom::legacy_texture::blender_legacy_noise_hnoise(size, x, y, z);
    let mut amplitude = 0.5_f32;
    let mut divisor = 1.0_f32;
    for _ in 0..depth {
        sum += amplitude
            * crate::geom::legacy_texture::blender_legacy_noise_hnoise(size * amplitude, x, y, z);
        divisor += amplitude;
        amplitude *= 0.5;
    }
    sum / divisor
}

fn perlin_fbm(x: f32, y: f32, detail: i64, roughness: f32, lacunarity: f32) -> f32 {
    let mut frequency = 1.0_f32;
    let mut amplitude = 1.0_f32;
    let mut max_amplitude = 0.0_f32;
    let mut sum = 0.0_f32;
    for _ in 0..=detail {
        let value = perlin_signed(frequency * x, frequency * y);
        sum += value * amplitude;
        max_amplitude += amplitude;
        amplitude *= roughness;
        frequency *= lacunarity;
    }
    0.5 * sum / max_amplitude + 0.5
}

fn perlin_signed(x: f32, y: f32) -> f32 {
    let x = x % 100_000.0 + if x.abs() >= 1_000_000.0 { 0.5 } else { 0.0 };
    let y = y % 100_000.0 + if y.abs() >= 1_000_000.0 { 0.5 } else { 0.0 };
    0.6616 * perlin_noise(x, y)
}

fn perlin_noise(x: f32, y: f32) -> f32 {
    let floor_x = x.floor();
    let floor_y = y.floor();
    let lattice_x = floor_x as i32;
    let lattice_y = floor_y as i32;
    let fraction_x = x - floor_x;
    let fraction_y = y - floor_y;
    let fade_x = fade(fraction_x);
    let fade_y = fade(fraction_y);
    let lower_left = gradient(hash2(lattice_x, lattice_y), fraction_x, fraction_y);
    let lower_right = gradient(
        hash2(lattice_x + 1, lattice_y),
        fraction_x - 1.0,
        fraction_y,
    );
    let upper_left = gradient(
        hash2(lattice_x, lattice_y + 1),
        fraction_x,
        fraction_y - 1.0,
    );
    let upper_right = gradient(
        hash2(lattice_x + 1, lattice_y + 1),
        fraction_x - 1.0,
        fraction_y - 1.0,
    );
    let inverse_x = 1.0 - fade_x;
    (1.0 - fade_y) * (lower_left * inverse_x + lower_right * fade_x)
        + fade_y * (upper_left * inverse_x + upper_right * fade_x)
}

fn fade(value: f32) -> f32 {
    value * value * value * (value * (value * 6.0 - 15.0) + 10.0)
}

fn gradient(hash: u32, x: f32, y: f32) -> f32 {
    let hash = hash & 7;
    let u = if hash < 4 { x } else { y };
    let v = 2.0 * if hash < 4 { y } else { x };
    (if hash & 1 != 0 { -u } else { u }) + if hash & 2 != 0 { -v } else { v }
}

fn hash2(x: i32, y: i32) -> u32 {
    let initial = 0xdead_beef_u32.wrapping_add((2 << 2) + 13);
    let (mut a, mut b, mut c) = (initial, initial, initial);
    a = a.wrapping_add(x as u32);
    b = b.wrapping_add(y as u32);
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(14));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(11));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(25));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(16));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(4));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(14));
    c ^= b;
    c.wrapping_sub(b.rotate_left(24))
}

fn invalid_modifier(curve: &FCurve, message: &str) -> PotError {
    PotError::with_details(
        ErrorCode::SceneInvalid,
        message,
        json!({"path":curve.path,"index":curve.index}),
    )
}
