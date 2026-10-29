// Blender 5.2.2 BLI_noise.cc kernels, kept in f32 to preserve shader results.
#[derive(Clone, Copy)]
pub(super) enum NoiseType {
    Multifractal,
    Fbm,
    HybridMultifractal,
    RidgedMultifractal,
    HeteroTerrain,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum VoronoiFeature {
    F1,
    F2,
    SmoothF1,
    DistanceToEdge,
    NSphereRadius,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum VoronoiMetric {
    Euclidean,
    Manhattan,
    Chebychev,
    Minkowski,
}

#[derive(Clone, Copy)]
pub(super) struct VoronoiParams {
    pub(super) feature: VoronoiFeature,
    pub(super) metric: VoronoiMetric,
    pub(super) scale: f32,
    pub(super) detail: f32,
    pub(super) roughness: f32,
    pub(super) lacunarity: f32,
    pub(super) smoothness: f32,
    pub(super) exponent: f32,
    pub(super) randomness: f32,
    pub(super) normalize: bool,
    pub(super) max_distance: f32,
}

#[derive(Clone, Copy, Default)]
pub(super) struct VoronoiOutput {
    pub(super) distance: f32,
    pub(super) color: [f32; 3],
    pub(super) position: [f32; 4],
    pub(super) radius: f32,
}

pub(super) fn noise_factor(
    position: [f32; 4],
    dimensions: usize,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    offset: f32,
    gain: f32,
    distortion: f32,
    noise_type: NoiseType,
    normalize: bool,
) -> f32 {
    let position = distorted_position(position, dimensions, distortion);
    perlin_select(
        position, dimensions, detail, roughness, lacunarity, offset, gain, noise_type, normalize,
    )
}

pub(super) fn noise_color(
    position: [f32; 4],
    dimensions: usize,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    offset: f32,
    gain: f32,
    distortion: f32,
    noise_type: NoiseType,
    normalize: bool,
) -> [f32; 3] {
    let position = distorted_position(position, dimensions, distortion);
    let mut color = [0.0; 3];
    color[0] = perlin_select(
        position, dimensions, detail, roughness, lacunarity, offset, gain, noise_type, normalize,
    );
    let seed_start = (dimensions - 1) as f32;
    for (channel, color_channel) in color.iter_mut().enumerate().skip(1) {
        let offset_position = random_offset(seed_start + channel as f32, dimensions);
        let mut seeded_position = position;
        for axis in 0..dimensions {
            seeded_position[axis] += offset_position[axis];
        }
        *color_channel = perlin_select(
            seeded_position,
            dimensions,
            detail,
            roughness,
            lacunarity,
            offset,
            gain,
            noise_type,
            normalize,
        );
    }
    color
}

fn perlin_select(
    position: [f32; 4],
    dimensions: usize,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    offset: f32,
    gain: f32,
    noise_type: NoiseType,
    normalize: bool,
) -> f32 {
    match noise_type {
        NoiseType::Multifractal => {
            let mut value = 1.0;
            let mut power = 1.0;
            let mut point = position;
            for _ in 0..=detail as i32 {
                value *= power * perlin_signed(point, dimensions) + 1.0;
                power *= roughness;
                scale_point(&mut point, dimensions, lacunarity);
            }
            let remainder = detail - detail.floor();
            if remainder != 0.0 {
                value *= remainder * power * perlin_signed(point, dimensions) + 1.0;
            }
            value
        }
        NoiseType::Fbm => perlin_fbm(
            position, dimensions, detail, roughness, lacunarity, normalize,
        ),
        NoiseType::HybridMultifractal => {
            let mut power = 1.0;
            let mut value = 0.0;
            let mut weight: f32 = 1.0;
            let mut point = position;
            for _ in 0..=detail as i32 {
                if weight.is_nan() || weight <= 0.001 {
                    break;
                }
                weight = weight.min(1.0);
                let signal = (perlin_signed(point, dimensions) + offset) * power;
                power *= roughness;
                value += weight * signal;
                weight *= gain * signal;
                scale_point(&mut point, dimensions, lacunarity);
            }
            let remainder = detail - detail.floor();
            if remainder != 0.0 && weight > 0.001 {
                weight = weight.min(1.0);
                let signal = (perlin_signed(point, dimensions) + offset) * power;
                value += remainder * weight * signal;
            }
            value
        }
        NoiseType::RidgedMultifractal => {
            let mut power = roughness;
            let mut signal = offset - perlin_signed(position, dimensions).abs();
            signal *= signal;
            let mut value = signal;
            let mut weight;
            let mut point = position;
            for _ in 1..=detail as i32 {
                scale_point(&mut point, dimensions, lacunarity);
                weight = (signal * gain).clamp(0.0, 1.0);
                signal = offset - perlin_signed(point, dimensions).abs();
                signal *= signal;
                signal *= weight;
                value += signal * power;
                power *= roughness;
            }
            value
        }
        NoiseType::HeteroTerrain => {
            let mut power = roughness;
            let mut value = offset + perlin_signed(position, dimensions);
            let mut point = position;
            scale_point(&mut point, dimensions, lacunarity);
            for _ in 1..=detail as i32 {
                let increment = (perlin_signed(point, dimensions) + offset) * power * value;
                value += increment;
                power *= roughness;
                scale_point(&mut point, dimensions, lacunarity);
            }
            let remainder = detail - detail.floor();
            if remainder != 0.0 {
                let increment = (perlin_signed(point, dimensions) + offset) * power * value;
                value += remainder * increment;
            }
            value
        }
    }
}

fn perlin_fbm(
    position: [f32; 4],
    dimensions: usize,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    normalize: bool,
) -> f32 {
    let mut frequency = 1.0;
    let mut amplitude = 1.0;
    let mut max_amplitude = 0.0;
    let mut sum = 0.0;
    for _ in 0..=detail as i32 {
        let value = perlin_signed(
            scale_point_copy(position, dimensions, frequency),
            dimensions,
        );
        sum += value * amplitude;
        max_amplitude += amplitude;
        amplitude *= roughness;
        frequency *= lacunarity;
    }
    let remainder = detail - detail.floor();
    if remainder != 0.0 {
        let value = perlin_signed(
            scale_point_copy(position, dimensions, frequency),
            dimensions,
        );
        let partial_sum = sum + value * amplitude;
        if normalize {
            let full = 0.5 * sum / max_amplitude + 0.5;
            let partial = 0.5 * partial_sum / (max_amplitude + amplitude) + 0.5;
            mix(full, partial, remainder)
        } else {
            mix(sum, partial_sum, remainder)
        }
    } else if normalize {
        0.5 * sum / max_amplitude + 0.5
    } else {
        sum
    }
}

fn distorted_position(position: [f32; 4], dimensions: usize, strength: f32) -> [f32; 4] {
    let mut result = position;
    for (channel, result_channel) in result.iter_mut().enumerate().take(dimensions) {
        let seed = random_offset(channel as f32, dimensions);
        let mut sample = position;
        for axis in 0..dimensions {
            sample[axis] += seed[axis];
        }
        *result_channel += perlin_signed(sample, dimensions) * strength;
    }
    result
}

fn random_offset(seed: f32, dimensions: usize) -> [f32; 4] {
    let mut offset = [0.0; 4];
    match dimensions {
        1 => offset[0] = 100.0 + hash_float_to_float(seed) * 100.0,
        2 => {
            for (axis, value) in offset.iter_mut().take(2).enumerate() {
                *value = 100.0 + hash_float_to_float2(seed, axis as f32) * 100.0;
            }
        }
        3 => {
            for (axis, value) in offset.iter_mut().take(3).enumerate() {
                *value = 100.0 + hash_float_to_float2(seed, axis as f32) * 100.0;
            }
        }
        4 => {
            for (axis, value) in offset.iter_mut().enumerate() {
                *value = 100.0 + hash_float_to_float2(seed, axis as f32) * 100.0;
            }
        }
        _ => {}
    }
    offset
}

fn scale_point(point: &mut [f32; 4], dimensions: usize, scale: f32) {
    for value in point.iter_mut().take(dimensions) {
        *value *= scale;
    }
}

fn scale_point_copy(mut point: [f32; 4], dimensions: usize, scale: f32) -> [f32; 4] {
    scale_point(&mut point, dimensions, scale);
    point
}

fn perlin_signed(mut position: [f32; 4], dimensions: usize) -> f32 {
    for value in position.iter_mut().take(dimensions) {
        let correction = if value.abs() >= 1_000_000.0 { 0.5 } else { 0.0 };
        *value = *value % 100_000.0 + correction;
    }
    let value = perlin_noise(position, dimensions);
    value
        * match dimensions {
            1 => 0.25,
            2 => 0.6616,
            3 => 0.982,
            4 => 0.8344,
            _ => 0.0,
        }
}

fn perlin_noise(position: [f32; 4], dimensions: usize) -> f32 {
    let mut base = [0_i32; 4];
    let mut fraction = [0.0; 4];
    let mut blend = [0.0; 4];
    for axis in 0..dimensions {
        let floor = position[axis].floor();
        base[axis] = floor as i32;
        fraction[axis] = position[axis] - floor;
        blend[axis] = fade(fraction[axis]);
    }

    let mut values = [0.0; 16];
    let count = 1_usize << dimensions;
    for (corner, value) in values.iter_mut().enumerate().take(count) {
        let mut lattice = [0_i32; 4];
        let mut delta = [0.0; 4];
        for axis in 0..dimensions {
            let bit = ((corner >> axis) & 1) as i32;
            lattice[axis] = base[axis].wrapping_add(bit);
            delta[axis] = fraction[axis] - bit as f32;
        }
        *value = noise_gradient(hash_lattice(lattice, dimensions), delta, dimensions);
    }
    let mut count = count;
    for factor in blend.iter().take(dimensions) {
        for index in 0..count / 2 {
            values[index] = mix(values[index * 2], values[index * 2 + 1], *factor);
        }
        count /= 2;
    }
    values[0]
}

fn noise_gradient(hash: u32, delta: [f32; 4], dimensions: usize) -> f32 {
    match dimensions {
        1 => {
            let h = hash & 15;
            let gradient = (1 + (h & 7)) as f32;
            if h & 8 != 0 {
                -gradient * delta[0]
            } else {
                gradient * delta[0]
            }
        }
        2 => {
            let h = hash & 7;
            let u = if h < 4 { delta[0] } else { delta[1] };
            let v = 2.0 * if h < 4 { delta[1] } else { delta[0] };
            negate_if(u, h & 1 != 0) + negate_if(v, h & 2 != 0)
        }
        3 => {
            let h = hash & 15;
            let u = if h < 8 { delta[0] } else { delta[1] };
            let vt = if h == 12 || h == 14 {
                delta[0]
            } else {
                delta[2]
            };
            let v = if h < 4 { delta[1] } else { vt };
            negate_if(u, h & 1 != 0) + negate_if(v, h & 2 != 0)
        }
        4 => {
            let h = hash & 31;
            let u = if h < 24 { delta[0] } else { delta[1] };
            let v = if h < 16 { delta[1] } else { delta[2] };
            let s = if h < 8 { delta[2] } else { delta[3] };
            negate_if(u, h & 1 != 0) + negate_if(v, h & 2 != 0) + negate_if(s, h & 4 != 0)
        }
        _ => 0.0,
    }
}

fn negate_if(value: f32, condition: bool) -> f32 {
    if condition { -value } else { value }
}

fn fade(value: f32) -> f32 {
    value * value * value * (value * (value * 6.0 - 15.0) + 10.0)
}

fn mix(first: f32, second: f32, factor: f32) -> f32 {
    (1.0 - factor) * first + factor * second
}

// Match Cycles' `mix(a, b, t)` operation order without fusing its multiply and add.
fn voronoi_mix(first: f32, second: f32, factor: f32) -> f32 {
    first + factor * (second - first)
}

fn hash_lattice(lattice: [i32; 4], dimensions: usize) -> u32 {
    match dimensions {
        1 => hash_uint(lattice[0] as u32),
        2 => hash_uint2(lattice[0] as u32, lattice[1] as u32),
        3 => hash_uint3(lattice[0] as u32, lattice[1] as u32, lattice[2] as u32),
        4 => hash_uint4(
            lattice[0] as u32,
            lattice[1] as u32,
            lattice[2] as u32,
            lattice[3] as u32,
        ),
        _ => 0,
    }
}

fn hash_float_to_float(value: f32) -> f32 {
    uint_to_float(hash_uint(value.to_bits()))
}

fn hash_float_to_float2(first: f32, second: f32) -> f32 {
    uint_to_float(hash_uint2(first.to_bits(), second.to_bits()))
}

fn hash_float_to_float3_scalar(value: f32) -> [f32; 3] {
    [
        hash_float_to_float(value),
        hash_float_to_float2(value, 1.0),
        hash_float_to_float2(value, 2.0),
    ]
}

fn uint_to_float(value: u32) -> f32 {
    value as f32 / u32::MAX as f32
}

fn hash_uint(value: u32) -> u32 {
    let mut a = 0xdead_beef_u32.wrapping_add(17);
    let mut b = a;
    let mut c = a;
    a = a.wrapping_add(value);
    hash_final(&mut a, &mut b, &mut c);
    c
}

fn hash_uint2(first: u32, second: u32) -> u32 {
    let mut a = 0xdead_beef_u32.wrapping_add(21);
    let mut b = a.wrapping_add(second);
    let mut c = a;
    a = a.wrapping_add(first);
    hash_final(&mut a, &mut b, &mut c);
    c
}

fn hash_uint3(first: u32, second: u32, third: u32) -> u32 {
    let mut a = 0xdead_beef_u32.wrapping_add(25);
    let mut b = a.wrapping_add(second);
    let mut c = a.wrapping_add(third);
    a = a.wrapping_add(first);
    hash_final(&mut a, &mut b, &mut c);
    c
}

fn hash_uint4(first: u32, second: u32, third: u32, fourth: u32) -> u32 {
    let mut a = 0xdead_beef_u32.wrapping_add(29).wrapping_add(first);
    let mut b = 0xdead_beef_u32.wrapping_add(29).wrapping_add(second);
    let mut c = 0xdead_beef_u32.wrapping_add(29).wrapping_add(third);
    hash_mix(&mut a, &mut b, &mut c);
    a = a.wrapping_add(fourth);
    hash_final(&mut a, &mut b, &mut c);
    c
}

fn hash_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = (*a).wrapping_sub(*c);
    *a ^= (*c).rotate_left(4);
    *c = (*c).wrapping_add(*b);
    *b = (*b).wrapping_sub(*a);
    *b ^= (*a).rotate_left(6);
    *a = (*a).wrapping_add(*c);
    *c = (*c).wrapping_sub(*b);
    *c ^= (*b).rotate_left(8);
    *b = (*b).wrapping_add(*a);
    *a = (*a).wrapping_sub(*c);
    *a ^= (*c).rotate_left(16);
    *c = (*c).wrapping_add(*b);
    *b = (*b).wrapping_sub(*a);
    *b ^= (*a).rotate_left(19);
    *a = (*a).wrapping_add(*c);
    *c = (*c).wrapping_sub(*b);
    *c ^= (*b).rotate_left(4);
    *b = (*b).wrapping_add(*a);
}

fn hash_final(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = (*c).wrapping_sub((*b).rotate_left(14));
    *a ^= *c;
    *a = (*a).wrapping_sub((*c).rotate_left(11));
    *b ^= *a;
    *b = (*b).wrapping_sub((*a).rotate_left(25));
    *c ^= *b;
    *c = (*c).wrapping_sub((*b).rotate_left(16));
    *a ^= *c;
    *a = (*a).wrapping_sub((*c).rotate_left(4));
    *b ^= *a;
    *b = (*b).wrapping_sub((*a).rotate_left(14));
    *c ^= *b;
    *c = (*c).wrapping_sub((*b).rotate_left(24));
}

pub(super) fn voronoi(
    coordinate: [f32; 4],
    dimensions: usize,
    mut params: VoronoiParams,
    calc_color: bool,
) -> VoronoiOutput {
    // Detail and Roughness limits are engine-specific and applied by the evaluator.
    match dimensions {
        1 => {
            let max_distance = 0.5 + 0.5 * params.randomness;
            params.max_distance = if params.feature == VoronoiFeature::F2 {
                max_distance * 2.0
            } else {
                max_distance
            };
            voronoi_dimension::<1>([coordinate[0]], params, calc_color)
        }
        2 => {
            params.max_distance = voronoi_max_distance::<2>(params);
            voronoi_dimension::<2>([coordinate[0], coordinate[1]], params, calc_color)
        }
        3 => {
            params.max_distance = voronoi_max_distance::<3>(params);
            voronoi_dimension::<3>(
                [coordinate[0], coordinate[1], coordinate[2]],
                params,
                calc_color,
            )
        }
        4 => {
            params.max_distance = voronoi_max_distance::<4>(params);
            voronoi_dimension::<4>(coordinate, params, calc_color)
        }
        _ => VoronoiOutput::default(),
    }
}

fn voronoi_max_distance<const N: usize>(params: VoronoiParams) -> f32 {
    if params.feature == VoronoiFeature::DistanceToEdge {
        return 0.5 + 0.5 * params.randomness;
    }
    let position = [0.0; N];
    let other = [0.5 + 0.5 * params.randomness; N];
    let factor = if params.feature == VoronoiFeature::F2 {
        2.0
    } else {
        1.0
    };
    voronoi_distance(position, other, params) * factor
}

fn voronoi_dimension<const N: usize>(
    coordinate: [f32; N],
    params: VoronoiParams,
    calc_color: bool,
) -> VoronoiOutput {
    match params.feature {
        VoronoiFeature::NSphereRadius => VoronoiOutput {
            radius: voronoi_n_sphere_radius(coordinate, params.randomness),
            ..Default::default()
        },
        VoronoiFeature::DistanceToEdge => VoronoiOutput {
            distance: fractal_voronoi_distance_to_edge(coordinate, params),
            ..Default::default()
        },
        _ => fractal_voronoi(coordinate, params, calc_color),
    }
}

fn fractal_voronoi<const N: usize>(
    coordinate: [f32; N],
    params: VoronoiParams,
    calc_color: bool,
) -> VoronoiOutput {
    let mut amplitude = 1.0;
    let mut max_amplitude = 0.0;
    let mut scale = 1.0;
    let mut output = VoronoiOutput::default();
    let zero_input = params.detail == 0.0 || params.roughness == 0.0;
    for octave_index in 0..=params.detail.ceil() as i32 {
        let scaled = coordinate.map(|value| value * scale);
        let octave = match params.feature {
            VoronoiFeature::F2 => voronoi_f2(scaled, params),
            VoronoiFeature::SmoothF1 if params.smoothness != 0.0 => {
                voronoi_smooth_f1(scaled, params, calc_color)
            }
            _ => voronoi_f1(scaled, params),
        };
        if zero_input {
            max_amplitude = 1.0;
            output = octave;
            break;
        }
        if octave_index as f32 <= params.detail {
            max_amplitude += amplitude;
            output.distance += octave.distance * amplitude;
            for channel in 0..3 {
                output.color[channel] += octave.color[channel] * amplitude;
            }
            for axis in 0..4 {
                output.position[axis] = voronoi_mix(
                    output.position[axis],
                    octave.position[axis] / scale,
                    amplitude,
                );
            }
            scale *= params.lacunarity;
            amplitude *= params.roughness;
        } else {
            let remainder = params.detail - params.detail.floor();
            if remainder != 0.0 {
                max_amplitude = voronoi_mix(max_amplitude, max_amplitude + amplitude, remainder);
                let partial_distance = output.distance + octave.distance * amplitude;
                output.distance = voronoi_mix(output.distance, partial_distance, remainder);
                for channel in 0..3 {
                    let partial_color = output.color[channel] + octave.color[channel] * amplitude;
                    output.color[channel] =
                        voronoi_mix(output.color[channel], partial_color, remainder);
                }
                for axis in 0..4 {
                    let partial = voronoi_mix(
                        output.position[axis],
                        octave.position[axis] / scale,
                        amplitude,
                    );
                    output.position[axis] = voronoi_mix(output.position[axis], partial, remainder);
                }
            }
        }
    }
    if params.normalize {
        output.distance /= max_amplitude * params.max_distance;
        for value in &mut output.color {
            *value /= max_amplitude;
        }
    }
    if params.scale == 0.0 {
        output.position = [0.0; 4];
    } else {
        for value in &mut output.position {
            *value /= params.scale;
        }
    }
    output
}

fn fractal_voronoi_distance_to_edge<const N: usize>(
    coordinate: [f32; N],
    params: VoronoiParams,
) -> f32 {
    let mut amplitude = 1.0;
    let mut max_amplitude = params.max_distance;
    let mut scale = 1.0;
    let mut distance = 8.0;
    let zero_input = params.detail == 0.0 || params.roughness == 0.0;
    for octave_index in 0..=params.detail.ceil() as i32 {
        let octave_distance =
            voronoi_distance_to_edge(coordinate.map(|value| value * scale), params.randomness);
        if zero_input {
            distance = octave_distance;
            break;
        }
        if octave_index as f32 <= params.detail {
            max_amplitude = voronoi_mix(max_amplitude, params.max_distance / scale, amplitude);
            distance = voronoi_mix(distance, distance.min(octave_distance / scale), amplitude);
            scale *= params.lacunarity;
            amplitude *= params.roughness;
        } else {
            let remainder = params.detail - params.detail.floor();
            if remainder != 0.0 {
                let lerp_amplitude =
                    voronoi_mix(max_amplitude, params.max_distance / scale, amplitude);
                max_amplitude = voronoi_mix(max_amplitude, lerp_amplitude, remainder);
                let lerp_distance =
                    voronoi_mix(distance, distance.min(octave_distance / scale), amplitude);
                distance = voronoi_mix(distance, distance.min(lerp_distance), remainder);
            }
        }
    }
    if params.normalize {
        distance / max_amplitude
    } else {
        distance
    }
}

fn voronoi_f1<const N: usize>(coordinate: [f32; N], params: VoronoiParams) -> VoronoiOutput {
    let (cell, local) = split_cell(coordinate);
    let mut min_distance = f32::MAX;
    let mut target_offset = [0_i32; N];
    let mut target_position = [0.0; N];
    for_each_offset::<N>(1, |offset| {
        let point = feature_point(cell, offset, params.randomness);
        let bound = voronoi_distance_bound(point, local, params);
        if bound < min_distance {
            target_offset = offset;
            min_distance = bound;
            target_position = point;
        }
    });
    VoronoiOutput {
        distance: voronoi_distance(target_position, local, params),
        color: cell_color(cell, target_offset),
        position: absolute_position::<N>(std::array::from_fn(|axis| {
            cell[axis] + target_position[axis]
        })),
        ..Default::default()
    }
}

fn voronoi_f2<const N: usize>(coordinate: [f32; N], params: VoronoiParams) -> VoronoiOutput {
    let (cell, local) = split_cell(coordinate);
    let mut distance_f1 = f32::MAX;
    let mut distance_f2 = f32::MAX;
    let mut offset_f1 = [0_i32; N];
    let mut position_f1 = [0.0; N];
    let mut offset_f2 = [0_i32; N];
    let mut position_f2 = [0.0; N];
    for_each_offset::<N>(1, |offset| {
        let point = feature_point(cell, offset, params.randomness);
        let distance = voronoi_distance(point, local, params);
        if distance < distance_f1 {
            distance_f2 = distance_f1;
            distance_f1 = distance;
            offset_f2 = offset_f1;
            offset_f1 = offset;
            position_f2 = position_f1;
            position_f1 = point;
        } else if distance < distance_f2 {
            distance_f2 = distance;
            offset_f2 = offset;
            position_f2 = point;
        }
    });
    VoronoiOutput {
        distance: distance_f2,
        color: cell_color(cell, offset_f2),
        position: absolute_position::<N>(std::array::from_fn(|axis| {
            cell[axis] + position_f2[axis]
        })),
        ..Default::default()
    }
}

fn voronoi_smooth_f1<const N: usize>(
    coordinate: [f32; N],
    params: VoronoiParams,
    calc_color: bool,
) -> VoronoiOutput {
    let (cell, local) = split_cell(coordinate);
    let mut distance = 0.0;
    let mut position = [0.0; N];
    let mut color = [0.0; 3];
    let mut h: f32 = -1.0;
    for_each_offset::<N>(2, |offset| {
        let point = feature_point(cell, offset, params.randomness);
        let distance_to_point = voronoi_distance(point, local, params);
        h = if h.to_bits() == (-1.0_f32).to_bits() {
            1.0
        } else {
            smoothstep(
                0.0,
                1.0,
                0.5 + 0.5 * (distance - distance_to_point) / params.smoothness,
            )
        };
        let mut correction = params.smoothness * h * (1.0 - h);
        distance = voronoi_mix(distance, distance_to_point, h) - correction;
        correction /= 1.0 + 3.0 * params.smoothness;
        if calc_color {
            let cell_color = cell_color(cell, offset);
            for channel in 0..3 {
                color[channel] = voronoi_mix(color[channel], cell_color[channel], h) - correction;
            }
        }
        for axis in 0..N {
            position[axis] = voronoi_mix(position[axis], point[axis], h) - correction;
        }
    });
    VoronoiOutput {
        distance,
        color,
        position: absolute_position::<N>(std::array::from_fn(|axis| cell[axis] + position[axis])),
        ..Default::default()
    }
}

fn split_cell<const N: usize>(coordinate: [f32; N]) -> ([f32; N], [f32; N]) {
    let cell = coordinate.map(f32::floor);
    let local = std::array::from_fn(|axis| coordinate[axis] - cell[axis]);
    (cell, local)
}

fn feature_point<const N: usize>(cell: [f32; N], offset: [i32; N], randomness: f32) -> [f32; N] {
    let mut random = [0.0; N];
    if N == 1 {
        let key = cell[0] + offset[0] as f32;
        random[0] = hash_float_to_float(key);
    } else {
        let key = std::array::from_fn(|axis| (cell[axis] as i32).wrapping_add(offset[axis]));
        random = hash_int_position(key);
    }
    std::array::from_fn(|axis| offset[axis] as f32 + random[axis] * randomness)
}

fn cell_color<const N: usize>(cell: [f32; N], offset: [i32; N]) -> [f32; 3] {
    if N == 1 {
        return hash_float_to_float3_scalar(cell[0] + offset[0] as f32);
    }
    let key: [i32; N] = std::array::from_fn(|axis| (cell[axis] as i32).wrapping_add(offset[axis]));
    hash_int_color(key)
}

fn hash_int_position<const N: usize>(key: [i32; N]) -> [f32; N] {
    let mut result = [0.0; N];
    match N {
        2 => {
            let hash = pcg2([key[0], key[1]]);
            result[0] = int_to_float(hash[0]);
            result[1] = int_to_float(hash[1]);
        }
        3 => {
            let hash = pcg3([key[0], key[1], key[2]]);
            for axis in 0..3 {
                result[axis] = int_to_float(hash[axis]);
            }
        }
        4 => {
            let hash = pcg4([key[0], key[1], key[2], key[3]]);
            for axis in 0..4 {
                result[axis] = int_to_float(hash[axis]);
            }
        }
        _ => {}
    }
    result
}

fn hash_int_color<const N: usize>(key: [i32; N]) -> [f32; 3] {
    let hash = match N {
        2 => pcg3([key[0], key[1], 0]),
        3 => pcg3([key[0], key[1], key[2]]),
        4 => {
            let value = pcg4([key[0], key[1], key[2], key[3]]);
            [value[0], value[1], value[2]]
        }
        _ => [0; 3],
    };
    [
        int_to_float(hash[0]),
        int_to_float(hash[1]),
        int_to_float(hash[2]),
    ]
}

fn pcg2(key: [i32; 2]) -> [u32; 2] {
    let mut x = (key[0] as u32)
        .wrapping_mul(1_664_525)
        .wrapping_add(1_013_904_223);
    let mut y = (key[1] as u32)
        .wrapping_mul(1_664_525)
        .wrapping_add(1_013_904_223);
    x = x.wrapping_add(y.wrapping_mul(1_664_525));
    y = y.wrapping_add(x.wrapping_mul(1_664_525));
    let old_x = x;
    let old_y = y;
    x ^= ((old_x as i32) >> 16) as u32;
    y ^= ((old_y as i32) >> 16) as u32;
    x = x.wrapping_add(y.wrapping_mul(1_664_525));
    y = y.wrapping_add(x.wrapping_mul(1_664_525));
    [x & 0x7fff_ffff, y & 0x7fff_ffff]
}

fn pcg3(key: [i32; 3]) -> [u32; 3] {
    let mut values = key.map(|value| {
        (value as u32)
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223)
    });
    values[0] = values[0].wrapping_add(values[1].wrapping_mul(values[2]));
    values[1] = values[1].wrapping_add(values[2].wrapping_mul(values[0]));
    values[2] = values[2].wrapping_add(values[0].wrapping_mul(values[1]));
    for value in &mut values {
        *value ^= ((*value as i32) >> 16) as u32;
    }
    values[0] = values[0].wrapping_add(values[1].wrapping_mul(values[2]));
    values[1] = values[1].wrapping_add(values[2].wrapping_mul(values[0]));
    values[2] = values[2].wrapping_add(values[0].wrapping_mul(values[1]));
    values.map(|value| value & 0x7fff_ffff)
}

fn pcg4(key: [i32; 4]) -> [u32; 4] {
    let mut values = key.map(|value| {
        (value as u32)
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223)
    });
    values[0] = values[0].wrapping_add(values[1].wrapping_mul(values[3]));
    values[1] = values[1].wrapping_add(values[2].wrapping_mul(values[0]));
    values[2] = values[2].wrapping_add(values[0].wrapping_mul(values[1]));
    values[3] = values[3].wrapping_add(values[1].wrapping_mul(values[2]));
    let old = values;
    for axis in 0..4 {
        values[axis] ^= ((old[axis] as i32) >> 16) as u32;
    }
    values[0] = values[0].wrapping_add(values[1].wrapping_mul(values[3]));
    values[1] = values[1].wrapping_add(values[2].wrapping_mul(values[0]));
    values[2] = values[2].wrapping_add(values[0].wrapping_mul(values[1]));
    values[3] = values[3].wrapping_add(values[1].wrapping_mul(values[2]));
    values.map(|value| value & 0x7fff_ffff)
}

fn int_to_float(value: u32) -> f32 {
    (value & 0x7fff_ffff) as f32 * (1.0 / 0x7fff_ffff_u32 as f32)
}

fn voronoi_distance<const N: usize>(
    first: [f32; N],
    second: [f32; N],
    params: VoronoiParams,
) -> f32 {
    if N == 1 {
        return (second[0] - first[0]).abs();
    }
    match params.metric {
        VoronoiMetric::Euclidean => distance_squared(first, second).sqrt(),
        VoronoiMetric::Manhattan => {
            let mut distance = 0.0;
            for axis in 0..N {
                distance += (first[axis] - second[axis]).abs();
            }
            distance
        }
        VoronoiMetric::Chebychev => {
            let mut distance: f32 = 0.0;
            for axis in 0..N {
                distance = distance.max((first[axis] - second[axis]).abs());
            }
            distance
        }
        VoronoiMetric::Minkowski => {
            let mut distance = 0.0;
            for axis in 0..N {
                distance += (first[axis] - second[axis]).abs().powf(params.exponent);
            }
            distance.powf(1.0 / params.exponent)
        }
    }
}

fn voronoi_distance_bound<const N: usize>(
    first: [f32; N],
    second: [f32; N],
    params: VoronoiParams,
) -> f32 {
    if N > 1 && params.metric == VoronoiMetric::Euclidean {
        distance_squared(first, second)
    } else {
        voronoi_distance(first, second, params)
    }
}

fn distance_squared<const N: usize>(first: [f32; N], second: [f32; N]) -> f32 {
    let mut distance = 0.0;
    for axis in 0..N {
        let difference = first[axis] - second[axis];
        distance += difference * difference;
    }
    distance
}

fn for_each_offset<const N: usize>(radius: i32, mut callback: impl FnMut([i32; N])) {
    let width = (2 * radius + 1) as usize;
    let total = width.pow(N as u32);
    for mut index in 0..total {
        let mut offset = [0; N];
        for value in &mut offset {
            *value = (index % width) as i32 - radius;
            index /= width;
        }
        callback(offset);
    }
}

fn absolute_position<const N: usize>(position: [f32; N]) -> [f32; 4] {
    match N {
        1 => [0.0, 0.0, 0.0, position[0]],
        2 => [position[0], position[1], 0.0, 0.0],
        3 => [position[0], position[1], position[2], 0.0],
        4 => [position[0], position[1], position[2], position[3]],
        _ => [0.0; 4],
    }
}

fn voronoi_distance_to_edge<const N: usize>(coordinate: [f32; N], randomness: f32) -> f32 {
    let (cell, local) = split_cell(coordinate);
    if N == 1 {
        let mid = hash_float_to_float(cell[0]) * randomness;
        let left = -1.0 + hash_float_to_float(cell[0] - 1.0) * randomness;
        let right = 1.0 + hash_float_to_float(cell[0] + 1.0) * randomness;
        let to_left = (f32::midpoint(mid, left) - local[0]).abs();
        let to_right = (f32::midpoint(mid, right) - local[0]).abs();
        return to_left.min(to_right);
    }
    let mut closest = [0.0; N];
    let mut minimum = f32::MAX;
    for_each_offset::<N>(1, |offset| {
        let point = feature_point(cell, offset, randomness);
        let vector = std::array::from_fn(|axis| point[axis] - local[axis]);
        let distance = distance_squared(vector, [0.0; N]);
        if distance < minimum {
            minimum = distance;
            closest = vector;
        }
    });
    minimum = f32::MAX;
    for_each_offset::<N>(1, |offset| {
        let point = feature_point(cell, offset, randomness);
        let vector: [f32; N] = std::array::from_fn(|axis| point[axis] - local[axis]);
        let perpendicular = std::array::from_fn(|axis| vector[axis] - closest[axis]);
        let length_squared = distance_squared(perpendicular, [0.0; N]);
        if length_squared > 0.0001 {
            let midpoint: [f32; N] =
                std::array::from_fn(|axis| f32::midpoint(closest[axis], vector[axis]));
            let inverse_length = 1.0 / length_squared.sqrt();
            let mut distance = 0.0;
            for axis in 0..N {
                distance += midpoint[axis] * perpendicular[axis] * inverse_length;
            }
            minimum = minimum.min(distance);
        }
    });
    minimum
}

fn voronoi_n_sphere_radius<const N: usize>(coordinate: [f32; N], randomness: f32) -> f32 {
    let (cell, local) = split_cell(coordinate);
    let mut closest_point = [0.0; N];
    let mut closest_offset = [0; N];
    if N == 1 {
        let mut minimum = f32::MAX;
        for_each_offset::<N>(1, |offset| {
            let point = feature_point(cell, offset, randomness);
            let distance = (point[0] - local[0]).abs();
            if distance < minimum {
                minimum = distance;
                closest_point = point;
                closest_offset = offset;
            }
        });
        minimum = f32::MAX;
        let mut closest_neighbor = 0.0;
        for_each_offset::<N>(1, |offset| {
            if offset[0] == 0 {
                return;
            }
            let cell_offset: [i32; N] =
                std::array::from_fn(|axis| offset[axis].wrapping_add(closest_offset[axis]));
            let point = feature_point(cell, cell_offset, randomness);
            let distance = (closest_point[0] - point[0]).abs();
            if distance < minimum {
                minimum = distance;
                closest_neighbor = point[0];
            }
        });
        return (closest_neighbor - closest_point[0]).abs() / 2.0;
    }
    let mut minimum = f32::MAX;
    for_each_offset::<N>(1, |offset| {
        let point = feature_point(cell, offset, randomness);
        let distance = distance_squared(point, local);
        if distance < minimum {
            minimum = distance;
            closest_point = point;
            closest_offset = offset;
        }
    });
    minimum = f32::MAX;
    let mut closest_neighbor = [0.0; N];
    for_each_offset::<N>(1, |offset| {
        if offset.iter().all(|value| *value == 0) {
            return;
        }
        let cell_offset =
            std::array::from_fn(|axis| offset[axis].wrapping_add(closest_offset[axis]));
        let point = feature_point(cell, cell_offset, randomness);
        let distance = distance_squared(closest_point, point);
        if distance < minimum {
            minimum = distance;
            closest_neighbor = point;
        }
    });
    distance_squared(closest_neighbor, closest_point).sqrt() / 2.0
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if value < edge0 {
        0.0
    } else if value >= edge1 {
        1.0
    } else {
        let t = (value - edge0) / (edge1 - edge0);
        (3.0 - 2.0 * t) * (t * t)
    }
}
