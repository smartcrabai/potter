// Blender 5.2.2 shader procedural texture kernels. Hash primitives and normalized fBM are shared
// with `texture_kernels`; the formulas here are the texture-node implementations.

use super::texture_kernels::{
    NoiseType, hash_uint, hash_uint2, hash_uint3, hash_uint4, noise_factor, uint_to_float,
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum WaveType {
    Bands,
    Rings,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum WaveBandsDirection {
    X,
    Y,
    Z,
    Diagonal,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum WaveRingsDirection {
    X,
    Y,
    Z,
    Spherical,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum WaveProfile {
    Sine,
    Saw,
    Triangle,
}

pub(crate) fn wave(
    vector: [f32; 3],
    scale: f32,
    distortion: f32,
    detail: f32,
    detail_scale: f32,
    detail_roughness: f32,
    phase: f32,
    wave_type: WaveType,
    bands_direction: WaveBandsDirection,
    rings_direction: WaveRingsDirection,
    profile: WaveProfile,
) -> f32 {
    let mut p = [vector[0] * scale, vector[1] * scale, vector[2] * scale];
    for value in &mut p {
        *value = (*value + 0.000_001) * 0.999_999;
    }

    let mut n = match wave_type {
        WaveType::Bands => match bands_direction {
            WaveBandsDirection::X => p[0] * 20.0,
            WaveBandsDirection::Y => p[1] * 20.0,
            WaveBandsDirection::Z => p[2] * 20.0,
            WaveBandsDirection::Diagonal => (p[0] + p[1] + p[2]) * 10.0,
        },
        WaveType::Rings => {
            let radius_squared = match rings_direction {
                WaveRingsDirection::X => p[1] * p[1] + p[2] * p[2],
                WaveRingsDirection::Y => p[0] * p[0] + p[2] * p[2],
                WaveRingsDirection::Z => p[0] * p[0] + p[1] * p[1],
                WaveRingsDirection::Spherical => p[0] * p[0] + p[1] * p[1] + p[2] * p[2],
            };
            radius_squared.sqrt() * 20.0
        }
    };
    n += phase;

    if distortion != 0.0 {
        let noise = noise_factor(
            [
                p[0] * detail_scale,
                p[1] * detail_scale,
                p[2] * detail_scale,
                0.0,
            ],
            3,
            detail,
            detail_roughness,
            2.0,
            0.0,
            0.0,
            0.0,
            NoiseType::Fbm,
            true,
        );
        n += distortion * (noise * 2.0 - 1.0);
    }

    match profile {
        WaveProfile::Sine => 0.5 + 0.5 * (n - std::f32::consts::FRAC_PI_2).sin(),
        WaveProfile::Saw => {
            let cycles = n / std::f32::consts::TAU;
            cycles - cycles.floor()
        }
        WaveProfile::Triangle => {
            let cycles = n / std::f32::consts::TAU;
            (cycles - (cycles + 0.5).floor()).abs() * 2.0
        }
    }
}

pub(crate) fn magic(vector: [f32; 3], scale: f32, distortion: f32, depth: u8) -> ([f32; 4], f32) {
    let co = [vector[0] * scale, vector[1] * scale, vector[2] * scale];
    magic_from_coordinates(co, distortion, depth)
}

pub(crate) fn magic_shader(
    vector: [f32; 3],
    scale: f32,
    distortion: f32,
    depth: u8,
    path_traced: bool,
) -> ([f32; 4], f32) {
    let mut co = [vector[0] * scale, vector[1] * scale, vector[2] * scale];
    // Cycles uses fmod while EEVEE uses floored mod; Geometry Nodes calls the unwrapped kernel.
    let period = std::f32::consts::TAU;
    for value in &mut co {
        *value = if path_traced {
            *value % period
        } else {
            *value - (*value / period).floor() * period
        };
    }
    magic_from_coordinates(co, distortion, depth)
}

fn magic_from_coordinates(co: [f32; 3], distortion: f32, depth: u8) -> ([f32; 4], f32) {
    let mut x = ((co[0] + co[1] + co[2]) * 5.0).sin();
    let mut y = ((-co[0] + co[1] - co[2]) * 5.0).cos();
    let mut z = -((-co[0] - co[1] + co[2]) * 5.0).cos();

    if depth > 0 {
        x *= distortion;
        y *= distortion;
        z *= distortion;
        y = -(x - y + z).cos();
        y *= distortion;
        if depth > 1 {
            x = (x - y - z).cos();
            x *= distortion;
            if depth > 2 {
                z = (-x - y - z).sin();
                z *= distortion;
                if depth > 3 {
                    x = -(-x + y - z).cos();
                    x *= distortion;
                    if depth > 4 {
                        y = -(-x + y + z).sin();
                        y *= distortion;
                        if depth > 5 {
                            y = -(-x + y + z).cos();
                            y *= distortion;
                            if depth > 6 {
                                x = (x + y + z).cos();
                                x *= distortion;
                                if depth > 7 {
                                    z = (x + y - z).sin();
                                    z *= distortion;
                                    if depth > 8 {
                                        x = -(-x - y + z).cos();
                                        x *= distortion;
                                        if depth > 9 {
                                            y = -(x - y + z).sin();
                                            y *= distortion;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if distortion != 0.0 {
        let divisor = distortion * 2.0;
        x /= divisor;
        y /= divisor;
        z /= divisor;
    }
    let color = [0.5 - x, 0.5 - y, 0.5 - z, 1.0];
    let factor = (color[0] + color[1] + color[2]) / 3.0;
    (color, factor)
}

pub(crate) fn brick(
    vector: [f32; 3],
    color1: [f32; 4],
    color2: [f32; 4],
    mortar_color: [f32; 4],
    scale: f32,
    mortar_size: f32,
    mortar_smooth: f32,
    bias: f32,
    brick_width: f32,
    row_height: f32,
    offset_amount: f32,
    offset_frequency: i32,
    squash_amount: f32,
    squash_frequency: i32,
) -> ([f32; 4], f32) {
    let p = [vector[0] * scale, vector[1] * scale, vector[2] * scale];
    let rownum = (p[1] / row_height).floor() as i32;
    let mut width = brick_width;
    let mut offset = 0.0;
    if offset_frequency != 0 && squash_frequency != 0 {
        if rownum % squash_frequency == 0 {
            width *= squash_amount;
        }
        if rownum % offset_frequency == 0 {
            offset = width * offset_amount;
        }
    }
    let bricknum = ((p[0] + offset) / width).floor() as i32;
    let x = (p[0] + offset) - width * bricknum as f32;
    let y = p[1] - row_height * rownum as f32;
    let noise_index =
        ((rownum as u32).wrapping_shl(16)).wrapping_add((bricknum as u32) & 0xffff) as i32;
    let tint = (brick_noise(noise_index) + bias).clamp(0.0, 1.0);

    let mut min_distance = x.min(y).min(width - x).min(row_height - y);
    let mortar = if min_distance >= mortar_size {
        0.0
    } else if mortar_smooth == 0.0 {
        1.0
    } else {
        min_distance = 1.0 - min_distance / mortar_size;
        if min_distance < mortar_smooth {
            let t = min_distance / mortar_smooth;
            t * t * (3.0 - 2.0 * t)
        } else {
            1.0
        }
    };

    let mut color = [0.0; 4];
    if mortar.to_bits() == 1.0_f32.to_bits() {
        color = color1;
    } else {
        let color1_weight = 1.0 - tint;
        for channel in 0..4 {
            color[channel] = color1[channel] * color1_weight + color2[channel] * tint;
        }
    }
    for channel in 0..4 {
        color[channel] = color[channel] * (1.0 - mortar) + mortar_color[channel] * mortar;
    }
    (color, mortar)
}

fn brick_noise(value: i32) -> f32 {
    let mut hash = (value as u32).wrapping_add(1013) & 0x7fff_ffff;
    hash = (hash >> 13) ^ hash;
    hash = hash
        .wrapping_mul(
            hash.wrapping_mul(hash)
                .wrapping_mul(60_493)
                .wrapping_add(19_990_303),
        )
        .wrapping_add(1_376_312_589)
        & 0x7fff_ffff;
    0.5 * (hash as f32 / 1_073_741_824.0)
}

pub(crate) fn white_noise_value(dimensions: u8, vector: [f32; 3], w: f32) -> Option<f32> {
    Some(match dimensions {
        1 => hash_float(w),
        2 => hash_vec2([vector[0], vector[1]]),
        3 => hash_vec3(vector),
        4 => hash_vec4([vector[0], vector[1], vector[2], w]),
        _ => return None,
    })
}

pub(crate) fn white_noise_color(dimensions: u8, vector: [f32; 3], w: f32) -> Option<[f32; 4]> {
    let rgb = match dimensions {
        1 => hash_float_to_vec3(w),
        2 => hash_vec2_to_vec3([vector[0], vector[1]]),
        3 => hash_vec3_to_vec3(vector),
        4 => hash_vec4_to_vec3([vector[0], vector[1], vector[2], w]),
        _ => return None,
    };
    Some([rgb[0], rgb[1], rgb[2], 1.0])
}

pub(crate) fn gabor_2d(
    vector: [f32; 3],
    scale: f32,
    frequency: f32,
    anisotropy: f32,
    orientation: f32,
) -> [f32; 3] {
    let coordinates = [vector[0] * scale, vector[1] * scale];
    let frequency = frequency.max(0.001);
    let isotropy = 1.0 - anisotropy.clamp(0.0, 1.0);
    let cell_position = [coordinates[0].floor(), coordinates[1].floor()];
    let local_position = [
        coordinates[0] - cell_position[0],
        coordinates[1] - cell_position[1],
    ];
    let mut phasor = [0.0; 2];

    for j in -1..=1 {
        for i in -1..=1 {
            let cell = [cell_position[0] + i as f32, cell_position[1] + j as f32];
            let position = [local_position[0] - i as f32, local_position[1] - j as f32];
            let sample = gabor_cell_2d(cell, position, frequency, isotropy, orientation);
            phasor[0] += sample[0];
            phasor[1] += sample[1];
        }
    }
    gabor_outputs(phasor, (8.0 * 0.5 * 0.25_f32).sqrt())
}

fn gabor_cell_2d(
    cell: [f32; 2],
    position: [f32; 2],
    frequency: f32,
    isotropy: f32,
    base_orientation: f32,
) -> [f32; 2] {
    let mut noise = [0.0; 2];
    for impulse in 0..8 {
        let seed = [cell[0], cell[1], (impulse * 3) as f32];
        let random_orientation = (hash_vec3(seed) - 0.5) * std::f32::consts::PI;
        let orientation = base_orientation + random_orientation * isotropy;
        let center_seed = [seed[0], seed[1], seed[2] + 1.0];
        let kernel_center = hash_vec3_to_vec2(center_seed);
        let position_in_kernel = [
            position[0] - kernel_center[0],
            position[1] - kernel_center[1],
        ];
        let distance_squared = position_in_kernel[0] * position_in_kernel[0]
            + position_in_kernel[1] * position_in_kernel[1];
        if distance_squared >= 1.0 {
            continue;
        }
        let weight_seed = [seed[0], seed[1], seed[2] + 2.0];
        let weight = if hash_vec3(weight_seed) < 0.5 {
            -1.0
        } else {
            1.0
        };
        let kernel = gabor_kernel_2d(position_in_kernel, frequency, orientation);
        noise[0] += weight * kernel[0];
        noise[1] += weight * kernel[1];
    }
    noise
}

fn gabor_kernel_2d(position: [f32; 2], frequency: f32, orientation: f32) -> [f32; 2] {
    let distance_squared = position[0] * position[0] + position[1] * position[1];
    let window = 0.5 + 0.5 * (std::f32::consts::PI * distance_squared).cos();
    let envelope = (-std::f32::consts::PI * distance_squared).exp() * window;
    let frequency_vector = [frequency * orientation.cos(), frequency * orientation.sin()];
    let angle = 2.0
        * std::f32::consts::PI
        * (position[0] * frequency_vector[0] + position[1] * frequency_vector[1]);
    [envelope * angle.cos(), envelope * angle.sin()]
}

pub(crate) fn gabor_3d(
    vector: [f32; 3],
    scale: f32,
    frequency: f32,
    anisotropy: f32,
    orientation: [f32; 3],
) -> [f32; 3] {
    gabor_3d_with_azimuth_sign(vector, scale, frequency, anisotropy, orientation, false)
}

pub(crate) fn gabor_3d_shader(
    vector: [f32; 3],
    scale: f32,
    frequency: f32,
    anisotropy: f32,
    orientation: [f32; 3],
    path_traced: bool,
) -> [f32; 3] {
    gabor_3d_with_azimuth_sign(
        vector,
        scale,
        frequency,
        anisotropy,
        orientation,
        path_traced,
    )
}

fn gabor_3d_with_azimuth_sign(
    vector: [f32; 3],
    scale: f32,
    frequency: f32,
    anisotropy: f32,
    orientation: [f32; 3],
    cycles_svm: bool,
) -> [f32; 3] {
    let coordinates = [vector[0] * scale, vector[1] * scale, vector[2] * scale];
    let frequency = frequency.max(0.001);
    let isotropy = 1.0 - anisotropy.clamp(0.0, 1.0);
    let orientation_length = (orientation[0] * orientation[0]
        + orientation[1] * orientation[1]
        + orientation[2] * orientation[2])
        .sqrt();
    let base_orientation = if orientation_length == 0.0 {
        [0.0; 3]
    } else {
        [
            orientation[0] / orientation_length,
            orientation[1] / orientation_length,
            orientation[2] / orientation_length,
        ]
    };
    let cell_position = [
        coordinates[0].floor(),
        coordinates[1].floor(),
        coordinates[2].floor(),
    ];
    let local_position = [
        coordinates[0] - cell_position[0],
        coordinates[1] - cell_position[1],
        coordinates[2] - cell_position[2],
    ];
    let mut phasor = [0.0; 2];

    for k in -1..=1 {
        for j in -1..=1 {
            for i in -1..=1 {
                let offset = [i as f32, j as f32, k as f32];
                let cell = [
                    cell_position[0] + offset[0],
                    cell_position[1] + offset[1],
                    cell_position[2] + offset[2],
                ];
                let position = [
                    local_position[0] - offset[0],
                    local_position[1] - offset[1],
                    local_position[2] - offset[2],
                ];
                let sample = gabor_cell_3d(
                    cell,
                    position,
                    frequency,
                    isotropy,
                    base_orientation,
                    cycles_svm,
                );
                phasor[0] += sample[0];
                phasor[1] += sample[1];
            }
        }
    }
    gabor_outputs(
        phasor,
        (8.0 * 0.5 / (4.0 * std::f32::consts::SQRT_2)).sqrt(),
    )
}

fn gabor_cell_3d(
    cell: [f32; 3],
    position: [f32; 3],
    frequency: f32,
    isotropy: f32,
    base_orientation: [f32; 3],
    cycles_svm: bool,
) -> [f32; 2] {
    let mut noise = [0.0; 2];
    for impulse in 0..8 {
        let seed = [cell[0], cell[1], cell[2], (impulse * 3) as f32];
        let orientation = gabor_orientation_3d(base_orientation, isotropy, seed, cycles_svm);
        let center_seed = [seed[0], seed[1], seed[2], seed[3] + 1.0];
        let kernel_center = hash_vec4_to_vec3(center_seed);
        let position_in_kernel = [
            position[0] - kernel_center[0],
            position[1] - kernel_center[1],
            position[2] - kernel_center[2],
        ];
        let distance_squared = position_in_kernel[0] * position_in_kernel[0]
            + position_in_kernel[1] * position_in_kernel[1]
            + position_in_kernel[2] * position_in_kernel[2];
        if distance_squared >= 1.0 {
            continue;
        }
        let weight_seed = [seed[0], seed[1], seed[2], seed[3] + 2.0];
        let weight = if hash_vec4(weight_seed) < 0.5 {
            -1.0
        } else {
            1.0
        };
        let kernel = gabor_kernel_3d(position_in_kernel, frequency, orientation);
        noise[0] += weight * kernel[0];
        noise[1] += weight * kernel[1];
    }
    noise
}

fn gabor_orientation_3d(
    orientation: [f32; 3],
    isotropy: f32,
    seed: [f32; 4],
    cycles_svm: bool,
) -> [f32; 3] {
    if isotropy == 0.0 {
        return orientation;
    }
    let inclination = orientation[2].acos();
    let xy_length = (orientation[0] * orientation[0] + orientation[1] * orientation[1]).sqrt();
    // Cycles SVM treats a zero Y direction as positive; EEVEE and Geometry Nodes use sign(0) = 0.
    let azimuth_sign = if cycles_svm {
        if orientation[1] < 0.0 { -1.0 } else { 1.0 }
    } else {
        sign(orientation[1])
    };
    let azimuth = azimuth_sign * (orientation[0] / xy_length).acos();
    let random_angles = hash_vec4_to_vec2(seed);
    let inclination = inclination + random_angles[0] * std::f32::consts::PI * isotropy;
    let azimuth = azimuth + random_angles[1] * std::f32::consts::PI * isotropy;
    [
        inclination.sin() * azimuth.cos(),
        inclination.sin() * azimuth.sin(),
        inclination.cos(),
    ]
}

fn gabor_kernel_3d(position: [f32; 3], frequency: f32, orientation: [f32; 3]) -> [f32; 2] {
    let distance_squared =
        position[0] * position[0] + position[1] * position[1] + position[2] * position[2];
    let window = 0.5 + 0.5 * (std::f32::consts::PI * distance_squared).cos();
    let envelope = (-std::f32::consts::PI * distance_squared).exp() * window;
    let angle = 2.0
        * std::f32::consts::PI
        * frequency
        * (position[0] * orientation[0]
            + position[1] * orientation[1]
            + position[2] * orientation[2]);
    [envelope * angle.cos(), envelope * angle.sin()]
}

fn gabor_outputs(phasor: [f32; 2], standard_deviation: f32) -> [f32; 3] {
    let normalization = 6.0 * standard_deviation;
    [
        (phasor[1] / normalization) * 0.5 + 0.5,
        (phasor[1].atan2(phasor[0]) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI),
        (phasor[0] * phasor[0] + phasor[1] * phasor[1]).sqrt() / normalization,
    ]
}

fn hash_float(value: f32) -> f32 {
    uint_to_float(hash_uint(value.to_bits()))
}

fn hash_vec2(value: [f32; 2]) -> f32 {
    uint_to_float(hash_uint2(value[0].to_bits(), value[1].to_bits()))
}

fn hash_vec3(value: [f32; 3]) -> f32 {
    uint_to_float(hash_uint3(
        value[0].to_bits(),
        value[1].to_bits(),
        value[2].to_bits(),
    ))
}

fn hash_vec4(value: [f32; 4]) -> f32 {
    uint_to_float(hash_uint4(
        value[0].to_bits(),
        value[1].to_bits(),
        value[2].to_bits(),
        value[3].to_bits(),
    ))
}

fn hash_float_to_vec3(value: f32) -> [f32; 3] {
    [
        hash_float(value),
        hash_vec2([value, 1.0]),
        hash_vec2([value, 2.0]),
    ]
}

fn hash_vec2_to_vec3(value: [f32; 2]) -> [f32; 3] {
    [
        hash_vec2(value),
        hash_vec3([value[0], value[1], 1.0]),
        hash_vec3([value[0], value[1], 2.0]),
    ]
}

fn hash_vec3_to_vec3(value: [f32; 3]) -> [f32; 3] {
    [
        hash_vec3(value),
        hash_vec4([value[0], value[1], value[2], 1.0]),
        hash_vec4([value[0], value[1], value[2], 2.0]),
    ]
}

fn hash_vec4_to_vec3(value: [f32; 4]) -> [f32; 3] {
    [
        hash_vec4(value),
        hash_vec4([value[2], value[0], value[3], value[1]]),
        hash_vec4([value[3], value[2], value[1], value[0]]),
    ]
}

fn hash_vec3_to_vec2(value: [f32; 3]) -> [f32; 2] {
    [hash_vec3(value), hash_vec3([value[2], value[0], value[1]])]
}

fn hash_vec4_to_vec2(value: [f32; 4]) -> [f32; 2] {
    [
        hash_vec4(value),
        hash_vec4([value[2], value[0], value[3], value[1]]),
    ]
}

fn sign(value: f32) -> f32 {
    if value < 0.0 {
        -1.0
    } else if value > 0.0 {
        1.0
    } else {
        0.0
    }
}
