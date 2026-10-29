use std::sync::LazyLock;

type Color = [f64; 4];

const CIE_COLOR_MATCH: [[f32; 3]; 81] = [
    [0.0014, 0.0000, 0.0065],
    [0.0022, 0.0001, 0.0105],
    [0.0042, 0.0001, 0.0201],
    [0.0076, 0.0002, 0.0362],
    [0.0143, 0.0004, 0.0679],
    [0.0232, 0.0006, 0.1102],
    [0.0435, 0.0012, 0.2074],
    [0.0776, 0.0022, 0.3713],
    [0.1344, 0.0040, 0.6456],
    [0.2148, 0.0073, 1.0391],
    [0.2839, 0.0116, 1.3856],
    [0.3285, 0.0168, 1.6230],
    [0.3483, 0.0230, 1.7471],
    [0.3481, 0.0298, 1.7826],
    [0.3362, 0.0380, 1.7721],
    [0.3187, 0.0480, 1.7441],
    [0.2908, 0.0600, 1.6692],
    [0.2511, 0.0739, 1.5281],
    [0.1954, 0.0910, 1.2876],
    [0.1421, 0.1126, 1.0419],
    [0.0956, 0.1390, 0.8130],
    [0.0580, 0.1693, 0.6162],
    [0.0320, 0.2080, 0.4652],
    [0.0147, 0.2586, 0.3533],
    [0.0049, 0.3230, 0.2720],
    [0.0024, 0.4073, 0.2123],
    [0.0093, 0.5030, 0.1582],
    [0.0291, 0.6082, 0.1117],
    [0.0633, 0.7100, 0.0782],
    [0.1096, 0.7932, 0.0573],
    [0.1655, 0.8620, 0.0422],
    [0.2257, 0.9149, 0.0298],
    [0.2904, 0.9540, 0.0203],
    [0.3597, 0.9803, 0.0134],
    [0.4334, 0.9950, 0.0087],
    [0.5121, 1.0000, 0.0057],
    [0.5945, 0.9950, 0.0039],
    [0.6784, 0.9786, 0.0027],
    [0.7621, 0.9520, 0.0021],
    [0.8425, 0.9154, 0.0018],
    [0.9163, 0.8700, 0.0017],
    [0.9786, 0.8163, 0.0014],
    [1.0263, 0.7570, 0.0011],
    [1.0567, 0.6949, 0.0010],
    [1.0622, 0.6310, 0.0008],
    [1.0456, 0.5668, 0.0006],
    [1.0026, 0.5030, 0.0003],
    [0.9384, 0.4412, 0.0002],
    [0.8544, 0.3810, 0.0002],
    [0.7514, 0.3210, 0.0001],
    [0.6424, 0.2650, 0.0000],
    [0.5419, 0.2170, 0.0000],
    [0.4479, 0.1750, 0.0000],
    [0.3608, 0.1382, 0.0000],
    [0.2835, 0.1070, 0.0000],
    [0.2187, 0.0816, 0.0000],
    [0.1649, 0.0610, 0.0000],
    [0.1212, 0.0446, 0.0000],
    [0.0874, 0.0320, 0.0000],
    [0.0636, 0.0232, 0.0000],
    [0.0468, 0.0170, 0.0000],
    [0.0329, 0.0119, 0.0000],
    [0.0227, 0.0082, 0.0000],
    [0.0158, 0.0057, 0.0000],
    [0.0114, 0.0041, 0.0000],
    [0.0081, 0.0029, 0.0000],
    [0.0058, 0.0021, 0.0000],
    [0.0041, 0.0015, 0.0000],
    [0.0029, 0.0010, 0.0000],
    [0.0020, 0.0007, 0.0000],
    [0.0014, 0.0005, 0.0000],
    [0.0010, 0.0004, 0.0000],
    [0.0007, 0.0002, 0.0000],
    [0.0005, 0.0002, 0.0000],
    [0.0003, 0.0001, 0.0000],
    [0.0002, 0.0001, 0.0000],
    [0.0002, 0.0001, 0.0000],
    [0.0001, 0.0000, 0.0000],
    [0.0001, 0.0000, 0.0000],
    [0.0001, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000],
];

#[must_use]
pub(crate) fn combine_color(mode: &str, red: f64, green: f64, blue: f64) -> Option<Color> {
    let channels = [red as f32, green as f32, blue as f32];
    let rgb = match mode {
        "RGB" => channels,
        "HSV" => hsv_to_rgb([channels[0], channels[1], channels[2]]),
        "HSL" => hsl_to_rgb([channels[0], channels[1], channels[2]]),
        _ => return None,
    };
    Some([f64::from(rgb[0]), f64::from(rgb[1]), f64::from(rgb[2]), 1.0])
}

#[must_use]
pub(crate) fn separate_color(mode: &str, color: Color) -> Option<[f64; 3]> {
    let rgb = [color[0] as f32, color[1] as f32, color[2] as f32];
    let values = match mode {
        "RGB" => rgb,
        "HSV" => rgb_to_hsv(rgb),
        "HSL" => rgb_to_hsl(rgb),
        _ => return None,
    };
    Some(values.map(f64::from))
}

#[must_use]
pub(crate) fn rgb_to_bw(color: Color) -> f64 {
    let [red, green, blue] = [color[0] as f32, color[1] as f32, color[2] as f32];
    f64::from(red * 0.2126 + green * 0.7152 + blue * 0.0722)
}

#[must_use]
pub(crate) fn invert_color(color: Color, factor: f64) -> Color {
    let factor = factor as f32;
    let mut result = color;
    for channel in 0..3 {
        let original = color[channel] as f32;
        result[channel] = f64::from(original + (1.0 - 2.0 * original) * factor);
    }
    result
}

#[must_use]
pub(crate) fn gamma_color(color: Color, gamma: f64) -> Color {
    let gamma = gamma as f32;
    let mut result = color;
    for channel in 0..3 {
        let value = color[channel] as f32;
        if value > 0.0 {
            result[channel] = f64::from(value.powf(gamma));
        }
    }
    result
}

#[must_use]
pub(crate) fn brightness_contrast(color: Color, brightness: f64, contrast: f64) -> Color {
    let brightness = brightness as f32;
    let contrast = contrast as f32;
    let a = 1.0 + contrast;
    let b = brightness - contrast * 0.5;
    let mut result = color;
    for channel in 0..3 {
        result[channel] = f64::from((a * color[channel] as f32 + b).max(0.0));
    }
    result
}

#[must_use]
pub(crate) fn hue_saturation(
    color: Color,
    hue: f64,
    saturation: f64,
    value: f64,
    factor: f64,
) -> Color {
    let mut hsv = rgb_to_hsv([color[0] as f32, color[1] as f32, color[2] as f32]);
    hsv[0] = fract(hsv[0] + hue as f32 + 0.5);
    hsv[1] = (hsv[1] * saturation as f32).clamp(0.0, 1.0);
    hsv[2] *= value as f32;
    let adjusted = hsv_to_rgb(hsv);
    let factor = factor as f32;
    let mut result = color;
    for channel in 0..3 {
        let original = color[channel] as f32;
        result[channel] = f64::from(original + (adjusted[channel] - original) * factor);
    }
    result
}

#[must_use]
pub(crate) fn wavelength_rgb(wavelength_nm: f32) -> [f32; 3] {
    wavelength_scene_linear_rgb(wavelength_nm).map(|channel| (channel / 2.52).max(0.0))
}

fn wavelength_scene_linear_rgb(wavelength_nm: f32) -> [f32; 3] {
    let scaled = (wavelength_nm - 380.0) * 0.2;
    let index = scaled as i32;
    if !(0..80).contains(&index) {
        return [0.0; 3];
    }
    let factor = scaled - index as f32;
    let start = CIE_COLOR_MATCH[index as usize];
    let end = CIE_COLOR_MATCH[index as usize + 1];
    let xyz = [
        start[0] + factor * (end[0] - start[0]),
        start[1] + factor * (end[1] - start[1]),
        start[2] + factor * (end[2] - start[2]),
    ];
    // Potter's working RGB is linear Rec.709, the scene-linear transform of XYZ.
    [
        3.240_454_2 * xyz[0] - 1.537_138_5 * xyz[1] - 0.498_531_4 * xyz[2],
        -0.969_266 * xyz[0] + 1.876_010_8 * xyz[1] + 0.041_556 * xyz[2],
        0.055_643_4 * xyz[0] - 0.204_025_9 * xyz[1] + 1.057_225_2 * xyz[2],
    ]
    .map(|channel| channel.max(0.0))
}

const EEVEE_SPECTRUM_TEXELS: usize = 257;

type EeveeSpectrumTable = [[f32; 3]; EEVEE_SPECTRUM_TEXELS];

static EEVEE_WAVELENGTH_TABLE: LazyLock<EeveeSpectrumTable> = LazyLock::new(|| {
    std::array::from_fn(|index| {
        let wavelength = 380.0 + 400.0 / EEVEE_SPECTRUM_TEXELS as f32 * index as f32;
        wavelength_scene_linear_rgb(wavelength).map(f32_to_f16_roundtrip)
    })
});
static EEVEE_BLACKBODY_TABLE: LazyLock<EeveeSpectrumTable> = LazyLock::new(|| {
    std::array::from_fn(|index| {
        let temperature = 800.0 + 11_200.0 / EEVEE_SPECTRUM_TEXELS as f32 * index as f32;
        super::blackbody::blackbody_rgb(temperature)
            .map(|channel| f32_to_f16_roundtrip(channel.max(0.0)))
    })
});

fn f32_to_f16_roundtrip(value: f32) -> f32 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ff_ff;
    let half = if exponent >= 31 {
        sign | 0x7c00
    } else if exponent > 0 {
        let mut half_exponent = exponent as u32;
        let mut half_mantissa = round_shift_to_even(mantissa, 13);
        if half_mantissa == 0x400 {
            half_exponent += 1;
            half_mantissa = 0;
        }
        if half_exponent >= 31 {
            sign | 0x7c00
        } else {
            sign | (half_exponent << 10) | half_mantissa
        }
    } else if exponent >= -10 {
        let significand = 0x80_0000 | mantissa;
        let half_mantissa = round_shift_to_even(significand, (14 - exponent) as u32);
        sign | half_mantissa
    } else {
        sign
    };
    f16_to_f32(half)
}

fn round_shift_to_even(value: u32, shift: u32) -> u32 {
    let truncated = value >> shift;
    let remainder = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

fn f16_to_f32(half: u32) -> f32 {
    let sign = (half & 0x8000) << 16;
    let exponent = (half >> 10) & 0x1f;
    let mantissa = half & 0x3ff;
    if exponent == 0 {
        let value = mantissa as f32 * (1.0 / 16_777_216.0);
        if sign == 0 { value } else { -value }
    } else if exponent == 0x1f {
        f32::INFINITY
    } else {
        f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13))
    }
}

fn eevee_spectrum_sample(
    value: f32,
    minimum: f32,
    maximum: f32,
    table: &EeveeSpectrumTable,
) -> [f32; 3] {
    let coordinate =
        ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0) * EEVEE_SPECTRUM_TEXELS as f32;
    let coordinate = coordinate.clamp(0.0, (EEVEE_SPECTRUM_TEXELS - 1) as f32);
    let index = coordinate as usize;
    let fraction = coordinate - index as f32;
    let first = table[index];
    let second = table[(index + 1).min(EEVEE_SPECTRUM_TEXELS - 1)];
    std::array::from_fn(|channel| first[channel] + (second[channel] - first[channel]) * fraction)
}

#[must_use]
pub(crate) fn eevee_wavelength_rgb(wavelength_nm: f32) -> [f32; 3] {
    eevee_spectrum_sample(wavelength_nm, 380.0, 780.0, &EEVEE_WAVELENGTH_TABLE)
        .map(|channel| (channel / 2.52).clamp(0.0, 1.0))
}

#[must_use]
pub(crate) fn eevee_blackbody_rgb(temperature_kelvin: f32) -> [f32; 3] {
    eevee_spectrum_sample(temperature_kelvin, 800.0, 12_000.0, &EEVEE_BLACKBODY_TABLE)
}

fn same_f32(first: f32, second: f32) -> bool {
    first.partial_cmp(&second) == Some(std::cmp::Ordering::Equal)
}

fn rgb_to_hsv(rgb: [f32; 3]) -> [f32; 3] {
    let cmax = rgb[0].max(rgb[1].max(rgb[2]));
    let cmin = rgb[0].min(rgb[1].min(rgb[2]));
    let delta = cmax - cmin;
    let saturation = if same_f32(cmax, 0.0) {
        0.0
    } else {
        delta / cmax
    };
    let mut hue = if same_f32(cmax, 0.0) || same_f32(saturation, 0.0) {
        0.0
    } else {
        let c = [cmax - rgb[0], cmax - rgb[1], cmax - rgb[2]];
        let raw = if same_f32(rgb[0], cmax) {
            c[2] / delta - c[1] / delta
        } else if same_f32(rgb[1], cmax) {
            2.0 + c[0] / delta - c[2] / delta
        } else {
            4.0 + c[1] / delta - c[0] / delta
        };
        raw / 6.0
    };
    if hue < 0.0 {
        hue += 1.0;
    }
    [hue, saturation, cmax]
}

fn hsv_to_rgb(mut hsv: [f32; 3]) -> [f32; 3] {
    let [hue, saturation, value] = hsv;
    if same_f32(saturation, 0.0) {
        return [value; 3];
    }
    if same_f32(hue, 1.0) {
        hsv[0] = 0.0;
    }
    let hue = hsv[0] * 6.0;
    let sector = hue.floor();
    let fraction = hue - sector;
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * fraction);
    let t = value * (1.0 - saturation * (1.0 - fraction));
    if same_f32(sector, 0.0) {
        [value, t, p]
    } else if same_f32(sector, 1.0) {
        [q, value, p]
    } else if same_f32(sector, 2.0) {
        [p, value, t]
    } else if same_f32(sector, 3.0) {
        [p, q, value]
    } else if same_f32(sector, 4.0) {
        [t, p, value]
    } else {
        [value, p, q]
    }
}

fn rgb_to_hsl(rgb: [f32; 3]) -> [f32; 3] {
    let cmax = rgb[0].max(rgb[1].max(rgb[2]));
    let cmin = rgb[0].min(rgb[1].min(rgb[2]));
    let lightness = (cmax + cmin).mul_add(0.5, 0.0).min(1.0);
    if same_f32(cmax, cmin) {
        return [0.0, 0.0, lightness];
    }
    let delta = cmax - cmin;
    let saturation = if lightness > 0.5 {
        delta / (2.0 - cmax - cmin)
    } else {
        delta / (cmax + cmin)
    };
    let hue = (if same_f32(cmax, rgb[0]) {
        (rgb[1] - rgb[2]) / delta + if rgb[1] < rgb[2] { 6.0 } else { 0.0 }
    } else if same_f32(cmax, rgb[1]) {
        (rgb[2] - rgb[0]) / delta + 2.0
    } else {
        (rgb[0] - rgb[1]) / delta + 4.0
    }) / 6.0;
    [hue, saturation, lightness]
}

fn hsl_to_rgb([hue, saturation, lightness]: [f32; 3]) -> [f32; 3] {
    let red = (hue * 6.0 - 3.0).abs() - 1.0;
    let green = 2.0 - (hue * 6.0 - 2.0).abs();
    let blue = 2.0 - (hue * 6.0 - 4.0).abs();
    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    [
        (red.clamp(0.0, 1.0) - 0.5) * chroma + lightness,
        (green.clamp(0.0, 1.0) - 0.5) * chroma + lightness,
        (blue.clamp(0.0, 1.0) - 0.5) * chroma + lightness,
    ]
}

fn fract(value: f32) -> f32 {
    value - value.floor()
}
