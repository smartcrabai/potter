/// Cycles' Rec.709 blackbody polynomial (`intern/cycles/kernel/svm/math_util.h`).
///
/// This returns the raw kernel color before `svm_node_blackbody` converts Rec.709 to the render
/// working space and clamps negative channels.
#[must_use]
pub(crate) fn blackbody_rgb(temperature_kelvin: f32) -> [f32; 3] {
    const RED: [[f32; 3]; 7] = [
        [1_619.191, -0.002_050_109_2, 5.029_958],
        [2_488.454_7, -0.001_113_309_1, 3.226_215_4],
        [3_341.432, -0.000_486_551_2, 1.764_867_7],
        [4_094.617_4, -0.000_127_446_58, 0.725_731_6],
        [4_670.280_4, 2.912_582e-05, 0.126_703_44],
        [4_595.092, 2.874_956_4e-05, 0.150_345_02],
        [3_787.174_5, 9.359_078e-06, 0.399_075_87],
    ];
    const GREEN: [[f32; 3]; 7] = [
        [-488.999_75, 0.000_604_330_75, -0.075_580_75],
        [-755.994_28, 0.000_316_730_1, 0.478_306_14],
        [-1_023.639_8, 0.000_120_223_47, 0.936_662_3],
        [-1_265.713_1, 4.873_409e-06, 1.270_545],
        [-1_425.293_3, -4.011_504_3e-05, 1.439_727_8],
        [-1_175.548_2, -2.163_780_5e-05, 1.304_080_2],
        [-500.799_57, -4.598_32e-06, 1.090_987_7],
    ];
    const BLUE: [[f32; 4]; 7] = [
        [
            5.969_453_1e-11,
            -4.857_429e-08,
            -9.706_223e-05,
            -0.004_079_361_5,
        ],
        [
            2.404_303_7e-11,
            5.550_210_7e-08,
            -0.000_198_503_71,
            0.028_931_286,
        ],
        [
            -1.409_497_32e-11,
            1.898_789_7e-07,
            -0.000_356_632_84,
            0.091_076_78,
        ],
        [
            -3.614_608_7e-11,
            2.848_220_1e-07,
            -0.000_493_211_3,
            0.156_723_44,
        ],
        [
            -1.970_757_4e-11,
            1.753_593_5e-07,
            -0.000_250_542_82,
            -0.022_278_327,
        ],
        [
            -1.619_979_5e-13,
            -1.642_16e-08,
            0.000_386_216_27,
            -0.738_077_4,
        ],
        [
            6.726_503e-13,
            -2.730_788e-08,
            0.000_424_098_26,
            -0.752_335_69,
        ],
    ];

    if temperature_kelvin >= 12_000.0 {
        return [0.826_295_5, 0.994_508_1, 1.566_307_7];
    }
    if temperature_kelvin < 800.0 {
        return [5.413_294_3, -0.203_193_9, -0.082_253_52];
    }

    let index = if temperature_kelvin >= 6_365.0 {
        6
    } else if temperature_kelvin >= 3_315.0 {
        5
    } else if temperature_kelvin >= 1_902.0 {
        4
    } else if temperature_kelvin >= 1_449.0 {
        3
    } else if temperature_kelvin >= 1_167.0 {
        2
    } else {
        usize::from(temperature_kelvin >= 965.0)
    };

    let inverse = 1.0 / temperature_kelvin;
    let [r0, r1, r2] = RED[index];
    let [g0, g1, g2] = GREEN[index];
    let [b0, b1, b2, b3] = BLUE[index];
    [
        r0 * inverse + r1 * temperature_kelvin + r2,
        g0 * inverse + g1 * temperature_kelvin + g2,
        ((b0 * temperature_kelvin + b1) * temperature_kelvin + b2) * temperature_kelvin + b3,
    ]
}
