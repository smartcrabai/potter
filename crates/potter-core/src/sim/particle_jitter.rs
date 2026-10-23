// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org): `init_mv_jit` and `hammersley_create`
// (`particle_distribute.cc`), `BLI_jitterate1` and `BLI_jitterate2` (`blenlib/intern/jitter_2d.cc`),
// `BLI_array_randomize` and `BLI_hammersley_1d` (`blenlib/intern/rand.cc`).

//! Blender's jitter table for `PART_DISTR_JIT`: `jitlevel` offsets in `[0, 1)`, one pair per
//! emission position of a face. Fewer than 25 pairs use a relaxed regular grid (`init_mv_jit`);
//! more use a Hammersley sequence (`hammersley_create`), which is faster for large counts.

use crate::blender_rng::BlenderRng;

/// Tables from this many pairs up use the Hammersley sequence (`distribute_from_faces`).
const HAMMERSLEY_FROM: usize = 25;
/// Rounds of the `jitterate` relaxation in `init_mv_jit`.
const RELAX_ROUNDS: usize = 4;

/// The `jit` table of `distribute_particles` for `jitlevel` pairs: `init_mv_jit` (or
/// `hammersley_create` for `jitlevel + 1` pairs), then `BLI_array_randomize` over the first
/// `jitlevel` pairs. `seed` is `ParticleSystem.seed`; `amount` is `ParticleSettings.jitter_factor`.
pub(crate) fn jitter_table(jitlevel: usize, seed: u32, amount: f32) -> Vec<[f32; 2]> {
    let mut table = if jitlevel < HAMMERSLEY_FROM {
        init_mv_jit(jitlevel, seed, amount)
    } else {
        hammersley_create(jitlevel + 1, seed, amount)
    };
    table.truncate(jitlevel);
    array_randomize(&mut table, seed);
    table
}

/// `init_mv_jit`: a regular grid with random offsets, relaxed by `BLI_jitterate1` twice and
/// `BLI_jitterate2` once per round.
fn init_mv_jit(num: usize, seed: u32, amount: f32) -> Vec<[f32; 2]> {
    let mut jit = vec![[0.0_f32; 2]; num];
    if num == 0 {
        return jit;
    }
    let count = num as f32;
    let rad1 = 1.0_f32 / count.sqrt();
    let rad2 = 1.0_f32 / count;
    let rad3 = count.sqrt() / count;
    let mut rng = BlenderRng::seed(31_415_926_u32.wrapping_add(num as u32).wrapping_add(seed));
    let mut x = 0.0_f32;
    for (index, pair) in jit.iter_mut().enumerate() {
        // Blender's second coordinate is written at float index `i`, so its row is `i / (2 * num)`.
        let row = (2 * index) as f32 / (2.0 * count);
        pair[0] = x + amount * rad1 * (0.5 - rng.get_float());
        pair[1] = row + amount * rad1 * (0.5 - rng.get_float());
        pair[0] -= pair[0].floor();
        pair[1] -= pair[1].floor();
        x += rad3;
        x -= x.floor();
    }
    for _ in 0..RELAX_ROUNDS {
        jitterate1(&mut jit, rad1);
        jitterate1(&mut jit, rad1);
        jitterate2(&mut jit, rad2);
    }
    jit
}

/// `hammersley_create`: `count` points of the Hammersley sequence, shifted by two random offsets
/// (`BLI_rng_new(31415926 + count + seed)`, double precision) and wrapped to `[0, 1)`.
fn hammersley_create(count: usize, seed: u32, amount: f32) -> Vec<[f32; 2]> {
    let mut rng = BlenderRng::seed(31_415_926_u32.wrapping_add(count as u32).wrapping_add(seed));
    let offset_u = rng.get_double() + f64::from(amount);
    let offset_v = rng.get_double() + f64::from(amount);
    (0..count)
        .map(|index| {
            let radical = radical_inverse(index as u32);
            [
                ((index as f64 / count as f64 + offset_u) % 1.0) as f32,
                ((radical + offset_v) % 1.0) as f32,
            ]
        })
        .collect()
}

/// `BLI_hammersley_1d` (`radical_inverse`): the base-2 van der Corput value of `n`.
fn radical_inverse(mut n: u32) -> f64 {
    let mut value = 0.0_f64;
    let mut place = 0.5_f64;
    while n != 0 {
        if n & 1 == 1 {
            value += place;
        }
        place *= 0.5;
        n >>= 1;
    }
    value
}

/// `BLI_jitterate1`: each pair moves by the summed unit directions to the pairs within `radius`,
/// over the nine wrapped copies of every other pair.
fn jitterate1(jit: &mut [[f32; 2]], radius: f32) {
    let num = jit.len();
    let mut moved = vec![[0.0_f32; 2]; num];
    for i in (0..num).rev() {
        let mut dvecx = 0.0_f32;
        let mut dvecy = 0.0_f32;
        let x = jit[i][0];
        let y = jit[i][1];
        for j in (0..num).rev() {
            if i == j {
                continue;
            }
            let mut vecx = jit[j][0] - x - 1.0;
            let mut vecy = jit[j][1] - y - 1.0;
            for _ in 0..3 {
                repel(&mut dvecx, &mut dvecy, vecx, vecy, radius);
                vecx += 1.0;
                repel(&mut dvecx, &mut dvecy, vecx, vecy, radius);
                vecx += 1.0;
                repel(&mut dvecx, &mut dvecy, vecx, vecy, radius);
                vecx -= 2.0;
                vecy += 1.0;
            }
        }
        moved[i] = wrap_pair(x - dvecx / 18.0, y - dvecy / 18.0);
    }
    jit.copy_from_slice(&moved);
}

/// `BLI_jitterate2`: each pair moves by the summed offsets to the pairs within `radius`, over the
/// same nine wrapped copies, weighted by the offset itself.
fn jitterate2(jit: &mut [[f32; 2]], radius: f32) {
    let num = jit.len();
    let mut moved = vec![[0.0_f32; 2]; num];
    for i in (0..num).rev() {
        let mut dvecx = 0.0_f32;
        let mut dvecy = 0.0_f32;
        let x = jit[i][0];
        let y = jit[i][1];
        for j in (0..num).rev() {
            if i == j {
                continue;
            }
            let mut vecx = jit[j][0] - x - 1.0;
            let mut vecy = jit[j][1] - y - 1.0;
            weigh(&mut dvecx, vecx, radius);
            vecx += 1.0;
            weigh(&mut dvecx, vecx, radius);
            vecx += 1.0;
            weigh(&mut dvecx, vecx, radius);
            weigh(&mut dvecy, vecy, radius);
            vecy += 1.0;
            weigh(&mut dvecy, vecy, radius);
            vecy += 1.0;
            weigh(&mut dvecy, vecy, radius);
        }
        moved[i] = wrap_pair(x - dvecx / 2.0, y - dvecy / 2.0);
    }
    jit.copy_from_slice(&moved);
}

/// One `BLI_jitterate1` neighbour: adds the unit direction of `(vecx, vecy)` when both components
/// are within `radius` and the distance is inside it.
fn repel(dvecx: &mut f32, dvecy: &mut f32, vecx: f32, vecy: f32, radius: f32) {
    if vecx.abs() < radius && vecy.abs() < radius {
        let mut len = (vecx * vecx + vecy * vecy).sqrt();
        if len > 0.0 && len < radius {
            len /= radius;
            *dvecx += vecx / len;
            *dvecy += vecy / len;
        }
    }
}

/// One `BLI_jitterate2` neighbour component: adds `value * radius` when `value` is within `radius`.
fn weigh(sum: &mut f32, value: f32, radius: f32) {
    if value.abs() < radius {
        *sum += value * radius;
    }
}

/// `x - floor(x)` on both components, as Blender wraps each relaxed pair.
fn wrap_pair(x: f32, y: f32) -> [f32; 2] {
    [x - x.floor(), y - y.floor()]
}

/// `BLI_array_randomize` over pairs: a Fisher-Yates shuffle from `BLI_rng_seed(seed)`.
fn array_randomize(jit: &mut [[f32; 2]], seed: u32) {
    let num = jit.len();
    if num <= 1 {
        return;
    }
    let mut rng = BlenderRng::seed(seed);
    for i in (0..num).rev() {
        let j = (rng.get_uint32() % num as u32) as usize;
        if i != j {
            jit.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::jitter_table;

    /// Asserts each pair matches Blender's value to `1.0e-6`.
    fn assert_table(actual: &[[f32; 2]], expected: &[[f32; 2]]) {
        assert_eq!(actual.len(), expected.len());
        for (pair, reference) in actual.iter().zip(expected) {
            assert!(
                (pair[0] - reference[0]).abs() < 1.0e-6,
                "{pair:?} != {reference:?}"
            );
            assert!(
                (pair[1] - reference[1]).abs() < 1.0e-6,
                "{pair:?} != {reference:?}"
            );
        }
    }

    // Blender 5.2.2 dumped these with `tests/fixtures/particle_texture/dump_jitter_table.py`: a
    // unit-square quad with `userjit` equal to the count and no random placement puts particle `p`
    // on jitter pair `p`, so its location is the pair.

    #[test]
    fn regular_jitter_table_matches_blender_for_three_pairs() {
        assert_table(
            &jitter_table(3, 3, 1.0),
            &[
                [0.051_146_038, 0.782_502_06],
                [0.097_578_16, 0.480_573_95],
                [0.598_415_3, 0.140_965_83],
            ],
        );
    }

    #[test]
    fn regular_jitter_table_scales_offsets_by_jitter_factor() {
        assert_table(
            &jitter_table(7, 3, 0.5),
            &[
                [0.511_432_2, 0.523_592_35],
                [0.016_074_533, 0.986_256_7],
                [0.286_276_22, 0.796_595_04],
                [0.758_639_9, 0.248_708_58],
                [0.131_656_74, 0.393_380_22],
                [0.419_655_62, 0.134_697_88],
                [0.880_650_4, 0.660_609_9],
            ],
        );
    }

    #[test]
    fn hammersley_table_matches_blender_for_thirty_pairs() {
        assert_table(
            &jitter_table(30, 3, 1.0)[..12],
            &[
                [0.538_741_6, 0.434_712_4],
                [0.474_225_37, 0.184_712_41],
                [0.570_999_7, 0.809_712_4],
                [0.022_612_525, 0.403_462_38],
                [0.732_289_97, 0.840_962_4],
                [0.506_483_44, 0.934_712_4],
                [0.280_677_05, 0.372_212_44],
                [0.829_064_13, 0.215_962_41],
                [0.087_128_65, 0.278_462_4],
                [0.216_160_91, 0.122_212_395],
                [0.345_193_18, 0.247_212_41],
                [0.409_709_27, 0.497_212_4],
            ],
        );
    }
}
