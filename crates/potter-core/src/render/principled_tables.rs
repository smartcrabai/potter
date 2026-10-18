//! Interpolated lookups into the Cycles GGX albedo tables, ported from
//! `intern/cycles/kernel/util/lookup_table.h` at Blender v5.2.2.
//!
//! The data comes from `table_ggx_E`, `table_ggx_Eavg` and `table_ggx_gen_schlick_ior_s` in
//! `intern/cycles/scene/shader.tables`, held in [`super::principled_tables_data`]. Each lookup
//! clamps its coordinates to `[0, 1]`, maps them onto the table's index range, and linearly
//! interpolates between neighbouring entries. Layout is `table[z][y][x]` with `x` fastest.

use super::principled_tables_data::{GGX_E, GGX_E_AVG, GGX_GEN_SCHLICK_IOR_S};

/// Port of `lookup_table_read` (`lookup_1d`) over a table of `n` entries.
fn lookup_1d(table: &[f32], n: usize, x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0) * (n - 1) as f64;
    let index = (x.floor() as usize).min(n - 1);
    let nindex = (index + 1).min(n - 1);
    let t = x - index as f64;
    let d0 = f64::from(table[index]);
    if t == 0.0 {
        return d0;
    }
    let d1 = f64::from(table[nindex]);
    (1.0 - t) * d0 + t * d1
}

/// Port of `lookup_table_read_2D` (`lookup_2d`) over a `xsize` × `ysize` table.
fn lookup_2d(table: &[f32], xsize: usize, ysize: usize, x: f64, y: f64) -> f64 {
    let y = y.clamp(0.0, 1.0) * (ysize - 1) as f64;
    let index = (y.floor() as usize).min(ysize - 1);
    let nindex = (index + 1).min(ysize - 1);
    let t = y - index as f64;
    let row = |r: usize| lookup_1d(&table[r * xsize..(r + 1) * xsize], xsize, x);
    let d0 = row(index);
    if t == 0.0 {
        return d0;
    }
    let d1 = row(nindex);
    (1.0 - t) * d0 + t * d1
}

/// Port of `lookup_table_read_3D` (`lookup_3d`) over a `xsize` × `ysize` × `zsize` table.
fn lookup_3d(
    table: &[f32],
    xsize: usize,
    ysize: usize,
    zsize: usize,
    x: f64,
    y: f64,
    z: f64,
) -> f64 {
    let z = z.clamp(0.0, 1.0) * (zsize - 1) as f64;
    let index = (z.floor() as usize).min(zsize - 1);
    let nindex = (index + 1).min(zsize - 1);
    let t = z - index as f64;
    let plane = xsize * ysize;
    let slice = |p: usize| lookup_2d(&table[p * plane..(p + 1) * plane], xsize, ysize, x, y);
    let d0 = slice(index);
    if t == 0.0 {
        return d0;
    }
    let d1 = slice(nindex);
    (1.0 - t) * d0 + t * d1
}

/// GGX directional albedo `E(mu)` at roughness `rough`, with `mu = dot(N, view)` (`ggx_E`).
pub(crate) fn ggx_energy(rough: f64, mu: f64) -> f64 {
    lookup_2d(&GGX_E, 32, 32, rough, mu)
}

/// GGX average albedo `E_avg` at roughness `rough` (`ggx_Eavg`).
pub(crate) fn ggx_energy_average(rough: f64) -> f64 {
    lookup_1d(&GGX_E_AVG, 32, rough)
}

/// GGX Schlick-IOR scale term at roughness `rough`, `cos_ni`, and
/// `z = sqrt(|(ior - 1) / (ior + 1)|)` (`ggx_gen_schlick_ior_s`).
pub(crate) fn ggx_gen_schlick_ior_s(rough: f64, cos_ni: f64, z: f64) -> f64 {
    lookup_3d(&GGX_GEN_SCHLICK_IOR_S, 16, 16, 16, rough, cos_ni, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggx_energy_at_zero_roughness_and_normal_incidence_is_one() {
        // `GGX_E[31 * 32]` (row `y = 31`, i.e. `mu = 1`, column `x = 0`) is the source's first
        // value of the last row of `table_ggx_E`: `1.000000f`.
        assert!((ggx_energy(0.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ggx_energy_average_at_zero_roughness_is_first_entry() {
        assert_eq!(ggx_energy_average(0.0), f64::from(GGX_E_AVG[0]));
    }

    #[test]
    fn ggx_gen_schlick_ior_s_at_origin_is_first_entry() {
        assert_eq!(
            ggx_gen_schlick_ior_s(0.0, 0.0, 0.0),
            f64::from(GGX_GEN_SCHLICK_IOR_S[0])
        );
    }

    #[test]
    fn ggx_gen_schlick_ior_s_at_far_corner_is_last_entry() {
        // Index 4095 is `x = y = z = 15`, the far corner of the 16 × 16 × 16 table.
        assert_eq!(
            ggx_gen_schlick_ior_s(1.0, 1.0, 1.0),
            f64::from(GGX_GEN_SCHLICK_IOR_S[4095])
        );
    }
}
