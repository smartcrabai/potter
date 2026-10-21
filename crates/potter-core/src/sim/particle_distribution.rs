// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org): `distribute_particles_on_dm` and
// `distribute_binary_search` (`particle_distribute.cc`), for PART_FROM_VERT.

//! Blender's emitter element assignment for vertex emission.
//!
//! `distribute_particles` gives each vertex a weight (uniform here: a density group is not
//! modelled), builds the cumulative weights, and assigns particle `p` a vertex. With `use_emit_random`
//! (`PART_TRAND`, Blender's default) the position is `BLI_rng_get_float` on a generator seeded from
//! `31415926 + seed`, one draw per particle; without it the particles sit at evenly spaced
//! positions along the cumulative weights.

use crate::blender_rng::BlenderRng;

/// Blender's `distribute_binary_search`: the first cumulative weight that reaches `value`.
fn binary_search(sum: &[f32], value: f32) -> usize {
    let mut low = 0;
    let mut high = sum.len() - 1;
    if high == low {
        return low;
    }
    if sum[low] >= value {
        return low;
    }
    if sum[high - 1] < value {
        return high;
    }
    while low < high {
        let mid = (low + high) / 2;
        if sum[mid] >= value && sum[mid - 1] < value {
            return mid;
        }
        if sum[mid] > value {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    low
}

/// The vertex index that each of the `count` particles is emitted from (`particle_element`).
///
/// `vertex_count` is the number of emitting vertices. `system_seed` is `ParticleSystem.seed`.
pub(crate) fn vertex_elements(
    vertex_count: usize,
    count: usize,
    system_seed: u32,
    random: bool,
) -> Vec<usize> {
    if vertex_count == 0 || count == 0 {
        return Vec::new();
    }
    // Every vertex has the same weight `1 / min(vertices, particles)`.
    let weight = 1.0_f32 / (vertex_count.min(count) as f32);
    let mut total_weight = 0.0_f32;
    for _ in 0..vertex_count {
        total_weight += weight;
    }
    let inverse_total = 1.0_f32 / total_weight;
    // Blender drops an element whose addition does not change the sum (`element_map`).
    let mut element_sum: Vec<f32> = Vec::with_capacity(vertex_count);
    let mut element_map: Vec<usize> = Vec::with_capacity(vertex_count);
    element_sum.push(weight * inverse_total);
    element_map.push(0);
    for index in 1..vertex_count {
        let next = element_sum[element_sum.len() - 1] + weight * inverse_total;
        if next > element_sum[element_sum.len() - 1] {
            element_sum.push(next);
            element_map.push(index);
        }
    }
    let mapped = element_sum.len();
    let mut elements = Vec::with_capacity(count);
    if random {
        // `BLI_rng_new_srandom(31415926 + psys->seed)`, one `BLI_rng_get_float` per particle.
        let mut rng = BlenderRng::seed_random(31_415_926_u32.wrapping_add(system_seed));
        for _ in 0..count {
            let position = rng.get_float() * element_sum[mapped - 1];
            elements.push(element_map[binary_search(&element_sum, position)]);
        }
    } else {
        let step = if count < 2 {
            0.5_f64
        } else {
            1.0_f64 / count as f64
        };
        let mut position = if count < mapped {
            0.5_f64 / mapped as f64
        } else {
            step * 0.5
        };
        let mut index = 0;
        for _ in 0..count {
            while index < mapped - 1 && position > f64::from(element_sum[index]) {
                index += 1;
            }
            elements.push(element_map[index]);
            position += step;
        }
    }
    elements
}

#[cfg(test)]
mod tests {
    use super::vertex_elements;

    #[test]
    fn one_vertex_per_particle_when_counts_match_without_randomness() {
        // Evenly spaced positions over equal weights hit each vertex once, in order.
        assert_eq!(vertex_elements(4, 4, 0, false), vec![0, 1, 2, 3]);
    }

    #[test]
    fn every_particle_names_a_vertex() {
        let elements = vertex_elements(8, 20, 7, true);
        assert_eq!(elements.len(), 20);
        assert!(elements.iter().all(|index| *index < 8));
    }
}
