// SPDX-FileCopyrightText: 2025 Blender Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from Blender Cycles 5.2.2 (source commit d13f752e3b9c4f8c261cda552b1021f8bcc0382c):
// intern/cycles/bvh/octree.cpp, intern/cycles/kernel/bake/bake.h,
// intern/cycles/kernel/sample/sobol_burley.h, intern/cycles/kernel/sample/util.h,
// intern/cycles/util/hash.h, and intern/cycles/kernel/tables.h.

use glam::DVec3;

use crate::error::Result;

const VOLUME_OCTREE_MAX_DEPTH: u32 = 7;
const DENSITY_RESOLUTION: usize = 1 << VOLUME_OCTREE_MAX_DEPTH;
const DENSITY_CELL_COUNT: usize = DENSITY_RESOLUTION * DENSITY_RESOLUTION * DENSITY_RESOLUTION;
const DENSITY_SAMPLES_PER_CELL: u32 = 16;
const ROOT_MIN: f32 = -10_000.0;
const ROOT_MAX: f32 = 10_000.0;
const VOXEL_SIZE: f32 = (ROOT_MAX - ROOT_MIN) / DENSITY_RESOLUTION as f32;
const SPLIT_THRESHOLD: f32 = 1.442;
const LEAF: usize = usize::MAX;

#[derive(Clone, Copy)]
struct OctreeNode {
    sigma_min: f32,
    sigma_max: f32,
    first_child: usize,
}

impl OctreeNode {
    const EMPTY: Self = Self {
        sigma_min: 0.0,
        sigma_max: 0.0,
        first_child: LEAF,
    };
}

pub(crate) struct WorldVolumeOctree {
    nodes: Vec<OctreeNode>,
}

#[derive(Clone, Copy)]
pub(crate) struct WorldVolumeInterval {
    pub(crate) end: f64,
    pub(crate) sigma_min: f32,
    pub(crate) sigma_max: f32,
}

impl WorldVolumeOctree {
    pub(crate) fn build_with_workers<E>(
        worker_count: usize,
        mut make_evaluator: impl FnMut() -> E,
    ) -> Result<(Self, usize)>
    where
        E: FnMut(DVec3) -> Result<(f32, usize)> + Send,
    {
        let mut cells = vec![(f32::MAX, -f32::MAX); DENSITY_CELL_COUNT];
        let worker_count = worker_count.clamp(1, DENSITY_CELL_COUNT);
        let chunk_size = DENSITY_CELL_COUNT.div_ceil(worker_count);
        let max_closures = std::thread::scope(|scope| {
            let mut workers = Vec::with_capacity(worker_count);
            for (chunk_index, chunk) in cells.chunks_mut(chunk_size).enumerate() {
                let offset = chunk_index * chunk_size;
                let mut evaluate = make_evaluator();
                workers.push(
                    std::thread::Builder::new()
                        .stack_size(super::path::WORKER_STACK_BYTES)
                        .spawn_scoped(scope, move || {
                            let mut max_closures = 0;
                            for (index, cell) in chunk.iter_mut().enumerate() {
                                let (extrema, capacity) =
                                    cell_extrema_with_capacity(offset + index, &mut evaluate)?;
                                *cell = extrema;
                                max_closures = max_closures.max(capacity);
                            }
                            Ok::<_, crate::error::PotError>(max_closures)
                        }),
                );
            }

            let mut max_closures = 0;
            let mut worker_error = None;
            for worker in workers {
                let worker_result = match worker {
                    Ok(worker) => worker.join().unwrap_or_else(|_| {
                        Err(crate::error::PotError::new(
                            crate::error::ErrorCode::InternalError,
                            "World volume density worker panicked",
                        ))
                    }),
                    Err(error) => Err(crate::error::PotError::new(
                        crate::error::ErrorCode::InternalError,
                        format!("cannot start a World volume density worker: {error}"),
                    )),
                };
                match worker_result {
                    Ok(capacity) => max_closures = max_closures.max(capacity),
                    Err(error) => {
                        if worker_error.is_none() {
                            worker_error = Some(error);
                        }
                    }
                }
            }
            match worker_error {
                Some(error) => Err(error),
                None => Ok(max_closures),
            }
        })?;

        let mut nodes = vec![OctreeNode::EMPTY];
        build_node(&mut nodes, 0, &cells, 0, [0; 3], DENSITY_RESOLUTION);
        Ok((Self { nodes }, max_closures))
    }

    pub(crate) fn interval(
        &self,
        origin: DVec3,
        direction: DVec3,
        start: f64,
        limit: f64,
    ) -> WorldVolumeInterval {
        let root = self.nodes[0];
        let mut root_near = f64::NEG_INFINITY;
        let mut root_far = f64::INFINITY;
        let mut intersects_root = true;

        for axis in 0..3 {
            let o = origin[axis];
            let d = direction[axis];
            let min = f64::from(ROOT_MIN);
            let max = f64::from(ROOT_MAX);
            if d > 0.0 {
                let near = (min - o) / d;
                let far = (max - o) / d;
                root_near = root_near.max(near);
                root_far = root_far.min(far);
            } else if d < 0.0 {
                let near = (max - o) / d;
                let far = (min - o) / d;
                root_near = root_near.max(near);
                root_far = root_far.min(far);
            } else if o < min || o > max {
                intersects_root = false;
            }
        }

        let position = origin + direction * start;
        let inside_root = intersects_root
            && start >= root_near
            && start < root_far
            && position.x >= f64::from(ROOT_MIN)
            && position.x <= f64::from(ROOT_MAX)
            && position.y >= f64::from(ROOT_MIN)
            && position.y <= f64::from(ROOT_MAX)
            && position.z >= f64::from(ROOT_MIN)
            && position.z <= f64::from(ROOT_MAX);
        if !inside_root {
            let entry = root_near.max(start);
            let end = if intersects_root && root_far > entry && root_near > start {
                entry.min(limit)
            } else {
                limit
            };
            return WorldVolumeInterval {
                end,
                sigma_min: root.sigma_min,
                sigma_max: root.sigma_max,
            };
        }

        let grid_position = [
            (position.x - f64::from(ROOT_MIN)) / f64::from(VOXEL_SIZE),
            (position.y - f64::from(ROOT_MIN)) / f64::from(VOXEL_SIZE),
            (position.z - f64::from(ROOT_MIN)) / f64::from(VOXEL_SIZE),
        ];
        let mut cell_min = [0_usize; 3];
        let mut cell_span = DENSITY_RESOLUTION;
        let mut node_index = 0;

        while self.nodes[node_index].first_child != LEAF {
            let half = cell_span / 2;
            let mut child = 0_usize;
            for axis in 0..3 {
                let boundary = (cell_min[axis] + half) as f64;
                let upper = if direction[axis] > 0.0 {
                    let crossing = (f64::from(ROOT_MIN) + boundary * f64::from(VOXEL_SIZE)
                        - origin[axis])
                        / direction[axis];
                    start >= crossing
                } else if direction[axis] < 0.0 {
                    let crossing = (f64::from(ROOT_MIN) + boundary * f64::from(VOXEL_SIZE)
                        - origin[axis])
                        / direction[axis];
                    start < crossing
                } else {
                    grid_position[axis] >= boundary
                };
                if upper {
                    child |= 1 << axis;
                    cell_min[axis] += half;
                }
            }
            node_index = self.nodes[node_index].first_child + child;
            cell_span = half;
        }

        let mut end = limit;
        for axis in 0..3 {
            let d = direction[axis];
            if d > 0.0 {
                let boundary = f64::from(ROOT_MIN)
                    + (cell_min[axis] + cell_span) as f64 * f64::from(VOXEL_SIZE);
                end = end.min((boundary - origin[axis]) / d);
            } else if d < 0.0 {
                let boundary = f64::from(ROOT_MIN) + cell_min[axis] as f64 * f64::from(VOXEL_SIZE);
                end = end.min((boundary - origin[axis]) / d);
            }
        }

        let node = self.nodes[node_index];
        WorldVolumeInterval {
            end,
            sigma_min: node.sigma_min,
            sigma_max: node.sigma_max,
        }
    }
}

fn cell_extrema_with_capacity(
    offset: usize,
    eval_sigma: &mut impl FnMut(DVec3) -> Result<(f32, usize)>,
) -> Result<((f32, f32), usize)> {
    let x = offset % DENSITY_RESOLUTION;
    let y = (offset / DENSITY_RESOLUTION) % DENSITY_RESOLUTION;
    let z = offset / (DENSITY_RESOLUTION * DENSITY_RESOLUTION);
    let pad = 0.2_f32 * VOXEL_SIZE;
    let padded_size = VOXEL_SIZE + pad * 2.0;
    let cell_min = [x, y, z].map(|index| ROOT_MIN + index as f32 * VOXEL_SIZE - pad);
    let mut sigma_min = f32::MAX;
    let mut sigma_max = -f32::MAX;
    let mut max_closures = 0;

    for sample in 0..DENSITY_SAMPLES_PER_CELL {
        let sample_index = sample + offset as u32 * DENSITY_SAMPLES_PER_CELL;
        let sample_position = sobol_burley_sample_3d(sample_index);
        let sigma = eval_sigma(DVec3::new(
            f64::from(cell_min[0] + sample_position[0] * padded_size),
            f64::from(cell_min[1] + sample_position[1] * padded_size),
            f64::from(cell_min[2] + sample_position[2] * padded_size),
        ))?;
        sigma_min = sigma_min.min(sigma.0);
        sigma_max = sigma_max.max(sigma.0);
        max_closures = max_closures.max(sigma.1);
    }

    Ok(((sigma_min, sigma_max), max_closures))
}

fn build_node(
    nodes: &mut Vec<OctreeNode>,
    node_index: usize,
    cells: &[(f32, f32)],
    depth: u32,
    cell_min: [usize; 3],
    cell_span: usize,
) {
    let mut sigma_min = f32::MAX;
    let mut sigma_max = -f32::MAX;
    for z in cell_min[2]..cell_min[2] + cell_span {
        for y in cell_min[1]..cell_min[1] + cell_span {
            for x in cell_min[0]..cell_min[0] + cell_span {
                let offset = x + DENSITY_RESOLUTION * (y + z * DENSITY_RESOLUTION);
                sigma_min = sigma_min.min(cells[offset].0);
                sigma_max = sigma_max.max(cells[offset].1);
            }
        }
    }
    nodes[node_index].sigma_min = sigma_min;
    nodes[node_index].sigma_max = sigma_max;

    let edge = VOXEL_SIZE * cell_span as f32;
    let bbox_length = (edge * edge + edge * edge + edge * edge).sqrt();
    if depth == VOLUME_OCTREE_MAX_DEPTH || (sigma_max - sigma_min) * bbox_length <= SPLIT_THRESHOLD
    {
        return;
    }

    let first_child = nodes.len();
    nodes[node_index].first_child = first_child;
    nodes.extend([OctreeNode::EMPTY; 8]);
    let child_span = cell_span / 2;
    for child in 0..8 {
        let child_min = [
            cell_min[0] + if child & 1 != 0 { child_span } else { 0 },
            cell_min[1] + if child & 2 != 0 { child_span } else { 0 },
            cell_min[2] + if child & 4 != 0 { child_span } else { 0 },
        ];
        build_node(
            nodes,
            first_child + child,
            cells,
            depth + 1,
            child_min,
            child_span,
        );
    }
}

#[inline]
fn sobol_burley_sample_3d(index: u32) -> [f32; 3] {
    let seed = hash_hp_uint(0);
    let index = reversed_bit_owen(index.reverse_bits(), seed ^ 0xcaa7_26ac);
    [
        sobol_burley(index, 0, seed ^ 0x9e78_e391),
        sobol_burley(index, 1, seed ^ 0x67c3_3241),
        sobol_burley(index, 2, seed ^ 0x78c3_95c5),
    ]
}

#[inline]
fn sobol_burley(mut rev_bit_index: u32, dimension: usize, scramble_seed: u32) -> f32 {
    let mut result = if dimension == 0 {
        rev_bit_index.reverse_bits()
    } else {
        let mut result = 0;
        let mut i = 0;
        while rev_bit_index != 0 {
            let j = rev_bit_index.leading_zeros();
            result ^= SOBOL_BURLEY_TABLE[dimension][i + j as usize];
            i += j as usize + 1;
            rev_bit_index <<= j;
            rev_bit_index <<= 1;
        }
        result
    };
    result = reversed_bit_owen(result, scramble_seed).reverse_bits();
    // Match Cycles' uint_to_float_excl rounding with a single f32 conversion.
    (f64::from(result) / 4_294_967_808.0) as f32
}

#[inline]
fn reversed_bit_owen(mut n: u32, seed: u32) -> u32 {
    n ^= n.wrapping_mul(0x3d20_adea);
    n = n.wrapping_add(seed);
    n = n.wrapping_mul((seed >> 16) | 1);
    n ^= n.wrapping_mul(0x0552_6c56);
    n ^= n.wrapping_mul(0x53a2_2864);
    n
}

#[inline]
fn hash_hp_uint(mut i: u32) -> u32 {
    i ^= i >> 16;
    i = i.wrapping_mul(0x21f0_aaad);
    i ^= i >> 15;
    i = i.wrapping_mul(0xd35a_2d97);
    i ^= i >> 15;
    i ^ 0xe6fe_3beb
}

const SOBOL_BURLEY_TABLE: [[u32; 32]; 4] = [
    [
        0x0000_0001,
        0x0000_0002,
        0x0000_0004,
        0x0000_0008,
        0x0000_0010,
        0x0000_0020,
        0x0000_0040,
        0x0000_0080,
        0x0000_0100,
        0x0000_0200,
        0x0000_0400,
        0x0000_0800,
        0x0000_1000,
        0x0000_2000,
        0x0000_4000,
        0x0000_8000,
        0x0001_0000,
        0x0002_0000,
        0x0004_0000,
        0x0008_0000,
        0x0010_0000,
        0x0020_0000,
        0x0040_0000,
        0x0080_0000,
        0x0100_0000,
        0x0200_0000,
        0x0400_0000,
        0x0800_0000,
        0x1000_0000,
        0x2000_0000,
        0x4000_0000,
        0x8000_0000,
    ],
    [
        0x0000_0001,
        0x0000_0003,
        0x0000_0005,
        0x0000_000f,
        0x0000_0011,
        0x0000_0033,
        0x0000_0055,
        0x0000_00ff,
        0x0000_0101,
        0x0000_0303,
        0x0000_0505,
        0x0000_0f0f,
        0x0000_1111,
        0x0000_3333,
        0x0000_5555,
        0x0000_ffff,
        0x0001_0001,
        0x0003_0003,
        0x0005_0005,
        0x000f_000f,
        0x0011_0011,
        0x0033_0033,
        0x0055_0055,
        0x00ff_00ff,
        0x0101_0101,
        0x0303_0303,
        0x0505_0505,
        0x0f0f_0f0f,
        0x1111_1111,
        0x3333_3333,
        0x5555_5555,
        0xffff_ffff,
    ],
    [
        0x0000_0001,
        0x0000_0003,
        0x0000_0006,
        0x0000_0009,
        0x0000_0017,
        0x0000_003a,
        0x0000_0071,
        0x0000_00a3,
        0x0000_0116,
        0x0000_0339,
        0x0000_0677,
        0x0000_09aa,
        0x0000_1601,
        0x0000_3903,
        0x0000_7706,
        0x0000_aa09,
        0x0001_0117,
        0x0003_033a,
        0x0006_0671,
        0x0009_09a3,
        0x0017_1616,
        0x003a_3939,
        0x0071_7777,
        0x00a3_aaaa,
        0x0117_0001,
        0x033a_0003,
        0x0671_0006,
        0x09a3_0009,
        0x1616_0017,
        0x3939_003a,
        0x7777_0071,
        0xaaaa_00a3,
    ],
    [
        0x0000_0001,
        0x0000_0003,
        0x0000_0004,
        0x0000_000a,
        0x0000_001f,
        0x0000_002e,
        0x0000_0045,
        0x0000_00c9,
        0x0000_011b,
        0x0000_02a4,
        0x0000_079a,
        0x0000_0b67,
        0x0000_101e,
        0x0000_302d,
        0x0000_4041,
        0x0000_a0c3,
        0x0001_f104,
        0x0002_e28a,
        0x0004_57df,
        0x000c_9bae,
        0x0011_a105,
        0x002a_7289,
        0x0079_e7db,
        0x00b6_dba4,
        0x0100_011a,
        0x0300_02a7,
        0x0400_079e,
        0x0a00_0b6d,
        0x1f00_1001,
        0x2e00_3003,
        0x4500_4004,
        0xc900_a00a,
    ],
];

#[cfg(test)]
mod tests {
    use super::{
        DENSITY_CELL_COUNT, DENSITY_RESOLUTION, OctreeNode, WorldVolumeOctree, build_node,
        cell_extrema_with_capacity,
    };
    use glam::DVec3;

    #[test]
    fn sobol_burley_samples_match_cycles_table_and_hash() {
        assert_eq!(
            super::sobol_burley_sample_3d(0).map(f32::to_bits),
            [0x3f50_5455, 0x3ebb_c2e6, 0x3f7d_e3ea]
        );
        assert_eq!(
            super::sobol_burley_sample_3d(1).map(f32::to_bits),
            [0x3d7e_88f3, 0x3f65_bb61, 0x3e8f_2b81]
        );
        assert_eq!(
            super::sobol_burley_sample_3d(65_535).map(f32::to_bits),
            [0x3e60_4818, 0x3f03_982b, 0x3f0e_90d5]
        );
    }

    #[test]
    fn cell_extrema_use_all_sixteen_padded_sobol_samples() -> crate::error::Result<()> {
        let mut count = 0;
        let ((minimum, maximum), _) = cell_extrema_with_capacity(0, &mut |position| {
            count += 1;
            Ok((position.x as f32, 0))
        })?;
        assert_eq!(count, 16);

        let pad = 0.2_f32 * super::VOXEL_SIZE;
        let padded_size = super::VOXEL_SIZE + 2.0 * pad;
        let expected = (0..16)
            .map(|sample| {
                super::ROOT_MIN - pad + super::sobol_burley_sample_3d(sample)[0] * padded_size
            })
            .fold((f32::MAX, -f32::MAX), |(minimum, maximum), value| {
                (minimum.min(value), maximum.max(value))
            });
        assert_eq!(
            (minimum.to_bits(), maximum.to_bits()),
            (expected.0.to_bits(), expected.1.to_bits())
        );
        Ok(())
    }

    #[test]
    fn split_nodes_agree_with_exact_ray_leaf_intervals() {
        let mut cells = vec![(0.0, 0.0); DENSITY_CELL_COUNT];
        let hot_offset = 96 + DENSITY_RESOLUTION * (64 + 64 * DENSITY_RESOLUTION);
        cells[hot_offset] = (1.0, 1.0);
        let mut nodes = vec![OctreeNode::EMPTY];
        build_node(&mut nodes, 0, &cells, 0, [0; 3], DENSITY_RESOLUTION);
        let octree = WorldVolumeOctree { nodes };
        let origin = DVec3::new(
            -10_000.0,
            -10_000.0 + 64.5 * 156.25,
            -10_000.0 + 64.5 * 156.25,
        );
        let direction = DVec3::X;

        let first = octree.interval(origin, direction, 0.0, f64::INFINITY);
        assert_eq!(first.end, 10_000.0);
        assert_eq!((first.sigma_min, first.sigma_max), (0.0, 0.0));

        let second = octree.interval(origin, direction, first.end, f64::INFINITY);
        assert_eq!(second.end, 15_000.0);
        assert_eq!((second.sigma_min, second.sigma_max), (0.0, 0.0));

        let hot = octree.interval(origin, direction, second.end, f64::INFINITY);
        assert_eq!(hot.end, 15_156.25);
        assert_eq!((hot.sigma_min, hot.sigma_max), (1.0, 1.0));

        let outside = octree.interval(
            DVec3::new(-20_000.0, origin.y, origin.z),
            direction,
            0.0,
            f64::INFINITY,
        );
        assert_eq!(outside.end, 10_000.0);
        assert_eq!((outside.sigma_min, outside.sigma_max), (0.0, 1.0));
    }
}
