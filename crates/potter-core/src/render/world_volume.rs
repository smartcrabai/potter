// SPDX-FileCopyrightText: 2025 Blender Foundation
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from Blender Cycles 5.2.2 (source commit d13f752e3b9c4f8c261cda552b1021f8bcc0382c):
// intern/cycles/bvh/octree.cpp, intern/cycles/kernel/bake/bake.h,
// intern/cycles/kernel/integrator/shade_volume.h,
// intern/cycles/kernel/sample/sobol_burley.h, intern/cycles/kernel/sample/util.h,
// intern/cycles/util/hash.h, and intern/cycles/kernel/tables.h.

use glam::DVec3;

use crate::error::Result;
use crate::shader::hash_uint;

const VOLUME_OCTREE_MAX_DEPTH: u32 = 7;
const MANTISSA_BITS: u8 = 23;
const OVERLAP_EXP: f32 = 5e-4;
const DENSITY_RESOLUTION: usize = 1 << VOLUME_OCTREE_MAX_DEPTH;
const DENSITY_CELL_COUNT: usize = DENSITY_RESOLUTION * DENSITY_RESOLUTION * DENSITY_RESOLUTION;
const DENSITY_SAMPLES_PER_CELL: u32 = 16;
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

pub(crate) struct VolumeOctree {
    nodes: Vec<OctreeNode>,
    root_scale: [f32; 3],
    root_translation: [f32; 3],
    split_scale: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct VolumeInterval {
    pub(crate) end: f64,
    pub(crate) sigma_min: f32,
    pub(crate) sigma_max: f32,
}

impl VolumeOctree {
    pub(crate) fn build_with_workers<E>(
        worker_count: usize,
        bounds_min: DVec3,
        bounds_max: DVec3,
        split_scale: f32,
        mut make_evaluator: impl FnMut() -> E,
    ) -> Result<(Self, usize)>
    where
        E: FnMut(DVec3) -> Result<(f32, usize)> + Send,
    {
        let bounds_min_f64 = bounds_min.to_array();
        let bounds_max_f64 = bounds_max.to_array();
        let bounds_min = bounds_min_f64.map(|component| component as f32);
        let bounds_max = bounds_max_f64.map(|component| component as f32);
        let mut root_scale = [0.0; 3];
        let mut root_translation = [0.0; 3];
        let mut voxel_size = [0.0; 3];
        let mut position_to_index_scale = [0.0; 3];
        for axis in 0..3 {
            if !bounds_min_f64[axis].is_finite()
                || !bounds_max_f64[axis].is_finite()
                || !bounds_min[axis].is_finite()
                || !bounds_max[axis].is_finite()
            {
                return Err(crate::error::PotError::invalid_argument(
                    "volume octree bounds must be finite f32 values",
                ));
            }
            let extent = bounds_max[axis] - bounds_min[axis];
            root_scale[axis] = 1.0 / extent;
            root_translation[axis] = -bounds_min[axis] * root_scale[axis] + 1.0;
            voxel_size[axis] = extent / DENSITY_RESOLUTION as f32;
            if !extent.is_finite()
                || extent <= 0.0
                || !root_scale[axis].is_finite()
                || root_scale[axis] <= 0.0
                || !root_translation[axis].is_finite()
                || !voxel_size[axis].is_finite()
                || voxel_size[axis] <= 0.0
            {
                return Err(crate::error::PotError::invalid_argument(
                    "volume octree bounds must have finite positive f32 extents",
                ));
            }
            position_to_index_scale[axis] = 1.0 / voxel_size[axis];
        }

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
                                let (extrema, capacity) = cell_extrema_with_capacity(
                                    offset + index,
                                    bounds_min,
                                    voxel_size,
                                    &mut evaluate,
                                )?;
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
                    Ok(worker) => match worker.join() {
                        Ok(result) => result,
                        Err(_) => Err(crate::error::PotError::new(
                            crate::error::ErrorCode::InternalError,
                            "volume density worker panicked",
                        )),
                    },
                    Err(error) => Err(crate::error::PotError::new(
                        crate::error::ErrorCode::InternalError,
                        format!("cannot start a volume density worker: {error}"),
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
        build_node(
            &mut nodes,
            0,
            &cells,
            0,
            bounds_min,
            bounds_max,
            bounds_min,
            position_to_index_scale,
            split_scale,
        );
        Ok((
            Self {
                nodes,
                root_scale,
                root_translation,
                split_scale,
            },
            max_closures,
        ))
    }

    pub(crate) fn split_scale(&self) -> f32 {
        self.split_scale
    }

    pub(crate) fn trace(
        &self,
        origin: DVec3,
        direction: DVec3,
        start: f64,
        limit: f64,
    ) -> VolumeTrace {
        VolumeTrace::new(self, origin, direction, start, limit)
    }
}

/// Preserves Cycles' initial clamp while monotonically advancing this ray's leaf traversal.
pub(crate) struct VolumeTrace {
    ray_p: [f32; 3],
    ray_d: [f32; 3],
    current_p: [u32; 3],
    octant_mask: u8,
    scale: u8,
    next_scale: u8,
    node_index: usize,
    active_min: f32,
    active_max: f32,
    ray_max: f32,
    root_mode: bool,
}

impl VolumeTrace {
    fn new(octree: &VolumeOctree, origin: DVec3, direction: DVec3, start: f64, limit: f64) -> Self {
        let origin = origin.to_array().map(|component| component as f32);
        let direction = direction.to_array().map(|component| component as f32);
        let start = start as f32;
        let ray_max = (limit as f32).min(f32::MAX);
        let mut ray_d = std::array::from_fn(|axis| direction[axis] * octree.root_scale[axis]);
        let mut local_p = std::array::from_fn(|axis| {
            (origin[axis] + direction[axis] * start) * octree.root_scale[axis]
                + octree.root_translation[axis]
        });
        let mut octant_mask = 0_u8;
        for axis in 0..3 {
            if ray_d[axis] > 0.0 {
                octant_mask |= 1 << axis;
                local_p[axis] = 3.0 - local_p[axis];
            }
            local_p[axis] = local_p[axis].min(1.999_999_9);
            ray_d[axis] = -ray_d[axis].abs();
        }

        let mut trace = Self {
            ray_p: std::array::from_fn(|axis| local_p[axis] - ray_d[axis] * start),
            ray_d,
            current_p: local_p.map(f32::to_bits),
            octant_mask,
            scale: MANTISSA_BITS,
            next_scale: 0,
            node_index: 0,
            active_min: start,
            active_max: ray_max,
            ray_max,
            root_mode: !local_p.iter().all(|component| *component > 1.0),
        };
        if !trace.root_mode {
            trace.locate_leaf(octree);
            trace.active_max = trace.ray_voxel_intersect();
        }
        trace
    }

    /// Returns `None` at ray completion or when Cycles skips the overlapping root tail.
    pub(crate) fn interval(
        &mut self,
        octree: &VolumeOctree,
        start: f64,
        limit: f64,
    ) -> Option<VolumeInterval> {
        let start = start as f32;
        if start >= self.ray_max {
            return None;
        }
        while !self.root_mode && self.active_max < self.ray_max && start >= self.active_max {
            self.active_min = self.active_max;
            if self.next_scale > MANTISSA_BITS {
                // Match Cycles' overlap termination before extending traversal to the root.
                if (self.active_max - self.ray_max).abs() <= OVERLAP_EXP {
                    return None;
                }
                self.root_mode = true;
                self.node_index = 0;
                self.active_max = self.ray_max;
                break;
            }
            self.locate_leaf(octree);
            self.active_max = self.ray_voxel_intersect();
        }
        let node = octree.nodes[if self.root_mode { 0 } else { self.node_index }];
        Some(VolumeInterval {
            end: f64::from(self.active_max).min(limit),
            sigma_min: node.sigma_min,
            sigma_max: node.sigma_max,
        })
    }

    fn locate_leaf(&mut self, octree: &VolumeOctree) {
        let mut node_index = 0;
        let mut scale = MANTISSA_BITS;
        while octree.nodes[node_index].first_child != LEAF {
            scale -= 1;
            let mut child = 0;
            for axis in 0..3 {
                child |= (((self.current_p[axis] >> scale) & 1) as usize) << axis;
            }
            node_index = octree.nodes[node_index].first_child + (child ^ self.octant_mask as usize);
        }
        self.node_index = node_index;
        self.scale = scale;
    }

    fn ray_voxel_intersect(&mut self) -> f32 {
        let mask = u32::MAX << self.scale;
        let bbox_min = self
            .current_p
            .map(|coordinate| f32::from_bits(coordinate & mask));
        let mut intersect_t = [f32::MAX; 3];
        for axis in 0..3 {
            let t = (bbox_min[axis] - self.ray_p[axis]) / self.ray_d[axis];
            if t > self.active_min {
                intersect_t[axis] = t;
            }
        }
        let t_max = intersect_t.into_iter().fold(f32::MAX, f32::min);
        let half_size = 1.0 / (2 << VOLUME_OCTREE_MAX_DEPTH) as f32;
        let next_p = std::array::from_fn(|axis| {
            if intersect_t[axis].partial_cmp(&t_max) == Some(std::cmp::Ordering::Equal) {
                bbox_min[axis] - half_size
            } else {
                self.ray_d[axis] * t_max + self.ray_p[axis]
            }
        });
        let diff = (self.current_p[0] ^ next_p[0].to_bits())
            | (self.current_p[1] ^ next_p[1].to_bits())
            | (self.current_p[2] ^ next_p[2].to_bits());
        self.current_p = next_p.map(f32::to_bits);
        self.next_scale = (32 - diff.leading_zeros()) as u8;
        t_max.min(self.ray_max)
    }
}

fn cell_extrema_with_capacity(
    offset: usize,
    bounds_min: [f32; 3],
    voxel_size: [f32; 3],
    eval_sigma: &mut impl FnMut(DVec3) -> Result<(f32, usize)>,
) -> Result<((f32, f32), usize)> {
    let x = offset % DENSITY_RESOLUTION;
    let y = (offset / DENSITY_RESOLUTION) % DENSITY_RESOLUTION;
    let z = offset / (DENSITY_RESOLUTION * DENSITY_RESOLUTION);
    let pad = voxel_size.map(|size| 0.2_f32 * size);
    let padded_size: [f32; 3] = std::array::from_fn(|axis| voxel_size[axis] + pad[axis] * 2.0);
    let cell_index = [x, y, z];
    let cell_min: [f32; 3] = std::array::from_fn(|axis| {
        bounds_min[axis] + cell_index[axis] as f32 * voxel_size[axis] - pad[axis]
    });
    let mut sigma_min = f32::MAX;
    let mut sigma_max = -f32::MAX;
    let mut max_closures = 0;

    for sample in 0..DENSITY_SAMPLES_PER_CELL {
        let sample_index = sample + offset as u32 * DENSITY_SAMPLES_PER_CELL;
        let sample_position = sobol_burley_sample_3d(sample_index);
        let sigma = eval_sigma(DVec3::new(
            f64::from(cell_min[0] + sample_position[0] * padded_size[0]),
            f64::from(cell_min[1] + sample_position[1] * padded_size[1]),
            f64::from(cell_min[2] + sample_position[2] * padded_size[2]),
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
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
    root_min: [f32; 3],
    position_to_index_scale: [f32; 3],
    split_scale: f32,
) {
    let index_min: [usize; 3] = std::array::from_fn(|axis| {
        let index =
            ((bbox_min[axis] - root_min[axis]) * position_to_index_scale[axis]).round() as i32;
        index.clamp(0, DENSITY_RESOLUTION as i32 - 1) as usize
    });
    let index_max: [usize; 3] = std::array::from_fn(|axis| {
        let index =
            ((bbox_max[axis] - root_min[axis]) * position_to_index_scale[axis]).round() as i32;
        index.clamp(1, DENSITY_RESOLUTION as i32) as usize
    });
    let mut sigma_min = f32::MAX;
    let mut sigma_max = -f32::MAX;
    for z in index_min[2]..index_max[2] {
        for y in index_min[1]..index_max[1] {
            for x in index_min[0]..index_max[0] {
                let offset = x + DENSITY_RESOLUTION * (y + z * DENSITY_RESOLUTION);
                sigma_min = sigma_min.min(cells[offset].0);
                sigma_max = sigma_max.max(cells[offset].1);
            }
        }
    }
    nodes[node_index].sigma_min = sigma_min;
    nodes[node_index].sigma_max = sigma_max;

    let bbox_size = std::array::from_fn::<_, 3, _>(|axis| bbox_max[axis] - bbox_min[axis]);
    if bbox_size.contains(&0.0) {
        return;
    }
    let bbox_length =
        (bbox_size[0] * bbox_size[0] + bbox_size[1] * bbox_size[1] + bbox_size[2] * bbox_size[2])
            .sqrt();
    let split_demand = (sigma_max - sigma_min) * bbox_length * split_scale;
    if split_demand.partial_cmp(&SPLIT_THRESHOLD) != Some(std::cmp::Ordering::Greater)
        || depth >= VOLUME_OCTREE_MAX_DEPTH
    {
        return;
    }

    let first_child = nodes.len();
    nodes[node_index].first_child = first_child;
    nodes.extend([OctreeNode::EMPTY; 8]);
    let center = std::array::from_fn::<_, 3, _>(|axis| {
        let source_sum = bbox_min[axis] + bbox_max[axis];
        0.5 * source_sum
    });
    for child in 0..8 {
        let t = [
            if child & 1 != 0 { 1.0 } else { 0.0 },
            if child & 2 != 0 { 1.0 } else { 0.0 },
            if child & 4 != 0 { 1.0 } else { 0.0 },
        ];
        let child_min =
            std::array::from_fn(|axis| bbox_min[axis] + t[axis] * (center[axis] - bbox_min[axis]));
        let child_max =
            std::array::from_fn(|axis| center[axis] + t[axis] * (bbox_max[axis] - center[axis]));
        build_node(
            nodes,
            first_child + child,
            cells,
            depth + 1,
            child_min,
            child_max,
            root_min,
            position_to_index_scale,
            split_scale,
        );
    }
}

#[inline]
fn sobol_burley_sample_3d(index: u32) -> [f32; 3] {
    sobol_burley_sample_3d_set(index, 0)
}

#[inline]
fn sobol_burley_sample_3d_set(index: u32, dimension_set: u32) -> [f32; 3] {
    let seed = hash_hp_uint(dimension_set);
    let index = reversed_bit_owen(index.reverse_bits(), seed ^ 0xcaa7_26ac);
    [
        sobol_burley(index, 0, seed ^ 0x9e78_e391),
        sobol_burley(index, 1, seed ^ 0x67c3_3241),
        sobol_burley(index, 2, seed ^ 0x78c3_95c5),
    ]
}

#[inline]
pub(crate) fn volume_uniform_2d(index: u32, dimension_set: u32) -> [f32; 2] {
    let seed = hash_hp_uint(dimension_set);
    let index = reversed_bit_owen(index.reverse_bits(), seed ^ 0xf8ad_e99a);
    [
        sobol_burley(index, 0, seed ^ 0xe0aa_af76),
        sobol_burley(index, 1, seed ^ 0x9496_4d4e),
    ]
}

#[inline]
pub(crate) fn volume_uniform_3d(index: u32, dimension_set: u32) -> [f32; 3] {
    sobol_burley_sample_3d_set(index, dimension_set)
}

#[inline]
pub(crate) fn volume_pixel_base(seed: u32, x: u32, source_y: u32, nominal_samples: u32) -> u32 {
    nested_uniform_scramble_base4(morton2d(x, source_y), hash_uint(seed))
        .wrapping_mul(nominal_samples)
}

#[inline]
fn expand_bits(mut value: u32) -> u32 {
    value &= 0x0000_ffff;
    value = (value ^ (value << 8)) & 0x00ff_00ff;
    value = (value ^ (value << 4)) & 0x0f0f_0f0f;
    value = (value ^ (value << 2)) & 0x3333_3333;
    value = (value ^ (value << 1)) & 0x5555_5555;
    value
}

#[inline]
fn morton2d(x: u32, y: u32) -> u32 {
    (expand_bits(x) << 1) | expand_bits(y)
}

#[inline]
fn nested_uniform_scramble_base4(index: u32, seed: u32) -> u32 {
    reversed_bit_owen_base4(index.reverse_bits(), seed).reverse_bits()
}

#[inline]
fn reversed_bit_owen_base4(mut value: u32, seed: u32) -> u32 {
    value ^= value.wrapping_mul(0x3d20_adea);
    value ^= (value >> 1) & (value << 1) & 0x5555_5555;
    value = value.wrapping_add(seed);
    value = value.wrapping_mul((seed >> 16) | 1);
    value ^= (value >> 1) & (value << 1) & 0x5555_5555;
    value ^= value.wrapping_mul(0x0552_6c56);
    value ^= value.wrapping_mul(0x53a2_2864);
    value
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
    uint_to_float_excl(result)
}

#[inline]
fn uint_to_float_excl(value: u32) -> f32 {
    value as f32 * (1.0_f32 / 4_294_967_808.0_f32)
}

#[inline]
pub(crate) fn volume_uniform_1d(sample: u32, dimension: u32) -> f32 {
    let seed = hash_hp_uint(dimension);
    let index = reversed_bit_owen(sample.reverse_bits(), seed ^ 0xbff9_5bfe);
    let value = reversed_bit_owen(index.reverse_bits(), seed ^ 0x635c_77bd).reverse_bits();
    uint_to_float_excl(value)
}

#[inline]
pub(crate) fn volume_rng_scramble(offset: u32, scramble_seed: u32) -> u32 {
    let hash_seed = scramble_seed ^ scramble_seed.wrapping_shl(19);
    hash_hp_uint(offset ^ hash_seed) & !3_u32
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
        DENSITY_CELL_COUNT, DENSITY_RESOLUTION, OctreeNode, VolumeInterval, VolumeOctree,
        VolumeTrace, build_node, cell_extrema_with_capacity,
    };
    use crate::error::{PotError, Result};
    use glam::DVec3;

    fn world_octree(nodes: Vec<OctreeNode>) -> VolumeOctree {
        VolumeOctree {
            nodes,
            root_scale: [1.0 / 20_000.0; 3],
            root_translation: [1.5; 3],
            split_scale: 1.0,
        }
    }

    fn next_interval(
        trace: &mut VolumeTrace,
        octree: &VolumeOctree,
        start: f64,
        limit: f64,
    ) -> Result<VolumeInterval> {
        trace.interval(octree, start, limit).ok_or_else(|| {
            PotError::new(
                crate::error::ErrorCode::InternalError,
                "Source octree test ray unexpectedly completed",
            )
        })
    }

    #[test]
    fn sobol_burley_samples_match_cycles_table_and_hash() {
        assert_eq!(
            super::sobol_burley_sample_3d(0).map(f32::to_bits),
            [0x3f50_5454, 0x3ebb_c2e7, 0x3f7d_e3ea]
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
    fn volume_uniform_2d_and_3d_match_independent_source_goldens() {
        // Independent C++ reference from Blender d13f752 Sobol-Burley sampling.
        let cases_2d: [(u32, u32, [u32; 2]); 6] = [
            (0, 0, [0x3f07_e461, 0x3929_e29f]),
            (1, 0, [0x3ec7_11f0, 0x3f71_bfd8]),
            (17, 1, [0x3f10_faee, 0x3e89_4a8b]),
            (65_535, 3, [0x3f2a_4795, 0x3ef6_df03]),
            (u32::MAX, u32::MAX, [0x3e20_cabd, 0x3f00_584b]),
            (0x8000_0001, 7, [0x3e7f_4ac4, 0x3e35_7e19]),
        ];
        for (index, dimension_set, expected) in cases_2d {
            assert_eq!(
                super::volume_uniform_2d(index, dimension_set).map(f32::to_bits),
                expected
            );
        }

        let cases_3d: [(u32, u32, [u32; 3]); 6] = [
            (0, 0, [0x3f50_5454, 0x3ebb_c2e7, 0x3f7d_e3ea]),
            (1, 0, [0x3d7e_88f3, 0x3f65_bb61, 0x3e8f_2b81]),
            (17, 1, [0x3f70_e0f2, 0x3f5e_004e, 0x3f19_d7ef]),
            (65_535, 3, [0x3d8b_178f, 0x3f60_9da4, 0x3f7d_16a8]),
            (u32::MAX, u32::MAX, [0x3f13_977c, 0x3f26_21f4, 0x3cad_b5c3]),
            (0x8000_0001, 7, [0x3ed2_2806, 0x3f1d_94d1, 0x3f45_ae25]),
        ];
        for (index, dimension_set, expected) in cases_3d {
            assert_eq!(
                super::volume_uniform_3d(index, dimension_set).map(f32::to_bits),
                expected
            );
        }
    }

    #[test]
    fn volume_pixel_base_matches_independent_source_goldens() {
        // Independent C++ reference from Blender d13f752 Morton and base-4 Owen sampling.
        let cases: [(u32, u32, u32, u32, u32); 5] = [
            (0, 0, 0, 1, 0xf2f2_cf06),
            (1, 13, 27, 37, 0xc17f_2c38),
            (0x1234_5678, 65_535, 65_536, 7, 0x462c_7c3a),
            (u32::MAX, u32::MAX, 0x8000_0001, u32::MAX, 0xfbb0_7b99),
            (0x8765_4321, 456, 789, 1_000, 0xa19e_de28),
        ];
        for (seed, x, source_y, nominal_samples, expected) in cases {
            assert_eq!(
                super::volume_pixel_base(seed, x, source_y, nominal_samples),
                expected
            );
        }
    }

    #[test]
    fn uint_to_float_excl_matches_cycles_at_f32_rounding_boundary() {
        let value = 0xffff_da80;
        assert_eq!(super::uint_to_float_excl(value).to_bits(), 0x3f7f_ffd8);
        assert_ne!(
            super::uint_to_float_excl(value).to_bits(),
            ((f64::from(value) / 4_294_967_808.0) as f32).to_bits()
        );
    }

    #[test]
    fn volume_uniform_1d_matches_independent_source_goldens() {
        // Independent C++ goldens: Blender d13f752 bit-reversal only (no byte swap), Owen, hash, float conversion.
        let cases: [(u32, u32, u32); 6] = [
            (0, 0, 0x3e4d_a96e),
            (0, 6, 0x3da7_12fe),
            (1, 6, 0x3f73_214a),
            (511, 22, 0x3ed6_7207),
            (u32::MAX, u32::MAX, 0x3f00_16bc),
            (0x8000_0001, 7, 0x3f19_ed3e),
        ];
        for (sample, dimension, expected_bits) in cases {
            assert_eq!(
                super::volume_uniform_1d(sample, dimension).to_bits(),
                expected_bits
            );
        }
    }

    #[test]
    fn volume_rng_scramble_matches_independent_source_goldens() {
        // Independent C++ goldens from Blender d13f752 hash_hp_seeded_uint.
        let cases: [(u32, u32, u32); 5] = [
            (16, 0xe35f_ad82, 0x1598_2c28),
            (32, 0xe35f_ad82, 0x84cd_41bc),
            (16, 0x8647_ace4, 0xf6fa_14e4),
            (32, 0x8647_ace4, 0xc449_36a0),
            (0, 0, 0xe6fe_3be8),
        ];
        for (offset, seed, expected) in cases {
            assert_eq!(super::volume_rng_scramble(offset, seed), expected);
        }
    }

    #[test]
    fn anisotropic_bake_samples_each_padded_root_axis() -> Result<()> {
        let bounds_min = [-3.0, 2.0, 10.0];
        let voxel_size = [
            8.0 / DENSITY_RESOLUTION as f32,
            2.0 / DENSITY_RESOLUTION as f32,
            4.0 / DENSITY_RESOLUTION as f32,
        ];
        let cell_index = [41, 73, 95];
        let offset = cell_index[0]
            + DENSITY_RESOLUTION * (cell_index[1] + cell_index[2] * DENSITY_RESOLUTION);
        let mut points = Vec::with_capacity(16);
        let mut capacity = 0;
        let (_, max_closures) =
            cell_extrema_with_capacity(offset, bounds_min, voxel_size, &mut |position| {
                points.push(position);
                capacity += 1;
                Ok((position.x as f32, capacity))
            })?;

        assert_eq!(points.len(), 16);
        assert_eq!(max_closures, 16);
        let pad = voxel_size.map(|size| 0.2 * size);
        let padded_size = std::array::from_fn::<_, 3, _>(|axis| voxel_size[axis] + 2.0 * pad[axis]);
        let cell_min = std::array::from_fn::<_, 3, _>(|axis| {
            bounds_min[axis] + cell_index[axis] as f32 * voxel_size[axis] - pad[axis]
        });
        for point in points {
            let coordinates = point.to_array();
            for axis in 0..3 {
                assert!(coordinates[axis] >= f64::from(cell_min[axis]));
                assert!(coordinates[axis] <= f64::from(cell_min[axis] + padded_size[axis]));
            }
        }
        Ok(())
    }

    #[test]
    fn split_nodes_and_outside_root_match_cycles_octree_traversal() -> Result<()> {
        let mut cells = vec![(0.0, 0.0); DENSITY_CELL_COUNT];
        let hot_offset = 96 + DENSITY_RESOLUTION * (64 + 64 * DENSITY_RESOLUTION);
        cells[hot_offset] = (1.0, 1.0);
        let world_min = [-10_000.0; 3];
        let world_max = [10_000.0; 3];
        let position_to_index_scale = [1.0 / 156.25; 3];
        let mut nodes = vec![OctreeNode::EMPTY];
        build_node(
            &mut nodes,
            0,
            &cells,
            0,
            world_min,
            world_max,
            world_min,
            position_to_index_scale,
            1.0,
        );
        let octree = world_octree(nodes);
        let y = -10_000.0 + 64.5 * 156.25;
        let origin = DVec3::new(-10_000.0, y, y);
        let direction = DVec3::X;
        let limit = f64::INFINITY;
        let mut trace = octree.trace(origin, direction, 0.0, limit);

        let clipped = next_interval(&mut trace, &octree, 0.0, 1.0)?;
        assert_eq!(clipped.end, 1.0);
        assert_eq!((clipped.sigma_min, clipped.sigma_max), (0.0, 0.0));
        let first = next_interval(&mut trace, &octree, clipped.end, limit)?;
        assert_eq!(first.end, 9_999.998_046_875);
        assert_eq!((first.sigma_min, first.sigma_max), (0.0, 0.0));
        let second = next_interval(&mut trace, &octree, first.end, limit)?;
        assert_eq!(second.end, 14_999.998_046_875);
        assert_eq!((second.sigma_min, second.sigma_max), (0.0, 0.0));

        let hot = next_interval(&mut trace, &octree, second.end, limit)?;
        assert_eq!(hot.end, 15_156.248_046_875);
        assert_eq!((hot.sigma_min, hot.sigma_max), (1.0, 1.0));

        let mut approaching = octree.trace(DVec3::new(-20_000.0, y, y), direction, 0.0, limit);
        let approaching = next_interval(&mut approaching, &octree, 0.0, limit)?;
        assert_eq!(approaching.end, 9_999.998_046_875);
        assert_eq!((approaching.sigma_min, approaching.sigma_max), (0.0, 0.0));

        let mut outgoing = octree.trace(DVec3::new(20_000.0, y, y), direction, 0.0, 40_000.0);
        let outgoing = next_interval(&mut outgoing, &octree, 0.0, 40_000.0)?;
        assert_eq!(outgoing.end, 40_000.0);
        assert_eq!((outgoing.sigma_min, outgoing.sigma_max), (0.0, 1.0));

        let mut corner = octree.trace(
            DVec3::new(-20_000.0, -20_000.0, y),
            DVec3::new(1.0, 1.0, 0.0),
            0.0,
            limit,
        );
        let corner = next_interval(&mut corner, &octree, 0.0, limit)?;
        assert_eq!(corner.end, 9_999.998_046_875);
        assert_eq!((corner.sigma_min, corner.sigma_max), (0.0, 0.0));

        let grazing_y = -10_000.0 + 65.0 * 156.25;
        let grazing_limit = 19_990.0;
        let mut grazing = octree.trace(
            DVec3::new(-10_000.0, grazing_y, y),
            direction,
            0.0,
            grazing_limit,
        );
        let mut grazing_start = 0.0;
        for _ in 0..=DENSITY_RESOLUTION {
            let Some(interval) = grazing.interval(&octree, grazing_start, grazing_limit) else {
                break;
            };
            assert_eq!(interval.sigma_max, 0.0);
            assert!(interval.end > grazing_start);
            grazing_start = interval.end;
        }
        assert_eq!(grazing_start, grazing_limit);

        let mut stationary_above_root =
            octree.trace(DVec3::new(20_000.0, y, y), DVec3::Y, 0.0, 1.0);
        let stationary_above_root = next_interval(&mut stationary_above_root, &octree, 0.0, 1.0)?;
        assert_eq!(stationary_above_root.end, 1.0);
        assert_eq!(
            (
                stationary_above_root.sigma_min,
                stationary_above_root.sigma_max
            ),
            (0.0, 0.0)
        );

        let mut parallel_outside = octree.trace(
            DVec3::new(15_000.0, 0.0, 20_000.0),
            DVec3::new(0.0, 0.0, -1.0),
            0.0,
            100.0,
        );
        let parallel_outside = next_interval(&mut parallel_outside, &octree, 0.0, 100.0)?;
        assert_eq!(parallel_outside.end, 100.0);
        assert_eq!(
            (parallel_outside.sigma_min, parallel_outside.sigma_max),
            (0.0, 0.0)
        );

        let mut stationary_below_root =
            octree.trace(DVec3::new(-10_000.0, y, y), DVec3::Y, 0.0, 1.0);
        let stationary_below_root = next_interval(&mut stationary_below_root, &octree, 0.0, 1.0)?;
        assert_eq!(stationary_below_root.end, 1.0);
        assert_eq!(
            (
                stationary_below_root.sigma_min,
                stationary_below_root.sigma_max
            ),
            (0.0, 1.0)
        );

        let mut leaving_on_face = octree.trace(DVec3::new(10_000.0, y, y), direction, 0.0, 100.0);
        let leaving_on_face = next_interval(&mut leaving_on_face, &octree, 0.0, 100.0)?;
        assert_eq!(leaving_on_face.end, 100.0);
        assert_eq!(
            (leaving_on_face.sigma_min, leaving_on_face.sigma_max),
            (0.0, 1.0)
        );
        let mut nodes = vec![OctreeNode {
            sigma_min: 0.0,
            sigma_max: 1.0,
            first_child: 1,
        }];
        nodes.extend([OctreeNode::EMPTY; 8]);
        let octree = world_octree(nodes);
        let mut leaving_root = octree.trace(DVec3::new(1.0, y, y), direction, 0.0, 20_000.0);
        let edge_leaf = next_interval(&mut leaving_root, &octree, 0.0, 20_000.0)?;
        assert_eq!(edge_leaf.end, 9_999.000_976_562_5);
        assert_eq!((edge_leaf.sigma_min, edge_leaf.sigma_max), (0.0, 0.0));
        let root_extrema = next_interval(&mut leaving_root, &octree, edge_leaf.end, 20_000.0)?;
        assert_eq!(root_extrema.end, 20_000.0);
        assert_eq!((root_extrema.sigma_min, root_extrema.sigma_max), (0.0, 1.0));

        let root_exit = f32::from_bits(0x4479_fff1);
        let within_limit = f32::from_bits(0x4479_fff4);
        let mut within_tolerance = octree.trace(
            DVec3::new(9_000.0, y, y),
            direction,
            0.0,
            f64::from(within_limit),
        );
        let edge = next_interval(&mut within_tolerance, &octree, 0.0, f64::from(within_limit))?;
        assert_eq!(edge.end, f64::from(root_exit));
        assert!(
            within_tolerance
                .interval(&octree, edge.end, f64::from(within_limit))
                .is_none()
        );

        let outside_limit = f32::from_bits(0x4479_fffa);
        let mut outside_tolerance = octree.trace(
            DVec3::new(9_000.0, y, y),
            direction,
            0.0,
            f64::from(outside_limit),
        );
        let edge = next_interval(
            &mut outside_tolerance,
            &octree,
            0.0,
            f64::from(outside_limit),
        )?;
        assert_eq!(edge.end, f64::from(root_exit));
        let root_extrema = next_interval(
            &mut outside_tolerance,
            &octree,
            edge.end,
            f64::from(outside_limit),
        )?;
        assert_eq!(root_extrema.end, f64::from(outside_limit));
        assert_eq!((root_extrema.sigma_min, root_extrema.sigma_max), (0.0, 1.0));
        Ok(())
    }

    #[test]
    fn anisotropic_root_diagonal_drives_splitting_and_trace() -> Result<()> {
        let bounds_min = [-3.0, 2.0, 10.0];
        let bounds_max = [-2.75, 22.0, 10.25];
        let extent = std::array::from_fn::<_, 3, _>(|axis| bounds_max[axis] - bounds_min[axis]);
        let voxel_size = extent.map(|axis| axis / DENSITY_RESOLUTION as f32);
        let position_to_index_scale = voxel_size.map(|size| 1.0 / size);
        let mut cells = vec![(0.0, 0.0); DENSITY_CELL_COUNT];
        for z in 0..DENSITY_RESOLUTION {
            for y in 0..DENSITY_RESOLUTION {
                for x in DENSITY_RESOLUTION / 2..DENSITY_RESOLUTION {
                    let offset = x + DENSITY_RESOLUTION * (y + z * DENSITY_RESOLUTION);
                    cells[offset] = (0.08, 0.08);
                }
            }
        }

        let mut nodes = vec![OctreeNode::EMPTY];
        build_node(
            &mut nodes,
            0,
            &cells,
            0,
            bounds_min,
            bounds_max,
            bounds_min,
            position_to_index_scale,
            1.0,
        );
        assert_eq!(nodes[0].first_child, 1);
        assert_eq!(nodes.len(), 9);
        for child in 0..8 {
            let sigma = if child & 1 == 0 { 0.0 } else { 0.08 };
            assert_eq!(
                (nodes[1 + child].sigma_min, nodes[1 + child].sigma_max),
                (sigma, sigma)
            );
        }

        let mut coarse_nodes = vec![OctreeNode::EMPTY];
        build_node(
            &mut coarse_nodes,
            0,
            &cells,
            0,
            bounds_min,
            bounds_max,
            bounds_min,
            position_to_index_scale,
            0.5,
        );
        assert_eq!(coarse_nodes.len(), 1);

        let root_scale = extent.map(|axis| 1.0 / axis);
        let root_translation =
            std::array::from_fn(|axis| -bounds_min[axis] * root_scale[axis] + 1.0);
        let octree = VolumeOctree {
            nodes,
            root_scale,
            root_translation,
            split_scale: 1.0,
        };
        let origin = DVec3::new(f64::from(bounds_min[0]), 12.0, 10.125);
        let root_limit = f64::from(extent[0]);
        let mut trace = octree.trace(origin, DVec3::X, 0.0, root_limit);
        let low_half = next_interval(&mut trace, &octree, 0.0, root_limit)?;
        // Source clamps the mirrored root entry to nextafter(2, 0) before the first split-plane hit.
        let clamped_root_entry = f32::from_bits(0x3fff_ffff);
        let first_split_plane = f32::from_bits(0x3fc0_0000);
        let source_first_boundary_t =
            f64::from((first_split_plane - clamped_root_entry) / -4.0_f32);
        assert_eq!(low_half.end, source_first_boundary_t);
        assert_eq!((low_half.sigma_min, low_half.sigma_max), (0.0, 0.0));
        let high_half = next_interval(&mut trace, &octree, low_half.end, root_limit)?;
        let source_root_exit_t = f64::from((1.0_f32 - clamped_root_entry) / -4.0_f32);
        assert_eq!(high_half.end, source_root_exit_t);
        assert_eq!((high_half.sigma_min, high_half.sigma_max), (0.08, 0.08));
        assert!(trace.interval(&octree, high_half.end, root_limit).is_none());
        Ok(())
    }

    #[test]
    fn invalid_root_bounds_fail_before_creating_evaluators() {
        let mut factory_called = false;
        let result = VolumeOctree::build_with_workers(
            1,
            DVec3::ZERO,
            DVec3::new(1.0, 0.0, 1.0),
            1.0,
            || {
                factory_called = true;
                |_position: DVec3| Ok((0.0, 0))
            },
        );
        assert!(result.is_err());
        assert!(!factory_called);
    }
}
