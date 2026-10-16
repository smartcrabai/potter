// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/dualcon/intern/Projections.cpp and Projections.h, SPDX-FileCopyrightText: 2002-2022
//   Blender Authors, GPL-2.0-or-later.

//! Separating-axis triangle/cube tests in integer grid space (`CubeTriangleIsect`).

use super::tables::VERTMAP;

/// Number of separating axes: three cube normals, the triangle normal, and nine edge crosses.
pub(super) const NUM_AXES: usize = 13;

/// Triangle data shared by all cube projections derived from one input triangle.
pub(super) struct TriangleProjection {
    /// Min and max projection of the triangle on each axis.
    pub(super) tri_proj: [[i64; 2]; NUM_AXES],
    /// Unit triangle normal computed in double precision from grid-space vertices.
    pub(super) norm: [f64; 3],
}

#[derive(Clone, Copy, Default)]
pub(super) struct CubeProjection {
    pub(super) origin: i64,
    pub(super) edges: [i64; 3],
    pub(super) min: i64,
    pub(super) max: i64,
}

/// Projection of a cube onto every separating axis.
#[derive(Clone)]
pub(super) struct CubeTriangleIsect {
    pub(super) cube: [CubeProjection; NUM_AXES],
}

fn cross_i64(a: [i64; 3], b: [i64; 3]) -> [i64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn cross_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot_i64(a: [i64; 3], b: [i64; 3]) -> i64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(vector: &mut [f64; 3]) {
    let mut magnitude = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    if magnitude > 0.0 {
        magnitude = magnitude.sqrt();
        vector[0] /= magnitude;
        vector[1] /= magnitude;
        vector[2] /= magnitude;
    }
}

/// `create_projection_axes`: cube normals, triangle normal, then cross products of each cube
/// normal with each triangle edge.
fn projection_axes(tri: &[[i64; 3]; 3]) -> [[i64; 3]; NUM_AXES] {
    let mut axes = [[0_i64; 3]; NUM_AXES];
    axes[0] = [1, 0, 0];
    axes[1] = [0, 1, 0];
    axes[2] = [0, 0, 1];

    let mut tri_edges = [[0_i64; 3]; 3];
    for (index, edge) in tri_edges.iter_mut().enumerate() {
        for axis in 0..3 {
            edge[axis] = tri[(index + 1) % 3][axis] - tri[index][axis];
        }
    }

    axes[3] = cross_i64(tri_edges[0], tri_edges[1]);

    let mut count = 4;
    for edge in &tri_edges {
        for face in 0..3 {
            axes[count] = cross_i64(axes[face], *edge);
            count += 1;
        }
    }
    axes
}

impl TriangleProjection {
    fn new(axes: &[[i64; 3]; NUM_AXES], tri: &[[i64; 3]; 3]) -> Self {
        let edge1 = std::array::from_fn(|axis| tri[1][axis] as f64 - tri[0][axis] as f64);
        let edge2 = std::array::from_fn(|axis| tri[2][axis] as f64 - tri[1][axis] as f64);
        let mut norm = cross_f64(edge1, edge2);
        normalize(&mut norm);

        let mut tri_proj = [[0_i64; 2]; NUM_AXES];
        for (axis, projection) in tri_proj.iter_mut().enumerate() {
            let values = [
                dot_i64(axes[axis], tri[0]),
                dot_i64(axes[axis], tri[1]),
                dot_i64(axes[axis], tri[2]),
            ];
            let mut low = values[0];
            let mut high = values[0];
            for value in &values[1..] {
                low = low.min(*value);
                high = high.max(*value);
            }
            *projection = [low, high];
        }
        Self { tri_proj, norm }
    }
}

impl CubeTriangleIsect {
    /// Constructs the root projection of the grid cube `cube` against triangle `tri`, returning the
    /// inheritable triangle data alongside it. Mirrors the `CubeTriangleIsect(cube, tri, ...)`
    /// constructor.
    pub(super) fn root(cube: [[i64; 3]; 2], tri: [[i64; 3]; 3]) -> (Self, TriangleProjection) {
        let axes = projection_axes(&tri);
        let inherit = TriangleProjection::new(&axes, &tri);

        let mut cube_edges = [[0_i64; 3]; 3];
        for (axis, edge) in cube_edges.iter_mut().enumerate() {
            edge[axis] = cube[1][axis] - cube[0][axis];
        }

        let mut cube_projection = [CubeProjection::default(); NUM_AXES];
        for (axis, projection) in cube_projection.iter_mut().enumerate() {
            let origin = dot_i64(axes[axis], cube[0]);
            let edges = std::array::from_fn(|index| dot_i64(axes[axis], cube_edges[index]));
            let mut max = 0_i64;
            let mut min = 0_i64;
            for vertex in &VERTMAP[1..] {
                let proj = i64::from(vertex[0]) * edges[0]
                    + i64::from(vertex[1]) * edges[1]
                    + i64::from(vertex[2]) * edges[2];
                max = max.max(proj);
                min = min.min(proj);
            }
            *projection = CubeProjection {
                origin,
                edges,
                min,
                max,
            };
        }
        (
            Self {
                cube: cube_projection,
            },
            inherit,
        )
    }

    /// Child projection: same origin, halved extents (`CubeTriangleIsect(CubeTriangleIsect *)`).
    pub(super) fn child(&self) -> Self {
        let mut cube = self.cube;
        for projection in &mut cube {
            for edge in &mut projection.edges {
                *edge >>= 1;
            }
            projection.min >>= 1;
            projection.max >>= 1;
        }
        Self { cube }
    }

    /// Bit `4x + 2y + z` is set when child `(x, y, z)` may overlap the triangle bounds on the three
    /// cube-normal axes (`getBoxMask`).
    pub(super) fn box_mask(&self, inherit: &TriangleProjection) -> u8 {
        let child_len = self.cube[0].edges[0] >> 1;
        let mut bmask = [[false; 2]; 3];
        for (axis, flags) in bmask.iter_mut().enumerate() {
            let mid = self.cube[axis].origin + child_len;
            if mid >= inherit.tri_proj[axis][0] {
                flags[0] = true;
            }
            if mid < inherit.tri_proj[axis][1] {
                flags[1] = true;
            }
        }
        let mut mask = 0_u8;
        let mut count = 0;
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    if bmask[0][x] && bmask[1][y] && bmask[2][z] {
                        mask |= 1 << count;
                    }
                    count += 1;
                }
            }
        }
        mask
    }

    /// Moves the cube origin by `offset` in child-cell units (`shift`).
    pub(super) fn shift(&mut self, offset: [i32; 3]) {
        for projection in &mut self.cube {
            let delta: i64 = (0..3)
                .map(|axis| i64::from(offset[axis]) * projection.edges[axis])
                .sum();
            projection.origin += delta;
        }
    }

    /// Full cube/triangle separating-axis test (`isIntersecting`).
    pub(super) fn is_intersecting(&self, inherit: &TriangleProjection) -> bool {
        for (axis, projection) in self.cube.iter().enumerate() {
            let proj0 = projection.origin + projection.min;
            let proj1 = projection.origin + projection.max;
            if proj0 > inherit.tri_proj[axis][1] || proj1 < inherit.tri_proj[axis][0] {
                return false;
            }
        }
        true
    }

    /// Edge-segment test for a primary cube edge (`isIntersectingPrimary`).
    pub(super) fn is_intersecting_primary(
        &self,
        edge: usize,
        inherit: &TriangleProjection,
    ) -> bool {
        for (axis, projection) in self.cube.iter().enumerate() {
            let proj0 = projection.origin;
            let proj1 = projection.origin + projection.edges[edge];
            if proj0 < proj1 {
                if proj0 > inherit.tri_proj[axis][1] || proj1 < inherit.tri_proj[axis][0] {
                    return false;
                }
            } else if proj1 > inherit.tri_proj[axis][1] || proj0 < inherit.tri_proj[axis][0] {
                return false;
            }
        }
        true
    }

    /// Offset along a primary cube edge where the triangle plane crosses it
    /// (`getIntersectionPrimary`), returned as float32.
    pub(super) fn intersection_primary(&self, edge: usize, inherit: &TriangleProjection) -> f32 {
        let axis = 3;
        let proj0 = self.cube[axis].origin;
        let proj1 = self.cube[axis].origin + self.cube[axis].edges[edge];
        let proj2 = inherit.tri_proj[axis][1];
        let delta = proj1 - proj0;
        let alpha = if delta == 0 {
            0.5_f64
        } else {
            let value = (proj2 - proj0) as f64 / delta as f64;
            if (0.0..=1.0).contains(&value) {
                value
            } else {
                0.5
            }
        };
        alpha as f32
    }
}
