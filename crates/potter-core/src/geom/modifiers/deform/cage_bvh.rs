//! Blender's cage ray casts for Mesh Deform's harmonic bind, ported in `f32` from the pinned 5.2.2 source.
//!
//! The chain is what `meshlaplacian.cc` reads through `mdb->cagemesh_cache` and `mdb->bvhtree`:
//!
//! - corner triangles: `Mesh::tessellate` (`geom/mod.rs`), the one Blender loop-triangle implementation
//!   (`mesh_tessellate.cc`). This module takes them as input and does not triangulate. A quad splits on
//!   the `(0, 2)` diagonal unless `is_quad_flip_v3_first_third_fast` (`math_geom.cc`) says it folds, in
//!   which case it splits on `(1, 3)`.
//! - the tree: `KdopTree` in `geom::bvh`, built over the corner triangles in triangle order. Leaf
//!   `i` is corner triangle `i`, as `bvhtree_new_common` (`bvhutils.cc`) builds Blender's tree.
//! - the ray cast: `closest_hit` runs `KdopTree::ray_cast` (`BLI_bvhtree_ray_cast_ex`) with the
//!   callback `harmonic_ray_callback` (`meshlaplacian.cc:939-983`). The callback stores
//!   `dist / vec_length` in `hit.dist`, but the tree prunes with absolute slab distances. That
//!   mismatch decides which triangle the bind sees, so it is reproduced as written.
//!
//! The inside rule is `meshdeform_inside_cage` (`meshlaplacian.cc:1060-1082`).

use crate::{
    error::{PotError, Result},
    geom::bvh::{KdopTree, Ray, RayHit, Vec3, cross, dot, kdop_bounds, normalize, sub},
};

/// `isect_mdef.lambda` before the ray test (`meshdeform_ray_tree_intersect`).
const LAMBDA_START: f32 = 1.0e10;
/// `MESHDEFORM_OFFSET[1..=6]` (`meshlaplacian.cc:852-860`).
const RAY_OFFSETS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

/// Polygon normal as `face_normal` in `harmonic.rs` (quads: cross of diagonals; triangles: edge cross;
/// larger polygons: Newell's method), normalized.
fn polygon_normal(corners: &[Vec3]) -> Vec3 {
    let normal = match corners.len() {
        3 => cross(sub(corners[0], corners[1]), sub(corners[1], corners[2])),
        4 => cross(sub(corners[0], corners[2]), sub(corners[1], corners[3])),
        count => {
            let mut normal = [0.0_f32; 3];
            for index in 0..count {
                let current = corners[index];
                let previous = corners[(index + count - 1) % count];
                normal[0] += (previous[1] - current[1]) * (previous[2] + current[2]);
                normal[1] += (previous[2] - current[2]) * (previous[0] + current[0]);
                normal[2] += (previous[0] - current[0]) * (previous[1] + current[1]);
            }
            normal
        }
    };
    let (normal, length) = normalize(normal);
    if length <= 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        normal
    }
}

/// The first cage triangle a segment meets (`MeshDeformIsect`).
#[derive(Clone, Copy)]
pub(super) struct CageHit {
    pub(super) triangle: usize,
    /// `isec->lambda`: the hit fraction along the segment (the normalised distance).
    pub(super) lambda: f32,
    /// `isec->isect`: `dot(face normal, dir) <= 0`.
    pub(super) facing: bool,
}

/// The cage as Blender's `cagemesh_cache` and `bvhtree` see it during `harmonic_coordinates_bind`.
pub(super) struct CageRayCache {
    points: Vec<Vec3>,
    corner_tris: Vec<[usize; 3]>,
    tri_face: Vec<usize>,
    face_normals: Vec<Vec3>,
    tree: KdopTree,
}

impl CageRayCache {
    /// `points` are cage vertices in cage space. `faces` are polygons as vertex indices, in mesh
    /// order, with at least three vertices each. `corner_tris` are their loop triangles as vertex
    /// indices, in Blender's order (`Mesh::tessellate`): each polygon contributes `len - 2` triangles.
    pub(super) fn new(
        points: &[Vec3],
        faces: &[Vec<usize>],
        corner_tris: Vec<[usize; 3]>,
    ) -> Result<Self> {
        let mut tri_face = Vec::with_capacity(corner_tris.len());
        let mut face_normals = Vec::with_capacity(faces.len());
        for (face_index, face) in faces.iter().enumerate() {
            tri_face.extend(std::iter::repeat_n(face_index, face.len() - 2));
            let corners: Vec<Vec3> = face.iter().map(|&vertex| points[vertex]).collect();
            face_normals.push(polygon_normal(&corners));
        }
        if tri_face.len() != corner_tris.len() {
            return Err(PotError::invalid_operation(
                "mesh deform cage triangles do not match its polygons",
            ));
        }
        // Mesh Deform binds on `mdb->bvhdata = cagemesh->bvh_corner_tris()` (`meshlaplacian.cc:1596-1597`),
        // whose tree is `create_tree_from_tris` (`bvhutils.cc:510-520`): `bvhtree_new_common`
        // (`bvhutils.cc:431-436`) = `BLI_bvhtree_new(n, 0.0, 2, 6)`, the epsilon clamped to `FLT_EPSILON`
        // (`BLI_kdopbvh.cc:870`), with each corner triangle inserted as three points. Leaf `i` is corner triangle `i`.
        let leaf_bv: Vec<_> = corner_tris
            .iter()
            .map(|t| kdop_bounds(&[points[t[0]], points[t[1]], points[t[2]]], f32::EPSILON, 6))
            .collect();
        Ok(Self {
            points: points.to_vec(),
            corner_tris,
            tri_face,
            face_normals,
            tree: KdopTree::new(leaf_bv, 2, 6),
        })
    }

    /// The polygon that owns a cage triangle (`mdb->cagemesh_cache.tri_faces`).
    pub(super) fn triangle_face(&self, triangle: usize) -> usize {
        self.tri_face[triangle]
    }

    /// The closest-hit query of `meshdeform_ray_tree_intersect` with `harmonic_ray_callback`.
    pub(super) fn closest_hit(&self, start: Vec3, end: Vec3) -> Option<CageHit> {
        let (dir, vec_length) = normalize(sub(end, start));
        let mut lambda = LAMBDA_START;
        let mut facing = false;
        let mut hit = RayHit::open();
        let index = self.tree.ray_cast(
            start,
            dir,
            &mut hit,
            &mut |triangle: usize, ray: &Ray, record: &mut RayHit| {
                let [a, b, c] = self.corner_tris[triangle];
                let Some(dist) = ray.watertight(self.points[a], self.points[b], self.points[c])
                else {
                    return;
                };
                if dist > vec_length {
                    return;
                }
                let normal = self.face_normals[self.tri_face[triangle]];
                let normalized = dist / vec_length;
                if normalized < record.dist {
                    record.index = triangle as isize;
                    record.dist = normalized;
                    facing = dot(normal, ray.dir) <= 0.0;
                    lambda = normalized;
                }
            },
        );
        (index != -1).then_some(CageHit {
            triangle: index as usize,
            lambda,
            facing,
        })
    }

    /// `meshdeform_inside_cage` (`meshlaplacian.cc:1060-1082`): six rays of length `max - min + 1` per axis
    /// from `point` (cage space); inside when the first hit of any ray faces away from it.
    pub(super) fn point_inside(&self, point: Vec3, min: Vec3, max: Vec3) -> bool {
        let extent = [
            max[0] - min[0] + 1.0,
            max[1] - min[1] + 1.0,
            max[2] - min[2] + 1.0,
        ];
        RAY_OFFSETS.iter().any(|offset| {
            let outside =
                [0_usize, 1, 2].map(|axis| point[axis] + extent[axis] * offset[axis] as f32);
            matches!(self.closest_hit(point, outside), Some(hit) if !hit.facing)
        })
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "small cage fixtures")]

    use super::*;

    // Corner order is `Mesh::box_mesh` (`geom/mod.rs`, `BOX_CORNER_SIGNS`, `BOX_FACES`): corner `i` has sign
    // vector `BOX_CORNER_SIGNS[i]` (x, y, z in {-1, +1}) with `i = 4*[x>0] + 2*[y>0] + [z>0]`. Blender's
    // `primitive_cube_add(size=2)` has the same corner order and face table; checked headless (vertex `b<i>`
    // at sign vector `i`; its polygons and `calc_loop_triangles()` equal the lists below).
    //
    // Derivation without Blender: `ASYMMETRIC` is the cage at its bind positions in corner order, the array of
    // `asymmetric_mesh_operations` in `tests/pr18f_deform_bind.rs`. `BOX_FACES` below is the `Mesh::box_mesh`
    // face table. `CORNER_TRIS` applies the quad rule in its doc comment to these positions. Triangle indices
    // count `CORNER_TRIS` in order.
    /// The p2 cage at its asymmetric bind positions, in `Mesh::box_mesh` corner order.
    const ASYMMETRIC: [Vec3; 8] = [
        [-2.2, -1.8, -2.0],
        [-1.8, -1.9, 2.0],
        [-2.0, 1.7, -1.8],
        [-2.1, 1.6, 1.7],
        [2.1, -1.7, -2.1],
        [2.1, -1.7, 2.2],
        [2.1, 1.9, -2.2],
        [2.2, 1.8, 2.3],
    ];
    /// `Mesh::box_mesh` polygons (`BOX_FACES` in `geom/mod.rs`): faces x-, y+, x+, y-, z-, z+ in that order.
    const BOX_FACES: [[usize; 4]; 6] = [
        [0, 1, 3, 2],
        [2, 3, 7, 6],
        [6, 7, 5, 4],
        [4, 5, 1, 0],
        [2, 6, 4, 0],
        [7, 3, 1, 5],
    ];
    /// The padded grid of that bind in Blender 5.2.2 (precision 2, size 6).
    const GRID_MIN: Vec3 = [-6.975, -6.975, -6.925];
    const GRID_MAX: Vec3 = [6.975, 6.975, 7.024_999_6];
    /// Source vertex 0 and vertex 1 in cage space (Blender 5.2.2 bind), `f32`.
    const VERTEX0: Vec3 = [0.348_456_95, -2.196_758_7, 0.636_363_6];
    const VERTEX1: Vec3 = [0.652_235_5, -1.927_082, 0.636_363_6];

    /// Loop triangles of `BOX_FACES` in face order, at the `ASYMMETRIC` positions. A quad with corners
    /// `c0..c3` (face order) splits as `loop_triangles` (`geom/triangulate.rs`, Blender `mesh_tessellate.cc`):
    /// with `s = dot((c1-c0) x (c2-c0), (c3-c0) x (c2-c0))`, `s > 0` gives `[0,1,3], [1,2,3]` (1-3 diagonal),
    /// otherwise `[0,1,2], [0,2,3]` (0-2 diagonal). At these positions `s` is about -170 to -290 for every face,
    /// so every face splits on 0-2. Face `f` owns triangles `2f` and `2f+1`:
    ///
    /// - face 0 `[0,1,3,2]`: `[0,1,3]` (0), `[0,3,2]` (1)
    /// - face 1 `[2,3,7,6]`: `[2,3,7]` (2), `[2,7,6]` (3)
    /// - face 2 `[6,7,5,4]`: `[6,7,5]` (4), `[6,5,4]` (5)
    /// - face 3 `[4,5,1,0]`: `[4,5,1]` (6), `[4,1,0]` (7)
    /// - face 4 `[2,6,4,0]`: `[2,6,4]` (8), `[2,4,0]` (9)
    /// - face 5 `[7,3,1,5]`: `[7,3,1]` (10), `[7,1,5]` (11)
    const CORNER_TRIS: [[usize; 3]; 12] = [
        [0, 1, 3],
        [0, 3, 2],
        [2, 3, 7],
        [2, 7, 6],
        [6, 7, 5],
        [6, 5, 4],
        [4, 5, 1],
        [4, 1, 0],
        [2, 6, 4],
        [2, 4, 0],
        [7, 3, 1],
        [7, 1, 5],
    ];

    fn cage() -> CageRayCache {
        let faces: Vec<Vec<usize>> = BOX_FACES.iter().map(|face| face.to_vec()).collect();
        CageRayCache::new(&ASYMMETRIC, &faces, CORNER_TRIS.to_vec())
            .expect("the box cage is consistent")
    }

    #[test]
    fn vertex0_plus_y_ray_is_decided_by_the_exit_face() {
        // Blender's culling: the entry triangle is pruned by the normalised exit distance. The root's
        // children are visited right-first (its main axis is z, orthogonal to the ray), so the right subtree,
        // which holds triangle 2, records the exit at 0.264 before the left subtree (box entry 0.297) is
        // tested. The first triangle the bind accepts is the back-facing exit face, triangle 2, and vertex 0
        // counts as inside.
        let cage = cage();
        let length = GRID_MAX[1] - GRID_MIN[1] + 1.0;
        let outside = [VERTEX0[0], VERTEX0[1] + length, VERTEX0[2]];
        let hit = cage
            .closest_hit(VERTEX0, outside)
            .expect("the +y ray meets the cage");
        assert_eq!(hit.triangle, 2);
        assert!(!hit.facing);
    }

    #[test]
    fn vertex1_plus_y_ray_is_decided_by_the_entry_face() {
        let cage = cage();
        let length = GRID_MAX[1] - GRID_MIN[1] + 1.0;
        let outside = [VERTEX1[0], VERTEX1[1] + length, VERTEX1[2]];
        let hit = cage
            .closest_hit(VERTEX1, outside)
            .expect("the +y ray meets the cage");
        assert_eq!(hit.triangle, 6);
        assert!(hit.facing);
    }

    #[test]
    fn inside_rule_matches_blender_for_vertices_0_and_1() {
        let cage = cage();
        assert!(cage.point_inside(VERTEX0, GRID_MIN, GRID_MAX));
        assert!(!cage.point_inside(VERTEX1, GRID_MIN, GRID_MAX));
    }
}
