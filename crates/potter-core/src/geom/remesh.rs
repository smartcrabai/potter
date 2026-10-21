//! Source sampling for the voxel remesh attribute transfer, Blender's `mesh_remesh_reproject_attributes`
//! (`blenkernel/intern/mesh_remesh_voxel.cc`). The remeshing itself is Blender's `BKE_mesh_remesh_voxel`
//! (`geom/modifiers/remesh_modes/voxel.rs`); this module finds, for each new vertex and face, the nearest
//! source triangle and its barycentric weights.

use glam::Vec3;
use std::collections::HashMap;

use crate::{
    error::{PotError, Result},
    geom::{
        Mesh,
        bvh::{KdopTree, NearestHit},
        modifiers::shrinkwrap::{closest_on_triangle, tri_weights},
    },
};

/// One triangle of the source tessellation, with what the transfer reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SourceTriangle {
    /// Source vertex indices (into `mesh.vertices`), in tessellation order.
    pub(crate) vertices: [usize; 3],
    /// Source face position (into `mesh.faces`).
    pub(crate) face: usize,
    /// Corner position inside `face` of each triangle vertex, in the same order.
    pub(crate) corners: [usize; 3],
}

/// The nearest source triangle of a query point, and the barycentric weights of the point in it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sample {
    /// Index into [`SourceSampler::triangle`].
    pub(crate) triangle: usize,
    /// Blender's `interp_weights_tri_v3` of the query point in that triangle.
    pub(crate) weights: Vec3,
    /// The source face of the triangle, so face attributes follow the nearest source face.
    pub(crate) face: usize,
}

/// Blender's corner-triangle tree of the source mesh (`BKE_bvhtree_from_mesh_get` with corner tris), used
/// for `find_nearest_tris`.
pub(crate) struct SourceSampler {
    triangles: Vec<SourceTriangle>,
    points: Vec<Vec3>,
    tree: KdopTree,
}

impl SourceSampler {
    /// Builds the tree of `source`'s tessellation. A source without triangles cannot be sampled.
    pub(crate) fn new(source: &Mesh) -> Result<Self> {
        let index_by_id: HashMap<u32, usize> = source
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect();
        let points: Vec<Vec3> = source
            .vertices
            .iter()
            .map(|vertex| vertex.co.as_vec3())
            .collect();
        let tessellation = source.tessellate().map_err(|error| {
            PotError::invalid_argument(format!("invalid source triangulation: {error}"))
        })?;
        let expected: usize = source
            .faces
            .iter()
            .map(|face| face.vertices.len().saturating_sub(2))
            .sum();
        if tessellation.len() != expected {
            return Err(PotError::invalid_argument(
                "source tessellation does not match its faces",
            ));
        }
        if expected == 0 {
            return Err(PotError::invalid_argument(
                "voxel remesh needs a source surface with faces",
            ));
        }
        let mut triangles = Vec::with_capacity(expected);
        let mut next = 0;
        for (face_index, face) in source.faces.iter().enumerate() {
            for _ in 0..face.vertices.len().saturating_sub(2) {
                let ids = tessellation[next];
                next += 1;
                let mut vertices = [0; 3];
                let mut corners = [0; 3];
                for (slot, id) in ids.into_iter().enumerate() {
                    vertices[slot] = *index_by_id.get(&id).ok_or_else(|| {
                        PotError::invalid_argument("source triangle references a missing vertex")
                    })?;
                    corners[slot] = face
                        .vertices
                        .iter()
                        .position(|face_id| *face_id == id)
                        .ok_or_else(|| {
                            PotError::invalid_argument("source triangle corner is missing")
                        })?;
                }
                triangles.push(SourceTriangle {
                    vertices,
                    face: face_index,
                    corners,
                });
            }
        }
        // `BLI_bvhtree_new` clamps the epsilon to FLT_EPSILON, so the leaves are the inflated kdop boxes.
        let bounds: Vec<[f32; 6]> = triangles
            .iter()
            .map(|triangle| {
                KdopTree::leaf_bounds_for_triangle(
                    triangle.vertices.map(|vertex| points[vertex].to_array()),
                )
            })
            .collect();
        Ok(Self {
            triangles,
            points,
            tree: KdopTree::from_leaf_bounds(&bounds),
        })
    }

    /// The triangle at `index`, as returned in [`Sample::triangle`].
    pub(crate) fn triangle(&self, index: usize) -> Option<&SourceTriangle> {
        self.triangles.get(index)
    }

    /// Blender's `find_nearest_tris` (`BLI_bvhtree_find_nearest` with `mesh_corner_tris_nearest_point`,
    /// depth-first, strict `<` so the first leaf wins ties) and `sample_barycentric_weights`.
    pub(crate) fn nearest(&self, point: Vec3) -> Result<Sample> {
        let mut nearest = NearestHit {
            index: -1,
            co: [0.0; 3],
            dist_sq: f32::MAX,
        };
        self.tree.find_nearest(
            point.to_array(),
            &mut nearest,
            &mut |index: usize, query: [f32; 3], hit: &mut NearestHit| {
                let query = Vec3::from(query);
                let [a, b, c] = self.triangles[index]
                    .vertices
                    .map(|vertex| self.points[vertex]);
                let closest = closest_on_triangle(query, a, b, c);
                let dist_sq = (query - closest).length_squared();
                if dist_sq < hit.dist_sq {
                    hit.index = index as isize;
                    hit.dist_sq = dist_sq;
                    hit.co = closest.to_array();
                }
            },
            false,
        );
        let triangle = usize::try_from(nearest.index)
            .map_err(|_| PotError::invalid_argument("voxel remesh found no source triangle"))?;
        let source = self.triangles[triangle];
        let [a, b, c] = source.vertices.map(|vertex| self.points[vertex]);
        Ok(Sample {
            triangle,
            weights: Vec3::from(tri_weights(a, b, c, point)),
            face: source.face,
        })
    }
}

/// Blender's `attribute_math::mix3` for float values, in f32 as Blender stores them:
/// `weights.x * v0 + weights.y * v1 + weights.z * v2`.
pub(crate) fn mix3(weights: Vec3, values: [f64; 3]) -> f64 {
    let [v0, v1, v2] = values.map(|value| value as f32);
    f64::from(weights.x * v0 + weights.y * v1 + weights.z * v2)
}

/// Blender's `face_center_calc` (`blenkernel/intern/mesh_evaluate.cc`) in f32: triangles and quads average
/// with `/ 3` and `/ 4`, and other sizes add each corner times `1 / n`.
pub(crate) fn face_center(corners: &[Vec3]) -> Vec3 {
    match corners {
        [a, b, c] => (*a + *b + *c) / 3.0,
        [a, b, c, d] => (*a + *b + *c + *d) / 4.0,
        _ => {
            let weight = 1.0 / f32::from(u16::try_from(corners.len()).unwrap_or(u16::MAX));
            corners
                .iter()
                .fold(Vec3::ZERO, |center, corner| center + *corner * weight)
        }
    }
}
