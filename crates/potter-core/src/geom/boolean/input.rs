//! Joins boolean operands into arena triangles (Blender `meshes_to_imesh`,
//! `geometry/intern/mesh_boolean.cc` l.448–575, and `clean_transform`, l.326–346).
//!
//! Index scheme (Blender's offset scheme):
//! - a vertex's `orig` is its storage index in its own operand (first creator wins in the arena);
//! - a triangle's `orig` is the global face index: operands in order, then faces in storage order;
//! - a boundary edge's `orig` is the operand's edge offset plus its storage index.
//!
//! Blender's `join_meshes_with_transforms` only builds the output attribute container, so it is
//! not needed for these indices and is not ported.
//!
//! Deviations from Blender:
//! - Blender stores positions as `float` and replaces non-finite values with zero
//!   (`clean_float3`). Here positions are rounded to `f32` as Blender stores them, and a
//!   non-finite position after transformation is an error.
//! - Blender's `join_meshes_with_transforms` is not ported (see above).

use std::collections::HashMap;

use glam::{DMat4, Mat3, Mat4, Vec3};

use crate::error::{PotError, Result};
use crate::geom::{Mesh, edge_key};

use super::exact::rat3_from_dvec3;
use super::imesh::{Arena, NO_INDEX, Tri};
use super::triangulate::triangulate_face;

/// One boolean operand: its mesh and the transform from object space to the joined space.
pub(crate) struct Operand<'a> {
    pub mesh: &'a Mesh,
    pub transform: DMat4,
}

/// Blender `meshes_to_imesh`: interns every operand's vertices in `arena` and returns the
/// triangulated faces of all operands, in operand then face order.
///
/// Operands whose transform has a negative determinant have their faces reversed when operand 0
/// is not negative (Blender `need_face_flip`), so every operand keeps operand 0's orientation.
pub(crate) fn meshes_to_tris(operands: &[Operand<'_>], arena: &mut Arena) -> Result<Vec<Tri>> {
    let prepared: Vec<(Mat4, bool)> = operands
        .iter()
        .map(|operand| {
            let transform = clean_transform(operand.transform.as_mat4());
            (transform, is_negative(transform))
        })
        .collect();
    let Some(&(_, first_negative)) = prepared.first() else {
        return Err(PotError::invalid_argument(
            "boolean requires at least one operand",
        ));
    };

    let mut tris = Vec::new();
    let mut edge_base = 0_usize;
    let mut face_base = 0_usize;
    for (mi, (operand, &(transform, negative))) in operands.iter().zip(&prepared).enumerate() {
        let mesh = operand.mesh;
        mesh.validate().map_err(|error| {
            PotError::invalid_argument(format!("invalid boolean operand {mi} mesh: {error}"))
        })?;
        let flip = negative != first_negative;
        let arena_ids = add_vertices(mesh, transform, mi, arena)?;

        let vertex_index: HashMap<u32, usize> = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(i, vertex)| (vertex.id, i))
            .collect();
        let edge_index: HashMap<(u32, u32), usize> = mesh
            .edges
            .iter()
            .enumerate()
            .map(|(i, edge)| (edge_key(edge.vertices[0], edge.vertices[1]), i))
            .collect();

        for (fi, face) in mesh.faces.iter().enumerate() {
            let ids = &face.vertices;
            let n = ids.len();
            let mut verts = Vec::with_capacity(n);
            let mut edges = Vec::with_capacity(n);
            for (k, &id) in ids.iter().enumerate() {
                let next = ids[(k + 1) % n];
                let local_vertex = *vertex_index.get(&id).ok_or_else(|| {
                    PotError::invalid_argument("boolean face references a missing vertex")
                })?;
                verts.push(arena_ids[local_vertex]);
                let local_edge = *edge_index.get(&edge_key(id, next)).ok_or_else(|| {
                    PotError::invalid_argument("boolean face boundary edge is missing")
                })?;
                edges.push(to_index(edge_base + local_edge)?);
            }
            let (verts, edges) = if flip {
                reversed(&verts, &edges)
            } else {
                (verts, edges)
            };
            let face_index = to_index(face_base + fi)?;
            tris.extend(triangulate_face(arena, &verts, face_index, &edges));
        }

        edge_base += mesh.edges.len();
        face_base += mesh.faces.len();
    }
    Ok(tris)
}

/// Blender's vertex loop in `meshes_to_imesh`: each position is rounded to `f32`, transformed in
/// `f32` (skipped for the identity), and interned by exact position with its storage index as
/// `orig`. Returns arena indices in storage order.
fn add_vertices(mesh: &Mesh, transform: Mat4, mi: usize, arena: &mut Arena) -> Result<Vec<u32>> {
    let identity = transform == Mat4::IDENTITY;
    let mut ids = Vec::with_capacity(mesh.vertices.len());
    for (i, vertex) in mesh.vertices.iter().enumerate() {
        let mut co = Vec3::new(vertex.co.x as f32, vertex.co.y as f32, vertex.co.z as f32);
        if !identity {
            co = transform.transform_point3(co);
        }
        if !co.is_finite() {
            return Err(PotError::invalid_argument(format!(
                "boolean operand {mi} vertex {i} is not finite after transform"
            )));
        }
        ids.push(arena.add_or_find_vert(rat3_from_dvec3(co.as_dvec3()), to_index(i)?));
    }
    Ok(ids)
}

/// Blender's face flip in `meshes_to_imesh`: reverses the corners of a face. Edge `i` (from corner
/// `i` to `i + 1`) moves to the reversed position, and the closing edge stays last.
fn reversed(verts: &[u32], edges: &[i32]) -> (Vec<u32>, Vec<i32>) {
    let n = verts.len();
    let mut rev_verts = vec![0; n];
    let mut rev_edges = vec![NO_INDEX; n];
    for (i, (&v, &e)) in verts.iter().zip(edges).enumerate() {
        rev_verts[n - 1 - i] = v;
        let iedge = if i + 1 < n { n - i - 2 } else { n - 1 };
        rev_edges[iedge] = e;
    }
    (rev_verts, rev_edges)
}

fn to_index(index: usize) -> Result<i32> {
    i32::try_from(index).map_err(|_| {
        PotError::invalid_argument("boolean input exceeds the supported element count")
    })
}

/// Blender `clean_transform`: entries within `1e-6` of `0`, `1`, or `-1` snap to that value.
fn clean_transform(mat: Mat4) -> Mat4 {
    Mat4::from_cols_array_2d(&mat.to_cols_array_2d().map(|column| column.map(clean_entry)))
}

fn clean_entry(value: f32) -> f32 {
    const FUZZ: f32 = 1e-6;
    if value.abs() <= FUZZ {
        0.0
    } else if (value - 1.0).abs() <= FUZZ {
        1.0
    } else if (value + 1.0).abs() <= FUZZ {
        -1.0
    } else {
        value
    }
}

/// Blender `math::is_negative` for a 4x4 matrix: negative determinant of its 3x3 block.
fn is_negative(transform: Mat4) -> bool {
    Mat3::from_mat4(transform).determinant() < 0.0
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use glam::{DMat4, DVec3};

    use crate::geom::{BoxParams, Mesh, edge_key};

    use super::*;

    type TestResult = std::result::Result<(), Box<dyn Error>>;

    /// Storage index of the edge joining two persistent vertex IDs in `mesh`.
    fn local_edge(mesh: &Mesh, a: u32, b: u32) -> std::result::Result<i32, Box<dyn Error>> {
        let wanted = edge_key(a, b);
        let index = mesh
            .edges
            .iter()
            .position(|edge| edge_key(edge.vertices[0], edge.vertices[1]) == wanted)
            .ok_or("edge missing")?;
        Ok(i32::try_from(index)?)
    }

    #[test]
    fn cube_becomes_twelve_triangles_with_face_and_edge_origins() -> TestResult {
        let mesh = Mesh::box_mesh(BoxParams::default())?;
        let mut arena = Arena::new();
        let tris = meshes_to_tris(
            &[Operand {
                mesh: &mesh,
                transform: DMat4::IDENTITY,
            }],
            &mut arena,
        )?;
        assert_eq!(tris.len(), 12);
        assert_eq!(arena.num_verts(), 8);
        for face in 0..6 {
            assert_eq!(tris.iter().filter(|tri| tri.orig == face).count(), 2);
        }
        for id in 0..8_u32 {
            assert_eq!(arena.vert(id).orig, i32::try_from(id)?);
        }
        // Every quad gives two triangles sharing one diagonal: 12 diagonals, 24 input edges.
        let entries: Vec<i32> = tris.iter().flat_map(|tri| tri.edge_orig).collect();
        assert_eq!(entries.iter().filter(|&&e| e == NO_INDEX).count(), 12);
        assert_eq!(
            entries.iter().filter(|&&e| (0..12).contains(&e)).count(),
            24
        );
        for tri in &tris {
            assert_eq!(tri.edge_orig.iter().filter(|&&e| e == NO_INDEX).count(), 1);
        }
        Ok(())
    }

    #[test]
    fn negative_operand_faces_are_flipped_to_operand_zero_winding() -> TestResult {
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )?;
        let mirror = DMat4::from_scale(DVec3::new(-1.0, 1.0, 1.0));
        let mut arena = Arena::new();
        let tris = meshes_to_tris(
            &[
                Operand {
                    mesh: &mesh,
                    transform: DMat4::IDENTITY,
                },
                Operand {
                    mesh: &mesh,
                    transform: mirror,
                },
            ],
            &mut arena,
        )?;
        assert_eq!(tris.len(), 2);

        let (e01, e12, e20) = (
            local_edge(&mesh, 0, 1)?,
            local_edge(&mesh, 1, 2)?,
            local_edge(&mesh, 2, 0)?,
        );
        let base = i32::try_from(mesh.edges.len())?;

        // Operand 0 is not flipped.
        assert_eq!(tris[0].v, [0, 1, 2]);
        assert_eq!(tris[0].orig, 0);
        assert_eq!(tris[0].edge_orig, [e01, e12, e20]);

        // The mirrored operand maps (1,0,0) to a new vertex 3 and shares the other two corners.
        // Its corners are reversed to [2, 3, 0] so it stays counter-clockwise in xy.
        assert_eq!(tris[1].v, [2, 3, 0]);
        assert_eq!(tris[1].orig, 1);
        assert_eq!(tris[1].edge_orig, [base + e12, base + e01, base + e20]);
        let [a, b, c] = tris[1].v.map(|v| arena.vert(v).co_f);
        assert!((b - a).cross(c - a).z > 0.0);
        Ok(())
    }

    #[test]
    fn non_finite_transformed_coordinate_is_rejected() -> TestResult {
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )?;
        // The f32 transform overflows to infinity, and infinity times zero is NaN.
        let huge = DMat4::from_scale(DVec3::splat(1.0e300));
        let mut arena = Arena::new();
        assert!(
            meshes_to_tris(
                &[Operand {
                    mesh: &mesh,
                    transform: huge,
                }],
                &mut arena,
            )
            .is_err()
        );
        Ok(())
    }
}
