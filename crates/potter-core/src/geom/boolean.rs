//! Boolean operations on polygon meshes.
//!
//! `EXACT` (the default solver) is Blender's Mesh Arrangements, ported under `boolean/`: exact
//! rational intersection, classification of the intersected triangles, and assembly of the output
//! polygons with Blender's attribute rules.
//!
//! `FLOAT` ports Blender's `BMesh` intersection solver in `boolean/float.rs`: f32 operand transforms,
//! triangle BVH intersections, edgenet splitting, face-group ray parity, and `BMesh` attribute rules.
//!
//! Operand and material semantics follow Blender's `non_float_boolean_mesh`: an unset Object
//! operand leaves the mesh unchanged, and `material_remaps` gives the INDEX and TRANSFER slot maps.
//! FLOAT always remaps by INDEX, from the effective (OBJECT-resolved) slots.

mod cdt;
mod classify;
mod dissolve;
mod exact;
mod float;
mod imesh;
mod input;
mod intersect;
mod output;
#[cfg(test)]
mod tests;
mod triangulate;

use glam::DMat4;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
    model::Id,
};

use classify::{BoolOp, boolean_tris};
use dissolve::polymesh_from_tris;
use imesh::Arena;
use input::{Operand, meshes_to_tris};
use output::{OutputMaterials, polys_to_mesh};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Operation {
    Union,
    Difference,
    Intersect,
}

impl Operation {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_uppercase().as_str() {
            "UNION" => Ok(Self::Union),
            "DIFFERENCE" => Ok(Self::Difference),
            "INTERSECT" => Ok(Self::Intersect),
            _ => Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                "boolean operation must be UNION, DIFFERENCE, or INTERSECT",
                serde_json::json!({"parameter":"operation","pointer":"/params/operation"}),
            )),
        }
    }
}

/// Blender `material_mode`: how operand material indices map to output material indices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MaterialMode {
    /// Each operand slot takes the subject slot holding the same material, or keeps its index.
    Index,
    /// The subject's slots come first, then each operand's new materials are appended to one pool.
    Transfer,
}

/// One operand of a Boolean solver, resolved by the evaluator in the subject's space.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanOperand {
    /// The operand's mesh in its object space.
    pub mesh: Mesh,
    /// Object space to subject space (Blender `world_to_object * operand_world`).
    pub to_subject: DMat4,
    /// Whether Blender's scene-context FLOAT solver flips this operand's faces before
    /// tessellation: the subject and the operand differ in negative-determinant state. `None`
    /// derives it from the determinant of `to_subject` for context-free callers. EXACT ignores it.
    pub float_flip: Option<bool>,
    /// Output material index for each local material index, from [`material_remaps`]. An empty
    /// list keeps every index.
    pub material_remap: Vec<u32>,
}

/// Blender `non_float_boolean_mesh` for the EXACT solver: `subject` is operand 0 with the identity
/// transform, and `operands` follow in order. One N-ary call, so a collection is never folded
/// pairwise. An empty `operands` runs the solver on the subject alone, which removes its
/// self-intersections (a collection with no members).
///
/// # Errors
/// Returns `INVALID_ARGUMENT` for an invalid operand mesh, and the errors of the exact pipeline
/// stages.
pub(crate) fn exact_boolean(
    subject: &Mesh,
    operands: &[BooleanOperand],
    operation: Operation,
    use_self: bool,
    hole_tolerant: bool,
) -> Result<Mesh> {
    let mut joined = Vec::with_capacity(operands.len() + 1);
    joined.push(Operand {
        mesh: subject,
        transform: DMat4::IDENTITY,
    });
    joined.extend(operands.iter().map(|operand| Operand {
        mesh: &operand.mesh,
        transform: operand.to_subject,
    }));
    // The subject keeps its own indices (empty remap); each operand uses its table.
    let mut remap = vec![Vec::new()];
    remap.extend(
        operands
            .iter()
            .map(|operand| operand.material_remap.clone()),
    );

    // Blender `mesh_id_for_face`: a triangle's `orig` is a global face index, mapped to the
    // operand whose face range holds it.
    let mut face_starts = Vec::with_capacity(joined.len());
    let mut face_total = 0_usize;
    for operand in &joined {
        face_starts.push(face_total);
        face_total += operand.mesh.faces.len();
    }
    let shape_of = |orig: i32| -> usize {
        usize::try_from(orig).map_or(0, |face| {
            face_starts
                .partition_point(|&start| start <= face)
                .saturating_sub(1)
        })
    };

    let mut arena = Arena::new();
    let tris = meshes_to_tris(&joined, &mut arena)?;
    let intersected = boolean_tris(
        &mut arena,
        &tris,
        joined.len(),
        &shape_of,
        bool_op(operation),
        use_self,
        hole_tolerant,
    )?;
    let polys = polymesh_from_tris(&arena, &intersected);
    polys_to_mesh(&polys, &arena, &joined, &OutputMaterials { remap: &remap })
}

/// The result of Blender's FLOAT object-operand shortcut (`get_quick_mesh`).
pub(crate) enum FloatObjectQuickMesh {
    /// The subject is the result unchanged.
    Subject,
    /// The result is an empty mesh with no material slots.
    Empty,
    /// The result is the operand's mesh, moved into the subject's space.
    Operand(Mesh),
}

/// Blender's FLOAT shortcut for an OBJECT operand when the subject or the operand has no faces.
/// Returns `None` when both have faces, so the solver runs. Collections never take this path.
pub(crate) fn float_object_quick_mesh(
    subject: &Mesh,
    operand: &BooleanOperand,
    operation: Operation,
) -> Option<FloatObjectQuickMesh> {
    if !subject.faces.is_empty() && !operand.mesh.faces.is_empty() {
        return None;
    }
    Some(match operation {
        Operation::Intersect => FloatObjectQuickMesh::Empty,
        Operation::Union if subject.faces.is_empty() => {
            let transform = operand.to_subject.to_cols_array().map(|value| value as f32);
            let mut mesh = operand.mesh.clone();
            for vertex in &mut mesh.vertices {
                let point =
                    glam::Vec3::new(vertex.co.x as f32, vertex.co.y as f32, vertex.co.z as f32);
                vertex.co =
                    crate::geom::blender_matrix::blender_transform_point_f32(&transform, point)
                        .as_dvec3();
            }
            FloatObjectQuickMesh::Operand(mesh)
        }
        Operation::Union | Operation::Difference => FloatObjectQuickMesh::Subject,
    })
}

/// Blender's FLOAT `BMesh` Boolean over operands already resolved into the subject's space.
pub(crate) fn float_boolean(
    subject: &Mesh,
    operands: &[BooleanOperand],
    operation: Operation,
    double_threshold: f32,
) -> Result<Mesh> {
    float::boolean_float(subject, operands, operation, double_threshold)
}

/// Blender `BoolOpType` for an operation.
fn bool_op(operation: Operation) -> BoolOp {
    match operation {
        Operation::Union => BoolOp::Union,
        Operation::Difference => BoolOp::Difference,
        Operation::Intersect => BoolOp::Intersect,
    }
}

/// Blender `material_remaps` (`non_float_boolean_mesh`): the output slot index of each operand's
/// local slot, and the output node's slot list.
///
/// `subject` and every entry of `operands` list one object's material slots by material identity,
/// in slot order; `None` is an empty slot. `remaps` has one entry per operand, subject first. The
/// subject's entry is empty, so its indices stay. `pool` is the output node's slot list.
///
/// - INDEX: `pool` is the joined table of the subject and operands ([`index_pool`]). An operand slot
///   maps to the subject slot at its index when that slot holds the same material. Otherwise it maps
///   to the first subject slot holding
///   the material (`BKE_object_material_remap_calc`, material.cc:1307-1351, and `Map::add`, which
///   keeps the first key, BLI_map.hh:289-295). Otherwise it keeps its local index when the subject
///   has that slot, and maps to 0 when it does not. An operand with no slots maps to `[0]`
///   (MOD_boolean.cc:366-375).
/// - TRANSFER: `pool` is the subject's distinct materials in first-occurrence order (Blender's
///   `VectorSet`, MOD_boolean.cc:422-430), or `[None]` when the subject has no slots. Each operand
///   slot maps to the pool index of its material, appending a material the pool lacks. An empty
///   operand slot maps to the pooled `None`. Blender writes `-1` there (`MOD_boolean.cc:387` and
///   `geometry_mesh_boolean.cc:276-280`), which `u32` cannot hold: both mean no material.
///
/// As in Blender, the subject keeps its own indices, so with duplicate subject slots a face index
/// can point past the end of `pool`.
#[must_use]
pub(crate) fn material_remaps<'a>(
    mode: MaterialMode,
    subject: &[Option<&'a Id>],
    operands: &[Vec<Option<&'a Id>>],
) -> (Vec<Vec<u32>>, Vec<Option<Id>>) {
    let mut pool: Vec<Option<Id>> = match mode {
        MaterialMode::Index => index_pool(subject, operands),
        MaterialMode::Transfer => transfer_pool(subject),
    };
    let mut remaps = vec![Vec::new()];
    for slots in operands {
        remaps.push(match mode {
            MaterialMode::Index => index_remap(subject, slots),
            MaterialMode::Transfer => transfer_remap(&mut pool, slots),
        });
    }
    (remaps, pool)
}

/// The TRANSFER pool: the subject's distinct materials in first-occurrence order, or one empty slot
/// when the subject has none.
fn transfer_pool(subject: &[Option<&Id>]) -> Vec<Option<Id>> {
    if subject.is_empty() {
        return vec![None];
    }
    let mut pool: Vec<Option<Id>> = Vec::with_capacity(subject.len());
    for &slot in subject {
        if !pool.iter().any(|pooled| pooled.as_ref() == slot) {
            pool.push(slot.cloned());
        }
    }
    pool
}

/// The INDEX result table. Blender joins the subject and operand meshes (`join_meshes`, the source of
/// the output's material array), and each mesh contributes its slots, or one empty slot when it has
/// none; a material already in the table is not repeated. The subject keeps its own slots as they are,
/// so its face indices are unchanged. Measured on Blender 5.2.2 (EXACT, INDEX, default mode): an empty
/// subject gives `[None]`, an empty subject with an operand gives `[None, O0]`, and a subject `[H0]`
/// with an empty operand gives `[H0, None]`.
fn index_pool(subject: &[Option<&Id>], operands: &[Vec<Option<&Id>>]) -> Vec<Option<Id>> {
    let empty: [Option<&Id>; 1] = [None];
    let mut pool: Vec<Option<Id>> = if subject.is_empty() {
        vec![None]
    } else {
        subject.iter().map(|&slot| slot.cloned()).collect()
    };
    for slots in operands {
        let entries: &[Option<&Id>] = if slots.is_empty() { &empty } else { slots };
        for &slot in entries {
            if !pool.iter().any(|pooled| pooled.as_ref() == slot) {
                pool.push(slot.cloned());
            }
        }
    }
    pool
}

/// The INDEX remap of one operand (Blender `BKE_object_material_remap_calc`, see
/// [`material_remaps`]). An operand with no slots maps to `[0]`.
fn index_remap(subject: &[Option<&Id>], slots: &[Option<&Id>]) -> Vec<u32> {
    if slots.is_empty() {
        return vec![0];
    }
    slots
        .iter()
        .enumerate()
        .map(|(local, &slot)| {
            let index = if subject.get(local).copied() == Some(slot) {
                local
            } else if let Some(first) = subject.iter().position(|&held| held == slot) {
                first
            } else if local < subject.len() {
                local
            } else {
                0
            };
            slot_index(index)
        })
        .collect()
}

/// The TRANSFER remap of one operand: each slot's index in `pool`, appending the materials the
/// pool lacks (Blender `VectorSet::index_of_or_add`).
fn transfer_remap(pool: &mut Vec<Option<Id>>, slots: &[Option<&Id>]) -> Vec<u32> {
    slots
        .iter()
        .map(|&slot| {
            let index = pool
                .iter()
                .position(|pooled| pooled.as_ref() == slot)
                .unwrap_or_else(|| {
                    pool.push(slot.cloned());
                    pool.len() - 1
                });
            slot_index(index)
        })
        .collect()
}

/// Output material indices are slot numbers, which always fit in `u32`.
fn slot_index(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}
