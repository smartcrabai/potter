//! Blender `Solidify` modifier. `EXTRUDE` follows `MOD_solidify_extrude.cc` and `NON_MANIFOLD`
//! follows `MOD_solidify_nonmanifold.cc`; this module parses the schema into Blender's
//! `SolidifyModifierData` names and resolves the vertex groups both algorithms read.

mod extrude;
mod nonmanifold;
#[cfg(test)]
mod parity_tests;
mod util;

use serde_json::Value;

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::ModifierEvaluationContext,
    model::Modifier,
};

use super::{
    Mesh, bool_param, invalid_parameter, number_param, schema_value, string_param, validated,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Extrude,
    NonManifold,
}

/// `nonmanifold_thickness_mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ThicknessMode {
    Fixed,
    Even,
    Constraints,
}

/// `nonmanifold_boundary_mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BoundaryMode {
    None,
    Round,
    Flat,
}

/// Input weights of the thickness group (`vertex_group`), in `mesh.vertices` storage order. `None`
/// when the name is empty or the mesh has no such group (Blender's `defgrp_index == -1`). Unassigned
/// vertices weigh zero. The weights are raw: `invert_vertex_group` is applied by the algorithms.
pub(super) struct VertexGroups {
    pub(super) thickness: Option<Vec<f64>>,
}

/// `SolidifyModifierData` after schema validation, under Blender's field names.
pub(super) struct Params {
    pub(super) mode: Mode,
    /// `smd->offset`: schema `thickness`.
    pub(super) thickness: f64,
    /// `smd->offset_fac`: schema `offset`, from -1 (inward) through 1 (outward).
    pub(super) offset_fac: f64,
    /// `smd->offset_clamp`: schema `thickness_clamp`.
    pub(super) offset_clamp: f64,
    /// `MOD_SOLIDIFY_OFFSET_ANGLE_CLAMP`: schema `use_thickness_angle_clamp`.
    pub(super) angle_clamp: bool,
    /// `smd->offset_fac_vg`: schema `thickness_vertex_group`.
    pub(super) offset_fac_vg: f64,
    /// `MOD_SOLIDIFY_VGROUP_INV`: schema `invert_vertex_group`.
    pub(super) invert_vertex_group: bool,
    pub(super) crease_inner: f64,
    pub(super) crease_outer: f64,
    pub(super) crease_rim: f64,
    /// `smd->mat_ofs`: schema `material_offset`.
    pub(super) material_offset: i64,
    /// `smd->mat_ofs_rim`: schema `material_offset_rim`.
    pub(super) material_offset_rim: i64,
    /// `MOD_SOLIDIFY_RIM`: schema `use_rim`.
    pub(super) rim: bool,
    /// `MOD_SOLIDIFY_NOSHELL`: schema `use_rim_only`.
    pub(super) rim_only: bool,
    /// `MOD_SOLIDIFY_EVEN`: schema `use_even_offset`.
    pub(super) even_offset: bool,
    /// `MOD_SOLIDIFY_NORMAL_CALC`: schema `use_quality_normals`.
    pub(super) quality_normals: bool,
    /// `MOD_SOLIDIFY_FLIP`: schema `use_flip_normals`.
    pub(super) flip_normals: bool,
    /// Schema `use_flat_faces`.
    pub(super) flat_faces: bool,
    pub(super) thickness_mode: ThicknessMode,
    pub(super) boundary_mode: BoundaryMode,
    /// `nonmanifold_merge_threshold`.
    pub(super) merge_threshold: f64,
    /// `bevel_convex`.
    pub(super) bevel_convex: f64,
    pub(super) groups: VertexGroups,
    /// Output group names `shell_vertex_group` and `rim_vertex_group`; empty means unnamed.
    pub(super) shell_group: String,
    pub(super) rim_group: String,
    /// Material slots of the evaluated object (`ctx->object->totcol`): the scene count when the node is
    /// evaluated, else the mesh's own face-addressed count.
    pub(super) material_slot_count: usize,
}

pub(super) fn solidify(
    mesh: &Mesh,
    modifier: &Modifier,
    context: ModifierEvaluationContext<'_>,
) -> Result<Mesh> {
    validated(mesh)?;
    let material_slot_count = context
        .material_slot_count
        .unwrap_or_else(|| face_slot_count(mesh));
    let params = Params::parse(mesh, modifier, material_slot_count)?;
    match params.mode {
        Mode::Extrude => extrude::solidify_extrude(mesh, modifier, &params),
        Mode::NonManifold => nonmanifold::solidify_nonmanifold(mesh, modifier, &params),
    }
}

/// Material slots the mesh's faces address: one more than the largest `material_index`, or zero for a
/// mesh without faces. Used when no scene object supplies the count.
fn face_slot_count(mesh: &Mesh) -> usize {
    mesh.faces
        .iter()
        .map(|face| {
            usize::try_from(face.material_index)
                .unwrap_or(usize::MAX)
                .saturating_add(1)
        })
        .max()
        .unwrap_or(0)
}

impl Params {
    fn parse(mesh: &Mesh, modifier: &Modifier, material_slot_count: usize) -> Result<Self> {
        let mode = match string_param(modifier, "solidify_mode")? {
            "EXTRUDE" => Mode::Extrude,
            "NON_MANIFOLD" => Mode::NonManifold,
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "solidify_mode",
                    "EXTRUDE or NON_MANIFOLD",
                ));
            }
        };
        let thickness_mode = match string_param(modifier, "nonmanifold_thickness_mode")? {
            "FIXED" => ThicknessMode::Fixed,
            "EVEN" => ThicknessMode::Even,
            "CONSTRAINTS" => ThicknessMode::Constraints,
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "nonmanifold_thickness_mode",
                    "FIXED, EVEN, or CONSTRAINTS",
                ));
            }
        };
        let boundary_mode = match string_param(modifier, "nonmanifold_boundary_mode")? {
            "NONE" => BoundaryMode::None,
            "ROUND" => BoundaryMode::Round,
            "FLAT" => BoundaryMode::Flat,
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "nonmanifold_boundary_mode",
                    "NONE, ROUND, or FLAT",
                ));
            }
        };
        let thickness_group = string_param(modifier, "vertex_group")?;
        Ok(Self {
            mode,
            thickness: number_param(modifier, "thickness")?,
            offset_fac: number_param(modifier, "offset")?,
            offset_clamp: number_param(modifier, "thickness_clamp")?,
            angle_clamp: bool_param(modifier, "use_thickness_angle_clamp")?,
            offset_fac_vg: number_param(modifier, "thickness_vertex_group")?,
            invert_vertex_group: bool_param(modifier, "invert_vertex_group")?,
            crease_inner: number_param(modifier, "edge_crease_inner")?,
            crease_outer: number_param(modifier, "edge_crease_outer")?,
            crease_rim: number_param(modifier, "edge_crease_rim")?,
            material_offset: integer_param(modifier, "material_offset")?,
            material_offset_rim: integer_param(modifier, "material_offset_rim")?,
            rim: bool_param(modifier, "use_rim")?,
            rim_only: bool_param(modifier, "use_rim_only")?,
            even_offset: bool_param(modifier, "use_even_offset")?,
            quality_normals: bool_param(modifier, "use_quality_normals")?,
            flip_normals: bool_param(modifier, "use_flip_normals")?,
            flat_faces: bool_param(modifier, "use_flat_faces")?,
            thickness_mode,
            boundary_mode,
            merge_threshold: number_param(modifier, "nonmanifold_merge_threshold")?,
            bevel_convex: number_param(modifier, "bevel_convex")?,
            groups: VertexGroups {
                thickness: vertex_group_weights(mesh, thickness_group)?,
            },
            shell_group: string_param(modifier, "shell_vertex_group")?.to_owned(),
            rim_group: string_param(modifier, "rim_vertex_group")?.to_owned(),
            material_slot_count,
        })
    }
}

fn integer_param(modifier: &Modifier, name: &str) -> Result<i64> {
    schema_value(modifier, name)?
        .as_i64()
        .ok_or_else(|| invalid_parameter(modifier, name, "an integer"))
}

fn vertex_group_weights(mesh: &Mesh, name: &str) -> Result<Option<Vec<f64>>> {
    if name.is_empty() {
        return Ok(None);
    }
    let Some(weights) = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .and_then(|groups| groups.get(name))
        .and_then(Value::as_object)
    else {
        return Ok(None);
    };
    mesh.vertices
        .iter()
        .map(|vertex| {
            let weight = weights
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if (0.0..=1.0).contains(&weight) {
                Ok(weight)
            } else {
                Err(PotError::with_details(
                    ErrorCode::InvalidArgument,
                    "Solidify vertex-group weights must be finite values from 0 through 1",
                    serde_json::json!({"vertex_id":vertex.id,"group":name,"weight":weight}),
                ))
            }
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}
