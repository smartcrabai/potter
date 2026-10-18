// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/modifiers/intern/MOD_array.cc, source/blender/modifiers/intern/MOD_displace.cc, source/blender/modifiers/intern/MOD_smooth.cc, source/blender/modifiers/intern/MOD_solidify.cc, source/blender/modifiers/intern/MOD_subsurf.cc, SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.
//   source/blender/modifiers/intern/MOD_solidify_extrude.cc, source/blender/bmesh/tools/bmesh_bevel.cc, SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.
//   source/blender/blenlib/intern/math_matrix_c.cc, SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved., GPL-2.0-or-later.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use glam::{DMat4, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::ModifierEvaluationContext,
    graph::{self, NodeGroup},
    model::{Id, Modifier, Registry},
    params::{self, ParameterFamily},
};

use super::attribute_keys;
use super::clnor::{self, CornerNormalCode, NormalCodeOptions, settle_custom_normals};
use super::triangulate::{NgonMethod, QuadMethod, triangulate_mesh};
use super::{
    Edge, Face, Mesh, MeshError, Vertex, edge_key,
    normals::{corner_normals, face_normal, safe_acos_approx, sharp_edge_flags},
    provenance::{Origin, edge_flags, mean_value_corners, remap_attributes},
    uv_layers,
};

mod advanced;
pub mod attributes;
mod bevel_shading;
mod bevel_vmesh;
mod decimate;
pub(crate) mod deform;
mod ocean;
mod remesh_modes;
pub mod simulation;
mod subdivision_limit;
mod subdivision_normals;
mod subdivision_uv;
use subdivision_limit::Samples;
type Polygon = (Vec<usize>, u32);
type BevelEdgeFaces = BTreeMap<(usize, usize), Vec<(usize, usize, u32)>>;
type BevelEdgeFacesWithMaterial = BTreeMap<(usize, usize), Vec<(usize, usize, usize, u32)>>;
/// Merged points, polygons, and edges, with the input element each output point and polygon came from.
struct MergedGeometry {
    points: Vec<DVec3>,
    polygons: Vec<Polygon>,
    edges: Vec<[usize; 2]>,
    /// Per merged point, the input point indices merged into it.
    point_members: Vec<Vec<usize>>,
    /// Per merged polygon, the input polygon index it was built from.
    polygon_sources: Vec<usize>,
}

impl MergedGeometry {
    /// Geometry that is not merged: every input point and polygon is kept in place.
    fn unmerged(points: Vec<DVec3>, polygons: Vec<Polygon>, edges: Vec<[usize; 2]>) -> Self {
        Self {
            point_members: (0..points.len()).map(|point| vec![point]).collect(),
            polygon_sources: (0..polygons.len()).collect(),
            points,
            polygons,
            edges,
        }
    }
}
const MAX_SUBDIVISION_TRIANGLES: u128 = 100_000_000;
const MAX_SUBDIVISION_MEMORY_BYTES: u128 = 8 * 1024 * 1024 * 1024;

/// Result of evaluating a modifier stack, including realized graph instance paths.
#[derive(Clone, Debug, PartialEq)]
pub struct ModifierEvaluation {
    pub mesh: Mesh,
    pub instance_paths: Vec<Vec<Id>>,
    pub warnings: Vec<PotError>,
}

/// Evaluate the enabled modifier stack over a source mesh without node-group context.
///
/// Disabled modifiers are ignored. Enabled modifiers are applied in the order supplied.
///
/// # Errors
///
/// Returns `UNSUPPORTED_FEATURE` for an unsupported modifier type, or `INVALID_ARGUMENT`
/// when a supported modifier has invalid parameters or input.
pub fn evaluate_modifiers(mesh: &Mesh, modifiers: &[Modifier]) -> Result<Mesh> {
    Ok(evaluate_modifiers_with_groups(mesh, modifiers, &BTreeMap::new())?.mesh)
}

/// Evaluate modifiers with scene geometry-node groups and ordered custom callbacks.
///
/// The mesh is owned so callers that have already produced a pre-modifier mesh can avoid
/// an extra clone. The callback runs for armature/lattice modifiers and volume-to-mesh
/// modifiers that reference scene volume data by ID.
///
/// # Errors
///
/// Returns an error for unsupported modifiers, invalid parameters, missing groups, or graph
/// and custom modifier evaluation failures.
pub fn evaluate_modifiers_with_groups_and_custom(
    evaluated: Mesh,
    modifiers: &[Modifier],
    node_groups: &Registry<NodeGroup>,
    custom_modifier: impl FnMut(&mut Mesh, &Modifier, &mut Vec<PotError>) -> Result<()>,
) -> Result<ModifierEvaluation> {
    evaluate_modifiers_with_groups_and_custom_at_frame(
        evaluated,
        modifiers,
        node_groups,
        1.0,
        custom_modifier,
    )
}

/// Evaluate modifiers with node groups and a scene frame.
///
/// # Errors
///
/// Returns an error for invalid frames, unsupported modifiers, invalid parameters,
/// missing groups, graph evaluation failures, or custom modifier failures.
pub fn evaluate_modifiers_with_groups_and_custom_at_frame(
    evaluated: Mesh,
    modifiers: &[Modifier],
    node_groups: &Registry<NodeGroup>,
    frame: f64,
    custom_modifier: impl FnMut(&mut Mesh, &Modifier, &mut Vec<PotError>) -> Result<()>,
) -> Result<ModifierEvaluation> {
    evaluate_modifiers_with_scene_context(
        evaluated,
        modifiers,
        node_groups,
        frame,
        DMat4::IDENTITY,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
        ModifierEvaluationContext::default(),
        |_, modifier| modifier.enabled,
        custom_modifier,
    )
}

pub(crate) fn evaluate_modifiers_with_scene_context(
    mut evaluated: Mesh,
    modifiers: &[Modifier],
    node_groups: &Registry<NodeGroup>,
    frame: f64,
    owner_world_matrix: DMat4,
    object_world_matrices: &BTreeMap<Id, DMat4>,
    object_meshes: &BTreeMap<Id, &Mesh>,
    evaluated_object_meshes: &BTreeMap<Id, Mesh>,
    curve_path_lengths: &BTreeMap<Id, f64>,
    context: ModifierEvaluationContext<'_>,
    is_active: impl Fn(usize, &Modifier) -> bool,
    mut custom_modifier: impl FnMut(&mut Mesh, &Modifier, &mut Vec<PotError>) -> Result<()>,
) -> Result<ModifierEvaluation> {
    if !frame.is_finite() {
        return Err(PotError::invalid_argument(
            "modifier evaluation frame must be finite",
        ));
    }
    let mut instance_paths = Vec::new();
    let mut warnings = Vec::new();
    for (modifier_index, modifier) in modifiers
        .iter()
        .enumerate()
        .filter(|&(modifier_index, modifier)| is_active(modifier_index, modifier))
    {
        validate_boolean_topology_suffix(modifiers, modifier_index, &is_active)?;
        let scene_operand_modifier = (matches!(
            modifier.modifier_type.as_str(),
            "boolean" | "shrinkwrap" | "surface_deform" | "mesh_deform"
        ) && ["object", "collection", "target"]
            .iter()
            .any(|key| modifier.params.get(*key).is_some_and(Value::is_string)))
            || (matches!(
                modifier.modifier_type.as_str(),
                "curve" | "hook" | "data_transfer"
            ) && modifier.params.get("object").is_some_and(Value::is_string))
            || (matches!(
                modifier.modifier_type.as_str(),
                "vertex_weight_proximity" | "normal_edit"
            ) && modifier.params.get("target").is_some_and(Value::is_string))
            || (modifier.modifier_type == "uv_project"
                && modifier
                    .params
                    .get("projectors")
                    .is_some_and(Value::is_array))
            || (modifier.modifier_type == "uv_warp"
                && ["object_from", "object_to"]
                    .iter()
                    .any(|key| modifier.params.get(*key).is_some_and(Value::is_string)))
            || (modifier.modifier_type == "warp"
                && ["object_from", "object_to"]
                    .iter()
                    .any(|key| modifier.params.get(*key).is_some_and(Value::is_string)))
            || (matches!(
                modifier.modifier_type.as_str(),
                "vertex_weight_edit" | "vertex_weight_mix" | "vertex_weight_proximity"
            ) && modifier
                .params
                .get("mask_texture")
                .is_some_and(|value| !value.is_null()));
        if modifier.modifier_type == "armature"
            || modifier.modifier_type == "lattice"
            || matches!(
                modifier.modifier_type.as_str(),
                "mesh_sequence_cache"
                    | "mesh_cache"
                    | "cloth"
                    | "soft_body"
                    | "collision"
                    | "dynamic_paint"
                    | "fluid"
                    | "particle_system"
                    | "particle_instance"
                    | "explode"
            )
            || (modifier.modifier_type == "volume_to_mesh"
                && modifier.params.get("object").is_some_and(Value::is_string))
            || scene_operand_modifier
        {
            custom_modifier(&mut evaluated, modifier, &mut warnings)?;
            continue;
        }
        if modifier.modifier_type == "nodes" {
            let group_text = modifier
                .params
                .get("node_group")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    invalid_parameter(modifier, "node_group", "a valid node-group ID")
                })?;
            let group_id = Id::new(group_text)
                .map_err(|_| invalid_parameter(modifier, "node_group", "a valid node-group ID"))?;
            let group = node_groups.get(&group_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    format!("node group `{group_id}` was not found"),
                    json!({"modifier_id":modifier.id,"node_group":group_id}),
                )
            })?;
            let overrides = match schema_value(modifier, "inputs")? {
                Value::Object(overrides) => overrides.clone(),
                _ => {
                    return Err(invalid_parameter(
                        modifier,
                        "inputs",
                        "an object keyed by interface socket ID or name",
                    ));
                }
            };
            let graph_result =
                graph::evaluate_at_frame(group, Some(&evaluated), &overrides, frame)?;
            evaluated = graph_result.mesh;
            instance_paths.extend(graph_result.instance_paths);
            continue;
        }
        if modifier.modifier_type == "mesh_to_volume" {
            continue;
        }
        if modifier.modifier_type == "surface_deform" {
            match deform::surface_deform(&evaluated, modifier) {
                Ok(deformed) => evaluated = deformed,
                Err(error)
                    if error
                        .details
                        .get("modifier_warning")
                        .and_then(Value::as_bool)
                        == Some(true) =>
                {
                    warnings.push(error);
                }
                Err(error) => return Err(error),
            }
            continue;
        }
        // Blender's setter has settled custom normals before a modifier reads them (clnor.rs).
        settle_custom_normals(&mut evaluated)?;
        let attributes_before = attribute_labels(&evaluated);
        evaluated = match modifier.modifier_type.as_str() {
            "mirror" => mirror(&evaluated, modifier)?,
            "array" => array(
                &evaluated,
                modifier,
                owner_world_matrix,
                object_world_matrices,
                object_meshes,
                evaluated_object_meshes,
                curve_path_lengths,
            )?,
            "subdivision" | "multires" => subdivision(&evaluated, modifier, context)?,
            "solidify" => solidify(&evaluated, modifier)?,
            "triangulate" => triangulate(&evaluated, modifier)?,
            "bevel" => bevel(&evaluated, modifier)?,
            "decimate" => without_custom_normals(decimate(&evaluated, modifier)?),
            "weld" => weld(&evaluated, modifier)?,
            "displace" => displace(&evaluated, modifier, owner_world_matrix)?,
            "smooth" => smooth(&evaluated, modifier)?,
            "simple_deform" => advanced::simple_deform(&evaluated, modifier)?,
            "screw" => advanced::screw(&evaluated, modifier, context)?,
            "shrinkwrap" => advanced::shrinkwrap(&evaluated, modifier)?,
            "curve" => advanced::curve(&evaluated, modifier)?,
            "hook" => advanced::hook(&evaluated, modifier)?,
            "skin" => advanced::skin_with_warnings(&evaluated, modifier, &mut warnings)?,
            "cast" => advanced::cast(&evaluated, modifier)?,
            "wave" => advanced::wave(&evaluated, modifier, frame)?,
            "warp" => advanced::warp(&evaluated, modifier)?,
            "vertex_weight_edit"
            | "vertex_weight_mix"
            | "vertex_weight_proximity"
            | "weighted_normal"
            | "normal_edit"
            | "uv_project"
            | "uv_warp"
            | "data_transfer" => attributes::evaluate(&evaluated, modifier)?,
            "laplacian_smooth" => advanced::laplacian_smooth(&evaluated, modifier)?,
            "laplacian_deform" => deform::laplacian_deform(&evaluated, modifier)?,
            "corrective_smooth" => advanced::corrective_smooth(&evaluated, modifier)?,
            "build" => advanced::build(&evaluated, modifier, frame)?,
            "edge_split" => advanced::edge_split(&evaluated, modifier)?,
            "mask" => advanced::mask(&evaluated, modifier)?,
            "wireframe" => advanced::wireframe(&evaluated, modifier)?,
            "boolean" => boolean_modifier(&evaluated, modifier)?,
            "mesh_deform" => deform::mesh_deform(&evaluated, modifier)?,
            "remesh" => remesh_modes::remesh_modifier(&evaluated, modifier)?,
            "ocean" => ocean::ocean(&evaluated, modifier, frame, context)?,
            _ => return Err(unsupported_modifier(modifier)),
        };
        let dropped: Vec<String> = attributes_before
            .difference(&attribute_labels(&evaluated))
            .cloned()
            .collect();
        if !dropped.is_empty() {
            warnings.push(attribute_dropped_warning(modifier, &dropped));
        }
    }
    Ok(ModifierEvaluation {
        mesh: evaluated,
        instance_paths,
        warnings,
    })
}

/// Evaluate the enabled modifier stack over a source mesh without node-group context.
///
/// # Errors
///
/// Returns an error for unsupported modifiers, invalid parameters, or invalid input.
pub fn evaluate_modifiers_with_groups(
    mesh: &Mesh,
    modifiers: &[Modifier],
    node_groups: &Registry<NodeGroup>,
) -> Result<ModifierEvaluation> {
    evaluate_modifiers_with_groups_at_frame(mesh, modifiers, node_groups, 1.0)
}

/// Evaluate the enabled modifier stack using the requested scene frame.
///
/// # Errors
///
/// Returns an error for unsupported modifiers, invalid parameters, invalid input,
/// or a non-finite frame.
pub fn evaluate_modifiers_with_groups_at_frame(
    mesh: &Mesh,
    modifiers: &[Modifier],
    node_groups: &Registry<NodeGroup>,
    frame: f64,
) -> Result<ModifierEvaluation> {
    evaluate_modifiers_with_groups_and_custom_at_frame(
        mesh.clone(),
        modifiers,
        node_groups,
        frame,
        |_, modifier, _warnings| Err(unsupported_modifier(modifier)),
    )
}

/// The `modifier.bevel.uv_profile` rejection for a bevel on UV layers whose beveled corners the
/// vertex-mesh construction does not cover; `rejections` lists them (never empty).
fn bevel_uv_profile_error(
    modifier: &Modifier,
    mesh: &Mesh,
    profile: f64,
    rejections: &[bevel_vmesh::CornerRejection],
) -> PotError {
    let rejected: Vec<Value> = rejections
        .iter()
        .map(|rejection| {
            json!({
                "vertex_id": mesh.vertices[rejection.vertex].id,
                "valence": rejection.valence,
                "selected_edges": rejection.selected,
                "reason": rejection.reason,
            })
        })
        .collect();
    let (parameter, message) = if bevel_vmesh::profile_super_r(profile)
        == bevel_vmesh::PRO_SQUARE_IN_R
    {
        (
            "profile",
            "bevel with UV layers does not support profile 0 (square inward) on EDGES; use a profile above 0"
                .to_owned(),
        )
    } else {
        let detail = rejections.first().map_or_else(String::new, |worst| {
            format!(
                "; corner at vertex {} has valence {} / selected {} edges: {}",
                mesh.vertices[worst.vertex].id, worst.valence, worst.selected, worst.reason
            )
        });
        (
            "segments",
            format!(
                "bevel with UV layers supports only three-edge cube-like corners with every edge selected{detail}"
            ),
        )
    };
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        message,
        json!({
            "feature_id": "modifier.bevel.uv_profile",
            "modifier_id": modifier.id,
            "parameter": parameter,
            "rejected_corners": rejected,
        }),
    )
}

fn unsupported_modifier(modifier: &Modifier) -> PotError {
    let feature_id = format!("modifier.{}", modifier.modifier_type);
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!(
            "modifier `{}` requires scene context that is unavailable in the standalone evaluator",
            modifier.modifier_type,
        ),
        json!({"feature_id":feature_id}),
    )
}
fn modifier_depends_on_topology(modifier: &Modifier) -> bool {
    matches!(
        modifier.modifier_type.as_str(),
        "array"
            | "bevel"
            | "boolean"
            | "build"
            | "decimate"
            | "edge_split"
            | "mask"
            | "mirror"
            | "multires"
            | "remesh"
            | "screw"
            | "skin"
            | "smooth"
            | "solidify"
            | "subdivision"
            | "triangulate"
            | "weld"
            | "wireframe"
    )
}
pub(crate) fn validate_boolean_topology_suffix(
    modifiers: &[Modifier],
    modifier_index: usize,
    is_active: &impl Fn(usize, &Modifier) -> bool,
) -> Result<()> {
    let Some(boolean_modifier) = modifiers.get(modifier_index) else {
        return Ok(());
    };
    if !is_active(modifier_index, boolean_modifier) || boolean_modifier.modifier_type != "boolean" {
        return Ok(());
    }
    let Some(downstream) = modifiers
        .iter()
        .enumerate()
        .skip(modifier_index + 1)
        .find(|(index, candidate)| {
            is_active(*index, candidate) && modifier_depends_on_topology(candidate)
        })
        .map(|(_, modifier)| modifier)
    else {
        return Ok(());
    };
    Err(PotError::with_details(
        ErrorCode::UnsupportedFeature,
        "Boolean output topology is not Blender-compatible for a downstream topology-dependent modifier",
        json!({
            "feature_id":"modifier.boolean.downstream_topology",
            "modifier_id":boolean_modifier.id,
            "solver":boolean_modifier.params.get("solver").and_then(Value::as_str),
            "downstream_modifier_id":downstream.id,
            "downstream_modifier_type":downstream.modifier_type
        }),
    ))
}

fn invalid_parameter(modifier: &Modifier, parameter: &str, expected: &str) -> PotError {
    let (name, pointer) = match parameter {
        "bind_data" | "binding_data" => ("binding_data", "/binding_data".to_owned()),
        _ => (parameter, format!("/params/{parameter}")),
    };
    PotError::with_details(
        ErrorCode::InvalidArgument,
        format!(
            "invalid `{name}` parameter for modifier `{}`; expected {expected}",
            modifier.modifier_type
        ),
        json!({
            "modifier_id": modifier.id,
            "parameter": name,
            "pointer": pointer
        }),
    )
}

/// Blender `solver` of the Boolean modifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BooleanSolver {
    /// Mesh Arrangements (`geom::boolean`).
    Exact,
    /// Blender's FLOAT solver. Potter uses the BSP shortcut until `PR17b` ports Blender's float path.
    Float,
    /// Blender's MANIFOLD solver, which Potter does not implement.
    Manifold,
}

/// Blender `operand_type` of the Boolean modifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BooleanOperandType {
    Object,
    Collection,
}

/// Validated Boolean modifier parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BooleanSettings {
    pub solver: BooleanSolver,
    pub operation: super::boolean::Operation,
    pub operand_type: BooleanOperandType,
    pub use_self: bool,
    pub hole_tolerant: bool,
    pub material_mode: super::boolean::MaterialMode,
}

impl BooleanSettings {
    /// Reads the Boolean parameters with Blender's defaults. `debug_options` is not read (Blender
    /// reads it only in debug builds), and `double_threshold` is FLOAT-only.
    pub(crate) fn parse(modifier: &Modifier) -> Result<Self> {
        let solver = match boolean_text(modifier, "solver", "EXACT")? {
            "EXACT" => BooleanSolver::Exact,
            "FLOAT" => BooleanSolver::Float,
            "MANIFOLD" => BooleanSolver::Manifold,
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "solver",
                    "EXACT, FLOAT, or MANIFOLD",
                ));
            }
        };
        let operation =
            super::boolean::Operation::parse(boolean_text(modifier, "operation", "DIFFERENCE")?)
                .map_err(|_| {
                    invalid_parameter(modifier, "operation", "UNION, DIFFERENCE, or INTERSECT")
                })?;
        let operand_type = match boolean_text(modifier, "operand_type", "OBJECT")? {
            "OBJECT" => BooleanOperandType::Object,
            "COLLECTION" => BooleanOperandType::Collection,
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "operand_type",
                    "OBJECT or COLLECTION",
                ));
            }
        };
        let material_mode = match boolean_text(modifier, "material_mode", "INDEX")? {
            "INDEX" => super::boolean::MaterialMode::Index,
            "TRANSFER" => super::boolean::MaterialMode::Transfer,
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "material_mode",
                    "INDEX or TRANSFER",
                ));
            }
        };
        Ok(Self {
            solver,
            operation,
            operand_type,
            use_self: boolean_flag(modifier, "use_self")?,
            hole_tolerant: boolean_flag(modifier, "use_hole_tolerant")?,
            material_mode,
        })
    }
}

/// A Boolean string parameter, or `default` when it is absent.
fn boolean_text<'m>(
    modifier: &'m Modifier,
    parameter: &str,
    default: &'static str,
) -> Result<&'m str> {
    match modifier.params.get(parameter) {
        None => Ok(default),
        Some(value) => value
            .as_str()
            .ok_or_else(|| invalid_parameter(modifier, parameter, "a string")),
    }
}

/// A Boolean flag parameter, `false` when it is absent.
fn boolean_flag(modifier: &Modifier, parameter: &str) -> Result<bool> {
    match modifier.params.get(parameter) {
        None => Ok(false),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| invalid_parameter(modifier, parameter, "true or false")),
    }
}

fn boolean_modifier(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let settings = BooleanSettings::parse(modifier)?;
    match settings.solver {
        BooleanSolver::Exact => {
            // Blender `non_float_boolean_mesh` returns the mesh unchanged when no Object is set. A
            // collection with no members still runs the solver, which removes self-intersections.
            if settings.operand_type == BooleanOperandType::Object
                && modifier.runtime.boolean_operands.is_empty()
            {
                return Ok(mesh.clone());
            }
            super::boolean::exact_boolean(
                mesh,
                &modifier.runtime.boolean_operands,
                settings.operation,
                settings.use_self,
                settings.hole_tolerant,
            )
        }
        BooleanSolver::Manifold => Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Boolean MANIFOLD solver is not supported",
            json!({
                "feature_id": "modifier.boolean.solver_manifold",
                "modifier_id": modifier.id,
                "solver": "MANIFOLD",
            }),
        )),
        // shortcut: FLOAT is the pre-PR17 BSP until PR17b ports Blender's float path.
        BooleanSolver::Float => {
            let operand =
                modifier.runtime.operand_mesh.clone().ok_or_else(|| {
                    invalid_parameter(modifier, "object", "a mesh object operand")
                })?;
            let operand: Mesh = serde_json::from_value(operand)
                .map_err(|_| invalid_parameter(modifier, "object", "a mesh object operand"))?;
            super::boolean::boolean_mesh_op(mesh, &operand, settings.operation)
        }
    }
}

fn mesh_error(error: &MeshError) -> PotError {
    PotError::with_details(
        ErrorCode::InvalidArgument,
        format!("modifier produced invalid mesh: {error}"),
        json!({ "mesh_error": error.to_string() }),
    )
}

fn validated(mesh: &Mesh) -> Result<()> {
    mesh.validate().map_err(|error| mesh_error(&error))
}

fn parameter_value<'a>(modifier: &'a Modifier, name: &str) -> Option<&'a Value> {
    modifier
        .params
        .get(name)
        .or_else(|| params::default_value(ParameterFamily::Modifier, &modifier.modifier_type, name))
}

/// The modifier's value for `name`, else its Blender default from `param_table.json`.
fn schema_value<'a>(modifier: &'a Modifier, name: &str) -> Result<&'a Value> {
    parameter_value(modifier, name).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::InternalError,
            format!(
                "modifier type `{}` has no value or schema default for parameter `{name}`",
                modifier.modifier_type
            ),
            json!({
                "modifier_id": modifier.id,
                "modifier_type": modifier.modifier_type,
                "parameter": name
            }),
        )
    })
}

fn string_param<'a>(modifier: &'a Modifier, name: &str) -> Result<&'a str> {
    string_param_with_expected(modifier, name, "a string")
}

fn string_param_with_expected<'a>(
    modifier: &'a Modifier,
    name: &str,
    expected: &str,
) -> Result<&'a str> {
    schema_value(modifier, name)?
        .as_str()
        .ok_or_else(|| invalid_parameter(modifier, name, expected))
}

/// Object-reference parameter: `None` when the value is null or absent.
fn optional_string_param<'a>(modifier: &'a Modifier, name: &str) -> Result<Option<&'a str>> {
    match parameter_value(modifier, name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(Some)
            .ok_or_else(|| invalid_parameter(modifier, name, "a string or null")),
    }
}

pub(super) fn number_param(modifier: &Modifier, name: &str) -> Result<f64> {
    schema_value(modifier, name)?
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| invalid_parameter(modifier, name, "a finite number"))
}

pub(super) fn bool_param(modifier: &Modifier, name: &str) -> Result<bool> {
    schema_value(modifier, name)?
        .as_bool()
        .ok_or_else(|| invalid_parameter(modifier, name, "a boolean"))
}

fn uint_param(modifier: &Modifier, name: &str) -> Result<u64> {
    schema_value(modifier, name)?
        .as_u64()
        .ok_or_else(|| invalid_parameter(modifier, name, "a non-negative integer"))
}

fn vec3_param(modifier: &Modifier, name: &str) -> Result<DVec3> {
    let value = schema_value(modifier, name)?;
    let components = if let Some(array) = value.as_array() {
        if array.len() != 3 {
            return Err(invalid_parameter(modifier, name, "three finite numbers"));
        }
        [array[0].as_f64(), array[1].as_f64(), array[2].as_f64()]
    } else if let Some(object) = value.as_object() {
        [
            object.get("x").and_then(Value::as_f64),
            object.get("y").and_then(Value::as_f64),
            object.get("z").and_then(Value::as_f64),
        ]
    } else {
        return Err(invalid_parameter(modifier, name, "three finite numbers"));
    };
    let [Some(x), Some(y), Some(z)] = components else {
        return Err(invalid_parameter(modifier, name, "three finite numbers"));
    };
    let result = DVec3::new(x, y, z);
    if result.is_finite() {
        Ok(result)
    } else {
        Err(invalid_parameter(modifier, name, "three finite numbers"))
    }
}

fn positions(mesh: &Mesh) -> Vec<DVec3> {
    mesh.vertices.iter().map(|vertex| vertex.co).collect()
}

fn closest_point_triangle(point: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denominator = 1.0 / (va + vb + vc);
    a + ab * (vb * denominator) + ac * (vc * denominator)
}

fn id_indices(mesh: &Mesh) -> HashMap<u32, usize> {
    mesh.vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect()
}

fn polygons(mesh: &Mesh) -> Result<Vec<Polygon>> {
    let indices = id_indices(mesh);
    mesh.faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| {
                    indices.get(id).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InvalidArgument,
                            "modifier input references a missing vertex",
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()
                .map(|vertices| (vertices, face.material_index))
        })
        .collect()
}

/// Every edge of `mesh` as a pair of storage indices, in `mesh.edges` order.
fn edge_pairs(mesh: &Mesh) -> Result<Vec<[usize; 2]>> {
    let indices = id_indices(mesh);
    mesh.edges
        .iter()
        .map(|edge| {
            Ok([
                *indices.get(&edge.vertices[0]).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InvalidArgument,
                        "modifier input edge references a missing vertex",
                    )
                })?,
                *indices.get(&edge.vertices[1]).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InvalidArgument,
                        "modifier input edge references a missing vertex",
                    )
                })?,
            ])
        })
        .collect()
}

fn loose_edges(mesh: &Mesh) -> Result<Vec<[usize; 2]>> {
    let mut face_edges = HashSet::new();
    for face in &mesh.faces {
        for (first, second) in cyclic_pairs(&face.vertices) {
            face_edges.insert(edge_key(*first, *second));
        }
    }
    Ok(edge_pairs(mesh)?
        .into_iter()
        .zip(&mesh.edges)
        .filter(|(_, edge)| !face_edges.contains(&edge_key(edge.vertices[0], edge.vertices[1])))
        .map(|(pair, _)| pair)
        .collect())
}

fn build_mesh(
    points: Vec<DVec3>,
    polygons: Vec<Polygon>,
    loose_edges: Vec<[usize; 2]>,
    origin: &Origin,
    source: &Mesh,
) -> Result<Mesh> {
    let mut result = Mesh::new();
    for point in points {
        result
            .insert_vertex(point)
            .map_err(|error| mesh_error(&error))?;
    }
    let vertex_ids = result
        .vertices
        .iter()
        .map(|vertex| vertex.id)
        .collect::<Vec<_>>();
    let faces = polygons.into_iter().map(|(indices, material_index)| {
        let ids = indices
            .into_iter()
            .map(|index| {
                vertex_ids
                    .get(index)
                    .copied()
                    .ok_or(MeshError::InvalidTopology(
                        "generated polygon index is out of range",
                    ))
            })
            .collect::<std::result::Result<Vec<_>, MeshError>>()?;
        Ok((ids, material_index))
    });
    result
        .insert_faces(faces)
        .map_err(|error| mesh_error(&error))?;
    for [first, second] in loose_edges {
        let Some(first_id) = result.vertices.get(first).map(|vertex| vertex.id) else {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "generated edge index is out of range",
            ));
        };
        let Some(second_id) = result.vertices.get(second).map(|vertex| vertex.id) else {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "generated edge index is out of range",
            ));
        };
        if first_id != second_id
            && !result.edges.iter().any(|edge| {
                edge_key(edge.vertices[0], edge.vertices[1]) == edge_key(first_id, second_id)
            })
        {
            result
                .insert_edge([first_id, second_id])
                .map_err(|error| mesh_error(&error))?;
        }
    }
    remap_attributes(source, &mut result, origin)?;
    validated(&result)?;
    Ok(result)
}

/// Labels of the element-keyed data a modifier must carry through a rebuild: each top-level
/// attribute, and each `blender_attributes` entry by name. Skin inputs are consumed by the Skin
/// modifier itself, and wireframe settings are Potter-only modifier state, so neither is a label.
fn attribute_labels(mesh: &Mesh) -> BTreeSet<String> {
    let mut labels = BTreeSet::new();
    for (key, value) in &mesh.attributes {
        if matches!(
            key.as_str(),
            "skin_radii"
                | "skin_roots"
                | "blender_skin_vertices"
                | "wireframe_use_crease"
                | "wireframe_crease_weight"
        ) {
            continue;
        }
        labels.insert(key.clone());
        if key == "blender_attributes" {
            for attribute in value.as_array().into_iter().flatten() {
                if let Some(name) = attribute.get("name").and_then(Value::as_str) {
                    labels.insert(format!("blender_attributes:{name}"));
                }
            }
        }
    }
    labels
}

fn attribute_dropped_warning(modifier: &Modifier, dropped: &[String]) -> PotError {
    PotError::with_details(
        ErrorCode::UnrepresentableFeature,
        format!(
            "modifier `{}` cannot carry mesh attributes through this rebuild: {}",
            modifier.id,
            dropped.join(", ")
        ),
        json!({
            "feature_id": "modifier.attribute_dropped",
            "status": "dropped",
            "modifier_id": modifier.id,
            "modifier_type": modifier.modifier_type,
            "attributes": dropped,
        }),
    )
}

fn cyclic_pairs<T>(items: &[T]) -> impl Iterator<Item = (&T, &T)> {
    items
        .iter()
        .zip(items.iter().cycle().skip(1))
        .take(items.len())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "mesh element counts are bounded by u32 and exactly representable in f64"
)]
fn usize_as_f64(value: usize) -> f64 {
    value as f64
}

/// Origin weights from `(source, weight)` pairs: repeated sources add up and zero weights drop out.
pub(super) fn blend_weights(pairs: impl IntoIterator<Item = (usize, f64)>) -> Vec<(usize, f64)> {
    let mut weights = BTreeMap::new();
    for (source, weight) in pairs {
        *weights.entry(source).or_insert(0.0) += weight;
    }
    weights
        .into_iter()
        .filter(|(_, weight)| *weight > 0.0)
        .collect()
}

/// Origin of a point at the equal-weight mean of `members`. `None` members carry no input data and
/// are left out; a point whose members all lack input data is new.
fn mean_origin(members: impl IntoIterator<Item = Option<usize>>) -> Vec<(usize, f64)> {
    let members: Vec<usize> = members.into_iter().flatten().collect();
    if members.is_empty() {
        return Vec::new();
    }
    let share = 1.0 / usize_as_f64(members.len());
    blend_weights(members.into_iter().map(|member| (member, share)))
}

/// The group a mirrored copy takes weight from: the `.L`/`.R` side-swapped name when the mesh has
/// that group, otherwise the name itself.
fn mirrored_group_name(name: &str, groups: &Map<String, Value>, flipped: bool) -> String {
    let swapped = name
        .strip_suffix(".L")
        .map(|stem| format!("{stem}.R"))
        .or_else(|| name.strip_suffix(".R").map(|stem| format!("{stem}.L")));
    match swapped {
        Some(swapped) if flipped && groups.contains_key(&swapped) => swapped,
        _ => name.to_owned(),
    }
}

/// Rewrites `vertex_groups` of a mirrored `output`. `members[o]` lists the input points merged into
/// output vertex `o`; input point `c * vertex_count + v` is copy `c` of source vertex `v`. A copy
/// with an odd number of mirrored axes takes each group under its side-swapped name, and a merged
/// point averages its members with absent weights counted as zero.
fn mirror_vertex_groups(
    source: &Mesh,
    output: &mut Mesh,
    members: &[Vec<usize>],
    vertex_count: usize,
) {
    let Some(source_groups) = source
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
    else {
        return;
    };
    let names: Vec<&str> = source_groups.keys().map(String::as_str).collect();
    let position: HashMap<&str, usize> = names
        .iter()
        .copied()
        .enumerate()
        .map(|(index, name)| (name, index))
        .collect();
    let group_values: Vec<&Value> = source_groups.values().collect();
    // `targets[usize::from(flipped)][group]`: the group a copy with that flip state takes weight from.
    // A mirrored name is an existing group or the name itself, so the fallback is never taken.
    let targets: [Vec<usize>; 2] = [false, true].map(|flipped| {
        names
            .iter()
            .enumerate()
            .map(|(group, name)| {
                let target = mirrored_group_name(name, source_groups, flipped);
                position.get(target.as_str()).copied().unwrap_or(group)
            })
            .collect()
    });
    let source_keys: Vec<String> = source
        .vertices
        .iter()
        .map(|vertex| format!("v{}", vertex.id))
        .collect();
    // Summed weight per group and output vertex; `None` when no copy has an entry for the group.
    let mut sums = vec![vec![None::<f64>; output.vertices.len()]; names.len()];
    for (vertex, copies) in members.iter().enumerate().take(output.vertices.len()) {
        let share = 1.0 / usize_as_f64(copies.len());
        for &copy in copies {
            let flipped = (copy / vertex_count).count_ones() % 2 == 1;
            let source_key = source_keys[copy % vertex_count].as_str();
            for (group, values) in group_values.iter().enumerate() {
                let Some(weight) = values.get(source_key).and_then(Value::as_f64) else {
                    continue;
                };
                let slot = &mut sums[targets[usize::from(flipped)][group]][vertex];
                *slot = Some(slot.unwrap_or(0.0) + weight * share);
            }
        }
    }
    let groups: Map<String, Value> = names
        .iter()
        .zip(sums)
        .map(|(name, sums)| {
            let weights: Map<String, Value> = sums
                .into_iter()
                .enumerate()
                .filter_map(|(vertex, sum)| {
                    Some((format!("v{}", output.vertices[vertex].id), json!(sum?)))
                })
                .collect();
            ((*name).to_owned(), Value::Object(weights))
        })
        .collect();
    output
        .attributes
        .insert("vertex_groups".to_owned(), Value::Object(groups));
}

fn clean_polygon(vertices: Vec<usize>) -> Option<Vec<usize>> {
    let mut clean = Vec::with_capacity(vertices.len());
    for vertex in vertices {
        if clean.last() != Some(&vertex) {
            clean.push(vertex);
        }
    }
    if clean.len() > 1 && clean.first() == clean.last() {
        clean.pop();
    }
    let unique: HashSet<_> = clean.iter().copied().collect();
    (clean.len() >= 3 && clean.len() == unique.len()).then_some(clean)
}

/// Blender's decimate rebuilds corner normals from geometry, so its output has no custom normals.
fn without_custom_normals(mut mesh: Mesh) -> Mesh {
    mesh.attributes.remove("custom_normal");
    mesh
}

/// `vector` mirrored across each axis `axes[bit]` whose bit is set in `mask`.
fn reflect_vector(mut vector: DVec3, mask: usize, axes: &[usize]) -> DVec3 {
    for (bit, &axis) in axes.iter().enumerate() {
        if mask & (1 << bit) != 0 {
            vector[axis] = -vector[axis];
        }
    }
    vector
}

/// Blender's mirror loop numbering and reflected vectors for every copy corner, before the merge, in
/// output polygon order (`mesh_mirror.cc`). Copy `c` is appended after the copies below its top axis
/// bit `h`, so its loops start at `N * 2^h` plus the first loop of the copied face in that prefix. A
/// reversed copy (odd reflection count) keeps its first corner and reverses the rest, as
/// `mesh_flip_faces()` in `mesh_flip_faces.cc` and the `mirrorj` mapping in
/// `BKE_mesh_mirror_apply_mirror_on_axis_for_modifier` (`mesh_mirror.cc`) assume: copy corner `m`
/// holds source corner `(size - m) % size`.
fn mirror_copy_states(
    mesh: &Mesh,
    source_order: &[usize],
    source_vectors: Option<&[Option<DVec3>]>,
    copy_count: usize,
    axes: &[usize],
) -> (Vec<usize>, Vec<Option<DVec3>>) {
    let corners = source_order.len();
    let mut ordinals = vec![0; copy_count * corners];
    let mut vectors = vec![None; copy_count * corners];
    for copy in 0..copy_count {
        let reversed = copy.count_ones() % 2 == 1;
        let (prefix, parent) = if copy == 0 {
            (0, 0)
        } else {
            let top = copy.ilog2();
            (corners << top, copy ^ (1 << top))
        };
        let mut start = 0;
        for face in &mesh.faces {
            let size = face.vertices.len();
            let parent_first = if copy == 0 {
                0
            } else {
                (start..start + size)
                    .map(|corner| ordinals[parent * corners + corner])
                    .min()
                    .unwrap_or(0)
            };
            for m in 0..size {
                let slot = copy * corners + start + m;
                let source = start + if reversed { (size - m) % size } else { m };
                ordinals[slot] = if copy == 0 {
                    source_order[source]
                } else {
                    prefix + parent_first + m
                };
                vectors[slot] = source_vectors
                    .and_then(|vectors| vectors[source])
                    .map(|vector| reflect_vector(vector, copy, axes));
            }
            start += size;
        }
    }
    (ordinals, vectors)
}

/// The merged output's vectors and loop basis. Each kept corner takes the vector and loop of the copy
/// corner whose point merged into it; its loop is ranked among the kept corners, so the basis is a
/// permutation of the output corners.
fn mirror_output_states(
    merged: &MergedGeometry,
    copy_corners: &[Vec<usize>],
    ordinals: &[usize],
    vectors: &[Option<DVec3>],
) -> Result<(Vec<Option<DVec3>>, Vec<usize>)> {
    let mut offsets = Vec::with_capacity(copy_corners.len());
    let mut total = 0;
    for points in copy_corners {
        offsets.push(total);
        total += points.len();
    }
    let mut merged_vectors = Vec::new();
    let mut merged_loops = Vec::new();
    for ((polygon, _), &source) in merged.polygons.iter().zip(&merged.polygon_sources) {
        for &point in polygon {
            let members = &merged.point_members[point];
            let corner = copy_corners[source]
                .iter()
                .position(|copy_point| members.contains(copy_point))
                .ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "mirror merge lost a copy corner")
                })?;
            let copy = offsets[source] + corner;
            merged_vectors.push(vectors[copy]);
            merged_loops.push(ordinals[copy]);
        }
    }
    let mut by_loop: Vec<usize> = (0..merged_loops.len()).collect();
    by_loop.sort_unstable_by_key(|&corner| merged_loops[corner]);
    let mut order = vec![0; merged_loops.len()];
    for (rank, corner) in by_loop.into_iter().enumerate() {
        order[corner] = rank;
    }
    Ok((merged_vectors, order))
}

fn mirror(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    let enabled = if let Some(value) = parameter_value(modifier, "use_axis") {
        let values = value
            .as_array()
            .filter(|values| values.len() == 3)
            .ok_or_else(|| invalid_parameter(modifier, "use_axis", "three booleans"))?;
        [
            values[0]
                .as_bool()
                .ok_or_else(|| invalid_parameter(modifier, "use_axis", "three booleans"))?,
            values[1]
                .as_bool()
                .ok_or_else(|| invalid_parameter(modifier, "use_axis", "three booleans"))?,
            values[2]
                .as_bool()
                .ok_or_else(|| invalid_parameter(modifier, "use_axis", "three booleans"))?,
        ]
    } else {
        [true, false, false]
    };
    // Blender cuts the mesh at a mirrored axis's plane when that axis is bisected; a bisect on an
    // axis that is not mirrored changes nothing, so only the mirrored case is refused.
    let bisect: Vec<bool> = parameter_value(modifier, "use_bisect_axis")
        .and_then(Value::as_array)
        .map(|flags| {
            flags
                .iter()
                .map(|flag| flag.as_bool() == Some(true))
                .collect()
        })
        .unwrap_or_default();
    if let Some(axis) = (0..3).find(|&axis| enabled[axis] && bisect.get(axis) == Some(&true)) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Blender mirror bisect on a mirrored axis is not supported",
            json!({"feature_id": "modifier.mirror.bisect", "modifier_id": modifier.id, "axis": axis}),
        ));
    }
    let selected: Vec<_> = enabled
        .iter()
        .enumerate()
        .filter_map(|(i, value)| value.then_some(i))
        .collect();
    if selected.is_empty() {
        return Ok(mesh.clone());
    }
    let merge = bool_param(modifier, "use_mirror_merge")?;
    let threshold = number_param(modifier, "merge_threshold")?;
    if threshold < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "merge_threshold",
            "a non-negative finite number",
        ));
    }
    let source_points = positions(mesh);
    let source_polygons = polygons(mesh)?;
    let source_loose_edges = loose_edges(mesh)?;
    let copy_count = 1_usize
        .checked_shl(
            u32::try_from(selected.len()).map_err(|_| mesh_error(&MeshError::IdExhausted))?,
        )
        .ok_or_else(|| mesh_error(&MeshError::IdExhausted))?;
    let total_points = source_points
        .len()
        .checked_mul(copy_count)
        .ok_or_else(|| mesh_error(&MeshError::IdExhausted))?;
    let mut points = Vec::with_capacity(total_points);
    let mut output_polygons = Vec::with_capacity(source_polygons.len().saturating_mul(copy_count));
    let mut output_loose_edges =
        Vec::with_capacity(source_loose_edges.len().saturating_mul(copy_count));
    for mask in 0..copy_count {
        let base = points.len();
        for point in &source_points {
            let mut reflected = *point;
            for (bit, axis_index) in selected.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    reflected[*axis_index] = -reflected[*axis_index];
                }
            }
            points.push(reflected);
        }
        let reverse = mask.count_ones() % 2 == 1;
        for (face, material_index) in &source_polygons {
            let mut mapped: Vec<_> = face.iter().map(|index| base + *index).collect();
            if reverse && !mapped.is_empty() {
                // Blender's flip keeps the first corner and reverses the rest: `mesh_flip_faces()`
                // (`mesh_flip_faces.cc`), applied to the mirrored faces by
                // `BKE_mesh_mirror_apply_mirror_on_axis_for_modifier` (`mesh_mirror.cc`).
                mapped[1..].reverse();
            }
            output_polygons.push((mapped, *material_index));
        }
        output_loose_edges.extend(
            source_loose_edges
                .iter()
                .map(|[a, b]| [base + *a, base + *b]),
        );
    }
    // Blender numbers the mirror's loops after the source's and reflects each copy's custom normals
    // (`mesh_mirror.cc`). The merged output keeps the loops of the corners that survive the merge.
    let has_values = clnor::corner_codes(mesh)?.is_some();
    let has_state = has_values || clnor::corner_order(mesh)?.is_some();
    let copy_corners: Vec<Vec<usize>> = output_polygons
        .iter()
        .map(|(face, _)| face.clone())
        .collect();
    let copy_states = if has_state {
        let source_order =
            clnor::corner_order(mesh)?.unwrap_or_else(|| clnor::stored_corner_order(mesh));
        let source_vectors = if has_values {
            clnor::corner_custom_normals(mesh, &source_points, NormalCodeOptions::default())?
        } else {
            None
        };
        Some(mirror_copy_states(
            mesh,
            &source_order,
            source_vectors.as_deref(),
            copy_count,
            &selected,
        ))
    } else {
        None
    };
    let merged = if merge {
        merge_geometry(&points, output_polygons, output_loose_edges, threshold)?
    } else {
        MergedGeometry::unmerged(points, output_polygons, output_loose_edges)
    };
    let merged_state = copy_states
        .map(|(ordinals, vectors)| {
            mirror_output_states(&merged, &copy_corners, &ordinals, &vectors)
        })
        .transpose()?;
    // Blender's merge keeps one source edge's flags per merged edge (see `first_source_edges`).
    let first_sources = if merge {
        let edges = edge_pairs(mesh)?;
        let pre_merge: Vec<[usize; 2]> = (0..copy_count)
            .flat_map(|copy| {
                let base = copy * source_points.len();
                edges
                    .iter()
                    .map(move |[first, second]| [base + first, base + second])
            })
            .collect();
        Some(first_source_edges(&merged, &pre_merge))
    } else {
        None
    };
    let vertex_count = source_points.len();
    let origin = Origin {
        vertices: merged
            .point_members
            .iter()
            .map(|members| mean_origin(members.iter().map(|&point| Some(point % vertex_count))))
            .collect(),
        faces: merged
            .polygon_sources
            .iter()
            .map(|&polygon| Some(polygon % source_polygons.len()))
            .collect(),
        corners: None,
    };
    // Each copy has one mirror pass per mirrored axis in its copy mask, so its UVs take that many
    // passes; merged polygons keep the lowest copy they came from, as in the origin above.
    let passes: Vec<usize> = merged
        .polygon_sources
        .iter()
        .map(|&polygon| (polygon / source_polygons.len()).count_ones() as usize)
        .collect();
    let mut result = build_mesh(merged.points, merged.polygons, merged.edges, &origin, mesh)?;
    if let Some(first) = &first_sources {
        let edge_count = mesh.edges.len();
        apply_first_source_edge_flags(&mut result, mesh, first, |edge| edge % edge_count)?;
    }
    if let Some((vectors, order)) = &merged_state {
        // build_mesh carried a provisional basis across; the mirror's own numbering replaces it.
        result.attributes.remove("custom_normal");
        let options = NormalCodeOptions {
            corner_order: Some(order.as_slice()),
            setter_input: clnor::SetterInput::NormalizeInput,
        };
        if has_values {
            clnor::encode_vectors(&mut result, Some(vectors.as_slice()), options)?;
        } else {
            clnor::write_codes(&mut result, None, options)?;
        }
    }
    if bool_param(modifier, "use_mirror_vertex_groups")? {
        mirror_vertex_groups(mesh, &mut result, &merged.point_members, vertex_count);
    }
    let flip = [
        bool_param(modifier, "use_mirror_u")?,
        bool_param(modifier, "use_mirror_v")?,
    ];
    let mirror_offset = [
        number_param(modifier, "mirror_offset_u")?,
        number_param(modifier, "mirror_offset_v")?,
    ];
    let offset = [
        number_param(modifier, "offset_u")?,
        number_param(modifier, "offset_v")?,
    ];
    let udim = bool_param(modifier, "use_mirror_udim")?;
    map_uv_values(&mut result, |face, _, uv| {
        (0..passes[face]).fold(uv, |uv, _| {
            mirror_uv_pass(uv, flip, mirror_offset, offset, udim)
        })
    });
    Ok(result)
}

fn array(
    mesh: &Mesh,
    modifier: &Modifier,
    owner_world_matrix: DMat4,
    object_world_matrices: &BTreeMap<Id, DMat4>,
    object_meshes: &BTreeMap<Id, &Mesh>,
    evaluated_object_meshes: &BTreeMap<Id, Mesh>,
    curve_path_lengths: &BTreeMap<Id, f64>,
) -> Result<Mesh> {
    validated(mesh)?;
    let mut count = uint_param(modifier, "count")?;
    if count == 0 || count > 10_000 {
        return Err(invalid_parameter(
            modifier,
            "count",
            "an integer between 1 and 10000",
        ));
    }
    let start_cap = array_cap_mesh(
        modifier,
        "start_cap",
        object_meshes,
        evaluated_object_meshes,
    )?;
    let end_cap = array_cap_mesh(modifier, "end_cap", object_meshes, evaluated_object_meshes)?;
    let use_relative = bool_param(modifier, "use_relative_offset")?;
    let use_constant = bool_param(modifier, "use_constant_offset")?;
    let relative = vec3_param(modifier, "relative_offset_displace")?;
    let constant = vec3_param(modifier, "constant_offset_displace")?;
    let dimensions = mesh.bounds().map_or(DVec3::ZERO, super::Aabb::size);
    let increment = (if use_relative {
        dimensions * relative
    } else {
        DVec3::ZERO
    }) + if use_constant { constant } else { DVec3::ZERO };
    if !increment.is_finite() {
        return Err(invalid_parameter(
            modifier,
            "relative_offset_displace",
            "an offset that produces finite coordinates",
        ));
    }
    let mut step = DMat4::from_translation(increment);
    if bool_param(modifier, "use_object_offset")?
        && let Some(target) = optional_string_param(modifier, "offset_object")
            .map_err(|_| invalid_parameter(modifier, "offset_object", "an object ID or null"))?
    {
        let target_id = Id::new(target)
            .map_err(|_| invalid_parameter(modifier, "offset_object", "a valid object ID"))?;
        let target_world = object_world_matrices.get(&target_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "array offset object was not found",
                json!({"modifier_id":modifier.id,"offset_object":target_id}),
            )
        })?;
        step = DMat4::from_translation(increment) * owner_world_matrix.inverse() * *target_world;
    }
    if !step
        .to_cols_array()
        .iter()
        .all(|component| component.is_finite())
    {
        return Err(invalid_parameter(
            modifier,
            "offset_object",
            "an object offset that produces a finite transform",
        ));
    }
    let fit_type = string_param_with_expected(modifier, "fit_type", "a Blender array fit type")?;
    if !matches!(fit_type, "FIXED_COUNT" | "FIT_LENGTH" | "FIT_CURVE") {
        return Err(invalid_parameter(
            modifier,
            "fit_type",
            "FIXED_COUNT, FIT_LENGTH, or FIT_CURVE",
        ));
    }
    let fit_length = match fit_type {
        "FIT_LENGTH" => number_param(modifier, "fit_length")?,
        "FIT_CURVE" => array_curve_length(modifier, curve_path_lengths)?,
        _ => 0.0,
    };
    if fit_length < 0.0 {
        let parameter = if fit_type == "FIT_CURVE" {
            "curve"
        } else {
            "fit_length"
        };
        return Err(invalid_parameter(
            modifier,
            parameter,
            "a non-negative finite path length",
        ));
    }
    if fit_type == "FIT_LENGTH" || fit_type == "FIT_CURVE" {
        let distance = step.w_axis.truncate().length();
        count = if distance <= 1.0e-6 {
            1
        } else {
            let fitted = ((fit_length + 1.0e-6) / distance + 1.0).floor();
            if !fitted.is_finite() || fitted > 10_000.0 {
                return Err(invalid_parameter(
                    modifier,
                    if fit_type == "FIT_CURVE" {
                        "curve"
                    } else {
                        "fit_length"
                    },
                    "a length that generates at most 10000 copies",
                ));
            }
            fitted as u64
        };
    }
    let use_merge = bool_param(modifier, "use_merge_vertices")?;
    let merge_final = bool_param(modifier, "use_merge_vertices_cap")?;
    let merge_threshold = number_param(modifier, "merge_threshold")?;
    if merge_threshold < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "merge_threshold",
            "a non-negative finite number",
        ));
    }
    let offset_u = number_param(modifier, "offset_u")?;
    let offset_v = number_param(modifier, "offset_v")?;
    let source_points = positions(mesh);
    let source_polygons = polygons(mesh)?;
    let source_loose_edges = loose_edges(mesh)?;
    let count_usize = usize::try_from(count).map_err(|_| mesh_error(&MeshError::IdExhausted))?;
    let start_cap_parts = if let Some(cap) = start_cap {
        validated(cap)?;
        Some((positions(cap), polygons(cap)?, loose_edges(cap)?))
    } else {
        None
    };
    let end_cap_parts = if let Some(cap) = end_cap {
        validated(cap)?;
        Some((positions(cap), polygons(cap)?, loose_edges(cap)?))
    } else {
        None
    };
    let caps: Vec<&Mesh> = start_cap.into_iter().chain(end_cap).collect();
    let provenance = (!caps.is_empty())
        .then(|| array_provenance_source(mesh, &caps))
        .transpose()?;
    let provenance_source = provenance.as_ref().unwrap_or(mesh);
    let point_count = source_points
        .len()
        .checked_mul(count_usize)
        .and_then(|value| {
            value.checked_add(start_cap_parts.as_ref().map_or(0, |parts| parts.0.len()))
        })
        .and_then(|value| {
            value.checked_add(end_cap_parts.as_ref().map_or(0, |parts| parts.0.len()))
        })
        .ok_or_else(|| mesh_error(&MeshError::IdExhausted))?;
    let mut points = Vec::with_capacity(point_count);
    let mut output_polygons = Vec::with_capacity(
        source_polygons
            .len()
            .saturating_mul(count_usize)
            .saturating_add(start_cap_parts.as_ref().map_or(0, |parts| parts.1.len()))
            .saturating_add(end_cap_parts.as_ref().map_or(0, |parts| parts.1.len())),
    );
    let mut output_loose_edges = Vec::with_capacity(
        source_loose_edges
            .len()
            .saturating_mul(count_usize)
            .saturating_add(start_cap_parts.as_ref().map_or(0, |parts| parts.2.len()))
            .saturating_add(end_cap_parts.as_ref().map_or(0, |parts| parts.2.len())),
    );
    let mut copy_ranges = Vec::with_capacity(count_usize);
    let mut transform = DMat4::IDENTITY;
    for _ in 0..count_usize {
        copy_ranges.push(append_array_geometry(
            &mut points,
            &mut output_polygons,
            &mut output_loose_edges,
            &source_points,
            &source_polygons,
            &source_loose_edges,
            transform,
        ));
        transform *= step;
    }
    let start_cap_range = if let Some((cap_points, cap_polygons, cap_edges)) = &start_cap_parts {
        let cap_transform = step.inverse();
        if !cap_transform
            .to_cols_array()
            .iter()
            .all(|component| component.is_finite())
        {
            return Err(invalid_parameter(
                modifier,
                "start_cap",
                "a cap transform that produces finite coordinates",
            ));
        }
        Some(append_array_geometry(
            &mut points,
            &mut output_polygons,
            &mut output_loose_edges,
            cap_points,
            cap_polygons,
            cap_edges,
            cap_transform,
        ))
    } else {
        None
    };
    let end_cap_range = if let Some((cap_points, cap_polygons, cap_edges)) = &end_cap_parts {
        Some(append_array_geometry(
            &mut points,
            &mut output_polygons,
            &mut output_loose_edges,
            cap_points,
            cap_polygons,
            cap_edges,
            transform,
        ))
    } else {
        None
    };
    if points.iter().any(|point| !point.is_finite()) {
        return Err(invalid_parameter(
            modifier,
            "count",
            "a count that produces finite coordinates",
        ));
    }
    // Merging can drop or reorder a polygon's corners, so keep each pre-merge polygon's points.
    let pre_merge_faces: Option<Vec<Vec<usize>>> = use_merge.then(|| {
        output_polygons
            .iter()
            .map(|(face, _)| face.clone())
            .collect()
    });
    // Blender's merge keeps one source edge's flags per merged edge (see `first_source_edges`). The
    // pre-merge edges are every edge of each copy and cap, in the order `provenance_source` lists them.
    let pre_merge_edges = ["sharp", "seam"]
        .iter()
        .any(|name| provenance_source.attributes.contains_key(*name))
        .then(|| -> Result<Vec<[usize; 2]>> {
            let edges = edge_pairs(mesh)?;
            let mut pairs = Vec::with_capacity(edges.len() * count_usize);
            for copy in 0..count_usize {
                let base = copy * source_points.len();
                pairs.extend(
                    edges
                        .iter()
                        .map(|[first, second]| [base + first, base + second]),
                );
            }
            for (cap, range) in [(start_cap, &start_cap_range), (end_cap, &end_cap_range)] {
                if let (Some(cap), Some(range)) = (cap, range) {
                    let base = range.start;
                    pairs.extend(
                        edge_pairs(cap)?
                            .iter()
                            .map(|[first, second]| [base + first, base + second]),
                    );
                }
            }
            Ok(pairs)
        })
        .transpose()?;
    let merged = if use_merge {
        let mut pairs = Vec::with_capacity(count_usize + usize::from(merge_final));
        for index in 1..copy_ranges.len() {
            pairs.push((copy_ranges[index - 1].clone(), copy_ranges[index].clone()));
        }
        if let Some(cap_range) = start_cap_range {
            pairs.push((copy_ranges[0].clone(), cap_range));
        }
        if let Some(cap_range) = end_cap_range {
            pairs.push((copy_ranges[copy_ranges.len() - 1].clone(), cap_range));
        }
        if merge_final && copy_ranges.len() > 1 {
            pairs.push((
                copy_ranges[copy_ranges.len() - 1].clone(),
                copy_ranges[0].clone(),
            ));
        }
        merge_geometry_pairs(
            &points,
            output_polygons,
            output_loose_edges,
            merge_threshold,
            &pairs,
        )?
    } else {
        MergedGeometry::unmerged(points, output_polygons, output_loose_edges)
    };
    let first_sources = pre_merge_edges
        .as_deref()
        .map(|pre_merge| first_source_edges(&merged, pre_merge));
    // Copies come first in point and polygon order, then the start cap and the end cap. A copy element
    // maps to its input element by modulo; a cap element maps to its index in `provenance_source`,
    // where caps follow the input. Only copies carry vertex data: a merged vertex takes the data of its
    // lowest-index member (copies precede caps), as Blender's merge keeps the target vertex.
    let copy_vertex_count = count_usize * source_points.len();
    let copy_face_count = count_usize * source_polygons.len();
    let source_face = |polygon: usize| {
        if polygon < copy_face_count {
            polygon % source_polygons.len()
        } else {
            source_polygons.len() + (polygon - copy_face_count)
        }
    };
    let origin = Origin {
        vertices: merged
            .point_members
            .iter()
            .map(|members| match members.first() {
                Some(&point) if point < copy_vertex_count => {
                    vec![(point % source_points.len(), 1.0)]
                }
                _ => Vec::new(),
            })
            .collect(),
        faces: merged
            .polygon_sources
            .iter()
            .map(|&polygon| Some(source_face(polygon)))
            .collect(),
        corners: Some(array_corner_origin(
            &merged,
            pre_merge_faces.as_deref(),
            provenance_source,
            source_face,
        )),
    };
    let passes: Vec<usize> = merged
        .polygon_sources
        .iter()
        .map(|&polygon| {
            if polygon < copy_face_count {
                polygon / source_polygons.len()
            } else {
                0
            }
        })
        .collect();
    let mut result = build_mesh(
        merged.points,
        merged.polygons,
        merged.edges,
        &origin,
        provenance_source,
    )?;
    if let Some(first) = &first_sources {
        let edge_count = mesh.edges.len();
        let copy_edges = count_usize * edge_count;
        apply_first_source_edge_flags(&mut result, provenance_source, first, |edge| {
            if edge < copy_edges {
                edge % edge_count
            } else {
                edge_count + edge - copy_edges
            }
        })?;
    }
    map_uv_values(&mut result, |face, _, [u, v]| {
        let copy = usize_as_f64(passes[face]);
        [u + offset_u * copy, v + offset_v * copy]
    });
    Ok(result)
}

fn array_cap_mesh<'a>(
    modifier: &Modifier,
    name: &str,
    object_meshes: &'a BTreeMap<Id, &'a Mesh>,
    evaluated_object_meshes: &'a BTreeMap<Id, Mesh>,
) -> Result<Option<&'a Mesh>> {
    let Some(target) = optional_string_param(modifier, name)
        .map_err(|_| invalid_parameter(modifier, name, "a mesh object ID or null"))?
    else {
        return Ok(None);
    };
    let target_id =
        Id::new(target).map_err(|_| invalid_parameter(modifier, name, "a valid object ID"))?;
    evaluated_object_meshes
        .get(&target_id)
        .or_else(|| object_meshes.get(&target_id).copied())
        .map(Some)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                format!("array {name} object was not found"),
                json!({"modifier_id":modifier.id,"cap":name,"target":target_id}),
            )
        })
}

fn array_curve_length(modifier: &Modifier, curve_path_lengths: &BTreeMap<Id, f64>) -> Result<f64> {
    let Some(target) = optional_string_param(modifier, "curve")
        .map_err(|_| invalid_parameter(modifier, "curve", "a curve object ID or null"))?
    else {
        return Ok(0.0);
    };
    let target_id =
        Id::new(target).map_err(|_| invalid_parameter(modifier, "curve", "a valid object ID"))?;
    curve_path_lengths.get(&target_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "array fit curve object was not found or has no curve path",
            json!({"modifier_id":modifier.id,"curve":target_id}),
        )
    })
}

pub(crate) fn evaluated_curve_path_length(
    curve: &crate::model::CurveData,
    world_matrix: DMat4,
) -> Result<f64> {
    let mut path = curve.clone();
    // Blender's animation-path length helper measures only the first curve spline.
    path.splines.truncate(1);
    path.bevel_depth = 0.0;
    path.bevel_resolution = 0;
    path.extrude = 0.0;
    path.fill_mode = crate::model::CurveFillMode::None;
    let path_mesh =
        crate::geom::curve::evaluate_curve(&path).map_err(|error| mesh_error(&error))?;
    let indices = id_indices(&path_mesh);
    let points = positions(&path_mesh);
    let mut length = 0.0;
    for edge in &path_mesh.edges {
        let first = *indices.get(&edge.vertices[0]).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "evaluated curve path references a missing vertex",
            )
        })?;
        let second = *indices.get(&edge.vertices[1]).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "evaluated curve path references a missing vertex",
            )
        })?;
        length += points[first].distance(points[second]);
    }
    let scale = world_matrix
        .transform_vector3(DVec3::splat(1.0 / 3.0_f64.sqrt()))
        .length();
    let world_length = length * scale;
    if !world_length.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "evaluated curve path length is non-finite",
        ));
    }
    Ok(world_length)
}

fn append_array_geometry(
    points: &mut Vec<DVec3>,
    polygons: &mut Vec<Polygon>,
    edges: &mut Vec<[usize; 2]>,
    source_points: &[DVec3],
    source_polygons: &[Polygon],
    source_edges: &[[usize; 2]],
    transform: DMat4,
) -> std::ops::Range<usize> {
    let base = points.len();
    points.extend(
        source_points
            .iter()
            .map(|point| transform.transform_point3(*point)),
    );
    polygons.extend(source_polygons.iter().map(|(face, material_index)| {
        (
            face.iter().map(|index| base + *index).collect(),
            *material_index,
        )
    }));
    edges.extend(
        source_edges
            .iter()
            .map(|[first, second]| [base + *first, base + *second]),
    );
    base..points.len()
}

/// Copies the keyed entry `from` of `cap`'s attribute `name` into `source` under the key `to`, creating
/// the attribute with the cap's domain and type when `source` lacks it.
fn copy_keyed_value(cap: &Mesh, source: &mut Mesh, name: &str, from: &str, to: String) {
    let Some(attribute) = cap.attributes.get(name) else {
        return;
    };
    let Some(value) = attribute.get("values").and_then(|values| values.get(from)) else {
        return;
    };
    let target = source.attributes.entry(name.to_owned()).or_insert_with(
        || json!({"domain": attribute["domain"], "type": attribute["type"], "values": {}}),
    );
    if let Some(values) = target.get_mut("values").and_then(Value::as_object_mut) {
        values.insert(to, value.clone());
    }
}

/// The input mesh with its cap meshes appended: the provenance source of an array with caps.
///
/// Storage order is the input, then the start cap, then the end cap, matching the source indices
/// `array` reports. Input elements keep their IDs; cap elements get fresh IDs after them. Like
/// Blender's array modifier, a cap contributes UV data only: each input UV layer gains the matching
/// cap layer's corner values, face by face. Where the cap lacks the layer, its corners take Blender's
/// `[0, 0]` in V-up, stored as `[0, 1]`. Every
/// other attribute stays input-only, so cap elements take defaults.
fn array_provenance_source(mesh: &Mesh, caps: &[&Mesh]) -> Result<Mesh> {
    let mut source = mesh.clone();
    let mut vertex_next = next_element_id(mesh.vertices.iter().map(|vertex| vertex.id))?;
    let mut edge_next = next_element_id(mesh.edges.iter().map(|edge| edge.id))?;
    let mut face_next = next_element_id(mesh.faces.iter().map(|face| face.id))?;
    // Input UV layers, each extended face by face with every cap in turn.
    let mut uv_store = uv_layers::read(mesh);
    for cap in caps {
        let mut vertex_ids = HashMap::with_capacity(cap.vertices.len());
        for vertex in &cap.vertices {
            let id = take_element_id(&mut vertex_next)?;
            vertex_ids.insert(vertex.id, id);
            source.vertices.push(Vertex { id, co: vertex.co });
        }
        let vertex_of = |id: u32| {
            vertex_ids.get(&id).copied().ok_or_else(|| {
                mesh_error(&MeshError::InvalidTopology(
                    "array cap references a missing vertex",
                ))
            })
        };
        let mut edge_ids = HashMap::with_capacity(cap.edges.len());
        for edge in &cap.edges {
            let vertices = [vertex_of(edge.vertices[0])?, vertex_of(edge.vertices[1])?];
            let id = take_element_id(&mut edge_next)?;
            edge_ids.insert(edge.id, id);
            source.edges.push(Edge { id, vertices });
        }
        let mut face_ids = HashMap::with_capacity(cap.faces.len());
        for face in &cap.faces {
            let id = take_element_id(&mut face_next)?;
            face_ids.insert(face.id, id);
            let vertices = face
                .vertices
                .iter()
                .map(|&vertex| vertex_of(vertex))
                .collect::<Result<Vec<_>>>()?;
            source.faces.push(Face {
                id,
                vertices,
                material_index: 0,
            });
        }
        for (old, new) in &face_ids {
            copy_keyed_value(
                cap,
                &mut source,
                "shade_smooth",
                &format!("f{old}"),
                format!("f{new}"),
            );
        }
        for (old, new) in &edge_ids {
            copy_keyed_value(
                cap,
                &mut source,
                "sharp",
                &format!("e{old}"),
                format!("e{new}"),
            );
        }
        let cap_uv = uv_layers::read_layers(cap);
        for layer in &mut uv_store.layers {
            let cap_layer = cap_uv.iter().find(|known| known.name == layer.name);
            for (index, face) in cap.faces.iter().enumerate() {
                let uv = cap_layer
                    .and_then(|known| known.faces.get(index))
                    .cloned()
                    .flatten()
                    .unwrap_or_else(|| vec![[0.0, 1.0]; face.vertices.len()]);
                layer.faces.push(Some(uv));
                layer.pinned.push(None);
            }
        }
    }
    array_provenance_normals(mesh, caps, &mut source)?;
    uv_layers::write(&mut source, &uv_store);
    Ok(source)
}

/// Extends `source`'s custom-normal state over the appended caps, in storage order. A part without a
/// normal layer is automatic, and each part's loop ordinals follow the corners before it.
fn array_provenance_normals(mesh: &Mesh, caps: &[&Mesh], source: &mut Mesh) -> Result<()> {
    let parts: Vec<&Mesh> = std::iter::once(mesh).chain(caps.iter().copied()).collect();
    let layers: Vec<Option<Vec<CornerNormalCode>>> = parts
        .iter()
        .map(|part| clnor::corner_codes(part))
        .collect::<Result<_>>()?;
    let orders: Vec<Option<Vec<usize>>> = parts
        .iter()
        .map(|part| clnor::corner_order(part))
        .collect::<Result<_>>()?;
    source.attributes.remove("custom_normal");
    let has_layer = layers.iter().any(Option::is_some);
    let has_order = orders.iter().any(Option::is_some);
    if !has_layer && !has_order {
        return Ok(());
    }
    let mut codes = Vec::new();
    let mut order = Vec::new();
    for ((part, layer), part_order) in parts.iter().zip(layers).zip(orders) {
        let base = order.len();
        let count: usize = part.faces.iter().map(|face| face.vertices.len()).sum();
        match layer {
            Some(part_codes) => codes.extend(part_codes),
            None => codes.extend(std::iter::repeat_n(CornerNormalCode::Auto, count)),
        }
        match part_order {
            Some(part_order) => order.extend(part_order.into_iter().map(|ordinal| base + ordinal)),
            None => order.extend(base..base + count),
        }
    }
    clnor::write_codes(
        source,
        has_layer.then_some(codes.as_slice()),
        NormalCodeOptions {
            corner_order: has_order.then_some(order.as_slice()),
            ..NormalCodeOptions::default()
        },
    )
}

/// The first element ID after `ids`, or zero when there are none.
fn next_element_id(ids: impl Iterator<Item = u32>) -> Result<u32> {
    ids.max().map_or(Ok(0), |id| {
        id.checked_add(1)
            .ok_or_else(|| mesh_error(&MeshError::IdExhausted))
    })
}

/// Takes `next` as an element ID and advances it.
fn take_element_id(next: &mut u32) -> Result<u32> {
    let id = *next;
    *next = next
        .checked_add(1)
        .ok_or_else(|| mesh_error(&MeshError::IdExhausted))?;
    Ok(id)
}

/// Corner sources of a rebuilt array. Each output corner comes from the one pre-merge corner that
/// merged into it, so it takes that corner's UVs unblended. Without a merge (`pre_merge` is `None`)
/// output corner `k` of a polygon is pre-merge corner `k`. Otherwise `pre_merge[p]` lists the points
/// of pre-merge polygon `p` in corner order. Those corners are the corners of `provenance` face
/// `source_face(p)`.
fn array_corner_origin(
    merged: &MergedGeometry,
    pre_merge: Option<&[Vec<usize>]>,
    provenance: &Mesh,
    source_face: impl Fn(usize) -> usize,
) -> Vec<Vec<(usize, f64)>> {
    let mut face_corners = Vec::with_capacity(provenance.faces.len());
    let mut total = 0;
    for face in &provenance.faces {
        face_corners.push(total);
        total += face.vertices.len();
    }
    let mut corners = Vec::new();
    for ((polygon, _), &source) in merged.polygons.iter().zip(&merged.polygon_sources) {
        let first = face_corners[source_face(source)];
        for (corner, point) in polygon.iter().enumerate() {
            let pre_corner = match pre_merge {
                None => Some(corner),
                Some(pre_merge) => {
                    let members = &merged.point_members[*point];
                    pre_merge[source]
                        .iter()
                        .position(|candidate| members.contains(candidate))
                }
            };
            corners.push(
                pre_corner
                    .map(|pre_corner| vec![(first + pre_corner, 1.0)])
                    .unwrap_or_default(),
            );
        }
    }
    corners
}

/// Rewrites the UV values of a rebuilt `mesh` in every store of each layer. `transform(face, corner,
/// uv)` maps one corner's `[u, v]` in Blender's V-up convention, where `face` is the output polygon
/// index and `corner` the corner position within it. The store is V-down, so values convert here.
fn map_uv_values(mesh: &mut Mesh, transform: impl Fn(usize, usize, [f64; 2]) -> [f64; 2]) {
    let mut layers = uv_layers::read_layers(mesh);
    for layer in &mut layers {
        for (face, values) in layer.faces.iter_mut().enumerate() {
            for (corner, uv) in values.iter_mut().flatten().enumerate() {
                *uv = flip_v(transform(face, corner, flip_v(*uv)));
            }
        }
    }
    uv_layers::write_layers(mesh, &layers);
}

/// Converts `[u, v]` between the store's V-down and Blender's V-up. The map is its own inverse.
fn flip_v([u, v]: [f64; 2]) -> [f64; 2] {
    [u, 1.0 - v]
}

/// Each merged edge's first source: the pre-merge edge with the lowest index among those merging onto
/// its endpoints, keyed by the merged point pair. Blender's `mesh_merge_verts` keeps the destination
/// (lowest-index) edge's data alone, so a merged edge takes its flags from that edge.
fn first_source_edges(
    merged: &MergedGeometry,
    pre_merge: &[[usize; 2]],
) -> HashMap<(usize, usize), usize> {
    let mut merged_point = vec![0; merged.point_members.iter().map(Vec::len).sum()];
    for (point, members) in merged.point_members.iter().enumerate() {
        for &member in members {
            merged_point[member] = point;
        }
    }
    let mut first = HashMap::new();
    for (edge, [first_point, second_point]) in pre_merge.iter().enumerate() {
        let key = edge_key(merged_point[*first_point], merged_point[*second_point]);
        if key.0 != key.1 {
            first.entry(key).or_insert(edge);
        }
    }
    first
}

/// Writes the shading flags of `result`'s edges: `sharp` and `seam` each take the flag of the first
/// source edge (see [`first_source_edges`]). `source_edge` maps a pre-merge edge to its index in
/// `provenance`, whose flags are read.
fn apply_first_source_edge_flags(
    result: &mut Mesh,
    provenance: &Mesh,
    first: &HashMap<(usize, usize), usize>,
    source_edge: impl Fn(usize) -> usize,
) -> Result<()> {
    let index_of: HashMap<u32, usize> = result
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let sources: Vec<Option<usize>> = result
        .edges
        .iter()
        .map(|edge| {
            let first_point = *index_of.get(&edge.vertices[0])?;
            let second_point = *index_of.get(&edge.vertices[1])?;
            first
                .get(&edge_key(first_point, second_point))
                .map(|&pre_merge| source_edge(pre_merge))
        })
        .collect();
    let sharp = sharp_edge_flags(provenance)?;
    let seam = edge_flags(provenance, "seam");
    for (name, flags) in [("sharp", sharp.as_slice()), ("seam", seam.as_slice())] {
        if let Some(values) = result
            .attributes
            .get_mut(name)
            .and_then(|attribute| attribute.get_mut("values"))
            .and_then(Value::as_object_mut)
        {
            for (edge, source) in result.edges.iter().zip(&sources) {
                let flag = source.and_then(|source| flags.get(source)).copied();
                values.insert(format!("e{}", edge.id), json!(flag == Some(true)));
            }
        }
    }
    Ok(())
}

/// One mirror pass over a UV coordinate. A flipped axis is reflected about 0.5, or about the centre
/// of its UDIM tile with `udim`; the mirror offset applies to flipped axes only. The UV offset applies
/// to every copy.
fn mirror_uv_pass(
    uv: [f64; 2],
    flip: [bool; 2],
    mirror_offset: [f64; 2],
    offset: [f64; 2],
    udim: bool,
) -> [f64; 2] {
    std::array::from_fn(|axis| {
        let value = if flip[axis] {
            // Blender's UDIM reflection: a whole-number coordinate maps to itself.
            let reflected = if udim {
                uv[axis].ceil() - uv[axis] % 1.0
            } else {
                1.0 - uv[axis]
            };
            reflected + mirror_offset[axis]
        } else {
            uv[axis]
        };
        value + offset[axis]
    })
}

fn merge_geometry_pairs(
    points: &[DVec3],
    polygons: Vec<Polygon>,
    edges: Vec<[usize; 2]>,
    threshold: f64,
    pairs: &[(std::ops::Range<usize>, std::ops::Range<usize>)],
) -> Result<MergedGeometry> {
    if points.len() > u32::MAX as usize {
        return Err(mesh_error(&MeshError::IdExhausted));
    }
    let mut parent: Vec<_> = (0..points.len()).collect();
    let mut cells = HashMap::<[i64; 3], Vec<usize>>::new();
    let bucket_size = if threshold == 0.0 { 1.0 } else { threshold };
    let threshold_squared = threshold * threshold;
    for (target, source) in pairs {
        cells.clear();
        for index in target.clone() {
            cells
                .entry(spatial_cell(points[index], bucket_size))
                .or_default()
                .push(index);
        }
        for source_index in source.clone() {
            let cell = spatial_cell(points[source_index], bucket_size);
            let mut closest = None;
            for x in -1_i64..=1 {
                for y in -1_i64..=1 {
                    for z in -1_i64..=1 {
                        let nearby = [
                            cell[0].saturating_add(x),
                            cell[1].saturating_add(y),
                            cell[2].saturating_add(z),
                        ];
                        if let Some(candidates) = cells.get(&nearby) {
                            for target_index in candidates {
                                let distance =
                                    points[source_index].distance_squared(points[*target_index]);
                                if distance <= threshold_squared
                                    && closest.is_none_or(|(best, _)| distance < best)
                                {
                                    closest = Some((distance, *target_index));
                                }
                            }
                        }
                    }
                }
            }
            if let Some((_, target_index)) = closest {
                union(&mut parent, source_index, target_index);
            }
        }
    }
    Ok(collapse_clusters(points, parent, polygons, edges))
}

#[derive(Clone, Copy)]
struct SubdivisionOptions {
    preserve_corners: bool,
    use_creases: bool,
}

fn subdivision(
    mesh: &Mesh,
    modifier: &Modifier,
    context: ModifierEvaluationContext<'_>,
) -> Result<Mesh> {
    validated(mesh)?;
    let level_name = if context.purpose == crate::eval::EvaluationPurpose::Render {
        "render_levels"
    } else {
        "levels"
    };
    let levels = uint_param(modifier, level_name)?;
    if levels > 5 {
        return Err(invalid_parameter(
            modifier,
            level_name,
            "an integer between 0 and 5",
        ));
    }
    let boundary_smooth =
        string_param_with_expected(modifier, "boundary_smooth", "ALL or PRESERVE_CORNERS")?;
    if !matches!(boundary_smooth, "ALL" | "PRESERVE_CORNERS") {
        return Err(invalid_parameter(
            modifier,
            "boundary_smooth",
            "ALL or PRESERVE_CORNERS",
        ));
    }
    let uv_smooth =
        string_param_with_expected(modifier, "uv_smooth", "a Blender UV smoothing mode")?;
    if !matches!(
        uv_smooth,
        "NONE"
            | "PRESERVE_CORNERS"
            | "PRESERVE_CORNERS_AND_JUNCTIONS"
            | "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE"
            | "PRESERVE_BOUNDARIES"
            | "SMOOTH_ALL"
    ) {
        return Err(invalid_parameter(
            modifier,
            "uv_smooth",
            "a Blender UV smoothing mode",
        ));
    }
    let quality = uint_param(modifier, "quality")?;
    if !(1..=10).contains(&quality) {
        return Err(invalid_parameter(
            modifier,
            "quality",
            "an integer between 1 and 10",
        ));
    }
    let simple = match modifier.params.get("subdivision_type").map(Value::as_str) {
        None | Some(Some("CATMULL_CLARK")) => false,
        Some(Some("SIMPLE")) => true,
        Some(_) => {
            return Err(invalid_parameter(
                modifier,
                "subdivision_type",
                "CATMULL_CLARK or SIMPLE",
            ));
        }
    };
    let options = SubdivisionOptions {
        preserve_corners: boundary_smooth == "PRESERVE_CORNERS",
        use_creases: bool_param(modifier, "use_creases")?,
    };
    let linear = subdivision_uv::FvarLinear::from_uv_smooth(uv_smooth);
    // Bilinear subdivision has no limit surface: its positions are already the limit.
    // `use_limit_surface` is not in the multires schema; multires uses `true` when it is unset.
    let multires_unset =
        modifier.modifier_type == "multires" && !modifier.params.contains_key("use_limit_surface");
    let use_limit_surface =
        levels > 0 && !simple && (multires_unset || bool_param(modifier, "use_limit_surface")?);
    if levels > 0 {
        let first_level_faces = mesh
            .faces
            .iter()
            .try_fold(0_usize, |count, face| {
                count.checked_add(face.vertices.len())
            })
            .ok_or_else(|| {
                PotError::new(ErrorCode::LimitExceeded, "subdivision face count overflows")
            })?;
        let uv_layer_count = uv_layers::layer_count(mesh);
        check_subdivision_budget(
            first_level_faces,
            mesh.vertices.len(),
            mesh.edges.len(),
            levels,
            uv_layer_count,
        )?;
    }
    let use_custom_normals = bool_param(modifier, "use_custom_normals")?;
    let mut result = mesh.clone();
    if levels == 0 {
        // Without levels, only Use Custom Normals keeps the custom-normal state.
        if !use_custom_normals {
            result.attributes.remove("custom_normal");
        }
        return Ok(result);
    }
    // Subsurf interpolates the evaluated input loop normals only when the source has a custom layer
    // and the option is enabled; automatic corners participate as their evaluated fan normals.
    let input_normals = if use_custom_normals && clnor::corner_codes(mesh)?.is_some() {
        Some(corner_normals(mesh, &positions(mesh))?)
    } else {
        None
    };
    // With levels, the output custom normals are re-encoded from the subdivided loop normals below.
    result.attributes.remove("custom_normal");
    let mut source_polygons = polygons(&result)?;
    let mut source_edges = loose_edges(&result)?;
    let mut edge_creases =
        subdivision_uv::Creases::from_values(edge_crease_values(&result, options.use_creases)?);
    let mut position = subdivision_uv::Position::new(
        &source_polygons,
        &source_edges,
        &edge_creases.sharpness,
        result.vertices.len(),
    );
    let mut uv_maps = subdivision_uv::value_maps(mesh, &position);
    // Blender refines SIMPLE as Catmull-Clark with infinitely sharp edges and vertices, so its
    // positions are limit points too (see `BaseSharpness::Simple`). Loop normals use the same grid.
    let mut samples = (simple || use_limit_surface || input_normals.is_some()).then(|| {
        let base_faces: Vec<Vec<usize>> = source_polygons
            .iter()
            .map(|(face, _)| face.clone())
            .collect();
        Samples::base(&base_faces, result.vertices.len())
    });
    for level in 0..levels {
        let (next, edge_points) = if simple {
            let (next, next_creases, edge_points) = subdivision_uv::subdivide_simple_once(
                &result,
                &position,
                &source_polygons,
                &source_edges,
                &edge_creases,
                options.use_creases,
            )?;
            edge_creases = next_creases;
            (next, edge_points)
        } else {
            let (next, edge_points, next_creases) = subdivide_once(
                &result,
                options,
                &position,
                &source_polygons,
                &source_edges,
                &edge_creases,
            )?;
            edge_creases = next_creases;
            (next, edge_points)
        };
        samples = samples.map(|samples| {
            let faces: Vec<Vec<usize>> = source_polygons
                .iter()
                .map(|(face, _)| face.clone())
                .collect();
            samples.next(&faces, &edge_points, level == 0)
        });
        for layer in &mut uv_maps {
            layer.pins =
                subdivision_uv::refine_pins(std::mem::take(&mut layer.pins), &position.starts);
        }
        result = next;
        if level + 1 < levels {
            source_polygons = polygons(&result)?;
            source_edges = loose_edges(&result)?;
            position = subdivision_uv::Position::new(
                &source_polygons,
                &source_edges,
                &edge_creases.sharpness,
                result.vertices.len(),
            );
        }
    }
    // Positions and UVs share the base level of the limit surface. SIMPLE is evaluated at isolation 1
    // whatever the quality; without `use_limit_surface` only the base level serves the UVs.
    let limit_samples = samples.as_ref().filter(|_| simple || use_limit_surface);
    if simple || use_limit_surface || !uv_maps.is_empty() {
        let base_faces: Vec<Vec<usize>> =
            polygons(mesh)?.into_iter().map(|(face, _)| face).collect();
        let creases =
            subdivision_uv::Creases::from_values(edge_crease_values(mesh, options.use_creases)?)
                .sharpness;
        // OpenSubdiv sharpens non-manifold edges and corners infinitely (as for the positions).
        let non_manifold =
            NonManifold::new(mesh.vertices.len(), base_faces.iter().map(Vec::as_slice));
        let edge_sharpness = non_manifold.sharpened(&creases);
        let non_manifold_vertices: Vec<bool> = (0..mesh.vertices.len())
            .map(|vertex| non_manifold.corner(vertex))
            .collect();
        let sharpness = if simple {
            subdivision_limit::BaseSharpness::Simple
        } else {
            subdivision_limit::BaseSharpness::Creases {
                edges: &edge_sharpness,
                non_manifold: &non_manifold_vertices,
                preserve_corners: options.preserve_corners,
            }
        };
        // Blender evaluates SIMPLE at refinement level 1, whatever the quality.
        let isolation = if simple {
            1
        } else if use_limit_surface {
            quality as usize
        } else {
            0
        };
        let limit = subdivision_limit::LimitSurface::build(
            &base_faces,
            &positions(mesh),
            sharpness,
            isolation,
        );
        if let Some(samples) = limit_samples {
            for (vertex, sample) in result.vertices.iter_mut().zip(&samples.vertices) {
                if let Some(sample) = sample {
                    vertex.co = limit.evaluate(*sample);
                }
            }
        }
        if !uv_maps.is_empty() {
            let loops: Vec<_> = uv_maps
                .into_iter()
                .map(|layer| {
                    let values = match limit_samples {
                        Some(samples) => limit.face_varying_loops(
                            linear,
                            options.preserve_corners,
                            &layer.map,
                            samples,
                        ),
                        None => limit.uniform_face_varying(
                            linear,
                            options.preserve_corners,
                            &layer.map,
                            levels as usize,
                        ),
                    };
                    (layer.name, values, layer.pins)
                })
                .collect();
            subdivision_uv::write_loop_uvs(&mut result, loops)?;
        }
    }
    // Loop normals follow the final positions: Blender's fan spaces are measured on them.
    if let Some(normals) = input_normals {
        let samples = samples.as_ref().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "subdivision custom normals are missing output samples",
            )
        })?;
        let face_sizes: Vec<usize> = mesh.faces.iter().map(|face| face.vertices.len()).collect();
        let loops = subdivision_normals::loop_normals(&normals, &face_sizes, &samples.corners);
        let numbering =
            subdivision_normals::corner_numbering(&samples.corners, &face_sizes, levels as u32);
        let vectors: Vec<Option<DVec3>> = loops.into_iter().map(Some).collect();
        clnor::encode_vectors(
            &mut result,
            Some(&vectors),
            NormalCodeOptions {
                corner_order: Some(&numbering),
                setter_input: clnor::SetterInput::AlreadyNormalized,
            },
        )?;
    }
    Ok(result)
}

fn check_subdivision_budget(
    first_level_faces: usize,
    input_vertices: usize,
    input_edges: usize,
    levels: u64,
    uv_layer_count: usize,
) -> Result<()> {
    let convert = |count: usize| {
        u128::try_from(count)
            .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "subdivision budget overflows"))
    };
    let input_vertices = convert(input_vertices)?;
    let input_edges = convert(input_edges)?;
    let mut face_count = 0_u128;
    let mut edge_multiplier = 1_u128;
    for level in 0..levels {
        face_count = if level == 0 {
            convert(first_level_faces)?
        } else {
            face_count.checked_mul(4).ok_or_else(|| {
                PotError::new(ErrorCode::LimitExceeded, "subdivision face count overflows")
            })?
        };
        let triangle_count = face_count.checked_mul(2).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "subdivision triangle count overflows",
            )
        })?;
        if triangle_count > MAX_SUBDIVISION_TRIANGLES {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "subdivision exceeds the 100-million evaluated triangle budget",
            ));
        }
        edge_multiplier = edge_multiplier.checked_mul(2).ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "subdivision edge count overflows")
        })?;
    }

    let estimated_vertices = input_vertices
        .checked_add(face_count.checked_mul(5).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "subdivision vertex estimate overflows",
            )
        })?)
        .and_then(|count| count.checked_add(input_edges.checked_mul(edge_multiplier)?))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "subdivision vertex estimate overflows",
            )
        })?;
    let estimated_edges = face_count
        .checked_mul(4)
        .and_then(|count| count.checked_add(input_edges.checked_mul(edge_multiplier)?))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "subdivision edge estimate overflows",
            )
        })?;
    let estimated_corners = face_count.checked_mul(4).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "subdivision corner estimate overflows",
        )
    })?;
    // Include simultaneous input/output mesh storage and subdivision adjacency indexes. Each UV
    // layer adds face-varying channels and UV attribute copies, measured at about 260 bytes per
    // output corner.
    let uv_corners = estimated_corners
        .checked_mul(convert(uv_layer_count)?)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "subdivision UV estimate overflows",
            )
        })?;
    let estimated_bytes = [
        (estimated_vertices, 64_u128),
        (estimated_edges, 128_u128),
        (face_count, 128_u128),
        (estimated_corners, 32_u128),
        (uv_corners, 260_u128),
    ]
    .into_iter()
    .try_fold(0_u128, |total, (count, bytes_per_element)| {
        total.checked_add(count.checked_mul(bytes_per_element)?)
    })
    .ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "subdivision memory estimate overflows",
        )
    })?;
    if estimated_bytes > MAX_SUBDIVISION_MEMORY_BYTES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "subdivision exceeds the 8-GiB estimated evaluation memory budget",
        ));
    }
    Ok(())
}

const BLENDER_EDGE_CREASE_MAX_SHARPNESS: f64 = 10.0;

fn blender_edge_crease_sharpness(crease: f64) -> f64 {
    BLENDER_EDGE_CREASE_MAX_SHARPNESS * crease * crease
}

fn edge_crease_values(mesh: &Mesh, enabled: bool) -> Result<BTreeMap<(usize, usize), f64>> {
    if !enabled {
        return Ok(BTreeMap::new());
    }
    let Some(values) = mesh
        .attributes
        .get("crease_edge")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
    else {
        return Ok(BTreeMap::new());
    };
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    let mut creases = BTreeMap::new();
    for edge in &mesh.edges {
        let Some(value) = values.get(&format!("e{}", edge.id)).and_then(Value::as_f64) else {
            continue;
        };
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(PotError::invalid_argument(
                "subdivision edge creases must be finite values in [0, 1]",
            ));
        }
        if value > 0.0 {
            let first = vertex_indices
                .get(&edge.vertices[0])
                .copied()
                .ok_or_else(|| {
                    PotError::invalid_argument("crease edge references a missing vertex")
                })?;
            let second = vertex_indices
                .get(&edge.vertices[1])
                .copied()
                .ok_or_else(|| {
                    PotError::invalid_argument("crease edge references a missing vertex")
                })?;
            creases.insert(edge_key(first, second), value);
        }
    }
    Ok(creases)
}

/// Non-manifold topology, which `OpenSubdiv` sharpens infinitely
/// (`prepareComponentTagsAndSharpness`). An edge with more than two faces, or two faces that
/// traverse it in the same direction, is an infinitely sharp crease. A vertex on one, or whose
/// faces are not one fan, is an infinitely sharp corner unless it is a non-manifold crease
/// (`testVertexNonManifoldCrease`).
struct NonManifold<'a> {
    faces: Vec<&'a [usize]>,
    /// Faces using each edge, with whether the face traverses it from its lower vertex index.
    uses: BTreeMap<(usize, usize), Vec<(usize, bool)>>,
    edges: BTreeSet<(usize, usize)>,
    vertex_edges: Vec<Vec<(usize, usize)>>,
    vertex_faces: Vec<Vec<usize>>,
    /// Whether each vertex is an infinitely sharp corner.
    corners: Vec<bool>,
}

impl<'a> NonManifold<'a> {
    fn new(vertex_count: usize, faces: impl Iterator<Item = &'a [usize]>) -> Self {
        let faces: Vec<&'a [usize]> = faces.collect();
        let mut uses = BTreeMap::<(usize, usize), Vec<(usize, bool)>>::new();
        let mut vertex_faces = vec![Vec::<usize>::new(); vertex_count];
        for (face_index, face) in faces.iter().copied().enumerate() {
            for (first, second) in cyclic_pairs(face) {
                uses.entry(edge_key(*first, *second))
                    .or_default()
                    .push((face_index, first < second));
            }
            for vertex in face {
                if vertex_faces[*vertex].last() != Some(&face_index) {
                    vertex_faces[*vertex].push(face_index);
                }
            }
        }
        let edges = uses
            .iter()
            .filter(|(_, faces)| faces.len() > 2 || (faces.len() == 2 && faces[0].1 == faces[1].1))
            .map(|(edge, _)| *edge)
            .collect();
        let mut vertex_edges = vec![Vec::<(usize, usize)>::new(); vertex_count];
        for edge in uses.keys() {
            vertex_edges[edge.0].push(*edge);
            vertex_edges[edge.1].push(*edge);
        }
        let mut topology = Self {
            faces,
            uses,
            edges,
            vertex_edges,
            vertex_faces,
            corners: Vec::new(),
        };
        topology.corners = (0..vertex_count)
            .map(|vertex| topology.is_corner(vertex))
            .collect();
        topology
    }

    /// The crease weights with every non-manifold edge infinitely sharp.
    fn sharpened(&self, creases: &BTreeMap<(usize, usize), f64>) -> BTreeMap<(usize, usize), f64> {
        let mut sharp = creases.clone();
        for edge in &self.edges {
            sharp.insert(*edge, f64::INFINITY);
        }
        sharp
    }

    /// Whether `vertex` is an infinitely sharp corner.
    fn corner(&self, vertex: usize) -> bool {
        self.corners[vertex]
    }

    fn is_corner(&self, vertex: usize) -> bool {
        let edges = &self.vertex_edges[vertex];
        let faces = &self.vertex_faces[vertex];
        if faces.is_empty() {
            return false;
        }
        let non_manifold_edges = edges
            .iter()
            .filter(|edge| self.edges.contains(*edge))
            .count();
        if non_manifold_edges == 0 && self.is_single_fan(vertex) {
            return false;
        }
        let boundary_edges = edges
            .iter()
            .filter(|edge| self.uses[*edge].len() == 1)
            .count();
        let crease = non_manifold_edges == 2
            && boundary_edges == 0
            && faces.len() > edges.len()
            && self.is_crease(vertex);
        !crease
    }

    /// Whether the faces at `vertex` form one fan, connected across the edges at `vertex`.
    fn is_single_fan(&self, vertex: usize) -> bool {
        let faces = &self.vertex_faces[vertex];
        let mut seen = BTreeSet::from([faces[0]]);
        let mut pending = vec![faces[0]];
        while let Some(face) = pending.pop() {
            for (first, second) in cyclic_pairs(self.faces[face]) {
                if *first != vertex && *second != vertex {
                    continue;
                }
                for (other, _) in &self.uses[&edge_key(*first, *second)] {
                    if *other != face && seen.insert(*other) {
                        pending.push(*other);
                    }
                }
            }
        }
        seen.len() == faces.len()
    }

    /// `testVertexNonManifoldCrease`: the faces at `vertex` split into two runs of manifold
    /// faces, each running from one of its two non-manifold edges to the other and leaving
    /// `vertex` along it.
    fn is_crease(&self, vertex: usize) -> bool {
        let mut ends = Vec::with_capacity(2);
        for edge in &self.vertex_edges[vertex] {
            if self.edges.contains(edge) {
                if ends.len() == 2 {
                    return false;
                }
                ends.push(*edge);
            } else if self.uses[edge].len() != 2 {
                return false;
            }
        }
        let &[first, second] = ends.as_slice() else {
            return false;
        };
        if self.uses[&first].len() != self.uses[&second].len() {
            return false;
        }
        let mut traversed = 0;
        for (start, end) in [(first, second), (second, first)] {
            for &(face, _) in &self.uses[&start] {
                let polygon = self.faces[face];
                let Some(index) = edge_index(polygon, start) else {
                    return false;
                };
                if polygon[index] != vertex {
                    continue;
                }
                let (mut edge, mut current) = (start, face);
                while edge != end {
                    if edge != start {
                        let Some(next) = self.other_face(edge, current) else {
                            return false;
                        };
                        current = next;
                    }
                    traversed += 1;
                    let polygon = self.faces[current];
                    let Some(index) = edge_index(polygon, edge) else {
                        return false;
                    };
                    let previous = (index + polygon.len() - 1) % polygon.len();
                    edge = edge_key(polygon[previous], polygon[index]);
                    if edge == start {
                        return false;
                    }
                }
            }
        }
        traversed == self.vertex_faces[vertex].len()
    }

    /// The face across `edge` from `face`.
    fn other_face(&self, edge: (usize, usize), face: usize) -> Option<usize> {
        self.uses[&edge]
            .iter()
            .map(|(other, _)| *other)
            .find(|other| *other != face)
    }
}

/// The index `k` such that `(polygon[k], polygon[k + 1])` is `edge`.
fn edge_index(polygon: &[usize], edge: (usize, usize)) -> Option<usize> {
    (0..polygon.len())
        .find(|&index| edge_key(polygon[index], polygon[(index + 1) % polygon.len()]) == edge)
}

fn subdivide_once(
    mesh: &Mesh,
    options: SubdivisionOptions,
    position: &subdivision_uv::Position,
    source_polygons: &[Polygon],
    source_edges: &[[usize; 2]],
    edge_creases: &subdivision_uv::Creases,
) -> Result<(
    Mesh,
    BTreeMap<(usize, usize), usize>,
    subdivision_uv::Creases,
)> {
    let source_points = positions(mesh);
    let mut face_points = Vec::with_capacity(source_polygons.len());
    for (face, _) in source_polygons {
        face_points.push(
            face.iter()
                .fold(DVec3::ZERO, |sum, index| sum + source_points[*index])
                / usize_as_f64(face.len()),
        );
    }

    let mut updated_points =
        Vec::with_capacity(source_points.len() + position.edge_ends.len() + source_polygons.len());
    for (index, point) in source_points.iter().enumerate() {
        let edges = &position.vertex_edges
            [position.vertex_edge_starts[index]..position.vertex_edge_starts[index + 1]];
        let corners = &position.vertex_corners
            [position.vertex_starts[index]..position.vertex_starts[index + 1]];
        let mut boundary_neighbors = [0_usize; 2];
        let mut boundary_count = 0;
        for &edge in edges {
            let sides =
                &position.edge_sides[position.edge_starts[edge]..position.edge_starts[edge + 1]];
            if sides.len() == 1 {
                if boundary_count < boundary_neighbors.len() {
                    let ends = position.edge_ends[edge];
                    boundary_neighbors[boundary_count] =
                        if ends.0 == index { ends.1 } else { ends.0 };
                }
                boundary_count += 1;
            }
        }
        let smooth = if boundary_count == 2 {
            *point * 0.75
                + (source_points[boundary_neighbors[0]] + source_points[boundary_neighbors[1]])
                    * 0.125
        } else if !edges.is_empty() && !corners.is_empty() {
            let face_average = corners.iter().fold(DVec3::ZERO, |sum, corner| {
                sum + face_points[position.face_of[*corner]]
            }) / usize_as_f64(corners.len());
            let edge_midpoint_average = edges.iter().fold(DVec3::ZERO, |sum, edge| {
                let ends = position.edge_ends[*edge];
                sum + (source_points[ends.0] + source_points[ends.1]) * 0.5
            }) / usize_as_f64(edges.len());
            let valence = usize_as_f64(edges.len());
            (face_average + edge_midpoint_average * 2.0 + *point * (valence - 3.0)) / valence
        } else {
            *point
        };

        let mut crease_neighbors = [(0_usize, 0.0_f64); 3];
        let mut crease_count = 0_usize;
        for &edge in edges {
            // OpenSubdiv sharpens boundary edges infinitely, so they are creases of the vertex rule.
            let sides =
                &position.edge_sides[position.edge_starts[edge]..position.edge_starts[edge + 1]];
            let sharpness = if sides.len() == 1 {
                BLENDER_EDGE_CREASE_MAX_SHARPNESS
            } else {
                position.edge_sharp[edge]
            };
            if sharpness <= 0.0 {
                continue;
            }
            let ends = position.edge_ends[edge];
            let neighbor = if ends.0 == index { ends.1 } else { ends.0 };
            let mut slot = 0;
            while slot < crease_neighbors.len() && sharpness <= crease_neighbors[slot].1 {
                slot += 1;
            }
            if slot < crease_neighbors.len() {
                for next in (slot + 1..crease_neighbors.len()).rev() {
                    crease_neighbors[next] = crease_neighbors[next - 1];
                }
                crease_neighbors[slot] = (neighbor, sharpness);
            }
            crease_count = (crease_count + 1).min(crease_neighbors.len());
        }
        // A topological corner of a preserved boundary is an infinitely sharp corner.
        let topological_corner = corners.len() == 1 && edges.len() == 2;
        let updated = if position.non_manifold_corners[index]
            || (options.preserve_corners && topological_corner)
        {
            *point
        } else if crease_count >= 3 {
            let crease = *point * 0.75
                + (source_points[crease_neighbors[0].0] + source_points[crease_neighbors[1].0])
                    * 0.125;
            let crease_weight = crease_neighbors[0].1.min(crease_neighbors[1].1).min(1.0);
            let corner_weight = crease_neighbors[0]
                .1
                .min(crease_neighbors[1].1)
                .min(crease_neighbors[2].1)
                .min(1.0);
            smooth
                .lerp(crease, crease_weight)
                .lerp(*point, corner_weight)
        } else if crease_count == 2 {
            let sharpness = crease_neighbors[0].1.min(crease_neighbors[1].1).min(1.0);
            let crease = *point * 0.75
                + (source_points[crease_neighbors[0].0] + source_points[crease_neighbors[1].0])
                    * 0.125;
            smooth.lerp(crease, sharpness)
        } else {
            smooth
        };
        updated_points.push(updated);
    }

    let mut edge_point_indices = BTreeMap::new();
    for &edge_index in &position.edge_order {
        let edge = position.edge_ends[edge_index];
        let midpoint = (source_points[edge.0] + source_points[edge.1]) * 0.5;
        let sides = &position.edge_sides
            [position.edge_starts[edge_index]..position.edge_starts[edge_index + 1]];
        let smooth = if sides.len() == 2 {
            let first_face = position.face_of[sides[0]];
            let second_face = position.face_of[sides[1]];
            (source_points[edge.0]
                + source_points[edge.1]
                + face_points[first_face]
                + face_points[second_face])
                * 0.25
        } else {
            midpoint
        };
        let point = smooth.lerp(midpoint, position.edge_sharp[edge_index].min(1.0));
        edge_point_indices.insert(edge, updated_points.len());
        updated_points.push(point);
    }
    updated_points.extend(face_points);
    let next_creases = edge_creases.subdivide(&edge_point_indices);
    let subdivided = subdivision_uv::subdivided_mesh(
        mesh,
        updated_points,
        source_polygons,
        source_edges,
        &edge_point_indices,
        &next_creases.canonical,
        options.use_creases,
    )?;
    Ok((subdivided, edge_point_indices, next_creases))
}

fn triangulate_quad_method(modifier: &Modifier) -> Result<QuadMethod> {
    const EXPECTED: &str = "BEAUTY, FIXED, FIXED_ALTERNATE, SHORTEST_DIAGONAL, or LONGEST_DIAGONAL";
    Ok(
        match string_param_with_expected(modifier, "quad_method", EXPECTED)? {
            "BEAUTY" => QuadMethod::Beauty,
            "FIXED" => QuadMethod::Fixed,
            "FIXED_ALTERNATE" => QuadMethod::FixedAlternate,
            "SHORTEST_DIAGONAL" => QuadMethod::ShortestDiagonal,
            "LONGEST_DIAGONAL" => QuadMethod::LongestDiagonal,
            _ => return Err(invalid_parameter(modifier, "quad_method", EXPECTED)),
        },
    )
}

fn triangulate_ngon_method(modifier: &Modifier) -> Result<NgonMethod> {
    const EXPECTED: &str = "BEAUTY or CLIP";
    Ok(
        match string_param_with_expected(modifier, "ngon_method", EXPECTED)? {
            "BEAUTY" => NgonMethod::Beauty,
            "CLIP" => NgonMethod::Clip,
            _ => return Err(invalid_parameter(modifier, "ngon_method", EXPECTED)),
        },
    )
}

fn triangulate(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    let keep_custom_normals = bool_param(modifier, "keep_custom_normals")?;
    let quad_method = triangulate_quad_method(modifier)?;
    let ngon_method = triangulate_ngon_method(modifier)?;
    let min_vertices = usize::try_from(uint_param(modifier, "min_vertices")?)
        .ok()
        .filter(|count| *count >= 4)
        .ok_or_else(|| invalid_parameter(modifier, "min_vertices", "an integer of at least 4"))?;
    let indices = id_indices(mesh);
    let coords: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3().to_array())
        .collect();
    let face_vertices = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| {
                    indices.get(id).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InvalidArgument,
                            "face references an unknown vertex",
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let face_corners: Vec<Vec<[f32; 3]>> = face_vertices
        .iter()
        .map(|vertices| vertices.iter().map(|&index| coords[index]).collect())
        .collect();
    let triangles = triangulate_mesh(&face_corners, quad_method, ngon_method, min_vertices);
    // The first source corner of each face, so a face corner maps to `corner_offsets[face] + corner`.
    let corner_offsets: Vec<usize> = mesh
        .faces
        .iter()
        .scan(0, |offset, face| {
            let start = *offset;
            *offset += face.vertices.len();
            Some(start)
        })
        .collect();
    let mut output_polygons = Vec::with_capacity(triangles.len());
    // The source corner each output corner copies, in output order.
    let mut sources = Vec::with_capacity(3 * triangles.len());
    for triangle in &triangles {
        let polygon: Vec<usize> = triangle
            .corners
            .iter()
            .map(|&corner| face_vertices[triangle.face][corner])
            .collect();
        output_polygons.push((polygon, mesh.faces[triangle.face].material_index));
        sources.extend(
            triangle
                .corners
                .iter()
                .map(|&corner| Some(corner_offsets[triangle.face] + corner)),
        );
    }
    let origin = Origin::copies(
        (0..mesh.vertices.len()).map(Some),
        triangles
            .iter()
            .map(|triangle| Some(triangle.face))
            .collect(),
    );
    let mut result = build_mesh(
        positions(mesh),
        output_polygons,
        loose_edges(mesh)?,
        &origin,
        mesh,
    )?;
    if keep_custom_normals {
        // MOD_triangulate.cc copies the evaluated corner normals (`mesh->corner_normals()`, which
        // include custom normals) onto the triangulated loops and writes them with
        // `bke::mesh_set_custom_normals_normalized` (already-normalised input, no renormalisation),
        // so every triangle corner gets the vector of the source corner it came from. The loops are
        // rebuilt in storage order, so no loop basis carries over.
        result.attributes.remove("custom_normal");
        let computed = corner_normals(mesh, &positions(mesh))?;
        let vectors: Vec<Option<DVec3>> = sources
            .iter()
            .map(|source| source.map(|corner| computed[corner]))
            .collect();
        clnor::encode_vectors(
            &mut result,
            Some(&vectors),
            NormalCodeOptions {
                corner_order: None,
                setter_input: clnor::SetterInput::AlreadyNormalized,
            },
        )?;
    } else if clnor::corner_codes(mesh)?.is_some() || clnor::corner_order(mesh)?.is_some() {
        clnor::transfer_custom_normals(mesh, &mut result, &sources, NormalCodeOptions::default())?;
    }
    Ok(result)
}

#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
fn solidify(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    let thickness = number_param(modifier, "thickness")?;
    if thickness < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "thickness",
            "a non-negative finite number",
        ));
    }
    let offset = number_param(modifier, "offset")?;
    if !(-1.0..=1.0).contains(&offset) {
        return Err(invalid_parameter(
            modifier,
            "offset",
            "a number from -1 through 1",
        ));
    }
    let use_even_offset = bool_param(modifier, "use_even_offset")?;
    let use_quality_normals = bool_param(modifier, "use_quality_normals")?;
    let use_rim = bool_param(modifier, "use_rim")?;
    let use_rim_only = bool_param(modifier, "use_rim_only")?;
    let flip = bool_param(modifier, "use_flip_normals")?;
    if mesh.faces.is_empty() {
        return Ok(mesh.clone());
    }

    let source_points = positions(mesh);
    let source_polygons = polygons(mesh)?;
    let source_loose_edges = loose_edges(mesh)?;
    let normals = vertex_normals(&source_points, &source_polygons, use_quality_normals);
    let mut even_scales = vec![0.0; source_points.len()];
    let mut angle_totals = vec![0.0; source_points.len()];
    let mut edge_faces: BevelEdgeFaces = BTreeMap::new();
    // Per edge: the first face that uses it, and that face's corner at the edge's first vertex.
    let mut edge_owner: BTreeMap<(usize, usize), (usize, usize)> = BTreeMap::new();
    let mut corner_base = Vec::with_capacity(source_polygons.len());
    let mut corner_total = 0;
    for (face_index, (face, material)) in source_polygons.iter().enumerate() {
        corner_base.push(corner_total);
        corner_total += face.len();
        for (corner, (first, second)) in cyclic_pairs(face).enumerate() {
            edge_faces
                .entry(edge_key(*first, *second))
                .or_default()
                .push((*first, *second, *material));
            edge_owner
                .entry(edge_key(*first, *second))
                .or_insert((face_index, corner));
        }
    }
    if use_even_offset {
        for (face, _) in &source_polygons {
            let mut face_normal = DVec3::ZERO;
            for (first, second) in cyclic_pairs(face) {
                face_normal += source_points[*first].cross(source_points[*second]);
            }
            face_normal = face_normal.normalize_or_zero();
            for (index, vertex) in face.iter().enumerate() {
                let current = source_points[*vertex];
                let previous = source_points[face[(index + face.len() - 1) % face.len()]];
                let next = source_points[face[(index + 1) % face.len()]];
                let angle = (previous - current)
                    .normalize_or_zero()
                    .dot((next - current).normalize_or_zero())
                    .clamp(-1.0, 1.0)
                    .acos()
                    .max(f64::EPSILON);
                let alignment = normals[*vertex].dot(face_normal).abs().max(1.0e-6);
                even_scales[*vertex] += angle / alignment;
                angle_totals[*vertex] += angle;
            }
        }
    }
    let even_scale = |index: usize| {
        if use_even_offset && angle_totals[index] > 0.0 {
            even_scales[index] / angle_totals[index]
        } else {
            1.0
        }
    };
    let original_offset = (offset + 1.0) * 0.5 * thickness;
    let shell_offset = (offset - 1.0) * 0.5 * thickness;
    let mut points = Vec::with_capacity(source_points.len() * 2);
    let mut point_sources: Vec<usize> = (0..source_points.len()).collect();
    points.extend(
        source_points
            .iter()
            .enumerate()
            .map(|(index, point)| *point + normals[index] * (original_offset * even_scale(index))),
    );
    let rim_only = use_rim && use_rim_only;
    let mut duplicate_indices = vec![None; source_points.len()];
    if rim_only {
        for faces in edge_faces.values().filter(|faces| faces.len() == 1) {
            let (first, second, _) = faces[0];
            for vertex in [first, second] {
                if duplicate_indices[vertex].is_none() {
                    duplicate_indices[vertex] = Some(points.len());
                    points.push(
                        source_points[vertex]
                            + normals[vertex] * (shell_offset * even_scale(vertex)),
                    );
                    point_sources.push(vertex);
                }
            }
        }
    } else {
        for (index, point) in source_points.iter().enumerate() {
            duplicate_indices[index] = Some(source_points.len() + index);
            points.push(*point + normals[index] * (shell_offset * even_scale(index)));
            point_sources.push(index);
        }
    }
    if points.iter().any(|point| !point.is_finite()) {
        return Err(invalid_parameter(
            modifier,
            "thickness",
            "a thickness that produces finite coordinates",
        ));
    }
    let mut output_polygons = Vec::with_capacity(source_polygons.len() * 2 + edge_faces.len());
    let mut face_sources = Vec::with_capacity(source_polygons.len() * 2 + edge_faces.len());
    // Input corner each output corner copies, flattened in output order.
    let mut corner_sources: Vec<Option<usize>> = Vec::new();
    let mut push_polygon = |loops: Vec<(usize, usize)>, material: u32| {
        output_polygons.push((loops.iter().map(|&(vertex, _)| vertex).collect(), material));
        corner_sources.extend(loops.iter().map(|&(_, corner)| Some(corner)));
    };
    for (face_index, (face, material)) in source_polygons.iter().enumerate() {
        let base = corner_base[face_index];
        let mut outer: Vec<(usize, usize)> = face
            .iter()
            .enumerate()
            .map(|(corner, &vertex)| (vertex, base + corner))
            .collect();
        if flip {
            outer.reverse();
        }
        push_polygon(outer, *material);
        face_sources.push(Some(face_index));
        if !rim_only {
            let mut shell: Vec<(usize, usize)> = face
                .iter()
                .enumerate()
                .map(|(corner, &vertex)| {
                    (duplicate_indices[vertex].unwrap_or(vertex), base + corner)
                })
                .collect();
            // Blender copies the shell with reversed winding; a flipped mesh keeps the input winding.
            shell.reverse();
            if flip {
                shell.reverse();
            }
            push_polygon(shell, *material);
            face_sources.push(Some(face_index));
        }
    }
    if use_rim {
        for (edge, faces) in edge_faces.iter().filter(|(_, faces)| faces.len() == 1) {
            let (first, second, material) = faces[0];
            let (owner, first_corner) = edge_owner[edge];
            let owner_count = source_polygons[owner].0.len();
            let first_loop = corner_base[owner] + first_corner;
            let second_loop = corner_base[owner] + (first_corner + 1) % owner_count;
            let duplicate_first = duplicate_indices[first].unwrap_or(first);
            let duplicate_second = duplicate_indices[second].unwrap_or(second);
            // Blender's rim runs against its owner face along the boundary edge: second, first,
            // then the shell copies of first and second.
            let mut rim = vec![
                (second, second_loop),
                (first, first_loop),
                (duplicate_first, first_loop),
                (duplicate_second, second_loop),
            ];
            if flip {
                rim.reverse();
            }
            push_polygon(rim, material);
            face_sources.push(Some(owner));
        }
    }
    let mut output_loose_edges = Vec::with_capacity(source_loose_edges.len().saturating_mul(2));
    output_loose_edges.extend(source_loose_edges.iter().copied());
    if !rim_only {
        output_loose_edges.extend(source_loose_edges.iter().filter_map(|[first, second]| {
            Some([duplicate_indices[*first]?, duplicate_indices[*second]?])
        }));
    }
    let origin = Origin::copies(point_sources.into_iter().map(Some), face_sources);
    let mut result = build_mesh(points, output_polygons, output_loose_edges, &origin, mesh)?;
    clnor::transfer_custom_normals(
        mesh,
        &mut result,
        &corner_sources,
        NormalCodeOptions::default(),
    )?;
    Ok(result)
}

/// Vertex normals the solidify offset moves along. The default is Blender's `vert_normals_true`:
/// face normals weighted by the approximate corner angle. With `use_quality_normals` it is Blender's
/// `mesh_calc_hq_normal`: each edge adds the angle-weighted midpoint of its faces' normals (or its
/// one face's normal), and a zero sum falls back to the default.
fn vertex_normals(points: &[DVec3], polygons: &[Polygon], quality: bool) -> Vec<DVec3> {
    let face_normals: Vec<DVec3> = polygons
        .iter()
        .map(|(face, _)| {
            let corners: Vec<DVec3> = face.iter().map(|&vertex| points[vertex]).collect();
            face_normal(&corners)
        })
        .collect();
    let mut angle_normals = vec![DVec3::ZERO; points.len()];
    for ((face, _), normal) in polygons.iter().zip(&face_normals) {
        let count = face.len();
        for (corner, &vertex) in face.iter().enumerate() {
            let previous =
                (points[face[(corner + count - 1) % count]] - points[vertex]).normalize_or_zero();
            let next = (points[face[(corner + 1) % count]] - points[vertex]).normalize_or_zero();
            angle_normals[vertex] += *normal * safe_acos_approx(previous.dot(next));
        }
    }
    let angle_normals: Vec<DVec3> = angle_normals
        .into_iter()
        .map(DVec3::normalize_or_zero)
        .collect();
    if !quality {
        return angle_normals;
    }
    let mut faces_by_edge: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (index, (face, _)) in polygons.iter().enumerate() {
        for (first, second) in cyclic_pairs(face) {
            faces_by_edge
                .entry(edge_key(*first, *second))
                .or_default()
                .push(index);
        }
    }
    let mut sums = vec![DVec3::ZERO; points.len()];
    for (&(first, second), faces) in &faces_by_edge {
        let edge_normal = match *faces.as_slice() {
            [one] => face_normals[one],
            [one, other] => angle_weighted_midpoint(face_normals[one], face_normals[other]),
            _ => continue,
        };
        sums[first] += edge_normal;
        sums[second] += edge_normal;
    }
    sums.into_iter()
        .zip(angle_normals)
        .map(|(sum, fallback)| {
            let normal = sum.normalize_or_zero();
            if normal == DVec3::ZERO {
                fallback
            } else {
                normal
            }
        })
        .collect()
}

/// Blender's `mid_v3_v3v3_angle_weighted`: the midpoint direction of two unit normals, scaled by
/// `4 / pi` times the arc of their angle.
fn angle_weighted_midpoint(first: DVec3, second: DVec3) -> DVec3 {
    let sum = first + second;
    let length = sum.length();
    sum.normalize_or_zero() * (4.0 / std::f64::consts::PI * (length / 2.0).min(1.0).acos())
}

/// The values of the element attribute `name` in storage order. A native point or edge attribute is read
/// through its element IDs. Otherwise a `blender_attributes` entry is read by position.
fn mesh_attribute_values(mesh: &Mesh, name: &str) -> Option<Vec<Value>> {
    if let Some(attribute) = mesh
        .attributes
        .get(attribute_keys::storage_key(name).as_ref())
    {
        let keyed = attribute.get("values").and_then(Value::as_object)?;
        let keys: Vec<String> = match attribute.get("domain").and_then(Value::as_str)? {
            "point" => mesh
                .vertices
                .iter()
                .map(|vertex| format!("v{}", vertex.id))
                .collect(),
            "edge" => mesh
                .edges
                .iter()
                .map(|edge| format!("e{}", edge.id))
                .collect(),
            _ => return None,
        };
        return Some(
            keys.iter()
                .map(|key| keyed.get(key).cloned().unwrap_or(Value::Null))
                .collect(),
        );
    }
    mesh.attributes
        .get("blender_attributes")?
        .as_array()?
        .iter()
        .find(|attribute| attribute.get("name").and_then(Value::as_str) == Some(name))?
        .get("values")?
        .as_array()
        .cloned()
}
#[expect(
    clippy::float_cmp,
    reason = "exact profile exponents select Blender's straight and circular profiles"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "bevel segment counts are bounded by 64"
)]
// Blender's BMesh bevel modifier distributes superellipse profile samples by chord length.
#[expect(
    clippy::too_many_lines,
    reason = "the profile sampler mirrors Blender's special cases and iterative chord solver"
)]
fn bevel_profile_coordinates(segments: usize, exponent: f64) -> Vec<(f64, f64)> {
    let mut coordinates = vec![(0.0, 0.0); segments + 1];
    if segments == 0 {
        return coordinates;
    }
    if exponent == 0.0 {
        let half = segments / 2;
        if segments.is_multiple_of(2) {
            for index in 0..=half {
                let y = 1.0 - index as f64 / half.max(1) as f64;
                coordinates[index] = (0.0, y);
                coordinates[segments - index] = (y, 0.0);
            }
        } else {
            let step = 1.0 / (half as f64 + std::f64::consts::FRAC_1_SQRT_2);
            for index in 0..=half {
                let y = 1.0 - index as f64 * step;
                coordinates[index] = (0.0, y);
                coordinates[segments - index] = (y, 0.0);
            }
        }
        return coordinates;
    }
    if exponent >= 10_000.0 {
        let half = segments / 2;
        if segments.is_multiple_of(2) {
            for index in 0..=half {
                let x = index as f64 / half.max(1) as f64;
                coordinates[index] = (x, 1.0);
                coordinates[segments - index] = (1.0, x);
            }
        } else {
            let step = 1.0 / (half as f64 + std::f64::consts::FRAC_1_SQRT_2);
            for index in 0..=half {
                let x = index as f64 * step;
                coordinates[index] = (x, 1.0);
                coordinates[segments - index] = (1.0, x);
            }
        }
        return coordinates;
    }
    if exponent == 1.0 {
        for (index, coordinate) in coordinates.iter_mut().enumerate() {
            let fraction = index as f64 / segments as f64;
            *coordinate = (fraction, 1.0 - fraction);
        }
        return coordinates;
    }
    if exponent == 2.0 {
        for (index, coordinate) in coordinates.iter_mut().enumerate() {
            let angle = index as f64 / segments as f64 * std::f64::consts::FRAC_PI_2;
            *coordinate = (angle.sin(), angle.cos());
        }
        return coordinates;
    }

    let rbig = exponent > 1.0;
    let midpoint_x = if rbig {
        0.5_f64.powf(1.0 / exponent)
    } else {
        1.0 - 0.5_f64.powf(1.0 / exponent)
    };
    let last_unique_index = segments.div_ceil(2) - 1;
    for (index, coordinate) in coordinates
        .iter_mut()
        .enumerate()
        .take(last_unique_index + 1)
    {
        let x = index as f64 * midpoint_x / segments as f64 * 2.0;
        *coordinate = (x, superellipse_y(x, exponent, rbig));
    }
    coordinates[0].1 = 1.0;

    for _ in 0..10 {
        let mut distance_sum = 0.0;
        let mut minimum_distance = 2.0_f64;
        let mut maximum_distance = 0.0_f64;
        for index in 0..last_unique_index {
            let (first_x, first_y) = coordinates[index];
            let (second_x, second_y) = coordinates[index + 1];
            let distance = (second_x - first_x).hypot(second_y - first_y);
            distance_sum += distance;
            maximum_distance = maximum_distance.max(distance);
            minimum_distance = minimum_distance.min(distance);
        }
        let average_distance = if segments % 2 == 1 {
            let (x, y) = coordinates[last_unique_index];
            distance_sum += std::f64::consts::FRAC_1_SQRT_2 * (y - x);
            distance_sum / (last_unique_index as f64 + 0.5)
        } else {
            let (x, y) = coordinates[last_unique_index];
            distance_sum += (x - midpoint_x).hypot(y - midpoint_x);
            distance_sum / (last_unique_index as f64 + 1.0)
        };
        if maximum_distance - average_distance <= 1.0e-7
            && minimum_distance - average_distance >= 1.0e-7
        {
            break;
        }
        for index in 1..=last_unique_index {
            let x = superellipse_chord_endpoint(
                coordinates[index - 1].0,
                average_distance,
                exponent,
                rbig,
            );
            coordinates[index] = (x, superellipse_y(x, exponent, rbig));
        }
    }
    if segments.is_multiple_of(2) {
        coordinates[last_unique_index + 1] = (midpoint_x, midpoint_x);
    }
    for index in last_unique_index + 1..=segments {
        let (x, y) = coordinates[segments - index];
        coordinates[index] = (y, x);
    }
    coordinates
}

fn superellipse_y(x: f64, exponent: f64, rbig: bool) -> f64 {
    if rbig {
        (1.0 - x.powf(exponent)).powf(1.0 / exponent)
    } else {
        1.0 - (1.0 - (1.0 - x).powf(exponent)).powf(1.0 / exponent)
    }
}

fn superellipse_chord_endpoint(x: f64, distance: f64, exponent: f64, rbig: bool) -> f64 {
    let y = superellipse_y(x, exponent, rbig);
    let minimum_x = (x + std::f64::consts::FRAC_1_SQRT_2 * distance).min(1.0);
    let maximum_x = (x + distance).min(1.0);
    let minimum_error =
        (minimum_x - x).hypot(superellipse_y(minimum_x, exponent, rbig) - y) - distance;
    let maximum_error =
        (maximum_x - x).hypot(superellipse_y(maximum_x, exponent, rbig) - y) - distance;
    let mut x_new =
        maximum_x - maximum_error * (maximum_x - minimum_x) / (maximum_error - minimum_error);
    let mut minimum_x = minimum_x;
    let mut maximum_x = maximum_x;
    let mut minimum_error = minimum_error;
    let mut maximum_error = maximum_error;
    let mut updated_upper = true;

    for _ in 0..10 {
        let new_error = (x_new - x).hypot(superellipse_y(x_new, exponent, rbig) - y) - distance;
        if new_error.abs() < 1.0e-13 {
            break;
        }
        if new_error < 0.0 {
            minimum_x = x_new;
            minimum_error = new_error;
            x_new = if updated_upper {
                maximum_x
                    - maximum_error * (maximum_x - minimum_x) / (maximum_error - minimum_error)
            } else {
                (maximum_error / 2.0 * minimum_x - minimum_error * maximum_x)
                    / (maximum_error / 2.0 - minimum_error)
            };
            updated_upper = false;
        } else {
            maximum_x = x_new;
            maximum_error = new_error;
            x_new = if updated_upper {
                (maximum_error * minimum_x - minimum_error / 2.0 * maximum_x)
                    / (maximum_error - minimum_error / 2.0)
            } else {
                maximum_x
                    - maximum_error * (maximum_x - minimum_x) / (maximum_error - minimum_error)
            };
            updated_upper = true;
        }
    }
    x_new
}

fn bevel_profile_point(start: DVec3, end: DVec3, middle: DVec3, x: f64, y: f64) -> DVec3 {
    start * (1.0 - x) + end * (1.0 - y) + middle * (x + y - 1.0)
}

fn bevel_profile_middle(start: DVec3, end: DVec3, edge_start: DVec3, edge_end: DVec3) -> DVec3 {
    let edge_direction = (edge_end - edge_start).normalize_or_zero();
    let profile_direction = (end - start).normalize_or_zero();
    let cosine = edge_direction.dot(profile_direction);
    let denominator = 1.0 - cosine * cosine;
    let offset = edge_start - start;
    let distance_along_edge = if denominator.abs() > 1.0e-12 {
        (cosine * profile_direction.dot(offset) - edge_direction.dot(offset)) / denominator
    } else {
        ((start + end) * 0.5 - edge_start).dot(edge_direction)
    };
    edge_start + edge_direction * distance_along_edge
}
#[expect(
    clippy::too_many_lines,
    reason = "bevel builds the face, edge-profile, and corner patches in one topology pass"
)]
fn bevel(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    let width = number_param(modifier, "width")?;
    if width < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "width",
            "a non-negative finite number",
        ));
    }
    if width == 0.0 || mesh.edges.is_empty() {
        return Ok(mesh.clone());
    }

    let segments = usize::try_from(uint_param(modifier, "segments")?)
        .ok()
        .filter(|value| (1..=64).contains(value))
        .ok_or_else(|| invalid_parameter(modifier, "segments", "an integer from 1 through 64"))?;
    let limit_method = string_param(modifier, "limit_method")?.to_ascii_uppercase();
    if !matches!(
        limit_method.as_str(),
        "NONE" | "ANGLE" | "WEIGHT" | "VGROUP"
    ) {
        return Err(invalid_parameter(
            modifier,
            "limit_method",
            "NONE, ANGLE, WEIGHT, or VGROUP",
        ));
    }
    let affect = string_param(modifier, "affect")?.to_ascii_uppercase();
    if !matches!(affect.as_str(), "EDGES" | "VERTICES") {
        return Err(invalid_parameter(modifier, "affect", "EDGES or VERTICES"));
    }
    let profile = number_param(modifier, "profile")?;
    if !(0.0..=1.0).contains(&profile) {
        return Err(invalid_parameter(
            modifier,
            "profile",
            "a number from 0 through 1",
        ));
    }
    let clamp_overlap = bool_param(modifier, "use_clamp_overlap")?;
    let loop_slide = bool_param(modifier, "loop_slide")?;
    let harden_normals = bool_param(modifier, "harden_normals")?;
    let mark_sharp = bool_param(modifier, "mark_sharp")?;
    // Blender's mark_seam copies the input's seams onto the new rungs (check_edge_data_seam_sharp_edges),
    // so it changes nothing on a mesh without seams.
    let input_seam = edge_flags(mesh, "seam").contains(&true);
    if bool_param(modifier, "mark_seam")? && input_seam {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "bevel mark_seam on a mesh with seam edges is not supported",
            json!({
                "feature_id": "modifier.bevel.mark_seam",
                "modifier_id": modifier.id,
            }),
        ));
    }
    let offset_type = string_param(modifier, "offset_type")?.to_ascii_uppercase();
    if !matches!(
        offset_type.as_str(),
        "OFFSET" | "WIDTH" | "DEPTH" | "PERCENT"
    ) {
        return Err(invalid_parameter(
            modifier,
            "offset_type",
            "OFFSET, WIDTH, DEPTH, or PERCENT",
        ));
    }

    // harden_normals needs the single-edge bevel. mark_sharp only marks rungs of input sharp edges
    // (mark_sharp_rungs), so it needs that bevel only when the input has sharp edges.
    let input_sharp = sharp_edge_flags(mesh)?.contains(&true);
    let single_edge_bevel = affect == "EDGES" && segments == 1;
    if (harden_normals || (mark_sharp && input_sharp)) && !single_edge_bevel {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "bevel harden_normals and mark_sharp on sharp edges are evaluated for single-segment EDGES bevels only",
            json!({
                "feature_id": "modifier.bevel.shading_segments",
                "modifier_id": modifier.id,
            }),
        ));
    }
    let source_points = positions(mesh);
    let source_polygons = polygons(mesh)?;
    if source_polygons.is_empty() {
        return Ok(mesh.clone());
    }
    let mut face_normals = Vec::with_capacity(source_polygons.len());
    let mut edge_faces: BevelEdgeFacesWithMaterial = BTreeMap::new();
    for (face_index, (face, material)) in source_polygons.iter().enumerate() {
        let mut normal = DVec3::ZERO;
        for (first, second) in cyclic_pairs(face) {
            normal += source_points[*first].cross(source_points[*second]);
            edge_faces
                .entry(edge_key(*first, *second))
                .or_default()
                .push((face_index, *first, *second, *material));
        }
        face_normals.push(normal.normalize_or_zero());
    }

    let mut selected_edges = HashSet::new();
    let angle_limit = number_param(modifier, "angle_limit")?;
    if !(0.0..=std::f64::consts::PI).contains(&angle_limit) {
        return Err(invalid_parameter(
            modifier,
            "angle_limit",
            "an angle from 0 through pi",
        ));
    }
    let weight_attribute_name = if limit_method == "VGROUP" {
        modifier.params.get("vertex_group").and_then(Value::as_str)
    } else if affect == "VERTICES" {
        modifier
            .params
            .get("vertex_weight")
            .and_then(Value::as_str)
            .or(Some("bevel_weight_vert"))
    } else {
        modifier
            .params
            .get("edge_weight")
            .and_then(Value::as_str)
            .or(Some("bevel_weight_edge"))
    };
    let weight_attribute = weight_attribute_name.and_then(|name| mesh_attribute_values(mesh, name));
    let weight_attribute = weight_attribute.as_deref();
    let vertex_id_indices = id_indices(mesh);
    let edge_id_indices: HashMap<_, _> = mesh
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| {
            (
                edge_key(
                    vertex_id_indices[&edge.vertices[0]],
                    vertex_id_indices[&edge.vertices[1]],
                ),
                index,
            )
        })
        .collect();
    let mut selected_vertices = HashSet::new();
    for (edge, sides) in &edge_faces {
        if sides.len() != 2 {
            continue;
        }
        let mut selected = match limit_method.as_str() {
            "NONE" => true,
            "ANGLE" => face_normals[sides[0].0].dot(face_normals[sides[1].0]) < angle_limit.cos(),
            "WEIGHT" => {
                if affect == "VERTICES" {
                    false
                } else {
                    edge_id_indices
                        .get(edge)
                        .and_then(|index| weight_attribute.and_then(|weights| weights.get(*index)))
                        .and_then(Value::as_f64)
                        .is_some_and(|value| value > 0.0)
                }
            }
            "VGROUP" => {
                let first = weight_attribute
                    .and_then(|weights| weights.get(sides[0].1))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let second = weight_attribute
                    .and_then(|weights| weights.get(sides[0].2))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                first >= 0.5 && second >= 0.5
            }
            _ => false,
        };
        if affect == "VERTICES" {
            if selected && !matches!(limit_method.as_str(), "WEIGHT" | "VGROUP") {
                selected_vertices.insert(edge.0);
                selected_vertices.insert(edge.1);
            }
            selected = false;
        }
        if selected {
            selected_edges.insert(*edge);
        }
    }
    if affect == "VERTICES" {
        match limit_method.as_str() {
            "NONE" => {
                for (face, _) in &source_polygons {
                    selected_vertices.extend(face.iter().copied());
                }
            }
            "WEIGHT" | "VGROUP" => {
                for (vertex, _) in mesh.vertices.iter().enumerate() {
                    let weight = weight_attribute
                        .and_then(|weights| weights.get(vertex))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    let selected = if limit_method == "VGROUP" {
                        weight >= 0.5
                    } else {
                        weight > 0.0
                    };
                    if selected {
                        selected_vertices.insert(vertex);
                    }
                }
            }
            _ => {}
        }
        if selected_vertices.is_empty() {
            return Ok(mesh.clone());
        }
    } else if selected_edges.is_empty() {
        return Ok(mesh.clone());
    }

    if affect == "VERTICES" {
        return bevel_vertices(
            mesh,
            &source_points,
            &source_polygons,
            &selected_vertices,
            width,
            segments,
            profile,
            clamp_overlap,
            loop_slide,
            &offset_type,
        );
    }
    if segments > 1 {
        let mut vertex_edge_counts = vec![(0_usize, 0_usize); source_points.len()];
        for edge in edge_faces.keys() {
            for vertex in [edge.0, edge.1] {
                vertex_edge_counts[vertex].0 += 1;
                if selected_edges.contains(edge) {
                    vertex_edge_counts[vertex].1 += 1;
                }
            }
        }
        if let Some((vertex_index, (incident_edges, selected_edges))) = vertex_edge_counts
            .iter()
            .enumerate()
            .find(|(_, (incident, selected))| *incident >= 4 && *selected >= 2)
        {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Bevel vertex meshes with four or more incident edges and multiple beveled edges are not evaluated",
                json!({
                    "feature_id":"modifier.bevel.high_valence_vertex_mesh",
                    "modifier_id":modifier.id,
                    "vertex_id":mesh.vertices.get(vertex_index).map(|vertex|vertex.id),
                    "incident_edges":incident_edges,
                    "selected_edges":selected_edges
                }),
            ));
        }
    }
    let exponent = if profile <= 0.0 {
        0.0
    } else if profile >= 0.95 {
        1.0e4
    } else {
        (-2.0_f64.ln() / profile.sqrt().ln()).clamp(0.0, 1.0e4)
    };
    let profile_coordinates = bevel_profile_coordinates(segments, exponent);
    let mut points = Vec::new();
    let mut point_vertices = Vec::new();
    let mut face_corner = HashMap::new();
    let mut corner_width = HashMap::new();
    for (face_index, (face, _)) in source_polygons.iter().enumerate() {
        for (corner, vertex) in face.iter().copied().enumerate() {
            let previous = face[(corner + face.len() - 1) % face.len()];
            let next = face[(corner + 1) % face.len()];
            let incoming_key = edge_key(previous, vertex);
            let outgoing_key = edge_key(vertex, next);
            let incoming_selected = selected_edges.contains(&incoming_key);
            let outgoing_selected = selected_edges.contains(&outgoing_key);
            let vertex_selected = selected_vertices.contains(&vertex);
            let mut amount = width;
            let corner_angle = (source_points[previous] - source_points[vertex])
                .normalize_or_zero()
                .dot((source_points[next] - source_points[vertex]).normalize_or_zero())
                .clamp(-1.0, 1.0)
                .acos()
                .max(1.0e-6);
            let shorter_edge = source_points[vertex]
                .distance(source_points[previous])
                .min(source_points[vertex].distance(source_points[next]));
            if !loop_slide && (incoming_selected || outgoing_selected || vertex_selected) {
                amount /= (corner_angle * 0.5).sin().max(1.0e-6);
            }
            if offset_type == "PERCENT" {
                amount *= shorter_edge / 100.0;
            } else if offset_type == "WIDTH" || offset_type == "DEPTH" {
                let angle = (source_points[previous] - source_points[vertex])
                    .normalize_or_zero()
                    .dot((source_points[next] - source_points[vertex]).normalize_or_zero())
                    .clamp(-1.0, 1.0)
                    .acos()
                    .max(1.0e-6);
                if offset_type == "WIDTH" {
                    amount *= (angle * 0.5).sin();
                } else {
                    amount /= (angle * 0.5).sin().max(1.0e-6);
                }
            }
            if clamp_overlap {
                amount = amount.min(shorter_edge * 0.49);
            }
            corner_width.insert((face_index, vertex), amount);
            let inward_incoming = face_normals[face_index]
                .cross((source_points[vertex] - source_points[previous]).normalize_or_zero());
            let inward_outgoing = face_normals[face_index]
                .cross((source_points[next] - source_points[vertex]).normalize_or_zero());
            let mut point = source_points[vertex];
            if incoming_selected && outgoing_selected {
                let denominator = 1.0 + inward_incoming.dot(inward_outgoing);
                if denominator.abs() > 1.0e-12 {
                    point += (inward_incoming + inward_outgoing) * (amount / denominator);
                } else {
                    point += inward_incoming * amount;
                }
            } else if incoming_selected {
                point += inward_incoming * amount;
            } else if outgoing_selected {
                point += inward_outgoing * amount;
            }
            let index = points.len();
            points.push(point);
            // shortcut: corner vertices take one source vertex; Blender blends the bevel stencil of several corners, upgrade with those coefficients
            point_vertices.push(vertex);
            face_corner.insert((face_index, vertex), index);
        }
    }

    // Blender's square profile also uses its even-segment corner mesh (`square_out_adj_vmesh`) for
    // two segments, which the generic path below does not build.
    let square_two_segments =
        segments == 2 && bevel_vmesh::profile_super_r(profile) == bevel_vmesh::PRO_SQUARE_R;
    if affect == "EDGES" && (segments > 2 || square_two_segments) {
        let bound_of = |face: usize, vertex: usize| points[face_corner[&(face, vertex)]];
        match bevel_vmesh::bevel_edges(
            mesh,
            segments,
            bevel_vmesh::profile_super_r(profile),
            &source_points,
            &source_polygons,
            &selected_edges,
            &bound_of,
        ) {
            Ok(emission) => {
                return bevel_vmesh::finish(mesh, &source_polygons, emission, loose_edges(mesh)?);
            }
            Err(rejections) if uv_layers::has_uv_layers(mesh) => {
                return Err(bevel_uv_profile_error(modifier, mesh, profile, &rejections));
            }
            Err(_) => {}
        }
    }
    // Two segments on UV layers: the old path's UV rules are verified only for cube-like corners.
    if affect == "EDGES" && segments == 2 && uv_layers::has_uv_layers(mesh) {
        let bound_of = |face: usize, vertex: usize| points[face_corner[&(face, vertex)]];
        let rejections = bevel_vmesh::cube_corner_rejections(
            &source_points,
            &source_polygons,
            &selected_edges,
            &bound_of,
            bevel_vmesh::profile_super_r(profile),
        );
        if !rejections.is_empty() {
            return Err(bevel_uv_profile_error(modifier, mesh, profile, &rejections));
        }
    }
    let mut output_polygons = Vec::new();
    let mut face_sources = Vec::new();
    // Blender's smooth-flag source of each output face (`bevel_shading`), kept apart from the
    // interpolation source so corner data still blends from the face it came from.
    let mut shade_sources = Vec::new();
    let blender_order = bevel_shading::blender_edge_order(&source_polygons);
    for (face_index, (face, material)) in source_polygons.iter().enumerate() {
        face_sources.push(Some(face_index));
        shade_sources.push(Some(face_index));
        output_polygons.push((
            face.iter()
                .map(|vertex| face_corner[&(face_index, *vertex)])
                .collect(),
            *material,
        ));
    }
    let mut profile_points: HashMap<((usize, usize), usize, usize), usize> = HashMap::new();
    // Blender evaluates each bevel loop's UV at the edge's middle point (profile loops) or at the
    // center projected onto the incoming edge (corner-center loops); these are the query points.
    let mut profile_uv_points: HashMap<usize, DVec3> = HashMap::new();
    let mut uv_queries: HashMap<(usize, usize), DVec3> = HashMap::new();
    let mut uv_face_overrides: HashMap<usize, usize> = HashMap::new();
    if affect == "EDGES" {
        for (edge, sides) in &edge_faces {
            if !selected_edges.contains(edge) || sides.len() != 2 {
                continue;
            }
            let [side_a, side_b] = [sides[0], sides[1]];
            for vertex in [edge.0, edge.1] {
                let a = face_corner[&(side_a.0, vertex)];
                let b = face_corner[&(side_b.0, vertex)];
                let p = source_points[vertex];
                let other = if edge.0 == vertex { edge.1 } else { edge.0 };
                let middle = bevel_profile_middle(points[a], points[b], p, source_points[other]);
                for (segment, &(x, y)) in profile_coordinates
                    .iter()
                    .enumerate()
                    .take(segments)
                    .skip(1)
                {
                    let index = points.len();
                    points.push(bevel_profile_point(points[a], points[b], middle, x, y));
                    point_vertices.push(vertex);
                    profile_points.insert((*edge, vertex, segment), index);
                    profile_uv_points.insert(index, middle);
                }
            }
            let point_at = |vertex: usize, side: usize, segment: usize| {
                if segment == 0 {
                    face_corner[&(sides[side].0, vertex)]
                } else if segment == segments {
                    face_corner[&(sides[1 - side].0, vertex)]
                } else {
                    profile_points[&(*edge, vertex, segment)]
                }
            };
            let (before, after) = bevel_shading::strip_faces(*edge, side_a, side_b);
            // Pieces keep side A's layout (the UV overrides index them so). A piece whose `before`
            // face is side B runs the other way round, as Blender's strip does.
            let reversed = before != side_a.0;
            for segment in 0..segments {
                let (start, end) = *edge;
                let material = side_a.3;
                // The strip's second half takes its loop UVs from the far face, as Blender does.
                if 2 * segment >= segments {
                    uv_face_overrides.insert(output_polygons.len(), side_b.0);
                }
                face_sources.push(Some(side_a.0));
                let before_relative = if reversed {
                    segments - 1 - segment
                } else {
                    segment
                };
                shade_sources.push(Some(bevel_shading::strip_face_for_segment(
                    before,
                    after,
                    segments,
                    before_relative,
                )));
                let mut corners = vec![
                    point_at(start, 0, segment),
                    point_at(end, 0, segment),
                    point_at(end, 0, segment + 1),
                    point_at(start, 0, segment + 1),
                ];
                if reversed {
                    corners.reverse();
                }
                output_polygons.push((corners, material));
            }
        }
    }

    // Blender's loops are rebuilt from the strips and corners emitted so far (see `bevel_shading`).
    let corner_start = output_polygons.len();
    let input_sharp: HashSet<(usize, usize)> = super::normals::sharp_edge_flags(mesh)?
        .into_iter()
        .zip(&mesh.edges)
        .filter(|(flag, _)| *flag)
        .map(|(_, edge)| {
            edge_key(
                vertex_id_indices[&edge.vertices[0]],
                vertex_id_indices[&edge.vertices[1]],
            )
        })
        .collect();
    let sharp_points = if mark_sharp {
        bevel_shading::mark_sharp_rungs(
            &edge_faces,
            &selected_edges,
            &input_sharp,
            &face_corner,
            source_points.len(),
        )
    } else {
        Vec::new()
    };
    let input_custom = clnor::corner_codes(mesh)?.is_some();
    let mut incident_faces = vec![Vec::new(); source_points.len()];
    for (face_index, (face, material)) in source_polygons.iter().enumerate() {
        for (corner, vertex) in face.iter().copied().enumerate() {
            incident_faces[vertex].push((face_index, corner, *material));
        }
    }
    for (vertex, faces) in incident_faces.iter().enumerate() {
        let affected = if affect == "VERTICES" {
            selected_vertices.contains(&vertex)
        } else {
            edge_faces.iter().any(|(edge, _)| {
                selected_edges.contains(edge) && (edge.0 == vertex || edge.1 == vertex)
            })
        };
        if !affected || faces.len() < 3 {
            continue;
        }
        let mut corner_center = source_points[vertex];
        if affect == "EDGES"
            && faces.len() == 3
            && selected_edges
                .iter()
                .filter(|edge| edge.0 == vertex || edge.1 == vertex)
                .count()
                == 3
        {
            let mut normal_sum = DVec3::ZERO;
            for (face_index, _, _) in faces {
                normal_sum += face_normals[*face_index];
            }
            let amount = faces
                .iter()
                .map(|(face_index, _, _)| corner_width[&(*face_index, vertex)])
                .sum::<f64>()
                / faces.len() as f64;
            // Blender's corner center lies on the superellipsoid of exponent e (e=2: sphere).
            let displacement = amount * (1.0 - 3.0_f64.powf(-1.0 / exponent));
            corner_center -= normal_sum * displacement;
        } else {
            let corners = faces
                .iter()
                .map(|(face_index, _, _)| points[face_corner[&(*face_index, vertex)]])
                .fold(DVec3::ZERO, |sum, point| sum + point);
            corner_center = corners / faces.len() as f64;
        }
        if affect == "VERTICES" || segments == 1 {
            let boundary_faces: Vec<usize> =
                faces.iter().map(|(face_index, _, _)| *face_index).collect();
            let boundary: Vec<usize> = boundary_faces
                .iter()
                .map(|face_index| face_corner[&(*face_index, vertex)])
                .collect();
            let boundary =
                bevel_shading::orient_corner(vertex, &boundary_faces, boundary, &edge_faces);
            if boundary.len() >= 3 {
                face_sources.push(Some(faces[0].0));
                shade_sources.push(Some(
                    bevel_shading::corner_source(
                        vertex,
                        &blender_order,
                        &edge_faces,
                        &selected_edges,
                    )
                    .unwrap_or(faces[0].0),
                ));
                output_polygons.push((boundary, faces[0].2));
            }
            continue;
        }
        let center_index = points.len();
        // shortcut: corner vertices take one source vertex; Blender blends the bevel stencil of several corners, upgrade with those coefficients
        points.push(corner_center);
        point_vertices.push(vertex);
        for (face_index, corner, material) in faces {
            let face = &source_polygons[*face_index].0;
            let previous = face[(*corner + face.len() - 1) % face.len()];
            let next = face[(*corner + 1) % face.len()];
            let mut patch = vec![face_corner[&(*face_index, vertex)]];
            let outgoing_edge = edge_key(vertex, next);
            let incoming_edge = edge_key(previous, vertex);
            if affect == "EDGES" && selected_edges.contains(&outgoing_edge) {
                patch.extend(
                    (1..segments).map(|segment| profile_points[&(outgoing_edge, vertex, segment)]),
                );
            }
            let center_corner = patch.len();
            patch.push(center_index);
            if affect == "EDGES" && selected_edges.contains(&incoming_edge) {
                patch.extend(
                    (1..segments)
                        .rev()
                        .map(|segment| profile_points[&(incoming_edge, vertex, segment)]),
                );
            }
            let incoming = (source_points[previous] - source_points[vertex]).normalize_or_zero();
            uv_queries.insert(
                (output_polygons.len(), center_corner),
                source_points[vertex]
                    + incoming * (corner_center - source_points[vertex]).dot(incoming),
            );
            face_sources.push(Some(*face_index));
            shade_sources.push(Some(*face_index));
            output_polygons.push((patch, *material));
        }
    }
    if output_polygons.is_empty() {
        return Ok(mesh.clone());
    }
    // Blender interpolates each bevel loop from its source face at the loop's query point.
    let mut query_points = Vec::new();
    let mut query_polygons = Vec::with_capacity(output_polygons.len());
    for (polygon, (vertices, _)) in output_polygons.iter().enumerate() {
        let mut ids = Vec::with_capacity(vertices.len());
        for (corner, vertex) in vertices.iter().copied().enumerate() {
            ids.push(query_points.len());
            query_points.push(
                uv_queries
                    .get(&(polygon, corner))
                    .or_else(|| profile_uv_points.get(&vertex))
                    .copied()
                    .unwrap_or(points[vertex]),
            );
        }
        query_polygons.push(ids);
    }
    let uv_sources: Vec<Option<usize>> = face_sources
        .iter()
        .enumerate()
        .map(|(polygon, source)| match uv_face_overrides.get(&polygon) {
            Some(face) => Some(*face),
            None => *source,
        })
        .collect();
    let corners = mean_value_corners(
        mesh,
        &query_points,
        query_polygons
            .iter()
            .zip(&uv_sources)
            .map(|(ids, source)| (ids.as_slice(), *source)),
    );
    let origin = Origin {
        corners: Some(corners),
        ..Origin::copies(point_vertices.iter().copied().map(Some), shade_sources)
    };
    let source_count = source_polygons.len();
    let kinds: Vec<bevel_shading::Kind> = (0..output_polygons.len())
        .map(|polygon| {
            if polygon < source_count {
                bevel_shading::Kind::Source
            } else if polygon < corner_start {
                bevel_shading::Kind::Strip
            } else {
                bevel_shading::Kind::Corner
            }
        })
        .collect();
    let mut result = build_mesh(points, output_polygons, loose_edges(mesh)?, &origin, mesh)?;
    bevel_shading::shade_output(
        &mut result,
        &kinds,
        &sharp_points,
        harden_normals,
        input_custom,
    )?;
    Ok(result)
}
/// Boundary point of a VERTICES bevel on the edge from `vertex` to `neighbor`, seen from the face
/// corner `previous, vertex, next`: the offset along that edge, as a point and as its fraction of
/// the edge.
#[expect(
    clippy::too_many_arguments,
    reason = "the boundary point depends on each bevel setting and on the face corner"
)]
fn vertex_bevel_edge_point(
    source_points: &[DVec3],
    vertex: usize,
    previous: usize,
    next: usize,
    neighbor: usize,
    width: f64,
    offset_type: &str,
    loop_slide: bool,
    clamp_overlap: bool,
) -> (DVec3, f64) {
    let edge_length = source_points[vertex].distance(source_points[neighbor]);
    let mut amount = if offset_type == "PERCENT" {
        edge_length * width / 100.0
    } else {
        width
    };
    let other = if neighbor == previous { next } else { previous };
    let angle = (source_points[other] - source_points[vertex])
        .normalize_or_zero()
        .dot((source_points[neighbor] - source_points[vertex]).normalize_or_zero())
        .clamp(-1.0, 1.0)
        .acos();
    if offset_type == "WIDTH" {
        amount *= (angle * 0.5).sin();
    } else if offset_type == "DEPTH" {
        amount /= (angle * 0.5).sin().max(1.0e-6);
    }
    if !loop_slide {
        amount /= (angle * 0.5).sin().max(1.0e-6);
    }
    if clamp_overlap {
        amount = amount.min(edge_length * 0.49);
    }
    let fraction = amount / edge_length.max(f64::EPSILON);
    (
        source_points[vertex].lerp(source_points[neighbor], fraction),
        fraction,
    )
}
#[expect(
    clippy::too_many_lines,
    reason = "vertex bevel shares edge offsets between original faces and fills each corner patch"
)]
#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
fn bevel_vertices(
    mesh: &Mesh,
    source_points: &[DVec3],
    source_polygons: &[Polygon],
    selected_vertices: &HashSet<usize>,
    width: f64,
    segments: usize,
    profile: f64,
    clamp_overlap: bool,
    loop_slide: bool,
    offset_type: &str,
) -> Result<Mesh> {
    if segments > 2 {
        let mut edge_point = |vertex: usize, previous: usize, next: usize, neighbor: usize| {
            vertex_bevel_edge_point(
                source_points,
                vertex,
                previous,
                next,
                neighbor,
                width,
                offset_type,
                loop_slide,
                clamp_overlap,
            )
            .0
        };
        if let Some(emission) = bevel_vmesh::bevel_vertices(
            mesh,
            segments,
            bevel_vmesh::profile_super_r(profile),
            profile,
            source_points,
            source_polygons,
            selected_vertices,
            &mut edge_point,
        ) {
            return bevel_vmesh::finish(mesh, source_polygons, emission, loose_edges(mesh)?);
        }
    }
    let exponent = if profile <= 0.0 {
        1.0e4
    } else {
        (-2.0_f64.ln() / profile.sqrt().ln()).clamp(0.0, 1.0e4)
    };
    let mut points = source_points.to_vec();
    // shortcut: corner vertices take one source vertex; Blender blends the bevel stencil of several corners, upgrade with those coefficients
    let mut point_origins: Vec<Vec<(usize, f64)>> = (0..source_points.len())
        .map(|vertex| vec![(vertex, 1.0)])
        .collect();
    let mut edge_points = HashMap::new();
    let mut patch_graph = vec![HashMap::<usize, Vec<usize>>::new(); source_points.len()];
    let mut corner_sectors = vec![Vec::new(); source_points.len()];
    let mut output_polygons = Vec::with_capacity(source_polygons.len());
    let mut face_sources = Vec::with_capacity(source_polygons.len());
    let mut uv_queries: HashMap<(usize, usize), DVec3> = HashMap::new();
    let mut uv_face_overrides: HashMap<usize, usize> = HashMap::new();
    for (face_index, (face, material)) in source_polygons.iter().enumerate() {
        let mut output_face = Vec::with_capacity(face.len() * (segments + 1));
        for (corner, vertex) in face.iter().copied().enumerate() {
            if !selected_vertices.contains(&vertex) {
                output_face.push(vertex);
                continue;
            }
            let previous = face[(corner + face.len() - 1) % face.len()];
            let next = face[(corner + 1) % face.len()];
            let mut edge_point = |neighbor: usize| {
                let key = (edge_key(vertex, neighbor), vertex);
                *edge_points.entry(key).or_insert_with(|| {
                    let (point, fraction) = vertex_bevel_edge_point(
                        source_points,
                        vertex,
                        previous,
                        next,
                        neighbor,
                        width,
                        offset_type,
                        loop_slide,
                        clamp_overlap,
                    );
                    let index = points.len();
                    points.push(point);
                    let bounded = fraction.clamp(0.0, 1.0);
                    point_origins.push(blend_weights([
                        (vertex, 1.0 - bounded),
                        (neighbor, bounded),
                    ]));
                    index
                })
            };
            let incoming = edge_point(previous);
            let outgoing = edge_point(next);
            let incoming_width = points[incoming].distance(source_points[vertex]);
            let outgoing_width = points[outgoing].distance(source_points[vertex]);
            let previous_direction =
                (source_points[previous] - source_points[vertex]).normalize_or_zero();
            let next_direction = (source_points[next] - source_points[vertex]).normalize_or_zero();
            let arc_center = source_points[vertex]
                + previous_direction * incoming_width
                + next_direction * outgoing_width;
            let start = (points[incoming] - arc_center).normalize_or_zero();
            let end = (points[outgoing] - arc_center).normalize_or_zero();
            let angle = start.dot(end).clamp(-1.0, 1.0).acos();
            let radius = (points[incoming].distance(arc_center)
                + points[outgoing].distance(arc_center))
                * 0.5;
            let mut arc_points = Vec::with_capacity(segments.saturating_sub(1));
            for segment in 1..segments {
                let t = segment as f64 / segments as f64;
                let mapped = t.powf(2.0 / exponent);
                let direction = if angle < 1.0e-8 {
                    start
                } else {
                    (start * (angle * (1.0 - mapped)).sin() + end * (angle * mapped).sin())
                        / angle.sin()
                };
                let index = points.len();
                points.push(arc_center + direction * radius);
                let origin = blend_weights(
                    point_origins[incoming]
                        .iter()
                        .map(|&(source, weight)| (source, weight * (1.0 - t)))
                        .chain(
                            point_origins[outgoing]
                                .iter()
                                .map(|&(source, weight)| (source, weight * t)),
                        ),
                );
                point_origins.push(origin);
                arc_points.push(index);
            }
            output_face.push(incoming);
            output_face.extend(arc_points.iter().copied());
            output_face.push(outgoing);
            patch_graph[vertex]
                .entry(incoming)
                .or_default()
                .push(outgoing);
            patch_graph[vertex]
                .entry(outgoing)
                .or_default()
                .push(incoming);
            corner_sectors[vertex].push((incoming, arc_points, outgoing, *material, face_index));
        }
        face_sources.push(Some(face_index));
        output_polygons.push((output_face, *material));
    }
    let circle_fullness = [
        0.0, 0.559, 0.642, 0.551, 0.646, 0.624, 0.646, 0.619, 0.647, 0.639, 0.647,
    ];
    for (vertex, graph) in patch_graph.iter().enumerate() {
        if graph.len() < 3 || graph.values().any(|neighbors| neighbors.len() != 2) {
            continue;
        }
        let Some(start) = graph.keys().copied().min() else {
            continue;
        };
        let Some(mut current) = graph[&start].iter().copied().min() else {
            continue;
        };
        let mut boundary = vec![start];
        let mut previous = start;
        for _ in 0..=graph.len() {
            if current == start {
                break;
            }
            if boundary.contains(&current) {
                break;
            }
            boundary.push(current);
            let Some(next) = graph[&current]
                .iter()
                .copied()
                .find(|neighbor| *neighbor != previous)
            else {
                break;
            };
            previous = current;
            current = next;
        }
        if current != start || boundary.len() != graph.len() {
            continue;
        }
        if segments == 1 {
            let sector = corner_sectors[vertex].first();
            let material = sector.map_or(0, |sector| sector.3);
            face_sources.push(sector.map(|sector| sector.4));
            output_polygons.push((boundary, material));
            continue;
        }
        let mut normal_sum = DVec3::ZERO;
        for (face, _) in source_polygons {
            if face.contains(&vertex) {
                normal_sum += decimate::decimate_face_normal(source_points, face);
            }
        }
        let amount = boundary
            .iter()
            .map(|point| points[*point].distance(source_points[vertex]))
            .sum::<f64>()
            / boundary.len() as f64;
        let fullness = if (profile - 0.5).abs() < 1.0e-6 && segments <= circle_fullness.len() {
            circle_fullness[segments - 1]
        } else if segments.is_multiple_of(2) {
            (2.4506 * profile - 0.000_003 * segments as f64 - 0.6266).max(0.0)
        } else {
            (2.3635 * profile + 0.000_152 * segments as f64 - 0.6060).max(0.0)
        };
        let center = source_points[vertex] - normal_sum * (amount * fullness * 0.301_073_345);
        let center_index = points.len();
        points.push(center);
        point_origins.push(vec![(vertex, 1.0)]);
        for (incoming, arc_points, outgoing, material, source) in &corner_sectors[vertex] {
            if segments == 2 {
                // Blender fills each edge-end point with one quad through the center, spanning the
                // arcs of the two faces that meet there. Its loop UVs come from the face where the
                // point is outgoing, and its center loop from the center projected onto that edge.
                if let Some(other) = corner_sectors[vertex]
                    .iter()
                    .find(|other| other.2 == *incoming)
                {
                    let edge_direction =
                        (points[*incoming] - source_points[vertex]).normalize_or_zero();
                    let along =
                        edge_direction * (center - source_points[vertex]).dot(edge_direction);
                    uv_queries.insert((output_polygons.len(), 2), source_points[vertex] + along);
                    uv_face_overrides.insert(output_polygons.len(), other.4);
                    face_sources.push(Some(*source));
                    output_polygons.push((
                        vec![*incoming, arc_points[0], center_index, other.1[0]],
                        *material,
                    ));
                }
            } else {
                let mut boundary = Vec::with_capacity(arc_points.len() + 2);
                boundary.push(*incoming);
                boundary.extend(arc_points.iter().copied());
                boundary.push(*outgoing);
                for pair in boundary.windows(2) {
                    face_sources.push(Some(*source));
                    output_polygons.push((vec![pair[0], pair[1], center_index], *material));
                }
            }
        }
    }
    let loose = loose_edges(mesh)?;
    let mut used = vec![false; points.len()];
    for (face, _) in &output_polygons {
        for vertex in face {
            used[*vertex] = true;
        }
    }
    for [first, second] in &loose {
        used[*first] = true;
        used[*second] = true;
    }
    let mut query_points = Vec::new();
    let mut query_polygons = Vec::with_capacity(output_polygons.len());
    for (polygon, (vertices, _)) in output_polygons.iter().enumerate() {
        let mut ids = Vec::with_capacity(vertices.len());
        for (corner, vertex) in vertices.iter().copied().enumerate() {
            ids.push(query_points.len());
            query_points.push(
                uv_queries
                    .get(&(polygon, corner))
                    .copied()
                    .unwrap_or(points[vertex]),
            );
        }
        query_polygons.push(ids);
    }
    let mut remap = vec![0; points.len()];
    let mut compact_points = Vec::with_capacity(points.len());
    let mut compact_origins = Vec::with_capacity(points.len());
    for ((index, point), origin) in points.into_iter().enumerate().zip(point_origins) {
        if used[index] {
            remap[index] = compact_points.len();
            compact_points.push(point);
            compact_origins.push(origin);
        }
    }
    for (face, _) in &mut output_polygons {
        for vertex in face {
            *vertex = remap[*vertex];
        }
    }
    let compact_loose = loose
        .into_iter()
        .map(|[first, second]| [remap[first], remap[second]])
        .collect();
    // Blender interpolates each bevel loop from its source face at the loop's query point.
    let uv_sources: Vec<Option<usize>> = face_sources
        .iter()
        .enumerate()
        .map(|(polygon, source)| match uv_face_overrides.get(&polygon) {
            Some(face) => Some(*face),
            None => *source,
        })
        .collect();
    let corners = mean_value_corners(
        mesh,
        &query_points,
        query_polygons
            .iter()
            .zip(&uv_sources)
            .map(|(ids, source)| (ids.as_slice(), *source)),
    );
    let origin = Origin {
        vertices: compact_origins,
        faces: face_sources,
        corners: Some(corners),
    };
    build_mesh(
        compact_points,
        output_polygons,
        compact_loose,
        &origin,
        mesh,
    )
}

fn decimate(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    if mesh.faces.is_empty() {
        return Ok(mesh.clone());
    }
    let mode = string_param(modifier, "decimate_type")?.to_ascii_uppercase();
    match mode.as_str() {
        "COLLAPSE" => {
            let ratio = number_param(modifier, "ratio")?;
            if !(0.0..=1.0).contains(&ratio) {
                return Err(invalid_parameter(
                    modifier,
                    "ratio",
                    "a number from 0 through 1",
                ));
            }
            if crate::float::equal_f64(ratio, 1.0) {
                return Ok(mesh.clone());
            }
            decimate::decimate_collapse(mesh, ratio, modifier)
        }
        "UNSUBDIV" => {
            let iterations = usize::try_from(uint_param(modifier, "iterations")?)
                .ok()
                .filter(|value| *value <= 64)
                .ok_or_else(|| {
                    invalid_parameter(modifier, "iterations", "an integer from 0 through 64")
                })?;
            decimate::decimate_unsubdivide(mesh, iterations)
        }
        "DISSOLVE" => {
            let angle = number_param(modifier, "angle_limit")?;
            if !(0.0..=std::f64::consts::PI).contains(&angle) {
                return Err(invalid_parameter(
                    modifier,
                    "angle_limit",
                    "an angle from 0 through pi",
                ));
            }
            decimate::decimate_dissolve(mesh, angle, modifier)
        }
        _ => Err(invalid_parameter(
            modifier,
            "decimate_type",
            "COLLAPSE, UNSUBDIV, or DISSOLVE",
        )),
    }
}

fn decimate_compact(
    points: Vec<DVec3>,
    mut polygons: Vec<Polygon>,
    mut loose: Vec<[usize; 2]>,
    origin: Origin,
    source: &Mesh,
    keep_vertices: &[bool],
) -> Result<Mesh> {
    // Blender's BMesh keeps the live vertices no face or edge uses; `keep_vertices` marks them.
    let mut used: Vec<bool> = (0..points.len())
        .map(|index| keep_vertices.get(index).copied().unwrap_or(false))
        .collect();
    for (face, _) in &polygons {
        for vertex in face {
            used[*vertex] = true;
        }
    }
    for [first, second] in &loose {
        used[*first] = true;
        used[*second] = true;
    }
    let mut remap = vec![0; points.len()];
    let mut compact_points = Vec::with_capacity(points.len());
    let mut compact_origin = Vec::with_capacity(points.len());
    for ((index, point), sources) in points.into_iter().enumerate().zip(origin.vertices) {
        if used[index] {
            remap[index] = compact_points.len();
            compact_points.push(point);
            compact_origin.push(sources);
        }
    }
    for (face, _) in &mut polygons {
        for vertex in face {
            *vertex = remap[*vertex];
        }
    }
    for [first, second] in &mut loose {
        *first = remap[*first];
        *second = remap[*second];
    }
    let origin = Origin {
        vertices: compact_origin,
        faces: origin.faces,
        corners: origin.corners,
    };
    build_mesh(compact_points, polygons, loose, &origin, source)
}

fn weld(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    let threshold_name = "merge_threshold";
    let threshold = number_param(modifier, threshold_name)?;
    if threshold < 0.0 {
        return Err(invalid_parameter(
            modifier,
            threshold_name,
            "a non-negative finite number",
        ));
    }
    let connected = match schema_value(modifier, "mode")?.as_str() {
        Some("ALL") => false,
        Some("CONNECTED") => true,
        _ => return Err(invalid_parameter(modifier, "mode", "ALL or CONNECTED")),
    };
    let only_loose_edges = bool_param(modifier, "loose_edges")?;
    if mesh.vertices.len() < 2 {
        return Ok(mesh.clone());
    }
    let points = positions(mesh);
    let merged = if connected {
        match connected_clusters(mesh, &points, threshold, only_loose_edges)? {
            Some(parent) => collapse_clusters(&points, parent, polygons(mesh)?, loose_edges(mesh)?),
            None => return Ok(mesh.clone()),
        }
    } else {
        merge_geometry(&points, polygons(mesh)?, loose_edges(mesh)?, threshold)?
    };
    // Blender returns the input unchanged when nothing merged.
    if merged.points.len() == points.len() {
        return Ok(mesh.clone());
    }
    let source_polygons = polygons(mesh)?;
    let mut corner_offsets = Vec::with_capacity(source_polygons.len());
    let mut corner_total = 0;
    for (vertices, _) in &source_polygons {
        corner_offsets.push(corner_total);
        corner_total += vertices.len();
    }
    // Each output corner copies the first source corner whose vertex merged into its vertex.
    let mut sources = Vec::with_capacity(corner_total);
    for ((vertices, _), &source) in merged.polygons.iter().zip(&merged.polygon_sources) {
        for &point in vertices {
            let members = &merged.point_members[point];
            let corner = source_polygons[source]
                .0
                .iter()
                .position(|vertex| members.contains(vertex));
            sources.push(corner.map(|corner| corner_offsets[source] + corner));
        }
    }
    let origin = Origin {
        vertices: merged
            .point_members
            .iter()
            .map(|members| mean_origin(members.iter().map(|&vertex| Some(vertex))))
            .collect(),
        faces: merged
            .polygon_sources
            .iter()
            .map(|&face| Some(face))
            .collect(),
        corners: None,
    };
    let mut result = build_mesh(merged.points, merged.polygons, merged.edges, &origin, mesh)?;
    clnor::transfer_custom_normals(mesh, &mut result, &sources, NormalCodeOptions::default())?;
    Ok(result)
}

fn displace(mesh: &Mesh, modifier: &Modifier, owner_world_matrix: DMat4) -> Result<Mesh> {
    validated(mesh)?;
    let strength = number_param(modifier, "strength")?;
    let mid_level = number_param(modifier, "mid_level")?;
    let direction = string_param_with_expected(
        modifier,
        "direction",
        "NORMAL, X, Y, Z, RGB_TO_XYZ, or CUSTOM_NORMAL",
    )?;
    if !matches!(
        direction,
        "NORMAL" | "NOR" | "X" | "Y" | "Z" | "RGB_TO_XYZ" | "CUSTOM_NORMAL" | "CLNOR"
    ) {
        return Err(invalid_parameter(
            modifier,
            "direction",
            "NORMAL, X, Y, Z, RGB_TO_XYZ, or CUSTOM_NORMAL",
        ));
    }
    let space = string_param_with_expected(modifier, "space", "LOCAL or GLOBAL")?;
    if !matches!(space, "LOCAL" | "GLOBAL") {
        return Err(invalid_parameter(modifier, "space", "LOCAL or GLOBAL"));
    }
    let has_texture = modifier
        .params
        .get("texture")
        .is_some_and(|value| !value.is_null());
    if strength == 0.0 || mesh.faces.is_empty() || (direction == "RGB_TO_XYZ" && !has_texture) {
        return Ok(mesh.clone());
    }
    if has_texture {
        return Err(PotError::new(
            ErrorCode::UnsupportedFeature,
            "displace modifier textures are not supported by the geometry evaluator",
        ));
    }

    let group_name = string_param_with_expected(modifier, "vertex_group", "a vertex-group name")?;
    let group_name = (!group_name.is_empty()).then_some(group_name);
    let group_weights = if let Some(group_name) = group_name {
        let group = mesh
            .attributes
            .get("vertex_groups")
            .and_then(Value::as_object)
            .and_then(|groups| groups.get(group_name))
            .and_then(Value::as_object)
            .ok_or_else(|| {
                invalid_parameter(
                    modifier,
                    "vertex_group",
                    "a vertex group present in the mesh attributes",
                )
            })?;
        Some(
            mesh.vertices
                .iter()
                .map(|vertex| {
                    let weight = group
                        .get(&format!("v{}", vertex.id))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    if weight.is_finite() && (0.0..=1.0).contains(&weight) {
                        Ok(weight)
                    } else {
                        Err(invalid_parameter(
                            modifier,
                            "vertex_group",
                            "finite weights in [0, 1]",
                        ))
                    }
                })
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    let invert_group = bool_param(modifier, "invert_vertex_group")?;
    let points = positions(mesh);
    // Blender's NORMAL direction is `mesh->vert_normals_true()` (MOD_displace.cc, NOR case): the
    // angle-weighted unit face normals of `normals_calc_verts` (mesh_normals.cc L194-224).
    let normals = vertex_normals(&points, &polygons(mesh)?, false);
    let delta = ((1.0 - mid_level) * strength).clamp(-10_000.0, 10_000.0);
    let mut result = mesh.clone();
    for (index, vertex) in result.vertices.iter_mut().enumerate() {
        let weight = group_weights.as_ref().map_or(1.0, |weights| {
            if invert_group {
                1.0 - weights[index]
            } else {
                weights[index]
            }
        });
        let displacement = delta * weight;
        let axis = match direction {
            "X" => Some(DVec3::X),
            "Y" => Some(DVec3::Y),
            "Z" => Some(DVec3::Z),
            _ => None,
        };
        let direction_vector = axis.map_or_else(
            || normals[index],
            |local| {
                if space == "GLOBAL" {
                    owner_world_matrix.transpose().transform_vector3(local)
                } else {
                    local
                }
            },
        );
        vertex.co = points[index] + direction_vector * displacement;
    }
    validated(&result)?;
    Ok(result)
}

fn smooth(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    let factor = number_param(modifier, "factor")?;
    if !(0.0..=1.0).contains(&factor) {
        return Err(invalid_parameter(
            modifier,
            "factor",
            "a number from 0 through 1",
        ));
    }
    let iterations = uint_param(modifier, "iterations")?;
    if iterations > 100 {
        return Err(invalid_parameter(
            modifier,
            "iterations",
            "an integer from 0 through 100",
        ));
    }
    let axes = [
        bool_param(modifier, "use_x")?,
        bool_param(modifier, "use_y")?,
        bool_param(modifier, "use_z")?,
    ];
    if factor == 0.0 || iterations == 0 || !axes.into_iter().any(|enabled| enabled) {
        return Ok(mesh.clone());
    }
    let group_name = string_param_with_expected(modifier, "vertex_group", "a vertex-group name")?;
    let group_name = (!group_name.is_empty()).then_some(group_name);
    let group_weights = if let Some(group_name) = group_name {
        let group = mesh
            .attributes
            .get("vertex_groups")
            .and_then(Value::as_object)
            .and_then(|groups| groups.get(group_name))
            .and_then(Value::as_object)
            .ok_or_else(|| {
                invalid_parameter(
                    modifier,
                    "vertex_group",
                    "a vertex group present in the mesh attributes",
                )
            })?;
        Some(
            mesh.vertices
                .iter()
                .map(|vertex| {
                    let weight = group
                        .get(&format!("v{}", vertex.id))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    if weight.is_finite() && (0.0..=1.0).contains(&weight) {
                        Ok(weight)
                    } else {
                        Err(invalid_parameter(
                            modifier,
                            "vertex_group",
                            "finite weights in [0, 1]",
                        ))
                    }
                })
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    let invert_group = bool_param(modifier, "invert_vertex_group")?;
    let index_by_id = id_indices(mesh);
    let mut neighbors = vec![BTreeSet::new(); mesh.vertices.len()];
    for edge in &mesh.edges {
        let first = index_by_id[&edge.vertices[0]];
        let second = index_by_id[&edge.vertices[1]];
        neighbors[first].insert(second);
        neighbors[second].insert(first);
    }
    let mut points = positions(mesh);
    for _ in 0..iterations {
        let previous = points.clone();
        let mut sums = vec![DVec3::ZERO; points.len()];
        let mut counts = vec![0_usize; points.len()];
        for (index, adjacent) in neighbors.iter().enumerate() {
            for neighbor in adjacent {
                sums[index] += (previous[index] + previous[*neighbor]) * 0.5;
                counts[index] += 1;
            }
        }
        for index in 0..points.len() {
            if counts[index] == 0 {
                continue;
            }
            let average = sums[index] / usize_as_f64(counts[index]);
            let group_factor = group_weights.as_ref().map_or(1.0, |weights| {
                if invert_group {
                    1.0 - weights[index]
                } else {
                    weights[index]
                }
            });
            let weight = factor * group_factor;
            for axis in 0..3 {
                if axes[axis] {
                    points[index][axis] =
                        previous[index][axis] * (1.0 - weight) + average[axis] * weight;
                }
            }
        }
    }
    let mut result = mesh.clone();
    for (vertex, point) in result.vertices.iter_mut().zip(points) {
        vertex.co = point;
    }
    validated(&result)?;
    Ok(result)
}

/// Keeps the first polygon for each vertex set, with the tag it arrived with.
pub(super) fn unique_tagged_polygons<T>(polygons: Vec<(T, Polygon)>) -> Vec<(T, Polygon)> {
    let mut seen = HashSet::new();
    polygons
        .into_iter()
        .filter(|(_, (face, _))| {
            let mut key = face.clone();
            key.sort_unstable();
            seen.insert(key)
        })
        .collect()
}

/// Merges each union-find cluster of `parent` into one point at the cluster centroid, recording the
/// input points of each cluster so that its provenance is their equal-weight mean.
fn collapse_clusters(
    points: &[DVec3],
    mut parent: Vec<usize>,
    polygons: Vec<Polygon>,
    edges: Vec<[usize; 2]>,
) -> MergedGeometry {
    let mut root_to_index = HashMap::new();
    let mut sums = Vec::<DVec3>::new();
    let mut point_members = Vec::<Vec<usize>>::new();
    let mut remap = vec![0; points.len()];
    for index in 0..points.len() {
        let root = find(&mut parent, index);
        let mapped = *root_to_index.entry(root).or_insert_with(|| {
            sums.push(DVec3::ZERO);
            point_members.push(Vec::new());
            sums.len() - 1
        });
        remap[index] = mapped;
        sums[mapped] += points[index];
        point_members[mapped].push(index);
    }
    let merged_points = sums
        .into_iter()
        .zip(&point_members)
        .map(|(sum, members)| sum / usize_as_f64(members.len()))
        .collect::<Vec<_>>();
    let (merged_polygons, polygon_sources): (Vec<Polygon>, Vec<usize>) = unique_tagged_polygons(
        polygons
            .into_iter()
            .enumerate()
            .filter_map(|(source, (face, material))| {
                clean_polygon(face.into_iter().map(|index| remap[index]).collect())
                    .map(|clean| (source, (clean, material)))
            })
            .collect(),
    )
    .into_iter()
    .map(|(source, polygon)| (polygon, source))
    .unzip();
    let merged_edges = edges
        .into_iter()
        .filter_map(|[first, second]| {
            let pair = [remap[first], remap[second]];
            (pair[0] != pair[1]).then_some(pair)
        })
        .collect();
    MergedGeometry {
        points: merged_points,
        polygons: merged_polygons,
        edges: merged_edges,
        point_members,
        polygon_sources,
    }
}

fn merge_geometry(
    points: &[DVec3],
    polygons: Vec<Polygon>,
    edges: Vec<[usize; 2]>,
    threshold: f64,
) -> Result<MergedGeometry> {
    if points.len() > u32::MAX as usize {
        return Err(mesh_error(&MeshError::IdExhausted));
    }
    let mut parent: Vec<_> = (0..points.len()).collect();
    let mut cells: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (index, point) in points.iter().enumerate() {
        let cell = spatial_cell(*point, threshold);
        for x in -1_i64..=1 {
            for y in -1_i64..=1 {
                for z in -1_i64..=1 {
                    let nearby = [
                        cell[0].saturating_add(x),
                        cell[1].saturating_add(y),
                        cell[2].saturating_add(z),
                    ];
                    if let Some(indices) = cells.get(&nearby) {
                        for other in indices {
                            if point.distance_squared(points[*other]) <= threshold * threshold {
                                union(&mut parent, index, *other);
                            }
                        }
                    }
                }
            }
        }
        cells.entry(cell).or_default().push(index);
    }
    Ok(collapse_clusters(points, parent, polygons, edges))
}

/// The parent of each vertex after Blender's CONNECTED weld (`mesh_merge_by_distance_connected`).
/// Edges in mesh order collapse their two clusters when the cluster positions are within the
/// threshold; a cluster's position is the running mean of its members. The lower root survives.
/// `None` when no edge collapses.
fn connected_clusters(
    mesh: &Mesh,
    points: &[DVec3],
    threshold: f64,
    only_loose_edges: bool,
) -> Result<Option<Vec<usize>>> {
    let edges = if only_loose_edges {
        loose_edges(mesh)?
    } else {
        let indices = id_indices(mesh);
        mesh.edges
            .iter()
            .map(|edge| {
                match (
                    indices.get(&edge.vertices[0]),
                    indices.get(&edge.vertices[1]),
                ) {
                    (Some(&first), Some(&second)) => Ok([first, second]),
                    _ => Err(PotError::new(
                        ErrorCode::InvalidArgument,
                        "modifier input edge references a missing vertex",
                    )),
                }
            })
            .collect::<Result<Vec<_>>>()?
    };
    let mut parent: Vec<usize> = (0..points.len()).collect();
    let mut cluster_points = points.to_vec();
    let mut members = vec![0_usize; points.len()];
    let limit = threshold * threshold;
    let mut collapsed = false;
    for [first, second] in edges {
        let mut first = find(&mut parent, first);
        let mut second = find(&mut parent, second);
        if first == second {
            continue;
        }
        if first > second {
            std::mem::swap(&mut first, &mut second);
        }
        let direction = cluster_points[second] - cluster_points[first];
        if direction.length_squared() > limit {
            continue;
        }
        let influence =
            (members[second] + 1) as f64 / (members[first] + members[second] + 2) as f64;
        cluster_points[first] += direction * influence;
        members[first] += members[second] + 1;
        parent[second] = first;
        collapsed = true;
    }
    Ok(collapsed.then_some(parent))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "spatial cells use saturating float-to-integer conversion"
)]
fn spatial_cell(point: DVec3, threshold: f64) -> [i64; 3] {
    [
        (point.x / threshold).floor() as i64,
        (point.y / threshold).floor() as i64,
        (point.z / threshold).floor() as i64,
    ]
}

fn find(parent: &mut [usize], index: usize) -> usize {
    if parent[index] != index {
        parent[index] = find(parent, parent[index]);
    }
    parent[index]
}

fn union(parent: &mut [usize], first: usize, second: usize) {
    let first_root = find(parent, first);
    let second_root = find(parent, second);
    if first_root != second_root {
        let (root, child) = if first_root < second_root {
            (first_root, second_root)
        } else {
            (second_root, first_root)
        };
        parent[child] = root;
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use std::collections::HashMap;

    use glam::{DMat4, DVec3};
    use proptest::prelude::*;
    use serde_json::json;

    use crate::{
        error::{ErrorCode, PotError},
        geom::{
            BoxParams, Mesh,
            uv_layers::{self, UvLayer},
        },
        model::{CurveData, CurvePoint, CurveSpline, Id, Modifier},
    };

    use super::{check_subdivision_budget, evaluate_modifiers, evaluated_curve_path_length};

    fn modifier(enabled: bool, modifier_type: &str) -> Modifier {
        Modifier {
            id: Id::new("modifier_one").unwrap(),
            modifier_type: modifier_type.to_owned(),
            name: "Modifier".to_owned(),
            enabled,
            show_viewport: true,
            show_render: true,
            params: serde_json::Map::new(),
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        }
    }

    fn with_params(modifier_type: &str, params: serde_json::Value) -> Modifier {
        let mut modifier = modifier(true, modifier_type);
        modifier.params = serde_json::from_value(params).unwrap();
        modifier
    }

    #[test]
    fn disabled_modifiers_preserve_the_source_mesh() {
        let mesh = Mesh::box_mesh(BoxParams {
            size: DVec3::splat(2.0),
        })
        .unwrap();
        let evaluated = evaluate_modifiers(&mesh, &[modifier(false, "unknown")]).unwrap();
        assert_eq!(evaluated, mesh);
    }

    #[test]
    fn unknown_enabled_modifiers_report_the_feature_id() {
        let mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let error = evaluate_modifiers(&mesh, &[modifier(true, "unknown")]).unwrap_err();
        assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(error.details["feature_id"], "modifier.unknown");
    }

    #[test]
    fn array_offsets_instances_and_bounds() {
        let mesh = Mesh::box_mesh(BoxParams {
            size: DVec3::new(2.0, 4.0, 6.0),
        })
        .unwrap();
        let modifier = with_params(
            "array",
            json!({"count": 3, "use_relative_offset": true, "relative_offset_displace": [1.0, 0.0, 0.0], "use_constant_offset": true, "constant_offset_displace": [0.0, 1.0, 0.0]}),
        );
        let result = evaluate_modifiers(&mesh, &[modifier]).unwrap();
        assert_eq!(result.vertices.len(), 24);
        let bounds = result.bounds().unwrap();
        assert_eq!(bounds.min, DVec3::new(-1.0, -2.0, -3.0));
        assert_eq!(bounds.max, DVec3::new(5.0, 4.0, 3.0));
        assert!(result.validate().is_ok());
    }
    #[test]
    fn array_duplicates_and_offsets_uv_layer_values() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        uv_layers::write_layers(
            &mut mesh,
            &[UvLayer::new(
                "UVMap",
                vec![Some(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])],
            )],
        );
        let modifier = with_params(
            "array",
            json!({
                "count": 2,
                "use_relative_offset": false,
                "use_constant_offset": false,
                "offset_u": 0.25,
                "offset_v": -0.5
            }),
        );

        let result = evaluate_modifiers(&mesh, &[modifier]).unwrap();

        let layers = uv_layers::read_layers(&result);
        assert_eq!(layers.len(), 1);
        assert_eq!(
            layers[0].faces,
            vec![
                Some(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
                // Offsets apply in Blender's V-up space, so stored v is 1 - (Blender v + offset_v).
                Some(vec![[0.25, 0.5], [1.25, 0.5], [1.25, 1.5]]),
            ]
        );
    }

    /// Blender's UDIM mirror (values from a Blender 5.2.2 probe): whole-number coordinates map to
    /// themselves, a tile's interior reflects about its centre, and negative values follow `fmod`.
    #[test]
    fn mirror_udim_reflection_matches_blender_on_tile_edges_and_negatives() {
        let mirrored = |uv| super::mirror_uv_pass(uv, [true, true], [0.0, 0.0], [0.0, 0.0], true);
        assert_eq!(mirrored([0.0, 0.0]), [0.0, 0.0]);
        assert_eq!(mirrored([1.0, 1.0]), [1.0, 1.0]);
        assert_eq!(mirrored([2.0, 0.5]), [2.0, 0.5]);
        assert_eq!(mirrored([-0.25, 1.25]), [0.25, 1.75]);
        assert_eq!(mirrored([1.25, 0.25]), [1.75, 0.75]);
    }

    /// A UV warp followed by a mirror carries the warped store UVs into every mirrored face.
    #[test]
    fn uv_warp_then_mirror_rebuilds_every_store_face() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y],
            vec![vec![0, 1, 2, 3]],
        )
        .unwrap();
        uv_layers::write_layers(
            &mut mesh,
            &[UvLayer::new(
                "UVMap",
                vec![Some(vec![[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]])],
            )],
        );
        let warp = with_params(
            "uv_warp",
            json!({"offset": [0.5, 0.0], "uv_layer": "UVMap"}),
        );
        let mirror = with_params(
            "mirror",
            json!({"use_axis": [true, false, false], "use_mirror_merge": false}),
        );
        let out = evaluate_modifiers(&mesh, &[warp, mirror]).unwrap();
        let warped: [[f64; 2]; 4] = [[0.6, 0.2], [0.8, 0.2], [0.8, 0.4], [0.6, 0.4]];
        let is_warped = |uv: &[f64; 2]| {
            warped
                .iter()
                .any(|want| (want[0] - uv[0]).abs() < 1e-9 && (want[1] - uv[1]).abs() < 1e-9)
        };
        let layers = uv_layers::read_layers(&out);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].faces.len(), 2);
        for face in &layers[0].faces {
            let uv = face.as_ref().unwrap();
            assert_eq!(uv.len(), 4);
            assert!(uv.iter().all(is_warped), "stale store UV: {uv:?}");
        }
    }

    #[test]
    fn array_fit_curve_length_uses_curve_object_scale() {
        let curve = CurveData {
            splines: vec![CurveSpline {
                points: vec![
                    CurvePoint {
                        co: [0.0, 0.0, 0.0],
                        ..CurvePoint::default()
                    },
                    CurvePoint {
                        co: [2.0, 0.0, 0.0],
                        ..CurvePoint::default()
                    },
                ],
                ..CurveSpline::default()
            }],
            ..CurveData::default()
        };

        let length =
            evaluated_curve_path_length(&curve, DMat4::from_scale(DVec3::splat(1.5))).unwrap();

        assert!((length - 3.0).abs() <= 1.0e-12);
    }

    #[test]
    fn mirror_merges_coincident_geometry() {
        let mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let result = evaluate_modifiers(
            &mesh,
            &[with_params(
                "mirror",
                json!({"use_axis": [true, false, false], "use_mirror_merge": true}),
            )],
        )
        .unwrap();
        assert_eq!(result.vertices.len(), 8);
        assert_eq!(result.faces.len(), 6);
        assert!(result.validate().is_ok());

        let plane_mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::Y, DVec3::new(0.0, 0.0, 1.0)],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        let mirrored = evaluate_modifiers(
            &plane_mesh,
            &[with_params(
                "mirror",
                json!({"use_axis": [true, false, false], "use_mirror_merge": true}),
            )],
        )
        .unwrap();
        assert_eq!(mirrored.vertices.len(), 3);
        assert_eq!(mirrored.faces.len(), 1);
    }

    /// A box with every face smooth and the given custom normals: `(face index, vector)` pairs, one
    /// vector per corner of that face.
    fn smooth_box_with_custom(faces: &[(usize, DVec3)]) -> Mesh {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        crate::geom::normals::shade_faces_smooth(&mut mesh);
        let vectors: Vec<Option<DVec3>> = mesh
            .faces
            .iter()
            .enumerate()
            .flat_map(|(index, face)| {
                let normal = faces
                    .iter()
                    .find(|(face_index, _)| *face_index == index)
                    .map(|(_, normal)| *normal);
                std::iter::repeat_n(normal, face.vertices.len())
            })
            .collect();
        crate::geom::clnor::encode_vectors(
            &mut mesh,
            Some(&vectors),
            crate::geom::clnor::NormalCodeOptions::default(),
        )
        .unwrap();
        mesh
    }

    /// Vertex positions of `mesh` by vertex ID.
    fn positions_by_id(mesh: &Mesh) -> HashMap<u32, DVec3> {
        mesh.vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co))
            .collect()
    }

    /// Case A: a planar Triangulate changes no shading. The authored normal stays on the two
    /// triangles of its face, the four edges where it disagrees with its neighbours become sharp
    /// (Blender's setter), and no other triangle has a custom normal.
    #[test]
    fn planar_triangulate_keeps_authored_normal_and_marks_split_edges() {
        let authored = DVec3::new(0.195, 0.098, -0.976).normalize();
        let base = Mesh::box_mesh(BoxParams::default()).unwrap();
        let bottom = base
            .faces
            .iter()
            .position(|face| {
                let points = positions_by_id(&base);
                face.vertices.iter().all(|id| points[id].z < -0.999)
            })
            .unwrap();
        let mesh = smooth_box_with_custom(&[(bottom, authored)]);
        let mut settled = mesh.clone();
        crate::geom::clnor::settle_custom_normals(&mut settled).unwrap();
        let input_normals = crate::geom::normals::corner_normals(
            &settled,
            &mesh
                .vertices
                .iter()
                .map(|vertex| vertex.co)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let result = evaluate_modifiers(&mesh, &[with_params("triangulate", json!({}))]).unwrap();
        let points = positions_by_id(&result);
        let sharp = crate::geom::normals::sharp_edge_flags(&result).unwrap();
        let bottom_edges = result
            .edges
            .iter()
            .zip(&sharp)
            .filter(|(edge, _)| edge.vertices.iter().all(|id| points[id].z < -0.999))
            .filter(|(_, flag)| **flag)
            .count();
        assert_eq!(
            bottom_edges, 4,
            "the four bottom edges split the bottom fan"
        );

        let output = crate::geom::normals::corner_normals(
            &result,
            &result
                .vertices
                .iter()
                .map(|vertex| vertex.co)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        // Every output corner keeps the shading of the input corner it was cut from.
        let mut corner = 0;
        for face in &result.faces {
            let corner_points: Vec<DVec3> = face.vertices.iter().map(|id| points[id]).collect();
            let source = mesh
                .faces
                .iter()
                .enumerate()
                .find(|(_, source)| {
                    corner_points.iter().all(|point| {
                        source
                            .vertices
                            .iter()
                            .any(|id| (positions_by_id(&mesh)[id] - *point).length() < 1.0e-9)
                    })
                })
                .map(|(index, _)| index)
                .unwrap();
            for (position, point) in corner_points.iter().enumerate() {
                let source_position = mesh.faces[source]
                    .vertices
                    .iter()
                    .position(|id| (positions_by_id(&mesh)[id] - *point).length() < 1.0e-9)
                    .unwrap();
                let source_corner = mesh.faces[..source]
                    .iter()
                    .map(|face| face.vertices.len())
                    .sum::<usize>()
                    + source_position;
                assert!(
                    (output[corner + position] - input_normals[source_corner]).length() < 1.0e-3,
                    "triangle corner shading changed"
                );
            }
            corner += face.vertices.len();
        }
    }

    /// Case B: Mirror does not average two custom normals across an edge their faces split on. The
    /// two faces disagree, so the shared edge is sharp, and the copy of the first face keeps its own
    /// vector along that edge.
    #[test]
    fn mirror_keeps_custom_normals_on_either_side_of_a_split_edge() {
        let base = Mesh::box_mesh(BoxParams::default()).unwrap();
        let points = positions_by_id(&base);
        let centre = |face: &crate::geom::Face| {
            face.vertices.iter().map(|id| points[id]).sum::<DVec3>() / face.vertices.len() as f64
        };
        let first = (0..base.faces.len())
            .min_by(|a, b| {
                centre(&base.faces[*a])
                    .x
                    .total_cmp(&centre(&base.faces[*b]).x)
            })
            .unwrap();
        let second = (0..base.faces.len())
            .filter(|index| *index != first)
            .find(|index| {
                base.faces[*index]
                    .vertices
                    .iter()
                    .filter(|id| base.faces[first].vertices.contains(id))
                    .count()
                    == 2
            })
            .unwrap();
        let first_normal = DVec3::new(-1.0, 0.3, 0.1).normalize();
        let second_normal = DVec3::new(0.2, -1.0, 0.4).normalize();
        let mesh = smooth_box_with_custom(&[(first, first_normal), (second, second_normal)]);
        let shared: Vec<u32> = base.faces[first]
            .vertices
            .iter()
            .copied()
            .filter(|id| base.faces[second].vertices.contains(id))
            .collect();

        let result = evaluate_modifiers(
            &mesh,
            &[with_params(
                "mirror",
                json!({"use_axis": [true, false, false], "use_mirror_merge": false}),
            )],
        )
        .unwrap();
        let result_points = positions_by_id(&result);
        let sharp = crate::geom::normals::sharp_edge_flags(&result).unwrap();
        let split = result
            .edges
            .iter()
            .zip(&sharp)
            .any(|(edge, flag)| *flag && shared.iter().all(|id| edge.vertices.contains(id)));
        assert!(split, "the edge between the two faces must be sharp");

        let normals = crate::geom::normals::corner_normals(
            &result,
            &result
                .vertices
                .iter()
                .map(|vertex| vertex.co)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut corner = 0;
        let mut checked = 0;
        for face in &result.faces {
            let face_points: Vec<u32> = face.vertices.clone();
            let is_first_copy = base.faces[first]
                .vertices
                .iter()
                .all(|id| face_points.contains(id))
                && face_points.len() == 4;
            if is_first_copy {
                for (position, id) in face_points.iter().enumerate() {
                    if shared.contains(id) {
                        assert!(
                            (normals[corner + position] - first_normal).length() < 1.0e-3,
                            "a corner on the split edge took the other face's normal"
                        );
                        checked += 1;
                    }
                }
            }
            corner += face.vertices.len();
        }
        assert_eq!(
            checked, 2,
            "both corners of the shared edge on the original face are checked"
        );
        let _ = result_points;
    }

    /// A Triangulate with keep on a mesh without custom normals stores its pre-split normals: the
    /// output's custom normals equal the input's corner normals for every corner of a flat face.
    #[test]
    fn triangulate_keep_without_custom_normals_stores_pre_split_normals() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        crate::geom::normals::shade_faces_smooth(&mut mesh);
        let result = evaluate_modifiers(
            &mesh,
            &[with_params(
                "triangulate",
                json!({"keep_custom_normals": true}),
            )],
        )
        .unwrap();
        assert!(result.attributes.contains_key("custom_normal"));
        let without = evaluate_modifiers(
            &mesh,
            &[with_params(
                "triangulate",
                json!({"keep_custom_normals": false}),
            )],
        )
        .unwrap();
        assert!(!without.attributes.contains_key("custom_normal"));
    }

    /// `use_normal_flip` reverses every screw quad, so each face's normal changes sign.
    #[test]
    fn screw_normal_flip_reverses_every_face() {
        // A profile quad in the XZ plane, off the axis: each edge sweeps into faces that do not cross.
        let mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(3.0, 0.0, 0.0),
                DVec3::new(4.0, 0.0, 0.0),
                DVec3::new(4.0, 0.0, 1.0),
                DVec3::new(3.0, 0.0, 1.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .unwrap();
        // Faces keyed by their sorted, rounded corner positions, so the comparison does not depend on
        // the order the faces are emitted in.
        let normals_by_corners = |result: &Mesh| -> HashMap<Vec<[i64; 3]>, DVec3> {
            let points = positions_by_id(result);
            result
                .faces
                .iter()
                .map(|face| {
                    let corners: Vec<DVec3> = face.vertices.iter().map(|id| points[id]).collect();
                    let mut key: Vec<[i64; 3]> = corners
                        .iter()
                        .map(|point| point.to_array().map(|c| (c * 1.0e6).round() as i64))
                        .collect();
                    key.sort_unstable();
                    (key, crate::geom::normals::face_normal(&corners))
                })
                .collect()
        };
        // Three steps close the full turn into three distinct rings.
        let plain = normals_by_corners(
            &evaluate_modifiers(&mesh, &[with_params("screw", json!({"steps": 3}))]).unwrap(),
        );
        let flipped = normals_by_corners(
            &evaluate_modifiers(
                &mesh,
                &[with_params(
                    "screw",
                    json!({"steps": 3, "use_normal_flip": true}),
                )],
            )
            .unwrap(),
        );
        assert_eq!(plain.len(), 12);
        assert_eq!(flipped.len(), 12);
        for (corners, before) in &plain {
            let after = flipped
                .get(corners)
                .unwrap_or_else(|| panic!("flipped screw lacks a face {corners:?}"));
            assert!(
                before.dot(*after) < -0.99,
                "screw face did not flip: {before:?} then {after:?} corners {corners:?}"
            );
        }
    }

    fn refused_feature(error: &PotError) -> Option<&str> {
        error
            .details
            .get("feature_id")
            .and_then(serde_json::Value::as_str)
    }

    /// Each typed refusal is raised where Blender acts, and each stays silent where Blender does
    /// nothing (`mirror bisect` on an unmirrored axis, `mark_seam` without seams, `mark_sharp`
    /// without sharp edges).
    #[test]
    fn shading_refusals_name_their_feature_only_where_blender_acts() {
        let mesh = Mesh::box_mesh(BoxParams::default()).unwrap();

        let error = evaluate_modifiers(
            &mesh,
            &[with_params(
                "mirror",
                json!({"use_axis": [true, false, false], "use_bisect_axis": [true, false, false]}),
            )],
        )
        .unwrap_err();
        assert_eq!(refused_feature(&error), Some("modifier.mirror.bisect"));
        evaluate_modifiers(
            &mesh,
            &[with_params(
                "mirror",
                json!({"use_axis": [true, false, false], "use_bisect_axis": [false, true, false]}),
            )],
        )
        .unwrap();

        let mut seamed = mesh.clone();
        let first_edge = seamed.edges[0].id;
        let mut seam_values = serde_json::Map::new();
        seam_values.insert(format!("e{first_edge}"), json!(true));
        seamed.attributes.insert(
            "seam".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": seam_values}),
        );
        let error = evaluate_modifiers(
            &seamed,
            &[with_params(
                "bevel",
                json!({"width": 0.1, "mark_seam": true}),
            )],
        )
        .unwrap_err();
        assert_eq!(refused_feature(&error), Some("modifier.bevel.mark_seam"));
        evaluate_modifiers(
            &mesh,
            &[with_params(
                "bevel",
                json!({"width": 0.1, "mark_seam": true}),
            )],
        )
        .unwrap();

        let error = evaluate_modifiers(
            &mesh,
            &[with_params(
                "bevel",
                json!({"width": 0.1, "segments": 2, "harden_normals": true}),
            )],
        )
        .unwrap_err();
        assert_eq!(
            refused_feature(&error),
            Some("modifier.bevel.shading_segments")
        );
        evaluate_modifiers(
            &mesh,
            &[with_params(
                "bevel",
                json!({"width": 0.1, "segments": 2, "mark_sharp": true}),
            )],
        )
        .unwrap();

        let mut sharp = mesh.clone();
        let key = format!("e{}", sharp.edges[0].id);
        sharp.attributes.insert(
            "sharp".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": {key: true}}),
        );
        let error = evaluate_modifiers(
            &sharp,
            &[with_params(
                "bevel",
                json!({"width": 0.1, "segments": 2, "mark_sharp": true}),
            )],
        )
        .unwrap_err();
        assert_eq!(
            refused_feature(&error),
            Some("modifier.bevel.shading_segments")
        );
    }

    #[test]
    fn bevel_creates_inset_faces_and_valid_chamfer_topology() {
        let mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let result =
            evaluate_modifiers(&mesh, &[with_params("bevel", json!({"width": 0.1}))]).unwrap();
        assert_eq!(result.faces.len(), 26);
        assert_eq!(result.vertices.len(), 24);
        assert!(result.validate().is_ok());
    }

    #[test]
    fn subdivision_resource_budget_preserves_fixture_and_rejects_excess() {
        assert!(check_subdivision_budget(24, 8, 12, 2, 0).is_ok());
        let triangle_error = check_subdivision_budget(3_072, 8, 12, 8, 0).unwrap_err();
        assert_eq!(triangle_error.code, ErrorCode::LimitExceeded);
        let memory_error = check_subdivision_budget(10_000_000, 8, 10_000_000, 1, 0).unwrap_err();
        assert_eq!(memory_error.code, ErrorCode::LimitExceeded);
    }

    #[test]
    fn subdivision_budget_counts_uv_layers() {
        // A 50x50 quad grid at level 5: 10_000 corners, about 2.82e9 bytes without UVs and 2.66e9 per
        // UV layer. Two layers still fit the 8 GiB budget; four do not.
        assert!(check_subdivision_budget(10_000, 2_601, 5_100, 5, 0).is_ok());
        assert!(check_subdivision_budget(10_000, 2_601, 5_100, 5, 2).is_ok());
        let error = check_subdivision_budget(10_000, 2_601, 5_100, 5, 4).unwrap_err();
        assert_eq!(error.code, ErrorCode::LimitExceeded);
    }

    #[test]
    fn subdivision_rejects_four_uv_layers_over_budget_before_allocating() {
        let mut mesh = Mesh::grid(crate::geom::GridParams {
            size_x: 2.0,
            size_y: 2.0,
            x_subdivisions: 51,
            y_subdivisions: 51,
        })
        .unwrap();
        let corners: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
        let layers: Vec<UvLayer> = ["UV0", "UV1", "UV2", "UV3"]
            .into_iter()
            .map(|name| {
                let mut values =
                    (0..corners).map(|index| [(index % 7) as f64 / 7.0, (index % 5) as f64 / 5.0]);
                let faces: Vec<Option<Vec<[f64; 2]>>> = mesh
                    .faces
                    .iter()
                    .map(|face| Some(values.by_ref().take(face.vertices.len()).collect()))
                    .collect();
                UvLayer::new(name, faces)
            })
            .collect();
        uv_layers::write_layers(&mut mesh, &layers);
        let error = evaluate_modifiers(&mesh, &[with_params("subdivision", json!({"levels": 5}))])
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::LimitExceeded);
    }

    #[test]
    fn subdivision_refines_topology_with_catmull_clark_quads() {
        let mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let result =
            evaluate_modifiers(&mesh, &[with_params("subdivision", json!({"levels": 1}))]).unwrap();
        assert_eq!(result.vertices.len(), 26);
        assert_eq!(result.faces.len(), 24);
        assert!(result.faces.iter().all(|face| face.vertices.len() == 4));
        assert!(result.validate().is_ok());
    }

    #[test]
    fn modifier_stack_applies_transforms_in_declared_order() {
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        let array = || {
            with_params(
                "array",
                json!({
                    "count":2,
                    "use_relative_offset":false,
                    "use_constant_offset":true,
                    "constant_offset_displace":[2.0,0.0,0.0]
                }),
            )
        };
        let mirror = || {
            with_params(
                "mirror",
                json!({"use_axis":[true,false,false],"use_mirror_merge":false}),
            )
        };
        let array_then_mirror = evaluate_modifiers(&mesh, &[array(), mirror()]).unwrap();
        let mirror_then_array = evaluate_modifiers(&mesh, &[mirror(), array()]).unwrap();

        assert_eq!(array_then_mirror.vertices.len(), 12);
        assert_eq!(array_then_mirror.faces.len(), 4);
        let ordered_bounds = array_then_mirror.bounds().unwrap();
        assert!((ordered_bounds.min.x + 3.0).abs() < 1.0e-12);
        assert!((ordered_bounds.max.x - 3.0).abs() < 1.0e-12);

        let reversed_bounds = mirror_then_array.bounds().unwrap();
        assert!((reversed_bounds.min.x + 1.0).abs() < 1.0e-12);
        assert!((reversed_bounds.max.x - 3.0).abs() < 1.0e-12);
    }

    proptest! {
        #[test]
        fn mirror_without_merging_doubles_vertices(size in 0.25_f64..10.0) {
            let mesh = Mesh::box_mesh(BoxParams { size: DVec3::splat(size) }).unwrap();
            let result = evaluate_modifiers(
                &mesh,
                &[with_params("mirror", json!({"use_axis": [true, false, false], "use_mirror_merge": false}))],
            ).unwrap();
            prop_assert_eq!(result.vertices.len(), mesh.vertices.len() * 2);
            prop_assert!(result.validate().is_ok());
        }

        #[test]
        fn mirror_merge_coalesces_vertices_on_the_axis(
            y in 0.01_f64..100.0,
            z in 0.01_f64..100.0,
        ) {
            let mesh = Mesh::from_positions_and_faces(
                vec![DVec3::ZERO, DVec3::new(0.0, y, 0.0), DVec3::new(0.0, 0.0, z)],
                vec![vec![0, 1, 2]],
            ).unwrap();
            let result = evaluate_modifiers(
                &mesh,
                &[with_params("mirror", json!({"use_axis": [true, false, false], "use_mirror_merge": true, "merge_threshold": 0.0}))],
            ).unwrap();
            prop_assert_eq!(result.vertices.len(), mesh.vertices.len());
            prop_assert_eq!(result.faces.len(), mesh.faces.len());
            prop_assert!(result.validate().is_ok());
        }

        #[test]
        fn array_bounds_follow_count_and_component_offset(
            size in 0.25_f64..10.0,
            count in 1_u32..=8,
            relative_x in -2.0_f64..2.0,
            constant_x in -5.0_f64..5.0,
        ) {
            let mesh = Mesh::box_mesh(BoxParams { size: DVec3::splat(size) }).unwrap();
            let result = evaluate_modifiers(
                &mesh,
                &[with_params(
                    "array",
                    json!({
                        "count": count,
                        "use_relative_offset": true,
                        "relative_offset_displace": [relative_x, 0.0, 0.0],
                        "use_constant_offset": true,
                        "constant_offset_displace": [constant_x, 0.0, 0.0],
                    }),
                )],
            ).unwrap();
            let travel = (size * relative_x + constant_x) * f64::from(count - 1);
            let expected_min_x = -size * 0.5 + travel.min(0.0);
            let expected_max_x = size * 0.5 + travel.max(0.0);
            let bounds = result.bounds().unwrap();
            prop_assert_eq!(result.vertices.len(), mesh.vertices.len() * usize::try_from(count).unwrap());
            prop_assert!((bounds.min.x - expected_min_x).abs() < 1.0e-9);
            prop_assert!((bounds.max.x - expected_max_x).abs() < 1.0e-9);
            prop_assert!((bounds.min.y + size * 0.5).abs() < 1.0e-12);
            prop_assert!((bounds.max.y - size * 0.5).abs() < 1.0e-12);
            prop_assert!((bounds.min.z + size * 0.5).abs() < 1.0e-12);
            prop_assert!((bounds.max.z - size * 0.5).abs() < 1.0e-12);
        }
    }
}
