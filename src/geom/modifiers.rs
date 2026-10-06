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
    graph::{self, NodeGroup},
    model::{Id, Modifier, Registry},
    params::{self, ParameterFamily},
};

use super::{Mesh, MeshError, edge_key};

mod advanced;
pub mod attributes;
mod decimate;
pub(crate) mod deform;
mod ocean;
mod remesh_modes;
pub mod simulation;
type Polygon = (Vec<usize>, u32);
type BevelEdgeFaces = BTreeMap<(usize, usize), Vec<(usize, usize, u32)>>;
type BevelEdgeFacesWithMaterial = BTreeMap<(usize, usize), Vec<(usize, usize, usize, u32)>>;
type MergedGeometry = (Vec<DVec3>, Vec<Polygon>, Vec<[usize; 2]>);
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
        .filter(|(_, modifier)| modifier.enabled)
    {
        validate_boolean_topology_suffix(modifiers, modifier_index)?;
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
            let overrides = match modifier.params.get("inputs") {
                None => Map::new(),
                Some(Value::Object(overrides)) => overrides.clone(),
                Some(_) => {
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
            "subdivision" | "multires" => subdivision(&evaluated, modifier)?,
            "solidify" => solidify(&evaluated, modifier)?,
            "triangulate" => triangulate(&evaluated)?,
            "bevel" => bevel(&evaluated, modifier)?,
            "decimate" => decimate(&evaluated, modifier)?,
            "weld" => weld(&evaluated, modifier)?,
            "displace" => displace(&evaluated, modifier, owner_world_matrix)?,
            "smooth" => smooth(&evaluated, modifier)?,
            "simple_deform" => advanced::simple_deform(&evaluated, modifier)?,
            "screw" => advanced::screw(&evaluated, modifier)?,
            "shrinkwrap" => advanced::shrinkwrap(&evaluated, modifier)?,
            "curve" => advanced::curve(&evaluated, modifier)?,
            "hook" => advanced::hook(&evaluated, modifier)?,
            "skin" => advanced::skin(&evaluated, modifier)?,
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
            "ocean" => ocean::ocean(&evaluated, modifier, frame)?,
            _ => return Err(unsupported_modifier(modifier)),
        };
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
) -> Result<()> {
    let Some(boolean_modifier) = modifiers.get(modifier_index) else {
        return Ok(());
    };
    if !boolean_modifier.enabled || boolean_modifier.modifier_type != "boolean" {
        return Ok(());
    }
    let Some(downstream) = modifiers
        .iter()
        .skip(modifier_index + 1)
        .find(|candidate| candidate.enabled && modifier_depends_on_topology(candidate))
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

fn boolean_modifier(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let operand = modifier
        .runtime
        .operand_mesh
        .clone()
        .ok_or_else(|| invalid_parameter(modifier, "object", "a mesh object operand"))?;
    let operand: Mesh = serde_json::from_value(operand)
        .map_err(|_| invalid_parameter(modifier, "object", "a mesh object operand"))?;
    let operation = match modifier.params.get("operation") {
        None => "DIFFERENCE",
        Some(value) => value.as_str().ok_or_else(|| {
            invalid_parameter(modifier, "operation", "UNION, DIFFERENCE, or INTERSECT")
        })?,
    };
    super::boolean::boolean_mesh(mesh, &operand, operation)
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

fn string_param<'a>(modifier: &'a Modifier, name: &str, default: &'a str) -> Result<&'a str> {
    string_param_with_expected(modifier, name, default, "a string")
}

fn string_param_with_expected<'a>(
    modifier: &'a Modifier,
    name: &str,
    default: &'a str,
    expected: &str,
) -> Result<&'a str> {
    modifier.params.get(name).map_or(Ok(default), |value| {
        value
            .as_str()
            .ok_or_else(|| invalid_parameter(modifier, name, expected))
    })
}

pub(super) fn number_param(modifier: &Modifier, name: &str, default: f64) -> Result<f64> {
    match parameter_value(modifier, name) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| invalid_parameter(modifier, name, "a finite number")),
    }
}

pub(super) fn bool_param(modifier: &Modifier, name: &str, default: bool) -> Result<bool> {
    match parameter_value(modifier, name) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| invalid_parameter(modifier, name, "a boolean")),
    }
}

fn uint_param(modifier: &Modifier, name: &str, default: u64) -> Result<u64> {
    match parameter_value(modifier, name) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .ok_or_else(|| invalid_parameter(modifier, name, "a non-negative integer")),
    }
}

fn vec3_param(modifier: &Modifier, name: &str, default: DVec3) -> Result<DVec3> {
    let Some(value) = parameter_value(modifier, name) else {
        return Ok(default);
    };
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

fn loose_edges(mesh: &Mesh) -> Result<Vec<[usize; 2]>> {
    let indices = id_indices(mesh);
    let mut face_edges = HashSet::new();
    for face in &mesh.faces {
        for (first, second) in cyclic_pairs(&face.vertices) {
            face_edges.insert(edge_key(*first, *second));
        }
    }
    mesh.edges
        .iter()
        .filter(|edge| !face_edges.contains(&edge_key(edge.vertices[0], edge.vertices[1])))
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

fn build_mesh(
    points: Vec<DVec3>,
    polygons: Vec<Polygon>,
    loose_edges: Vec<[usize; 2]>,
    attributes: &serde_json::Map<String, Value>,
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
    result.attributes.clone_from(attributes);
    validated(&result)?;
    Ok(result)
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
    let selected: Vec<_> = enabled
        .iter()
        .enumerate()
        .filter_map(|(i, value)| value.then_some(i))
        .collect();
    if selected.is_empty() {
        return Ok(mesh.clone());
    }
    let merge = bool_param(modifier, "use_mirror_merge", true)?;
    let threshold = number_param(modifier, "merge_threshold", 0.001)?;
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
            if reverse {
                mapped.reverse();
            }
            output_polygons.push((mapped, *material_index));
        }
        output_loose_edges.extend(
            source_loose_edges
                .iter()
                .map(|[a, b]| [base + *a, base + *b]),
        );
    }
    let (points, output_polygons, output_loose_edges) = if merge {
        merge_geometry(&points, output_polygons, output_loose_edges, threshold)?
    } else {
        (points, output_polygons, output_loose_edges)
    };
    build_mesh(
        points,
        output_polygons,
        output_loose_edges,
        &mesh.attributes,
    )
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
    let mut count = uint_param(modifier, "count", 2)?;
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
    let use_relative = bool_param(modifier, "use_relative_offset", true)?;
    let use_constant = bool_param(modifier, "use_constant_offset", false)?;
    let relative = vec3_param(modifier, "relative_offset_displace", DVec3::X)?;
    let constant = vec3_param(modifier, "constant_offset_displace", DVec3::ZERO)?;
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
    if bool_param(modifier, "use_object_offset", false)?
        && let Some(value) = modifier
            .params
            .get("offset_object")
            .filter(|value| !value.is_null())
    {
        let target = value
            .as_str()
            .ok_or_else(|| invalid_parameter(modifier, "offset_object", "an object ID or null"))?;
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
    let fit_type = match modifier.params.get("fit_type") {
        None => "FIXED_COUNT",
        Some(value) => value
            .as_str()
            .ok_or_else(|| invalid_parameter(modifier, "fit_type", "a Blender array fit type"))?,
    };
    if !matches!(fit_type, "FIXED_COUNT" | "FIT_LENGTH" | "FIT_CURVE") {
        return Err(invalid_parameter(
            modifier,
            "fit_type",
            "FIXED_COUNT, FIT_LENGTH, or FIT_CURVE",
        ));
    }
    let fit_length = match fit_type {
        "FIT_LENGTH" => number_param(modifier, "fit_length", 0.0)?,
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
    let use_merge = bool_param(modifier, "use_merge_vertices", false)?;
    let merge_final = bool_param(modifier, "use_merge_vertices_cap", false)?;
    let merge_threshold = number_param(modifier, "merge_threshold", 0.01)?;
    if merge_threshold < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "merge_threshold",
            "a non-negative finite number",
        ));
    }
    let offset_u = number_param(modifier, "offset_u", 0.0)?;
    let offset_v = number_param(modifier, "offset_v", 0.0)?;
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
    let (points, output_polygons, output_loose_edges) = if use_merge {
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
        (points, output_polygons, output_loose_edges)
    };
    let source_loop_count = source_polygons.iter().map(|(face, _)| face.len()).sum();
    let mut attributes = mesh.attributes.clone();
    array_uv_attributes(
        &mut attributes,
        source_loop_count,
        count_usize,
        offset_u,
        offset_v,
    )?;
    build_mesh(points, output_polygons, output_loose_edges, &attributes)
}

fn array_cap_mesh<'a>(
    modifier: &Modifier,
    name: &str,
    object_meshes: &'a BTreeMap<Id, &'a Mesh>,
    evaluated_object_meshes: &'a BTreeMap<Id, Mesh>,
) -> Result<Option<&'a Mesh>> {
    let Some(value) = modifier.params.get(name).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let target = value
        .as_str()
        .ok_or_else(|| invalid_parameter(modifier, name, "a mesh object ID or null"))?;
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
    let Some(value) = modifier
        .params
        .get("curve")
        .filter(|value| !value.is_null())
    else {
        return Ok(0.0);
    };
    let target = value
        .as_str()
        .ok_or_else(|| invalid_parameter(modifier, "curve", "a curve object ID or null"))?;
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

fn array_uv_attributes(
    attributes: &mut Map<String, Value>,
    loop_count: usize,
    copy_count: usize,
    offset_u: f64,
    offset_v: f64,
) -> Result<()> {
    let Some(layers) = attributes
        .get_mut("blender_uv_layers")
        .and_then(Value::as_array_mut)
    else {
        return Ok(());
    };
    let capacity = loop_count
        .checked_mul(copy_count)
        .ok_or_else(|| mesh_error(&MeshError::IdExhausted))?;
    let mut uv_names = HashSet::new();
    for layer in layers {
        let Some(name) = layer.get("name").and_then(Value::as_str) else {
            continue;
        };
        uv_names.insert(name.to_owned());
        let Some(source_values) = layer.get("values").and_then(Value::as_array) else {
            continue;
        };
        let mut values = Vec::with_capacity(capacity);
        for copy in 0..copy_count {
            let u_offset = offset_u * usize_as_f64(copy);
            let v_offset = offset_v * usize_as_f64(copy);
            for value in source_values.iter().take(loop_count) {
                values.push(offset_uv_value(value, u_offset, v_offset));
            }
        }
        layer["values"] = Value::Array(values);
    }
    if let Some(attributes) = attributes
        .get_mut("blender_attributes")
        .and_then(Value::as_array_mut)
    {
        for attribute in attributes {
            let name = attribute.get("name").and_then(Value::as_str);
            if !name.is_some_and(|name| uv_names.contains(name))
                || attribute.get("domain").and_then(Value::as_str) != Some("CORNER")
                || attribute.get("data_type").and_then(Value::as_str) != Some("FLOAT2")
            {
                continue;
            }
            let Some(source_values) = attribute.get("values").and_then(Value::as_array) else {
                continue;
            };
            let mut values = Vec::with_capacity(capacity);
            for copy in 0..copy_count {
                let u_offset = offset_u * usize_as_f64(copy);
                let v_offset = offset_v * usize_as_f64(copy);
                for value in source_values.iter().take(loop_count) {
                    values.push(offset_uv_value(value, u_offset, v_offset));
                }
            }
            attribute["values"] = Value::Array(values);
        }
    }
    Ok(())
}

fn offset_uv_value(value: &Value, offset_u: f64, offset_v: f64) -> Value {
    let Some(coordinates) = value
        .as_array()
        .filter(|coordinates| coordinates.len() == 2)
    else {
        return value.clone();
    };
    let (Some(u), Some(v)) = (coordinates[0].as_f64(), coordinates[1].as_f64()) else {
        return value.clone();
    };
    json!([u + offset_u, v + offset_v])
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
    let mut root_to_index = HashMap::new();
    let mut sums = Vec::<DVec3>::new();
    let mut counts = Vec::<usize>::new();
    let mut remap = vec![0; points.len()];
    for index in 0..points.len() {
        let root = find(&mut parent, index);
        let mapped = *root_to_index.entry(root).or_insert_with(|| {
            sums.push(DVec3::ZERO);
            counts.push(0);
            sums.len() - 1
        });
        remap[index] = mapped;
        sums[mapped] += points[index];
        counts[mapped] += 1;
    }
    let merged_points = sums
        .into_iter()
        .zip(counts)
        .map(|(sum, count)| sum / usize_as_f64(count))
        .collect();
    let merged_polygons = unique_polygons(
        polygons
            .into_iter()
            .filter_map(|(face, material)| {
                clean_polygon(face.into_iter().map(|index| remap[index]).collect())
                    .map(|clean| (clean, material))
            })
            .collect(),
    );
    let merged_edges = edges
        .into_iter()
        .filter_map(|[first, second]| {
            let pair = [remap[first], remap[second]];
            (pair[0] != pair[1]).then_some(pair)
        })
        .collect();
    Ok((merged_points, merged_polygons, merged_edges))
}

#[derive(Clone, Copy)]
struct SubdivisionOptions {
    preserve_corners: bool,
    use_creases: bool,
}

fn subdivision(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    let level_name = if modifier.runtime.use_render_levels {
        "render_levels"
    } else {
        "levels"
    };
    let levels = uint_param(modifier, level_name, 1)?;
    if levels > 5 {
        return Err(invalid_parameter(
            modifier,
            level_name,
            "an integer between 0 and 5",
        ));
    }
    let boundary_smooth = match modifier.params.get("boundary_smooth") {
        None => "ALL",
        Some(value) => value.as_str().ok_or_else(|| {
            invalid_parameter(modifier, "boundary_smooth", "ALL or PRESERVE_CORNERS")
        })?,
    };
    if !matches!(boundary_smooth, "ALL" | "PRESERVE_CORNERS") {
        return Err(invalid_parameter(
            modifier,
            "boundary_smooth",
            "ALL or PRESERVE_CORNERS",
        ));
    }
    let uv_smooth = match modifier.params.get("uv_smooth") {
        None => "PRESERVE_BOUNDARIES",
        Some(value) => value.as_str().ok_or_else(|| {
            invalid_parameter(modifier, "uv_smooth", "a Blender UV smoothing mode")
        })?,
    };
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
    let quality = uint_param(modifier, "quality", 3)?;
    if !(1..=10).contains(&quality) {
        return Err(invalid_parameter(
            modifier,
            "quality",
            "an integer between 1 and 10",
        ));
    }
    let options = SubdivisionOptions {
        preserve_corners: boundary_smooth == "PRESERVE_CORNERS",
        use_creases: bool_param(modifier, "use_creases", true)?,
    };
    let use_limit_surface = levels > 0 && bool_param(modifier, "use_limit_surface", true)?;
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
        check_subdivision_budget(
            first_level_faces,
            mesh.vertices.len(),
            mesh.edges.len(),
            levels,
        )?;
    }
    let mut result = mesh.clone();
    for _ in 0..levels {
        result = subdivide_once(&result, options)?;
    }
    if use_limit_surface {
        let output_vertex_count = result.vertices.len();
        let limit_positions = limit_surface_positions(&result, options, quality)?;
        let (face_counts, boundary_neighbors) = boundary_stencils(&result)?;
        let control_points = positions(mesh);
        for index in 0..output_vertex_count {
            let limit_position = limit_positions[index];
            let is_boundary_corner =
                face_counts[index] == 1 && boundary_neighbors[index].len() == 2;
            result.vertices[index].co = if is_boundary_corner && options.preserve_corners {
                control_points.get(index).copied().unwrap_or(limit_position)
            } else {
                limit_position
            };
        }
    }
    Ok(result)
}

fn check_subdivision_budget(
    first_level_faces: usize,
    input_vertices: usize,
    input_edges: usize,
    levels: u64,
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
    // Include simultaneous input/output mesh storage and subdivision adjacency indexes.
    let estimated_bytes = [
        (estimated_vertices, 64_u128),
        (estimated_edges, 128_u128),
        (face_count, 128_u128),
        (estimated_corners, 32_u128),
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

fn limit_surface_positions(
    mesh: &Mesh,
    options: SubdivisionOptions,
    quality: u64,
) -> Result<Vec<DVec3>> {
    let source_points = positions(mesh);
    let source_polygons = polygons(mesh)?;
    let mut edge_faces = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (face_index, (face, _)) in source_polygons.iter().enumerate() {
        for (first, second) in cyclic_pairs(face) {
            edge_faces
                .entry(edge_key(*first, *second))
                .or_default()
                .push(face_index);
        }
    }
    for [first, second] in loose_edges(mesh)? {
        edge_faces.entry(edge_key(first, second)).or_default();
    }
    let edge_creases = edge_crease_weights(mesh, options.use_creases)?;
    let mut vertex_faces = vec![Vec::new(); source_points.len()];
    let mut vertex_edges = vec![Vec::new(); source_points.len()];
    for (face_index, (face, _)) in source_polygons.iter().enumerate() {
        for index in face {
            vertex_faces[*index].push(face_index);
        }
    }
    for edge in edge_faces.keys() {
        vertex_edges[edge.0].push(*edge);
        vertex_edges[edge.1].push(*edge);
    }

    let mut limit_points = Vec::with_capacity(source_points.len());
    for (index, point) in source_points.iter().copied().enumerate() {
        let faces = &vertex_faces[index];
        let edges = &vertex_edges[index];
        if faces.is_empty() || edges.is_empty() {
            limit_points.push(point);
            continue;
        }

        let mut boundary_neighbors = [0_usize; 2];
        let mut boundary_count = 0_usize;
        let mut crease_neighbors = [(0_usize, 0.0_f64); 3];
        let mut crease_count = 0_usize;
        for edge in edges {
            let neighbor = if edge.0 == index { edge.1 } else { edge.0 };
            if edge_faces
                .get(edge)
                .is_some_and(|incident| incident.len() == 1)
            {
                if boundary_count < boundary_neighbors.len() {
                    boundary_neighbors[boundary_count] = neighbor;
                }
                boundary_count += 1;
            }
            if let Some(sharpness) = edge_creases.get(edge)
                && *sharpness > 0.0
            {
                let mut candidate = (neighbor, *sharpness);
                for slot in &mut crease_neighbors {
                    if candidate.1 > slot.1 {
                        std::mem::swap(&mut candidate, slot);
                    }
                }
                crease_count = crease_count.saturating_add(1).min(crease_neighbors.len());
            }
        }

        let boundary_point = if boundary_count == 2 {
            let point = (point * 4.0
                + source_points[boundary_neighbors[0]]
                + source_points[boundary_neighbors[1]])
                / 6.0;
            Some(if options.preserve_corners && faces.len() == 1 {
                source_points[index]
            } else if faces.len() == 1 {
                // OpenSubdiv's adaptive evaluation refines extraordinary boundary
                // corners according to quality before approximating their limit point.
                let refinement_weight = (1..quality).fold(1.0, |weight, _| weight * 0.25);
                point.lerp(source_points[index], refinement_weight)
            } else {
                point
            })
        } else {
            None
        };
        let limit_point = if let Some(boundary_point) = boundary_point {
            boundary_point
        } else {
            let valence = usize_as_f64(edges.len());
            let mut edge_neighbor_sum = DVec3::ZERO;
            for edge in edges {
                let neighbor = if edge.0 == index { edge.1 } else { edge.0 };
                edge_neighbor_sum += source_points[neighbor];
            }
            let mut opposite_point_sum = DVec3::ZERO;
            for face_index in faces {
                let face = &source_polygons[*face_index].0;
                if face.len() != 4 {
                    return Err(PotError::new(
                        ErrorCode::EvaluationFailed,
                        "limit-surface evaluation requires subdivided quad faces",
                    ));
                }
                let corner = face
                    .iter()
                    .position(|vertex| *vertex == index)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "subdivision vertex-face adjacency is inconsistent",
                        )
                    })?;
                opposite_point_sum += source_points[face[(corner + 2) % 4]];
            }
            let smooth_limit =
                (point * (valence * valence) + edge_neighbor_sum * 4.0 + opposite_point_sum)
                    / (valence * (valence + 5.0));
            if crease_count >= 3 && crease_neighbors[2].1 >= 1.0 {
                point
            } else if crease_count >= 2 {
                let sharpness = crease_neighbors[0].1.min(crease_neighbors[1].1).min(1.0);
                let crease_limit = (point * 4.0
                    + source_points[crease_neighbors[0].0]
                    + source_points[crease_neighbors[1].0])
                    / 6.0;
                smooth_limit.lerp(crease_limit, sharpness)
            } else {
                smooth_limit
            }
        };
        limit_points.push(limit_point);
    }
    Ok(limit_points)
}

fn boundary_stencils(mesh: &Mesh) -> Result<(Vec<usize>, Vec<Vec<usize>>)> {
    let polygons = polygons(mesh)?;
    let mut vertex_face_count = vec![0_usize; mesh.vertices.len()];
    let mut edge_face_count = BTreeMap::<(usize, usize), usize>::new();
    for (face, _) in polygons {
        for vertex in &face {
            vertex_face_count[*vertex] += 1;
        }
        for (first, second) in cyclic_pairs(&face) {
            *edge_face_count
                .entry(edge_key(*first, *second))
                .or_default() += 1;
        }
    }
    let mut boundary_neighbors = vec![Vec::new(); mesh.vertices.len()];
    for (edge, face_count) in edge_face_count {
        if face_count == 1 {
            boundary_neighbors[edge.0].push(edge.1);
            boundary_neighbors[edge.1].push(edge.0);
        }
    }
    Ok((vertex_face_count, boundary_neighbors))
}

fn edge_crease_weights(mesh: &Mesh, enabled: bool) -> Result<BTreeMap<(usize, usize), f64>> {
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
fn subdivide_once(mesh: &Mesh, options: SubdivisionOptions) -> Result<Mesh> {
    let source_points = positions(mesh);
    let source_polygons = polygons(mesh)?;
    let source_edges = loose_edges(mesh)?;
    let mut edge_faces: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (face_index, (face, _)) in source_polygons.iter().enumerate() {
        for (first, second) in cyclic_pairs(face) {
            edge_faces
                .entry(edge_key(*first, *second))
                .or_default()
                .push(face_index);
        }
    }
    for [first, second] in &source_edges {
        edge_faces.entry(edge_key(*first, *second)).or_default();
    }
    let edge_creases = edge_crease_weights(mesh, options.use_creases)?;
    let mut face_points = Vec::with_capacity(source_polygons.len());
    for (face, _) in &source_polygons {
        face_points.push(
            face.iter()
                .fold(DVec3::ZERO, |sum, index| sum + source_points[*index])
                / usize_as_f64(face.len()),
        );
    }
    let mut vertex_faces = vec![Vec::new(); source_points.len()];
    let mut vertex_edges = vec![Vec::new(); source_points.len()];
    for (face_index, (face, _)) in source_polygons.iter().enumerate() {
        for index in face {
            vertex_faces[*index].push(face_index);
        }
    }
    for edge in edge_faces.keys() {
        vertex_edges[edge.0].push(*edge);
        vertex_edges[edge.1].push(*edge);
    }
    let mut updated_points =
        Vec::with_capacity(source_points.len() + edge_faces.len() + source_polygons.len());
    for (index, point) in source_points.iter().enumerate() {
        let boundary_neighbors: Vec<_> = vertex_edges[index]
            .iter()
            .filter(|edge| edge_faces.get(edge).is_some_and(|faces| faces.len() == 1))
            .map(|edge| if edge.0 == index { edge.1 } else { edge.0 })
            .collect();
        let smooth = if boundary_neighbors.len() == 2 {
            if options.preserve_corners && vertex_faces[index].len() == 1 {
                *point
            } else {
                *point * 0.75
                    + (source_points[boundary_neighbors[0]] + source_points[boundary_neighbors[1]])
                        * 0.125
            }
        } else if !vertex_edges[index].is_empty() && !vertex_faces[index].is_empty() {
            let face_average = vertex_faces[index]
                .iter()
                .fold(DVec3::ZERO, |sum, face| sum + face_points[*face])
                / usize_as_f64(vertex_faces[index].len());
            let edge_midpoint_average =
                vertex_edges[index].iter().fold(DVec3::ZERO, |sum, edge| {
                    sum + (source_points[edge.0] + source_points[edge.1]) * 0.5
                }) / usize_as_f64(vertex_edges[index].len());
            let n = usize_as_f64(vertex_edges[index].len());
            (face_average + edge_midpoint_average * 2.0 + *point * (n - 3.0)) / n
        } else {
            *point
        };
        let mut crease_neighbors = vertex_edges[index]
            .iter()
            .filter_map(|edge| {
                edge_creases
                    .get(edge)
                    .map(|sharpness| (if edge.0 == index { edge.1 } else { edge.0 }, *sharpness))
            })
            .collect::<Vec<_>>();
        crease_neighbors.sort_by(|first, second| second.1.total_cmp(&first.1));
        let updated = if crease_neighbors.len() >= 2 {
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
    for (edge, faces) in &edge_faces {
        let smooth = if faces.len() == 2 {
            (source_points[edge.0]
                + source_points[edge.1]
                + face_points[faces[0]]
                + face_points[faces[1]])
                * 0.25
        } else if faces.is_empty() || faces.len() == 1 {
            (source_points[edge.0] + source_points[edge.1]) * 0.5
        } else {
            faces.iter().fold(
                source_points[edge.0] + source_points[edge.1],
                |sum, face| sum + face_points[*face],
            ) / usize_as_f64(faces.len() + 2)
        };
        let point = edge_creases.get(edge).map_or(smooth, |sharpness| {
            smooth.lerp(
                (source_points[edge.0] + source_points[edge.1]) * 0.5,
                sharpness.min(1.0),
            )
        });
        edge_point_indices.insert(*edge, updated_points.len());
        updated_points.push(point);
    }
    let face_point_base = updated_points.len();
    updated_points.extend(face_points);
    let mut output_polygons =
        Vec::with_capacity(source_polygons.iter().map(|(face, _)| face.len()).sum());
    for (face_index, (face, material)) in source_polygons.iter().enumerate() {
        for corner in 0..face.len() {
            let previous = face[(corner + face.len() - 1) % face.len()];
            let current = face[corner];
            let next = face[(corner + 1) % face.len()];
            let edge_previous = edge_point_indices[&edge_key(previous, current)];
            let edge_next = edge_point_indices[&edge_key(current, next)];
            output_polygons.push((
                vec![
                    current,
                    edge_next,
                    face_point_base + face_index,
                    edge_previous,
                ],
                *material,
            ));
        }
    }
    let mut output_loose_edges = Vec::with_capacity(source_edges.len() * 2);
    for [first, second] in source_edges {
        let midpoint = edge_point_indices[&edge_key(first, second)];
        output_loose_edges.push([first, midpoint]);
        output_loose_edges.push([midpoint, second]);
    }
    let attributes = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .map(|_| {
            let mut attributes = mesh.attributes.clone();
            attributes.insert(
                "vertex_groups".to_owned(),
                subdivided_vertex_group_attributes(mesh, &source_polygons, &edge_faces),
            );
            attributes
        });
    let mut output = build_mesh(
        updated_points,
        output_polygons,
        output_loose_edges,
        attributes.as_ref().unwrap_or(&mesh.attributes),
    )?;
    if options.use_creases && mesh.attributes.contains_key("crease_edge") {
        let mut next_creases = BTreeMap::new();
        for (edge, sharpness) in &edge_creases {
            let remaining = sharpness - 1.0;
            if remaining <= 0.0 {
                continue;
            }
            let midpoint = edge_point_indices[edge];
            let first_id = output.vertices[edge.0].id;
            let second_id = output.vertices[edge.1].id;
            let midpoint_id = output.vertices[midpoint].id;
            let ordered = |first: u32, second: u32| {
                if first <= second {
                    (first, second)
                } else {
                    (second, first)
                }
            };
            next_creases.insert(ordered(first_id, midpoint_id), remaining);
            next_creases.insert(ordered(midpoint_id, second_id), remaining);
        }
        let mut values = Map::new();
        for edge in &output.edges {
            let ordered = if edge.vertices[0] <= edge.vertices[1] {
                (edge.vertices[0], edge.vertices[1])
            } else {
                (edge.vertices[1], edge.vertices[0])
            };
            if let Some(weight) = next_creases.get(&ordered) {
                values.insert(format!("e{}", edge.id), json!(weight));
            }
        }
        output.attributes.insert(
            "crease_edge".to_owned(),
            json!({"domain":"edge","type":"float","values":values}),
        );
    }
    Ok(output)
}

fn subdivided_vertex_group_attributes(
    mesh: &Mesh,
    polygons: &[Polygon],
    edges: &BTreeMap<(usize, usize), Vec<usize>>,
) -> Value {
    let source_groups = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut output_groups = Map::new();
    for (name, group) in source_groups {
        let Some(group_weights) = group.as_object() else {
            continue;
        };
        let weights = mesh
            .vertices
            .iter()
            .map(|vertex| {
                group_weights
                    .get(&format!("v{}", vertex.id))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
            })
            .collect::<Vec<_>>();
        let mut output_weights = weights.clone();
        for edge in edges.keys() {
            output_weights.push(f64::midpoint(weights[edge.0], weights[edge.1]));
        }
        for (face, _) in polygons {
            let average =
                face.iter().map(|index| weights[*index]).sum::<f64>() / usize_as_f64(face.len());
            output_weights.push(average);
        }
        let values = output_weights
            .into_iter()
            .enumerate()
            .map(|(index, weight)| (format!("v{index}"), json!(weight)))
            .collect::<Map<_, _>>();
        output_groups.insert(name, Value::Object(values));
    }
    Value::Object(output_groups)
}

fn triangulate(mesh: &Mesh) -> Result<Mesh> {
    validated(mesh)?;
    let triangles = mesh.triangulate().map_err(|error| mesh_error(&error))?;
    let indices = id_indices(mesh);
    let mut output_polygons = Vec::with_capacity(triangles.len());
    let mut triangle_index = 0;
    for face in &mesh.faces {
        let triangle_count = face.vertices.len() - 2;
        for triangle in &triangles[triangle_index..triangle_index + triangle_count] {
            let mut polygon = Vec::with_capacity(3);
            for id in triangle {
                polygon.push(*indices.get(id).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InvalidArgument,
                        "triangulation returned an unknown vertex",
                    )
                })?);
            }
            output_polygons.push((polygon, face.material_index));
        }
        triangle_index += triangle_count;
    }
    build_mesh(
        positions(mesh),
        output_polygons,
        loose_edges(mesh)?,
        &mesh.attributes,
    )
}

#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
fn solidify(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    let thickness = number_param(modifier, "thickness", 0.01)?;
    if thickness < 0.0 {
        return Err(invalid_parameter(
            modifier,
            "thickness",
            "a non-negative finite number",
        ));
    }
    let offset = number_param(modifier, "offset", -1.0)?;
    if !(-1.0..=1.0).contains(&offset) {
        return Err(invalid_parameter(
            modifier,
            "offset",
            "a number from -1 through 1",
        ));
    }
    let use_even_offset = bool_param(modifier, "use_even_offset", false)?;
    let use_quality_normals = bool_param(modifier, "use_quality_normals", false)?;
    let use_rim = bool_param(modifier, "use_rim", true)?;
    let use_rim_only = bool_param(modifier, "use_rim_only", false)?;
    let flip = bool_param(modifier, "use_flip_normals", false)?;
    if mesh.faces.is_empty() {
        return Ok(mesh.clone());
    }

    let source_points = positions(mesh);
    let source_polygons = polygons(mesh)?;
    let source_loose_edges = loose_edges(mesh)?;
    let mut normals = vec![DVec3::ZERO; source_points.len()];
    let mut even_scales = vec![0.0; source_points.len()];
    let mut angle_totals = vec![0.0; source_points.len()];
    let mut edge_faces: BevelEdgeFaces = BTreeMap::new();
    for (face, material) in &source_polygons {
        let mut face_normal = DVec3::ZERO;
        for (first, second) in cyclic_pairs(face) {
            face_normal += source_points[*first].cross(source_points[*second]);
            edge_faces
                .entry(edge_key(*first, *second))
                .or_default()
                .push((*first, *second, *material));
        }
        let face_area = face_normal.length();
        face_normal = face_normal.normalize_or_zero();
        if face_normal == DVec3::ZERO {
            continue;
        }
        for (index, vertex) in face.iter().enumerate() {
            let current = source_points[*vertex];
            let previous = source_points[face[(index + face.len() - 1) % face.len()]];
            let next = source_points[face[(index + 1) % face.len()]];
            let incoming = (previous - current).normalize_or_zero();
            let outgoing = (next - current).normalize_or_zero();
            let angle = incoming
                .dot(outgoing)
                .clamp(-1.0, 1.0)
                .acos()
                .max(f64::EPSILON);
            let weight = if use_quality_normals {
                angle
            } else {
                face_area
            };
            normals[*vertex] += face_normal * weight;
        }
    }
    for normal in &mut normals {
        *normal = normal.normalize_or_zero();
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
                }
            }
        }
    } else {
        for (index, point) in source_points.iter().enumerate() {
            duplicate_indices[index] = Some(source_points.len() + index);
            points.push(*point + normals[index] * (shell_offset * even_scale(index)));
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
    for (face, material) in &source_polygons {
        let mut outer = face.clone();
        if flip {
            outer.reverse();
        }
        output_polygons.push((outer, *material));
        if !rim_only {
            let mut shell: Vec<_> = face
                .iter()
                .map(|index| duplicate_indices[*index].unwrap_or(*index))
                .collect();
            shell.reverse();
            if flip {
                shell.reverse();
            }
            output_polygons.push((shell, *material));
        }
    }
    if use_rim {
        for faces in edge_faces.values().filter(|faces| faces.len() == 1) {
            let (first, second, material) = faces[0];
            let duplicate_first = duplicate_indices[first].unwrap_or(first);
            let duplicate_second = duplicate_indices[second].unwrap_or(second);
            let mut rim = vec![first, second, duplicate_second, duplicate_first];
            if flip {
                rim.reverse();
            }
            output_polygons.push((rim, material));
        }
    }
    let mut output_loose_edges = Vec::with_capacity(source_loose_edges.len().saturating_mul(2));
    output_loose_edges.extend(source_loose_edges.iter().copied());
    if !rim_only {
        output_loose_edges.extend(source_loose_edges.iter().filter_map(|[first, second]| {
            Some([duplicate_indices[*first]?, duplicate_indices[*second]?])
        }));
    }
    build_mesh(
        points,
        output_polygons,
        output_loose_edges,
        &mesh.attributes,
    )
}

fn mesh_attribute_values<'a>(mesh: &'a Mesh, name: &str) -> Option<&'a [Value]> {
    if let Some(values) = mesh.attributes.get(name).and_then(Value::as_array) {
        return Some(values);
    }
    mesh.attributes
        .get("blender_attributes")?
        .as_array()?
        .iter()
        .find(|attribute| attribute.get("name").and_then(Value::as_str) == Some(name))?
        .get("values")?
        .as_array()
        .map(Vec::as_slice)
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
    let width = number_param(modifier, "width", 0.1)?;
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

    let segments = usize::try_from(uint_param(modifier, "segments", 1)?)
        .ok()
        .filter(|value| (1..=64).contains(value))
        .ok_or_else(|| invalid_parameter(modifier, "segments", "an integer from 1 through 64"))?;
    let limit_method = modifier
        .params
        .get("limit_method")
        .and_then(Value::as_str)
        .unwrap_or("NONE")
        .to_ascii_uppercase();
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
    let affect = modifier
        .params
        .get("affect")
        .and_then(Value::as_str)
        .unwrap_or("EDGES")
        .to_ascii_uppercase();
    if !matches!(affect.as_str(), "EDGES" | "VERTICES") {
        return Err(invalid_parameter(modifier, "affect", "EDGES or VERTICES"));
    }
    let profile = number_param(modifier, "profile", 0.5)?;
    if !(0.0..=1.0).contains(&profile) {
        return Err(invalid_parameter(
            modifier,
            "profile",
            "a number from 0 through 1",
        ));
    }
    let clamp_overlap = bool_param(modifier, "use_clamp_overlap", true)?;
    let loop_slide = bool_param(modifier, "loop_slide", true)?;
    let offset_type = modifier
        .params
        .get("offset_type")
        .and_then(Value::as_str)
        .unwrap_or("OFFSET")
        .to_ascii_uppercase();
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
    let angle_limit = number_param(modifier, "angle_limit", std::f64::consts::FRAC_PI_6)?;
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
            face_corner.insert((face_index, vertex), index);
        }
    }

    let mut output_polygons = Vec::new();
    for (face_index, (face, material)) in source_polygons.iter().enumerate() {
        output_polygons.push((
            face.iter()
                .map(|vertex| face_corner[&(face_index, *vertex)])
                .collect(),
            *material,
        ));
    }
    let mut profile_points: HashMap<((usize, usize), usize, usize), usize> = HashMap::new();
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
                    profile_points.insert((*edge, vertex, segment), index);
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
            for segment in 0..segments {
                let (start, end) = (side_a.1, side_a.2);
                let material = side_a.3;
                output_polygons.push((
                    vec![
                        point_at(start, 0, segment),
                        point_at(end, 0, segment),
                        point_at(end, 0, segment + 1),
                        point_at(start, 0, segment + 1),
                    ],
                    material,
                ));
            }
        }
    }

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
            let displacement = amount * (1.0 - 1.0 / 3.0_f64.sqrt());
            corner_center -= normal_sum * displacement;
        } else {
            let corners = faces
                .iter()
                .map(|(face_index, _, _)| points[face_corner[&(*face_index, vertex)]])
                .fold(DVec3::ZERO, |sum, point| sum + point);
            corner_center = corners / faces.len() as f64;
        }
        if affect == "VERTICES" || segments == 1 {
            let boundary: Vec<_> = faces
                .iter()
                .map(|(face_index, _, _)| face_corner[&(*face_index, vertex)])
                .collect();
            if boundary.len() >= 3 {
                output_polygons.push((boundary, faces[0].2));
            }
            continue;
        }
        let center_index = points.len();
        points.push(corner_center);
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
            patch.push(center_index);
            if affect == "EDGES" && selected_edges.contains(&incoming_edge) {
                patch.extend(
                    (1..segments)
                        .rev()
                        .map(|segment| profile_points[&(incoming_edge, vertex, segment)]),
                );
            }
            output_polygons.push((patch, *material));
        }
    }
    if output_polygons.is_empty() {
        return Ok(mesh.clone());
    }
    build_mesh(
        points,
        output_polygons,
        loose_edges(mesh)?,
        &mesh.attributes,
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
    let exponent = if profile <= 0.0 {
        1.0e4
    } else {
        (-2.0_f64.ln() / profile.sqrt().ln()).clamp(0.0, 1.0e4)
    };
    let mut points = source_points.to_vec();
    let mut edge_points = HashMap::new();
    let mut patch_graph = vec![HashMap::<usize, Vec<usize>>::new(); source_points.len()];
    let mut corner_sectors = vec![Vec::new(); source_points.len()];
    let mut output_polygons = Vec::with_capacity(source_polygons.len());
    for (face, material) in source_polygons {
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
                    let index = points.len();
                    points.push(source_points[vertex].lerp(
                        source_points[neighbor],
                        amount / edge_length.max(f64::EPSILON),
                    ));
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
            corner_sectors[vertex].push((incoming, arc_points, outgoing, *material));
        }
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
            let material = corner_sectors[vertex].first().map_or(0, |sector| sector.3);
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
        for (incoming, arc_points, outgoing, material) in &corner_sectors[vertex] {
            if segments == 2 {
                output_polygons.push((
                    vec![*incoming, arc_points[0], *outgoing, center_index],
                    *material,
                ));
            } else {
                let mut boundary = Vec::with_capacity(arc_points.len() + 2);
                boundary.push(*incoming);
                boundary.extend(arc_points.iter().copied());
                boundary.push(*outgoing);
                for pair in boundary.windows(2) {
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
    let mut remap = vec![0; points.len()];
    let mut compact_points = Vec::with_capacity(points.len());
    for (index, point) in points.into_iter().enumerate() {
        if used[index] {
            remap[index] = compact_points.len();
            compact_points.push(point);
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
    build_mesh(
        compact_points,
        output_polygons,
        compact_loose,
        &mesh.attributes,
    )
}

fn decimate(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    if mesh.faces.is_empty() {
        return Ok(mesh.clone());
    }
    let mode = parameter_value(modifier, "decimate_type")
        .and_then(Value::as_str)
        .unwrap_or("COLLAPSE")
        .to_ascii_uppercase();
    match mode.as_str() {
        "COLLAPSE" => {
            let ratio = number_param(modifier, "ratio", 1.0)?;
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
            let iterations = usize::try_from(uint_param(modifier, "iterations", 2)?)
                .ok()
                .filter(|value| *value <= 64)
                .ok_or_else(|| {
                    invalid_parameter(modifier, "iterations", "an integer from 0 through 64")
                })?;
            decimate::decimate_unsubdivide(mesh, iterations)
        }
        "DISSOLVE" => {
            let angle = number_param(modifier, "angle_limit", 0.0)?;
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

fn decimate_merge_faces(
    first: &[usize],
    second: &[usize],
    edge: (usize, usize),
) -> Option<Vec<usize>> {
    let orientation =
        [(edge.0, edge.1), (edge.1, edge.0)]
            .into_iter()
            .find_map(|(start, end)| {
                let start_a = first.iter().position(|vertex| *vertex == start)?;
                let end_a = (start_a + 1) % first.len();
                if first[end_a] != end {
                    return None;
                }
                let start_b = second.iter().position(|vertex| *vertex == start)?;
                let end_b = second.iter().position(|vertex| *vertex == end)?;
                ((end_b + 1) % second.len() == start_b)
                    .then_some((start, start_a, end_a, start_b, end_b))
            })?;
    let (_, start_a, end_a, start_b, end_b) = orientation;
    let mut merged = Vec::with_capacity(first.len() + second.len() - 2);
    let mut index = end_a;
    loop {
        merged.push(first[index]);
        if index == start_a {
            break;
        }
        index = (index + 1) % first.len();
    }
    index = (start_b + 1) % second.len();
    while index != end_b {
        merged.push(second[index]);
        index = (index + 1) % second.len();
    }
    clean_polygon(merged)
}

fn decimate_compact(
    points: Vec<DVec3>,
    mut polygons: Vec<Polygon>,
    mut loose: Vec<[usize; 2]>,
    attributes: &serde_json::Map<String, Value>,
) -> Result<Mesh> {
    let mut used = vec![false; points.len()];
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
    for (index, point) in points.into_iter().enumerate() {
        if used[index] {
            remap[index] = compact_points.len();
            compact_points.push(point);
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
    build_mesh(compact_points, polygons, loose, attributes)
}

fn weld(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    validated(mesh)?;
    let threshold_name = "merge_threshold";
    let threshold = number_param(modifier, threshold_name, 0.001)?;
    if threshold < 0.0 {
        return Err(invalid_parameter(
            modifier,
            threshold_name,
            "a non-negative finite number",
        ));
    }
    if mesh.vertices.len() < 2 {
        return Ok(mesh.clone());
    }
    let (points, output_polygons, output_edges) = merge_geometry(
        &positions(mesh),
        polygons(mesh)?,
        loose_edges(mesh)?,
        threshold,
    )?;
    build_mesh(points, output_polygons, output_edges, &mesh.attributes)
}

fn displace(mesh: &Mesh, modifier: &Modifier, owner_world_matrix: DMat4) -> Result<Mesh> {
    validated(mesh)?;
    let strength = number_param(modifier, "strength", 1.0)?;
    let mid_level = number_param(modifier, "mid_level", 0.5)?;
    let direction = match modifier.params.get("direction") {
        None => "NORMAL",
        Some(value) => value.as_str().ok_or_else(|| {
            invalid_parameter(
                modifier,
                "direction",
                "NORMAL, X, Y, Z, RGB_TO_XYZ, or CUSTOM_NORMAL",
            )
        })?,
    };
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
    let space = match modifier.params.get("space") {
        None => "LOCAL",
        Some(value) => value
            .as_str()
            .ok_or_else(|| invalid_parameter(modifier, "space", "LOCAL or GLOBAL"))?,
    };
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

    let group_name = match modifier.params.get("vertex_group") {
        None => None,
        Some(value) => {
            let name = value.as_str().ok_or_else(|| {
                invalid_parameter(modifier, "vertex_group", "a vertex-group name")
            })?;
            (!name.is_empty()).then_some(name)
        }
    };
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
    let invert_group = bool_param(modifier, "invert_vertex_group", false)?;
    let points = positions(mesh);
    let indices = id_indices(mesh);
    let mut normals = vec![DVec3::ZERO; points.len()];
    for face in &mesh.faces {
        let mut normal = DVec3::ZERO;
        for (first, second) in cyclic_pairs(&face.vertices) {
            normal += points[indices[first]].cross(points[indices[second]]);
        }
        for id in &face.vertices {
            normals[indices[id]] += normal;
        }
    }
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
            || normals[index].normalize_or_zero(),
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
    let factor = number_param(modifier, "factor", 0.5)?;
    if !(0.0..=1.0).contains(&factor) {
        return Err(invalid_parameter(
            modifier,
            "factor",
            "a number from 0 through 1",
        ));
    }
    let iterations = uint_param(modifier, "iterations", 1)?;
    if iterations > 100 {
        return Err(invalid_parameter(
            modifier,
            "iterations",
            "an integer from 0 through 100",
        ));
    }
    let axes = [
        bool_param(modifier, "use_x", true)?,
        bool_param(modifier, "use_y", true)?,
        bool_param(modifier, "use_z", true)?,
    ];
    if factor == 0.0 || iterations == 0 || !axes.into_iter().any(|enabled| enabled) {
        return Ok(mesh.clone());
    }
    let group_name = match modifier.params.get("vertex_group") {
        None => None,
        Some(value) => {
            let name = value.as_str().ok_or_else(|| {
                invalid_parameter(modifier, "vertex_group", "a vertex-group name")
            })?;
            (!name.is_empty()).then_some(name)
        }
    };
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
    let invert_group = bool_param(modifier, "invert_vertex_group", false)?;
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

fn unique_polygons(polygons: Vec<Polygon>) -> Vec<Polygon> {
    let mut seen = HashSet::new();
    polygons
        .into_iter()
        .filter(|(face, _)| {
            let mut key = face.clone();
            key.sort_unstable();
            seen.insert(key)
        })
        .collect()
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
    let mut root_to_index = HashMap::new();
    let mut sums = Vec::<DVec3>::new();
    let mut counts = Vec::<usize>::new();
    let mut remap = vec![0; points.len()];
    for index in 0..points.len() {
        let root = find(&mut parent, index);
        let mapped = *root_to_index.entry(root).or_insert_with(|| {
            sums.push(DVec3::ZERO);
            counts.push(0);
            sums.len() - 1
        });
        remap[index] = mapped;
        sums[mapped] += points[index];
        counts[mapped] += 1;
    }
    let merged_points = sums
        .into_iter()
        .zip(counts)
        .map(|(sum, count)| sum / usize_as_f64(count))
        .collect::<Vec<_>>();
    let merged_polygons = unique_polygons(
        polygons
            .into_iter()
            .filter_map(|(face, material)| {
                clean_polygon(face.into_iter().map(|index| remap[index]).collect())
                    .map(|clean| (clean, material))
            })
            .collect(),
    );
    let merged_edges = edges
        .into_iter()
        .filter_map(|[first, second]| {
            let pair = [remap[first], remap[second]];
            (pair[0] != pair[1]).then_some(pair)
        })
        .collect();
    Ok((merged_points, merged_polygons, merged_edges))
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

    use glam::{DMat4, DVec3};
    use proptest::prelude::*;
    use serde_json::json;

    use crate::{
        error::ErrorCode,
        geom::{BoxParams, Mesh},
        model::{CurveData, CurvePoint, CurveSpline, Id, Modifier},
    };

    use super::{check_subdivision_budget, evaluate_modifiers, evaluated_curve_path_length};

    fn modifier(enabled: bool, modifier_type: &str) -> Modifier {
        Modifier {
            id: Id::new("modifier_one").unwrap(),
            modifier_type: modifier_type.to_owned(),
            name: "Modifier".to_owned(),
            enabled,
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
        mesh.attributes.insert(
            "blender_uv_layers".to_owned(),
            json!([{
                "name": "UVMap",
                "values": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
            }]),
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

        assert_eq!(
            result.attributes["blender_uv_layers"][0]["values"],
            json!([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.25, -0.5],
                [1.25, -0.5],
                [1.25, 0.5]
            ])
        );
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
        assert!(check_subdivision_budget(24, 8, 12, 2).is_ok());
        let triangle_error = check_subdivision_budget(3_072, 8, 12, 8).unwrap_err();
        assert_eq!(triangle_error.code, ErrorCode::LimitExceeded);
        let memory_error = check_subdivision_budget(10_000_000, 8, 10_000_000, 1).unwrap_err();
        assert_eq!(memory_error.code, ErrorCode::LimitExceeded);
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
