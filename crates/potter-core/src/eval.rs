use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use glam::{DMat4, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use serde_json::{Map, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Aabb, CylinderParams, IcosphereParams, Mesh},
    hash,
    model::{ConstraintType, DataBlock, Id, Modifier, Node, SceneDoc},
};

pub mod animation;
mod ik_solver;
pub mod rig;
const EVALUATION_ENGINE: &str = "potter-eval-4";

/// Scene context used to produce one immutable evaluated snapshot.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationContext {
    #[serde(default)]
    pub scene_id: Option<Id>,
    #[serde(default)]
    pub view_layer: Option<Id>,
    #[serde(default)]
    pub frame: Option<f64>,
}

/// Evaluated node state for the selected scene and frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatedNode {
    /// World transform in column-major order.
    pub world_matrix: [f64; 16],
    /// Evaluated world-space axis-aligned bounds, or null for geometry-free nodes.
    pub bounds: Option<Aabb>,
    /// World-space bounds size, or null for geometry-free nodes.
    pub dimensions: Option<[f64; 3]>,
}

/// Evaluated world-space Grease Pencil stroke used by vector exporters and preview.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedStroke {
    pub points_world: Vec<[f64; 3]>,
    pub radius: Vec<f64>,
    pub color: [f64; 4],
    pub cyclic: bool,
}

/// Stable evaluation of one scene document and its context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub scene_id: Id,
    pub view_layer: Option<Id>,
    pub frame: f64,
    pub revision: u64,
    pub scene_hash: String,
    pub evaluation_hash: String,
    #[serde(default)]
    pub warnings: Vec<PotError>,
    pub nodes: BTreeMap<Id, EvaluatedNode>,
    /// Local-space geometry after the node's enabled modifier stack.
    #[serde(skip)]
    pub meshes: BTreeMap<Id, Mesh>,
    /// Evaluated armature bone-head matrices in world space.
    #[serde(skip)]
    pub bone_matrices: BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
    /// Instance source paths recorded by geometry-node modifiers.
    #[serde(skip)]
    pub instance_paths: BTreeMap<Id, Vec<Vec<Id>>>,
    /// World-space Grease Pencil strokes held at the selected frame.
    #[serde(skip)]
    pub strokes: BTreeMap<Id, Vec<EvaluatedStroke>>,
    /// Content address of the deterministic physics cache result, when enabled.
    #[serde(skip)]
    pub simulation_cache_key: Option<String>,
    /// World-space final velocities for simulated rigid bodies.
    #[serde(skip)]
    pub simulated_velocities: BTreeMap<Id, [f64; 3]>,
    /// World-space particle states evaluated for the selected frame.
    #[serde(skip)]
    pub simulated_particles: BTreeMap<Id, Vec<crate::sim::ParticleState>>,
    /// World-space SPH liquid samples evaluated for the selected frame.
    #[serde(skip)]
    pub fluid_particles: BTreeMap<Id, Vec<[f64; 3]>>,
    /// Dynamic-paint colors evaluated for each canvas vertex.
    #[serde(skip)]
    pub paint_colors: BTreeMap<Id, Vec<[f64; 4]>>,
    /// Dynamic-paint weights evaluated for each canvas vertex.
    #[serde(skip)]
    pub paint_weights: BTreeMap<Id, Vec<f64>>,
    /// VDB-aware volume payloads resolved for this evaluation; omitted from snapshots on disk.
    #[serde(skip)]
    pub volume_data: BTreeMap<Id, crate::geom::volume::VolumeData>,
}

impl Snapshot {
    /// Evaluate world matrices, post-modifier meshes and bounds, and a reproducible key.
    ///
    /// # Errors
    ///
    /// Returns scene, evaluation, modifier, context, and serialization errors.
    pub fn evaluate(doc: &SceneDoc, context: &EvaluationContext) -> Result<Self> {
        Self::evaluate_with_cache(doc, context, None)
    }

    /// Evaluate the Scene and read/write deterministic simulation cache entries.
    pub fn evaluate_with_cache(
        doc: &SceneDoc,
        context: &EvaluationContext,
        cache_directory: Option<&Path>,
    ) -> Result<Self> {
        Self::evaluate_internal(doc, context, cache_directory, None)
    }

    /// Evaluate only the geometry required for the selected nodes and their mesh operands.
    pub fn evaluate_nodes_with_cache(
        doc: &SceneDoc,
        context: &EvaluationContext,
        cache_directory: Option<&Path>,
        node_ids: &BTreeSet<Id>,
    ) -> Result<Self> {
        for node_id in node_ids {
            if !doc.nodes.contains_key(node_id) {
                return Err(PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "selected evaluation node was not found",
                    json!({"node_id":node_id}),
                ));
            }
        }
        Self::evaluate_internal(doc, context, cache_directory, Some(node_ids))
    }

    /// Evaluate every supported node, recording modifier-feature failures per owner.
    pub fn evaluate_available_with_cache(
        doc: &SceneDoc,
        context: &EvaluationContext,
        cache_directory: Option<&Path>,
    ) -> Result<(Self, BTreeMap<Id, PotError>)> {
        let mut evaluation_doc = None;
        let mut node_errors = BTreeMap::new();
        loop {
            let evaluation = if let Some(evaluation_doc) = &evaluation_doc {
                Self::evaluate_with_cache(evaluation_doc, context, cache_directory)
            } else {
                Self::evaluate_with_cache(doc, context, cache_directory)
            };
            match evaluation {
                Ok(snapshot) if node_errors.is_empty() => {
                    return Ok((snapshot, node_errors));
                }
                Ok(mut snapshot) => {
                    let scene_value = serde_json::to_value(doc).map_err(PotError::internal_json)?;
                    let scene_hash = hash::sha256(&hash::canonicalize(&scene_value)?);
                    snapshot.scene_hash.clone_from(&scene_hash);
                    let evaluation_key = json!({
                        "engine": EVALUATION_ENGINE,
                        "simulation_engine": crate::sim::SOLVER_VERSION,
                        "scene_hash": scene_hash,
                        "scene_id": snapshot.scene_id,
                        "view_layer": snapshot.view_layer,
                        "frame": snapshot.frame,
                        "simulation_cache_key": snapshot.simulation_cache_key,
                    });
                    snapshot.evaluation_hash = hash::sha256(&hash::canonicalize(&evaluation_key)?);
                    for node_id in node_errors.keys() {
                        snapshot.meshes.remove(node_id);
                        snapshot.instance_paths.remove(node_id);
                        snapshot.strokes.remove(node_id);
                    }
                    return Ok((snapshot, node_errors));
                }
                Err(error) if error.code == ErrorCode::UnsupportedFeature => {
                    let source_doc = evaluation_doc.as_ref().unwrap_or(doc);
                    let Some(node_id) = unsupported_feature_owner(source_doc, &error) else {
                        return Err(error);
                    };
                    if node_errors.contains_key(&node_id) {
                        return Err(error);
                    }
                    let evaluation_doc = evaluation_doc.get_or_insert_with(|| doc.clone());
                    {
                        let node = evaluation_doc.nodes.get_mut(&node_id).ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "unsupported feature owner disappeared during evaluation",
                            )
                        })?;
                        if node.modifiers.iter().all(|modifier| !modifier.enabled) {
                            return Err(error);
                        }
                        for modifier in &mut node.modifiers {
                            modifier.enabled = false;
                        }
                    }
                    let mut details = error.details.clone();
                    if let Some(details) = details.as_object_mut() {
                        details
                            .entry("node_id".to_owned())
                            .or_insert_with(|| json!(node_id));
                    }
                    node_errors.insert(
                        node_id.clone(),
                        PotError::with_details(error.code, error.message.clone(), details),
                    );
                    let dependencies = object_mesh_dependencies(doc)?;
                    let mut affected = vec![node_id];
                    while let Some(source_id) = affected.pop() {
                        for (dependent_id, targets) in &dependencies {
                            if dependent_id == &source_id
                                || !targets.contains(&source_id)
                                || node_errors.contains_key(dependent_id)
                            {
                                continue;
                            }
                            let Some(dependent) = evaluation_doc.nodes.get_mut(dependent_id) else {
                                continue;
                            };
                            for modifier in &mut dependent.modifiers {
                                modifier.enabled = false;
                            }
                            let mut details = error.details.clone();
                            if let Some(details) = details.as_object_mut() {
                                details.insert("node_id".to_owned(), json!(dependent_id));
                                details.insert("dependency_node_id".to_owned(), json!(source_id));
                            }
                            node_errors.insert(
                                dependent_id.clone(),
                                PotError::with_details(
                                    error.code,
                                    format!(
                                        "object depends on unevaluated geometry `{source_id}`: {}",
                                        error.message
                                    ),
                                    details,
                                ),
                            );
                            affected.push(dependent_id.clone());
                        }
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn evaluate_internal(
        doc: &SceneDoc,
        context: &EvaluationContext,
        cache_directory: Option<&Path>,
        selected_nodes: Option<&BTreeSet<Id>>,
    ) -> Result<Self> {
        doc.validate()?;
        crate::library::validate_linked_assets(doc, cache_directory)?;
        let scene_id = context
            .scene_id
            .clone()
            .unwrap_or_else(|| doc.active_scene.clone());
        let scene = doc.scenes.get(&scene_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "scene context not found",
                json!({ "scene_id": scene_id }),
            )
        })?;
        let view_layer = match &context.view_layer {
            Some(view_layer) if scene.view_layers.contains_key(view_layer) => {
                Some(view_layer.clone())
            }
            Some(view_layer) => {
                return Err(PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "view layer context not found",
                    json!({ "view_layer": view_layer }),
                ));
            }
            None => scene.view_layers.keys().next().cloned(),
        };
        let frame = context.frame.unwrap_or(scene.frame_current);
        if !frame.is_finite() {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "frame must be finite",
            ));
        }
        let mesh_evaluation_order = object_mesh_evaluation_order(doc, selected_nodes)?;
        let evaluated_node_ids: BTreeSet<Id> = mesh_evaluation_order.iter().cloned().collect();

        let scene_value = serde_json::to_value(doc).map_err(PotError::internal_json)?;
        let scene_hash = hash::sha256(&hash::canonicalize(&scene_value)?);
        let driver_values = rig::evaluate_drivers(doc, frame)?;
        let mut matrices = rig::evaluate_world_matrices_with_resources(
            doc,
            frame,
            &driver_values,
            cache_directory,
            scene.fps,
            scene.fps_base,
        )?;
        let object_meshes: BTreeMap<_, _> = doc
            .nodes
            .iter()
            .filter_map(|(node_id, node)| {
                let data_id = node.data.as_ref()?;
                let data_block = doc.data_blocks.get(data_id)?;
                (data_block.data_type == "mesh")
                    .then(|| data_block.mesh.as_ref().map(|mesh| (node_id.clone(), mesh)))
                    .flatten()
            })
            .collect();
        let mut curve_path_lengths = BTreeMap::new();
        for owner_id in &mesh_evaluation_order {
            let Some(owner) = doc.nodes.get(owner_id) else {
                continue;
            };
            for modifier in owner.modifiers.iter().filter(|modifier| {
                modifier.enabled
                    && modifier.modifier_type == "array"
                    && modifier
                        .params
                        .get("fit_type")
                        .and_then(serde_json::Value::as_str)
                        == Some("FIT_CURVE")
            }) {
                let Some(curve_reference) = modifier
                    .params
                    .get("curve")
                    .and_then(serde_json::Value::as_str)
                else {
                    continue;
                };
                let Ok(curve_id) = Id::new(curve_reference) else {
                    continue;
                };
                if curve_path_lengths.contains_key(&curve_id) {
                    continue;
                }
                let Some(curve_node) = doc.nodes.get(&curve_id) else {
                    continue;
                };
                let Some(curve_data_id) = curve_node.data.as_ref() else {
                    continue;
                };
                let Some(curve_data) = doc
                    .data_blocks
                    .get(curve_data_id)
                    .filter(|data_block| data_block.data_type == "curve")
                    .and_then(|data_block| data_block.curve.as_ref())
                else {
                    continue;
                };
                let Some(curve_world_matrix) = matrices.get(&curve_id).copied() else {
                    continue;
                };
                let length = crate::geom::modifiers::evaluated_curve_path_length(
                    curve_data,
                    curve_world_matrix,
                )?;
                curve_path_lengths.insert(curve_id, length);
            }
        }
        let simulation = crate::sim::evaluate(doc, &scene_id, frame, &matrices, cache_directory)?;
        let mut simulated_velocities = BTreeMap::new();
        let mut deformed_vertices = BTreeMap::new();
        let mut fluid_surfaces = BTreeMap::new();
        let mut simulated_particles = BTreeMap::new();
        let mut fluid_particles = BTreeMap::new();
        let mut paint_colors = BTreeMap::new();
        let mut paint_weights = BTreeMap::new();
        let simulation_cache_key = simulation.as_ref().map(|result| result.cache_key.clone());
        if let Some(simulation) = simulation {
            simulated_velocities = simulation.linear_velocities;
            deformed_vertices = simulation.deformed_vertices;
            fluid_surfaces = simulation.fluid_surfaces;
            fluid_particles = simulation.fluid_particles;
            paint_colors = simulation.paint_colors;
            paint_weights = simulation.paint_weights;
            for (id, states) in simulation.particle_states {
                simulated_particles.insert(id, states);
            }
            for (id, positions) in &simulation.particle_positions {
                if simulated_particles.contains_key(id) {
                    continue;
                }
                let velocities = simulation.particle_velocities.get(id).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "simulation particle velocities are missing",
                    )
                })?;
                let birth_frames = simulation.particle_birth_frames.get(id).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "simulation particle birth frames are missing",
                    )
                })?;
                if positions.len() != velocities.len() || positions.len() != birth_frames.len() {
                    return Err(PotError::new(
                        ErrorCode::EvaluationFailed,
                        "simulation particle arrays have inconsistent lengths",
                    ));
                }
                let states = positions
                    .iter()
                    .zip(velocities)
                    .zip(birth_frames)
                    .map(
                        |((position, velocity), birth_frame)| crate::sim::ParticleState {
                            position: *position,
                            velocity: *velocity,
                            birth_frame: *birth_frame,
                            ..crate::sim::ParticleState::default()
                        },
                    )
                    .collect();
                simulated_particles.insert(id.clone(), states);
            }
            for (id, matrix) in simulation.world_matrices {
                matrices.insert(id, DMat4::from_cols_array(&matrix));
            }
        }
        let mut bone_matrices = BTreeMap::new();
        for (id, node) in &doc.nodes {
            if node.kind == "armature" && evaluated_node_ids.contains(id) {
                let armature_world = matrices.get(id).copied().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "evaluated armature world matrix is missing",
                    )
                })?;
                let bones = rig::evaluate_armature_bone_world_matrices_with_resources(
                    doc,
                    id,
                    frame,
                    armature_world,
                    &matrices,
                    &driver_values,
                    cache_directory,
                    scene.fps,
                    scene.fps_base,
                )?;
                bone_matrices.insert(
                    id.clone(),
                    bones
                        .into_iter()
                        .map(|(bone_id, matrix)| (bone_id, matrix.to_cols_array()))
                        .collect(),
                );
            }
        }
        let evaluation_key = json!({
            "engine": EVALUATION_ENGINE,
            "simulation_engine": crate::sim::SOLVER_VERSION,
            "scene_hash": scene_hash,
            "scene_id": scene_id,
            "view_layer": view_layer,
            "frame": frame,
            "simulation_cache_key": simulation_cache_key,
        });
        let evaluation_hash = hash::sha256(&hash::canonicalize(&evaluation_key)?);
        let mut volume_data = load_vdb_volume_data(doc, cache_directory)?;

        let mut nodes = BTreeMap::new();
        let mut meshes = BTreeMap::new();
        let mut instance_paths = BTreeMap::new();
        let mut strokes = BTreeMap::new();
        let mut modifier_warnings = Vec::new();
        for id in &mesh_evaluation_order {
            let node = doc.nodes.get(id).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "mesh evaluation node is missing")
            })?;
            let matrix = matrices.get(id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "evaluated world matrix is missing",
                )
            })?;
            let bounds = if instance_collection_target(node)?.is_some() {
                None
            } else if let Some(data_id) = &node.data {
                let data_block = doc.data_blocks.get(data_id).ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "node Data-Block reference does not exist",
                        json!({ "id": id, "data": data_id }),
                    )
                })?;
                if data_block.data_type == "mesh" {
                    let source_mesh = data_block.mesh.as_ref().ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "mesh Data-Block has no mesh payload",
                            json!({ "data": data_id }),
                        )
                    })?;
                    let mut rigged_source = source_mesh.clone();
                    if let Some(shape_keys) = &data_block.shape_keys {
                        let values = rig::evaluate_shape_key_values(
                            id,
                            node,
                            shape_keys,
                            doc,
                            frame,
                            &driver_values,
                        )?;
                        rig::apply_shape_keys_with_values(
                            &mut rigged_source,
                            shape_keys,
                            data_block,
                            &values,
                        )?;
                    }
                    if !has_stack_modifier(node, "cloth")
                        && !has_stack_modifier(node, "soft_body")
                        && let Some(positions) = deformed_vertices.get(id)
                    {
                        if positions.len() != rigged_source.vertices.len() {
                            return Err(PotError::with_details(
                                ErrorCode::EvaluationFailed,
                                "simulated vertex count does not match source mesh",
                                json!({"node_id":id}),
                            ));
                        }
                        for (vertex, position) in rigged_source.vertices.iter_mut().zip(positions) {
                            vertex.co = DVec3::from_array(*position);
                        }
                    }
                    if !has_stack_modifier(node, "dynamic_paint")
                        && let Some(colors) = paint_colors.get(id)
                    {
                        if colors.len() != rigged_source.vertices.len() {
                            return Err(PotError::with_details(
                                ErrorCode::EvaluationFailed,
                                "dynamic-paint color count does not match source mesh",
                                json!({"node_id":id}),
                            ));
                        }
                        let values = rigged_source
                            .vertices
                            .iter()
                            .zip(colors)
                            .map(|(vertex, color)| (format!("v{}", vertex.id), json!(color)))
                            .collect::<Map<_, _>>();
                        rigged_source.attributes.insert(
                            "color".to_owned(),
                            json!({"domain":"vertices","values":values}),
                        );
                    }
                    if !has_stack_modifier(node, "dynamic_paint")
                        && let Some(weights) = paint_weights.get(id)
                    {
                        if weights.len() != rigged_source.vertices.len()
                            || weights.iter().any(|weight| !weight.is_finite())
                        {
                            return Err(PotError::with_details(
                                ErrorCode::EvaluationFailed,
                                "dynamic-paint weight count or value is invalid",
                                json!({"node_id":id}),
                            ));
                        }
                        let values = rigged_source
                            .vertices
                            .iter()
                            .zip(weights)
                            .map(|(vertex, weight)| (format!("v{}", vertex.id), json!(weight)))
                            .collect::<Map<_, _>>();
                        rigged_source.attributes.insert(
                            "weight".to_owned(),
                            json!({"domain":"vertices","type":"float","values":values}),
                        );
                    }
                    attach_vertex_group_weights(&mut rigged_source, data_block);
                    let evaluated = crate::geom::modifiers::evaluate_modifiers_with_scene_context(
                        rigged_source,
                        &node.modifiers,
                        &doc.node_groups,
                        frame,
                        matrix,
                        &matrices,
                        &object_meshes,
                        &meshes,
                        &curve_path_lengths,
                        |mesh, modifier, modifier_warnings| match modifier.modifier_type.as_str() {
                            "armature" => rig::apply_armature_modifier_with_drivers(
                                mesh,
                                data_block,
                                modifier,
                                matrix,
                                doc,
                                frame,
                                &matrices,
                                &driver_values,
                                cache_directory,
                                scene.fps,
                                scene.fps_base,
                                &meshes,
                            ),
                            "lattice" => apply_lattice_modifier(
                                mesh, modifier, id, data_block, doc, &matrices,
                            ),
                            "volume_to_mesh" => apply_volume_to_mesh_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                &matrices,
                                &volume_data,
                            ),
                            "boolean" | "shrinkwrap" | "curve" | "hook" | "warp" => {
                                apply_scene_operand_modifier(
                                    mesh, modifier, id, doc, &matrices, &meshes,
                                )
                            }
                            "vertex_weight_edit"
                            | "vertex_weight_mix"
                            | "vertex_weight_proximity"
                            | "normal_edit"
                            | "uv_project"
                            | "uv_warp"
                            | "data_transfer" => apply_scene_attribute_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                &matrices,
                                &meshes,
                                cache_directory,
                            ),
                            "surface_deform" | "mesh_deform" => apply_scene_deform_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                &matrices,
                                &meshes,
                                modifier_warnings,
                            ),
                            "mesh_sequence_cache" => apply_mesh_sequence_cache_modifier(
                                mesh,
                                modifier,
                                doc,
                                cache_directory,
                                frame,
                                scene.fps,
                                scene.fps_base,
                            ),
                            "mesh_cache" => apply_mesh_cache_modifier(
                                mesh,
                                modifier,
                                doc,
                                cache_directory,
                                frame,
                                scene.fps,
                                scene.fps_base,
                            ),
                            "cloth" | "soft_body" | "collision" | "dynamic_paint" | "fluid"
                            | "particle_system" => apply_stack_physics_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                &scene_id,
                                frame,
                                &matrices,
                                cache_directory,
                                &mut simulated_particles,
                            ),
                            "particle_instance" => apply_particle_instance_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                frame,
                                f64::from(scene.fps) / scene.fps_base,
                                &matrices,
                                &simulated_particles,
                            ),
                            "explode" => apply_explode_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                frame,
                                &matrices,
                                &simulated_particles,
                            ),
                            _ => Err(PotError::new(
                                ErrorCode::InternalError,
                                "unexpected custom modifier callback",
                            )),
                        },
                    )
                    .map_err(|error| attach_node_owner(error, id))?;
                    modifier_warnings.extend(
                        evaluated
                            .warnings
                            .into_iter()
                            .map(|warning| attach_node_owner(warning, id)),
                    );
                    let evaluated_mesh = if has_stack_modifier(node, "fluid") {
                        evaluated.mesh
                    } else {
                        fluid_surfaces.get(id).cloned().unwrap_or(evaluated.mesh)
                    };
                    let bounds = mesh_bounds(&evaluated_mesh, matrix)?;
                    meshes.insert(id.clone(), evaluated_mesh);
                    if !evaluated.instance_paths.is_empty() {
                        instance_paths.insert(id.clone(), evaluated.instance_paths);
                    }
                    bounds
                } else if matches!(
                    data_block.data_type.as_str(),
                    "curve" | "surface" | "text" | "hair_curves" | "metaball" | "pointcloud"
                ) {
                    let mut source_mesh = match data_block.data_type.as_str() {
                        "curve" => {
                            let curve = data_block.curve.as_ref().ok_or_else(|| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    "curve Data-Block has no curve payload",
                                    json!({ "data": data_id }),
                                )
                            })?;
                            if let Some(taper) = curve.taper.as_ref() {
                                return Err(PotError::with_details(
                                    ErrorCode::UnsupportedFeature,
                                    "curve taper-object evaluation is not supported",
                                    json!({
                                        "feature_id": "curve.taper_object",
                                        "data": data_id,
                                        "taper": taper,
                                    }),
                                ));
                            }
                            if curve.fill_mode != crate::model::CurveFillMode::None
                                && curve.splines.len() > 1
                            {
                                return Err(PotError::with_details(
                                    ErrorCode::UnsupportedFeature,
                                    "multiple 2D fill contours are not supported",
                                    json!({
                                        "feature_id": "curve.fill.multiple_contours",
                                        "data": data_id,
                                    }),
                                ));
                            }
                            crate::geom::curve::evaluate_curve(curve).map_err(|error| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    format!("curve geometry is invalid: {error}"),
                                    json!({ "data": data_id }),
                                )
                            })?
                        }
                        "surface" => {
                            let surface = data_block.surface.as_ref().ok_or_else(|| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    "surface Data-Block has no surface payload",
                                    json!({ "data": data_id }),
                                )
                            })?;
                            crate::geom::curve::evaluate_surface(surface).map_err(|error| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    format!("surface geometry is invalid: {error}"),
                                    json!({ "data": data_id }),
                                )
                            })?
                        }
                        "text" => {
                            let text = data_block.text.as_ref().ok_or_else(|| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    "text Data-Block has no text payload",
                                    json!({ "data": data_id }),
                                )
                            })?;
                            crate::geom::text::evaluate_text(text).map_err(|error| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    format!("text geometry is invalid: {error}"),
                                    json!({ "data": data_id }),
                                )
                            })?
                        }
                        "metaball" => {
                            let metaball = data_block.metaball.as_ref().ok_or_else(|| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    "metaball Data-Block has no metaball payload",
                                    json!({ "data": data_id }),
                                )
                            })?;
                            crate::geom::metaball::to_mesh(metaball).map_err(|error| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    format!("metaball geometry is invalid: {}", error.message),
                                    json!({ "data": data_id }),
                                )
                            })?
                        }
                        "pointcloud" => {
                            let pointcloud = data_block.pointcloud.as_ref().ok_or_else(|| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    "point-cloud Data-Block has no point-cloud payload",
                                    json!({ "data": data_id }),
                                )
                            })?;
                            crate::geom::pointcloud::to_mesh(pointcloud).map_err(|error| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    format!("point-cloud geometry is invalid: {}", error.message),
                                    json!({ "data": data_id }),
                                )
                            })?
                        }
                        "hair_curves" => {
                            let hair = data_block.hair_curves.as_ref().ok_or_else(|| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    "hair-curves Data-Block has no hair-curves payload",
                                    json!({ "data": data_id }),
                                )
                            })?;
                            crate::geom::curve::evaluate_hair_curves(hair).map_err(|error| {
                                PotError::with_details(
                                    ErrorCode::SceneInvalid,
                                    format!("hair-curve geometry is invalid: {error}"),
                                    json!({ "data": data_id }),
                                )
                            })?
                        }
                        _ => {
                            return Err(PotError::new(
                                ErrorCode::InternalError,
                                "unexpected generated geometry Data-Block type",
                            ));
                        }
                    };
                    attach_vertex_group_weights(&mut source_mesh, data_block);
                    let evaluated = crate::geom::modifiers::evaluate_modifiers_with_scene_context(
                        source_mesh,
                        &node.modifiers,
                        &doc.node_groups,
                        frame,
                        matrix,
                        &matrices,
                        &object_meshes,
                        &meshes,
                        &curve_path_lengths,
                        |mesh, modifier, modifier_warnings| match modifier.modifier_type.as_str() {
                            "armature" => rig::apply_armature_modifier_with_drivers(
                                mesh,
                                data_block,
                                modifier,
                                matrix,
                                doc,
                                frame,
                                &matrices,
                                &driver_values,
                                cache_directory,
                                scene.fps,
                                scene.fps_base,
                                &meshes,
                            ),
                            "lattice" => apply_lattice_modifier(
                                mesh, modifier, id, data_block, doc, &matrices,
                            ),
                            "volume_to_mesh" => apply_volume_to_mesh_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                &matrices,
                                &volume_data,
                            ),
                            "boolean" | "shrinkwrap" | "curve" | "hook" | "warp" => {
                                apply_scene_operand_modifier(
                                    mesh, modifier, id, doc, &matrices, &meshes,
                                )
                            }
                            "vertex_weight_edit"
                            | "vertex_weight_mix"
                            | "vertex_weight_proximity"
                            | "normal_edit"
                            | "uv_project"
                            | "uv_warp"
                            | "data_transfer" => apply_scene_attribute_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                &matrices,
                                &meshes,
                                cache_directory,
                            ),
                            "surface_deform" | "mesh_deform" => apply_scene_deform_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                &matrices,
                                &meshes,
                                modifier_warnings,
                            ),
                            "mesh_sequence_cache" => apply_mesh_sequence_cache_modifier(
                                mesh,
                                modifier,
                                doc,
                                cache_directory,
                                frame,
                                scene.fps,
                                scene.fps_base,
                            ),
                            "mesh_cache" => apply_mesh_cache_modifier(
                                mesh,
                                modifier,
                                doc,
                                cache_directory,
                                frame,
                                scene.fps,
                                scene.fps_base,
                            ),
                            "cloth" | "soft_body" | "collision" | "dynamic_paint" | "fluid"
                            | "particle_system" => apply_stack_physics_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                &scene_id,
                                frame,
                                &matrices,
                                cache_directory,
                                &mut simulated_particles,
                            ),
                            "particle_instance" => apply_particle_instance_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                frame,
                                f64::from(scene.fps) / scene.fps_base,
                                &matrices,
                                &simulated_particles,
                            ),
                            "explode" => apply_explode_modifier(
                                mesh,
                                modifier,
                                id,
                                doc,
                                frame,
                                &matrices,
                                &simulated_particles,
                            ),
                            _ => Err(PotError::new(
                                ErrorCode::InternalError,
                                "unexpected custom modifier callback",
                            )),
                        },
                    )
                    .map_err(|error| attach_node_owner(error, id))?;
                    modifier_warnings.extend(
                        evaluated
                            .warnings
                            .into_iter()
                            .map(|warning| attach_node_owner(warning, id)),
                    );
                    let bounds = mesh_bounds(&evaluated.mesh, matrix)?;
                    meshes.insert(id.clone(), evaluated.mesh);
                    if !evaluated.instance_paths.is_empty() {
                        instance_paths.insert(id.clone(), evaluated.instance_paths);
                    }
                    bounds
                } else if data_block.data_type == "volume" {
                    let volume = data_block.volume.as_ref().ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "Volume Data-Block has no volume payload",
                            json!({ "data": data_id }),
                        )
                    })?;
                    let mut generated = None;
                    for modifier in node.modifiers.iter().filter(|modifier| {
                        modifier.enabled && modifier.modifier_type == "mesh_to_volume"
                    }) {
                        let object_text = modifier
                            .params
                            .get("object")
                            .and_then(serde_json::Value::as_str)
                            .ok_or_else(|| {
                                PotError::with_details(
                                    ErrorCode::InvalidArgument,
                                    "mesh-to-volume modifier requires an object operand",
                                    json!({"modifier_id":modifier.id,"parameter":"object"}),
                                )
                            })?;
                        let object_id = Id::new(object_text.to_owned()).map_err(|_| {
                            PotError::with_details(
                                ErrorCode::InvalidArgument,
                                "mesh-to-volume object operand is not a valid object ID",
                                json!({"modifier_id":modifier.id,"object":object_text}),
                            )
                        })?;
                        let operand_mesh = meshes
                            .get(&object_id)
                            .or_else(|| object_meshes.get(&object_id).copied())
                            .ok_or_else(|| {
                                PotError::with_details(
                                    ErrorCode::InvalidArgument,
                                    "mesh-to-volume object operand has no mesh geometry",
                                    json!({"modifier_id":modifier.id,"object":object_id}),
                                )
                            })?;
                        let operand_matrix =
                            matrices.get(&object_id).copied().ok_or_else(|| {
                                PotError::with_details(
                                    ErrorCode::EvaluationFailed,
                                    "mesh-to-volume object world matrix is missing",
                                    json!({"modifier_id":modifier.id,"object":object_id}),
                                )
                            })?;
                        let owner_inverse = matrix.inverse();
                        if !owner_inverse.is_finite() {
                            return Err(PotError::with_details(
                                ErrorCode::InvalidArgument,
                                "mesh-to-volume owner transform is singular",
                                json!({"modifier_id":modifier.id,"owner":id}),
                            ));
                        }
                        let operand_to_owner = owner_inverse * operand_matrix;
                        let mut transformed = (*operand_mesh).clone();
                        for vertex in &mut transformed.vertices {
                            vertex.co = operand_to_owner.transform_point3(vertex.co);
                        }
                        let number = |parameter: &str, default: f64| {
                            modifier
                                .params
                                .get(parameter)
                                .and_then(serde_json::Value::as_f64)
                                .unwrap_or(default)
                        };
                        let resolution_mode = modifier
                            .params
                            .get("resolution_mode")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("VOXEL_AMOUNT");
                        let voxel_size = match resolution_mode {
                            "VOXEL_SIZE" => number("voxel_size", 0.1),
                            "VOXEL_AMOUNT" => {
                                let amount = number("voxel_amount", 32.0);
                                let bounds = transformed.bounds().ok_or_else(|| {
                                    PotError::with_details(
                                        ErrorCode::InvalidArgument,
                                        "mesh-to-volume operand mesh is empty",
                                        json!({"modifier_id":modifier.id}),
                                    )
                                })?;
                                if !amount.is_finite() || amount <= 0.0 {
                                    return Err(PotError::with_details(
                                        ErrorCode::InvalidArgument,
                                        "mesh-to-volume voxel_amount must be positive",
                                        json!({"modifier_id":modifier.id,"parameter":"voxel_amount"}),
                                    ));
                                }
                                (bounds.size().length() / amount) as f32 as f64
                            }
                            _ => {
                                return Err(PotError::with_details(
                                    ErrorCode::InvalidArgument,
                                    "mesh-to-volume resolution_mode must be VOXEL_SIZE or VOXEL_AMOUNT",
                                    json!({"modifier_id":modifier.id,"parameter":"resolution_mode"}),
                                ));
                            }
                        };
                        generated = Some(crate::geom::volume::mesh_to_fog_volume(
                            &transformed,
                            voxel_size as f32 as f64,
                            number("interior_band_width", 0.2),
                            number("density", 1.0),
                            true,
                        )?);
                    }
                    if let Some(generated) = generated {
                        volume_data.insert(data_id.clone(), generated);
                    }
                    let base_volume = volume_data.get(data_id).unwrap_or(volume);
                    let mut displaced = None;
                    for modifier in node.modifiers.iter().filter(|modifier| {
                        modifier.enabled && modifier.modifier_type == "volume_displace"
                    }) {
                        let current = displaced.take().unwrap_or_else(|| base_volume.clone());
                        displaced = Some(apply_volume_displace_modifier(
                            &current,
                            modifier,
                            doc,
                            cache_directory,
                            &matrices,
                            id,
                        )?);
                    }
                    if let Some(displaced) = displaced {
                        volume_data.insert(data_id.clone(), displaced);
                    }
                    let evaluated_volume = volume_data.get(data_id).unwrap_or(volume);
                    volume_bounds(evaluated_volume, matrix)?
                } else if data_block.data_type == "grease_pencil" {
                    let grease_pencil = data_block.grease_pencil.as_ref().ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "Grease Pencil Data-Block has no payload",
                            json!({ "data": data_id }),
                        )
                    })?;
                    let evaluated =
                        evaluate_grease_pencil(grease_pencil, &doc.materials, matrix, frame)?;
                    if !evaluated.is_empty() {
                        strokes.insert(id.clone(), evaluated);
                    }
                    None
                } else {
                    None
                }
            } else {
                None
            };
            let dimensions = bounds.map(|bounds| bounds.size().to_array());
            nodes.insert(
                id.clone(),
                EvaluatedNode {
                    world_matrix: matrix.to_cols_array(),
                    bounds,
                    dimensions,
                },
            );
        }
        let has_mesh_dependent_constraints = doc.nodes.iter().any(|(id, node)| {
            evaluated_node_ids.contains(id)
                && node.constraints.iter().any(|constraint| {
                    constraint.enabled
                        && constraint.influence != 0.0
                        && matches!(
                            constraint.constraint_type,
                            ConstraintType::Shrinkwrap | ConstraintType::GeometryAttribute
                        )
                })
        });
        if has_mesh_dependent_constraints {
            matrices = rig::evaluate_world_matrices_with_evaluated_meshes(
                doc,
                frame,
                &driver_values,
                cache_directory,
                scene.fps,
                scene.fps_base,
                &meshes,
            )?;
            for (id, evaluated_node) in &mut nodes {
                let matrix = matrices.get(id).copied().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "evaluated world matrix is missing after mesh-dependent constraints",
                    )
                })?;
                evaluated_node.world_matrix = matrix.to_cols_array();
                if let Some(mesh) = meshes.get(id) {
                    evaluated_node.bounds = mesh_bounds(mesh, matrix)?;
                    evaluated_node.dimensions =
                        evaluated_node.bounds.map(|bounds| bounds.size().to_array());
                }
            }
            bone_matrices.clear();
            for (id, node) in &doc.nodes {
                if node.kind == "armature" && evaluated_node_ids.contains(id) {
                    let armature_world = matrices.get(id).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "evaluated armature world matrix is missing",
                        )
                    })?;
                    let bones = rig::evaluate_armature_bone_world_matrices_with_evaluated_meshes(
                        doc,
                        id,
                        frame,
                        armature_world,
                        &matrices,
                        &driver_values,
                        cache_directory,
                        scene.fps,
                        scene.fps_base,
                        &meshes,
                    )?;
                    bone_matrices.insert(
                        id.clone(),
                        bones
                            .into_iter()
                            .map(|(bone_id, matrix)| (bone_id, matrix.to_cols_array()))
                            .collect(),
                    );
                }
            }
        }
        render_simulated_particles(
            doc,
            &simulated_particles,
            &matrices,
            &mut meshes,
            &mut nodes,
            &evaluated_node_ids,
        )?;
        for (id, node) in &doc.nodes {
            if !nodes.contains_key(id) || instance_collection_target(node)?.is_none() {
                continue;
            }
            let matrix = matrices.get(id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "evaluated collection instance world matrix is missing",
                )
            })?;
            let (mesh, paths) =
                evaluate_collection_instance(doc, id, &matrices, &meshes, &mut BTreeSet::new())?;
            let bounds = mesh_bounds(&mesh, matrix)?;
            if !mesh.vertices.is_empty() {
                meshes.insert(id.clone(), mesh);
            }
            if !paths.is_empty() {
                instance_paths.insert(id.clone(), paths);
            }
            let evaluated_node = nodes.get_mut(id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "evaluated collection instance node is missing",
                )
            })?;
            evaluated_node.bounds = bounds;
            evaluated_node.dimensions = bounds.map(|value| value.size().to_array());
        }
        Ok(Self {
            scene_id,
            view_layer,
            frame,
            revision: doc.revision,
            scene_hash,
            evaluation_hash,
            nodes,
            instance_paths,
            meshes,
            strokes,
            volume_data,
            bone_matrices,
            simulation_cache_key,
            simulated_velocities,
            simulated_particles,
            fluid_particles,
            paint_colors,
            paint_weights,
            warnings: modifier_warnings,
        })
    }
}

/// Read and verify a cache resource from the scene's asset registry.
///
/// # Errors
///
/// Returns dependency, asset-integrity, or I/O errors.
pub(crate) fn read_cache_resource(
    doc: &SceneDoc,
    project_root: Option<&Path>,
    resource_id: &Id,
) -> Result<(Vec<u8>, String)> {
    let resource = doc.resources.get(resource_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            format!("Alembic cache resource `{resource_id}` is not registered"),
            json!({"resource_id":resource_id}),
        )
    })?;
    let expected_hash = resource
        .get("hash")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "Alembic cache resource has no expected content hash",
                json!({"resource_id":resource_id}),
            )
        })?;
    let bytes = if resource.get("packed").and_then(serde_json::Value::as_bool) == Some(true) {
        let values = resource
            .get("bytes")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "packed Alembic cache resource has no byte payload",
                    json!({"resource_id":resource_id}),
                )
            })?;
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_u64()
                    .and_then(|byte| u8::try_from(byte).ok())
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "packed Alembic cache byte is outside 0..255",
                            json!({"resource_id":resource_id,"index":index}),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let uri = resource
            .get("uri")
            .or_else(|| resource.get("path"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::DependencyMissing,
                    "Alembic cache resource has no file URI",
                    json!({"resource_id":resource_id}),
                )
            })?;
        let path_text = uri.strip_prefix("file://").unwrap_or(uri);
        let path = Path::new(path_text);
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else if let Some(root) = project_root {
            root.join(path)
        } else {
            return Err(PotError::with_details(
                ErrorCode::DependencyMissing,
                "relative Alembic cache resource cannot be resolved without a project root",
                json!({"resource_id":resource_id,"uri":uri}),
            ));
        };
        fs::read(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                PotError::with_details(
                    ErrorCode::DependencyMissing,
                    format!("Alembic cache file is missing: {}", path.display()),
                    json!({"resource_id":resource_id,"uri":uri}),
                )
            } else {
                PotError::io(&error)
            }
        })?
    };
    let actual_hash = hash::sha256(&bytes);
    if actual_hash != expected_hash {
        return Err(PotError::with_details(
            ErrorCode::AssetChanged,
            format!("Alembic cache resource `{resource_id}` changed after registration"),
            json!({
                "resource_id":resource_id,
                "uri":resource.get("uri").and_then(serde_json::Value::as_str),
                "expected_hash":expected_hash,
                "actual_hash":actual_hash,
            }),
        ));
    }
    Ok((bytes, actual_hash))
}

pub(crate) fn load_vdb_volume_data(
    doc: &SceneDoc,
    project_root: Option<&Path>,
) -> Result<BTreeMap<Id, crate::geom::volume::VolumeData>> {
    let referenced_data = doc
        .nodes
        .values()
        .filter_map(|node| node.data.clone())
        .collect::<BTreeSet<_>>();
    let mut volumes = BTreeMap::new();
    for data_id in referenced_data {
        let Some(data_block) = doc.data_blocks.get(&data_id) else {
            continue;
        };
        let Some(volume) = data_block.volume.as_ref() else {
            continue;
        };
        let crate::geom::volume::VolumeSource::File(source) = &volume.source else {
            continue;
        };
        if source.format != "vdb" {
            continue;
        }
        let reference = source.content_ref.as_deref().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::DependencyMissing,
                "VDB volume has no registered resource reference",
                json!({"feature_id":"volume.openvdb_evaluation.resource","data_id":data_id}),
            )
        })?;
        let resource_text = reference.strip_prefix("asset://").unwrap_or(reference);
        let resource_id = Id::new(resource_text.to_owned()).map_err(|_| {
            PotError::with_details(
                ErrorCode::DependencyMissing,
                "VDB volume resource reference is not a registered resource ID",
                json!({"feature_id":"volume.openvdb_evaluation.resource","data_id":data_id,"content_ref":reference}),
            )
        })?;
        let (bytes, content_hash) =
            read_cache_resource(doc, project_root, &resource_id).map_err(|mut error| {
                error.message = format!("VDB resource could not be loaded: {}", error.message);
                if let Some(details) = error.details.as_object_mut() {
                    details.insert(
                        "feature_id".to_owned(),
                        json!("volume.openvdb_evaluation.resource"),
                    );
                }
                error
            })?;
        let mut loaded = volume.clone();
        loaded.decoded_vdb = Some(crate::geom::vdb::VdbVolume::read_cached(
            &bytes,
            &content_hash,
        )?);
        volumes.insert(data_id, loaded);
    }
    Ok(volumes)
}

fn attach_node_owner(mut error: PotError, node_id: &Id) -> PotError {
    if error.code == ErrorCode::UnsupportedFeature
        && let Some(details) = error.details.as_object_mut()
    {
        details
            .entry("node_id".to_owned())
            .or_insert_with(|| json!(node_id));
    }
    error
}

fn unsupported_feature_owner(doc: &SceneDoc, error: &PotError) -> Option<Id> {
    let details = &error.details;
    for key in ["node_id", "object", "owner_id"] {
        if let Some(node_id) = details
            .get(key)
            .and_then(serde_json::Value::as_str)
            .and_then(|value| Id::new(value.to_owned()).ok())
            .filter(|node_id| doc.nodes.contains_key(node_id))
        {
            return Some(node_id);
        }
    }
    if let Some(modifier_id) = details
        .get("modifier_id")
        .and_then(serde_json::Value::as_str)
        && let Some((node_id, _)) = doc.nodes.iter().find(|(_, node)| {
            node.modifiers
                .iter()
                .any(|modifier| modifier.id.as_str() == modifier_id)
        })
    {
        return Some(node_id.clone());
    }
    if let Some(data_id) = details
        .get("data")
        .or_else(|| details.get("data_id"))
        .and_then(serde_json::Value::as_str)
        && let Some((node_id, _)) = doc
            .nodes
            .iter()
            .find(|(_, node)| node.data.as_ref().is_some_and(|id| id.as_str() == data_id))
    {
        return Some(node_id.clone());
    }
    let feature = details
        .get("feature_id")
        .and_then(serde_json::Value::as_str)?;
    let modifier_type = feature.strip_prefix("modifier.")?.split('.').next()?;
    doc.nodes
        .iter()
        .find(|(_, node)| {
            node.modifiers
                .iter()
                .any(|modifier| modifier.modifier_type == modifier_type)
        })
        .map(|(node_id, _)| node_id.clone())
}

fn add_mesh_operand_dependency(
    dependencies: &mut BTreeMap<Id, BTreeSet<Id>>,
    owner_id: &Id,
    value: Option<&serde_json::Value>,
) {
    let Some(reference) = value.and_then(serde_json::Value::as_str) else {
        return;
    };
    let Ok(target_id) = Id::new(reference.to_owned()) else {
        return;
    };
    if dependencies.contains_key(&target_id)
        && let Some(owner_dependencies) = dependencies.get_mut(owner_id)
    {
        owner_dependencies.insert(target_id);
    }
}

fn object_mesh_dependencies(doc: &SceneDoc) -> Result<BTreeMap<Id, BTreeSet<Id>>> {
    let mut dependencies = doc
        .nodes
        .keys()
        .cloned()
        .map(|id| (id, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    for (owner_id, node) in &doc.nodes {
        for modifier in node.modifiers.iter().filter(|modifier| modifier.enabled) {
            let parameters: &[&str] = match modifier.modifier_type.as_str() {
                "boolean" | "data_transfer" | "mesh_deform" | "curve" | "mesh_to_volume" => {
                    &["object"]
                }
                "shrinkwrap" => {
                    if modifier
                        .params
                        .get("target")
                        .is_some_and(|value| !value.is_null())
                    {
                        &["target"]
                    } else {
                        &["object"]
                    }
                }
                "surface_deform" | "vertex_weight_proximity" | "normal_edit" => &["target"],
                "array" => &["start_cap", "end_cap", "curve"],
                _ => &[],
            };
            for parameter in parameters {
                add_mesh_operand_dependency(
                    &mut dependencies,
                    owner_id,
                    modifier.params.get(*parameter),
                );
            }
            if modifier.modifier_type == "boolean"
                && let Some(collection_text) = modifier
                    .params
                    .get("collection")
                    .and_then(serde_json::Value::as_str)
                && let Ok(collection_id) = Id::new(collection_text.to_owned())
                && doc.collections.contains_key(&collection_id)
            {
                let mut object_ids = BTreeSet::new();
                collect_modifier_collection_objects(
                    doc,
                    &collection_id,
                    &mut BTreeSet::new(),
                    &mut object_ids,
                )?;
                for target_id in object_ids {
                    if dependencies.contains_key(&target_id)
                        && let Some(owner_dependencies) = dependencies.get_mut(owner_id)
                    {
                        owner_dependencies.insert(target_id);
                    }
                }
            }
        }
        if let Some(collection_id) = instance_collection_target(node)? {
            let mut object_ids = BTreeSet::new();
            collect_modifier_collection_objects(
                doc,
                &collection_id,
                &mut BTreeSet::new(),
                &mut object_ids,
            )?;
            for target_id in object_ids {
                if dependencies.contains_key(&target_id)
                    && let Some(owner_dependencies) = dependencies.get_mut(owner_id)
                {
                    owner_dependencies.insert(target_id);
                }
            }
        }
        for constraint in node.constraints.iter().filter(|constraint| {
            constraint.enabled
                && constraint.influence != 0.0
                && matches!(
                    constraint.constraint_type,
                    ConstraintType::Shrinkwrap | ConstraintType::GeometryAttribute
                )
        }) {
            if let Some(target_id) = &constraint.target
                && dependencies.contains_key(target_id)
                && let Some(owner_dependencies) = dependencies.get_mut(owner_id)
            {
                owner_dependencies.insert(target_id.clone());
            }
        }
    }
    Ok(dependencies)
}

fn object_mesh_evaluation_order(
    doc: &SceneDoc,
    selected_nodes: Option<&BTreeSet<Id>>,
) -> Result<Vec<Id>> {
    fn visit(
        id: &Id,
        dependencies: &BTreeMap<Id, BTreeSet<Id>>,
        complete: &mut BTreeSet<Id>,
        active: &mut Vec<Id>,
        output: &mut Vec<Id>,
    ) -> Result<()> {
        if complete.contains(id) {
            return Ok(());
        }
        if let Some(start) = active.iter().position(|active_id| active_id == id) {
            let mut cycle: Vec<String> = active[start..].iter().map(ToString::to_string).collect();
            cycle.push(id.to_string());
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "evaluated mesh dependency cycle",
                json!({"cycle":cycle,"dependency":"evaluated mesh"}),
            ));
        }
        active.push(id.clone());
        if let Some(targets) = dependencies.get(id) {
            for target in targets {
                visit(target, dependencies, complete, active, output)?;
            }
        }
        let popped = active.pop();
        debug_assert_eq!(popped.as_ref(), Some(id));
        complete.insert(id.clone());
        output.push(id.clone());
        Ok(())
    }
    fn include_dependencies(
        id: &Id,
        dependencies: &BTreeMap<Id, BTreeSet<Id>>,
        required: &mut BTreeSet<Id>,
    ) {
        if !required.insert(id.clone()) {
            return;
        }
        if let Some(targets) = dependencies.get(id) {
            for target in targets {
                include_dependencies(target, dependencies, required);
            }
        }
    }

    let dependencies = object_mesh_dependencies(doc)?;
    let roots = selected_nodes
        .cloned()
        .unwrap_or_else(|| doc.nodes.keys().cloned().collect());
    let mut required = BTreeSet::new();
    for id in roots {
        include_dependencies(&id, &dependencies, &mut required);
    }
    let mut order = Vec::with_capacity(required.len());
    let mut complete = BTreeSet::new();
    let mut active = Vec::new();
    for particle_system_first in [true, false] {
        for id in &required {
            let Some(node) = doc.nodes.get(id) else {
                continue;
            };
            let has_particle_system = node.properties.contains_key("physics_particle_emitter")
                || node.modifiers.iter().any(|modifier| {
                    modifier.enabled && modifier.modifier_type == "particle_system"
                });
            if has_particle_system == particle_system_first {
                visit(id, &dependencies, &mut complete, &mut active, &mut order)?;
            }
        }
    }
    Ok(order)
}

/// Resolve the object-reference parameters for a scene-bound modifier through one shared path.
///
/// Each operand retains its object-local mesh and carries the matrix from that object into
/// the modifier subject's local space. Camera data is preserved for projector modifiers.
pub(crate) fn resolve_modifier_object_operands(
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
) -> Result<BTreeMap<String, Vec<crate::geom::modifiers::attributes::SceneOperand>>> {
    let reference_parameters: &[&str] = match modifier.modifier_type.as_str() {
        "boolean" | "curve" | "hook" | "data_transfer" | "mesh_deform" => &["object"],
        "uv_project" => &["projectors"],
        "uv_warp" | "warp" => &["object_from", "object_to"],
        "shrinkwrap" => &["target", "object"],
        "vertex_weight_edit" | "vertex_weight_mix" => &["mask_tex_map_object"],
        "vertex_weight_proximity" => &["target", "mask_tex_map_object"],
        "normal_edit" | "surface_deform" => &["target"],
        "volume_displace" => &["texture_map_object"],
        _ => &[],
    };
    let subject_world = world_matrices.get(subject_id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "modifier subject world matrix is missing",
        )
    })?;
    let subject_inverse = subject_world.inverse();
    if !subject_inverse.is_finite() {
        return Err(PotError::invalid_argument(
            "modifier subject transform must be invertible",
        ));
    }
    let mut resolved = BTreeMap::new();
    for parameter in reference_parameters {
        let Some(value) = modifier.params.get(*parameter) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let object_texts = match value {
            serde_json::Value::String(text) => vec![text.as_str()],
            serde_json::Value::Array(values) if *parameter == "projectors" => values
                .iter()
                .map(|value| {
                    value.as_str().ok_or_else(|| {
                        PotError::invalid_argument("uv_project projectors must contain object IDs")
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            _ => {
                return Err(PotError::invalid_argument(format!(
                    "modifier `{}` `{parameter}` must be an object ID",
                    modifier.modifier_type
                )));
            }
        };
        let mut operands = Vec::with_capacity(object_texts.len());
        for object_text in object_texts {
            let object_id = Id::new(object_text.to_owned()).map_err(|error| {
                PotError::with_details(
                    ErrorCode::InvalidArgument,
                    error.message,
                    json!({"modifier_id":modifier.id,"parameter":parameter,"object":object_text}),
                )
            })?;
            let object = doc.nodes.get(&object_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    format!("modifier operand object `{object_id}` was not found"),
                    json!({"modifier_id":modifier.id,"parameter":parameter,"object":object_id}),
                )
            })?;
            let data_block = object
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id));
            let operand_mesh = if let Some(evaluated_mesh) = evaluated_meshes.get(&object_id) {
                evaluated_mesh.clone()
            } else {
                let mut base_mesh = match data_block {
                    Some(data) if data.data_type == "mesh" => {
                        data.mesh.clone().ok_or_else(|| {
                            PotError::with_details(
                                ErrorCode::SceneInvalid,
                                "mesh modifier operand has no mesh payload",
                                json!({"modifier_id":modifier.id,"object":object_id}),
                            )
                        })?
                    }
                    Some(data) if data.data_type == "curve" => crate::geom::curve::evaluate_curve(
                        data.curve.as_ref().ok_or_else(|| {
                            PotError::with_details(
                                ErrorCode::SceneInvalid,
                                "curve modifier operand has no curve payload",
                                json!({"modifier_id":modifier.id,"object":object_id}),
                            )
                        })?,
                    )
                    .map_err(|error| {
                        PotError::with_details(
                            ErrorCode::InvalidArgument,
                            format!("curve modifier operand is invalid: {error}"),
                            json!({"modifier_id":modifier.id,"object":object_id}),
                        )
                    })?,
                    _ => Mesh::default(),
                };
                if let Some(data) = data_block {
                    attach_vertex_group_weights(&mut base_mesh, data);
                }
                base_mesh
            };
            let object_world = world_matrices.get(&object_id).copied().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "modifier operand world matrix is missing",
                    json!({"modifier_id":modifier.id,"object":object_id}),
                )
            })?;
            operands.push(crate::geom::modifiers::attributes::SceneOperand {
                mesh: operand_mesh,
                local_to_subject: subject_inverse * object_world,
                camera: data_block.and_then(|data| data.camera.clone()),
                armature: object.kind == "armature"
                    || data_block.is_some_and(|data| data.data_type == "armature"),
            });
        }
        resolved.insert((*parameter).to_owned(), operands);
    }
    Ok(resolved)
}

pub(crate) fn apply_scene_attribute_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
    project_root: Option<&Path>,
) -> Result<()> {
    let operands = resolve_modifier_object_operands(
        modifier,
        subject_id,
        doc,
        world_matrices,
        evaluated_meshes,
    )?;
    let texture = load_weight_modifier_texture(modifier, doc, project_root)?;
    let subject_world = world_matrices.get(subject_id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "modifier subject world matrix is missing",
        )
    })?;
    *mesh = crate::geom::modifiers::attributes::evaluate_with_context(
        mesh,
        modifier,
        &operands,
        texture.as_ref(),
        subject_world,
    )?;
    Ok(())
}

fn load_weight_modifier_texture(
    modifier: &Modifier,
    doc: &SceneDoc,
    project_root: Option<&Path>,
) -> Result<Option<crate::image::ImageData>> {
    let Some(value) = modifier.params.get("mask_texture") else {
        return Ok(None);
    };
    let Some(image_text) = value.as_str().filter(|value| !value.is_empty()) else {
        if value.is_null() {
            return Ok(None);
        }
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "weight modifier texture is not an image texture",
            json!({"feature_id":format!("modifier.{}.texture_type",modifier.modifier_type),"modifier_id":modifier.id}),
        ));
    };
    let image_id = Id::new(image_text.to_owned()).map_err(|error| {
        PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "weight modifier texture is not an image texture",
            json!({"feature_id":format!("modifier.{}.texture_type",modifier.modifier_type),"modifier_id":modifier.id,"cause":error.message}),
        )
    })?;
    let image = doc.images.get(&image_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "procedural Blender texture masks are not modeled",
            json!({"feature_id":format!("modifier.{}.texture_type",modifier.modifier_type),"modifier_id":modifier.id,"texture":image_text}),
        )
    })?;
    let root = project_root.ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            "image texture mask requires a project root",
            json!({"feature_id":format!("modifier.{}.image_texture",modifier.modifier_type),"image":image_id}),
        )
    })?;
    crate::image::load_image_data(image, root, crate::image::ImageInterpolation::Linear).map(Some)
}

pub(crate) fn apply_scene_deform_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
    warnings: &mut Vec<PotError>,
) -> Result<()> {
    let operands = resolve_modifier_object_operands(
        modifier,
        subject_id,
        doc,
        world_matrices,
        evaluated_meshes,
    )?;
    let parameter = if modifier.modifier_type == "surface_deform" {
        "target"
    } else {
        "object"
    };
    let operand = operands
        .get(parameter)
        .and_then(|items| items.first())
        .ok_or_else(|| {
            PotError::invalid_argument(format!(
                "modifier `{}` requires an object operand",
                modifier.modifier_type
            ))
        })?;
    let mut target_mesh = operand.mesh.clone();
    for vertex in &mut target_mesh.vertices {
        vertex.co = operand.local_to_subject.transform_point3(vertex.co);
    }
    let mut resolved = modifier.clone();
    resolved.params.remove(parameter);
    resolved.runtime.target_to_subject = Some(operand.local_to_subject.to_cols_array());
    resolved.runtime.target_mesh = Some(serde_json::to_value(target_mesh).map_err(|error| {
        PotError::invalid_argument(format!("modifier operand serialization failed: {error}"))
    })?);
    let evaluation = crate::geom::modifiers::evaluate_modifiers_with_groups(
        mesh,
        &[resolved],
        &doc.node_groups,
    )?;
    *mesh = evaluation.mesh;
    warnings.extend(
        evaluation
            .warnings
            .into_iter()
            .map(|warning| attach_node_owner(warning, subject_id)),
    );
    Ok(())
}

fn apply_mesh_cache_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    doc: &SceneDoc,
    project_root: Option<&Path>,
    frame: f64,
    fps: u32,
    fps_base: f64,
) -> Result<()> {
    let resource_text = modifier
        .params
        .get("resource")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                "Mesh Cache resource must be a resource ID",
                json!({"modifier_id":modifier.id,"parameter":"resource"}),
            )
        })?;
    let resource_id = Id::new(resource_text.to_owned()).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            error.message,
            json!({"modifier_id":modifier.id,"resource":resource_text}),
        )
    })?;
    let (bytes, _) = read_cache_resource(doc, project_root, &resource_id)?;
    crate::geom::modifiers::attributes::evaluate_mesh_cache(
        mesh, modifier, frame, fps, fps_base, &bytes,
    )
}

fn apply_mesh_sequence_cache_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    doc: &SceneDoc,
    project_root: Option<&Path>,
    frame: f64,
    fps: u32,
    fps_base: f64,
) -> Result<()> {
    let resource_text = modifier
        .params
        .get("resource")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                "Mesh Sequence Cache resource must be a resource ID",
                json!({"modifier_id":modifier.id,"parameter":"resource"}),
            )
        })?;
    let resource_id = Id::new(resource_text).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            error.message,
            json!({"modifier_id":modifier.id,"resource":resource_text}),
        )
    })?;
    let (bytes, actual_hash) = read_cache_resource(doc, project_root, &resource_id)?;
    crate::geom::mesh_cache::evaluate(mesh, modifier, frame, fps, fps_base, &actual_hash, &bytes)
}

const MAX_RENDERED_PARTICLES: usize = 100_000;

fn render_simulated_particles(
    doc: &SceneDoc,
    particles: &BTreeMap<Id, Vec<crate::sim::ParticleState>>,
    matrices: &BTreeMap<Id, DMat4>,
    meshes: &mut BTreeMap<Id, Mesh>,
    nodes: &mut BTreeMap<Id, EvaluatedNode>,
    evaluated_node_ids: &BTreeSet<Id>,
) -> Result<()> {
    for (emitter_id, states) in particles {
        if !evaluated_node_ids.contains(emitter_id) {
            continue;
        }
        if doc.nodes.get(emitter_id).is_some_and(|node| {
            node.modifiers
                .iter()
                .any(|modifier| modifier.enabled && modifier.modifier_type == "explode")
        }) {
            continue;
        }
        let visible_states = states
            .iter()
            .filter(|state| state.life_state == crate::sim::ParticleLifeState::Alive)
            .collect::<Vec<_>>();
        if visible_states.is_empty() {
            continue;
        }
        if visible_states.len() > MAX_RENDERED_PARTICLES {
            return Err(PotError::with_details(
                ErrorCode::LimitExceeded,
                "particle visualization exceeds the 100000 particle limit",
                json!({"node_id":emitter_id,"count":visible_states.len()}),
            ));
        }
        let settings = doc
            .nodes
            .get(emitter_id)
            .and_then(|node| node.properties.get("physics_particle_emitter"))
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "particle emitter settings are missing",
                    json!({"node_id":emitter_id}),
                )
            })?;
        let render_type = settings
            .get("render_type")
            .and_then(serde_json::Value::as_str);
        let mode = match render_type {
            Some("HALO") => "points",
            Some("PATH") => "hair_curves",
            Some("OBJECT") => "object",
            Some("COLLECTION") => "collection",
            Some(_) => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "particle render_type is not supported",
                    json!({"feature_id":"physics.particle_emitter.render_type","node_id":emitter_id,"render_type":render_type}),
                ));
            }
            None => settings
                .get("render_as")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("points"),
        };
        let emitter_world = matrices.get(emitter_id).copied().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "particle emitter transform is missing",
                json!({"node_id":emitter_id}),
            )
        })?;
        let emitter_inverse = emitter_world.inverse();
        if !emitter_inverse.is_finite() {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "particle emitter transform cannot be inverted",
                json!({"node_id":emitter_id}),
            ));
        }
        let (template, object_world) = match mode {
            "points" => (
                Mesh::icosphere(IcosphereParams {
                    subdivisions: 1,
                    radius: 1.0,
                })
                .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?,
                None,
            ),
            "hair_curves" => (
                Mesh::cylinder(CylinderParams {
                    vertices: 6,
                    radius: 1.0,
                    depth: 2.0,
                    ..CylinderParams::default()
                })
                .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?,
                None,
            ),
            "object" => {
                let object_id = settings
                    .get("instance_object")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::InvalidOperation,
                            "object particle rendering requires instance_object",
                            json!({"node_id":emitter_id}),
                        )
                    })?;
                let object_id = Id::new(object_id.to_owned())
                    .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.message))?;
                if object_id == *emitter_id {
                    return Err(PotError::with_details(
                        ErrorCode::InvalidOperation,
                        "particle instances cannot reference their own emitter",
                        json!({"node_id":emitter_id}),
                    ));
                }
                let template = meshes.get(&object_id).cloned().ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::TargetNotFound,
                        "particle instance mesh was not evaluated",
                        json!({"node_id":emitter_id,"instance_object":object_id}),
                    )
                })?;
                let mut matrix = matrices.get(&object_id).copied().ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::EvaluationFailed,
                        "particle instance transform is missing",
                        json!({"instance_object":object_id}),
                    )
                })?;
                matrix.w_axis = DVec3::ZERO.extend(1.0);
                (template, Some(matrix))
            }
            "collection" => {
                let collection_text = settings
                    .get("instance_collection")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::InvalidOperation,
                            "collection particle rendering requires instance_collection",
                            json!({"node_id":emitter_id}),
                        )
                    })?;
                let collection_id = Id::new(collection_text.to_owned())
                    .map_err(|error| PotError::new(ErrorCode::InvalidOperation, error.message))?;
                let mut object_ids = BTreeSet::new();
                collect_modifier_collection_objects(
                    doc,
                    &collection_id,
                    &mut BTreeSet::new(),
                    &mut object_ids,
                )?;
                let mut template = Mesh::new();
                for object_id in object_ids {
                    if &object_id == emitter_id {
                        continue;
                    }
                    let Some(source) = meshes.get(&object_id) else {
                        continue;
                    };
                    let object_world = matrices.get(&object_id).copied().ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::EvaluationFailed,
                            "particle collection object transform is missing",
                            json!({"object":object_id}),
                        )
                    })?;
                    append_transformed_mesh(&mut template, source, emitter_inverse * object_world)?;
                }
                (template, None)
            }
            _ => {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "particle render_as setting is invalid",
                    json!({"node_id":emitter_id,"render_as":mode}),
                ));
            }
        };
        let output = meshes.get_mut(emitter_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "particle emitter mesh was not evaluated",
                json!({"node_id":emitter_id}),
            )
        })?;
        let size = settings
            .get("hair_length")
            .and_then(serde_json::Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(0.1);
        for particle in visible_states {
            let position = DVec3::from_array(particle.position);
            let world_transform = if let Some(object_world) = object_world {
                DMat4::from_translation(position)
                    * DMat4::from_scale(DVec3::splat(particle.size))
                    * object_world
            } else if mode == "hair_curves" {
                let velocity = DVec3::from_array(particle.velocity);
                let direction = if velocity.length_squared() > 1.0e-24 {
                    velocity.normalize()
                } else {
                    DVec3::Z
                };
                DMat4::from_scale_rotation_translation(
                    DVec3::new(
                        0.004 * particle.size,
                        0.004 * particle.size,
                        size * 0.5 * particle.size,
                    ),
                    DQuat::from_rotation_arc(DVec3::Z, direction),
                    position + direction * (size * 0.5 * particle.size),
                )
            } else {
                DMat4::from_translation(position)
                    * DMat4::from_scale(DVec3::splat(0.02 * particle.size))
            };
            append_transformed_mesh(output, &template, emitter_inverse * world_transform)?;
        }
        let bounds = mesh_bounds(output, emitter_world)?;
        let evaluated = nodes.get_mut(emitter_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "particle emitter node is missing")
        })?;
        evaluated.bounds = bounds;
        evaluated.dimensions = bounds.map(|value| value.size().to_array());
    }
    Ok(())
}

fn has_stack_modifier(node: &Node, modifier_type: &str) -> bool {
    node.modifiers
        .iter()
        .any(|modifier| modifier.modifier_type == modifier_type)
}

fn apply_stack_physics_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    node_id: &Id,
    doc: &SceneDoc,
    scene_id: &Id,
    frame: f64,
    matrices: &BTreeMap<Id, DMat4>,
    cache_directory: Option<&Path>,
    particles: &mut BTreeMap<Id, Vec<crate::sim::ParticleState>>,
) -> Result<()> {
    let result = crate::sim::evaluate_stack_modifier(
        doc,
        scene_id,
        node_id,
        modifier,
        mesh,
        frame,
        matrices,
        cache_directory,
    )?;
    if let Some(mut states) = result.particles {
        let system_index = doc
            .nodes
            .get(node_id)
            .into_iter()
            .flat_map(|node| node.modifiers.iter())
            .filter(|candidate| candidate.modifier_type == "particle_system")
            .position(|candidate| candidate.id == modifier.id)
            .and_then(|index| u32::try_from(index).ok())
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "particle system stack index is invalid",
                    json!({"node_id":node_id,"modifier_id":modifier.id}),
                )
            })?;
        for particle in &mut states {
            particle.system_index = system_index;
        }
        let cached = particles.entry(node_id.clone()).or_default();
        cached.retain(|particle| particle.system_index != system_index);
        cached.extend(states);
    }
    *mesh = result.mesh;
    Ok(())
}

fn apply_particle_instance_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    node_id: &Id,
    doc: &SceneDoc,
    frame: f64,
    fps: f64,
    matrices: &BTreeMap<Id, DMat4>,
    particles: &BTreeMap<Id, Vec<crate::sim::ParticleState>>,
) -> Result<()> {
    let object_text = modifier
        .params
        .get("object")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| PotError::invalid_argument("particle_instance requires an object ID"))?;
    let object_id = Id::new(object_text.to_owned()).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidOperation,
            error.message,
            json!({"modifier_id":modifier.id,"object":object_text}),
        )
    })?;
    let index = particle_system_index(modifier)?;
    let (system_modifier, _) = particle_system_for_index(doc, &object_id, index)?;
    let particle_states = if system_modifier.enabled {
        select_particle_states(particles.get(&object_id), index)
    } else {
        Vec::new()
    };
    let source_world = matrices.get(&object_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "particle system object transform is missing",
            json!({"modifier_id":modifier.id,"object":object_id}),
        )
    })?;
    let target_world = matrices.get(node_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "particle instance target transform is missing",
            json!({"node_id":node_id}),
        )
    })?;
    *mesh = crate::geom::modifiers::simulation::particle_instance(
        mesh,
        &particle_states,
        source_world,
        target_world,
        &modifier.params,
        frame,
        fps,
    )?;
    Ok(())
}

fn apply_explode_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    node_id: &Id,
    doc: &SceneDoc,
    frame: f64,
    matrices: &BTreeMap<Id, DMat4>,
    particles: &BTreeMap<Id, Vec<crate::sim::ParticleState>>,
) -> Result<()> {
    let index = particle_system_index(modifier)?;
    let (system_modifier, _) = particle_system_for_index(doc, node_id, index)?;
    let particle_states = if system_modifier.enabled {
        select_particle_states(particles.get(node_id), index)
    } else {
        Vec::new()
    };
    let matrix = matrices.get(node_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "explode object transform is missing",
            json!({"node_id":node_id}),
        )
    })?;
    *mesh = crate::geom::modifiers::simulation::explode(
        mesh,
        &particle_states,
        matrix,
        &modifier.params,
        frame,
        50.0,
    )?;
    Ok(())
}

fn particle_system_index(modifier: &Modifier) -> Result<u32> {
    let value = modifier
        .params
        .get("particle_system_index")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1)
        .checked_sub(1)
        .ok_or_else(|| PotError::invalid_argument("particle_system_index must be positive"))?;
    u32::try_from(value).map_err(|_| {
        PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "particle system index exceeds the supported range",
            json!({"feature_id":"modifier.particle_system_index","modifier_id":modifier.id}),
        )
    })
}

fn particle_system_for_index<'a>(
    doc: &'a SceneDoc,
    node_id: &Id,
    index: u32,
) -> Result<(&'a Modifier, &'a serde_json::Value)> {
    let node = doc.nodes.get(node_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::TargetNotFound,
            format!("particle system object `{node_id}` was not found"),
        )
    })?;
    let system_modifier = node
        .modifiers
        .iter()
        .filter(|modifier| modifier.modifier_type == "particle_system")
        .nth(
            usize::try_from(index)
                .map_err(|_| PotError::invalid_argument("particle system index is out of range"))?,
        )
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "particle system index does not exist on the object",
                json!({"node_id":node_id,"particle_system_index":index}),
            )
        })?;
    let settings_id = system_modifier
        .params
        .get("settings_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(system_modifier.id.as_str());
    let settings = node
        .properties
        .get("physics_particle_systems")
        .and_then(|systems| systems.get(settings_id))
        .or_else(|| {
            let primary_id = node
                .properties
                .get("physics_particle_emitter_modifier_id")
                .and_then(serde_json::Value::as_str);
            (primary_id.is_none() || primary_id == Some(settings_id))
                .then(|| node.properties.get("physics_particle_emitter"))
                .flatten()
        })
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "particle system settings are missing",
                json!({"node_id":node_id,"settings_id":settings_id}),
            )
        })?;
    Ok((system_modifier, settings))
}

fn select_particle_states(
    states: Option<&Vec<crate::sim::ParticleState>>,
    system_index: u32,
) -> Vec<crate::sim::ParticleState> {
    states
        .into_iter()
        .flatten()
        .filter(|state| state.system_index == system_index)
        .cloned()
        .collect()
}

fn apply_volume_displace_modifier(
    volume: &crate::geom::volume::VolumeData,
    modifier: &Modifier,
    doc: &SceneDoc,
    project_root: Option<&Path>,
    world_matrices: &BTreeMap<Id, DMat4>,
    volume_id: &Id,
) -> Result<crate::geom::volume::VolumeData> {
    let params = &modifier.params;
    let texture = params.get("texture").and_then(serde_json::Value::as_str);
    let (texture_type, image) = if let Some(texture) = texture {
        if texture == "noise" {
            ("NOISE", None)
        } else {
            let image_id = Id::new(texture.to_owned()).map_err(|error| {
                PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "Volume Displace texture is not a supported image or noise source",
                    json!({"feature_id":"modifier.volume_displace.texture_type","texture":texture,"cause":error.message}),
                )
            })?;
            let image = doc.images.get(&image_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "Volume Displace texture is not a supported image or noise source",
                    json!({"feature_id":"modifier.volume_displace.texture_type","texture":texture}),
                )
            })?;
            let root = project_root.ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::DependencyMissing,
                    "Volume Displace image texture requires a project root",
                    json!({"feature_id":"modifier.volume_displace.image_texture","image":image_id}),
                )
            })?;
            let image = crate::image::load_image_data(
                image,
                root,
                crate::image::ImageInterpolation::Linear,
            )?;
            ("IMAGE", Some(image))
        }
    } else {
        ("NONE", None)
    };
    let volume_world = world_matrices.get(volume_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Volume Displace target transform is missing",
            json!({"node_id":volume_id}),
        )
    })?;
    let texture_map_object_world = params
        .get("texture_map_object")
        .and_then(serde_json::Value::as_str)
        .map(|object_text| {
            let object_id = Id::new(object_text.to_owned()).map_err(|error| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    error.message,
                    json!({"texture_map_object":object_text}),
                )
            })?;
            world_matrices.get(&object_id).copied().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "Volume Displace texture map object was not found",
                    json!({"texture_map_object":object_id}),
                )
            })
        })
        .transpose()?;
    crate::geom::modifiers::simulation::displace_volume(
        volume,
        params,
        texture_type,
        image.as_ref(),
        volume_world,
        texture_map_object_world,
    )
}

fn apply_generic_scene_mesh_operand_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
) -> Result<()> {
    let operands = resolve_modifier_object_operands(
        modifier,
        subject_id,
        doc,
        world_matrices,
        evaluated_meshes,
    )?;
    let object_parameter =
        if modifier.modifier_type == "shrinkwrap" && modifier.params.contains_key("target") {
            "target"
        } else {
            "object"
        };
    let mut combined = None;
    for operand in operands.get(object_parameter).into_iter().flatten() {
        let mut transformed = operand.mesh.clone();
        for vertex in &mut transformed.vertices {
            vertex.co = operand.local_to_subject.transform_point3(vertex.co);
        }
        combined = Some(match combined {
            None => transformed,
            Some(existing) => {
                combine_scene_operand_meshes(&existing, &transformed, &modifier.modifier_type)?
            }
        });
    }
    let mut resolved = modifier.clone();
    for parameter in ["object", "target", "collection"] {
        resolved.params.remove(parameter);
    }
    let operand = combined.ok_or_else(|| {
        PotError::invalid_argument(format!(
            "modifier `{}` has no resolved object operand",
            modifier.modifier_type
        ))
    })?;
    let operand = serde_json::to_value(operand).map_err(|error| {
        PotError::invalid_argument(format!("modifier operand serialization failed: {error}"))
    })?;
    if matches!(modifier.modifier_type.as_str(), "shrinkwrap" | "curve") {
        resolved.runtime.target_mesh = Some(operand);
    } else {
        resolved.runtime.operand_mesh = Some(operand);
    }
    *mesh = crate::geom::modifiers::evaluate_modifiers(mesh, &[resolved])?;
    Ok(())
}

pub(crate) fn apply_scene_operand_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
) -> Result<()> {
    if modifier.modifier_type == "warp" {
        return apply_scene_warp_modifier(mesh, modifier, subject_id, doc, world_matrices);
    }
    if modifier.modifier_type == "hook" {
        return apply_scene_hook_modifier(mesh, modifier, subject_id, doc, world_matrices);
    }
    if matches!(
        modifier.modifier_type.as_str(),
        "boolean" | "shrinkwrap" | "curve"
    ) && !modifier.params.contains_key("collection")
    {
        return apply_generic_scene_mesh_operand_modifier(
            mesh,
            modifier,
            subject_id,
            doc,
            world_matrices,
            evaluated_meshes,
        );
    }
    let mut object_ids = BTreeSet::new();
    let mut collection_id = None;
    let object_parameter = if modifier.modifier_type == "shrinkwrap" {
        "target"
    } else {
        "object"
    };
    if let Some(value) = modifier.params.get(object_parameter) {
        let text = value.as_str().ok_or_else(|| {
            PotError::invalid_argument(format!(
                "modifier `{}` `{object_parameter}` must be an object ID",
                modifier.modifier_type
            ))
        })?;
        let object_id = Id::new(text).map_err(|_| {
            PotError::invalid_argument(format!(
                "modifier `{}` `{object_parameter}` must be a valid object ID",
                modifier.modifier_type
            ))
        })?;
        object_ids.insert(object_id);
    } else if modifier.modifier_type == "shrinkwrap"
        && let Some(value) = modifier.params.get("object")
    {
        let text = value.as_str().ok_or_else(|| {
            PotError::invalid_argument("shrinkwrap `object` must be an object ID")
        })?;
        object_ids.insert(Id::new(text).map_err(|_| {
            PotError::invalid_argument("shrinkwrap `object` must be a valid object ID")
        })?);
    }
    if let Some(value) = modifier.params.get("collection") {
        let text = value.as_str().ok_or_else(|| {
            PotError::invalid_argument("modifier `collection` must be a collection ID")
        })?;
        collection_id = Some(Id::new(text).map_err(|_| {
            PotError::invalid_argument("modifier `collection` must be a valid collection ID")
        })?);
    }
    if !object_ids.is_empty() && collection_id.is_some() {
        return Err(PotError::invalid_argument(
            "modifier operand must specify either an object or a collection",
        ));
    }
    if let Some(collection) = collection_id {
        collect_modifier_collection_objects(
            doc,
            &collection,
            &mut BTreeSet::new(),
            &mut object_ids,
        )?;
    }
    if object_ids.is_empty() {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            format!(
                "modifier `{}` requires an object or collection operand",
                modifier.modifier_type
            ),
            json!({"modifier_id":modifier.id,"pointer":format!("/params/{object_parameter}")}),
        ));
    }
    let subject_world = world_matrices.get(subject_id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "modifier subject world matrix is missing",
        )
    })?;
    let subject_inverse = subject_world.inverse();
    if !subject_inverse.is_finite() {
        return Err(PotError::invalid_argument(
            "modifier subject transform must be invertible",
        ));
    }
    let mut combined: Option<Mesh> = None;
    for object_id in object_ids {
        let object = doc.nodes.get(&object_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                format!("modifier operand object `{object_id}` was not found"),
                json!({"modifier_id":modifier.id,"object":object_id}),
            )
        })?;
        let data_id = object.data.as_ref().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                format!("modifier operand object `{object_id}` has no geometry"),
                json!({"modifier_id":modifier.id,"object":object_id}),
            )
        })?;
        let data = doc.data_blocks.get(data_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                format!("modifier operand data `{data_id}` was not found"),
                json!({"modifier_id":modifier.id,"data":data_id}),
            )
        })?;
        let mut operand = if let Some(evaluated) = evaluated_meshes.get(&object_id) {
            evaluated.clone()
        } else {
            match (modifier.modifier_type.as_str(), data.data_type.as_str()) {
                (_, "mesh") => data.mesh.clone().ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "mesh modifier operand has no mesh payload",
                        json!({"modifier_id":modifier.id,"object":object_id}),
                    )
                })?,
                ("curve", "curve") => {
                    crate::geom::curve::evaluate_curve(data.curve.as_ref().ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "curve modifier operand has no curve payload",
                            json!({"modifier_id":modifier.id,"object":object_id}),
                        )
                    })?)
                    .map_err(|error| {
                        PotError::with_details(
                            ErrorCode::InvalidArgument,
                            format!("curve modifier operand is invalid: {error}"),
                            json!({"modifier_id":modifier.id,"object":object_id}),
                        )
                    })?
                }
                _ => {
                    return Err(PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        "scene modifier operands require mesh objects (Curve accepts curve objects)",
                        json!({"feature_id":format!("modifier.{}",modifier.modifier_type),"modifier_id":modifier.id,"object":object_id}),
                    ));
                }
            }
        };
        let object_world = world_matrices.get(&object_id).copied().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "modifier operand world matrix is missing",
            )
        })?;
        let to_subject = subject_inverse * object_world;
        for vertex in &mut operand.vertices {
            vertex.co = to_subject.transform_point3(vertex.co);
        }
        combined = Some(match combined {
            None => operand,
            Some(existing) => {
                combine_scene_operand_meshes(&existing, &operand, &modifier.modifier_type)?
            }
        });
    }
    let mut resolved = modifier.clone();
    resolved.params.remove("object");
    resolved.params.remove("collection");
    resolved.params.remove("target");
    let operand =
        combined.ok_or_else(|| PotError::invalid_argument("modifier operand is empty"))?;
    let operand = serde_json::to_value(operand).map_err(|error| {
        PotError::invalid_argument(format!("modifier operand serialization failed: {error}"))
    })?;
    if matches!(modifier.modifier_type.as_str(), "shrinkwrap" | "curve") {
        resolved.runtime.target_mesh = Some(operand);
    } else {
        resolved.runtime.operand_mesh = Some(operand);
    }
    *mesh = crate::geom::modifiers::evaluate_modifiers(mesh, &[resolved])?;
    Ok(())
}
fn apply_scene_warp_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
) -> Result<()> {
    let subject_world = world_matrices.get(subject_id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "warp subject world matrix is missing",
        )
    })?;
    let subject_inverse = subject_world.inverse();
    if !subject_inverse.is_finite() {
        return Err(PotError::invalid_argument(
            "warp subject transform must be invertible",
        ));
    }
    let mut resolved = modifier.clone();
    for (source_parameter, is_from) in [("object_from", true), ("object_to", false)] {
        let object_id = modifier
            .params
            .get(source_parameter)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidArgument,
                    format!("warp modifier requires `{source_parameter}`"),
                    json!({
                        "modifier_id":modifier.id,
                        "pointer":format!("/params/{source_parameter}")
                    }),
                )
            })?;
        let object_id = Id::new(object_id).map_err(|_| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                format!("warp `{source_parameter}` must be a valid object ID"),
                json!({
                    "modifier_id":modifier.id,
                    "pointer":format!("/params/{source_parameter}")
                }),
            )
        })?;
        if !doc.nodes.contains_key(&object_id) {
            return Err(PotError::with_details(
                ErrorCode::TargetNotFound,
                format!("warp object `{object_id}` was not found"),
                json!({"modifier_id":modifier.id,"object":object_id}),
            ));
        }
        let world = world_matrices.get(&object_id).copied().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "warp object world matrix is missing",
            )
        })?;
        let point = subject_inverse.transform_point3(world.transform_point3(DVec3::ZERO));
        resolved.params.remove(source_parameter);
        if is_from {
            resolved.runtime.warp_from = Some(point.to_array());
        } else {
            resolved.runtime.warp_to = Some(point.to_array());
        }
    }
    *mesh = crate::geom::modifiers::evaluate_modifiers(mesh, &[resolved])?;
    Ok(())
}

fn apply_scene_hook_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
) -> Result<()> {
    let object_id = modifier
        .params
        .get("object")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                "hook modifier requires an object",
                json!({"modifier_id":modifier.id,"pointer":"/params/object"}),
            )
        })?;
    let object_id = Id::new(object_id).map_err(|_| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            "hook object must be a valid object ID",
            json!({"modifier_id":modifier.id,"pointer":"/params/object"}),
        )
    })?;
    if !doc.nodes.contains_key(&object_id) {
        return Err(PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("hook object `{object_id}` was not found"),
            json!({"modifier_id":modifier.id,"object":object_id}),
        ));
    }
    let subject_world = world_matrices.get(subject_id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "hook subject world matrix is missing",
        )
    })?;
    let subject_inverse = subject_world.inverse();
    let object_world = world_matrices.get(&object_id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "hook object world matrix is missing",
        )
    })?;
    let position = subject_inverse.transform_point3(object_world.transform_point3(DVec3::ZERO));
    let mut resolved = modifier.clone();
    resolved.params.remove("object");
    resolved.runtime.hook_target_position = Some(position.to_array());
    *mesh = crate::geom::modifiers::evaluate_modifiers(mesh, &[resolved])?;
    Ok(())
}
fn collect_modifier_collection_objects(
    doc: &SceneDoc,
    collection_id: &Id,
    visited: &mut BTreeSet<Id>,
    object_ids: &mut BTreeSet<Id>,
) -> Result<()> {
    if !visited.insert(collection_id.clone()) {
        return Ok(());
    }
    let collection = doc.collections.get(collection_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("modifier operand collection `{collection_id}` was not found"),
            json!({"collection":collection_id}),
        )
    })?;
    object_ids.extend(collection.objects.iter().cloned());
    for child in &collection.children {
        collect_modifier_collection_objects(doc, child, visited, object_ids)?;
    }
    Ok(())
}
fn combine_scene_operand_meshes(first: &Mesh, second: &Mesh, modifier_type: &str) -> Result<Mesh> {
    if modifier_type == "boolean" {
        return crate::geom::boolean::boolean_mesh(first, second, "UNION");
    }
    let mut result = Mesh::new();
    for source in [first, second] {
        let mut vertex_ids = std::collections::HashMap::with_capacity(source.vertices.len());
        for vertex in &source.vertices {
            let id = result.insert_vertex(vertex.co).map_err(|error| {
                PotError::invalid_argument(format!("could not combine modifier operands: {error}"))
            })?;
            vertex_ids.insert(vertex.id, id);
        }
        for face in &source.faces {
            let vertices = face
                .vertices
                .iter()
                .map(|id| {
                    vertex_ids.get(id).copied().ok_or_else(|| {
                        PotError::invalid_argument(
                            "modifier operand face references a missing vertex",
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            result
                .insert_face(vertices, face.material_index)
                .map_err(|error| {
                    PotError::invalid_argument(format!(
                        "could not combine modifier operand faces: {error}"
                    ))
                })?;
        }
        for edge in &source.edges {
            let first = vertex_ids.get(&edge.vertices[0]).copied().ok_or_else(|| {
                PotError::invalid_argument("modifier operand edge references a missing vertex")
            })?;
            let second = vertex_ids.get(&edge.vertices[1]).copied().ok_or_else(|| {
                PotError::invalid_argument("modifier operand edge references a missing vertex")
            })?;
            if !result.edges.iter().any(|candidate| {
                (candidate.vertices[0] == first && candidate.vertices[1] == second)
                    || (candidate.vertices[0] == second && candidate.vertices[1] == first)
            }) {
                result.insert_edge([first, second]).map_err(|error| {
                    PotError::invalid_argument(format!(
                        "could not combine modifier operand edges: {error}"
                    ))
                })?;
            }
        }
    }
    result.attributes.clone_from(&first.attributes);
    result.validate().map_err(|error| {
        PotError::invalid_argument(format!("combined modifier operand is invalid: {error}"))
    })?;
    Ok(result)
}
fn volume_bounds(volume: &crate::geom::volume::VolumeData, matrix: DMat4) -> Result<Option<Aabb>> {
    if let crate::geom::volume::VolumeSource::File(source) = &volume.source {
        let (min, max) = if source.format == "vdb" {
            let decoded = volume.decoded_vdb.as_ref().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::DependencyMissing,
                    "VDB grid data was not resolved before bounds evaluation",
                    json!({"feature_id":"volume.openvdb_evaluation.resource"}),
                )
            })?;
            let grid = decoded.density_grid().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "VDB archive contains no volume grid",
                    json!({"feature_id":"volume.openvdb_evaluation"}),
                )
            })?;
            let Some((minimum, maximum)) = grid.active_bbox_world() else {
                return Ok(None);
            };
            (minimum, maximum)
        } else {
            let (Some(minimum), Some(maximum)) = (source.bounds_min, source.bounds_max) else {
                return Ok(None);
            };
            (DVec3::from_array(minimum), DVec3::from_array(maximum))
        };
        if !min.is_finite() || !max.is_finite() || (0..3).any(|axis| min[axis] > max[axis]) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "file-backed volume bounds are invalid",
            ));
        }
        let mut bounds: Option<Aabb> = None;
        for corner in 0..8 {
            let local = DVec3::new(
                if corner & 1 == 0 { min.x } else { max.x },
                if corner & 2 == 0 { min.y } else { max.y },
                if corner & 4 == 0 { min.z } else { max.z },
            );
            let world = matrix.transform_point3(local);
            if !world.is_finite() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "file-backed volume world bounds are not finite",
                ));
            }
            bounds = Some(match bounds {
                Some(current) => Aabb {
                    min: current.min.min(world),
                    max: current.max.max(world),
                },
                None => Aabb {
                    min: world,
                    max: world,
                },
            });
        }
        return Ok(bounds);
    }
    crate::geom::volume::validate_data(volume)?;
    let mut bounds: Option<Aabb> = None;
    for grid in &volume.grids {
        if grid.dims.contains(&0)
            || !grid.voxel_size.is_finite()
            || grid.voxel_size <= 0.0
            || !grid.origin.is_finite()
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "volume grid bounds are invalid",
            ));
        }
        let half_voxel = match &volume.source {
            crate::geom::volume::VolumeSource::Generated(source)
                if source.algorithm == "mesh_to_volume_fog" =>
            {
                DVec3::splat(grid.voxel_size * 0.5)
            }
            _ => DVec3::ZERO,
        };
        let local_min = grid.origin - half_voxel;
        let local_max = grid.origin
            + grid.voxel_size
                * DVec3::new(
                    f64::from(grid.dims[0] - 1),
                    f64::from(grid.dims[1] - 1),
                    f64::from(grid.dims[2] - 1),
                )
            + half_voxel;
        if !local_min.is_finite() || !local_max.is_finite() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "volume grid bounds are not finite",
            ));
        }
        for corner in 0..8 {
            let local = DVec3::new(
                if corner & 1 == 0 {
                    local_min.x
                } else {
                    local_max.x
                },
                if corner & 2 == 0 {
                    local_min.y
                } else {
                    local_max.y
                },
                if corner & 4 == 0 {
                    local_min.z
                } else {
                    local_max.z
                },
            );
            let world = matrix.transform_point3(local);
            if !world.is_finite() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "volume world bounds are not finite",
                ));
            }
            bounds = Some(match bounds {
                Some(current) => Aabb {
                    min: current.min.min(world),
                    max: current.max.max(world),
                },
                None => Aabb {
                    min: world,
                    max: world,
                },
            });
        }
    }
    Ok(bounds)
}

fn apply_lattice_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    source_node_id: &Id,
    source_data: &DataBlock,
    doc: &SceneDoc,
    matrices: &BTreeMap<Id, DMat4>,
) -> Result<()> {
    let lattice_text = modifier
        .params
        .get("object")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                "lattice modifier requires an object node ID",
                json!({"modifier_id":modifier.id,"parameter":"object"}),
            )
        })?;
    let lattice_id = Id::new(lattice_text.to_owned()).map_err(|_| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            "lattice modifier target is not a valid node ID",
            json!({"modifier_id":modifier.id,"parameter":"object"}),
        )
    })?;
    let lattice_node = doc.nodes.get(&lattice_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("lattice node `{lattice_id}` was not found"),
            json!({"modifier_id":modifier.id,"object":lattice_id}),
        )
    })?;
    if lattice_node.kind != "lattice" {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "lattice modifier target must be a lattice node",
            json!({"modifier_id":modifier.id,"object":lattice_id}),
        ));
    }
    let lattice_data_id = lattice_node
        .data
        .as_ref()
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "lattice node has no data block"))?;
    let lattice_data_block = doc.data_blocks.get(lattice_data_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "lattice node data block is missing",
            json!({"data":lattice_data_id}),
        )
    })?;
    if lattice_data_block.data_type != "lattice" {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "lattice node references a non-lattice data block",
            json!({"data":lattice_data_id}),
        ));
    }
    let lattice = lattice_data_block.lattice.as_ref().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "lattice data block has no lattice payload",
        )
    })?;
    let source_world = matrices.get(source_node_id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "source mesh world matrix is missing",
        )
    })?;
    let lattice_world = matrices.get(&lattice_id).copied().ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "lattice world matrix is missing")
    })?;
    let source_inverse = source_world.inverse();
    if !source_inverse.is_finite() {
        return Err(PotError::invalid_argument(
            "lattice modifier source transform must be invertible",
        ));
    }
    let lattice_to_mesh = source_inverse * lattice_world;

    let weights = if let Some(group_name) = modifier
        .params
        .get("vertex_group")
        .and_then(serde_json::Value::as_str)
    {
        let mut matches = source_data
            .vertex_groups
            .iter()
            .filter(|group| group.id.as_str() == group_name || group.name == group_name);
        let group = matches.next().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                format!("vertex group `{group_name}` was not found"),
                json!({"modifier_id":modifier.id,"vertex_group":group_name}),
            )
        })?;
        if matches.next().is_some() {
            return Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                format!("vertex group name `{group_name}` is ambiguous"),
                json!({"modifier_id":modifier.id,"vertex_group":group_name}),
            ));
        }
        let group_id = &group.id;
        Some(
            mesh.vertices
                .iter()
                .map(|vertex| {
                    source_data
                        .vertex_weights
                        .get(&vertex.id)
                        .and_then(|groups| groups.get(group_id))
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect::<Vec<_>>(),
        )
    } else if modifier.params.contains_key("vertex_group") {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "lattice modifier vertex_group must be a group ID or name",
            json!({"modifier_id":modifier.id,"parameter":"vertex_group"}),
        ));
    } else {
        None
    };
    *mesh = crate::geom::lattice::deform_mesh_with_transform(
        mesh,
        lattice,
        weights.as_deref(),
        lattice_to_mesh,
    )?;
    Ok(())
}

pub(crate) fn apply_volume_to_mesh_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    source_node_id: &Id,
    doc: &SceneDoc,
    matrices: &BTreeMap<Id, DMat4>,
    volume_data: &BTreeMap<Id, crate::geom::volume::VolumeData>,
) -> Result<()> {
    let volume_text = modifier
        .params
        .get("object")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                "volume_to_mesh modifier requires a volume node ID",
                json!({"modifier_id":modifier.id,"parameter":"object"}),
            )
        })?;
    let volume_id = Id::new(volume_text.to_owned()).map_err(|_| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            "volume_to_mesh target is not a valid node ID",
            json!({"modifier_id":modifier.id,"parameter":"object"}),
        )
    })?;
    let node = doc.nodes.get(&volume_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("volume node `{volume_id}` was not found"),
            json!({"modifier_id":modifier.id,"object":volume_id}),
        )
    })?;
    if node.kind != "volume" {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "volume_to_mesh target must be a volume node",
            json!({"modifier_id":modifier.id,"object":volume_id}),
        ));
    }
    let data_id = node
        .data
        .as_ref()
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "volume node has no data block"))?;
    let data_block = doc.data_blocks.get(data_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "volume node data block is missing",
            json!({"data":data_id}),
        )
    })?;
    if data_block.data_type != "volume" {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "volume node references a non-volume data block",
            json!({"data":data_id}),
        ));
    }
    let volume = volume_data
        .get(data_id)
        .or(data_block.volume.as_ref())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "volume data block has no volume payload",
            )
        })?;
    let threshold = match modifier.params.get("threshold").or_else(|| {
        crate::params::default_value(
            crate::params::ParameterFamily::Modifier,
            "volume_to_mesh",
            "threshold",
        )
    }) {
        None => 0.1,
        Some(value) => value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidArgument,
                    "volume_to_mesh threshold must be finite",
                    json!({"modifier_id":modifier.id,"parameter":"threshold"}),
                )
            })?,
    };
    let source_world = matrices.get(source_node_id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "source mesh world matrix is missing",
        )
    })?;
    let volume_world = matrices
        .get(&volume_id)
        .copied()
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "volume world matrix is missing"))?;
    let source_inverse = source_world.inverse();
    if !source_inverse.is_finite() {
        return Err(PotError::invalid_argument(
            "volume_to_mesh source transform must be invertible",
        ));
    }
    let volume_to_mesh = source_inverse * volume_world;
    let mut extracted = crate::geom::volume::volume_to_mesh(volume, threshold)?;
    for vertex in &mut extracted.vertices {
        vertex.co = volume_to_mesh.transform_point3(vertex.co);
        if !vertex.co.is_finite() {
            return Err(PotError::invalid_argument(
                "volume_to_mesh produced non-finite mesh coordinates",
            ));
        }
    }
    *mesh = extracted;
    Ok(())
}

const INSTANCE_COLLECTION_PROPERTY: &str = "instance_collection";

fn instance_collection_target(node: &Node) -> Result<Option<Id>> {
    if node.kind != "collection_instance" {
        return Ok(None);
    }
    let value = node
        .properties
        .get(INSTANCE_COLLECTION_PROPERTY)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "collection instance node has no collection target",
            )
        })?;
    let collection = value.as_str().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "collection instance target must be a collection ID",
        )
    })?;
    collection.parse().map(Some).map_err(|_| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "collection instance target is not a valid ID",
        )
    })
}

fn evaluate_collection_instance(
    doc: &SceneDoc,
    instance_id: &Id,
    matrices: &BTreeMap<Id, DMat4>,
    meshes: &BTreeMap<Id, Mesh>,
    active_instances: &mut BTreeSet<Id>,
) -> Result<(Mesh, Vec<Vec<Id>>)> {
    let instance_node = doc.nodes.get(instance_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "collection instance node is missing",
        )
    })?;
    let target = instance_collection_target(instance_node)?.ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "node has no collection instance target",
        )
    })?;
    if !active_instances.insert(instance_id.clone()) {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "collection instance cycle detected",
            json!({ "node_id": instance_id, "collection": target }),
        ));
    }
    let mut mesh = Mesh::new();
    let mut paths = Vec::new();
    append_collection_geometry(
        doc,
        instance_id,
        &target,
        matrices,
        meshes,
        active_instances,
        &mut BTreeSet::new(),
        &mut BTreeSet::new(),
        &mut mesh,
        &mut paths,
    )?;
    active_instances.remove(instance_id);
    Ok((mesh, paths))
}

#[expect(
    clippy::too_many_arguments,
    reason = "keeps recursive instance traversal state explicit"
)]
fn append_collection_geometry(
    doc: &SceneDoc,
    instance_id: &Id,
    collection_id: &Id,
    matrices: &BTreeMap<Id, DMat4>,
    meshes: &BTreeMap<Id, Mesh>,
    active_instances: &mut BTreeSet<Id>,
    visited_collections: &mut BTreeSet<Id>,
    collection_stack: &mut BTreeSet<Id>,
    output: &mut Mesh,
    paths: &mut Vec<Vec<Id>>,
) -> Result<()> {
    if collection_stack.contains(collection_id) {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "collection hierarchy cycle detected during instance evaluation",
            json!({ "collection": collection_id }),
        ));
    }
    if !visited_collections.insert(collection_id.clone()) {
        return Ok(());
    }
    let collection = doc.collections.get(collection_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "collection instance target does not exist",
            json!({ "collection": collection_id, "node_id": instance_id }),
        )
    })?;
    collection_stack.insert(collection_id.clone());
    for object_id in &collection.objects {
        let source_node = doc.nodes.get(object_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "collection references a missing node",
                json!({ "collection": collection_id, "node_id": object_id }),
            )
        })?;
        if !source_node.visible || !source_node.render_visible {
            continue;
        }
        let (nested_mesh, nested_paths) = if instance_collection_target(source_node)?.is_some() {
            evaluate_collection_instance(doc, object_id, matrices, meshes, active_instances)?
        } else if let Some(source_mesh) = meshes.get(object_id) {
            (source_mesh.clone(), vec![vec![object_id.clone()]])
        } else {
            (Mesh::new(), Vec::new())
        };
        if !nested_mesh.vertices.is_empty() {
            let source_matrix = matrices.get(object_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "collection member world matrix is missing",
                )
            })?;
            append_transformed_mesh(output, &nested_mesh, source_matrix)?;
            for path in nested_paths {
                let mut full_path = Vec::with_capacity(path.len() + 1);
                full_path.push(instance_id.clone());
                full_path.extend(path);
                paths.push(full_path);
            }
        }
    }
    for child_id in &collection.children {
        append_collection_geometry(
            doc,
            instance_id,
            child_id,
            matrices,
            meshes,
            active_instances,
            visited_collections,
            collection_stack,
            output,
            paths,
        )?;
    }
    collection_stack.remove(collection_id);
    Ok(())
}

fn append_transformed_mesh(output: &mut Mesh, source: &Mesh, matrix: DMat4) -> Result<()> {
    if output.vertices.is_empty() {
        output.attributes.clone_from(&source.attributes);
    }
    let mut vertex_ids = BTreeMap::new();
    for vertex in &source.vertices {
        let id = output
            .insert_vertex(matrix.transform_point3(vertex.co))
            .map_err(|error| PotError::new(ErrorCode::EvaluationFailed, error.to_string()))?;
        vertex_ids.insert(vertex.id, id);
    }
    for edge in &source.edges {
        let first = vertex_ids.get(&edge.vertices[0]).copied().ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "instance edge references a missing vertex",
            )
        })?;
        let second = vertex_ids.get(&edge.vertices[1]).copied().ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "instance edge references a missing vertex",
            )
        })?;
        output
            .insert_edge([first, second])
            .map_err(|error| PotError::new(ErrorCode::EvaluationFailed, error.to_string()))?;
    }
    for face in &source.faces {
        let vertices = face
            .vertices
            .iter()
            .map(|vertex_id| {
                vertex_ids.get(vertex_id).copied().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "instance face references a missing vertex",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        output
            .insert_face(vertices, face.material_index)
            .map_err(|error| PotError::new(ErrorCode::EvaluationFailed, error.to_string()))?;
    }
    Ok(())
}

fn evaluate_grease_pencil(
    data: &crate::model::GreasePencilData,
    materials: &crate::model::Registry<crate::model::Material>,
    matrix: DMat4,
    frame: f64,
) -> Result<Vec<EvaluatedStroke>> {
    let scale = matrix
        .x_axis
        .truncate()
        .length()
        .max(matrix.y_axis.truncate().length())
        .max(matrix.z_axis.truncate().length());
    let mut output = Vec::new();
    for layer in data
        .layers
        .iter()
        .filter(|layer| layer.visible && layer.opacity > 0.0)
    {
        let Some(keyframe) = layer
            .frames
            .iter()
            .filter(|keyframe| keyframe.frame <= frame)
            .max_by(|left, right| left.frame.total_cmp(&right.frame))
        else {
            continue;
        };
        for stroke in &keyframe.strokes {
            if stroke.points.len() < 2 {
                continue;
            }
            let material_color = stroke
                .material
                .as_ref()
                .and_then(|id| materials.get(id))
                .map_or([0.6, 0.6, 0.6, 1.0], |material| material.base_color);
            let point_count = u32::try_from(stroke.points.len()).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "Grease Pencil stroke has too many points",
                )
            })?;
            let mean_opacity = stroke.points.iter().map(|point| point.opacity).sum::<f64>()
                / f64::from(point_count);
            let color = [
                material_color[0],
                material_color[1],
                material_color[2],
                material_color[3] * layer.opacity * mean_opacity,
            ];
            let points_world = stroke
                .points
                .iter()
                .map(|point| {
                    let position = glam::DVec3::from_array(point.position);
                    matrix.transform_point3(position).to_array()
                })
                .collect::<Vec<_>>();
            let radius = stroke
                .points
                .iter()
                .map(|point| point.radius * scale)
                .collect::<Vec<_>>();
            if color.iter().any(|value| !value.is_finite())
                || points_world
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
                || radius.iter().any(|value| !value.is_finite())
            {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    "Grease Pencil stroke evaluates to non-finite values",
                ));
            }
            output.push(EvaluatedStroke {
                points_world,
                radius,
                color,
                cyclic: stroke.cyclic,
            });
        }
    }
    Ok(output)
}

fn mesh_bounds(mesh: &Mesh, matrix: DMat4) -> Result<Option<Aabb>> {
    mesh.validate().map_err(|error| {
        PotError::with_details(ErrorCode::SceneInvalid, error.to_string(), json!({}))
    })?;
    let mut positions = mesh
        .vertices
        .iter()
        .map(|vertex| matrix.transform_point3(vertex.co));
    let Some(first) = positions.next() else {
        return Ok(None);
    };
    if !first.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "evaluated mesh contains non-finite positions",
        ));
    }
    let (min, max) = positions
        .try_fold((first, first), |(min, max), point| {
            point
                .is_finite()
                .then_some((min.min(point), max.max(point)))
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "evaluated mesh contains non-finite positions",
            )
        })?;
    Ok(Some(Aabb { min, max }))
}

pub(crate) fn attach_vertex_group_weights(mesh: &mut Mesh, data_block: &DataBlock) {
    if data_block.vertex_groups.is_empty() {
        return;
    }
    let mut groups = serde_json::Map::new();
    for group in &data_block.vertex_groups {
        let mut weights = serde_json::Map::new();
        for (vertex_id, groups_for_vertex) in &data_block.vertex_weights {
            if let Some(weight) = groups_for_vertex.get(&group.id) {
                weights.insert(format!("v{vertex_id}"), json!(weight));
            }
        }
        groups.insert(group.name.clone(), serde_json::Value::Object(weights));
    }
    mesh.attributes.insert(
        "vertex_groups".to_owned(),
        serde_json::Value::Object(groups),
    );
}
#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use glam::DVec3;

    use crate::{
        geom::{BoxParams, Mesh},
        model::{DataBlock, Id, Node, SceneDoc, Transform},
    };

    use super::{EvaluationContext, Snapshot};

    #[test]
    fn evaluates_world_matrix_and_transformed_mesh_bounds() {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let data_id = Id::new("cube_mesh").unwrap();
        let node_id = Id::new("cube").unwrap();
        let mesh = Mesh::box_mesh(BoxParams {
            size: DVec3::splat(2.0),
        })
        .unwrap();
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(mesh),
                camera: None,
                light: None,
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Cube".to_owned(),
                kind: "mesh".to_owned(),
                primitive: Some("box".to_owned()),
                tags: Vec::new(),
                parent: None,
                parent_inverse: None,
                transform: Transform::from_rotation_deg(
                    [1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
                    [2.0, 1.0, 1.0],
                ),
                data: Some(data_id),
                materials: Vec::new(),
                modifiers: Vec::new(),
                visible: true,
                render_visible: true,
                selectable: true,
                action: None,
                properties: serde_json::Map::new(),
                rigid_body: None,
                force_field: None,
                ..Node::default()
            },
        );
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
        let evaluated = snapshot.nodes.get(&node_id).unwrap();
        let bounds = evaluated.bounds.unwrap();
        assert_eq!(bounds.min, DVec3::new(-1.0, 1.0, 2.0));
        assert_eq!(bounds.max, DVec3::new(3.0, 3.0, 4.0));
        assert_eq!(evaluated.dimensions, Some([4.0, 2.0, 2.0]));
        let local_mesh = snapshot.meshes.get(&node_id).unwrap();
        let local_bounds = local_mesh.bounds().unwrap();
        assert_eq!(local_bounds.min, DVec3::splat(-1.0));
        assert_eq!(local_bounds.max, DVec3::splat(1.0));
        assert!(
            serde_json::to_value(&snapshot)
                .unwrap()
                .get("meshes")
                .is_none()
        );
    }

    #[test]
    fn rejects_missing_parent_and_parent_cycles() {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let node_id = Id::new("orphan").unwrap();
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Orphan".to_owned(),
                kind: "empty".to_owned(),
                primitive: None,
                tags: Vec::new(),
                parent: Some(Id::new("missing").unwrap()),
                parent_inverse: None,
                transform: Transform::default(),
                data: None,
                materials: Vec::new(),
                modifiers: Vec::new(),
                visible: true,
                render_visible: true,
                selectable: true,
                action: None,
                properties: serde_json::Map::new(),
                rigid_body: None,
                force_field: None,
                ..Node::default()
            },
        );
        assert!(Snapshot::evaluate(&doc, &EvaluationContext::default()).is_err());

        let child_id = Id::new("child").unwrap();
        let mut child = doc.nodes.get(&node_id).cloned().unwrap();
        child.name = "Child".to_owned();
        child.parent = Some(node_id.clone());
        doc.nodes.get_mut(&node_id).unwrap().parent = Some(child_id.clone());
        doc.nodes.insert(child_id, child);
        assert!(Snapshot::evaluate(&doc, &EvaluationContext::default()).is_err());
    }
}
