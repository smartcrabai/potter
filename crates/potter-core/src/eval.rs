use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::Path,
    sync::Arc,
};

use glam::{DMat4, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Aabb, CylinderParams, IcosphereParams, Mesh,
        concat::{Part, concatenate, concatenate_with_normals},
        modifiers::{AdaptiveRenderCamera, AdaptiveRenderContext},
        normals::SHADING_ATTRIBUTES,
        uv_layers::{self, UvLayer},
    },
    hash,
    model::{
        ArmatureData, CameraData, ConstraintType, DataBlock, Id, Modifier, Node, Scene, SceneDoc,
        TextureCoordinateMapping, VertexGroup,
    },
    params::{self, ParameterFamily},
};

/// Blender's default for a `mesh_to_volume` parameter, read from the schema table.
fn mesh_to_volume_default(parameter: &str) -> Result<&'static Value> {
    params::default_value(ParameterFamily::Modifier, "mesh_to_volume", parameter).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            format!("mesh_to_volume has no schema default for `{parameter}`"),
        )
    })
}

pub mod animation;
mod ik_solver;
pub mod rig;
const EVALUATION_ENGINE: &str = "potter-eval-5";

/// A render camera explicitly chosen by a preview view rather than a scene camera node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveCameraOverride {
    pub data: CameraData,
    pub world_to_camera: [f64; 16],
}

/// Camera-dependent dicing inputs attached only to native path render requests.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveRenderRequest {
    pub camera_id: Option<Id>,
    pub camera_override: Option<AdaptiveCameraOverride>,
    pub resolution: [u32; 2],
    /// Preview or final-render dicing rate, selected at the existing render boundary.
    pub dicing_rate: f64,
}

/// Which stack an evaluation serves, following Blender's depsgraph modes: viewport previews use
/// `show_viewport`, and saved renders use `show_render`. Each export format chooses its purpose
/// explicitly, following that format's Blender default; the purpose is not implied by "export".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationPurpose {
    #[default]
    Preview,
    Render,
}

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
    #[serde(default)]
    pub adaptive_render: Option<AdaptiveRenderRequest>,
    /// Modifier stack to evaluate; the viewport preview stack when omitted.
    #[serde(default)]
    pub purpose: EvaluationPurpose,
}

/// Borrowed project resources for one modifier evaluation.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ModifierEvaluationContext<'a> {
    pub(crate) purpose: EvaluationPurpose,
    pub(crate) project_root: Option<&'a Path>,
    pub(crate) cache_directory: Option<&'a Path>,
    /// The owning object's material slot count (`Node::material_slot_count`); `None` when there is
    /// no scene object, in which case consumers fall back to the mesh's own face material indices.
    pub(crate) material_slot_count: Option<usize>,
}
fn adaptive_render_context<'a>(
    doc: &'a SceneDoc,
    scene: &'a Scene,
    matrices: &BTreeMap<Id, DMat4>,
    request: &'a AdaptiveRenderRequest,
) -> Result<AdaptiveRenderContext<'a>> {
    if request.resolution.contains(&0)
        || !request.dicing_rate.is_finite()
        || request.dicing_rate <= 0.0
    {
        return Err(PotError::invalid_argument(
            "adaptive render raster size and dicing rate must be positive and finite",
        ));
    }
    let camera = if let Some(camera_id) = scene
        .render
        .dicing_camera
        .as_ref()
        .or(request.camera_id.as_ref())
    {
        let node = doc
            .nodes
            .get(camera_id)
            .filter(|node| node.kind == "camera")
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "adaptive dicing camera object does not exist",
                    json!({"camera_id": camera_id}),
                )
            })?;
        let data = node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data_block| data_block.camera.as_ref())
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "adaptive dicing camera has no camera data",
                    json!({"camera_id": camera_id}),
                )
            })?;
        let world = matrices.get(camera_id).copied().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "adaptive dicing camera has no evaluated world transform",
                json!({"camera_id": camera_id}),
            )
        })?;
        let world_to_camera = world.inverse();
        if !world_to_camera.is_finite() {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "adaptive dicing camera transform is singular",
                json!({"camera_id": camera_id}),
            ));
        }
        Some(AdaptiveRenderCamera {
            data,
            world_to_camera,
        })
    } else {
        request
            .camera_override
            .as_ref()
            .map(|camera| AdaptiveRenderCamera {
                data: &camera.data,
                world_to_camera: DMat4::from_cols_array(&camera.world_to_camera),
            })
    };
    Ok(AdaptiveRenderContext {
        settings: &scene.render,
        resolution: request.resolution,
        dicing_rate: request.dicing_rate,
        camera,
    })
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
    /// Effective material slots after the node's modifier stack: each slot resolved under its OBJECT
    /// or DATA link, or replaced by a Boolean TRANSFER pool. Face material indices refer to it;
    /// out-of-range indices clamp to its last entry (`BKE_object_material_get_eval`). Null entries are
    /// empty slots. Not serialized.
    #[serde(skip)]
    pub materials: Vec<Option<Id>>,
    /// Blender `me->mat` after the modifier stack: the evaluated mesh's own material array, padded to
    /// the evaluated count. Apply writes it back to DATA, and a quick Boolean operand hands it to its
    /// subject. Not serialized.
    #[serde(skip)]
    pub mesh_materials: Vec<Option<Id>>,
}

/// Evaluated world-space Grease Pencil stroke used by vector exporters and preview.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatedStroke {
    pub points_world: Vec<[f64; 3]>,
    pub radius: Vec<f64>,
    pub color: [f64; 4],
    pub cyclic: bool,
}

/// Per-member UV selection and availability in the joined layer-name order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UvBinding {
    /// Joined layer index of this member's source render layer, when present.
    pub render_index: Option<u8>,
    /// Bit `n` is set when this member supplied joined UV layer `n`.
    pub available_mask: u8,
    /// Whether every face using this slot has a real row in joined layer zero.
    pub texcoord0_covered: bool,
}

/// Material and member-local attribute selection of one evaluated instance slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotBinding {
    /// Material the slot binds, or `None` for an empty slot.
    pub material: Option<Id>,
    /// Member-local UV layer lookup in the joined name order.
    pub uv: UvBinding,
    /// Joined color-layer index named by this member's render-color designation.
    pub color_render_layer: Option<usize>,
    /// Dynamic bitset of color layers actually supplied by this member.
    pub color_available_mask: Arc<[u64]>,
}

/// Stable evaluation of one scene document and its context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub scene_id: Id,
    pub view_layer: Option<Id>,
    pub frame: f64,
    #[serde(default)]
    pub purpose: EvaluationPurpose,
    pub revision: u64,
    pub scene_hash: String,
    pub evaluation_hash: String,
    #[serde(default)]
    pub warnings: Vec<PotError>,
    pub nodes: BTreeMap<Id, EvaluatedNode>,
    /// World transforms captured for apply-only evaluations, including simulation results.
    #[serde(skip)]
    pub(crate) transform_world_matrices: BTreeMap<Id, DMat4>,
    /// Local-space geometry after the node's enabled modifier stack.
    #[serde(skip)]
    pub meshes: BTreeMap<Id, Mesh>,
    /// Ephemeral local-space adaptive surface normals, aligned with each evaluated mesh's vertices.
    #[serde(skip)]
    pub(crate) adaptive_normals: BTreeMap<Id, Vec<Option<DVec3>>>,
    /// Evaluated armature bone-head matrices in world space.
    #[serde(skip)]
    pub bone_matrices: BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
    /// Raw armature-local pose matrices for f32 bone references, keyed by armature then bone.
    #[serde(skip)]
    pub bone_pose_matrices: BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
    /// Instance source paths recorded by geometry-node modifiers.
    #[serde(skip)]
    pub instance_paths: BTreeMap<Id, Vec<Vec<Id>>>,
    /// Member material slots and UV/color layer bindings for evaluated collection instances.
    #[serde(skip)]
    pub slot_bindings: BTreeMap<Id, Vec<SlotBinding>>,
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
    /// Member-local binding for an evaluated collection instance; out-of-range slots clamp to
    /// its last entry. Empty binding tables return `None`.
    #[must_use]
    pub fn slot_binding<'a>(
        &'a self,
        node_id: &Id,
        _node: &Node,
        slot: usize,
    ) -> Option<&'a SlotBinding> {
        self.slot_bindings
            .get(node_id)
            .and_then(|table| table.get(slot.min(table.len().checked_sub(1)?)))
    }

    /// Material for a face slot; out-of-range indices clamp to the last available slot.
    #[must_use]
    pub fn slot_material<'a>(
        &'a self,
        node_id: &Id,
        node: &'a Node,
        data: Option<&'a DataBlock>,
        slot: usize,
    ) -> Option<&'a Id> {
        if self.slot_bindings.contains_key(node_id) {
            return self
                .slot_binding(node_id, node, slot)
                .and_then(|binding| binding.material.as_ref());
        }
        match self.nodes.get(node_id) {
            Some(evaluated) => {
                let table = evaluated.materials.as_slice();
                table.get(slot.min(table.len().checked_sub(1)?))?.as_ref()
            }
            None => crate::model::resolve_material_slot(
                node.materials.as_slice(),
                data.map_or(&[][..], |data| data.materials.as_slice()),
                node.material_links.as_slice(),
                slot,
            ),
        }
    }

    /// Number of effective material slots: the evaluated table's length, or the authoring count
    /// for a node without an evaluated entry.
    #[must_use]
    pub fn slot_count(&self, node_id: &Id, node: &Node, data: Option<&DataBlock>) -> usize {
        if let Some(bindings) = self.slot_bindings.get(node_id) {
            bindings.len()
        } else if let Some(evaluated) = self.nodes.get(node_id) {
            evaluated.materials.len()
        } else {
            node.material_slot_count(data)
        }
    }
}

impl Snapshot {
    /// Evaluate world matrices, post-modifier meshes and bounds, and a reproducible key.
    ///
    /// # Errors
    ///
    /// Returns scene, evaluation, modifier, context, and serialization errors.
    pub fn evaluate(doc: &SceneDoc, context: &EvaluationContext) -> Result<Self> {
        Self::evaluate_with_cache(doc, context, None, None)
    }

    /// Evaluate the Scene and read/write deterministic simulation cache entries.
    pub fn evaluate_with_cache(
        doc: &SceneDoc,
        context: &EvaluationContext,
        project_root: Option<&Path>,
        cache_directory: Option<&Path>,
    ) -> Result<Self> {
        Self::evaluate_internal(
            doc,
            context,
            project_root,
            cache_directory,
            None,
            None,
            None,
        )
    }

    /// Evaluate only the geometry required for the selected nodes and their mesh operands.
    pub fn evaluate_nodes_with_cache(
        doc: &SceneDoc,
        context: &EvaluationContext,
        project_root: Option<&Path>,
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
        Self::evaluate_internal(
            doc,
            context,
            project_root,
            cache_directory,
            Some(node_ids),
            None,
            None,
        )
    }

    /// Evaluate selected nodes with one node's modifier stack limited to a prefix.
    #[expect(
        clippy::too_many_arguments,
        reason = "bounded Snapshot evaluation keeps its inputs explicit"
    )]
    pub(crate) fn evaluate_nodes_with_modifier_prefix_with_cache(
        doc: &SceneDoc,
        context: &EvaluationContext,
        project_root: Option<&Path>,
        cache_directory: Option<&Path>,
        node_ids: &BTreeSet<Id>,
        prefix_node_id: &Id,
        keep: usize,
        driver_values: &BTreeMap<(Id, String, u32), f64>,
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
        let prefix_node = doc.nodes.get(prefix_node_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "modifier prefix node was not found",
                json!({"node_id":prefix_node_id}),
            )
        })?;
        Self::evaluate_internal(
            doc,
            context,
            project_root,
            cache_directory,
            Some(node_ids),
            Some(ModifierPrefix {
                node_id: prefix_node_id,
                modifiers: &prefix_node.modifiers,
                keep,
            }),
            Some(driver_values),
        )
    }

    /// Evaluate only a modifier's classified geometry operands and required world transforms.
    pub(crate) fn evaluate_modifier_operands_with_cache(
        doc: &SceneDoc,
        context: &EvaluationContext,
        project_root: Option<&Path>,
        cache_directory: Option<&Path>,
        target_id: &Id,
        modifier_id: &Id,
        driver_values: &BTreeMap<(Id, String, u32), f64>,
    ) -> Result<Self> {
        let modifier_context = ModifierEvaluationContext {
            purpose: context.purpose,
            project_root,
            cache_directory,
            material_slot_count: None,
        };
        let suppressed = BTreeSet::new();
        let mut world_matrix_ids = BTreeSet::from([target_id.clone()]);
        let mesh_dependencies = object_mesh_dependencies(
            doc,
            ModifierGate {
                context: modifier_context,
                suppressed: &suppressed,
                modifier_prefix: None,
            },
            Some((target_id, modifier_id)),
        )?;
        let mut selected_nodes = BTreeSet::new();
        if let Some(dependencies) = mesh_dependencies.modifiers.get(target_id) {
            selected_nodes.extend(dependencies.iter().cloned());
        }
        if let Some(dependencies) = mesh_dependencies.other.get(target_id) {
            selected_nodes.extend(dependencies.iter().cloned());
        }
        if let Some(modifier) = doc.nodes.get(target_id).and_then(|node| {
            node.modifiers
                .iter()
                .find(|modifier| &modifier.id == modifier_id)
        }) {
            let transform_references = modifier_transform_references(modifier);
            world_matrix_ids.extend(transform_references.iter().cloned());
            for reference_id in transform_references {
                let Some(reference) = doc.nodes.get(&reference_id) else {
                    continue;
                };
                for constraint in &reference.constraints {
                    if constraint.enabled
                        && constraint.influence != 0.0
                        && matches!(
                            constraint.constraint_type,
                            ConstraintType::Shrinkwrap | ConstraintType::GeometryAttribute
                        )
                        && let Some(dependency_id) = &constraint.target
                        && mesh_dependencies
                            .other
                            .get(&reference_id)
                            .is_some_and(|dependencies| dependencies.contains(dependency_id))
                    {
                        selected_nodes.insert(dependency_id.clone());
                    }
                }
            }
        }
        doc.validate()?;
        crate::library::validate_linked_assets(doc, modifier_context.project_root)?;
        let scene_hash = source_scene_hash(doc)?;
        Self::evaluate_gated(
            doc,
            &scene_hash,
            &scene_hash,
            context,
            modifier_context,
            Some(&selected_nodes),
            &suppressed,
            &mesh_dependencies,
            Some(&world_matrix_ids),
            None,
            Some(driver_values),
        )
    }

    /// Evaluate every supported node, recording modifier-feature failures per owner.
    pub fn evaluate_available_with_cache(
        doc: &SceneDoc,
        context: &EvaluationContext,
        project_root: Option<&Path>,
        cache_directory: Option<&Path>,
    ) -> Result<(Self, BTreeMap<Id, PotError>)> {
        doc.validate()?;
        let modifier_context = ModifierEvaluationContext {
            purpose: context.purpose,
            project_root,
            cache_directory,
            material_slot_count: None,
        };
        crate::library::validate_linked_assets(doc, modifier_context.project_root)?;
        let scene_hash = source_scene_hash(doc)?;
        let mut suppressed = BTreeSet::new();
        let mesh_dependencies = object_mesh_dependencies(
            doc,
            ModifierGate {
                context: modifier_context,
                suppressed: &suppressed,
                modifier_prefix: None,
            },
            None,
        )?;
        let mut node_errors = BTreeMap::new();
        loop {
            let evaluation = Self::evaluate_gated(
                doc,
                &scene_hash,
                &scene_hash,
                context,
                modifier_context,
                None,
                &suppressed,
                &mesh_dependencies,
                None,
                None,
                None,
            );
            match evaluation {
                Ok(snapshot) if node_errors.is_empty() => {
                    return Ok((snapshot, node_errors));
                }
                Ok(mut snapshot) => {
                    snapshot.scene_hash.clone_from(&scene_hash);
                    let evaluation_key = json!({
                        "engine": EVALUATION_ENGINE,
                        "build_identity": crate::BUILD_IDENTITY,
                        "simulation_engine": crate::sim::SOLVER_VERSION,
                        "scene_hash": scene_hash,
                        "scene_id": snapshot.scene_id,
                        "view_layer": snapshot.view_layer,
                        "purpose": snapshot.purpose,
                        "frame": snapshot.frame,
                        "adaptive_render": context.adaptive_render,
                        "simulation_cache_key": snapshot.simulation_cache_key,
                    });
                    snapshot.evaluation_hash = hash::sha256(&hash::canonicalize(&evaluation_key)?);
                    for node_id in node_errors.keys() {
                        snapshot.meshes.remove(node_id);
                        snapshot.instance_paths.remove(node_id);
                        snapshot.slot_bindings.remove(node_id);
                        snapshot.adaptive_normals.remove(node_id);
                        snapshot.strokes.remove(node_id);
                    }
                    return Ok((snapshot, node_errors));
                }
                Err(error) if is_isolated_failure(&error) => {
                    let Some(node_id) = unsupported_feature_owner(doc, &error) else {
                        return Err(error);
                    };
                    if node_errors.contains_key(&node_id) {
                        return Err(error);
                    }
                    let owner = doc.nodes.get(&node_id).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "unsupported feature owner disappeared during evaluation",
                        )
                    })?;
                    let is_instance = instance_collection_target(owner)?.is_some();
                    if !is_instance
                        && owner
                            .modifiers
                            .iter()
                            .all(|modifier| !modifier.is_active(context.purpose))
                    {
                        return Err(error);
                    }
                    suppressed.insert(node_id.clone());
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
                    let dependencies = &mesh_dependencies;
                    let mut affected = vec![node_id];
                    while let Some(source_id) = affected.pop() {
                        for (dependent_id, targets) in
                            dependencies.modifiers.iter().chain(&dependencies.other)
                        {
                            if dependent_id == &source_id
                                || !targets.contains(&source_id)
                                || node_errors.contains_key(dependent_id)
                            {
                                continue;
                            }
                            suppressed.insert(dependent_id.clone());
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
        source: &SceneDoc,
        context: &EvaluationContext,
        project_root: Option<&Path>,
        cache_directory: Option<&Path>,
        selected_nodes: Option<&BTreeSet<Id>>,
        modifier_prefix: Option<ModifierPrefix<'_>>,
        driver_values: Option<&BTreeMap<(Id, String, u32), f64>>,
    ) -> Result<Self> {
        source.validate()?;
        let modifier_context = ModifierEvaluationContext {
            purpose: context.purpose,
            project_root,
            cache_directory,
            material_slot_count: None,
        };
        crate::library::validate_linked_assets(source, modifier_context.project_root)?;
        let scene_hash = source_scene_hash(source)?;
        let suppressed = BTreeSet::new();
        let mesh_dependencies = object_mesh_dependencies(
            source,
            ModifierGate {
                context: modifier_context,
                suppressed: &suppressed,
                modifier_prefix,
            },
            None,
        )?;
        let mut selected_nodes_with_prefix = None;
        let mut world_matrix_ids = None;
        if let Some(prefix) = modifier_prefix {
            let mut selected = selected_nodes.cloned().unwrap_or_default();
            let mut world_ids = selected.clone();
            let target = source.nodes.get(prefix.node_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    "modifier prefix node was not found",
                    json!({"node_id":prefix.node_id}),
                )
            })?;
            let prefix_gate = ModifierGate {
                context: modifier_context,
                suppressed: &suppressed,
                modifier_prefix,
            };
            for modifier in target.modifiers.iter().take(prefix.keep) {
                if !prefix_gate.is_active(prefix.node_id, modifier) {
                    continue;
                }
                for reference_id in modifier_transform_references(modifier) {
                    world_ids.insert(reference_id.clone());
                    let Some(reference) = source.nodes.get(&reference_id) else {
                        continue;
                    };
                    if reference.kind == "armature" {
                        selected.insert(reference_id.clone());
                    }
                    for constraint in &reference.constraints {
                        if constraint.enabled
                            && constraint.influence != 0.0
                            && matches!(
                                constraint.constraint_type,
                                ConstraintType::Shrinkwrap | ConstraintType::GeometryAttribute
                            )
                            && let Some(dependency_id) = &constraint.target
                            && mesh_dependencies
                                .other
                                .get(&reference_id)
                                .is_some_and(|dependencies| dependencies.contains(dependency_id))
                        {
                            selected.insert(dependency_id.clone());
                        }
                    }
                }
            }
            selected_nodes_with_prefix = Some(selected);
            world_matrix_ids = Some(world_ids);
        }
        let selected_nodes = selected_nodes_with_prefix.as_ref().or(selected_nodes);
        let world_matrix_ids = world_matrix_ids.as_ref();
        let simulation_scene_hash = if let Some(prefix) = modifier_prefix {
            hash::sha256(&hash::canonicalize(&json!({
                "scene_hash": scene_hash,
                "modifier_prefix": {"node_id":prefix.node_id,"keep":prefix.keep},
            }))?)
        } else {
            scene_hash.clone()
        };
        Self::evaluate_gated(
            source,
            &scene_hash,
            &simulation_scene_hash,
            context,
            modifier_context,
            selected_nodes,
            &suppressed,
            &mesh_dependencies,
            world_matrix_ids,
            modifier_prefix,
            driver_values,
        )
    }

    /// Evaluates `doc` with the `suppressed` nodes' modifiers inactive. `scene_hash` addresses
    /// the authored source, not the document itself.
    fn evaluate_gated(
        doc: &SceneDoc,
        scene_hash: &str,
        simulation_scene_hash: &str,
        context: &EvaluationContext,
        modifier_context: ModifierEvaluationContext<'_>,
        selected_nodes: Option<&BTreeSet<Id>>,
        suppressed: &BTreeSet<Id>,
        mesh_dependencies: &ObjectMeshDependencies,
        world_matrix_ids: Option<&BTreeSet<Id>>,
        modifier_prefix: Option<ModifierPrefix<'_>>,
        driver_values: Option<&BTreeMap<(Id, String, u32), f64>>,
    ) -> Result<Self> {
        let gate = ModifierGate {
            context: modifier_context,
            suppressed,
            modifier_prefix,
        };
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
        let mesh_evaluation_order =
            object_mesh_evaluation_order(doc, selected_nodes, gate, mesh_dependencies)?;
        let evaluated_node_ids: BTreeSet<Id> = mesh_evaluation_order.iter().cloned().collect();

        let computed_driver_values;
        let driver_values = if let Some(driver_values) = driver_values {
            driver_values
        } else {
            computed_driver_values = rig::evaluate_drivers(doc, frame)?;
            &computed_driver_values
        };
        let mut matrices = rig::evaluate_world_matrices_with_resources(
            doc,
            frame,
            driver_values,
            modifier_context.project_root,
            scene.fps,
            scene.fps_base,
        )?;
        let adaptive_render = context
            .adaptive_render
            .as_ref()
            .map(|request| adaptive_render_context(doc, scene, &matrices, request))
            .transpose()?;
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
                gate.is_active(owner_id, modifier)
                    && modifier.modifier_type == "array"
                    && modifier
                        .params
                        .get("fit_type")
                        .and_then(serde_json::Value::as_str)
                        == Some("FIT_CURVE")
            }) {
                if let Some((curve_id, length)) =
                    evaluated_array_curve_path_length(doc, modifier, &matrices)?
                {
                    curve_path_lengths.entry(curve_id).or_insert(length);
                }
            }
        }
        let simulation = crate::sim::evaluate(
            doc,
            &scene_id,
            frame,
            &matrices,
            simulation_scene_hash,
            gate,
        )?;
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
        let mut bone_pose_matrices = BTreeMap::new();
        let modifier_pose_armatures = modifier_pose_armatures(doc, &evaluated_node_ids, gate, None);
        for (id, node) in &doc.nodes {
            if node.kind == "armature"
                && (evaluated_node_ids.contains(id) || modifier_pose_armatures.contains(id))
            {
                let armature_world = matrices.get(id).copied().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "evaluated armature world matrix is missing",
                    )
                })?;
                let (world, pose) = if modifier_pose_armatures.contains(id) {
                    let evaluated =
                        rig::evaluate_armature_bone_world_and_pose_matrices_with_resources(
                            doc,
                            id,
                            frame,
                            armature_world,
                            &matrices,
                            driver_values,
                            modifier_context.project_root,
                            scene.fps,
                            scene.fps_base,
                        )?;
                    (evaluated.world, evaluated.pose)
                } else {
                    (
                        rig::evaluate_armature_bone_world_matrices_with_resources(
                            doc,
                            id,
                            frame,
                            armature_world,
                            &matrices,
                            driver_values,
                            modifier_context.project_root,
                            scene.fps,
                            scene.fps_base,
                        )?,
                        None,
                    )
                };
                bone_matrices.insert(
                    id.clone(),
                    world
                        .into_iter()
                        .map(|(bone_id, matrix)| (bone_id, matrix.to_cols_array()))
                        .collect(),
                );
                if let Some(pose) = pose {
                    bone_pose_matrices.insert(
                        id.clone(),
                        pose.into_iter()
                            .map(|(bone_id, matrix)| (bone_id, matrix.to_cols_array()))
                            .collect(),
                    );
                }
            }
        }
        let mut evaluation_key = json!({
            "engine": EVALUATION_ENGINE,
            "build_identity": crate::BUILD_IDENTITY,
            "simulation_engine": crate::sim::SOLVER_VERSION,
            "scene_hash": scene_hash,
            "scene_id": scene_id,
            "view_layer": view_layer,
            "purpose": context.purpose,
            "frame": frame,
            "adaptive_render": context.adaptive_render,
            "simulation_cache_key": simulation_cache_key,
        });
        if let Some(prefix) = modifier_prefix {
            evaluation_key["modifier_prefix"] =
                json!({"node_id":prefix.node_id,"keep":prefix.keep});
        }
        let evaluation_hash = hash::sha256(&hash::canonicalize(&evaluation_key)?);
        let mut volume_data = if mesh_evaluation_order.is_empty() {
            BTreeMap::new()
        } else {
            load_vdb_volume_data(doc, modifier_context.project_root)?
        };

        let mut nodes = BTreeMap::new();
        let mut meshes = BTreeMap::new();
        let mut adaptive_normals = BTreeMap::new();
        let mut instance_paths = BTreeMap::new();
        let mut slot_bindings = BTreeMap::new();
        let mut strokes = BTreeMap::new();
        let mut modifier_warnings = Vec::new();
        for id in &mesh_evaluation_order {
            let node = doc.nodes.get(id).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "mesh evaluation node is missing")
            })?;
            let node_data = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id));
            // The mesh's own material array (DATA). Modifiers replace it (the Boolean TRANSFER pool);
            // the evaluated table is resolved from it once the stack has run.
            let mut node_materials = node_data.map_or_else(Vec::new, |data| data.materials.clone());
            let matrix = matrices.get(id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "evaluated world matrix is missing",
                )
            })?;
            let texture_seed = crate::geom::modifiers::TextureSeed::new(
                modifier_context.purpose,
                scene.render.seed,
                id.as_str(),
            );
            let references = SceneReferences {
                world_matrices: &matrices,
                bone_matrices: &bone_matrices,
                bone_pose_matrices: &bone_pose_matrices,
                evaluated_meshes: &meshes,
                project_root: modifier_context.project_root,
            };
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
                            driver_values,
                        )?;
                        rig::apply_shape_keys_with_values(
                            &mut rigged_source,
                            shape_keys,
                            data_block,
                            &values,
                        )?;
                    }
                    if !has_stack_modifier(gate, id, node, "cloth")
                        && !has_stack_modifier(gate, id, node, "soft_body")
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
                    if !has_stack_modifier(gate, id, node, "dynamic_paint")
                        && let Some(colors) = paint_colors.get(id)
                    {
                        if colors.len() != rigged_source.vertices.len() {
                            return Err(PotError::with_details(
                                ErrorCode::EvaluationFailed,
                                "dynamic-paint color count does not match source mesh",
                                json!({"node_id":id}),
                            ));
                        }
                        inject_dynamic_paint_colors(&mut rigged_source, colors);
                    }
                    if !has_stack_modifier(gate, id, node, "dynamic_paint")
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
                            crate::geom::attribute_keys::storage_key("weight").into_owned(),
                            json!({"domain":"vertices","type":"float","values":values}),
                        );
                    }
                    attach_vertex_group_weights(&mut rigged_source, data_block);
                    // Shape keys are applied above before modifiers, so adaptive derivatives use the same
                    // evaluated coordinates as the render mesh.
                    let mut stack_context = gate.context;
                    stack_context.material_slot_count = Some(crate::model::material_slot_count(
                        node.materials.len(),
                        node_materials.len(),
                    ));
                    let (evaluated, mesh_normals) =
                        crate::geom::modifiers::evaluate_modifiers_with_scene_context_and_normals(
                            rigged_source,
                            &node.modifiers,
                            &doc.node_groups,
                            frame,
                            node.name.as_str(),
                            matrix,
                            &matrices,
                            &object_meshes,
                            &meshes,
                            &curve_path_lengths,
                            stack_context,
                            |_, modifier| gate.is_active(id, modifier),
                            texture_seed,
                            adaptive_render.as_ref(),
                            |mesh, modifier, modifier_warnings, multi_input, texture_rng| {
                                let result = match modifier.modifier_type.as_str() {
                                    "armature" => rig::apply_armature_modifier_with_drivers(
                                        mesh,
                                        data_block,
                                        modifier,
                                        matrix,
                                        doc,
                                        frame,
                                        &matrices,
                                        driver_values,
                                        modifier_context.project_root,
                                        scene.fps,
                                        scene.fps_base,
                                        &meshes,
                                        multi_input,
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
                                    "displace" | "wave" | "warp" => {
                                        apply_scene_displacement_modifier(
                                            mesh,
                                            modifier,
                                            id,
                                            doc,
                                            &references,
                                            frame,
                                            texture_rng,
                                        )
                                    }
                                    "boolean" | "shrinkwrap" | "curve" | "hook" | "cast"
                                    | "simple_deform" => apply_scene_operand_modifier(
                                        mesh,
                                        modifier,
                                        id,
                                        doc,
                                        &references,
                                        &nodes,
                                        &mut node_materials,
                                    ),
                                    "vertex_weight_edit"
                                    | "vertex_weight_mix"
                                    | "vertex_weight_proximity"
                                    | "normal_edit"
                                    | "uv_project"
                                    | "uv_warp"
                                    | "data_transfer"
                                    | "mask" => apply_scene_attribute_modifier(
                                        mesh,
                                        modifier,
                                        id,
                                        doc,
                                        &matrices,
                                        &meshes,
                                        &bone_matrices,
                                        &bone_pose_matrices,
                                        modifier_context.project_root,
                                    ),
                                    "surface_deform" | "mesh_deform" => {
                                        apply_scene_deform_modifier(
                                            mesh,
                                            modifier,
                                            id,
                                            doc,
                                            &matrices,
                                            &meshes,
                                            modifier_warnings,
                                        )
                                    }
                                    "mesh_sequence_cache" => apply_mesh_sequence_cache_modifier(
                                        mesh,
                                        modifier,
                                        doc,
                                        modifier_context.project_root,
                                        frame,
                                        scene.fps,
                                        scene.fps_base,
                                    )
                                    .map(|warnings| modifier_warnings.extend(warnings)),
                                    "mesh_cache" => apply_mesh_cache_modifier(
                                        mesh,
                                        modifier,
                                        doc,
                                        modifier_context.project_root,
                                        frame,
                                        scene.fps,
                                        scene.fps_base,
                                    ),
                                    "cloth" | "soft_body" | "collision" | "dynamic_paint"
                                    | "fluid" | "particle_system" => apply_stack_physics_modifier(
                                        mesh,
                                        modifier,
                                        id,
                                        doc,
                                        simulation_scene_hash,
                                        gate,
                                        &scene_id,
                                        frame,
                                        &matrices,
                                        &mut simulated_particles,
                                    ),
                                    "particle_instance" => apply_particle_instance_modifier(
                                        mesh,
                                        modifier,
                                        id,
                                        doc,
                                        gate,
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
                                        gate,
                                        frame,
                                        &matrices,
                                        &simulated_particles,
                                    ),
                                    _ => Err(PotError::new(
                                        ErrorCode::InternalError,
                                        "unexpected custom modifier callback",
                                    )),
                                };
                                result.map(|()| {
                                    Some(crate::model::material_slot_count(
                                        node.materials.len(),
                                        node_materials.len(),
                                    ))
                                })
                            },
                        )
                        .map_err(|error| attach_node_owner(error, id))?;
                    modifier_warnings.extend(
                        evaluated
                            .warnings
                            .into_iter()
                            .map(|warning| attach_node_owner(warning, id)),
                    );
                    let uses_fluid_surface = !has_stack_modifier(gate, id, node, "fluid")
                        && fluid_surfaces.contains_key(id);
                    let evaluated_mesh = if has_stack_modifier(gate, id, node, "fluid") {
                        evaluated.mesh
                    } else {
                        fluid_surfaces.get(id).cloned().unwrap_or(evaluated.mesh)
                    };
                    let bounds = mesh_bounds(&evaluated_mesh, matrix)?;
                    meshes.insert(id.clone(), evaluated_mesh);
                    if !uses_fluid_surface && let Some(normals) = mesh_normals {
                        adaptive_normals.insert(id.clone(), normals);
                    }
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
                    let mut stack_context = gate.context;
                    stack_context.material_slot_count = Some(crate::model::material_slot_count(
                        node.materials.len(),
                        node_materials.len(),
                    ));
                    let (evaluated, evaluated_normals) =
                        crate::geom::modifiers::evaluate_modifiers_with_scene_context_and_normals(
                            source_mesh,
                            &node.modifiers,
                            &doc.node_groups,
                            frame,
                            node.name.as_str(),
                            matrix,
                            &matrices,
                            &object_meshes,
                            &meshes,
                            &curve_path_lengths,
                            stack_context,
                            |_, modifier| gate.is_active(id, modifier),
                            texture_seed,
                            adaptive_render.as_ref(),
                            |mesh, modifier, modifier_warnings, multi_input, texture_rng| {
                                let result = match modifier.modifier_type.as_str() {
                                    "armature" => rig::apply_armature_modifier_with_drivers(
                                        mesh,
                                        data_block,
                                        modifier,
                                        matrix,
                                        doc,
                                        frame,
                                        &matrices,
                                        driver_values,
                                        modifier_context.project_root,
                                        scene.fps,
                                        scene.fps_base,
                                        &meshes,
                                        multi_input,
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
                                    "displace" | "wave" | "warp" => {
                                        apply_scene_displacement_modifier(
                                            mesh,
                                            modifier,
                                            id,
                                            doc,
                                            &references,
                                            frame,
                                            texture_rng,
                                        )
                                    }
                                    "boolean" | "shrinkwrap" | "curve" | "hook" | "cast"
                                    | "simple_deform" => apply_scene_operand_modifier(
                                        mesh,
                                        modifier,
                                        id,
                                        doc,
                                        &references,
                                        &nodes,
                                        &mut node_materials,
                                    ),
                                    "vertex_weight_edit"
                                    | "vertex_weight_mix"
                                    | "vertex_weight_proximity"
                                    | "normal_edit"
                                    | "uv_project"
                                    | "uv_warp"
                                    | "data_transfer"
                                    | "mask" => apply_scene_attribute_modifier(
                                        mesh,
                                        modifier,
                                        id,
                                        doc,
                                        &matrices,
                                        &meshes,
                                        &bone_matrices,
                                        &bone_pose_matrices,
                                        modifier_context.project_root,
                                    ),
                                    "surface_deform" | "mesh_deform" => {
                                        apply_scene_deform_modifier(
                                            mesh,
                                            modifier,
                                            id,
                                            doc,
                                            &matrices,
                                            &meshes,
                                            modifier_warnings,
                                        )
                                    }
                                    "mesh_sequence_cache" => apply_mesh_sequence_cache_modifier(
                                        mesh,
                                        modifier,
                                        doc,
                                        modifier_context.project_root,
                                        frame,
                                        scene.fps,
                                        scene.fps_base,
                                    )
                                    .map(|warnings| modifier_warnings.extend(warnings)),
                                    "mesh_cache" => apply_mesh_cache_modifier(
                                        mesh,
                                        modifier,
                                        doc,
                                        modifier_context.project_root,
                                        frame,
                                        scene.fps,
                                        scene.fps_base,
                                    ),
                                    "cloth" | "soft_body" | "collision" | "dynamic_paint"
                                    | "fluid" | "particle_system" => apply_stack_physics_modifier(
                                        mesh,
                                        modifier,
                                        id,
                                        doc,
                                        simulation_scene_hash,
                                        gate,
                                        &scene_id,
                                        frame,
                                        &matrices,
                                        &mut simulated_particles,
                                    ),
                                    "particle_instance" => apply_particle_instance_modifier(
                                        mesh,
                                        modifier,
                                        id,
                                        doc,
                                        gate,
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
                                        gate,
                                        frame,
                                        &matrices,
                                        &simulated_particles,
                                    ),
                                    _ => Err(PotError::new(
                                        ErrorCode::InternalError,
                                        "unexpected custom modifier callback",
                                    )),
                                };
                                result.map(|()| {
                                    Some(crate::model::material_slot_count(
                                        node.materials.len(),
                                        node_materials.len(),
                                    ))
                                })
                            },
                        )
                        .map_err(|error| attach_node_owner(error, id))?;
                    modifier_warnings.extend(
                        evaluated
                            .warnings
                            .into_iter()
                            .map(|warning| attach_node_owner(warning, id)),
                    );
                    if let Some(normals) = evaluated_normals {
                        adaptive_normals.insert(id.clone(), normals);
                    }
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
                        gate.is_active(id, modifier) && modifier.modifier_type == "mesh_to_volume"
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
                        // A parameter the modifier does not store takes its Blender default from the schema table.
                        let number = |parameter: &str| -> Result<f64> {
                            match modifier.params.get(parameter).and_then(Value::as_f64) {
                                Some(value) => Ok(value),
                                None => mesh_to_volume_default(parameter)?.as_f64().ok_or_else(|| {
                                    PotError::new(
                                        ErrorCode::InternalError,
                                        format!("mesh_to_volume default `{parameter}` is not a number"),
                                    )
                                }),
                            }
                        };
                        let resolution_mode = match modifier
                            .params
                            .get("resolution_mode")
                            .and_then(Value::as_str)
                        {
                            Some(mode) => mode,
                            None => mesh_to_volume_default("resolution_mode")?
                                .as_str()
                                .ok_or_else(|| {
                                    PotError::new(
                                        ErrorCode::InternalError,
                                        "mesh_to_volume default `resolution_mode` is not a string",
                                    )
                                })?,
                        };
                        let voxel_size = match resolution_mode {
                            "VOXEL_SIZE" => number("voxel_size")?,
                            "VOXEL_AMOUNT" => {
                                let amount = number("voxel_amount")?;
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
                            number("interior_band_width")?,
                            number("density")?,
                            true,
                        )?);
                    }
                    if let Some(generated) = generated {
                        volume_data.insert(data_id.clone(), generated);
                    }
                    let base_volume = volume_data.get(data_id).unwrap_or(volume);
                    let mut displaced = None;
                    for (stack_index, modifier) in
                        node.modifiers
                            .iter()
                            .enumerate()
                            .filter_map(|(index, modifier)| {
                                (gate.is_active_at(id, index, modifier)
                                    && modifier.modifier_type == "volume_displace")
                                    .then_some((index, modifier))
                            })
                    {
                        let current = displaced.take().unwrap_or_else(|| base_volume.clone());
                        displaced = Some(apply_volume_displace_modifier(
                            &current,
                            modifier,
                            doc,
                            gate,
                            &mut texture_seed.rng_for(stack_index),
                            &matrices,
                            &bone_pose_matrices,
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
                    let (evaluated, bounds) =
                        evaluate_grease_pencil(grease_pencil, &doc.materials, matrix, frame)?;
                    if !evaluated.is_empty() {
                        strokes.insert(id.clone(), evaluated);
                    }
                    bounds
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
                    materials: effective_material_slots(node, &node_materials),
                    mesh_materials: node_materials,
                },
            );
        }
        let has_mesh_dependent_constraints = doc.nodes.iter().any(|(id, node)| {
            (evaluated_node_ids.contains(id)
                || world_matrix_ids.is_some_and(|ids| ids.contains(id)))
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
                driver_values,
                modifier_context.project_root,
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
            bone_pose_matrices.clear();
            for (id, node) in &doc.nodes {
                if node.kind == "armature"
                    && (evaluated_node_ids.contains(id) || modifier_pose_armatures.contains(id))
                {
                    let armature_world = matrices.get(id).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "evaluated armature world matrix is missing",
                        )
                    })?;
                    let (world, pose) = if modifier_pose_armatures.contains(id) {
                        let evaluated =
                            rig::evaluate_armature_bone_world_and_pose_matrices_with_evaluated_meshes(
                                doc,
                                id,
                                frame,
                                armature_world,
                                &matrices,
                                driver_values,
                                modifier_context.project_root,
                                scene.fps,
                                scene.fps_base,
                                &meshes,
                            )?;
                        (evaluated.world, evaluated.pose)
                    } else {
                        (
                            rig::evaluate_armature_bone_world_matrices_with_evaluated_meshes(
                                doc,
                                id,
                                frame,
                                armature_world,
                                &matrices,
                                driver_values,
                                modifier_context.project_root,
                                scene.fps,
                                scene.fps_base,
                                &meshes,
                            )?,
                            None,
                        )
                    };
                    bone_matrices.insert(
                        id.clone(),
                        world
                            .into_iter()
                            .map(|(bone_id, matrix)| (bone_id, matrix.to_cols_array()))
                            .collect(),
                    );
                    if let Some(pose) = pose {
                        bone_pose_matrices.insert(
                            id.clone(),
                            pose.into_iter()
                                .map(|(bone_id, matrix)| (bone_id, matrix.to_cols_array()))
                                .collect(),
                        );
                    }
                }
            }
        }
        render_simulated_particles(
            doc,
            gate,
            &simulated_particles,
            &matrices,
            &mut meshes,
            &mut nodes,
            &evaluated_node_ids,
            &mut adaptive_normals,
        )?;
        for (id, node) in &doc.nodes {
            if !nodes.contains_key(id)
                || gate.suppressed.contains(id)
                || instance_collection_target(node)?.is_none()
            {
                continue;
            }
            let matrix = matrices.get(id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "evaluated collection instance world matrix is missing",
                )
            })?;
            let CollectionEvaluation {
                mesh,
                paths,
                bindings,
                adaptive_normals: member_normals,
                ..
            } = evaluate_collection_instance(
                doc,
                id,
                &matrices,
                &meshes,
                &nodes,
                &adaptive_normals,
                &mut BTreeSet::new(),
            )?;
            let bounds = mesh_bounds(&mesh, matrix)?;
            if !mesh.vertices.is_empty() {
                slot_bindings.insert(id.clone(), bindings);
                meshes.insert(id.clone(), mesh);
                if let Some(normals) = member_normals {
                    adaptive_normals.insert(id.clone(), normals);
                }
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
        let transform_world_matrices: BTreeMap<Id, DMat4> =
            world_matrix_ids.map_or_else(BTreeMap::new, |ids| {
                ids.iter()
                    .filter_map(|id| matrices.get(id).copied().map(|matrix| (id.clone(), matrix)))
                    .collect()
            });
        Ok(Self {
            scene_id,
            view_layer,
            frame,
            purpose: context.purpose,
            revision: doc.revision,
            scene_hash: scene_hash.to_owned(),
            evaluation_hash,
            nodes,
            transform_world_matrices,
            instance_paths,
            slot_bindings,
            meshes,
            adaptive_normals,
            strokes,
            volume_data,
            bone_matrices,
            bone_pose_matrices,
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

/// The effective material slots of one node: each slot resolved under its OBJECT or DATA link by the
/// model's `resolve_material_slot`, matching Blender `BKE_object_material_get_eval`
/// (blenkernel/intern/material.cc:772): an object slot whose matbit is set and holds a material wins,
/// otherwise the DATA slot. `mesh_materials` is the mesh's material array after the modifier stack:
/// its DATA table, or the Boolean TRANSFER pool that replaced it. This is the evaluated table.
pub(crate) fn effective_material_slots(
    node: &Node,
    mesh_materials: &[Option<Id>],
) -> Vec<Option<Id>> {
    let count = crate::model::material_slot_count(node.materials.len(), mesh_materials.len());
    (0..count)
        .map(|slot| {
            crate::model::resolve_material_slot(
                &node.materials,
                mesh_materials,
                &node.material_links,
                slot,
            )
            .cloned()
        })
        .collect()
}

/// The UV Project slot count (`projectors_num`, MOD_uvproject.cc). Slots past it are ignored, empty
/// slots included, so their objects are neither resolved nor depended on.
fn uv_project_slot_count(modifier: &Modifier) -> usize {
    modifier
        .params
        .get("projector_count")
        .and_then(Value::as_u64)
        .map_or(1, |count| count.max(1) as usize)
}

fn modifier_transform_references(modifier: &Modifier) -> BTreeSet<Id> {
    fn add_reference(references: &mut BTreeSet<Id>, value: Option<&Value>) {
        if let Some(text) = value.and_then(Value::as_str)
            && let Ok(id) = Id::new(text.to_owned())
        {
            references.insert(id);
        }
    }

    let parameters: &[&str] = match modifier.modifier_type.as_str() {
        "armature" | "hook" | "lattice" | "screw" => &["object"],
        "mirror" => &["mirror_object"],
        "uv_project" => &["projectors"],
        "uv_warp" | "warp" => &["object_from", "object_to"],
        "vertex_weight_edit" | "vertex_weight_mix" | "vertex_weight_proximity" => {
            &["mask_tex_map_object"]
        }
        "volume_displace" => &["texture_map_object"],
        _ => &[],
    };
    let mut references = BTreeSet::new();
    for parameter in parameters {
        let Some(value) = modifier.params.get(*parameter) else {
            continue;
        };
        if *parameter == "projectors" {
            if let Some(projectors) = value.as_array() {
                for projector in projectors.iter().take(uv_project_slot_count(modifier)) {
                    add_reference(&mut references, Some(projector));
                }
            }
        } else {
            add_reference(&mut references, Some(value));
        }
    }
    if modifier.modifier_type == "array" {
        if modifier.params.get("fit_type").and_then(Value::as_str) == Some("FIT_CURVE") {
            add_reference(&mut references, modifier.params.get("curve"));
        }
        if modifier
            .params
            .get("use_object_offset")
            .and_then(Value::as_bool)
            == Some(true)
        {
            add_reference(&mut references, modifier.params.get("offset_object"));
        }
    }
    references
}

/// Content address of the authored scene. Purpose resolution never changes it.
fn source_scene_hash(doc: &SceneDoc) -> Result<String> {
    let scene_value = serde_json::to_value(doc).map_err(PotError::internal_json)?;
    Ok(hash::sha256(&hash::canonicalize(&scene_value)?))
}
/// Curve-path length comes from the Curve Data-Block and world transform, not its modifier stack.
pub(crate) fn evaluated_array_curve_path_length(
    doc: &SceneDoc,
    modifier: &Modifier,
    world_matrices: &BTreeMap<Id, DMat4>,
) -> Result<Option<(Id, f64)>> {
    if modifier.modifier_type != "array"
        || modifier
            .params
            .get("fit_type")
            .and_then(serde_json::Value::as_str)
            != Some("FIT_CURVE")
    {
        return Ok(None);
    }
    let Some(curve_reference) = modifier
        .params
        .get("curve")
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    let Ok(curve_id) = Id::new(curve_reference.to_owned()) else {
        return Ok(None);
    };
    let Some(curve_node) = doc.nodes.get(&curve_id) else {
        return Ok(None);
    };
    let Some(curve_data_id) = curve_node.data.as_ref() else {
        return Ok(None);
    };
    let Some(curve_data) = doc
        .data_blocks
        .get(curve_data_id)
        .filter(|data_block| data_block.data_type == "curve")
        .and_then(|data_block| data_block.curve.as_ref())
    else {
        return Ok(None);
    };
    let Some(curve_world_matrix) = world_matrices.get(&curve_id).copied() else {
        return Ok(None);
    };
    let length =
        crate::geom::modifiers::evaluated_curve_path_length(curve_data, curve_world_matrix)?;
    Ok(Some((curve_id, length)))
}

#[derive(Clone, Copy)]
struct ModifierPrefix<'a> {
    node_id: &'a Id,
    modifiers: &'a [Modifier],
    keep: usize,
}

/// Modifier state for one evaluation. `purpose` selects the viewport or render stack, and
/// `suppressed` nodes lost their modifiers to an unsupported-feature error. Every consumer asks
/// this type whether a modifier is active, so Preview and Render cannot diverge.
#[derive(Clone, Copy)]
pub(crate) struct ModifierGate<'a> {
    pub(crate) context: ModifierEvaluationContext<'a>,
    pub(crate) suppressed: &'a BTreeSet<Id>,
    modifier_prefix: Option<ModifierPrefix<'a>>,
}

impl ModifierGate<'_> {
    fn stack_modifiers<'m>(&self, node_id: &Id, modifiers: &'m [Modifier]) -> &'m [Modifier] {
        match self.modifier_prefix {
            Some(prefix) if prefix.node_id == node_id => {
                &modifiers[..prefix.keep.min(modifiers.len())]
            }
            _ => modifiers,
        }
    }

    /// Whether `modifier` on `node_id` takes part in this evaluation.
    pub(crate) fn is_active(&self, node_id: &Id, modifier: &Modifier) -> bool {
        let in_prefix = match self.modifier_prefix {
            Some(prefix) if prefix.node_id == node_id => prefix
                .modifiers
                .iter()
                .position(|candidate| candidate.id == modifier.id)
                .is_some_and(|index| index < prefix.keep),
            _ => true,
        };
        in_prefix && !self.suppressed.contains(node_id) && modifier.is_active(self.context.purpose)
    }

    /// Whether the modifier at stack `index` on `node_id` takes part in this evaluation (F's index form).
    fn is_active_at(&self, node_id: &Id, index: usize, modifier: &Modifier) -> bool {
        let in_prefix = self
            .modifier_prefix
            .is_none_or(|prefix| prefix.node_id != node_id || index < prefix.keep);
        in_prefix && !self.suppressed.contains(node_id) && modifier.is_active(self.context.purpose)
    }

    /// Whether a legacy physics property has an active stack modifier linked to its settings.
    pub(crate) fn is_property_active(
        &self,
        node_id: &Id,
        node: &Node,
        property_name: &str,
        modifier_type: &str,
    ) -> bool {
        if self.suppressed.contains(node_id) {
            return false;
        }
        let linked_modifier = if property_name == "physics_particle_emitter" {
            node.properties
                .get("physics_particle_emitter_modifier_id")
                .and_then(serde_json::Value::as_str)
                .and_then(|modifier_id| {
                    node.modifiers.iter().find(|modifier| {
                        modifier.id.as_str() == modifier_id
                            && modifier.modifier_type == modifier_type
                            && modifier
                                .params
                                .get("settings_id")
                                .and_then(serde_json::Value::as_str)
                                == Some(modifier.id.as_str())
                    })
                })
        } else {
            node.modifiers.iter().find(|modifier| {
                modifier.modifier_type == modifier_type
                    && modifier
                        .params
                        .get("settings_id")
                        .and_then(serde_json::Value::as_str)
                        == Some(modifier.id.as_str())
            })
        };
        linked_modifier.is_none_or(|modifier| self.is_active(node_id, modifier))
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

pub(crate) fn attach_node_owner(mut error: PotError, node_id: &Id) -> PotError {
    if matches!(
        error.code,
        ErrorCode::UnsupportedFeature | ErrorCode::UnrepresentableFeature
    ) && let Some(details) = error.details.as_object_mut()
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

/// Adds a Boolean's operand objects as dependencies of `owner_id`: its `object`, and every object
/// of its `collection`. The subject is never its own operand (`apply_boolean_operands_modifier`
/// drops it), so a self reference is skipped rather than reported as a dependency cycle.
fn add_boolean_operand_dependencies(
    dependencies: &mut BTreeMap<Id, BTreeSet<Id>>,
    owner_id: &Id,
    modifier: &Modifier,
    doc: &SceneDoc,
) -> Result<()> {
    let mut operand_ids = BTreeSet::new();
    if let Some(object_id) = modifier
        .params
        .get("object")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| Id::new(text).ok())
    {
        operand_ids.insert(object_id);
    }
    if let Some(collection_id) = modifier
        .params
        .get("collection")
        .and_then(serde_json::Value::as_str)
        .and_then(|text| Id::new(text).ok())
        && doc.collections.contains_key(&collection_id)
    {
        collect_modifier_collection_objects(
            doc,
            &collection_id,
            &mut BTreeSet::new(),
            &mut operand_ids,
        )?;
    }
    for target_id in operand_ids {
        if target_id != *owner_id
            && dependencies.contains_key(&target_id)
            && let Some(owner_dependencies) = dependencies.get_mut(owner_id)
        {
            owner_dependencies.insert(target_id);
        }
    }
    Ok(())
}

struct ObjectMeshDependencies {
    modifiers: BTreeMap<Id, BTreeSet<Id>>,
    other: BTreeMap<Id, BTreeSet<Id>>,
}

fn object_mesh_dependencies(
    doc: &SceneDoc,
    gate: ModifierGate<'_>,
    only_modifier: Option<(&Id, &Id)>,
) -> Result<ObjectMeshDependencies> {
    let node_dependencies = || {
        doc.nodes
            .keys()
            .cloned()
            .map(|id| (id, BTreeSet::new()))
            .collect::<BTreeMap<_, _>>()
    };
    let mut dependencies = ObjectMeshDependencies {
        modifiers: node_dependencies(),
        other: node_dependencies(),
    };
    for (owner_id, node) in &doc.nodes {
        for modifier in node.modifiers.iter().filter(|modifier| {
            if let Some((target_id, modifier_id)) = only_modifier
                && target_id == owner_id
            {
                modifier.enabled && &modifier.id == modifier_id
            } else {
                gate.is_active(owner_id, modifier)
            }
        }) {
            if modifier.modifier_type == "boolean" {
                add_boolean_operand_dependencies(
                    &mut dependencies.modifiers,
                    owner_id,
                    modifier,
                    doc,
                )?;
                continue;
            }
            let parameters: &[&str] = match modifier.modifier_type.as_str() {
                "data_transfer" | "mesh_deform" | "curve" | "mesh_to_volume" => &["object"],
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
                "array" => &["start_cap", "end_cap"],
                _ => &[],
            };
            for parameter in parameters {
                add_mesh_operand_dependency(
                    &mut dependencies.modifiers,
                    owner_id,
                    modifier.params.get(*parameter),
                );
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
                if dependencies.other.contains_key(&target_id)
                    && let Some(owner_dependencies) = dependencies.other.get_mut(owner_id)
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
                && dependencies.other.contains_key(target_id)
                && let Some(owner_dependencies) = dependencies.other.get_mut(owner_id)
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
    gate: ModifierGate<'_>,
    dependencies: &ObjectMeshDependencies,
) -> Result<Vec<Id>> {
    fn visit(
        id: &Id,
        dependencies: &ObjectMeshDependencies,
        suppressed: &BTreeSet<Id>,
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
        if !suppressed.contains(id)
            && let Some(targets) = dependencies.modifiers.get(id)
        {
            for target in targets {
                visit(target, dependencies, suppressed, complete, active, output)?;
            }
        }
        if let Some(targets) = dependencies.other.get(id) {
            for target in targets {
                visit(target, dependencies, suppressed, complete, active, output)?;
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
        dependencies: &ObjectMeshDependencies,
        suppressed: &BTreeSet<Id>,
        required: &mut BTreeSet<Id>,
    ) {
        if !required.insert(id.clone()) {
            return;
        }
        if !suppressed.contains(id)
            && let Some(targets) = dependencies.modifiers.get(id)
        {
            for target in targets {
                include_dependencies(target, dependencies, suppressed, required);
            }
        }
        if let Some(targets) = dependencies.other.get(id) {
            for target in targets {
                include_dependencies(target, dependencies, suppressed, required);
            }
        }
    }

    let roots = selected_nodes
        .cloned()
        .unwrap_or_else(|| doc.nodes.keys().cloned().collect());
    let mut required = BTreeSet::new();
    for id in roots {
        include_dependencies(&id, dependencies, gate.suppressed, &mut required);
    }
    let mut order = Vec::with_capacity(required.len());
    let mut complete = BTreeSet::new();
    let mut active = Vec::new();
    for particle_system_first in [true, false] {
        for id in &required {
            let Some(node) = doc.nodes.get(id) else {
                continue;
            };
            let has_particle_system = (node.properties.contains_key("physics_particle_emitter")
                && gate.is_property_active(
                    id,
                    node,
                    "physics_particle_emitter",
                    "particle_system",
                ))
                || node.modifiers.iter().any(|modifier| {
                    gate.is_active(id, modifier) && modifier.modifier_type == "particle_system"
                });
            if has_particle_system == particle_system_first {
                visit(
                    id,
                    dependencies,
                    gate.suppressed,
                    &mut complete,
                    &mut active,
                    &mut order,
                )?;
            }
        }
    }
    Ok(order)
}

/// A non-empty string parameter; absent, null, and empty values read as unset.
fn non_empty_text<'a>(modifier: &'a Modifier, parameter: &str) -> Option<&'a str> {
    modifier
        .params
        .get(parameter)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

/// The ID of the named bone in an armature object's own armature data.
fn armature_bone_id(object: &Node, doc: &SceneDoc, bone_name: &str) -> Option<Id> {
    object
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.armature.as_ref())?
        .bones
        .iter()
        .find(|(_, bone)| bone.name == bone_name)
        .map(|(bone_id, _)| bone_id.clone())
}

/// Whether a vertex-weight proximity modifier measures to a geometry target (its `proximity_mode`
/// defaults to the schema value when unset).
fn proximity_mode_is_geometry(modifier: &Modifier) -> bool {
    modifier
        .params
        .get("proximity_mode")
        .or_else(|| {
            params::default_value(
                ParameterFamily::Modifier,
                "vertex_weight_proximity",
                "proximity_mode",
            )
        })
        .and_then(Value::as_str)
        == Some("GEOMETRY")
}

fn proximity_geometry_is_empty(modifier: &Modifier) -> bool {
    modifier
        .params
        .get("proximity_geometry")
        .or_else(|| {
            params::default_value(
                ParameterFamily::Modifier,
                "vertex_weight_proximity",
                "proximity_geometry",
            )
        })
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty)
}

fn proximity_uses_geometry_mesh(modifier: &Modifier) -> bool {
    proximity_mode_is_geometry(modifier) && !proximity_geometry_is_empty(modifier)
}

/// The armature data an operand object contributes to bone selection: its own armature data, or
/// the armature data of the object its Armature modifier names.
fn armature_data_for_modifier_operand<'a>(
    object: &Node,
    doc: &'a SceneDoc,
) -> Option<&'a ArmatureData> {
    let own_armature = object
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.armature.as_ref());
    own_armature.or_else(|| {
        object
            .modifiers
            .iter()
            .filter(|modifier| modifier.modifier_type == "armature")
            .find_map(|modifier| {
                let object_text = modifier.params.get("object")?.as_str()?;
                let object_id = Id::new(object_text.to_owned()).ok()?;
                let armature_node = doc.nodes.get(&object_id)?;
                let armature_data_id = armature_node.data.as_ref()?;
                doc.data_blocks
                    .get(armature_data_id)
                    .and_then(|data| data.armature.as_ref())
            })
    })
}

/// One object referenced by a scene-bound modifier, with what its operand is built from.
struct ReferencedObject<'a> {
    id: &'a Id,
    node: &'a Node,
    world: DMat4,
}

/// The Blender-order f32 matrices an operand carries for the parameter that references it, or
/// `None` when the consuming modifier reads no matrix from this operand.
fn operand_blender_matrices_f32(
    modifier: &Modifier,
    parameter: &str,
    object: &ReferencedObject<'_>,
    subject_world: DMat4,
    doc: &SceneDoc,
    bone_matrices: &BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
) -> Option<crate::geom::modifiers::attributes::SceneOperandMatricesF32> {
    use crate::geom::blender_matrix::{blender_inverse_m4, blender_multiply_m4};
    let object_f32 = object.world.to_cols_array().map(|value| value as f32);
    let subject_f32 = subject_world.to_cols_array().map(|value| value as f32);
    let (world, local_to_subject, subject_to_local) =
        match (modifier.modifier_type.as_str(), parameter) {
            ("data_transfer", "object") => {
                let use_object_transform = modifier
                    .params
                    .get("use_object_transform")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                if !use_object_transform {
                    return None;
                }
                // Blender's `local2target` maps subject-local points into source-local space, so
                // it is the transfer's subject-to-source matrix; its inverse maps back.
                let (subject_to_source, source_to_subject) =
                    crate::geom::blender_matrix::blender_space_transform_f32(
                        subject_f32,
                        object_f32,
                    );
                (None, Some(source_to_subject), Some(subject_to_source))
            }
            ("normal_edit", "target") => (
                None,
                Some(blender_multiply_m4(
                    blender_inverse_m4(subject_f32),
                    object_f32,
                )),
                None,
            ),
            ("vertex_weight_proximity", "target") if proximity_uses_geometry_mesh(modifier) => (
                None,
                None,
                Some(blender_multiply_m4(
                    blender_inverse_m4(object_f32),
                    subject_f32,
                )),
            ),
            ("vertex_weight_proximity", "target") | ("uv_project", "projectors") => {
                (Some(object_f32), None, None)
            }
            (_, "mask_tex_map_object") => {
                // A mask bone is read only from an armature object. A bone the armature lacks uses
                // the object's world matrix, as Blender's MOD_util does.
                let target_f32 = non_empty_text(modifier, "mask_tex_map_bone")
                    .filter(|_| object.node.kind == "armature")
                    .and_then(|bone_name| armature_bone_id(object.node, doc, bone_name))
                    .map_or(Some(object_f32), |bone_id| {
                        bone_matrices
                            .get(object.id)?
                            .get(&bone_id)
                            .map(|matrix| matrix.map(|value| value as f32))
                    });
                (
                    None,
                    None,
                    target_f32
                        .map(|target| blender_multiply_m4(blender_inverse_m4(target), subject_f32)),
                )
            }
            _ => return None,
        };
    Some(
        crate::geom::modifiers::attributes::SceneOperandMatricesF32 {
            world,
            local_to_subject,
            subject_to_local,
        },
    )
}

/// The object-local mesh of an operand object that is not already evaluated: mesh data or the
/// evaluated curve, with its vertex-group weights attached.
fn operand_object_mesh(
    modifier: &Modifier,
    object_id: &Id,
    data_block: Option<&DataBlock>,
) -> Result<Mesh> {
    let mut base_mesh = match data_block {
        Some(data) if data.data_type == "mesh" => data.mesh.clone().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "mesh modifier operand has no mesh payload",
                json!({"modifier_id":modifier.id,"object":object_id}),
            )
        })?,
        Some(data) if data.data_type == "curve" => {
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
        _ => Mesh::default(),
    };
    if let Some(data) = data_block {
        attach_vertex_group_weights(&mut base_mesh, data);
    }
    Ok(base_mesh)
}

/// Resolve the object-reference parameters for a scene-bound modifier through one shared path.
///
/// Each operand retains its object-local mesh where the consuming modifier reads it, and carries
/// the Blender f32 matrices and armature state that its modifier reads. `bone_matrices` supplies
/// pose-bone world matrices for `mask_tex_map_bone`; callers whose modifiers never reference a
/// mask bone pass an empty map.
pub(crate) fn resolve_modifier_object_operands(
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
    bone_matrices: &BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
) -> Result<BTreeMap<String, Vec<crate::geom::modifiers::attributes::SceneOperand>>> {
    let reference_parameters: &[&str] = match modifier.modifier_type.as_str() {
        "boolean" | "curve" | "hook" | "mesh_deform" | "data_transfer" => &["object"],
        "uv_project" => &["projectors"],
        "uv_warp" | "warp" => &["object_from", "object_to"],
        "shrinkwrap" => &["target", "object"],
        "vertex_weight_edit" | "vertex_weight_mix" => &["mask_tex_map_object"],
        "vertex_weight_proximity" => &["target", "mask_tex_map_object"],
        "normal_edit" | "surface_deform" => &["target"],
        "volume_displace" => &["texture_map_object"],
        "mask" if modifier.params.get("mode").and_then(Value::as_str) == Some("ARMATURE") => {
            &["armature"]
        }
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
            Value::String(text) => vec![text.as_str()],
            Value::Array(values) if *parameter == "projectors" => values
                .iter()
                .take(uv_project_slot_count(modifier))
                .filter(|value| !value.is_null())
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
        for object_text in object_texts.into_iter().filter(|text| !text.is_empty()) {
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
            if modifier.modifier_type == "data_transfer"
                && data_block.is_none_or(|data| data.data_type != "mesh")
            {
                operands.clear();
                break;
            }
            let needs_operand_mesh = match modifier.modifier_type.as_str() {
                "normal_edit" | "uv_project" | "vertex_weight_edit" | "vertex_weight_mix"
                | "mask" => false,
                "vertex_weight_proximity" => proximity_uses_geometry_mesh(modifier),
                _ => true,
            };
            let operand_mesh = if !needs_operand_mesh {
                Mesh::default()
            } else if let Some(evaluated_mesh) = evaluated_meshes.get(&object_id) {
                evaluated_mesh.clone()
            } else {
                operand_object_mesh(modifier, &object_id, data_block)?
            };
            let object_world = world_matrices.get(&object_id).copied().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "modifier operand world matrix is missing",
                    json!({"modifier_id":modifier.id,"object":object_id}),
                )
            })?;
            let referenced = ReferencedObject {
                id: &object_id,
                node: object,
                world: object_world,
            };
            let blender_matrices_f32 = operand_blender_matrices_f32(
                modifier,
                parameter,
                &referenced,
                subject_world,
                doc,
                bone_matrices,
            );
            let armature = armature_data_for_modifier_operand(object, doc);
            operands.push(crate::geom::modifiers::attributes::SceneOperand {
                mesh: operand_mesh,
                local_to_subject: Some(subject_inverse * object_world),
                blender_matrices_f32,
                camera: data_block.and_then(|data| data.camera.clone()),
                armature: object.kind == "armature"
                    || data_block.is_some_and(|data| data.data_type == "armature"),
                object_scale: DVec3::from_array(object.transform.scale),
                selected_bones: armature.map_or_else(BTreeSet::new, |armature| {
                    armature
                        .bones
                        .values()
                        .filter(|bone| bone.select)
                        .map(|bone| bone.name.clone())
                        .collect()
                }),
                deforming_bones: armature.map_or_else(BTreeSet::new, |armature| {
                    armature
                        .bones
                        .values()
                        .filter(|bone| bone.deform)
                        .map(|bone| bone.name.clone())
                        .collect()
                }),
                layer_selection: crate::geom::modifiers::attributes::AttributeLayerSelection {
                    vertex_group_names: data_block.map_or_else(Vec::new, |data| {
                        data.vertex_groups
                            .iter()
                            .map(|group| group.name.clone())
                            .collect()
                    }),
                    active_vertex_group_index: object.active_vertex_group_index,
                },
            });
        }
        resolved.insert((*parameter).to_owned(), operands);
    }
    Ok(resolved)
}

/// Resolve a `uv_warp` reference to its Blender f32 world matrix. An unset, non-armature, or
/// bone-less reference uses the object's own world matrix, as Blender's UV Warp does; a named
/// bone uses that pose bone's world matrix.
fn resolve_uv_warp_reference_world_f32(
    modifier: &Modifier,
    object_parameter: &str,
    bone_parameter: &str,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    bone_matrices: &BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
) -> Result<Option<[f32; 16]>> {
    let object_text = match modifier.params.get(object_parameter) {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(text)) if text.is_empty() => return Ok(None),
        Some(Value::String(text)) => text.as_str(),
        Some(_) => {
            return Err(PotError::invalid_argument(format!(
                "UV Warp `{object_parameter}` must be an object ID or null"
            )));
        }
    };
    let object_id = Id::new(object_text.to_owned()).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            error.message,
            json!({"modifier_id":modifier.id,"parameter":object_parameter,"object":object_text}),
        )
    })?;
    let object = doc.nodes.get(&object_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("UV Warp operand object `{object_id}` was not found"),
            json!({"modifier_id":modifier.id,"parameter":object_parameter,"object":object_id}),
        )
    })?;
    let object_world = world_matrices.get(&object_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "UV Warp operand world matrix is missing",
            json!({"modifier_id":modifier.id,"parameter":object_parameter,"object":object_id}),
        )
    })?;
    let object_world_f32 = object_world.to_cols_array().map(|value| value as f32);
    let Some(bone_name) = non_empty_text(modifier, bone_parameter) else {
        return Ok(Some(object_world_f32));
    };
    if object.kind != "armature" {
        return Ok(Some(object_world_f32));
    }
    let Some(bone_id) = armature_bone_id(object, doc, bone_name) else {
        return Ok(Some(object_world_f32));
    };
    let bone_world = bone_matrices
        .get(&object_id)
        .and_then(|bones| bones.get(&bone_id))
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "UV Warp pose-bone matrix is missing",
                json!({"modifier_id":modifier.id,"parameter":bone_parameter,"object":object_id,"bone":bone_name}),
            )
        })?;
    Ok(Some(bone_world.map(|value| value as f32)))
}

fn matrix_f32_columns(matrix: [f32; 16]) -> [[f32; 4]; 4] {
    std::array::from_fn(|column| std::array::from_fn(|row| matrix[column * 4 + row]))
}

/// The subject node's name, which attribute and modifier evaluation uses as the subject identity.
fn scene_subject_name<'a>(doc: &'a SceneDoc, subject_id: &Id) -> Result<&'a str> {
    doc.nodes
        .get(subject_id)
        .map(|node| node.name.as_str())
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "modifier subject object is missing",
                json!({"subject_id": subject_id}),
            )
        })
}

/// The object-to-subject matrix of a resolved operand. The resolver always sets it, so a missing
/// matrix is an internal fault rather than an identity to fall back on.
fn operand_local_to_subject(
    operand: &crate::geom::modifiers::attributes::SceneOperand,
) -> Result<DMat4> {
    operand.local_to_subject.ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "scene operand is missing its object-to-subject transform",
        )
    })
}

/// Evaluate an attribute modifier against its resolved scene operands. `bone_matrices` supplies
/// the pose-bone world matrices that `uv_warp` bone references read; `bone_pose_matrices` supplies
/// the raw pose matrices that texture mapping through `mask_tex_map_bone` reads. `project_root`
/// resolves image mask textures.
pub(crate) fn apply_scene_attribute_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
    bone_matrices: &BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
    bone_pose_matrices: &BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
    project_root: Option<&Path>,
) -> Result<()> {
    let mut uv_warp_modifier = None;
    if modifier.modifier_type == "uv_warp" {
        let source_world = resolve_uv_warp_reference_world_f32(
            modifier,
            "object_from",
            "bone_from",
            doc,
            world_matrices,
            bone_matrices,
        )?;
        let destination_world = resolve_uv_warp_reference_world_f32(
            modifier,
            "object_to",
            "bone_to",
            doc,
            world_matrices,
            bone_matrices,
        )?;
        let matrix = crate::geom::modifiers::attributes::build_uv_warp_matrix_f32(
            source_world,
            destination_world,
            modifier,
        )?;
        let mut resolved = modifier.clone();
        resolved.runtime.uv_warp_matrix = Some(matrix_f32_columns(matrix));
        uv_warp_modifier = Some(resolved);
    }
    let modifier = uv_warp_modifier.as_ref().unwrap_or(modifier);
    let operands = resolve_modifier_object_operands(
        modifier,
        subject_id,
        doc,
        world_matrices,
        evaluated_meshes,
        bone_matrices,
    )?;
    let subject_world = world_matrices.get(subject_id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "modifier subject world matrix is missing",
        )
    })?;
    let subject = doc.nodes.get(subject_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "modifier subject object is missing",
            json!({"modifier_id": modifier.id, "subject": subject_id}),
        )
    })?;
    let subject_data = subject
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id));
    let subject_layer_selection = crate::geom::modifiers::attributes::AttributeLayerSelection {
        vertex_group_names: subject_data.map_or_else(Vec::new, |data| {
            data.vertex_groups
                .iter()
                .map(|group| group.name.clone())
                .collect()
        }),
        active_vertex_group_index: subject.active_vertex_group_index,
    };
    let (texture, texture_mapping) = match attribute_mask_texture_id(modifier)? {
        None => (None, None),
        Some(texture_id) => {
            let texture =
                crate::geom::legacy_texture::resolve_texture(doc, &texture_id, project_root)?;
            let mapping = texture_coordinate_mapping_with_parameters(
                modifier,
                subject_world,
                doc,
                world_matrices,
                bone_pose_matrices,
                "mask_tex_mapping",
                "mask_tex_map_object",
                Some("mask_tex_map_bone"),
            )?;
            (Some(texture), mapping)
        }
    };
    *mesh = crate::geom::modifiers::attributes::evaluate_with_context(
        mesh,
        modifier,
        &operands,
        texture.as_ref(),
        texture_mapping.as_ref(),
        subject_world,
        &subject.name,
        &subject_layer_selection,
    )?;
    Ok(())
}

/// The `mask_texture` ID of an attribute modifier. An unset or empty parameter has no texture.
fn attribute_mask_texture_id(modifier: &Modifier) -> Result<Option<Id>> {
    let text = match modifier.params.get("mask_texture") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(text)) if text.is_empty() => return Ok(None),
        Some(Value::String(text)) => text,
        Some(_) => {
            return Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                "mask_texture must be a texture ID or null",
                json!({"modifier_id":modifier.id,"parameter":"mask_texture"}),
            ));
        }
    };
    Id::new(text.to_owned()).map(Some).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            error.message,
            json!({"modifier_id":modifier.id,"parameter":"mask_texture","texture":text}),
        )
    })
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
        &BTreeMap::new(),
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
    let local_to_subject = operand_local_to_subject(operand)?;
    let mut target_mesh = operand.mesh.clone();
    for vertex in &mut target_mesh.vertices {
        vertex.co = local_to_subject.transform_point3(vertex.co);
    }
    let mut resolved = modifier.clone();
    resolved.params.remove(parameter);
    resolved.runtime.target_to_subject = Some(local_to_subject.to_cols_array());
    resolved.runtime.target_mesh = Some(target_mesh);
    let evaluation = crate::geom::modifiers::evaluate_modifiers_with_groups(
        mesh,
        &[resolved],
        &doc.node_groups,
        scene_subject_name(doc, subject_id)?,
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
) -> Result<Vec<PotError>> {
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
    let is_sequence = doc
        .resources
        .get(&resource_id)
        .and_then(|resource| resource.get("cache_file"))
        .and_then(|settings| settings.get("is_sequence"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    crate::geom::mesh_cache::evaluate(
        mesh,
        modifier,
        frame,
        fps,
        fps_base,
        &actual_hash,
        &bytes,
        is_sequence,
    )
}

const MAX_RENDERED_PARTICLES: usize = 100_000;

fn render_simulated_particles(
    doc: &SceneDoc,
    gate: ModifierGate<'_>,
    particles: &BTreeMap<Id, Vec<crate::sim::ParticleState>>,
    matrices: &BTreeMap<Id, DMat4>,
    meshes: &mut BTreeMap<Id, Mesh>,
    nodes: &mut BTreeMap<Id, EvaluatedNode>,
    evaluated_node_ids: &BTreeSet<Id>,
    adaptive_normals: &mut BTreeMap<Id, Vec<Option<DVec3>>>,
) -> Result<()> {
    for (emitter_id, states) in particles {
        if !evaluated_node_ids.contains(emitter_id) {
            continue;
        }
        if doc.nodes.get(emitter_id).is_some_and(|node| {
            node.modifiers.iter().any(|modifier| {
                gate.is_active(emitter_id, modifier) && modifier.modifier_type == "explode"
            })
        }) {
            continue;
        }
        let node = doc.nodes.get(emitter_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "particle emitter object was not found",
                json!({"node_id":emitter_id}),
            )
        })?;
        let primary_modifier_id = node
            .properties
            .get("physics_particle_emitter_modifier_id")
            .and_then(serde_json::Value::as_str);
        let mut visible_states_by_system = BTreeMap::<u32, Vec<&crate::sim::ParticleState>>::new();
        for state in states
            .iter()
            .filter(|state| state.life_state == crate::sim::ParticleLifeState::Alive)
        {
            let system_modifier = usize::try_from(state.system_index).ok().and_then(|index| {
                gate.stack_modifiers(emitter_id, &node.modifiers)
                    .iter()
                    .filter(|modifier| modifier.modifier_type == "particle_system")
                    .nth(index)
            });
            let active = match (system_modifier, primary_modifier_id) {
                (Some(modifier), Some(primary_id)) if modifier.id.as_str() == primary_id => gate
                    .is_property_active(
                        emitter_id,
                        node,
                        "physics_particle_emitter",
                        "particle_system",
                    ),
                (Some(_), None) if state.system_index == 0 => gate.is_property_active(
                    emitter_id,
                    node,
                    "physics_particle_emitter",
                    "particle_system",
                ),
                (Some(modifier), _) => gate.is_active(emitter_id, modifier),
                (None, _) => gate.is_property_active(
                    emitter_id,
                    node,
                    "physics_particle_emitter",
                    "particle_system",
                ),
            };
            if active {
                visible_states_by_system
                    .entry(state.system_index)
                    .or_default()
                    .push(state);
            }
        }
        let particle_count = visible_states_by_system
            .values()
            .fold(0_usize, |count, states| count.saturating_add(states.len()));
        if particle_count > MAX_RENDERED_PARTICLES {
            return Err(PotError::with_details(
                ErrorCode::LimitExceeded,
                "particle visualization exceeds the 100000 particle limit",
                json!({"node_id":emitter_id,"count":particle_count}),
            ));
        }
        for (system_index, visible_states) in visible_states_by_system {
            let has_stack_system = usize::try_from(system_index).is_ok_and(|index| {
                gate.stack_modifiers(emitter_id, &node.modifiers)
                    .iter()
                    .filter(|modifier| modifier.modifier_type == "particle_system")
                    .nth(index)
                    .is_some()
            });
            let settings = (if has_stack_system {
                particle_system_for_index(doc, emitter_id, system_index, gate)?.1
            } else {
                node.properties
                    .get("physics_particle_emitter")
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "particle emitter settings are missing",
                            json!({"node_id":emitter_id}),
                        )
                    })?
            })
            .as_object()
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
                    let object_id = Id::new(object_id.to_owned()).map_err(|error| {
                        PotError::new(ErrorCode::InvalidOperation, error.message)
                    })?;
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
                    let collection_id = Id::new(collection_text.to_owned()).map_err(|error| {
                        PotError::new(ErrorCode::InvalidOperation, error.message)
                    })?;
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
                        append_transformed_mesh(
                            &mut template,
                            source,
                            emitter_inverse * object_world,
                        )?;
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
            adaptive_normals.remove(emitter_id);
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
    }
    Ok(())
}

fn has_stack_modifier(
    gate: ModifierGate<'_>,
    node_id: &Id,
    node: &Node,
    modifier_type: &str,
) -> bool {
    gate.stack_modifiers(node_id, &node.modifiers)
        .iter()
        .any(|modifier| modifier.modifier_type == modifier_type)
}

fn apply_stack_physics_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    node_id: &Id,
    doc: &SceneDoc,
    scene_hash: &str,
    gate: ModifierGate<'_>,
    scene_id: &Id,
    frame: f64,
    matrices: &BTreeMap<Id, DMat4>,
    particles: &mut BTreeMap<Id, Vec<crate::sim::ParticleState>>,
) -> Result<()> {
    let result = crate::sim::evaluate_stack_modifier(
        doc,
        scene_id,
        node_id,
        modifier,
        std::mem::replace(mesh, Mesh::new()),
        frame,
        matrices,
        scene_hash,
        gate,
    )?;
    if let Some(mut states) = result.particles {
        let system_index = doc
            .nodes
            .get(node_id)
            .into_iter()
            .flat_map(|node| gate.stack_modifiers(node_id, &node.modifiers).iter())
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
    gate: ModifierGate<'_>,
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
    let (system_modifier, _) = particle_system_for_index(doc, &object_id, index, gate)?;
    let particle_states = if gate.is_active(&object_id, system_modifier) {
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
    gate: ModifierGate<'_>,
    frame: f64,
    matrices: &BTreeMap<Id, DMat4>,
    particles: &BTreeMap<Id, Vec<crate::sim::ParticleState>>,
) -> Result<()> {
    let index = particle_system_index(modifier)?;
    let (system_modifier, _) = particle_system_for_index(doc, node_id, index, gate)?;
    let particle_states = if gate.is_active(node_id, system_modifier) {
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
    gate: ModifierGate<'_>,
) -> Result<(&'a Modifier, &'a serde_json::Value)> {
    let node = doc.nodes.get(node_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::TargetNotFound,
            format!("particle system object `{node_id}` was not found"),
        )
    })?;
    let system_modifier = gate
        .stack_modifiers(node_id, &node.modifiers)
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
        &BTreeMap::new(),
    )?;
    let object_parameter =
        if modifier.modifier_type == "shrinkwrap" && modifier.params.contains_key("target") {
            "target"
        } else {
            "object"
        };
    let mut combined = None;
    let mut curve_object_matrix = None;
    for operand in operands.get(object_parameter).into_iter().flatten() {
        let local_to_subject = operand_local_to_subject(operand)?;
        if modifier.modifier_type == "curve" && curve_object_matrix.is_none() {
            curve_object_matrix = Some(local_to_subject.to_cols_array());
        }
        let mut transformed = operand.mesh.clone();
        for vertex in &mut transformed.vertices {
            vertex.co = local_to_subject.transform_point3(vertex.co);
        }
        combined = Some(match combined {
            None => transformed,
            Some(existing) => combine_scene_operand_meshes(&existing, &transformed)?,
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
    resolved.runtime.target_mesh = Some(operand);
    if let Some(matrix) = curve_object_matrix {
        resolved.runtime.object_matrix = Some(matrix);
    }
    *mesh = crate::geom::modifiers::evaluate_modifiers(
        mesh,
        &[resolved],
        scene_subject_name(doc, subject_id)?,
    )?;
    Ok(())
}

pub(crate) fn apply_scene_operand_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    references: &SceneReferences<'_>,
    evaluated_nodes: &BTreeMap<Id, EvaluatedNode>,
    material_slots: &mut Vec<Option<Id>>,
) -> Result<()> {
    match modifier.modifier_type.as_str() {
        "hook" => return apply_scene_hook_modifier(mesh, modifier, subject_id, doc, references),
        "cast" => {
            return apply_scene_single_reference_modifier(
                mesh, modifier, subject_id, doc, references, "object", None,
            );
        }
        "simple_deform" => {
            return apply_scene_single_reference_modifier(
                mesh, modifier, subject_id, doc, references, "origin", None,
            );
        }
        _ => {}
    }
    let world_matrices = references.world_matrices;
    let evaluated_meshes = references.evaluated_meshes;
    if modifier.modifier_type == "boolean" {
        *material_slots = apply_boolean_operands_modifier(
            mesh,
            modifier,
            subject_id,
            doc,
            world_matrices,
            evaluated_meshes,
            evaluated_nodes,
            material_slots.as_slice(),
        )?;
        return Ok(());
    }
    if matches!(modifier.modifier_type.as_str(), "shrinkwrap" | "curve")
        && !modifier.params.contains_key("collection")
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
            Some(existing) => combine_scene_operand_meshes(&existing, &operand)?,
        });
    }
    let mut resolved = modifier.clone();
    resolved.params.remove("object");
    resolved.params.remove("collection");
    resolved.params.remove("target");
    let operand =
        combined.ok_or_else(|| PotError::invalid_argument("modifier operand is empty"))?;
    resolved.runtime.target_mesh = Some(operand);
    *mesh = crate::geom::modifiers::evaluate_modifiers(
        mesh,
        &[resolved],
        scene_subject_name(doc, subject_id)?,
    )?;
    Ok(())
}
/// Object IDs that a Boolean modifier takes as operands for `operand_type`: the `object` parameter,
/// or, for COLLECTION, every object of the `collection` parameter in collection order. A missing or
/// null parameter has no operands; a named object that does not exist is an error when it is read.
fn boolean_operand_object_ids(
    modifier: &Modifier,
    doc: &SceneDoc,
    operand_type: crate::geom::modifiers::BooleanOperandType,
) -> Result<Vec<Id>> {
    use crate::geom::modifiers::BooleanOperandType;
    let (parameter, kind) = match operand_type {
        BooleanOperandType::Object => ("object", "object"),
        BooleanOperandType::Collection => ("collection", "collection"),
    };
    let text = match modifier.params.get(parameter) {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(value) => value.as_str().ok_or_else(|| {
            PotError::invalid_argument(format!(
                "modifier `{}` `{parameter}` must be a {kind} ID",
                modifier.modifier_type
            ))
        })?,
    };
    let id = Id::new(text).map_err(|_| {
        PotError::invalid_argument(format!(
            "modifier `{}` `{parameter}` must be a valid {kind} ID",
            modifier.modifier_type
        ))
    })?;
    if operand_type == BooleanOperandType::Object {
        return Ok(vec![id]);
    }
    let mut object_ids = Vec::new();
    collect_collection_objects_in_order(doc, &id, &mut BTreeSet::new(), &mut object_ids)?;
    Ok(object_ids)
}

/// Blender `FOREACH_COLLECTION_OBJECT_RECURSIVE` order: a collection's objects, then its child
/// collections, each object once.
fn collect_collection_objects_in_order(
    doc: &SceneDoc,
    collection_id: &Id,
    visited: &mut BTreeSet<Id>,
    object_ids: &mut Vec<Id>,
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
    for object_id in &collection.objects {
        if !object_ids.contains(object_id) {
            object_ids.push(object_id.clone());
        }
    }
    for child in &collection.children {
        collect_collection_objects_in_order(doc, child, visited, object_ids)?;
    }
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

/// The f64 fallback of the Boolean operand transform: the world matrix quantized to f32 before the
/// product (`world_to_object * object_world` in `non_float_boolean_mesh`). The exact f32 path is
/// `geom::blender_matrix::operand_to_subject`.
fn world_matrix_f32(matrix: DMat4) -> glam::Mat4 {
    glam::Mat4::from_cols_array(&matrix.to_cols_array().map(|value| value as f32))
}

type BooleanOperandParts<'d> = (
    crate::geom::boolean::BooleanOperand,
    Vec<Option<Id>>,
    Vec<Option<Id>>,
    &'d [VertexGroup],
);

/// One Boolean operand object: its mesh (the evaluated mesh when present, else the data mesh with
/// its vertex groups) and its transform into the subject's space: Blender's f32 product from
/// `geom::blender_matrix` when it applies, else the f64 fallback. Also returns its effective and
/// mesh-level material slots, and its DATA vertex groups.
fn boolean_operand<'d>(
    modifier: &Modifier,
    doc: &'d SceneDoc,
    object_id: &Id,
    subject_inverse: glam::Mat4,
    subject_id: &Id,
    subject_negative: bool,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
    evaluated_nodes: &BTreeMap<Id, EvaluatedNode>,
) -> Result<BooleanOperandParts<'d>> {
    let object = doc.nodes.get(object_id).ok_or_else(|| {
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
    if data.data_type != "mesh" {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Boolean operands must be mesh objects",
            json!({"feature_id":"modifier.boolean","modifier_id":modifier.id,"object":object_id}),
        ));
    }
    let mesh = if let Some(evaluated) = evaluated_meshes.get(object_id) {
        evaluated.clone()
    } else {
        let mut base = data.mesh.clone().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "mesh modifier operand has no mesh payload",
                json!({"modifier_id":modifier.id,"object":object_id}),
            )
        })?;
        attach_vertex_group_weights(&mut base, data);
        base
    };
    let object_world = world_matrices.get(object_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "modifier operand world matrix is missing",
            json!({"modifier_id":modifier.id,"object":object_id}),
        )
    })?;
    // Blender's f32 product `world_to_object(subject) * object_to_world(operand)`; the f64
    // fallback applies when either object has no exact replica.
    let to_subject = if let Some(exact) =
        crate::geom::blender_matrix::operand_to_subject(doc, subject_id, object_id, world_matrices)
    {
        exact
    } else {
        let to_subject = subject_inverse * world_matrix_f32(object_world);
        DMat4::from_cols_array(&to_subject.to_cols_array().map(f64::from))
    };
    let operand = crate::geom::boolean::BooleanOperand {
        mesh,
        to_subject,
        float_flip: Some(
            subject_negative != crate::geom::blender_matrix::is_negative_m4(object_world),
        ),
        material_remap: Vec::new(),
    };
    // An evaluated operand contributes its evaluated tables (after its own modifiers, e.g. a TRANSFER
    // pool). `slots` are effective, which its face material indices resolve through (remap); `mesh_slots`
    // is its mesh array, which a quick UNION copies. An operand without an evaluated mesh contributes
    // its authoring tables.
    let (slots, mesh_slots) = match evaluated_nodes.get(object_id) {
        Some(node) if evaluated_meshes.contains_key(object_id) => {
            (node.materials.clone(), node.mesh_materials.clone())
        }
        _ => (
            effective_material_slots(object, &data.materials),
            data.materials.clone(),
        ),
    };
    Ok((operand, slots, mesh_slots, data.vertex_groups.as_slice()))
}

/// Re-keys an operand's vertex-group weights from the operand's group names to the subject's, by
/// group position. Operand groups without a subject counterpart are dropped.
fn remap_float_operand_vertex_groups(
    mesh: &mut Mesh,
    operand_groups: &[VertexGroup],
    subject_groups: &[VertexGroup],
) {
    let Some(by_name) = mesh
        .attributes
        .get_mut("vertex_groups")
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    let mut source = std::mem::take(by_name);
    let mut remapped = Map::new();
    for (index, operand_group) in operand_groups.iter().enumerate() {
        let Some(subject_group) = subject_groups.get(index) else {
            break;
        };
        if let Some(weights) = source.remove(&operand_group.name) {
            remapped.insert(subject_group.name.clone(), weights);
        }
    }
    *by_name = remapped;
}

/// The Boolean operands of one modifier in Blender's order. The subject is operand 0 and is not
/// listed; each operand object keeps its own mesh and transform, and the modifier combines them in
/// one N-ary call, so a collection is never folded pairwise. Returns the output material slots.
///
/// EXACT returns the subject's DATA slots for INDEX, or the TRANSFER pool. FLOAT always remaps by
/// INDEX against the subject's effective (OBJECT-resolved) slots, and returns the subject's DATA
/// slots unchanged: the effective table of the result is the one it remapped against. A FLOAT
/// object shortcut that adopts the operand's mesh returns the operand's effective slots, which its
/// face indices refer to.
fn apply_boolean_operands_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    evaluated_meshes: &BTreeMap<Id, Mesh>,
    evaluated_nodes: &BTreeMap<Id, EvaluatedNode>,
    subject_slots: &[Option<Id>],
) -> Result<Vec<Option<Id>>> {
    use crate::geom::modifiers::{BooleanOperandType, BooleanSolver};

    let settings = crate::geom::modifiers::BooleanSettings::parse(modifier)?;
    let float = settings.solver == BooleanSolver::Float;
    crate::geom::modifiers::validate_boolean_float_collection(modifier, settings, false)?;
    let object_ids = boolean_operand_object_ids(modifier, doc, settings.operand_type)?;
    crate::geom::modifiers::validate_boolean_float_collection(
        modifier,
        settings,
        object_ids.is_empty(),
    )?;
    // Blender `non_float_boolean_mesh` returns the mesh unchanged for an OBJECT operand that is
    // unset (MOD_boolean.cc:406), so neither geometry nor material slots change.
    if settings.operand_type == BooleanOperandType::Object && object_ids.is_empty() {
        return Ok(subject_slots.to_vec());
    }
    let subject_world = world_matrices.get(subject_id).copied().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "modifier subject world matrix is missing",
        )
    })?;
    // The f64 fallback inverse; operands with an exact f32 replica skip it (`geom::blender_matrix`).
    let subject_inverse = world_matrix_f32(subject_world).inverse();
    if !subject_inverse.is_finite() {
        return Err(PotError::invalid_argument(
            "modifier subject transform must be invertible",
        ));
    }
    let subject_node = doc.nodes.get(subject_id).ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "modifier subject node is missing")
    })?;
    let subject_negative = crate::geom::blender_matrix::is_negative_m4(subject_world);
    let subject_groups = subject_node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .map_or(&[][..], |data| data.vertex_groups.as_slice());
    let mut operands = Vec::new();
    let mut operand_slots = Vec::new();
    let mut operand_mesh_slots = Vec::new();
    for object_id in object_ids
        .iter()
        .filter(|object_id| *object_id != subject_id)
    {
        let (mut operand, slots, mesh_slots, operand_groups) = boolean_operand(
            modifier,
            doc,
            object_id,
            subject_inverse,
            subject_id,
            subject_negative,
            world_matrices,
            evaluated_meshes,
            evaluated_nodes,
        )?;
        if float {
            remap_float_operand_vertex_groups(&mut operand.mesh, operand_groups, subject_groups);
        }
        operands.push(operand);
        operand_slots.push(slots);
        operand_mesh_slots.push(mesh_slots);
    }
    // FLOAT leaves the mesh unchanged when the subject is the only member of a collection.
    if float && settings.operand_type == BooleanOperandType::Collection && operands.is_empty() {
        return Ok(subject_slots.to_vec());
    }
    if float
        && settings.operand_type == BooleanOperandType::Object
        && let [operand] = operands.as_slice()
        && let Some(quick) =
            crate::geom::boolean::float_object_quick_mesh(mesh, operand, settings.operation)
    {
        return Ok(match quick {
            crate::geom::boolean::FloatObjectQuickMesh::Subject => subject_slots.to_vec(),
            crate::geom::boolean::FloatObjectQuickMesh::Empty => {
                *mesh = Mesh::new();
                Vec::new()
            }
            crate::geom::boolean::FloatObjectQuickMesh::Operand(result) => {
                *mesh = result;
                // Blender copies the operand's mesh, so its own material array (not its effective
                // OBJECT-resolved table) becomes the result's.
                operand_mesh_slots.first().cloned().unwrap_or_default()
            }
        });
    }
    let material_mode = if float {
        crate::geom::boolean::MaterialMode::Index
    } else {
        settings.material_mode
    };
    let effective_subject;
    let subject_table = if float {
        effective_subject = effective_material_slots(subject_node, subject_slots);
        effective_subject.as_slice()
    } else {
        subject_slots
    };
    // Blender `non_float_boolean_mesh` (source/blender/modifiers/intern/MOD_boolean.cc, v5.2.2):
    // the TRANSFER pool starts from the subject mesh's own material array (`mesh->mat`, the DATA
    // slots, here `subject_slots`), and each operand's slots come from `get_material_remap_transfer`,
    // which reads `BKE_object_material_get_eval` (blenkernel/intern/material.cc:772), i.e. the
    // effective slots `operand_slots` holds. The subject keeps its own indices in both modes.
    let subject_refs: Vec<Option<&Id>> = subject_table.iter().map(Option::as_ref).collect();
    let operand_refs: Vec<Vec<Option<&Id>>> = operand_slots
        .iter()
        .map(|slots| slots.iter().map(Option::as_ref).collect())
        .collect();
    let (remaps, pool) =
        crate::geom::boolean::material_remaps(material_mode, &subject_refs, &operand_refs);
    for (operand, remap) in operands.iter_mut().zip(remaps.into_iter().skip(1)) {
        operand.material_remap = remap;
    }
    let mut resolved = modifier.clone();
    for parameter in ["object", "collection", "target"] {
        resolved.params.remove(parameter);
    }
    resolved.runtime.boolean_operands = operands;
    *mesh = crate::geom::modifiers::evaluate_modifiers(
        mesh,
        &[resolved],
        scene_subject_name(doc, subject_id)?,
    )?;
    Ok(if float { subject_slots.to_vec() } else { pool })
}

fn combine_scene_operand_meshes(first: &Mesh, second: &Mesh) -> Result<Mesh> {
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

pub(crate) fn apply_lattice_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    source_node_id: &Id,
    source_data: &DataBlock,
    doc: &SceneDoc,
    matrices: &BTreeMap<Id, DMat4>,
) -> Result<()> {
    // Blender disables the modifier when no lattice object is set.
    let Some(lattice_id) = reference_object_id(modifier, "object", doc)? else {
        return Ok(());
    };
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
        PotError::with_details(
            ErrorCode::DependencyMissing,
            "lattice modifier subject transform is unavailable",
            json!({"modifier_id":modifier.id,"node_id":source_node_id}),
        )
    })?;
    let lattice_world = matrices.get(&lattice_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            "lattice modifier lattice transform is unavailable",
            json!({"modifier_id":modifier.id,"node_id":lattice_id}),
        )
    })?;
    let source_inverse = source_world.inverse();
    if !source_inverse.is_finite() {
        return Err(PotError::invalid_argument(
            "lattice modifier source transform must be invertible",
        ));
    }
    // The kernel maps lattice-space offsets through inv(subject) * lattice, as Blender's latmat does.
    let lattice_to_mesh = source_inverse * lattice_world;
    let strength = modifier_number_param(modifier, "strength")?;
    let weights = lattice_vertex_group_weights(mesh, modifier, source_data)?
        .unwrap_or_else(|| vec![1.0; mesh.vertices.len()]);
    let fully_deformed =
        crate::geom::lattice::deform_mesh_with_transform(mesh, lattice, lattice_to_mesh)?;
    // The full-weight result scaled by strength * weight is Blender's per-vertex blend.
    for ((vertex, deformed), weight) in mesh
        .vertices
        .iter_mut()
        .zip(&fully_deformed.vertices)
        .zip(weights)
    {
        vertex.co += (deformed.co - vertex.co) * (strength * weight);
    }
    Ok(())
}

fn lattice_vertex_group_weights(
    mesh: &Mesh,
    modifier: &Modifier,
    source_data: &DataBlock,
) -> Result<Option<Vec<f64>>> {
    let name = match modifier.params.get("vertex_group") {
        None | Some(Value::Null) => "",
        Some(Value::String(name)) => name.as_str(),
        Some(_) => {
            return Err(invalid_reference(modifier, "vertex_group", "a group name"));
        }
    };
    if name.is_empty() {
        return Ok(None);
    }
    let mut matches = source_data
        .vertex_groups
        .iter()
        .filter(|group| group.id.as_str() == name || group.name == name);
    let Some(group) = matches.next() else {
        // Blender reads a missing group as full influence.
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            format!("vertex group name `{name}` is ambiguous"),
            json!({"modifier_id":modifier.id,"vertex_group":name}),
        ));
    }
    let invert = modifier_bool_param(modifier, "invert_vertex_group")?;
    let weights_by_vertex = mesh_vertex_weights(mesh, &source_data.vertex_groups);
    if weights_by_vertex.values().all(BTreeMap::is_empty) {
        return Ok(None);
    }
    Ok(Some(
        mesh.vertices
            .iter()
            .map(|vertex| {
                let weight = weights_by_vertex
                    .get(&vertex.id)
                    .and_then(|groups| groups.get(&group.id))
                    .copied()
                    .unwrap_or(0.0);
                if invert { 1.0 - weight } else { weight }
            })
            .collect(),
    ))
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

/// Marks a collection instance's join failure with the instance as its owner, so evaluation isolates
/// that instance alone.
const INSTANCE_JOIN_FAILURE: &str = "collection_instance_join_failed";

/// A collection member before joining: its mesh and member-local UV/color bindings.
struct CollectionMember<'a> {
    mesh: Cow<'a, Mesh>,
    matrix: DMat4,
    slots: Vec<SlotBinding>,
    adaptive_normals: Option<Cow<'a, [Option<DVec3>]>>,
    uv_names: Vec<String>,
    uv_coverage: Vec<u8>,
    color_names: Vec<String>,
}

/// A joined collection instance and the source bindings/coverage to propagate through nesting.
struct CollectionEvaluation {
    mesh: Mesh,
    paths: Vec<Vec<Id>>,
    bindings: Vec<SlotBinding>,
    uv_coverage: Vec<u8>,
    adaptive_normals: Option<Vec<Option<DVec3>>>,
}

struct JoinedMemberBindings {
    slots: Vec<SlotBinding>,
    uv_names: Vec<String>,
    uv_coverage: Vec<u8>,
    color_names: Vec<String>,
    slot_base: usize,
}
type ColorMaskRemapCache = HashMap<*const [u64], (Arc<[u64]>, Arc<[u64]>)>;

/// A plain member's slot table: its material list, or one empty slot when it has none.
fn member_slots(materials: &[Option<Id>]) -> Vec<Option<Id>> {
    if materials.is_empty() {
        vec![None]
    } else {
        materials.to_vec()
    }
}

/// Source-face coverage by UV layer, before concatenation fills absent rows with defaults.
fn uv_face_coverage(layers: &[UvLayer], face_count: usize) -> Vec<u8> {
    let mut coverage = vec![0; face_count];
    for (layer_index, layer) in layers.iter().take(uv_layers::MAX_LAYERS).enumerate() {
        let bit = 1_u8 << layer_index;
        for (face_index, row) in layer.faces.iter().enumerate().take(face_count) {
            if row.is_some() {
                coverage[face_index] |= bit;
            }
        }
    }
    coverage
}

fn color_layer_names(mesh: &Mesh) -> Vec<String> {
    mesh.attributes
        .get("color_layers")
        .and_then(|value| value.get("layers"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn all_color_layers_mask(layer_count: usize) -> Arc<[u64]> {
    let mut mask = vec![u64::MAX; layer_count.div_ceil(64)];
    if !layer_count.is_multiple_of(64)
        && let Some(last) = mask.last_mut()
    {
        *last = (1_u64 << (layer_count % 64)) - 1;
    }
    mask.into()
}

fn member_slot_bindings(
    materials: &[Option<Id>],
    mesh: &Mesh,
    uv_store: &uv_layers::UvStore,
    color_names: &[String],
) -> Vec<SlotBinding> {
    let render_index = uv_store
        .render_index()
        .and_then(|index| u8::try_from(index).ok());
    let available_mask = (0..uv_store.layers.len().min(uv_layers::MAX_LAYERS))
        .fold(0_u8, |mask, index| mask | (1_u8 << index));
    let color_render_layer = mesh
        .attributes
        .get("color_layers")
        .and_then(|value| value.get("render"))
        .and_then(Value::as_str)
        .and_then(|name| color_names.iter().position(|layer| layer == name));
    let color_available_mask = all_color_layers_mask(color_names.len());
    member_slots(materials)
        .into_iter()
        .map(|material| SlotBinding {
            material,
            uv: UvBinding {
                render_index,
                available_mask,
                texcoord0_covered: false,
            },
            color_render_layer,
            color_available_mask: Arc::clone(&color_available_mask),
        })
        .collect()
}

/// Whether every face indexing a joined slot has real data for joined UV layer zero.
fn fold_slot_coverage(mesh: &Mesh, covered: &[u8], slot_count: usize) -> Vec<bool> {
    let mut slots = vec![None::<bool>; slot_count];
    for (face, &face_coverage) in mesh.faces.iter().zip(covered) {
        let Some(entry) = usize::try_from(face.material_index)
            .ok()
            .and_then(|slot| slots.get_mut(slot))
        else {
            continue;
        };
        *entry = Some(entry.unwrap_or(true) && face_coverage & 1 != 0);
    }
    slots
        .into_iter()
        .map(|entry| entry.unwrap_or(false))
        .collect()
}

fn layer_index_map(source: &[String], joined: &[String]) -> Vec<Option<usize>> {
    source
        .iter()
        .map(|name| joined.iter().position(|candidate| candidate == name))
        .collect()
}

fn remap_uv_mask(mask: u8, layer_map: &[Option<usize>]) -> u8 {
    layer_map
        .iter()
        .enumerate()
        .fold(0, |joined, (source, destination)| {
            if mask & (1_u8 << source) == 0 {
                return joined;
            }
            destination
                .and_then(|index| u8::try_from(index).ok())
                .map_or(joined, |index| joined | (1_u8 << index))
        })
}

fn remap_color_mask(mask: &[u64], layer_map: &[Option<usize>], joined_count: usize) -> Arc<[u64]> {
    let mut joined = vec![0; joined_count.div_ceil(64)];
    for (source, destination) in layer_map.iter().enumerate() {
        if mask
            .get(source / 64)
            .is_some_and(|word| word & (1_u64 << (source % 64)) != 0)
            && let Some(destination) = destination
        {
            joined[destination / 64] |= 1_u64 << (destination % 64);
        }
    }
    joined.into()
}

fn remap_coverage(coverage: &[u8], layer_map: &[Option<usize>]) -> Vec<u8> {
    coverage
        .iter()
        .map(|&mask| remap_uv_mask(mask, layer_map))
        .collect()
}

fn remap_slot_binding(
    mut binding: SlotBinding,
    uv_map: &[Option<usize>],
    color_map: &[Option<usize>],
    color_count: usize,
    color_masks: &mut ColorMaskRemapCache,
) -> SlotBinding {
    binding.uv.render_index = binding
        .uv
        .render_index
        .and_then(|index| uv_map.get(usize::from(index)).copied().flatten())
        .and_then(|index| u8::try_from(index).ok());
    binding.uv.available_mask = remap_uv_mask(binding.uv.available_mask, uv_map);
    binding.uv.texcoord0_covered = false;
    binding.color_render_layer = binding
        .color_render_layer
        .and_then(|index| color_map.get(index).copied().flatten());
    let pointer = Arc::as_ptr(&binding.color_available_mask);
    let mapped_mask = Arc::clone(
        &color_masks
            .entry(pointer)
            .or_insert_with(|| {
                let source_mask = Arc::clone(&binding.color_available_mask);
                let mapped = remap_color_mask(&source_mask, color_map, color_count);
                (source_mask, mapped)
            })
            .1,
    );
    binding.color_available_mask = mapped_mask;
    binding
}

/// Joins a collection instance's members and nested instances into one mesh.
fn evaluate_collection_instance<'a>(
    doc: &SceneDoc,
    instance_id: &Id,
    matrices: &BTreeMap<Id, DMat4>,
    meshes: &'a BTreeMap<Id, Mesh>,
    nodes: &BTreeMap<Id, EvaluatedNode>,
    adaptive_normals: &'a BTreeMap<Id, Vec<Option<DVec3>>>,
    active_instances: &mut BTreeSet<Id>,
) -> Result<CollectionEvaluation> {
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
    let mut members = Vec::new();
    let mut paths = Vec::new();
    append_collection_geometry(
        doc,
        instance_id,
        &target,
        matrices,
        meshes,
        adaptive_normals,
        nodes,
        active_instances,
        &mut BTreeSet::new(),
        &mut BTreeSet::new(),
        &mut members,
        &mut paths,
    )?;
    active_instances.remove(instance_id);
    let mut parts = Vec::with_capacity(members.len());
    let mut member_bindings = Vec::with_capacity(members.len());
    let mut slot_base = 0;
    for member in members {
        let material_offset = u32::try_from(slot_base).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "collection instance has too many material slots",
            )
        })?;
        let slot_count = member.slots.len();
        parts.push(Part {
            mesh: member.mesh,
            matrix: member.matrix,
            material_offset,
            slot_count,
            adaptive_normals: member.adaptive_normals,
        });
        member_bindings.push(JoinedMemberBindings {
            slots: member.slots,
            uv_names: member.uv_names,
            uv_coverage: member.uv_coverage,
            color_names: member.color_names,
            slot_base,
        });
        slot_base = slot_base.checked_add(slot_count).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "collection instance has too many material slots",
            )
        })?;
    }
    let include_normals = parts.iter().any(|part| part.adaptive_normals.is_some());
    let (mesh, adaptive_normals) = if include_normals {
        concatenate_with_normals(parts)
    } else {
        concatenate(parts).map(|mesh| (mesh, None))
    }
    .map_err(|error| instance_join_error(error, instance_id))?;
    let joined_uv_names = uv_layers::read_layers(&mesh)
        .into_iter()
        .map(|layer| layer.name)
        .collect::<Vec<_>>();
    let joined_color_names = color_layer_names(&mesh);
    let uv_maps = member_bindings
        .iter()
        .map(|member| layer_index_map(&member.uv_names, &joined_uv_names))
        .collect::<Vec<_>>();
    let color_maps = member_bindings
        .iter()
        .map(|member| layer_index_map(&member.color_names, &joined_color_names))
        .collect::<Vec<_>>();
    let mut uv_coverage = Vec::with_capacity(mesh.faces.len());
    for (member, layer_map) in member_bindings.iter().zip(&uv_maps) {
        uv_coverage.extend(remap_coverage(&member.uv_coverage, layer_map));
    }
    if uv_coverage.len() != mesh.faces.len() {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "collection instance UV coverage does not match its faces",
        ));
    }
    let slot_coverage = fold_slot_coverage(&mesh, &uv_coverage, slot_base);
    let mut color_masks = ColorMaskRemapCache::new();
    let mut bindings = Vec::with_capacity(slot_base);
    for ((member, uv_map), color_map) in member_bindings.into_iter().zip(&uv_maps).zip(&color_maps)
    {
        for (local_slot, binding) in member.slots.into_iter().enumerate() {
            let mut binding = remap_slot_binding(
                binding,
                uv_map,
                color_map,
                joined_color_names.len(),
                &mut color_masks,
            );
            binding.uv.texcoord0_covered = slot_coverage
                .get(member.slot_base + local_slot)
                .copied()
                .unwrap_or(false);
            bindings.push(binding);
        }
    }
    Ok(CollectionEvaluation {
        mesh,
        paths,
        bindings,
        uv_coverage,
        adaptive_normals,
    })
}

/// Tags a join failure with the collection instance that owns it.
fn instance_join_error(mut error: PotError, instance_id: &Id) -> PotError {
    let mut details = match std::mem::take(&mut error.details) {
        Value::Object(details) => details,
        _ => Map::new(),
    };
    details.insert("node_id".to_owned(), json!(instance_id));
    details.insert(INSTANCE_JOIN_FAILURE.to_owned(), json!(true));
    error.details = Value::Object(details);
    error
}

/// Whether evaluation isolates `error`'s owner: an unsupported feature, or a collection instance
/// join failure tagged by [`instance_join_error`].
fn is_isolated_failure(error: &PotError) -> bool {
    error.code == ErrorCode::UnsupportedFeature
        || error.details.get(INSTANCE_JOIN_FAILURE).is_some()
}

#[expect(
    clippy::too_many_arguments,
    reason = "keeps recursive instance traversal state explicit"
)]
fn append_collection_geometry<'a>(
    doc: &SceneDoc,
    instance_id: &Id,
    collection_id: &Id,
    matrices: &BTreeMap<Id, DMat4>,
    meshes: &'a BTreeMap<Id, Mesh>,
    adaptive_normals: &'a BTreeMap<Id, Vec<Option<DVec3>>>,
    nodes: &BTreeMap<Id, EvaluatedNode>,
    active_instances: &mut BTreeSet<Id>,
    visited_collections: &mut BTreeSet<Id>,
    collection_stack: &mut BTreeSet<Id>,
    members: &mut Vec<CollectionMember<'a>>,
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
        let member = if instance_collection_target(source_node)?.is_some() {
            let nested = evaluate_collection_instance(
                doc,
                object_id,
                matrices,
                meshes,
                nodes,
                adaptive_normals,
                active_instances,
            )?;
            let uv_names = uv_layers::read_layers(&nested.mesh)
                .into_iter()
                .map(|layer| layer.name)
                .collect();
            let color_names = color_layer_names(&nested.mesh);
            Some((
                Cow::Owned(nested.mesh),
                nested.paths,
                nested.bindings,
                nested.adaptive_normals.map(Cow::Owned),
                uv_names,
                nested.uv_coverage,
                color_names,
            ))
        } else {
            meshes.get(object_id).map(|source_mesh| {
                let uv_store = uv_layers::read(source_mesh);
                let uv_names = uv_store
                    .layers
                    .iter()
                    .map(|layer| layer.name.clone())
                    .collect();
                let uv_coverage = uv_face_coverage(&uv_store.layers, source_mesh.faces.len());
                let color_names = color_layer_names(source_mesh);
                let source_data = source_node
                    .data
                    .as_ref()
                    .and_then(|data_id| doc.data_blocks.get(data_id));
                let materials = nodes.get(object_id).map_or_else(
                    || {
                        effective_material_slots(
                            source_node,
                            source_data.map_or(&[][..], |data| data.materials.as_slice()),
                        )
                    },
                    |evaluated| evaluated.materials.clone(),
                );
                let slots = member_slot_bindings(&materials, source_mesh, &uv_store, &color_names);
                (
                    Cow::Borrowed(source_mesh),
                    vec![vec![object_id.clone()]],
                    slots,
                    adaptive_normals
                        .get(object_id)
                        .map(|normals| Cow::Borrowed(normals.as_slice())),
                    uv_names,
                    uv_coverage,
                    color_names,
                )
            })
        };
        let Some((
            nested_mesh,
            nested_paths,
            slots,
            member_normals,
            uv_names,
            uv_coverage,
            color_names,
        )) = member
        else {
            continue;
        };
        if !nested_mesh.vertices.is_empty() {
            let source_matrix = matrices.get(object_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "collection member world matrix is missing",
                )
            })?;
            members.push(CollectionMember {
                mesh: nested_mesh,
                matrix: source_matrix,
                slots,
                adaptive_normals: member_normals,
                uv_names,
                uv_coverage,
                color_names,
            });
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
            adaptive_normals,
            nodes,
            active_instances,
            visited_collections,
            collection_stack,
            members,
            paths,
        )?;
    }
    collection_stack.remove(collection_id);
    Ok(())
}

/// Injects the dynamic-paint colors of one canvas (one per vertex, in storage order) into the evaluation
/// copy as the point color layer `color`, and renders that layer. A `color` layer already on the mesh is
/// replaced. Other color layers are kept.
fn inject_dynamic_paint_colors(mesh: &mut Mesh, colors: &[[f64; 4]]) {
    let mut store = crate::geom::color_layers::read(mesh);
    store.layers.retain(|layer| layer.name != "color");
    store.layers.push(crate::geom::color_layers::ColorLayer {
        name: "color".to_owned(),
        domain: crate::geom::color_layers::ColorDomain::Point,
        values: colors.to_vec(),
    });
    store.render = Some("color".to_owned());
    crate::geom::color_layers::write(mesh, &store);
}

/// Instances copy each source's attributes, remapping the face- and edge-keyed shading ones.
/// Particle copies: each template's attributes, with its face- and edge-keyed shading remapped.
fn append_transformed_mesh(output: &mut Mesh, source: &Mesh, matrix: DMat4) -> Result<()> {
    if output.vertices.is_empty() {
        output.attributes = source
            .attributes
            .iter()
            .filter(|(name, _)| !SHADING_ATTRIBUTES.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
    }
    let mut vertex_ids = BTreeMap::new();
    for vertex in &source.vertices {
        let id = output
            .insert_vertex(matrix.transform_point3(vertex.co))
            .map_err(|error| PotError::new(ErrorCode::EvaluationFailed, error.to_string()))?;
        vertex_ids.insert(vertex.id, id);
    }
    let mut edge_ids = BTreeMap::new();
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
        let id = output
            .insert_edge([first, second])
            .map_err(|error| PotError::new(ErrorCode::EvaluationFailed, error.to_string()))?;
        edge_ids.insert(edge.id, id);
    }
    let mut face_ids = BTreeMap::new();
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
        let id = output
            .insert_face(vertices, face.material_index)
            .map_err(|error| PotError::new(ErrorCode::EvaluationFailed, error.to_string()))?;
        face_ids.insert(face.id, id);
    }
    merge_shading(output, source, &face_ids, &edge_ids)
}

/// Copies one source's shading attributes into the combined instance mesh under remapped IDs.
/// Custom-normal codes are relative to each fan, so they copy unchanged.
fn merge_shading(
    output: &mut Mesh,
    source: &Mesh,
    face_ids: &BTreeMap<u32, u32>,
    edge_ids: &BTreeMap<u32, u32>,
) -> Result<()> {
    for name in SHADING_ATTRIBUTES {
        let Some(source_attribute) = source.attributes.get(name) else {
            continue;
        };
        let (Some(domain), Some(kind), Some(source_values)) = (
            source_attribute.get("domain"),
            source_attribute.get("type"),
            source_attribute.get("values").and_then(Value::as_object),
        ) else {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                format!("instance source attribute `{name}` must have domain, type, and values"),
                json!({ "attribute": name }),
            ));
        };
        let (prefix, ids) = if name == "sharp" {
            ('e', edge_ids)
        } else {
            ('f', face_ids)
        };
        let target = output
            .attributes
            .entry(name.to_owned())
            .or_insert_with(|| json!({ "domain": domain, "type": kind, "values": {} }));
        let values = target
            .get_mut("values")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    format!("instance attribute `{name}` values must be an object"),
                )
            })?;
        for (key, value) in source_values {
            let Some(id) = key
                .strip_prefix(prefix)
                .and_then(|id| id.parse::<u32>().ok())
                .and_then(|id| ids.get(&id))
            else {
                continue;
            };
            values.insert(format!("{prefix}{id}"), value.clone());
        }
    }
    Ok(())
}

fn evaluate_grease_pencil(
    data: &crate::model::GreasePencilData,
    materials: &crate::model::Registry<crate::model::Material>,
    matrix: DMat4,
    frame: f64,
) -> Result<(Vec<EvaluatedStroke>, Option<Aabb>)> {
    let scale = matrix
        .x_axis
        .truncate()
        .length()
        .max(matrix.y_axis.truncate().length())
        .max(matrix.z_axis.truncate().length());
    let mut output = Vec::new();
    let mut bounds: Option<Aabb> = None;
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
            let mut points_world = Vec::with_capacity(stroke.points.len());
            let mut radius = Vec::with_capacity(stroke.points.len());
            for point in &stroke.points {
                let position = matrix.transform_point3(DVec3::from_array(point.position));
                let effective_radius = point.radius * scale;
                let radius_extent = DVec3::splat(effective_radius);
                let lower = position - radius_extent;
                let upper = position + radius_extent;
                let point_min = lower.min(upper);
                let point_max = lower.max(upper);
                if let Some(bounds) = &mut bounds {
                    bounds.min = bounds.min.min(point_min);
                    bounds.max = bounds.max.max(point_max);
                } else {
                    bounds = Some(Aabb {
                        min: point_min,
                        max: point_max,
                    });
                }
                points_world.push(position.to_array());
                radius.push(effective_radius);
            }
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
    Ok((output, bounds))
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

/// Inverse of [`attach_vertex_group_weights`]: vertex ID -> group ID -> weight, read from the
/// `vertex_groups` attribute of `mesh`. Group names resolve through `groups`; names absent from
/// `groups` and keys that are not `v{id}` are ignored.
pub(crate) fn mesh_vertex_weights(
    mesh: &Mesh,
    groups: &[VertexGroup],
) -> BTreeMap<u32, BTreeMap<Id, f64>> {
    let mut weights = BTreeMap::<u32, BTreeMap<Id, f64>>::new();
    let Some(by_group) = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
    else {
        return weights;
    };
    for group in groups {
        let Some(values) = by_group.get(&group.name).and_then(Value::as_object) else {
            continue;
        };
        for (key, value) in values {
            let vertex_id = key.strip_prefix('v').and_then(|id| id.parse::<u32>().ok());
            let (Some(vertex_id), Some(weight)) = (vertex_id, value.as_f64()) else {
                continue;
            };
            weights
                .entry(vertex_id)
                .or_default()
                .insert(group.id.clone(), weight);
        }
    }
    weights
}

// --- F reference helpers (ported unchanged from PR18-F; scene reference layer for displacement/warp/wave) ---
pub(crate) struct SceneReferences<'a> {
    /// World matrix of every evaluated node.
    pub(crate) world_matrices: &'a BTreeMap<Id, DMat4>,
    /// Pose-bone matrices in world space (`armature world * pose`), keyed by armature, then bone.
    pub(crate) bone_matrices: &'a BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
    /// Armature-local pose matrices used to build raw f32 bone reference worlds, keyed by armature
    /// then bone.
    pub(crate) bone_pose_matrices: &'a BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
    /// Meshes evaluated before the referencing modifier.
    pub(crate) evaluated_meshes: &'a BTreeMap<Id, Mesh>,
    /// Directory that image textures resolve against.
    pub(crate) project_root: Option<&'a Path>,
}

fn invalid_reference(modifier: &Modifier, parameter: &str, expected: &str) -> PotError {
    PotError::with_details(
        ErrorCode::InvalidArgument,
        format!(
            "modifier `{}` `{parameter}` must be {expected}",
            modifier.modifier_type
        ),
        json!({"modifier_id":modifier.id,"pointer":format!("/params/{parameter}")}),
    )
}

/// An object-reference parameter; an absent, null, or empty value is no reference.
fn reference_object_id(modifier: &Modifier, parameter: &str, doc: &SceneDoc) -> Result<Option<Id>> {
    let text = match modifier.params.get(parameter) {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(text)) if text.is_empty() => return Ok(None),
        Some(Value::String(text)) => text.as_str(),
        Some(_) => {
            return Err(invalid_reference(
                modifier,
                parameter,
                "an object ID or null",
            ));
        }
    };
    let object_id = Id::new(text).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            error.message,
            json!({"modifier_id":modifier.id,"pointer":format!("/params/{parameter}")}),
        )
    })?;
    if !doc.nodes.contains_key(&object_id) {
        return Err(PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("modifier reference object `{object_id}` was not found"),
            json!({"modifier_id":modifier.id,"object":object_id}),
        ));
    }
    Ok(Some(object_id))
}

fn modifier_str_param<'a>(modifier: &'a Modifier, name: &str) -> Result<&'a str> {
    modifier
        .params
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_reference(modifier, name, "a string"))
}

fn modifier_number_param(modifier: &Modifier, name: &str) -> Result<f64> {
    modifier
        .params
        .get(name)
        .or_else(|| params::default_value(ParameterFamily::Modifier, &modifier.modifier_type, name))
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid_reference(modifier, name, "a finite number"))
}

fn modifier_bool_param(modifier: &Modifier, name: &str) -> Result<bool> {
    modifier
        .params
        .get(name)
        .or_else(|| params::default_value(ParameterFamily::Modifier, &modifier.modifier_type, name))
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid_reference(modifier, name, "a boolean"))
}

fn subject_world_matrix(
    modifier: &Modifier,
    subject_id: &Id,
    references: &SceneReferences<'_>,
) -> Result<DMat4> {
    references
        .world_matrices
        .get(subject_id)
        .copied()
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::DependencyMissing,
                "modifier subject transform is unavailable",
                json!({"modifier_id":modifier.id,"node_id":subject_id}),
            )
        })
}

/// Inverse subject transform, which maps scene space into the subject's local frame.
fn subject_inverse_matrix(
    modifier: &Modifier,
    subject_id: &Id,
    references: &SceneReferences<'_>,
) -> Result<DMat4> {
    let inverse = subject_world_matrix(modifier, subject_id, references)?.inverse();
    if !inverse.is_finite() {
        return Err(PotError::invalid_argument(format!(
            "modifier `{}` subject transform must be invertible",
            modifier.modifier_type
        )));
    }
    Ok(inverse)
}

/// World matrix of a referenced object, or of its pose bone (`armature world * pose`) when named.
fn reference_world_matrix(
    modifier: &Modifier,
    object_id: &Id,
    bone: Option<&Id>,
    references: &SceneReferences<'_>,
) -> Result<DMat4> {
    let Some(bone_id) = bone else {
        return references
            .world_matrices
            .get(object_id)
            .copied()
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::DependencyMissing,
                    "referenced object transform is unavailable",
                    json!({"modifier_id":modifier.id,"object":object_id}),
                )
            });
    };
    references
        .bone_matrices
        .get(object_id)
        .and_then(|bones| bones.get(bone_id))
        .map(DMat4::from_cols_array)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::DependencyMissing,
                "referenced bone pose transform is unavailable",
                json!({"modifier_id":modifier.id,"object":object_id,"bone":bone_id}),
            )
        })
}

fn add_modifier_pose_armature_references(
    doc: &SceneDoc,
    modifier: &Modifier,
    result: &mut BTreeSet<Id>,
) {
    let pose_reference_parameters: &[(&str, &str)] = match modifier.modifier_type.as_str() {
        "hook" => &[("object", "subtarget")],
        "warp" => &[
            ("object_from", "bone_from"),
            ("object_to", "bone_to"),
            ("texture_coords_object", "texture_coords_bone"),
        ],
        "displace" | "wave" => &[("texture_coords_object", "texture_coords_bone")],
        "uv_warp" => &[("object_from", "bone_from"), ("object_to", "bone_to")],
        "vertex_weight_edit" | "vertex_weight_mix" | "vertex_weight_proximity" => {
            &[("mask_tex_map_object", "mask_tex_map_bone")]
        }
        _ => &[],
    };
    for (object_parameter, bone_parameter) in pose_reference_parameters {
        let Some(object_text) = modifier
            .params
            .get(*object_parameter)
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        else {
            continue;
        };
        let Ok(object_id) = Id::new(object_text.to_owned()) else {
            continue;
        };
        let Some(object) = doc.nodes.get(&object_id) else {
            continue;
        };
        if object.kind != "armature" {
            continue;
        }
        let Some(bone_name) = modifier
            .params
            .get(*bone_parameter)
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        else {
            continue;
        };
        let Some(armature) = armature_data_for_modifier_operand(object, doc) else {
            continue;
        };
        if armature
            .bones
            .iter()
            .any(|(bone_id, bone)| bone_id.as_str() == bone_name || bone.name == bone_name)
        {
            result.insert(object_id);
        }
    }
}

fn modifier_pose_armatures(
    doc: &SceneDoc,
    evaluated_node_ids: &BTreeSet<Id>,
    gate: ModifierGate<'_>,
    target_modifier: Option<(&Id, &Id)>,
) -> BTreeSet<Id> {
    let mut result = BTreeSet::new();
    for owner_id in evaluated_node_ids {
        let Some(owner) = doc.nodes.get(owner_id) else {
            continue;
        };
        for (index, modifier) in owner.modifiers.iter().enumerate() {
            if gate.is_active_at(owner_id, index, modifier) {
                add_modifier_pose_armature_references(doc, modifier, &mut result);
            }
        }
    }
    if let Some((target_id, modifier_id)) = target_modifier
        && let Some(owner) = doc.nodes.get(target_id)
    {
        for modifier in &owner.modifiers {
            if &modifier.id == modifier_id {
                if gate.is_active(target_id, modifier) {
                    add_modifier_pose_armature_references(doc, modifier, &mut result);
                }
                break;
            }
        }
    }
    result
}

fn reference_object_or_bone_world_f32(
    object_world: DMat4,
    bone_pose_matrix: Option<&[f64; 16]>,
) -> [f32; 16] {
    let object_world_f32 = object_world.to_cols_array().map(|value| value as f32);
    let Some(bone_pose_matrix) = bone_pose_matrix else {
        return object_world_f32;
    };
    crate::geom::modifiers::blender_multiply_m4(
        object_world_f32,
        bone_pose_matrix.map(|value| value as f32),
    )
}

fn apply_scene_single_reference_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    references: &SceneReferences<'_>,
    parameter: &str,
    bone_parameter: Option<&str>,
) -> Result<()> {
    let mut resolved = modifier.clone();
    resolved.params.remove(parameter);
    if let Some(object_id) = reference_object_id(modifier, parameter, doc)? {
        let bone_name = match bone_parameter {
            Some(name) => reference_bone_name(modifier, name)?,
            None => None,
        };
        let reference =
            warp_reference_world_matrix(modifier, &object_id, bone_name, doc, references)?;
        let inverse = subject_inverse_matrix(modifier, subject_id, references)?;
        resolved.runtime.object_matrix = Some((inverse * reference).to_cols_array());
    }
    *mesh = crate::geom::modifiers::evaluate_modifiers(
        mesh,
        &[resolved],
        scene_subject_name(doc, subject_id)?,
    )?;
    Ok(())
}

/// Warp's source and destination matrices, `inv(W_subject) * W_ref` with the bone pose folded in
/// when `bone_from`/`bone_to` names a bone. `None` when either object is unset; Blender leaves the
/// mesh unchanged in that case. Both object references are validated before that check.
fn warp_reference_matrices(
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    references: &SceneReferences<'_>,
) -> Result<Option<(DMat4, DMat4)>> {
    let from_id = reference_object_id(modifier, "object_from", doc)?;
    let to_id = reference_object_id(modifier, "object_to", doc)?;
    let (Some(from_id), Some(to_id)) = (from_id, to_id) else {
        return Ok(None);
    };
    let inverse = subject_inverse_matrix(modifier, subject_id, references)?;
    let from_bone = reference_bone_name(modifier, "bone_from")?;
    let to_bone = reference_bone_name(modifier, "bone_to")?;
    let from_world = warp_reference_world_matrix(modifier, &from_id, from_bone, doc, references)?;
    let to_world = warp_reference_world_matrix(modifier, &to_id, to_bone, doc, references)?;
    Ok(Some((inverse * from_world, inverse * to_world)))
}

/// A bone name (Warp's `bone_from`/`bone_to`, Hook's `subtarget`) is matched as
/// `armature_bone_id_by_name` does. A name that resolves to no bone uses the object's world
/// matrix, as Blender's `matrix_from_obj_pchan` (Warp) and the Hook fallback do.
fn warp_reference_world_matrix(
    modifier: &Modifier,
    object_id: &Id,
    bone_name: Option<&str>,
    doc: &SceneDoc,
    references: &SceneReferences<'_>,
) -> Result<DMat4> {
    let bone_id = bone_name.and_then(|name| armature_bone_id_by_name(doc, object_id, name));
    reference_world_matrix(modifier, object_id, bone_id.as_ref(), references)
}

/// A bone-name parameter as written; an absent, null, or empty value names no bone.
fn reference_bone_name<'a>(modifier: &'a Modifier, parameter: &str) -> Result<Option<&'a str>> {
    match modifier.params.get(parameter) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(name)) if name.is_empty() => Ok(None),
        Some(Value::String(name)) => Ok(Some(name.as_str())),
        Some(_) => Err(invalid_reference(modifier, parameter, "a bone name")),
    }
}

/// The armature bone of `object_id` whose ID or name is `name`: the match the pose-reference path
/// and the f32 bone fallback both use.
fn armature_bone_id_by_name(doc: &SceneDoc, object_id: &Id, name: &str) -> Option<Id> {
    let armature = doc
        .nodes
        .get(object_id)
        .and_then(|node| armature_data_for_modifier_operand(node, doc))?;
    armature
        .bones
        .iter()
        .find(|(bone_id, bone)| bone_id.as_str() == name || bone.name == name)
        .map(|(bone_id, _)| bone_id.clone())
}

/// Displace, Wave, and Warp resolve the references Blender reads from the scene (Warp's from/to
/// objects, Wave's start object, and texture mapping), then evaluate with the loaded texture.
pub(crate) fn apply_scene_displacement_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    references: &SceneReferences<'_>,
    frame: f64,
    texture_rng: &mut crate::blender_rng::BlenderRng,
) -> Result<()> {
    let mut resolved = modifier.clone();
    if modifier.modifier_type == "warp" {
        // Blender returns the mesh unchanged until both warp objects are set.
        let Some((from, to)) = warp_reference_matrices(modifier, subject_id, doc, references)?
        else {
            return Ok(());
        };
        resolved.params.remove("object_from");
        resolved.params.remove("object_to");
        resolved.runtime.warp_from_matrix = Some(from.to_cols_array());
        resolved.runtime.warp_to_matrix = Some(to.to_cols_array());
    }
    if modifier.modifier_type == "wave"
        && let Some(object_id) = reference_object_id(modifier, "start_position_object", doc)?
    {
        let inverse = subject_inverse_matrix(modifier, subject_id, references)?;
        let object_matrix = reference_world_matrix(modifier, &object_id, None, references)?;
        resolved.params.remove("start_position_object");
        resolved.runtime.object_matrix = Some((inverse * object_matrix).to_cols_array());
    }
    let subject_world = subject_world_matrix(modifier, subject_id, references)?;
    let texture = crate::geom::modifiers::load_modifier_texture(
        modifier,
        "texture",
        doc,
        references.project_root,
    )?;
    if texture.is_some() {
        resolved.runtime.texture_mapping =
            texture_coordinate_mapping(modifier, subject_world, doc, references)?;
    }
    let texture = texture.as_ref();
    let evaluated = match modifier.modifier_type.as_str() {
        "displace" => crate::geom::modifiers::displace_textured(
            mesh,
            &resolved,
            subject_world,
            texture,
            texture_rng,
        )?,
        "wave" => {
            crate::geom::modifiers::wave_textured(mesh, &resolved, frame, texture, texture_rng)?
        }
        "warp" => crate::geom::modifiers::warp_textured(mesh, &resolved, texture, texture_rng)?,
        _ => {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "unexpected displacement modifier callback",
            ));
        }
    };
    *mesh = evaluated;
    Ok(())
}

/// Resolves Blender texture mapping as ordered subject-to-world and optional world-to-object stages.
fn texture_coordinate_mapping(
    modifier: &Modifier,
    subject_world: DMat4,
    doc: &SceneDoc,
    references: &SceneReferences<'_>,
) -> Result<Option<TextureCoordinateMapping>> {
    texture_coordinate_mapping_with_parameters(
        modifier,
        subject_world,
        doc,
        references.world_matrices,
        references.bone_pose_matrices,
        "texture_coords",
        "texture_coords_object",
        Some("texture_coords_bone"),
    )
}

/// The OBJECT texture mapping for a mapping object whose f32 world matrix is `mapping_world`.
/// `world_to_mapping` is Blender's f32 `invert_m4_m4` of that matrix (`blender_invert_m4`), and the
/// subject's f32 transform is the first stage; a singular mapping transform is an error.
pub fn object_texture_mapping(
    subject_world: DMat4,
    mapping_world: [f32; 16],
) -> Result<TextureCoordinateMapping> {
    let world_to_mapping_f32 = crate::geom::blender_matrix::blender_invert_m4(
        &glam::Mat4::from_cols_array(&mapping_world).to_cols_array_2d(),
    )
    .map(|inverse| glam::Mat4::from_cols_array_2d(&inverse).to_cols_array())
    .filter(|inverse| inverse.iter().all(|value| value.is_finite()))
    .ok_or_else(|| {
        PotError::invalid_argument("texture coordinate mapping transform must be invertible")
    })?;
    Ok(TextureCoordinateMapping {
        subject_to_world_f32: subject_world.to_cols_array().map(|value| value as f32),
        world_to_mapping_f32: Some(world_to_mapping_f32),
    })
}

pub(crate) fn texture_coordinate_mapping_with_parameters(
    modifier: &Modifier,
    subject_world: DMat4,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    bone_pose_matrices: &BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
    mapping_parameter: &str,
    object_parameter: &str,
    bone_parameter: Option<&str>,
) -> Result<Option<TextureCoordinateMapping>> {
    let mapping = TextureCoordinateMapping::for_subject_world(subject_world);
    match modifier_str_param(modifier, mapping_parameter)? {
        "LOCAL" | "GLOBAL" => Ok(Some(mapping)),
        "OBJECT" => {
            let Some(object_id) = reference_object_id(modifier, object_parameter, doc)? else {
                return Ok(Some(mapping));
            };
            let mapping_world = reference_world_matrix_f32_with_bone_fallback(
                modifier,
                &object_id,
                bone_parameter,
                doc,
                world_matrices,
                bone_pose_matrices,
            )?;
            object_texture_mapping(subject_world, mapping_world).map(Some)
        }
        "UV" => Ok(None),
        _ => Err(invalid_reference(
            modifier,
            mapping_parameter,
            "LOCAL, GLOBAL, OBJECT, or UV",
        )),
    }
}

/// A missing bone name uses the object world matrix; a found bone multiplies its raw local pose by
/// the evaluated armature world matrix in Blender's f32 order.
fn reference_world_matrix_f32_with_bone_fallback(
    modifier: &Modifier,
    object_id: &Id,
    bone_parameter: Option<&str>,
    doc: &SceneDoc,
    world_matrices: &BTreeMap<Id, DMat4>,
    bone_pose_matrices: &BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
) -> Result<[f32; 16]> {
    let object_world = world_matrices.get(object_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            "referenced object transform is unavailable",
            json!({"modifier_id":modifier.id,"object":object_id}),
        )
    })?;
    let bone_name = bone_parameter
        .and_then(|parameter| modifier.params.get(parameter))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty());
    let bone_id = bone_name.and_then(|name| armature_bone_id_by_name(doc, object_id, name));
    let pose_matrix = if let Some(bone_id) = bone_id {
        Some(
            bone_pose_matrices
                .get(object_id)
                .and_then(|bones| bones.get(&bone_id))
                .ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::DependencyMissing,
                        "referenced armature bone pose transform is unavailable",
                        json!({"modifier_id":modifier.id,"object":object_id,"bone":bone_name}),
                    )
                })?,
        )
    } else {
        None
    };
    Ok(reference_object_or_bone_world_f32(
        object_world,
        pose_matrix,
    ))
}

fn apply_scene_hook_modifier(
    mesh: &mut Mesh,
    modifier: &Modifier,
    subject_id: &Id,
    doc: &SceneDoc,
    references: &SceneReferences<'_>,
) -> Result<()> {
    // Blender disables a Hook without an object, leaving the mesh unchanged.
    if reference_object_id(modifier, "object", doc)?.is_none() {
        return Ok(());
    }
    apply_scene_single_reference_modifier(
        mesh,
        modifier,
        subject_id,
        doc,
        references,
        "object",
        Some("subtarget"),
    )
}

impl TextureCoordinateMapping {
    pub(crate) fn for_subject_world(subject_world: DMat4) -> Self {
        Self {
            subject_to_world_f32: subject_world.to_cols_array().map(|value| value as f32),
            world_to_mapping_f32: None,
        }
    }

    pub(crate) fn global_coordinate(&self, point: DVec3) -> DVec3 {
        let world = crate::geom::modifiers::blender_transform_point_f32(
            &self.subject_to_world_f32,
            point.as_vec3(),
        );
        DVec3::new(f64::from(world.x), f64::from(world.y), f64::from(world.z))
    }

    pub(crate) fn object_coordinate(&self, point: DVec3) -> DVec3 {
        let Some(world_to_mapping_f32) = &self.world_to_mapping_f32 else {
            return point;
        };
        let world = crate::geom::modifiers::blender_transform_point_f32(
            &self.subject_to_world_f32,
            point.as_vec3(),
        );
        let mapped =
            crate::geom::modifiers::blender_transform_point_f32(world_to_mapping_f32, world);
        DVec3::new(
            f64::from(mapped.x),
            f64::from(mapped.y),
            f64::from(mapped.z),
        )
    }
}

fn apply_volume_displace_modifier(
    volume: &crate::geom::volume::VolumeData,
    modifier: &Modifier,
    doc: &SceneDoc,
    gate: ModifierGate<'_>,
    rng: &mut crate::blender_rng::BlenderRng,
    world_matrices: &BTreeMap<Id, DMat4>,
    bone_pose_matrices: &BTreeMap<Id, BTreeMap<Id, [f64; 16]>>,
    volume_id: &Id,
) -> Result<crate::geom::volume::VolumeData> {
    let params = &modifier.params;
    let texture = match params.get("texture").and_then(serde_json::Value::as_str) {
        Some(texture_text) => {
            let texture_id = Id::new(texture_text.to_owned()).map_err(|error| {
                PotError::with_details(
                    ErrorCode::InvalidArgument,
                    error.message,
                    json!({"modifier_id":modifier.id,"parameter":"texture"}),
                )
            })?;
            Some(crate::geom::resolve_texture(
                doc,
                &texture_id,
                gate.context.project_root,
            )?)
        }
        None => None,
    };
    let texture_source = texture.as_ref().map(|resolved| resolved.source());
    let volume_world = world_matrices.get(volume_id).copied().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "Volume Displace target transform is missing",
            json!({"node_id":volume_id}),
        )
    })?;
    let texture_mapping = if texture_source.is_some()
        && params
            .get("strength")
            .and_then(serde_json::Value::as_f64)
            .is_none_or(|strength| strength != 0.0)
    {
        texture_coordinate_mapping_with_parameters(
            modifier,
            volume_world,
            doc,
            world_matrices,
            bone_pose_matrices,
            "texture_map_mode",
            "texture_map_object",
            None,
        )?
    } else {
        None
    };
    crate::geom::modifiers::simulation::displace_volume(
        volume,
        params,
        texture_source,
        rng,
        texture_mapping,
    )
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use std::collections::{BTreeMap, BTreeSet};

    use glam::{DMat4, DVec3};
    use serde_json::json;

    use crate::{
        geom::{
            BoxParams, Mesh,
            uv_layers::{self, UvLayer, UvStore},
        },
        model::{
            Collection, DataBlock, GreasePencilData, GreasePencilFrame, GreasePencilLayer,
            GreasePencilPoint, GreasePencilStroke, Id, MaterialLink, Node, SceneDoc, Transform,
        },
    };

    use super::{
        EvaluationContext, SlotBinding, Snapshot, UvBinding, evaluate_collection_instance,
    };

    #[test]
    fn particle_render_limit_counts_all_systems_for_each_emitter()
    -> std::result::Result<(), &'static str> {
        let emitter_id = Id::new("emitter").unwrap();
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.nodes.insert(
            emitter_id.clone(),
            Node {
                name: "Emitter".to_owned(),
                kind: "mesh".to_owned(),
                ..Node::default()
            },
        );
        let mut states = Vec::with_capacity(120_002);
        for system_index in 0..2 {
            for _ in 0..60_001 {
                states.push(crate::sim::ParticleState {
                    system_index,
                    ..crate::sim::ParticleState::default()
                });
            }
        }
        let particles = std::collections::BTreeMap::from([(emitter_id.clone(), states)]);
        let suppressed = std::collections::BTreeSet::new();
        let gate = super::ModifierGate {
            context: super::ModifierEvaluationContext {
                purpose: super::EvaluationPurpose::Preview,
                project_root: None,
                cache_directory: None,
                material_slot_count: None,
            },
            suppressed: &suppressed,
            modifier_prefix: None,
        };
        let evaluated_nodes = std::collections::BTreeSet::from([emitter_id.clone()]);
        let Err(error) = super::render_simulated_particles(
            &doc,
            gate,
            &particles,
            &std::collections::BTreeMap::new(),
            &mut std::collections::BTreeMap::new(),
            &mut std::collections::BTreeMap::new(),
            &evaluated_nodes,
            &mut std::collections::BTreeMap::new(),
        ) else {
            return Err("combined visible systems must exceed the per-emitter limit");
        };
        assert_eq!(error.code, crate::error::ErrorCode::LimitExceeded);
        assert_eq!(error.details["count"], serde_json::json!(120_002));
        Ok(())
    }
    fn grease_pencil_scene(data: GreasePencilData, transform: Transform) -> (SceneDoc, Id) {
        let data_id = Id::new("grease_pencil_data").unwrap();
        let node_id = Id::new("grease_pencil_node").unwrap();
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "grease_pencil".to_owned(),
                grease_pencil: Some(data),
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Grease Pencil".to_owned(),
                kind: "grease_pencil".to_owned(),
                transform,
                data: Some(data_id),
                ..Node::default()
            },
        );
        (doc, node_id)
    }

    fn grease_pencil_layer(
        id: &str,
        visible: bool,
        opacity: f64,
        frames: Vec<GreasePencilFrame>,
    ) -> GreasePencilLayer {
        GreasePencilLayer {
            id: Id::new(id).unwrap(),
            name: id.to_owned(),
            opacity,
            visible,
            frames,
        }
    }

    fn grease_pencil_frame(frame: f64, strokes: Vec<GreasePencilStroke>) -> GreasePencilFrame {
        GreasePencilFrame { frame, strokes }
    }

    fn grease_pencil_stroke(id: &str, points: &[([f64; 3], f64)]) -> GreasePencilStroke {
        GreasePencilStroke {
            id: Id::new(id).unwrap(),
            points: points
                .iter()
                .map(|(position, radius)| GreasePencilPoint {
                    position: *position,
                    pressure: 1.0,
                    radius: *radius,
                    opacity: 1.0,
                    time: 0.0,
                })
                .collect(),
            material: None,
            cyclic: false,
            fill: None,
        }
    }

    fn assert_vec3_near(actual: DVec3, expected: DVec3) {
        for (actual, expected) in actual.to_array().into_iter().zip(expected.to_array()) {
            assert!(
                (actual - expected).abs() <= 1e-9,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn evaluates_grease_pencil_bounds_from_world_stroke_radii() {
        let actual_strokes = vec![
            grease_pencil_stroke("first", &[([0.0, 0.0, 0.0], 0.5), ([1.0, 2.0, -1.0], 0.25)]),
            grease_pencil_stroke(
                "second",
                &[([3.0, -1.0, 2.0], 0.75), ([2.0, 0.0, 0.0], 0.1)],
            ),
            grease_pencil_stroke(
                "zero_radius",
                &[([5.0, 4.0, 2.0], 0.0), ([0.0, 0.0, 0.0], 0.0)],
            ),
        ];
        let data = GreasePencilData {
            layers: vec![
                grease_pencil_layer(
                    "visible",
                    true,
                    1.0,
                    vec![
                        grease_pencil_frame(
                            0.0,
                            vec![grease_pencil_stroke(
                                "older",
                                &[
                                    ([-100.0, -100.0, -100.0], 1.0),
                                    ([-99.0, -99.0, -99.0], 1.0),
                                ],
                            )],
                        ),
                        grease_pencil_frame(1.0, actual_strokes),
                        grease_pencil_frame(
                            2.0,
                            vec![grease_pencil_stroke(
                                "future",
                                &[([100.0, 100.0, 100.0], 1.0), ([101.0, 101.0, 101.0], 1.0)],
                            )],
                        ),
                    ],
                ),
                grease_pencil_layer(
                    "invisible",
                    false,
                    1.0,
                    vec![grease_pencil_frame(
                        1.0,
                        vec![grease_pencil_stroke(
                            "hidden",
                            &[([200.0, 200.0, 200.0], 1.0), ([201.0, 201.0, 201.0], 1.0)],
                        )],
                    )],
                ),
                grease_pencil_layer(
                    "transparent",
                    true,
                    0.0,
                    vec![grease_pencil_frame(
                        1.0,
                        vec![grease_pencil_stroke(
                            "transparent_stroke",
                            &[([300.0, 300.0, 300.0], 1.0), ([301.0, 301.0, 301.0], 1.0)],
                        )],
                    )],
                ),
            ],
        };
        let transform = Transform::from_rotation_deg([10.0, -2.0, 5.0], [0.0; 3], [2.0, 3.0, 4.0]);
        let (doc, node_id) = grease_pencil_scene(data, transform);
        let snapshot = Snapshot::evaluate(
            &doc,
            &EvaluationContext {
                frame: Some(1.5),
                ..EvaluationContext::default()
            },
        )
        .unwrap();

        let evaluated = snapshot.nodes.get(&node_id).unwrap();
        let bounds = evaluated.bounds.unwrap();
        assert_vec3_near(bounds.min, DVec3::new(8.0, -8.0, 0.0));
        assert_vec3_near(bounds.max, DVec3::new(20.0, 10.0, 16.0));
        assert_vec3_near(
            DVec3::from_array(evaluated.dimensions.unwrap()),
            DVec3::new(12.0, 18.0, 16.0),
        );
        assert_eq!(snapshot.strokes.get(&node_id).unwrap().len(), 3);
    }

    #[test]
    fn grease_pencil_without_evaluated_strokes_has_no_bounds() {
        let cases = [
            ("empty", GreasePencilData { layers: Vec::new() }, 1.0),
            (
                "invisible",
                GreasePencilData {
                    layers: vec![grease_pencil_layer(
                        "invisible",
                        false,
                        1.0,
                        vec![grease_pencil_frame(
                            1.0,
                            vec![grease_pencil_stroke(
                                "hidden",
                                &[([0.0, 0.0, 0.0], 1.0), ([1.0, 1.0, 1.0], 1.0)],
                            )],
                        )],
                    )],
                },
                1.0,
            ),
            (
                "zero_opacity",
                GreasePencilData {
                    layers: vec![grease_pencil_layer(
                        "transparent",
                        true,
                        0.0,
                        vec![grease_pencil_frame(
                            1.0,
                            vec![grease_pencil_stroke(
                                "transparent_stroke",
                                &[([0.0, 0.0, 0.0], 1.0), ([1.0, 1.0, 1.0], 1.0)],
                            )],
                        )],
                    )],
                },
                1.0,
            ),
            (
                "future_only",
                GreasePencilData {
                    layers: vec![grease_pencil_layer(
                        "future",
                        true,
                        1.0,
                        vec![grease_pencil_frame(
                            2.0,
                            vec![grease_pencil_stroke(
                                "future_stroke",
                                &[([0.0, 0.0, 0.0], 1.0), ([1.0, 1.0, 1.0], 1.0)],
                            )],
                        )],
                    )],
                },
                1.0,
            ),
            (
                "one_point",
                GreasePencilData {
                    layers: vec![grease_pencil_layer(
                        "short",
                        true,
                        1.0,
                        vec![grease_pencil_frame(
                            1.0,
                            vec![grease_pencil_stroke(
                                "short_stroke",
                                &[([0.0, 0.0, 0.0], 1.0)],
                            )],
                        )],
                    )],
                },
                1.0,
            ),
        ];

        for (case, data, frame) in cases {
            let (doc, node_id) = grease_pencil_scene(data, Transform::default());
            let snapshot = Snapshot::evaluate(
                &doc,
                &EvaluationContext {
                    frame: Some(frame),
                    ..EvaluationContext::default()
                },
            )
            .unwrap();
            let evaluated = snapshot.nodes.get(&node_id).unwrap();
            assert!(evaluated.bounds.is_none(), "{case} unexpectedly has bounds");
            assert!(
                evaluated.dimensions.is_none(),
                "{case} unexpectedly has dimensions"
            );
            assert!(
                !snapshot.strokes.contains_key(&node_id),
                "{case} has strokes"
            );
        }
    }

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

    #[test]
    fn dynamic_paint_colors_become_the_render_point_layer() {
        use crate::geom::color_layers::{self, ColorDomain, ColorLayer};
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let corners: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
        let mut store = color_layers::read(&mesh);
        store.layers.push(ColorLayer {
            name: "color".to_owned(),
            domain: ColorDomain::Corner,
            values: vec![[1.0; 4]; corners],
        });
        color_layers::write(&mut mesh, &store);
        let colors: Vec<[f64; 4]> = (0..mesh.vertices.len())
            .map(|index| [index as f64 / 10.0, 0.25, 0.0, 1.0])
            .collect();

        super::inject_dynamic_paint_colors(&mut mesh, &colors);

        let store = color_layers::read(&mesh);
        assert_eq!(store.render.as_deref(), Some("color"));
        let layer = &store.layers[store.index_of("color").unwrap()];
        assert_eq!(layer.domain, ColorDomain::Point);
        assert_eq!(layer.values, colors);
        assert_eq!(
            mesh.attributes["color"]["domain"],
            serde_json::json!("point")
        );
        assert!(color_layers::validate(&mesh).is_ok());
    }

    fn member_mesh(layers: &[(&str, [f64; 2], Option<usize>)], render: &str, color: &str) -> Mesh {
        let mut mesh = Mesh::box_mesh(BoxParams {
            size: DVec3::splat(2.0),
        })
        .unwrap();
        let uv_layers = layers
            .iter()
            .map(|(name, uv, missing_face)| {
                UvLayer::new(
                    *name,
                    mesh.faces
                        .iter()
                        .enumerate()
                        .map(|(index, face)| {
                            (*missing_face != Some(index)).then(|| vec![*uv; face.vertices.len()])
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        uv_layers::write(
            &mut mesh,
            &UvStore {
                layers: uv_layers,
                active: Some(render.to_owned()),
                render: Some(render.to_owned()),
            },
        );
        mesh.attributes.insert(
            "color_layers".to_owned(),
            json!({"layers":[color],"active":color,"render":color}),
        );
        let color_values = mesh
            .faces
            .iter()
            .map(|face| {
                (
                    format!("f{}", face.id),
                    json!(vec![[0.2, 0.4, 0.8, 0.35]; face.vertices.len()]),
                )
            })
            .collect::<BTreeMap<_, _>>();
        mesh.attributes.insert(
            color.to_owned(),
            json!({"domain":"corner","type":"color","values":color_values}),
        );
        mesh
    }

    fn collection_instance(name: &str, collection: &Id) -> Node {
        let mut properties = serde_json::Map::new();
        properties.insert("instance_collection".to_owned(), json!(collection));
        Node {
            name: name.to_owned(),
            kind: "collection_instance".to_owned(),
            properties,
            ..Node::default()
        }
    }

    #[test]
    fn nested_collection_bindings_remap_member_layers_and_keep_real_face_coverage() {
        let outer_id = Id::new("outer").unwrap();
        let inner_id = Id::new("inner").unwrap();
        let outer_instance_id = Id::new("outer_instance").unwrap();
        let inner_instance_id = Id::new("inner_instance").unwrap();
        let prefix_id = Id::new("prefix").unwrap();
        let first_id = Id::new("first").unwrap();
        let second_id = Id::new("second").unwrap();
        let prefix_material = Id::new("prefix_material").unwrap();
        let shared_material = Id::new("shared_material").unwrap();
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.collections.insert(
            outer_id.clone(),
            Collection {
                name: "Outer".to_owned(),
                children: Vec::new(),
                objects: vec![prefix_id.clone(), inner_instance_id.clone()],
            },
        );
        doc.collections.insert(
            inner_id.clone(),
            Collection {
                name: "Inner".to_owned(),
                children: Vec::new(),
                objects: vec![first_id.clone(), second_id.clone()],
            },
        );
        doc.nodes.insert(
            outer_instance_id.clone(),
            collection_instance("Outer instance", &outer_id),
        );
        doc.nodes.insert(
            inner_instance_id.clone(),
            collection_instance("Inner instance", &inner_id),
        );
        doc.nodes.insert(
            prefix_id.clone(),
            Node {
                name: "Prefix".to_owned(),
                kind: "mesh".to_owned(),
                materials: vec![Some(prefix_material)],
                ..Node::default()
            },
        );
        doc.nodes.insert(
            first_id.clone(),
            Node {
                name: "First".to_owned(),
                kind: "mesh".to_owned(),
                materials: vec![Some(shared_material.clone()), Some(shared_material.clone())],
                ..Node::default()
            },
        );
        doc.nodes.insert(
            second_id.clone(),
            Node {
                name: "Second".to_owned(),
                kind: "mesh".to_owned(),
                materials: vec![Some(shared_material.clone())],
                ..Node::default()
            },
        );
        let prefix = member_mesh(&[("Other", [0.1, 0.2], None)], "Other", "Tint");
        let mut first = member_mesh(
            &[("Base", [0.3, 0.4], Some(0)), ("Detail", [0.5, 0.6], None)],
            "Detail",
            "Skin",
        );
        let second = member_mesh(
            &[("Detail", [0.7, 0.8], None), ("Base", [0.9, 0.25], None)],
            "Base",
            "Skin",
        );
        first.faces[1].material_index = 1;
        let mut meshes = BTreeMap::new();
        meshes.insert(prefix_id.clone(), prefix);
        meshes.insert(first_id.clone(), first);
        meshes.insert(second_id.clone(), second);
        let matrices = [
            outer_instance_id.clone(),
            inner_instance_id,
            prefix_id,
            first_id,
            second_id,
        ]
        .into_iter()
        .map(|id| (id, DMat4::IDENTITY))
        .collect();

        let evaluation = evaluate_collection_instance(
            &doc,
            &outer_instance_id,
            &matrices,
            &meshes,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &mut BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(evaluation.mesh.vertices.len(), 24);
        assert_eq!(evaluation.mesh.edges.len(), 36);
        assert_eq!(evaluation.mesh.faces.len(), 18);
        let corner_count = evaluation
            .mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>();
        assert_eq!(corner_count, 72);

        let uv = uv_layers::read(&evaluation.mesh);
        assert_eq!(
            uv.layers
                .iter()
                .map(|layer| layer.name.as_str())
                .collect::<Vec<_>>(),
            ["Other", "Base", "Detail"]
        );
        for layer in &uv.layers {
            assert_eq!(layer.faces.len(), evaluation.mesh.faces.len());
            for (face, row) in evaluation.mesh.faces.iter().zip(&layer.faces) {
                assert_eq!(row.as_ref().unwrap().len(), face.vertices.len());
            }
        }
        assert_eq!(
            uv.layers[0].faces[6].as_ref().unwrap()[0],
            [0.0, 1.0],
            "each absent joined UV row uses canonical Blender UV(0, 0)"
        );
        assert_eq!(
            uv.layers[1].faces[6].as_ref().unwrap()[0],
            [0.0, 1.0],
            "an originally absent row stays uncovered after its default is materialized"
        );
        assert_eq!(uv.layers[1].faces[7].as_ref().unwrap()[0], [0.3, 0.4]);
        assert_eq!(uv.layers[2].faces[6].as_ref().unwrap()[0], [0.5, 0.6]);
        assert_eq!(uv.layers[1].faces[12].as_ref().unwrap()[0], [0.9, 0.25]);
        assert_eq!(uv.layers[2].faces[12].as_ref().unwrap()[0], [0.7, 0.8]);
        assert_eq!(&evaluation.uv_coverage[..6], &[1, 1, 1, 1, 1, 1]);
        assert_eq!(&evaluation.uv_coverage[6..12], &[4, 6, 6, 6, 6, 6]);
        assert_eq!(&evaluation.uv_coverage[12..], &[6, 6, 6, 6, 6, 6]);

        assert_eq!(evaluation.bindings.len(), 4);
        assert_eq!(evaluation.bindings[0].uv.render_index, Some(0));
        assert_eq!(evaluation.bindings[0].uv.available_mask, 1);
        assert!(evaluation.bindings[0].uv.texcoord0_covered);
        assert_eq!(evaluation.bindings[1].uv.render_index, Some(2));
        assert_eq!(evaluation.bindings[1].uv.available_mask, 6);
        assert!(!evaluation.bindings[1].uv.texcoord0_covered);
        assert_eq!(evaluation.bindings[2].uv, evaluation.bindings[1].uv);
        assert_eq!(evaluation.bindings[3].uv.render_index, Some(1));
        assert_eq!(evaluation.bindings[3].uv.available_mask, 6);
        assert!(!evaluation.bindings[3].uv.texcoord0_covered);
        assert_eq!(
            evaluation.bindings[1].material,
            evaluation.bindings[2].material
        );
        assert_eq!(evaluation.bindings[1].color_render_layer, Some(1));
        assert_eq!(&*evaluation.bindings[1].color_available_mask, &[2]);
        assert!(std::sync::Arc::ptr_eq(
            &evaluation.bindings[1].color_available_mask,
            &evaluation.bindings[2].color_available_mask
        ));
        let positions_by_id = evaluation
            .mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co))
            .collect::<BTreeMap<_, _>>();
        let positions = evaluation
            .mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let normals = crate::geom::normals::corner_normals(&evaluation.mesh, &positions).unwrap();
        assert_eq!(normals.len(), corner_count);
        let mut corner_base = 0;
        for face in &evaluation.mesh.faces {
            let first = positions_by_id[&face.vertices[0]];
            let expected = (positions_by_id[&face.vertices[1]] - first)
                .cross(positions_by_id[&face.vertices[2]] - first)
                .normalize();
            let corner_end = corner_base + face.vertices.len();
            for actual in &normals[corner_base..corner_end] {
                assert!(
                    (*actual - expected).length() <= 1.0e-12,
                    "expected {expected}, got {actual}"
                );
            }
            corner_base = corner_end;
        }
    }

    #[test]
    fn slot_material_and_binding_clamp_to_the_last_slot() {
        let mut snapshot = Snapshot::evaluate(
            &SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned()),
            &EvaluationContext::default(),
        )
        .unwrap();
        let first = Id::new("first_material").unwrap();
        let last = Id::new("last_material").unwrap();
        let ordinary_id = Id::new("ordinary").unwrap();
        let ordinary = Node {
            materials: vec![Some(first), Some(last.clone())],
            ..Node::default()
        };
        assert_eq!(
            snapshot.slot_material(&ordinary_id, &ordinary, None, usize::MAX),
            Some(&last)
        );
        assert!(
            snapshot
                .slot_material(&ordinary_id, &Node::default(), None, 0)
                .is_none()
        );

        let instance_id = Id::new("instance").unwrap();
        let instance = Node::default();
        snapshot.slot_bindings.insert(
            instance_id.clone(),
            vec![
                SlotBinding {
                    material: None,
                    uv: UvBinding::default(),
                    color_render_layer: None,
                    color_available_mask: Vec::<u64>::new().into(),
                },
                SlotBinding {
                    material: Some(last.clone()),
                    uv: UvBinding::default(),
                    color_render_layer: None,
                    color_available_mask: Vec::<u64>::new().into(),
                },
            ],
        );
        assert!(
            snapshot
                .slot_binding(&instance_id, &instance, 0)
                .unwrap()
                .material
                .is_none()
        );
        assert!(
            snapshot
                .slot_material(&instance_id, &instance, None, 0)
                .is_none()
        );
        assert_eq!(
            snapshot
                .slot_binding(&instance_id, &instance, usize::MAX)
                .unwrap()
                .material
                .as_ref(),
            Some(&last)
        );
        assert_eq!(
            snapshot.slot_material(&instance_id, &instance, None, usize::MAX),
            Some(&last)
        );
        let empty_id = Id::new("empty_instance").unwrap();
        snapshot.slot_bindings.insert(empty_id.clone(), Vec::new());
        assert!(
            snapshot
                .slot_binding(&empty_id, &instance, usize::MAX)
                .is_none()
        );
        assert!(
            snapshot
                .slot_material(&empty_id, &instance, None, usize::MAX)
                .is_none()
        );
    }
    #[test]
    fn material_slots_resolve_object_data_links_and_null_fallback() {
        let snapshot = Snapshot::evaluate(
            &SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned()),
            &EvaluationContext::default(),
        )
        .unwrap();
        let data_first = Id::new("data_first").unwrap();
        let object_override = Id::new("object_override").unwrap();
        let data_last = Id::new("data_last").unwrap();
        let data = DataBlock {
            materials: vec![Some(data_first.clone()), None, Some(data_last.clone())],
            ..DataBlock::default()
        };
        let node_id = Id::new("mixed_links").unwrap();
        let node = Node {
            materials: vec![None, Some(object_override.clone()), None],
            material_links: vec![
                MaterialLink::Object,
                MaterialLink::Data,
                MaterialLink::Object,
            ],
            ..Node::default()
        };

        assert_eq!(node.material_slot_count(Some(&data)), 3);
        assert_eq!(node.material_slot(Some(&data), 0), Some(&data_first));
        assert_eq!(node.material_slot(Some(&data), 1), None);
        assert_eq!(node.material_slot(Some(&data), 2), Some(&data_last));
        assert_eq!(
            snapshot.slot_material(&node_id, &node, Some(&data), 0),
            Some(&data_first),
            "a null OBJECT slot falls back to the DATA material at the same index"
        );
        assert_eq!(
            snapshot.slot_material(&node_id, &node, Some(&data), 2),
            Some(&data_last)
        );
        assert_eq!(
            snapshot.slot_material(&node_id, &node, Some(&data), 1),
            None,
            "a DATA-linked empty middle slot keeps its numeric index"
        );
        assert_eq!(
            snapshot.slot_material(&node_id, &node, Some(&data), usize::MAX),
            Some(&data_last),
            "out-of-range access clamps to the last evaluated slot"
        );

        let legacy = Node {
            materials: vec![Some(object_override.clone())],
            ..Node::default()
        };
        assert_eq!(legacy.material_link(0), MaterialLink::Object);
        assert_eq!(legacy.material_link(1), MaterialLink::Data);
        assert_eq!(legacy.material_slot(Some(&data), 0), Some(&object_override));
        assert_eq!(legacy.material_slot(Some(&data), 1), None);
        assert_eq!(legacy.material_slot(Some(&data), 2), Some(&data_last));
    }
}
