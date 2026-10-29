use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque, hash_map::DefaultHasher},
    fmt::Write as _,
    hash::{Hash, Hasher},
    sync::{Arc, LazyLock, Mutex},
};

use glam::{DMat4, DQuat, DVec3, Vec3};
use serde_json::{Map, Value, json};

use crate::{
    blender_rng::BlenderRng,
    error::{ErrorCode, PotError, Result},
    geom::{
        GridParams, IcosphereParams, Mesh, MeshError, ORIGINDEX_NONE,
        modifiers::evaluate_modifiers, origindex_values, set_origindex,
    },
    model::{Id, Modifier, Registry},
    shader::texture_procedural_kernels::{
        self as procedural_texture, WaveBandsDirection, WaveProfile, WaveRingsDirection, WaveType,
    },
};

use super::{
    GraphKind, GraphNode, NodeBakeFrame, NodeBakeRecord, NodeBakeValue, NodeGroup, same_bake_frame,
    sockets::{bake_item_identifier, is_known_node_type, node_sockets, node_sockets_with_groups},
};

const MAX_GENERATED_POINTS: usize = 1_000_000;
const MAX_ZONE_ITERATIONS: i64 = 100_000;
const MAX_SIMULATION_CACHE_ENTRIES: usize = 10_000;

type SimulationCache = Mutex<BTreeMap<(String, Id, i64), GeometryValue>>;

static SIMULATION_CACHE: LazyLock<SimulationCache> = LazyLock::new(|| Mutex::new(BTreeMap::new()));

#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub(crate) struct EvaluatedInstances {
    pub(crate) transforms: Vec<[f64; 16]>,
    pub(crate) attributes: BTreeMap<String, InstanceAttributeColumn>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct InstanceAttributeColumn {
    pub(crate) data_type: String,
    pub(crate) values: Vec<Value>,
}

impl EvaluatedInstances {
    fn len(&self) -> usize {
        self.transforms.len()
    }

    fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    fn push_transform(&mut self, transform: [f64; 16]) {
        for column in self.attributes.values_mut() {
            column.values.push(Value::Null);
        }
        self.transforms.push(transform);
    }
    fn push_point_attributes(
        &mut self,
        transform: [f64; 16],
        columns: &BTreeMap<String, PointAttributeColumn<'_>>,
        mesh: &Mesh,
        index: usize,
    ) -> Result<()> {
        let vertex = mesh
            .vertices
            .get(index)
            .ok_or_else(|| evaluation("point attribute index is out of range"))?;
        let prior_len = self.len();
        self.push_transform(transform);
        for (name, column) in columns {
            let value = point_attribute_value(column, vertex.id, index)?;
            let data_type = match &column.data_type {
                Some(data_type) => data_type.clone(),
                None => infer_instance_data_type(&value)?,
            };
            if let Some(existing) = self.attributes.get_mut(name) {
                if existing.data_type != data_type {
                    return Err(evaluation("instance attribute has inconsistent data types"));
                }
                existing.values[prior_len] = value;
            } else {
                let mut values = vec![Value::Null; prior_len];
                values.push(value);
                self.attributes
                    .insert(name.clone(), InstanceAttributeColumn { data_type, values });
            }
        }
        Ok(())
    }

    fn append(&mut self, other: &Self) -> Result<()> {
        let self_len = self.len();
        let other_len = other.len();
        for (name, column) in &other.attributes {
            if let Some(existing) = self.attributes.get_mut(name) {
                if existing.data_type != column.data_type {
                    return Err(evaluation("instance attribute has inconsistent data types"));
                }
                existing.values.extend(column.values.iter().cloned());
            } else {
                let mut values = vec![Value::Null; self_len];
                values.extend(column.values.iter().cloned());
                self.attributes.insert(
                    name.clone(),
                    InstanceAttributeColumn {
                        data_type: column.data_type.clone(),
                        values,
                    },
                );
            }
        }
        for (name, column) in &mut self.attributes {
            if !other.attributes.contains_key(name) {
                column.values.resize(self_len + other_len, Value::Null);
            }
        }
        self.transforms.extend_from_slice(&other.transforms);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphEvaluation {
    pub mesh: Mesh,
    pub instance_paths: Vec<Vec<Id>>,
    pub(crate) instances: EvaluatedInstances,
    pub bake_captures: BTreeMap<String, NodeBakeFrame>,
    pub bake_material_refs: BTreeMap<String, Vec<Id>>,
}

#[derive(Clone, Debug)]
struct GeometryValue {
    mesh: Mesh,
    instance_paths: Vec<Vec<Id>>,
    instance_attributes: EvaluatedInstances,
}

#[derive(Clone, Debug)]
enum EvalValue {
    Geometry(GeometryValue),
    Field(Field),
    Scalar(Value),
}

#[derive(Clone, Debug)]
enum Field {
    Constant(Value),
    Position,
    Normal,
    Index,
    IdOrIndex,
    Attribute {
        name: String,
        data_type: Option<String>,
    },
    AttributeExists(String),
    Captured {
        domain: String,
        values: Vec<Value>,
    },
    Math {
        operation: String,
        clamp: bool,
        first: Box<Field>,
        second: Box<Field>,
        third: Box<Field>,
    },
    VectorMath {
        operation: String,
        first: Box<Field>,
        second: Box<Field>,
        third: Box<Field>,
        scale: Box<Field>,
        output: String,
    },
    IntegerMath {
        operation: String,
        first: Box<Field>,
        second: Box<Field>,
        third: Box<Field>,
    },
    BooleanMath {
        operation: String,
        first: Box<Field>,
        second: Box<Field>,
    },
    Compare {
        operation: String,
        data_type: String,
        mode: String,
        first: Box<Field>,
        second: Box<Field>,
        parameter: Box<Field>,
        epsilon: Box<Field>,
    },
    MapRange {
        data_type: String,
        interpolation: String,
        clamp: bool,
        value: Box<Field>,
        from_min: Box<Field>,
        from_max: Box<Field>,
        to_min: Box<Field>,
        to_max: Box<Field>,
        steps: Box<Field>,
    },
    Clamp {
        mode: String,
        value: Box<Field>,
        min: Box<Field>,
        max: Box<Field>,
    },
    Mix {
        node_type: &'static str,
        output: MixOutput,
        data_type: String,
        factor_mode: String,
        blend_mode: String,
        clamp_factor: bool,
        clamp_result: bool,
        factor: Box<Field>,
        first: Box<Field>,
        second: Box<Field>,
    },
    ColorRamp {
        elements: Arc<[(f32, [f32; 4])]>,
        interpolation: String,
        color_mode: String,
        hue_interpolation: String,
        factor: Box<Field>,
        output: String,
    },
    Curve {
        kind: String,
        mapping: Arc<crate::color::curve::CurveMapping>,
        prepared: Arc<crate::color::curve::PreparedCurve>,
        factor: Box<Field>,
        value: Box<Field>,
    },
    CombineXYZ {
        x: Box<Field>,
        y: Box<Field>,
        z: Box<Field>,
    },
    SeparateXYZ {
        vector: Box<Field>,
        axis: usize,
    },
    Random {
        seed: u32,
        data_type: String,
        id: Box<Field>,
        min: Box<Field>,
        max: Box<Field>,
    },
    Function {
        node_type: String,
        properties: Map<String, Value>,
        inputs: BTreeMap<String, Box<Field>>,
        output: String,
    },
    ProceduralTexture {
        kind: TextureFieldKind,
        inputs: Vec<Field>,
        output: String,
    },
}
#[derive(Clone, Copy, Debug)]
enum TextureFieldKind {
    Wave {
        wave_type: WaveType,
        bands_direction: WaveBandsDirection,
        rings_direction: WaveRingsDirection,
        profile: WaveProfile,
    },
    Magic {
        depth: u8,
    },
    Brick {
        offset: f32,
        offset_frequency: i32,
        squash: f32,
        squash_frequency: i32,
    },
    WhiteNoise {
        dimensions: u8,
    },
    Gabor {
        dimensions: u8,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum MixOutput {
    Value,
    Color,
}

struct Evaluator<'a> {
    group: &'a NodeGroup,
    node_groups: &'a Registry<NodeGroup>,
    group_stack: Vec<Id>,
    group_inputs: BTreeMap<String, EvalValue>,
    top_level: bool,
    subject_name: &'a str,
    input: Option<&'a Mesh>,
    input_instances: Option<&'a EvaluatedInstances>,
    input_instance_paths: Option<&'a [Vec<Id>]>,
    capture_instance_attributes: bool,
    require_instance_attributes: bool,
    overrides: &'a Map<String, Value>,
    frame: f64,
    frame_duration: f64,
    simulation_delta_times: HashMap<Id, f64>,
    simulation_signature: Option<String>,
    simulation_states: HashMap<Id, GeometryValue>,
    repeat_states: HashMap<Id, GeometryValue>,
    repeat_iterations: HashMap<Id, i64>,
    active: BTreeSet<Id>,
    cache: BTreeMap<(Id, String), EvalValue>,
    bake_records: &'a BTreeMap<String, NodeBakeRecord>,
    capture_bake_nodes: &'a BTreeSet<String>,
    bake_captures: BTreeMap<String, NodeBakeFrame>,
    bake_material_refs: BTreeMap<String, Vec<Id>>,
}

/// Evaluate a geometry node group against an optional incoming mesh.
///
/// Inputs are keyed by interface socket ID or name. Generated instances are realized
/// into `mesh` for bounds and rendering while `instance_paths` records each instance.
/// `subject_name` is forwarded to modifier nodes, which use it to seed texture noise.
///
/// # Errors
///
/// Returns `INVALID_OPERATION` for invalid links or unsupported graph inputs,
/// `EVALUATION_FAILED` for cycles or invalid evaluation, and `UNSUPPORTED_FEATURE` for
/// graph kinds or node behaviors not implemented by the native evaluator.
pub fn evaluate(
    group: &NodeGroup,
    input: Option<&Mesh>,
    overrides: &Map<String, Value>,
    subject_name: &str,
) -> Result<GraphEvaluation> {
    evaluate_at_frame(group, input, overrides, 1.0, subject_name)
}
/// Evaluate a registered geometry node group with nested `GeometryNodeGroup` calls.
///
/// Inputs are keyed by interface socket ID or name.
///
/// # Errors
///
/// Returns an error for a missing group, invalid graph, unsupported behavior, or recursive group cycle.
pub fn evaluate_with_node_groups(
    group_id: &Id,
    node_groups: &Registry<NodeGroup>,
    input: Option<&Mesh>,
    overrides: &Map<String, Value>,
    subject_name: &str,
) -> Result<GraphEvaluation> {
    let group = node_groups.get(group_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::TargetNotFound,
            format!("geometry node group `{group_id}` was not found"),
        )
    })?;
    evaluate_at_frame_with_bakes_and_instances(
        group,
        Some(group_id),
        node_groups,
        input,
        None,
        None,
        true,
        true,
        overrides,
        1.0,
        1.0 / 24.0,
        &BTreeMap::new(),
        &BTreeSet::new(),
        subject_name,
    )
}

/// Evaluate a geometry node group at a scene frame.
///
/// Simulation-zone states are cached by graph inputs and integer frame.
///
/// # Errors
///
/// Returns an error for invalid graphs, unsupported graph kinds, or invalid evaluation.
/// `subject_name` is the evaluating object's Blender name, which subject-bound modifiers use.
pub fn evaluate_at_frame(
    group: &NodeGroup,
    input: Option<&Mesh>,
    overrides: &Map<String, Value>,
    frame: f64,
    subject_name: &str,
) -> Result<GraphEvaluation> {
    evaluate_at_frame_with_bakes(
        group,
        input,
        overrides,
        frame,
        &BTreeMap::new(),
        &BTreeSet::new(),
        subject_name,
    )
}

/// Evaluate a geometry node group with modifier-owned Bake-node data and optional capture targets.
/// `subject_name` is forwarded to modifier nodes, which use it to seed texture noise.
pub fn evaluate_at_frame_with_bakes(
    group: &NodeGroup,
    input: Option<&Mesh>,
    overrides: &Map<String, Value>,
    frame: f64,
    bake_records: &BTreeMap<String, NodeBakeRecord>,
    capture_bake_nodes: &BTreeSet<String>,
    subject_name: &str,
) -> Result<GraphEvaluation> {
    let node_groups = Registry::new();
    evaluate_at_frame_with_bakes_and_instances(
        group,
        None,
        &node_groups,
        input,
        None,
        None,
        true,
        true,
        overrides,
        frame,
        1.0 / 24.0,
        bake_records,
        capture_bake_nodes,
        subject_name,
    )
}

pub(crate) fn evaluate_at_frame_with_bakes_and_instances(
    group: &NodeGroup,
    group_id: Option<&Id>,
    node_groups: &Registry<NodeGroup>,
    input: Option<&Mesh>,
    input_instances: Option<&EvaluatedInstances>,
    input_instance_paths: Option<&[Vec<Id>]>,
    collect_instance_attributes: bool,
    outer_instance_attributes_required: bool,
    overrides: &Map<String, Value>,
    frame: f64,
    frame_duration: f64,
    bake_records: &BTreeMap<String, NodeBakeRecord>,
    capture_bake_nodes: &BTreeSet<String>,
    subject_name: &str,
) -> Result<GraphEvaluation> {
    if !frame_duration.is_finite() || frame_duration <= 0.0 {
        return Err(PotError::invalid_argument(
            "graph evaluation frame duration must be finite and positive",
        ));
    }
    if !frame.is_finite() {
        return Err(PotError::invalid_argument(
            "graph evaluation frame must be finite",
        ));
    }
    if group.kind != GraphKind::Geometry {
        return Err(PotError::new(
            ErrorCode::UnsupportedFeature,
            "native evaluator supports geometry node groups only",
        ));
    }
    validate_graph(group, node_groups)?;
    let capture_instance_attributes = collect_instance_attributes
        || !capture_bake_nodes.is_empty()
        || group
            .nodes
            .values()
            .any(|node| node.node_type == "GeometryNodeInputNamedAttribute");
    let require_instance_attributes = outer_instance_attributes_required;
    let simulation_signature = group
        .nodes
        .values()
        .any(|node| node.node_type == "GeometryNodeSimulationOutput")
        .then(|| {
            let serialized = if capture_instance_attributes {
                serde_json::to_value((group, input, input_instances, overrides, frame_duration))
            } else {
                serde_json::to_value((
                    group,
                    input,
                    input_instances.map(|instances| &instances.transforms),
                    overrides,
                    frame_duration,
                ))
            };
            serialized
                .map_err(|error| {
                    PotError::new(
                        ErrorCode::InternalError,
                        format!("could not serialize graph simulation inputs: {error}"),
                    )
                })
                .and_then(|value| {
                    crate::hash::canonicalize(&value).map(|bytes| {
                        let mut signature = crate::hash::sha256(&bytes);
                        if let Some(indices) = input.and_then(origindex_values) {
                            let mut hasher = DefaultHasher::new();
                            indices.hash(&mut hasher);
                            let _ = write!(signature, "-{:016x}", hasher.finish());
                        }
                        signature
                    })
                })
        })
        .transpose()?;
    let output_node = select_group_output_node(group)?;
    let output_socket = if group.interface.outputs.is_empty() {
        "Geometry".to_owned()
    } else {
        let socket = group
            .interface
            .outputs
            .iter()
            .find(|socket| socket_is_geometry(&socket.socket_type))
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "geometry node group has no geometry output socket",
                )
            })?;
        let output_link = group
            .links
            .iter()
            .find(|link| link.to_node == output_node && link.to_socket == socket.id)
            .or_else(|| {
                group_socket_name_is_unambiguous(&group.interface.outputs, &socket.name)
                    .then(|| {
                        group.links.iter().find(|link| {
                            link.to_node == output_node && link.to_socket == socket.name
                        })
                    })
                    .flatten()
            });
        output_link
            .map(|link| link.to_socket.clone())
            .or_else(|| {
                group.nodes.get(&output_node).and_then(|node| {
                    if node.inputs.contains_key(&socket.id) {
                        Some(socket.id.clone())
                    } else if group_socket_name_is_unambiguous(
                        &group.interface.outputs,
                        &socket.name,
                    ) && node.inputs.contains_key(&socket.name)
                    {
                        Some(socket.name.clone())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_else(|| socket.id.clone())
    };
    let mut evaluator = Evaluator {
        group,
        node_groups,
        group_stack: group_id.cloned().into_iter().collect(),
        group_inputs: BTreeMap::new(),
        top_level: true,
        subject_name,
        input,
        input_instances,
        input_instance_paths,
        frame_duration,
        simulation_delta_times: HashMap::new(),
        frame,
        overrides,
        capture_instance_attributes,
        require_instance_attributes,
        simulation_signature,
        simulation_states: HashMap::new(),
        repeat_states: HashMap::new(),
        repeat_iterations: HashMap::new(),
        active: BTreeSet::new(),
        cache: BTreeMap::new(),
        bake_records,
        capture_bake_nodes,
        bake_material_refs: BTreeMap::new(),
        bake_captures: BTreeMap::new(),
    };
    let initial_instances = evaluator.input_instances_for_graph();
    let value = evaluator.input_value(
        &output_node,
        &output_socket,
        EvalValue::Geometry(GeometryValue {
            mesh: input.cloned().unwrap_or_default(),
            instance_paths: evaluator.input_instance_paths_for_graph(),
            instance_attributes: initial_instances,
        }),
    )?;
    let EvalValue::Geometry(geometry) = value else {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "geometry group output is not geometry",
        ));
    };
    for node_id in capture_bake_nodes {
        let parsed_id = Id::new(node_id.clone()).map_err(|_| {
            PotError::invalid_argument(format!("bake node ID `{node_id}` is invalid"))
        })?;
        let node = group.nodes.get(&parsed_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("Bake node `{node_id}` was not found"),
            )
        })?;
        if node.node_type != "GeometryNodeBake" {
            return Err(PotError::invalid_argument(format!(
                "node `{node_id}` is not a Bake node"
            )));
        }
        let (_, outputs) = node_sockets(group, &parsed_id)
            .ok_or_else(|| evaluation("Bake node has invalid bake items"))?;
        if let Some((socket, _)) = outputs.first() {
            let _ = evaluator.output_value(&parsed_id, socket)?;
        }
    }
    let mut instances = geometry.instance_attributes;
    if !collect_instance_attributes {
        instances.attributes.clear();
    }
    Ok(GraphEvaluation {
        mesh: geometry.mesh,
        instance_paths: geometry.instance_paths,
        instances,
        bake_captures: evaluator.bake_captures,
        bake_material_refs: evaluator.bake_material_refs,
    })
}

fn validate_graph(group: &NodeGroup, node_groups: &Registry<NodeGroup>) -> Result<()> {
    for (node_id, node) in &group.nodes {
        if !is_known_node_type(&node.node_type) {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("unsupported graph node type `{}`", node.node_type),
                json!({"feature_id": format!("graph.node.{}", node.node_type), "node_id": node_id}),
            ));
        }
        let (input_sockets, _) =
            node_sockets_with_groups(group, node_id, node_groups).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    "graph node has no socket schema",
                )
            })?;
        if let Some(unknown) = node
            .inputs
            .keys()
            .find(|socket| !input_sockets.iter().any(|(name, _)| name == *socket))
        {
            return Err(PotError::with_details(
                ErrorCode::InvalidOperation,
                format!("unknown input socket `{unknown}` on `{}`", node.node_type),
                json!({"node_id":node_id,"socket":unknown}),
            ));
        }
    }
    super::sockets::validate_group_links_with_groups(group, node_groups)?;
    validate_acyclic(group)
}

fn validate_acyclic(group: &NodeGroup) -> Result<()> {
    fn visit(
        id: &Id,
        group: &NodeGroup,
        active: &mut HashSet<Id>,
        complete: &mut HashSet<Id>,
    ) -> Result<()> {
        if complete.contains(id) {
            return Ok(());
        }
        if !active.insert(id.clone()) {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "geometry node graph contains a cycle",
                json!({"node_id":id}),
            ));
        }
        for link in group.links.iter().filter(|link| &link.to_node == id) {
            visit(&link.from_node, group, active, complete)?;
        }
        active.remove(id);
        complete.insert(id.clone());
        Ok(())
    }

    let mut active = HashSet::new();
    let mut complete = HashSet::new();
    for id in group.nodes.keys() {
        visit(id, group, &mut active, &mut complete)?;
    }
    Ok(())
}
fn select_group_output_node(group: &NodeGroup) -> Result<Id> {
    let mut output_nodes = group
        .nodes
        .iter()
        .filter(|(_, node)| node.node_type == "NodeGroupOutput");
    let Some((first_id, _)) = output_nodes.next() else {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "geometry node group has no NodeGroupOutput",
        ));
    };
    if output_nodes.next().is_none() {
        return Ok(first_id.clone());
    }
    let mut active_output = None;
    for id in group
        .nodes
        .iter()
        .filter(|(_, node)| {
            node.node_type == "NodeGroupOutput"
                && node
                    .properties
                    .get("is_active_output")
                    .and_then(Value::as_bool)
                    == Some(true)
        })
        .map(|(id, _)| id)
    {
        if active_output.replace(id).is_some() {
            active_output = None;
            break;
        }
    }
    if let Some(id) = active_output {
        return Ok(id.clone());
    }
    Err(PotError::with_details(
        ErrorCode::EvaluationFailed,
        "geometry node group must have exactly one active NodeGroupOutput",
        json!({"output_nodes":group.nodes.iter()
            .filter_map(|(id, node)| (node.node_type == "NodeGroupOutput").then_some(id))
            .collect::<Vec<_>>()}),
    ))
}

fn socket_is_geometry(socket_type: &str) -> bool {
    matches!(socket_type, "geometry" | "NodeSocketGeometry")
}

fn typed_socket_value(socket_type: &str, value: &Value) -> Result<Value> {
    let kind = match socket_type {
        "float"
        | "NodeSocketFloat"
        | "NodeSocketFloatFactor"
        | "NodeSocketFloatAngle"
        | "NodeSocketFloatDistance"
        | "NodeSocketFloatPercentage"
        | "NodeSocketFloatWavelength"
        | "NodeSocketFloatColorTemperature" => "float",
        "integer" | "NodeSocketInt" => "integer",
        "boolean" | "NodeSocketBool" => "boolean",
        "vector"
        | "NodeSocketVector"
        | "NodeSocketVectorDirection"
        | "NodeSocketVectorEuler"
        | "NodeSocketVectorTranslation"
        | "NodeSocketVectorXYZ" => "vector",
        "color" | "NodeSocketColor" => "color",
        "rotation" | "NodeSocketRotation" => "rotation",
        "matrix" | "NodeSocketMatrix" => "matrix",
        "menu" | "NodeSocketMenu" => "menu",
        "string" | "NodeSocketString" => "string",
        "object" | "NodeSocketObject" => "object",
        "collection" | "NodeSocketCollection" => "collection",
        "material" | "NodeSocketMaterial" => "material",
        "image" | "NodeSocketImage" => "image",
        "geometry" | "NodeSocketGeometry" => "geometry",
        "shader" | "NodeSocketShader" => "shader",
        _ => {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("geometry group socket type `{socket_type}` is not implemented"),
                json!({"feature_id":format!("graph.socket.{socket_type}")}),
            ));
        }
    };
    if value.is_null() {
        return Ok(match kind {
            "float" => json!(0.0),
            "integer" => json!(0),
            "vector" | "rotation" => json!([0.0, 0.0, 0.0]),
            "color" => json!([0.0, 0.0, 0.0, 1.0]),
            "matrix" => json!([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]),
            "string" | "menu" => json!(""),
            _ => Value::Null,
        });
    }
    let valid_number = |value: &Value| value.as_f64().is_some_and(f64::is_finite);
    match kind {
        "float" if valid_number(value) => Ok(value.clone()),
        "integer"
            if value
                .as_i64()
                .is_some_and(|value| i32::try_from(value).is_ok()) =>
        {
            Ok(value.clone())
        }
        "boolean" if value.is_boolean() => Ok(value.clone()),
        "vector" | "rotation" => {
            let values = value
                .as_array()
                .ok_or_else(|| evaluation(format!("{kind} socket input must be an array")))?;
            let valid_len = if kind == "rotation" {
                matches!(values.len(), 3 | 4)
            } else {
                values.len() == 3
            };
            if !valid_len || !values.iter().all(valid_number) {
                return Err(evaluation(format!(
                    "{kind} socket input has an invalid vector representation"
                )));
            }
            Ok(value.clone())
        }
        "color" => {
            let values = value
                .as_array()
                .filter(|values| matches!(values.len(), 3 | 4))
                .ok_or_else(|| evaluation("color socket input must be RGB or RGBA"))?;
            if !values.iter().all(valid_number) {
                return Err(evaluation("color socket input must be finite"));
            }
            if values.len() == 4 {
                Ok(value.clone())
            } else {
                Ok(json!([values[0], values[1], values[2], 1.0]))
            }
        }
        "matrix" => {
            let rows = value
                .as_array()
                .filter(|rows| rows.len() == 4)
                .ok_or_else(|| evaluation("matrix socket input must be a 4x4 row array"))?;
            if !rows.iter().all(|row| {
                row.as_array()
                    .is_some_and(|row| row.len() == 4 && row.iter().all(valid_number))
            }) {
                return Err(evaluation(
                    "matrix socket input must be a finite 4x4 row array",
                ));
            }
            Ok(value.clone())
        }
        "menu" | "string" if value.is_string() => Ok(value.clone()),
        "object" | "collection" | "material" | "image" => {
            let expected = match kind {
                "object" => "Object",
                "collection" => "Collection",
                "material" => "Material",
                "image" => "Image",
                _ => return Err(evaluation("invalid resource socket type")),
            };
            if let Some(reference) = value.as_str() {
                if reference.is_empty() {
                    return Err(evaluation(format!(
                        "{kind} socket reference must not be empty"
                    )));
                }
                return Ok(value.clone());
            }
            let reference = value.as_object().ok_or_else(|| {
                evaluation(format!("{kind} socket input must be a resource reference"))
            })?;
            let reference_id = reference
                .get("id")
                .or_else(|| reference.get("name"))
                .and_then(Value::as_str);
            if reference.get("id_type").and_then(Value::as_str) != Some(expected)
                || reference_id.is_none_or(str::is_empty)
            {
                return Err(evaluation(format!(
                    "{kind} socket input must reference a Blender {expected}"
                )));
            }
            Ok(value.clone())
        }
        "shader" | "geometry" => Err(evaluation(format!(
            "{kind} socket value must come from a graph link"
        ))),
        _ => Err(evaluation(format!(
            "{kind} socket input has an invalid JSON type"
        ))),
    }
}

fn group_socket_name_is_unambiguous(sockets: &[super::GraphSocket], name: &str) -> bool {
    sockets
        .iter()
        .filter(|candidate| candidate.name == name)
        .count()
        == 1
        && !sockets.iter().any(|candidate| candidate.id == name)
}

pub(super) fn group_interface_socket<'a>(
    sockets: &'a [super::GraphSocket],
    key: &str,
) -> Option<&'a super::GraphSocket> {
    sockets.iter().find(|socket| socket.id == key).or_else(|| {
        group_socket_name_is_unambiguous(sockets, key)
            .then(|| sockets.iter().find(|socket| socket.name == key))
            .flatten()
    })
}

fn reachable_group_inputs(
    group: &NodeGroup,
    node_groups: &Registry<NodeGroup>,
    output_node: &Id,
    output_socket: &str,
    group_stack: &mut Vec<Id>,
) -> Result<BTreeSet<String>> {
    fn visit_output(
        group: &NodeGroup,
        node_groups: &Registry<NodeGroup>,
        node_id: &Id,
        socket: &str,
        group_stack: &mut Vec<Id>,
        visited: &mut BTreeSet<(Id, String)>,
        active: &mut BTreeSet<(Id, String)>,
        required: &mut BTreeSet<String>,
    ) -> Result<()> {
        let key = (node_id.clone(), socket.to_owned());
        if visited.contains(&key) {
            return Ok(());
        }
        if !active.insert(key.clone()) {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "geometry node graph contains an evaluation cycle",
                json!({"node_id":node_id,"socket":socket}),
            ));
        }
        let node = group.nodes.get(node_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("graph node `{node_id}` was not found"),
            )
        })?;
        if node.node_type == "NodeGroupInput" {
            let input_socket = group_interface_socket(&group.interface.inputs, socket)
                .ok_or_else(|| evaluation(format!("unknown group input socket `{socket}`")))?;
            required.insert(input_socket.id.clone());
        } else if node.node_type == "GeometryNodeGroup" {
            let group_id = node
                .properties
                .get("node_tree")
                .and_then(Value::as_str)
                .ok_or_else(|| evaluation("Geometry Node Group has no node_tree reference"))?;
            let group_id = Id::new(group_id.to_owned()).map_err(|_| {
                evaluation("Geometry Node Group has an invalid node_tree reference")
            })?;
            if group_stack.contains(&group_id) {
                let mut cycle = group_stack.clone();
                cycle.push(group_id.clone());
                return Err(PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    "geometry node groups contain a recursive cycle",
                    json!({"group_ids":cycle}),
                ));
            }
            let child = node_groups.get(&group_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::TargetNotFound,
                    format!("geometry node group `{group_id}` was not found"),
                )
            })?;
            let output_socket = if child.interface.outputs.is_empty() && socket == "Geometry" {
                None
            } else {
                Some(
                    group_interface_socket(&child.interface.outputs, socket).ok_or_else(|| {
                        evaluation(format!("unknown output socket `{socket}` on node group"))
                    })?,
                )
            };
            let child_output_node = select_group_output_node(child)?;
            let child_output_socket = output_socket.map_or("Geometry", |socket| socket.id.as_str());
            group_stack.push(group_id);
            let child_inputs = reachable_group_inputs(
                child,
                node_groups,
                &child_output_node,
                child_output_socket,
                group_stack,
            )?;
            group_stack.pop();
            for child_input in &child.interface.inputs {
                if !child_inputs.contains(&child_input.id) {
                    continue;
                }
                let input_link_socket = if group
                    .links
                    .iter()
                    .any(|link| link.to_node == *node_id && link.to_socket == child_input.id)
                {
                    Some(child_input.id.as_str())
                } else if group_socket_name_is_unambiguous(
                    &child.interface.inputs,
                    &child_input.name,
                ) {
                    Some(child_input.name.as_str())
                } else {
                    None
                };
                let Some(input_link_socket) = input_link_socket else {
                    continue;
                };
                for link in group
                    .links
                    .iter()
                    .filter(|link| link.to_node == *node_id && link.to_socket == *input_link_socket)
                {
                    visit_output(
                        group,
                        node_groups,
                        &link.from_node,
                        &link.from_socket,
                        group_stack,
                        visited,
                        active,
                        required,
                    )?;
                }
            }
        } else {
            for link in group.links.iter().filter(|link| &link.to_node == node_id) {
                visit_output(
                    group,
                    node_groups,
                    &link.from_node,
                    &link.from_socket,
                    group_stack,
                    visited,
                    active,
                    required,
                )?;
            }
        }
        active.remove(&key);
        visited.insert(key);
        Ok(())
    }

    let mut required = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut active = BTreeSet::new();
    let output_socket_name = group
        .interface
        .outputs
        .iter()
        .find(|socket| socket.id == output_socket)
        .filter(|socket| group_socket_name_is_unambiguous(&group.interface.outputs, &socket.name))
        .map(|socket| socket.name.as_str());
    for link in group.links.iter().filter(|link| {
        link.to_node == *output_node
            && (link.to_socket == output_socket
                || output_socket_name.is_some_and(|name| link.to_socket == name))
    }) {
        visit_output(
            group,
            node_groups,
            &link.from_node,
            &link.from_socket,
            group_stack,
            &mut visited,
            &mut active,
            &mut required,
        )?;
    }
    Ok(required)
}

impl Evaluator<'_> {
    fn output_value(&mut self, id: &Id, socket: &str) -> Result<EvalValue> {
        let key = (id.clone(), socket.to_owned());
        if let Some(value) = self.cache.get(&key) {
            return Ok(value.clone());
        }
        if !self.active.insert(id.clone()) {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "geometry node graph contains an evaluation cycle",
                json!({"node_id":id}),
            ));
        }
        let node = self.group.nodes.get(id).cloned().ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("graph node `{id}` was not found"),
            )
        })?;
        let result = self.evaluate_node(id, &node, socket);
        self.active.remove(id);
        let mut value = result?;
        if let EvalValue::Geometry(geometry) = &mut value
            && origindex_values(&geometry.mesh).is_none()
            && self.input.and_then(origindex_values).is_some()
        {
            let vertex_count = geometry.mesh.vertices.len();
            set_origindex(&mut geometry.mesh, &vec![ORIGINDEX_NONE; vertex_count]);
        }
        self.cache.insert(key, value.clone());
        Ok(value)
    }

    fn evaluate_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        match node.node_type.as_str() {
            "NodeGroupInput" => self.group_input_value(socket),
            "GeometryNodeGroup" => self.group_node_value(id, node, socket),
            "NodeGroupOutput" => self.input_value(
                id,
                socket,
                EvalValue::Geometry(GeometryValue {
                    mesh: self.input.cloned().unwrap_or_default(),
                    instance_paths: Vec::new(),
                    instance_attributes: EvaluatedInstances::default(),
                }),
            ),
            "GeometryNodeMeshCube" => self.mesh_primitive(id, node, "box"),
            "GeometryNodeMeshGrid" => self.mesh_primitive(id, node, "grid"),
            "GeometryNodeMeshUVSphere" => self.mesh_primitive(id, node, "uv_sphere"),
            "GeometryNodeMeshIcoSphere" => self.mesh_primitive(id, node, "ico_sphere"),
            "GeometryNodeMeshCylinder" => self.mesh_primitive(id, node, "cylinder"),
            "GeometryNodeMeshCone" => self.mesh_primitive(id, node, "cone"),
            "GeometryNodeMeshCircle" => self.mesh_primitive(id, node, "circle"),
            "GeometryNodeMeshLine" => self.mesh_line(id, node),
            "GeometryNodeTransform" => self.transform_node(id, node),
            "GeometryNodeSetPosition" => self.set_position_node(id, node),
            "GeometryNodeJoinGeometry" => self.join_node(id),
            "GeometryNodeInstanceOnPoints" => self.instance_on_points(id, node),
            "GeometryNodeRealizeInstances" => self.realize_instances(id, node),
            "GeometryNodeDistributePointsOnFaces" => self.distribute_points(id, node, socket),
            "GeometryNodeMeshToPoints" => self.mesh_to_points(id, node),
            "GeometryNodeSubdivisionSurface" => self.subdivision(id, node),
            "GeometryNodeInputPosition" => Ok(EvalValue::Field(Field::Position)),
            "GeometryNodeInputNormal" => Ok(EvalValue::Field(Field::Normal)),
            "GeometryNodeInputIndex" => Ok(EvalValue::Field(Field::Index)),
            "GeometryNodeStoreNamedAttribute" => self.store_named_attribute(id, node),
            "GeometryNodeInputNamedAttribute" => self.input_named_attribute(id, node, socket),
            "GeometryNodeCaptureAttribute" => self.capture_attribute(id, node, socket),
            "ShaderNodeMath" => self.math_node(id, node),
            "ShaderNodeVectorMath" => self.vector_math_node(id, node, socket),
            "FunctionNodeIntegerMath" => self.integer_math_node(id, node),
            "FunctionNodeBooleanMath" => self.boolean_math_node(id, node),
            "FunctionNodeCompare" => self.compare_node(id, node, socket),
            "ShaderNodeMapRange" => self.map_range_node(id, node, socket),
            "ShaderNodeClamp" => self.clamp_node(id, node, socket),
            "ShaderNodeMixRGB" => self.mix_rgb_node(id, node, socket),
            "ShaderNodeMix" => self.mix_node(id, node, socket),
            "ShaderNodeValToRGB" => self.color_ramp_node(id, node, socket),
            "ShaderNodeFloatCurve" => self.curve_node(id, node, socket, "FLOAT"),
            "ShaderNodeVectorCurve" => self.curve_node(id, node, socket, "VECTOR"),
            "ShaderNodeRGBCurve" => self.curve_node(id, node, socket, "RGBA"),
            "ShaderNodeCombineXYZ" => self.combine_xyz(id, node),
            "ShaderNodeSeparateXYZ" => self.separate_xyz(id, node, socket),
            "ShaderNodeTexWave"
            | "ShaderNodeTexMagic"
            | "ShaderNodeTexBrick"
            | "ShaderNodeTexWhiteNoise"
            | "ShaderNodeTexGabor" => self.texture_node(id, node, socket),
            "FunctionNodeRandomValue" => self.random_node(id, node, socket),
            "FunctionNodeAlignEulerToVector"
            | "FunctionNodeAlignRotationToVector"
            | "FunctionNodeAxesToRotation"
            | "FunctionNodeAxisAngleToRotation"
            | "FunctionNodeCombineMatrix"
            | "FunctionNodeCombineTransform"
            | "FunctionNodeEulerToRotation"
            | "FunctionNodeInputRotation"
            | "FunctionNodeInvertMatrix"
            | "FunctionNodeInvertRotation"
            | "FunctionNodeMatrixDeterminant"
            | "FunctionNodeMatrixMultiply"
            | "FunctionNodeMatrixSVD"
            | "FunctionNodeProjectPoint"
            | "FunctionNodeQuaternionToRotation"
            | "FunctionNodeRotateEuler"
            | "FunctionNodeRotateRotation"
            | "FunctionNodeRotateVector"
            | "FunctionNodeRotationToAxisAngle"
            | "FunctionNodeRotationToEuler"
            | "FunctionNodeRotationToQuaternion"
            | "FunctionNodeSeparateMatrix"
            | "FunctionNodeSeparateTransform"
            | "FunctionNodeTransformDirection"
            | "FunctionNodeTransformPoint"
            | "FunctionNodeTransposeMatrix" => self.rotation_matrix_node(id, node, socket),
            "GeometryNodeBake" => self.bake_node(id, node, socket),
            "GeometryNodeRepeatInput" => self.repeat_input_value(id, node, socket),
            "GeometryNodeRepeatOutput" => self.repeat_output_value(id, node, socket),
            "GeometryNodeSimulationInput" => self.simulation_input_value(id, node, socket),
            "GeometryNodeSimulationOutput" => self.simulation_output_value(id, node, socket),
            _ => Err(PotError::new(
                ErrorCode::UnsupportedFeature,
                format!(
                    "native graph evaluator does not implement `{}`",
                    node.node_type
                ),
            )),
        }
    }
    fn rotation_matrix_node(
        &mut self,
        id: &Id,
        node: &GraphNode,
        output: &str,
    ) -> Result<EvalValue> {
        let (input_sockets, output_sockets) = node_sockets(self.group, id)
            .ok_or_else(|| evaluation("rotation or matrix node has no socket schema"))?;
        if !output_sockets.iter().any(|(socket, _)| socket == output) {
            return Err(evaluation(format!(
                "unknown output socket `{output}` on `{}`",
                node.node_type
            )));
        }
        let mut inputs = BTreeMap::new();
        for (socket, _) in input_sockets {
            let fallback = default_input(&node.node_type, &socket).unwrap_or(Value::Null);
            let input =
                self.input_value_with_node(id, node, &socket, EvalValue::Scalar(fallback))?;
            let field = match input {
                EvalValue::Field(field) => field,
                EvalValue::Scalar(value) => Field::Constant(value),
                EvalValue::Geometry(_) => {
                    return Err(evaluation(format!(
                        "socket `{socket}` on `{}` received geometry",
                        node.node_type
                    )));
                }
            };
            inputs.insert(socket, Box::new(field));
        }
        Ok(EvalValue::Field(Field::Function {
            node_type: node.node_type.clone(),
            properties: node.properties.clone(),
            inputs,
            output: output.to_owned(),
        }))
    }

    fn bake_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        let node_id = id.as_str();
        if let Some(record) = self.bake_records.get(node_id) {
            let baked = read_bake_value(record, self.frame, socket)?;
            if let Some(baked) = baked {
                if let NodeBakeValue::Geometry { material_refs, .. } = &baked {
                    self.bake_material_refs
                        .insert(node_id.to_owned(), material_refs.clone());
                }
                return baked.into_eval_value(id, self.require_instance_attributes);
            }
        }
        let items = node
            .properties
            .get("bake_items")
            .and_then(Value::as_array)
            .ok_or_else(|| evaluation("Bake node has no bake items"))?;
        let socket_type = node_sockets(self.group, id)
            .and_then(|(inputs, _)| {
                inputs
                    .into_iter()
                    .find(|(identifier, _)| identifier == socket)
                    .map(|(_, socket_type)| socket_type)
            })
            .ok_or_else(|| evaluation("Bake node output socket is unknown"))?;
        let fallback = if socket_type == "geometry" {
            EvalValue::Geometry(GeometryValue {
                mesh: self.input.cloned().unwrap_or_default(),
                instance_paths: Vec::new(),
                instance_attributes: self.input_instances_for_graph(),
            })
        } else {
            EvalValue::Scalar(Value::Null)
        };
        let value = self.input_value_with_node(id, node, socket, fallback)?;
        if self.capture_bake_nodes.contains(node_id) {
            let mut inputs = Vec::with_capacity(items.len());
            for item in items {
                let identifier = bake_item_identifier(item)
                    .ok_or_else(|| evaluation("Bake item has no socket identifier"))?;
                let item_type = node_sockets(self.group, id)
                    .and_then(|(input_sockets, _)| {
                        input_sockets
                            .into_iter()
                            .find(|(candidate, _)| candidate == &identifier)
                            .map(|(_, socket_type)| socket_type)
                    })
                    .ok_or_else(|| evaluation("Bake item has an unsupported socket type"))?;
                let input_fallback = if item_type == "geometry" {
                    EvalValue::Geometry(GeometryValue {
                        mesh: self.input.cloned().unwrap_or_default(),
                        instance_paths: Vec::new(),
                        instance_attributes: self.input_instances_for_graph(),
                    })
                } else {
                    EvalValue::Scalar(Value::Null)
                };
                let input_value =
                    self.input_value_with_node(id, node, &identifier, input_fallback)?;
                inputs.push((item, identifier, item_type, input_value));
            }
            let geometry_index = inputs
                .iter()
                .position(|(_, _, _, value)| matches!(value, EvalValue::Geometry(_)));
            let geometry = geometry_index.and_then(|index| match &inputs[index].3 {
                EvalValue::Geometry(geometry) => Some(geometry),
                _ => None,
            });
            let geometry_identifier = geometry_index.map(|index| inputs[index].1.clone());
            let mut captured_attributes = Vec::new();
            let mut values = BTreeMap::new();
            for (item, identifier, socket_type, input) in &inputs {
                let baked = match input {
                    EvalValue::Geometry(geometry) => NodeBakeValue::Geometry {
                        mesh: geometry.mesh.clone(),
                        instance_paths: geometry.instance_paths.clone(),
                        instance_columns: Some(
                            serde_json::to_value(&geometry.instance_attributes).map_err(
                                |error| {
                                    PotError::new(
                                        ErrorCode::InternalError,
                                        format!(
                                            "could not serialize bake instance columns: {error}"
                                        ),
                                    )
                                },
                            )?,
                        ),
                        material_refs: Vec::new(),
                    },
                    EvalValue::Scalar(value) => NodeBakeValue::Scalar {
                        value: value.clone(),
                    },
                    EvalValue::Field(field) => {
                        let geometry = geometry.ok_or_else(|| {
                            evaluation("Bake node field items require a geometry bake item")
                        })?;
                        let domain = item
                            .get("attribute_domain")
                            .and_then(Value::as_str)
                            .unwrap_or("POINT")
                            .to_owned();
                        if domain == "INSTANCE" && !geometry.instance_attributes.is_empty() {
                            return Err(unsupported_instance_feature(
                                id,
                                "graph.instance_component.GeometryNodeBake",
                                "Bake node field items on the INSTANCE domain are not implemented",
                            ));
                        }
                        let count = domain_element_count(&geometry.mesh, &domain)?;
                        let mut field_values = Vec::with_capacity(count);
                        for index in 0..count {
                            field_values.push(field_json(field, &geometry.mesh, index, &domain)?);
                        }
                        let attribute_name = format!(".bake_{node_id}_{identifier}");
                        captured_attributes.push((attribute_name.clone(), domain, field_values));
                        NodeBakeValue::Field { attribute_name }
                    }
                };
                if socket_type == "geometry" && !matches!(baked, NodeBakeValue::Geometry { .. }) {
                    return Err(evaluation(
                        "Bake geometry item did not evaluate to geometry",
                    ));
                }
                values.insert(identifier.clone(), baked);
            }
            if !captured_attributes.is_empty() {
                let geometry_identifier = geometry_identifier.ok_or_else(|| {
                    evaluation("Bake node field items require a geometry bake item")
                })?;
                let Some(NodeBakeValue::Geometry { mesh, .. }) =
                    values.get_mut(&geometry_identifier)
                else {
                    return Err(evaluation(
                        "Bake node geometry item did not evaluate to geometry",
                    ));
                };
                for (name, domain, data) in captured_attributes {
                    mesh.attributes
                        .insert(name, json!({"domain":domain,"data":data}));
                }
            }
            self.bake_captures.insert(
                node_id.to_owned(),
                NodeBakeFrame {
                    frame: self.frame,
                    values,
                },
            );
        }
        Ok(value)
    }

    fn repeat_input_value(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        match socket {
            "Geometry" => {
                if let Some(state) = self.repeat_states.get(id) {
                    return Ok(EvalValue::Geometry(state.clone()));
                }
                self.input_geometry(id, node, "Geometry")
            }
            "Iteration" => Ok(EvalValue::Scalar(json!(
                self.repeat_iterations.get(id).copied().unwrap_or(0)
            ))),
            _ => Err(evaluation("unknown Repeat Input output")),
        }
    }

    fn repeat_output_value(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
    ) -> Result<EvalValue> {
        if socket != "Geometry" {
            return Err(evaluation("unknown Repeat Output output"));
        }
        let input_id = self.paired_zone_input(id, "GeometryNodeRepeatInput")?;
        let input_node = self
            .group
            .nodes
            .get(&input_id)
            .cloned()
            .ok_or_else(|| evaluation("paired Repeat Input node disappeared"))?;
        let iterations = self.integer_input(&input_id, &input_node, "Iterations", 1)?;
        if !(0..=MAX_ZONE_ITERATIONS).contains(&iterations) {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "repeat-zone iteration count is outside the supported range",
            ));
        }
        let initial = self.input_value_with_node(
            &input_id,
            &input_node,
            "Geometry",
            EvalValue::Geometry(GeometryValue {
                mesh: self.input.cloned().unwrap_or_default(),
                instance_paths: Vec::new(),
                instance_attributes: EvaluatedInstances::default(),
            }),
        )?;
        let mut state = into_geometry(initial)?;
        for iteration in 0..iterations {
            self.repeat_states.insert(input_id.clone(), state.clone());
            self.repeat_iterations.insert(input_id.clone(), iteration);
            self.cache.clear();
            let next =
                self.input_value_with_node(id, node, "Geometry", EvalValue::Geometry(state))?;
            state = into_geometry(next)?;
        }
        self.repeat_states.remove(&input_id);
        self.repeat_iterations.remove(&input_id);
        Ok(EvalValue::Geometry(state))
    }

    fn simulation_input_value(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
    ) -> Result<EvalValue> {
        match socket {
            "Geometry" => {
                if let Some(state) = self.simulation_states.get(id) {
                    return Ok(EvalValue::Geometry(state.clone()));
                }
                self.input_geometry(id, node, "Geometry")
            }
            "Delta Time" => Ok(EvalValue::Scalar(json!(
                self.simulation_delta_times.get(id).copied().unwrap_or(0.0) as f32
            ))),
            _ => Err(evaluation("unknown Simulation Input output")),
        }
    }

    fn simulation_output_value(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
    ) -> Result<EvalValue> {
        if socket != "Geometry" {
            return Err(evaluation("unknown Simulation Output output"));
        }
        let input_id = self.paired_zone_input(id, "GeometryNodeSimulationInput")?;
        let input_node = self
            .group
            .nodes
            .get(&input_id)
            .cloned()
            .ok_or_else(|| evaluation("paired Simulation Input node disappeared"))?;
        let start_frame = property_i64(node, "start_frame", 1)?;
        let target_frame = floor_to_i64(self.frame)?;
        let initial = self.input_value_with_node(
            &input_id,
            &input_node,
            "Geometry",
            EvalValue::Geometry(GeometryValue {
                mesh: self.input.cloned().unwrap_or_default(),
                instance_paths: Vec::new(),
                instance_attributes: EvaluatedInstances::default(),
            }),
        )?;
        let mut state = into_geometry(initial)?;
        if target_frame < start_frame {
            return Ok(EvalValue::Geometry(state));
        }
        if target_frame.saturating_sub(start_frame) >= MAX_ZONE_ITERATIONS {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "simulation-zone frame range exceeds the supported limit",
            ));
        }
        let signature = self
            .simulation_signature
            .clone()
            .ok_or_else(|| evaluation("simulation graph cache signature is missing"))?;
        for frame in start_frame..=target_frame {
            self.simulation_delta_times.insert(
                input_id.clone(),
                if frame == start_frame {
                    0.0
                } else {
                    self.frame_duration
                },
            );
            if let Some(cached) = cached_simulation_state(&signature, &input_id, frame)? {
                state = cached;
                continue;
            }
            self.simulation_states
                .insert(input_id.clone(), state.clone());
            self.cache.clear();
            let skip = as_field(self.input_value_with_node(
                id,
                node,
                "Skip",
                EvalValue::Scalar(json!(false)),
            )?);
            let skip = field_bool(&skip, &state.mesh, 0, "POINT", false)?;
            let next = self.input_value_with_node(
                id,
                node,
                "Geometry",
                EvalValue::Geometry(state.clone()),
            )?;
            if !skip {
                state = into_geometry(next)?;
            }
            self.simulation_states.remove(&input_id);
            cache_simulation_state(&signature, &input_id, frame, state.clone())?;
        }
        self.simulation_states.remove(&input_id);
        Ok(EvalValue::Geometry(state))
    }

    fn paired_zone_input(&self, output_id: &Id, input_type: &str) -> Result<Id> {
        let mut frontier = VecDeque::from([output_id.clone()]);
        let mut visited = HashSet::from([output_id.clone()]);
        while !frontier.is_empty() {
            let breadth = frontier.len();
            let mut matches = Vec::new();
            for _ in 0..breadth {
                let Some(node_id) = frontier.pop_front() else {
                    continue;
                };
                for link in self
                    .group
                    .links
                    .iter()
                    .filter(|link| link.to_node == node_id)
                {
                    let Some(source) = self.group.nodes.get(&link.from_node) else {
                        continue;
                    };
                    if source.node_type == input_type {
                        matches.push(link.from_node.clone());
                    } else if visited.insert(link.from_node.clone()) {
                        frontier.push_back(link.from_node.clone());
                    }
                }
            }
            if !matches.is_empty() {
                matches.sort();
                matches.dedup();
                return if matches.len() == 1 {
                    Ok(matches.remove(0))
                } else {
                    Err(evaluation(
                        "zone output has multiple paired zone input nodes",
                    ))
                };
            }
        }
        Err(PotError::with_details(
            ErrorCode::InvalidOperation,
            "zone output is not connected to a matching zone input",
            json!({"node_id":output_id,"input_type":input_type}),
        ))
    }

    fn input_instances_for_graph(&self) -> EvaluatedInstances {
        let Some(instances) = self.input_instances else {
            return EvaluatedInstances::default();
        };
        if self.capture_instance_attributes {
            instances.clone()
        } else {
            EvaluatedInstances {
                transforms: instances.transforms.clone(),
                attributes: BTreeMap::new(),
            }
        }
    }

    fn input_instance_paths_for_graph(&self) -> Vec<Vec<Id>> {
        self.input_instance_paths
            .map(<[Vec<Id>]>::to_vec)
            .unwrap_or_default()
    }

    fn group_input_value(&self, socket: &str) -> Result<EvalValue> {
        let Some(interface_socket) = group_interface_socket(&self.group.interface.inputs, socket)
        else {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                format!("unknown group input socket `{socket}`"),
            ));
        };
        let Some(socket_index) = self
            .group
            .interface
            .inputs
            .iter()
            .position(|candidate| candidate.id == interface_socket.id)
        else {
            return Err(evaluation(
                "group input socket is missing from its interface",
            ));
        };
        if let Some(value) = self.group_inputs.get(&interface_socket.id) {
            return Ok(value.clone());
        }
        if socket_is_geometry(&interface_socket.socket_type) {
            let has_object_geometry = self.top_level && socket_index == 0;
            return Ok(EvalValue::Geometry(GeometryValue {
                mesh: if has_object_geometry {
                    self.input.cloned().unwrap_or_default()
                } else {
                    Mesh::new()
                },
                instance_paths: if has_object_geometry {
                    self.input_instance_paths_for_graph()
                } else {
                    Vec::new()
                },
                instance_attributes: if has_object_geometry {
                    self.input_instances_for_graph()
                } else {
                    EvaluatedInstances::default()
                },
            }));
        }
        let value = if self.top_level {
            self.overrides
                .get(&interface_socket.id)
                .or_else(|| {
                    group_socket_name_is_unambiguous(
                        &self.group.interface.inputs,
                        &interface_socket.name,
                    )
                    .then(|| self.overrides.get(&interface_socket.name))
                    .flatten()
                })
                .unwrap_or(&interface_socket.default)
        } else {
            &interface_socket.default
        };
        typed_socket_value(&interface_socket.socket_type, value).map(EvalValue::Scalar)
    }

    fn group_node_value(
        &mut self,
        node_id: &Id,
        node: &GraphNode,
        output_socket: &str,
    ) -> Result<EvalValue> {
        let group_id = node
            .properties
            .get("node_tree")
            .and_then(Value::as_str)
            .ok_or_else(|| evaluation("Geometry Node Group has no node_tree reference"))?;
        let group_id = Id::new(group_id.to_owned())
            .map_err(|_| evaluation("Geometry Node Group has an invalid node_tree reference"))?;
        if self.group_stack.contains(&group_id) {
            let mut cycle = self.group_stack.clone();
            cycle.push(group_id.clone());
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                "geometry node groups contain a recursive cycle",
                json!({"group_ids":cycle}),
            ));
        }
        let child_group = self.node_groups.get(&group_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("geometry node group `{group_id}` was not found"),
            )
        })?;
        if child_group.kind != GraphKind::Geometry {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Geometry Node Group references a non-geometry node tree",
                json!({"node_id":node_id,"node_group":group_id}),
            ));
        }
        validate_graph(child_group, self.node_groups)?;
        let child_output_socket =
            if child_group.interface.outputs.is_empty() && output_socket == "Geometry" {
                None
            } else {
                Some(
                    group_interface_socket(&child_group.interface.outputs, output_socket)
                        .cloned()
                        .ok_or_else(|| {
                            PotError::with_details(
                                ErrorCode::InvalidOperation,
                                format!(
                                    "unknown output socket `{output_socket}` on Geometry Node Group"
                                ),
                                json!({"node_id":node_id,"socket":output_socket}),
                            )
                        })?,
                )
            };
        let output_node = select_group_output_node(child_group)?;
        let output_link_socket = child_output_socket
            .as_ref()
            .map_or_else(|| "Geometry".to_owned(), |socket| socket.id.clone());
        let child_input_sockets = child_group.interface.inputs.clone();
        let mut group_stack = self.group_stack.clone();
        group_stack.push(group_id.clone());
        let required_inputs = reachable_group_inputs(
            child_group,
            self.node_groups,
            &output_node,
            &output_link_socket,
            &mut group_stack,
        )?;
        let mut group_inputs = BTreeMap::new();
        for interface_socket in &child_input_sockets {
            if !required_inputs.contains(&interface_socket.id) {
                continue;
            }
            let value =
                self.group_call_input(node_id, node, interface_socket, &child_input_sockets)?;
            group_inputs.insert(interface_socket.id.clone(), value);
        }
        let output_value = match child_output_socket.as_ref() {
            None => EvalValue::Geometry(GeometryValue {
                mesh: Mesh::new(),
                instance_paths: Vec::new(),
                instance_attributes: EvaluatedInstances::default(),
            }),
            Some(socket) if socket_is_geometry(&socket.socket_type) => {
                EvalValue::Geometry(GeometryValue {
                    mesh: Mesh::new(),
                    instance_paths: Vec::new(),
                    instance_attributes: EvaluatedInstances::default(),
                })
            }
            Some(socket) => {
                EvalValue::Scalar(typed_socket_value(&socket.socket_type, &socket.default)?)
            }
        };
        let child_group = self.node_groups.get(&group_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("geometry node group `{group_id}` was not found"),
            )
        })?;
        let output_node_value = child_group.nodes.get(&output_node).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "geometry node group output is missing",
            )
        })?;
        let output_link_socket_name = child_output_socket
            .as_ref()
            .filter(|socket| {
                group_socket_name_is_unambiguous(&child_group.interface.outputs, &socket.name)
            })
            .map(|socket| socket.name.as_str());
        let output_link = child_group
            .links
            .iter()
            .find(|link| link.to_node == output_node && link.to_socket == output_link_socket)
            .or_else(|| {
                output_link_socket_name.and_then(|name| {
                    child_group
                        .links
                        .iter()
                        .find(|link| link.to_node == output_node && link.to_socket == name)
                })
            });
        let output_input_socket = output_link
            .map(|link| link.to_socket.clone())
            .or_else(|| {
                output_node_value
                    .inputs
                    .get_key_value(&output_link_socket)
                    .map(|(key, _)| key.clone())
                    .or_else(|| {
                        output_link_socket_name.and_then(|name| {
                            output_node_value
                                .inputs
                                .get_key_value(name)
                                .map(|(key, _)| key.clone())
                        })
                    })
            })
            .unwrap_or_else(|| output_link_socket.clone());
        let simulation_signature = self.nested_group_signature(&group_id, node_id, child_group)?;
        let mut child = Evaluator {
            group: child_group,
            node_groups: self.node_groups,
            group_stack,
            group_inputs,
            top_level: false,
            subject_name: self.subject_name,
            input: None,
            input_instances: None,
            input_instance_paths: None,
            capture_instance_attributes: self.capture_instance_attributes,
            require_instance_attributes: self.require_instance_attributes,
            overrides: self.overrides,
            frame: self.frame,
            frame_duration: self.frame_duration,
            simulation_delta_times: HashMap::new(),
            simulation_signature: Some(simulation_signature),
            simulation_states: HashMap::new(),
            repeat_states: HashMap::new(),
            repeat_iterations: HashMap::new(),
            active: BTreeSet::new(),
            cache: BTreeMap::new(),
            bake_records: self.bake_records,
            capture_bake_nodes: self.capture_bake_nodes,
            bake_captures: BTreeMap::new(),
            bake_material_refs: BTreeMap::new(),
        };
        child.input_value(&output_node, &output_input_socket, output_value)
    }

    fn group_call_input(
        &mut self,
        node_id: &Id,
        node: &GraphNode,
        interface_socket: &super::GraphSocket,
        interface_sockets: &[super::GraphSocket],
    ) -> Result<EvalValue> {
        let name_is_unambiguous =
            group_socket_name_is_unambiguous(interface_sockets, &interface_socket.name);
        let linked = self
            .group
            .links
            .iter()
            .find(|link| link.to_node == *node_id && link.to_socket == interface_socket.id)
            .or_else(|| {
                name_is_unambiguous
                    .then(|| {
                        self.group.links.iter().find(|link| {
                            link.to_node == *node_id && link.to_socket == interface_socket.name
                        })
                    })
                    .flatten()
            });
        if let Some(link) = linked {
            return self.output_value(&link.from_node, &link.from_socket);
        }
        let value = node.inputs.get(&interface_socket.id).or_else(|| {
            name_is_unambiguous
                .then(|| node.inputs.get(&interface_socket.name))
                .flatten()
        });
        if let Some(value) = value {
            if socket_is_geometry(&interface_socket.socket_type) {
                return Ok(EvalValue::Geometry(GeometryValue {
                    mesh: Mesh::new(),
                    instance_paths: Vec::new(),
                    instance_attributes: EvaluatedInstances::default(),
                }));
            }
            return typed_socket_value(&interface_socket.socket_type, value).map(EvalValue::Scalar);
        }
        if socket_is_geometry(&interface_socket.socket_type) {
            return Ok(EvalValue::Geometry(GeometryValue {
                mesh: Mesh::new(),
                instance_paths: Vec::new(),
                instance_attributes: EvaluatedInstances::default(),
            }));
        }
        typed_socket_value(&interface_socket.socket_type, &interface_socket.default)
            .map(EvalValue::Scalar)
    }
    fn nested_group_signature(
        &self,
        group_id: &Id,
        callsite_id: &Id,
        child_group: &NodeGroup,
    ) -> Result<String> {
        let serialized = if self.capture_instance_attributes {
            serde_json::to_value((
                self.simulation_signature.as_deref(),
                self.group,
                child_group,
                group_id,
                callsite_id,
                self.input,
                self.input_instances,
                self.input_instance_paths,
                self.overrides,
                self.frame_duration,
            ))
        } else {
            serde_json::to_value((
                self.simulation_signature.as_deref(),
                self.group,
                child_group,
                group_id,
                callsite_id,
                self.input,
                self.input_instances.map(|instances| &instances.transforms),
                self.input_instance_paths,
                self.overrides,
                self.frame_duration,
            ))
        }
        .map_err(|error| {
            PotError::new(
                ErrorCode::InternalError,
                format!("could not serialize nested graph simulation inputs: {error}"),
            )
        })?;
        crate::hash::canonicalize(&serialized).map(|bytes| crate::hash::sha256(&bytes))
    }

    fn input_value(
        &mut self,
        node_id: &Id,
        socket: &str,
        fallback: EvalValue,
    ) -> Result<EvalValue> {
        if let Some(link) = self
            .group
            .links
            .iter()
            .find(|link| &link.to_node == node_id && link.to_socket == socket)
        {
            return self.output_value(&link.from_node, &link.from_socket);
        }
        let node = self.group.nodes.get(node_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::TargetNotFound,
                format!("graph node `{node_id}` was not found"),
            )
        })?;
        if let Some(value) = node.inputs.get(socket) {
            return Ok(EvalValue::Scalar(value.clone()));
        }
        Ok(match node_default_input(node, socket) {
            Some(value) => EvalValue::Scalar(value),
            None => fallback,
        })
    }

    fn input_value_with_node(
        &mut self,
        node_id: &Id,
        node: &GraphNode,
        socket: &str,
        fallback: EvalValue,
    ) -> Result<EvalValue> {
        if let Some(link) = self
            .group
            .links
            .iter()
            .find(|link| &link.to_node == node_id && link.to_socket == socket)
        {
            return self.output_value(&link.from_node, &link.from_socket);
        }
        if node.node_type == "GeometryNodeSetPosition" && socket == "Position" {
            return Ok(EvalValue::Field(Field::Position));
        }
        if let Some(value) = node.inputs.get(socket) {
            return Ok(EvalValue::Scalar(value.clone()));
        }
        Ok(match node_default_input(node, socket) {
            Some(value) => EvalValue::Scalar(value),
            None => fallback,
        })
    }

    fn input_geometry(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        self.input_value_with_node(
            id,
            node,
            socket,
            EvalValue::Geometry(GeometryValue {
                mesh: self.input.cloned().unwrap_or_default(),
                instance_paths: Vec::new(),
                instance_attributes: EvaluatedInstances::default(),
            }),
        )
    }

    fn geometry_input(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        match self.input_geometry(id, node, socket)? {
            EvalValue::Geometry(geometry) => {
                let instance_aware = matches!(
                    node.node_type.as_str(),
                    "GeometryNodeRealizeInstances"
                        | "GeometryNodeTransform"
                        | "GeometryNodeSetPosition"
                        | "GeometryNodeMeshToPoints"
                        | "GeometryNodeSubdivisionSurface"
                        | "GeometryNodeStoreNamedAttribute"
                );
                if !geometry.instance_attributes.is_empty() && !instance_aware {
                    return Err(unsupported_instance_feature(
                        id,
                        &format!("graph.instance_component.{}", node.node_type),
                        format!(
                            "node `{}` does not implement the Instances component",
                            node.node_type
                        ),
                    ));
                }
                Ok(EvalValue::Geometry(geometry))
            }
            _ => Err(PotError::new(
                ErrorCode::EvaluationFailed,
                format!("input `{socket}` on `{}` is not geometry", node.node_type),
            )),
        }
    }

    fn realize_instances(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let mut geometry = into_geometry(self.geometry_input(id, node, "Geometry")?)?;
        if !geometry.instance_attributes.is_empty() {
            let selection = as_field(self.input_value_with_node(
                id,
                node,
                "Selection",
                EvalValue::Scalar(json!(true)),
            )?);
            let realize_all = as_field(self.input_value_with_node(
                id,
                node,
                "Realize All",
                EvalValue::Scalar(json!(true)),
            )?);
            let instance_count = geometry.instance_attributes.len();
            let attributes = &geometry.instance_attributes.attributes;
            for (index, transform) in geometry.instance_attributes.transforms.iter().enumerate() {
                if !field_bool_for_instance(
                    &selection,
                    attributes,
                    transform,
                    index,
                    instance_count,
                    true,
                )? {
                    return Err(unsupported_instance_feature(
                        id,
                        "constraint.geometry_attribute.instances.realize_settings",
                        "Realize Instances with a non-default Selection cannot be represented from flattened instance geometry",
                    ));
                }
                if !field_bool_for_instance(
                    &realize_all,
                    attributes,
                    transform,
                    index,
                    instance_count,
                    true,
                )? {
                    return Err(unsupported_instance_feature(
                        id,
                        "constraint.geometry_attribute.instances.realize_settings",
                        "Realize Instances with Realize All disabled or finite-depth realization is not implemented",
                    ));
                }
            }
            if geometry
                .instance_attributes
                .attributes
                .values()
                .any(|column| !column.values.is_empty())
            {
                return Err(unsupported_instance_feature(
                    id,
                    "graph.instance_component.realize_attribute_columns",
                    "Realize Instances cannot propagate instance attribute columns to the realized mesh",
                ));
            }
            geometry.instance_attributes = EvaluatedInstances::default();
        }
        Ok(EvalValue::Geometry(geometry))
    }

    fn mesh_primitive(&mut self, id: &Id, node: &GraphNode, kind: &str) -> Result<EvalValue> {
        let mut mesh = match kind {
            "box" => self.cube_node_mesh(id, node)?,
            "grid" => self.grid_node_mesh(id, node)?,
            "uv_sphere" => self.uv_sphere_node_mesh(id, node)?,
            "cylinder" => self.cylinder_node_mesh(id, node)?,
            "cone" => self.cone_node_mesh(id, node)?,
            "circle" => self.circle_node_mesh(id, node)?,
            "ico_sphere" => self.ico_sphere_node_mesh(id, node)?,
            _ => {
                return Err(PotError::new(
                    ErrorCode::UnsupportedFeature,
                    format!("native graph evaluator does not implement mesh primitive `{kind}`"),
                ));
            }
        };
        mesh.attributes.remove("uv_layers");
        Ok(EvalValue::Geometry(GeometryValue {
            mesh,
            instance_paths: Vec::new(),
            instance_attributes: EvaluatedInstances::default(),
        }))
    }

    fn cube_node_mesh(&mut self, id: &Id, node: &GraphNode) -> Result<Mesh> {
        let default = property_vec3(node, "size", DVec3::ONE)?;
        let size = self.vector_input(id, node, "Size", default, &Mesh::new())?;
        let vertices_x =
            self.integer_input(id, node, "Vertices X", property_i64(node, "vertices_x", 2)?)?;
        let vertices_y =
            self.integer_input(id, node, "Vertices Y", property_i64(node, "vertices_y", 2)?)?;
        let vertices_z =
            self.integer_input(id, node, "Vertices Z", property_i64(node, "vertices_z", 2)?)?;
        geometry_node_cube(size, vertices_x, vertices_y, vertices_z)
    }

    fn grid_node_mesh(&mut self, id: &Id, node: &GraphNode) -> Result<Mesh> {
        let size_x = self.scalar_input(id, node, "Size X", property_f64(node, "size_x", 1.0)?)?;
        let size_y = self.scalar_input(id, node, "Size Y", property_f64(node, "size_y", 1.0)?)?;
        let vertices_x =
            self.integer_input(id, node, "Vertices X", property_i64(node, "vertices_x", 3)?)?;
        let vertices_y =
            self.integer_input(id, node, "Vertices Y", property_i64(node, "vertices_y", 3)?)?;
        let nx = segment_count(vertices_x, 2, "grid X vertices")?;
        let ny = segment_count(vertices_y, 2, "grid Y vertices")?;
        if nx
            .checked_mul(ny)
            .is_none_or(|count| count > MAX_GENERATED_POINTS)
        {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "generated primitive exceeds the vertex limit",
            ));
        }
        if !size_x.is_finite() || !size_y.is_finite() {
            return Err(evaluation("grid sizes must be finite"));
        }
        let mut mesh = Mesh::grid(GridParams {
            size_x: 1.0,
            size_y: 1.0,
            x_subdivisions: u32::try_from(nx)
                .map_err(|_| evaluation("grid X vertex count exceeds supported range"))?,
            y_subdivisions: u32::try_from(ny)
                .map_err(|_| evaluation("grid Y vertex count exceeds supported range"))?,
        })
        .map_err(|error| mesh_evaluation_error(&error))?;
        for vertex in &mut mesh.vertices {
            vertex.co.x *= size_x.abs();
            vertex.co.y *= size_y.abs();
        }
        Ok(mesh)
    }

    fn uv_sphere_node_mesh(&mut self, id: &Id, node: &GraphNode) -> Result<Mesh> {
        let segments =
            self.integer_input(id, node, "Segments", property_i64(node, "segments", 32)?)?;
        let rings = self.integer_input(id, node, "Rings", property_i64(node, "rings", 16)?)?;
        let radius = self.scalar_input(id, node, "Radius", property_f64(node, "radius", 1.0)?)?;
        geometry_node_uv_sphere(segments, rings, radius)
    }

    fn cylinder_node_mesh(&mut self, id: &Id, node: &GraphNode) -> Result<Mesh> {
        if node
            .properties
            .get("fill_type")
            .and_then(Value::as_str)
            .unwrap_or("NGON")
            != "NGON"
        {
            return Err(unsupported_graph_feature(
                id,
                "graph.node.mesh_cylinder.fill_type",
                "Cylinder fill types other than NGON are not implemented",
            ));
        }
        let vertices =
            self.integer_input(id, node, "Vertices", property_i64(node, "vertices", 32)?)?;
        let side_segments = self.integer_input(
            id,
            node,
            "Side Segments",
            property_i64(node, "side_segments", 1)?,
        )?;
        let fill_segments = self.integer_input(
            id,
            node,
            "Fill Segments",
            property_i64(node, "fill_segments", 1)?,
        )?;
        let radius = self.scalar_input(id, node, "Radius", property_f64(node, "radius", 1.0)?)?;
        let depth = self.scalar_input(id, node, "Depth", property_f64(node, "depth", 2.0)?)?;
        segmented_cylinder(
            vertices,
            side_segments,
            fill_segments,
            radius.abs(),
            depth.abs(),
        )
    }

    fn cone_node_mesh(&mut self, id: &Id, node: &GraphNode) -> Result<Mesh> {
        if node
            .properties
            .get("fill_type")
            .and_then(Value::as_str)
            .unwrap_or("NGON")
            != "NGON"
        {
            return Err(unsupported_graph_feature(
                id,
                "graph.node.mesh_cone.fill_type",
                "Cone fill types other than NGON are not implemented",
            ));
        }
        let vertices =
            self.integer_input(id, node, "Vertices", property_i64(node, "vertices", 32)?)?;
        let side_segments = self.integer_input(
            id,
            node,
            "Side Segments",
            property_i64(node, "side_segments", 1)?,
        )?;
        let fill_segments = self.integer_input(
            id,
            node,
            "Fill Segments",
            property_i64(node, "fill_segments", 1)?,
        )?;
        let radius_top =
            self.scalar_input(id, node, "Radius Top", property_f64(node, "radius2", 0.0)?)?;
        let radius_bottom = self.scalar_input(
            id,
            node,
            "Radius Bottom",
            property_f64(node, "radius1", 1.0)?,
        )?;
        let depth = self.scalar_input(id, node, "Depth", property_f64(node, "depth", 2.0)?)?;
        let mut mesh = segmented_cone(
            vertices,
            side_segments,
            fill_segments,
            radius_bottom.abs(),
            radius_top.abs(),
            depth,
        )?;
        for vertex in &mut mesh.vertices {
            vertex.co.z += depth * 0.5;
        }
        Ok(mesh)
    }

    fn circle_node_mesh(&mut self, id: &Id, node: &GraphNode) -> Result<Mesh> {
        if node
            .properties
            .get("fill_type")
            .and_then(Value::as_str)
            .unwrap_or("NONE")
            != "NONE"
        {
            return Err(unsupported_graph_feature(
                id,
                "graph.node.mesh_circle.fill_type",
                "Circle fill types other than NONE are not implemented",
            ));
        }
        let vertices =
            self.integer_input(id, node, "Vertices", property_i64(node, "vertices", 32)?)?;
        let radius = self.scalar_input(id, node, "Radius", property_f64(node, "radius", 1.0)?)?;
        geometry_node_circle(vertices, radius)
    }

    fn ico_sphere_node_mesh(&mut self, id: &Id, node: &GraphNode) -> Result<Mesh> {
        let radius = self.scalar_input(id, node, "Radius", property_f64(node, "radius", 1.0)?)?;
        let subdivisions = self.integer_input(
            id,
            node,
            "Subdivisions",
            property_i64(node, "subdivisions", 1)?,
        )?;
        let subdivisions = u32::try_from(subdivisions)
            .map_err(|_| evaluation("icosphere subdivisions are outside supported range"))?;
        let radius = radius.abs();
        let generation_radius = if radius == 0.0 { 1.0 } else { radius };
        let mut mesh = Mesh::icosphere(IcosphereParams {
            subdivisions,
            radius: generation_radius,
        })
        .map_err(|error| mesh_evaluation_error(&error))?;
        if radius == 0.0 {
            for vertex in &mut mesh.vertices {
                vertex.co = DVec3::ZERO;
            }
        }
        Ok(mesh)
    }

    fn mesh_line(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let mode = node
            .properties
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("OFFSET");
        if mode != "OFFSET" {
            return Err(unsupported_graph_feature(
                id,
                "graph.node.mesh_line.endpoint_mode",
                format!("Mesh Line mode `{mode}` is not implemented"),
            ));
        }
        let count_mode = node
            .properties
            .get("count_mode")
            .and_then(Value::as_str)
            .unwrap_or("TOTAL");
        if count_mode != "TOTAL" {
            return Err(unsupported_graph_feature(
                id,
                "graph.node.mesh_line.resolution_count_mode",
                format!("Mesh Line count mode `{count_mode}` is not implemented"),
            ));
        }
        let count = self.integer_input(id, node, "Count", property_i64(node, "count", 10)?)?;
        if count < 0 {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "mesh line exceeds point limit",
            ));
        }
        let count = usize::try_from(count)
            .map_err(|_| evaluation("mesh line count is outside supported range"))?;
        if count > MAX_GENERATED_POINTS {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "mesh line exceeds point limit",
            ));
        }
        let start_default = property_vec3(node, "start_location", DVec3::ZERO)?;
        let offset_default = property_vec3(node, "offset", DVec3::new(0.0, 0.0, 1.0))?;
        let start = self.vector_input(id, node, "Start Location", start_default, &Mesh::new())?;
        let offset = self.vector_input(id, node, "Offset", offset_default, &Mesh::new())?;

        let points = (0..count)
            .map(|index| {
                let t = interpolation_factor(index, count);
                start + offset * t
            })
            .collect::<Vec<_>>();
        let mut mesh = Mesh::new();
        let mut ids = Vec::with_capacity(points.len());
        for point in points {
            ids.push(
                mesh.insert_vertex(point)
                    .map_err(|error| mesh_evaluation_error(&error))?,
            );
        }
        for pair in ids.windows(2) {
            mesh.insert_edge([pair[0], pair[1]])
                .map_err(|error| mesh_evaluation_error(&error))?;
        }
        Ok(EvalValue::Geometry(GeometryValue {
            mesh,
            instance_paths: Vec::new(),
            instance_attributes: EvaluatedInstances::default(),
        }))
    }

    fn transform_node(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let geometry = self.geometry_input(id, node, "Geometry")?;
        let mut geometry = into_geometry(geometry)?;
        let mode = self.string_input(id, node, "Mode", "Components")?;
        if !mode.eq_ignore_ascii_case("Components") {
            return Err(unsupported_instance_feature(
                id,
                "constraint.geometry_attribute.instances.transform_matrix_mode",
                format!(
                    "GeometryNodeTransform mode `{mode}` is not implemented; only Components mode is supported"
                ),
            ));
        }
        let original = geometry.mesh.clone();
        let translation = as_field(self.input_value_with_node(
            id,
            node,
            "Translation",
            EvalValue::Scalar(json!([0.0, 0.0, 0.0])),
        )?);
        let scale = as_field(self.input_value_with_node(
            id,
            node,
            "Scale",
            EvalValue::Scalar(json!([1.0, 1.0, 1.0])),
        )?);
        let rotation = self.quaternion_input(id, node, "Rotation", &geometry.mesh)?;
        if !rotation.is_finite() {
            return Err(evaluation("transform rotation must be finite"));
        }
        for (index, vertex) in geometry.mesh.vertices.iter_mut().enumerate() {
            let local_translation = field_vec3(&translation, &original, index, DVec3::ZERO)?;
            let local_scale = field_vec3(&scale, &original, index, DVec3::ONE)?;
            if !local_translation.is_finite() || !local_scale.is_finite() {
                return Err(evaluation("transform fields must be finite"));
            }
            let matrix =
                DMat4::from_scale_rotation_translation(local_scale, rotation, local_translation);
            vertex.co = matrix.transform_point3(vertex.co);
        }
        let instance_count = geometry.instance_attributes.len();
        let attributes = &geometry.instance_attributes.attributes;
        let transforms = &mut geometry.instance_attributes.transforms;
        for (index, transform) in transforms.iter_mut().enumerate() {
            let local_translation = field_vec3_for_instance(
                &translation,
                attributes,
                transform,
                index,
                instance_count,
                DVec3::ZERO,
            )?;
            let local_scale = field_vec3_for_instance(
                &scale,
                attributes,
                transform,
                index,
                instance_count,
                DVec3::ONE,
            )?;
            if !local_translation.is_finite() || !local_scale.is_finite() {
                return Err(evaluation("transform fields must be finite"));
            }
            let matrix =
                DMat4::from_scale_rotation_translation(local_scale, rotation, local_translation)
                    * DMat4::from_cols_array(transform);
            let values = matrix.to_cols_array();
            if values.iter().any(|component| !component.is_finite()) {
                return Err(evaluation("transformed instance matrix is not finite"));
            }
            *transform = values;
        }
        Ok(EvalValue::Geometry(geometry))
    }

    fn set_position_node(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let geometry = self.geometry_input(id, node, "Geometry")?;
        let mut geometry = into_geometry(geometry)?;
        let original = geometry.mesh.clone();
        let selection =
            self.input_value_with_node(id, node, "Selection", EvalValue::Scalar(json!(true)))?;
        let position =
            self.input_value_with_node(id, node, "Position", EvalValue::Field(Field::Position))?;
        let offset = self.input_value_with_node(
            id,
            node,
            "Offset",
            EvalValue::Scalar(json!([0.0, 0.0, 0.0])),
        )?;
        let selection = as_field(selection);
        let position = as_field(position);
        let offset = as_field(offset);
        for (index, vertex) in geometry.mesh.vertices.iter_mut().enumerate() {
            if !field_bool(&selection, &original, index, "POINT", false)? {
                continue;
            }
            let base = field_vec3(&position, &original, index, vertex.co)?;
            let delta = field_vec3(&offset, &original, index, DVec3::ZERO)?;
            vertex.co = base + delta;
        }
        let instance_count = geometry.instance_attributes.len();
        let attributes = &geometry.instance_attributes.attributes;
        let transforms = &mut geometry.instance_attributes.transforms;
        for (index, transform) in transforms.iter_mut().enumerate() {
            if !field_bool_for_instance(
                &selection,
                attributes,
                transform,
                index,
                instance_count,
                false,
            )? {
                continue;
            }
            let current = instance_position(transform)?;
            let base = field_vec3_for_instance(
                &position,
                attributes,
                transform,
                index,
                instance_count,
                current,
            )?;
            let delta = field_vec3_for_instance(
                &offset,
                attributes,
                transform,
                index,
                instance_count,
                DVec3::ZERO,
            )?;
            let new_position = base + delta;
            if !new_position.is_finite() {
                return Err(evaluation("set position instance location is not finite"));
            }
            transform[12..15].copy_from_slice(&new_position.to_array());
        }
        Ok(EvalValue::Geometry(geometry))
    }

    fn join_node(&mut self, id: &Id) -> Result<EvalValue> {
        let links: Vec<_> = self
            .group
            .links
            .iter()
            .filter(|link| &link.to_node == id && link.to_socket == "Geometry")
            .cloned()
            .collect();
        let mut geometries = Vec::with_capacity(links.len());
        let mut paths = Vec::new();
        let mut instances = EvaluatedInstances::default();
        for link in links {
            match self.output_value(&link.from_node, &link.from_socket)? {
                EvalValue::Geometry(mut geometry) => {
                    paths.append(&mut geometry.instance_paths);
                    instances.append(&geometry.instance_attributes)?;
                    geometries.push(geometry.mesh);
                }
                _ => return Err(evaluation("Join Geometry input is not geometry")),
            }
        }
        let mesh = merge_meshes(&geometries)?;
        Ok(EvalValue::Geometry(GeometryValue {
            mesh,
            instance_paths: paths,
            instance_attributes: instances,
        }))
    }

    fn instance_on_points(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let points_value = self.input_value_with_node(
            id,
            node,
            "Points",
            EvalValue::Geometry(GeometryValue {
                mesh: Mesh::new(),
                instance_paths: Vec::new(),
                instance_attributes: EvaluatedInstances::default(),
            }),
        )?;
        let source_value = self.input_value_with_node(
            id,
            node,
            "Instance",
            EvalValue::Geometry(GeometryValue {
                mesh: Mesh::new(),
                instance_paths: Vec::new(),
                instance_attributes: EvaluatedInstances::default(),
            }),
        )?;
        let points = into_geometry(points_value)?;
        let source = into_geometry(source_value)?;
        let has_realized_point_geometry = !points.instance_attributes.is_empty();
        let selection = as_field(self.input_value_with_node(
            id,
            node,
            "Selection",
            EvalValue::Scalar(json!(true)),
        )?);
        let pick_instance = as_field(self.input_value_with_node(
            id,
            node,
            "Pick Instance",
            EvalValue::Scalar(json!(false)),
        )?);
        let scale = as_field(self.input_value_with_node(
            id,
            node,
            "Scale",
            EvalValue::Scalar(json!([1.0, 1.0, 1.0])),
        )?);
        let rotation = self.rotation_field_input(id, node, "Rotation")?;
        let source_node = self
            .group
            .links
            .iter()
            .find(|link| &link.to_node == id && link.to_socket == "Instance")
            .map(|link| link.from_node.clone());
        let mut meshes = Vec::new();
        let mut paths = points.instance_paths;
        let mut instances = points.instance_attributes;
        let mut point_columns = None;
        for (index, point) in points.mesh.vertices.iter().enumerate() {
            if !field_bool(&selection, &points.mesh, index, "POINT", false)? {
                continue;
            }
            if has_realized_point_geometry {
                return Err(unsupported_instance_feature(
                    id,
                    "constraint.geometry_attribute.instances.realized_points",
                    "Instance on Points cannot distinguish realized instance geometry in its Points mesh",
                ));
            }
            if field_bool(&pick_instance, &points.mesh, index, "POINT", false)? {
                return Err(unsupported_instance_feature(
                    id,
                    "constraint.geometry_attribute.instances.picked_instance",
                    "Instance on Points Pick Instance is not implemented",
                ));
            }
            let local_scale = field_vec3(&scale, &points.mesh, index, DVec3::ONE)?;
            let local_rotation = field_quaternion(&rotation, &points.mesh, index, "POINT")?;
            let matrix =
                DMat4::from_scale_rotation_translation(local_scale, local_rotation, point.co);
            if self.capture_instance_attributes {
                if point_columns.is_none() {
                    point_columns = Some(point_domain_attribute_columns(&points.mesh, id)?);
                }
                let point_columns = point_columns
                    .as_ref()
                    .ok_or_else(|| evaluation("point attribute columns were not initialized"))?;
                instances.push_point_attributes(
                    matrix.to_cols_array(),
                    point_columns,
                    &points.mesh,
                    index,
                )?;
            } else {
                instances.push_transform(matrix.to_cols_array());
            }
            let mut instance_mesh = source.mesh.clone();
            for vertex in &mut instance_mesh.vertices {
                vertex.co = matrix.transform_point3(vertex.co);
            }
            meshes.push(instance_mesh);
            let mut path = vec![id.clone()];
            if let Some(source_node) = &source_node {
                path.push(source_node.clone());
            }
            paths.push(path);
        }
        Ok(EvalValue::Geometry(GeometryValue {
            mesh: merge_meshes(&meshes)?,
            instance_paths: paths,
            instance_attributes: instances,
        }))
    }

    fn distribute_points(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        if socket != "Points" {
            return Err(unsupported_graph_feature(
                id,
                "graph.node.distribute_points.outputs",
                format!("Distribute Points on Faces output `{socket}` is not implemented"),
            ));
        }
        let method = node
            .properties
            .get("distribute_method")
            .and_then(Value::as_str)
            .unwrap_or("RANDOM");
        if !matches!(method, "RANDOM" | "POISSON") {
            return Err(unsupported_graph_feature(
                id,
                "graph.node.distribute_points.method",
                format!("Distribute Points on Faces method `{method}` is not implemented"),
            ));
        }
        let surface_value = self.input_value_with_node(
            id,
            node,
            "Mesh",
            EvalValue::Geometry(GeometryValue {
                mesh: self.input.cloned().unwrap_or_default(),
                instance_paths: Vec::new(),
                instance_attributes: EvaluatedInstances::default(),
            }),
        )?;
        let GeometryValue {
            mesh: surface,
            instance_paths,
            instance_attributes,
        } = into_geometry(surface_value)?;
        let selection = as_field(self.input_value_with_node(
            id,
            node,
            "Selection",
            EvalValue::Scalar(json!(true)),
        )?);
        let density = if method == "RANDOM" {
            as_field(self.input_value_with_node(
                id,
                node,
                "Density",
                EvalValue::Scalar(json!(10.0)),
            )?)
        } else {
            Field::Constant(Value::Null)
        };
        let density_factor = if method == "POISSON" {
            as_field(self.input_value_with_node(
                id,
                node,
                "Density Factor",
                EvalValue::Scalar(json!(1.0)),
            )?)
        } else {
            Field::Constant(Value::Null)
        };
        let density_max = if method == "POISSON" {
            self.scalar_input(id, node, "Density Max", 10.0)? as f32
        } else {
            10.0
        };
        let minimum_distance = if method == "POISSON" {
            self.scalar_input(id, node, "Distance Min", 0.0)? as f32
        } else {
            0.0
        };
        let seed = self.integer_input(id, node, "Seed", 0)?;
        let mesh = distribute_points_mesh(
            id,
            &surface,
            method,
            &density,
            &density_factor,
            &selection,
            seed,
            density_max,
            minimum_distance,
        )?;
        Ok(EvalValue::Geometry(GeometryValue {
            mesh,
            instance_paths,
            instance_attributes,
        }))
    }

    fn mesh_to_points(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let geometry = self.geometry_input(id, node, "Mesh")?;
        let geometry = into_geometry(geometry)?;
        let mode = node
            .properties
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("VERTICES");
        let domain = match mode {
            "VERTICES" => "POINT",
            "EDGES" => "EDGE",
            "FACES" => "FACE",
            "CORNERS" => "CORNER",
            _ => {
                return Err(unsupported_graph_feature(
                    id,
                    "graph.node.mesh_to_points.mode",
                    format!("Mesh to Points mode `{mode}` is not implemented"),
                ));
            }
        };
        let count = domain_element_count(&geometry.mesh, domain)?;
        let selection = as_field(self.input_value_with_node(
            id,
            node,
            "Selection",
            EvalValue::Scalar(json!(true)),
        )?);
        let position = as_field(self.input_value_with_node(
            id,
            node,
            "Position",
            EvalValue::Field(Field::Position),
        )?);
        let radius = as_field(self.input_value_with_node(
            id,
            node,
            "Radius",
            EvalValue::Scalar(json!(0.05)),
        )?);
        let mut points = Mesh::new();
        let source_indices = origindex_values(&geometry.mesh);
        let mut source_vertices = source_indices.as_ref().map(|_| Vec::new());
        let corner_vertices = if domain == "CORNER" && source_vertices.is_some() {
            let vertex_index: HashMap<u32, usize> = geometry
                .mesh
                .vertices
                .iter()
                .enumerate()
                .map(|(index, vertex)| (vertex.id, index))
                .collect();
            Some(
                geometry
                    .mesh
                    .faces
                    .iter()
                    .flat_map(|face| face.vertices.iter())
                    .map(|id| vertex_index.get(id).copied())
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let mut radii = Vec::new();
        for index in 0..count {
            if field_bool(&selection, &geometry.mesh, index, domain, false)? {
                let fallback = domain_position(&geometry.mesh, index, domain)?;
                let position =
                    field_vec3_in_domain(&position, &geometry.mesh, index, domain, fallback)?;
                let radius = field_json(&radius, &geometry.mesh, index, domain)?
                    .as_f64()
                    .filter(|radius| radius.is_finite() && *radius >= 0.0)
                    .ok_or_else(|| evaluation("point radius must be finite and non-negative"))?;
                points
                    .insert_vertex(position)
                    .map_err(|error| mesh_evaluation_error(&error))?;
                radii.push(json!(radius));
                if let Some(source_vertices) = &mut source_vertices {
                    let source_vertex = match domain {
                        "POINT" | "POINTS" => Some(index),
                        "CORNER" => corner_vertices
                            .as_ref()
                            .and_then(|vertices| vertices.get(index).copied().flatten()),
                        _ => None,
                    };
                    let original_index = source_vertex
                        .and_then(|source| source_indices.as_ref()?.get(source).copied())
                        .unwrap_or(ORIGINDEX_NONE);
                    source_vertices.push(original_index);
                }
            }
        }
        points
            .attributes
            .insert("radius".to_owned(), json!({"domain":"POINT","data":radii}));
        if let Some(vertices) = source_vertices {
            set_origindex(&mut points, &vertices);
        }
        Ok(EvalValue::Geometry(GeometryValue {
            mesh: points,
            instance_paths: geometry.instance_paths,
            instance_attributes: geometry.instance_attributes,
        }))
    }

    fn subdivision(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let geometry = self.geometry_input(id, node, "Mesh")?;
        let mut geometry = into_geometry(geometry)?;
        let level = self.integer_input(id, node, "Level", 1)?;
        if !(0..=6).contains(&level) {
            return Err(evaluation("subdivision level must be between 0 and 6"));
        }
        if level == 0 {
            return Ok(EvalValue::Geometry(geometry));
        }
        let edge_crease = as_field(self.input_value_with_node(
            id,
            node,
            "Edge Crease",
            EvalValue::Scalar(json!(0.0)),
        )?);
        let vertex_crease = as_field(self.input_value_with_node(
            id,
            node,
            "Vertex Crease",
            EvalValue::Scalar(json!(0.0)),
        )?);
        let limit_surface = match self.input_value_with_node(
            id,
            node,
            "Limit Surface",
            EvalValue::Scalar(json!(true)),
        )? {
            EvalValue::Scalar(value) => value
                .as_bool()
                .ok_or_else(|| evaluation("Limit Surface must be boolean"))?,
            _ => return Err(evaluation("Limit Surface must be a constant boolean")),
        };
        let quality = self.integer_input(id, node, "Quality", 3)?;
        if !(1..=10).contains(&quality) {
            return Err(evaluation("subdivision quality must be between 1 and 10"));
        }
        let uv_smooth = subdivision_uv_smooth(&self.string_input(
            id,
            node,
            "UV Smooth",
            "PRESERVE_BOUNDARIES",
        )?)?;
        let boundary_smooth =
            subdivision_boundary_smooth(&self.string_input(id, node, "Boundary Smooth", "ALL")?)?;

        let constant_crease_is_zero = |field: &Field| {
            matches!(
                field,
                Field::Constant(value)
                    if value
                        .as_f64()
                        .is_some_and(|value| (value as f32).clamp(0.0, 1.0) == 0.0)
            )
        };
        let use_creases =
            !(constant_crease_is_zero(&edge_crease) && constant_crease_is_zero(&vertex_crease));
        if use_creases {
            let mut edge_values = Map::new();
            for (index, edge) in geometry.mesh.edges.iter().enumerate() {
                let value = field_json(&edge_crease, &geometry.mesh, index, "EDGE")?
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| evaluation("Edge Crease must be finite and numeric"))?
                    as f32;
                if !value.is_finite() {
                    return Err(evaluation("Edge Crease must be finite"));
                }
                let value = value.clamp(0.0, 1.0);
                if value > 0.0 {
                    edge_values.insert(format!("e{}", edge.id), json!(value));
                }
            }
            let mut vertex_values = Map::new();
            for (index, vertex) in geometry.mesh.vertices.iter().enumerate() {
                let value = field_json(&vertex_crease, &geometry.mesh, index, "POINT")?
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| evaluation("Vertex Crease must be finite and numeric"))?
                    as f32;
                if !value.is_finite() {
                    return Err(evaluation("Vertex Crease must be finite"));
                }
                let value = value.clamp(0.0, 1.0);
                if value > 0.0 {
                    vertex_values.insert(format!("v{}", vertex.id), json!(value));
                }
            }
            geometry.mesh.attributes.remove("crease_edge");
            geometry.mesh.attributes.remove("user:crease_edge");
            geometry.mesh.attributes.remove("crease_vert");
            if !edge_values.is_empty() {
                geometry.mesh.attributes.insert(
                    "crease_edge".to_owned(),
                    json!({"domain":"EDGE","type":"FLOAT","values":edge_values}),
                );
            }
            if !vertex_values.is_empty() {
                geometry.mesh.attributes.insert(
                    "crease_vert".to_owned(),
                    json!({"domain":"POINT","type":"FLOAT","values":vertex_values}),
                );
            }
        }
        let modifier = Modifier {
            id: Id::new("graph_subdivision")
                .map_err(|error| PotError::new(ErrorCode::InternalError, error.message))?,
            modifier_type: "subdivision".to_owned(),
            name: "Graph Subdivision Surface".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: Map::from_iter([
                ("levels".to_owned(), json!(level)),
                ("quality".to_owned(), json!(quality)),
                ("use_creases".to_owned(), json!(use_creases)),
                ("use_limit_surface".to_owned(), json!(limit_surface)),
                ("uv_smooth".to_owned(), json!(uv_smooth)),
                ("boundary_smooth".to_owned(), json!(boundary_smooth)),
            ]),
            binding_data: None,
            baked_ocean_cache: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        let mut mesh = evaluate_modifiers(&geometry.mesh, &[modifier], self.subject_name)?;
        if use_creases {
            mesh.attributes.remove("crease_edge");
            mesh.attributes.remove("user:crease_edge");
            mesh.attributes.remove("crease_vert");
        }
        Ok(EvalValue::Geometry(GeometryValue {
            mesh,
            instance_paths: geometry.instance_paths,
            instance_attributes: geometry.instance_attributes,
        }))
    }

    fn store_named_attribute(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let geometry_value = self.geometry_input(id, node, "Geometry")?;
        let mut geometry = into_geometry(geometry_value)?;
        let name = self.string_input(id, node, "Name", "")?;
        if name.is_empty() {
            return Ok(EvalValue::Geometry(geometry));
        }
        let data_type =
            named_attribute_data_type(id, node, "graph.node.store_named_attribute.data_type")?;
        let domain = node
            .properties
            .get("domain")
            .and_then(Value::as_str)
            .unwrap_or("POINT");
        let domain = if domain == "POINTS" { "POINT" } else { domain };
        if domain == "INSTANCE" {
            if !self.capture_instance_attributes {
                return Err(unsupported_instance_feature(
                    id,
                    "graph.instance_component.store_named_attribute_uncaptured",
                    "Store Named Attribute on the INSTANCE domain requires instance attribute capture",
                ));
            }
            return self.store_instance_named_attribute(id, node, geometry, &name, &data_type);
        }
        let count = domain_element_count(&geometry.mesh, domain)?;
        let selection = as_field(self.input_value_with_node(
            id,
            node,
            "Selection",
            EvalValue::Scalar(json!(true)),
        )?);
        let value = match self.input_value_with_node(
            id,
            node,
            "Value",
            EvalValue::Scalar(named_attribute_input_default(&data_type)),
        )? {
            EvalValue::Scalar(Value::Null) => {
                Field::Constant(named_attribute_input_default(&data_type))
            }
            value => as_field(value),
        };
        let previous = if let Some(attribute) = geometry.mesh.attributes.get(&name) {
            if attribute.get("domain").and_then(Value::as_str) != Some(domain) {
                return Err(evaluation(
                    "stored attribute domain differs from its existing domain",
                ));
            }
            if attribute
                .get("data_type")
                .and_then(Value::as_str)
                .is_some_and(|previous_type| previous_type != data_type)
            {
                return Err(evaluation(
                    "stored attribute data type differs from its existing type",
                ));
            }
            let data = attribute
                .get("data")
                .and_then(Value::as_array)
                .ok_or_else(|| evaluation("stored attribute payload is malformed"))?;
            if data.len() != count {
                return Err(evaluation(
                    "stored attribute length does not match its domain",
                ));
            }
            Some(data)
        } else {
            None
        };
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            if field_bool(&selection, &geometry.mesh, index, domain, false)? {
                values.push(named_attribute_value(
                    &field_json(&value, &geometry.mesh, index, domain)?,
                    &data_type,
                )?);
            } else {
                values.push(
                    previous
                        .and_then(|data| data.get(index))
                        .cloned()
                        .unwrap_or_else(|| named_attribute_zero(&data_type)),
                );
            }
        }
        geometry.mesh.attributes.insert(
            name,
            json!({"domain":domain,"data_type":data_type,"data":values}),
        );
        Ok(EvalValue::Geometry(geometry))
    }

    fn store_instance_named_attribute(
        &mut self,
        id: &Id,
        node: &GraphNode,
        mut geometry: GeometryValue,
        name: &str,
        data_type: &str,
    ) -> Result<EvalValue> {
        if !geometry.instance_paths.is_empty() && geometry.instance_attributes.transforms.is_empty()
        {
            return Err(legacy_instance_columns_missing(id));
        }
        let selection = as_field(self.input_value_with_node(
            id,
            node,
            "Selection",
            EvalValue::Scalar(json!(true)),
        )?);
        let value = as_field(self.input_value_with_node(
            id,
            node,
            "Value",
            EvalValue::Scalar(named_attribute_input_default(data_type)),
        )?);
        let previous = geometry.instance_attributes.attributes.get(name);
        if previous.is_some_and(|column| column.values.len() != geometry.instance_attributes.len())
        {
            return Err(evaluation(
                "stored instance attribute column length is invalid",
            ));
        }
        if previous
            .is_some_and(|column| normalize_instance_data_type(&column.data_type) != data_type)
        {
            return Err(evaluation(
                "stored instance attribute data type differs from its existing type",
            ));
        }

        let mut values = Vec::with_capacity(geometry.instance_attributes.len());
        for index in 0..geometry.instance_attributes.len() {
            if field_bool_in_instances(&selection, &geometry.instance_attributes, index, false)? {
                values.push(named_attribute_value(
                    &field_json_instances(&value, &geometry.instance_attributes, index)?,
                    data_type,
                )?);
            } else {
                values.push(
                    previous
                        .and_then(|column| column.values.get(index))
                        .cloned()
                        .unwrap_or_else(|| named_attribute_zero(data_type)),
                );
            }
        }
        geometry.instance_attributes.attributes.insert(
            name.to_owned(),
            InstanceAttributeColumn {
                data_type: data_type.to_owned(),
                values,
            },
        );
        Ok(EvalValue::Geometry(geometry))
    }

    fn input_named_attribute(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
    ) -> Result<EvalValue> {
        let name = self.string_input(id, node, "Name", "")?;
        if socket == "Exists" {
            return Ok(EvalValue::Field(Field::AttributeExists(name)));
        }
        let data_type =
            named_attribute_data_type(id, node, "graph.node.input_named_attribute.data_type")?;
        Ok(EvalValue::Field(Field::Attribute {
            name,
            data_type: Some(data_type),
        }))
    }

    fn capture_attribute(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        if socket != "Attribute" {
            return self.geometry_input(id, node, "Geometry");
        }
        if node
            .properties
            .get("capture_items")
            .and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
        {
            return Err(unsupported_graph_feature(
                id,
                "graph.node.capture_attribute.dynamic_items",
                "Capture Attribute nodes with dynamic capture items are not implemented",
            ));
        }
        let data_type =
            named_attribute_data_type(id, node, "graph.node.capture_attribute.data_type")?;
        let geometry = into_geometry(self.geometry_input(id, node, "Geometry")?)?;
        let domain = node
            .properties
            .get("domain")
            .and_then(Value::as_str)
            .unwrap_or("POINT");
        let domain = if domain == "POINTS" { "POINT" } else { domain };
        let count = domain_element_count(&geometry.mesh, domain)?;
        let selection = as_field(self.input_value_with_node(
            id,
            node,
            "Selection",
            EvalValue::Scalar(json!(true)),
        )?);
        let value = match self.input_value_with_node(
            id,
            node,
            "Value",
            EvalValue::Scalar(named_attribute_input_default(&data_type)),
        )? {
            EvalValue::Scalar(Value::Null) => {
                Field::Constant(named_attribute_input_default(&data_type))
            }
            value => as_field(value),
        };
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            if field_bool(&selection, &geometry.mesh, index, domain, false)? {
                values.push(named_attribute_value(
                    &field_json(&value, &geometry.mesh, index, domain)?,
                    &data_type,
                )?);
            } else {
                values.push(named_attribute_zero(&data_type));
            }
        }
        Ok(EvalValue::Field(Field::Captured {
            domain: domain.to_owned(),
            values,
        }))
    }

    fn math_node(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let operation = node
            .properties
            .get("operation")
            .and_then(Value::as_str)
            .unwrap_or("ADD")
            .to_ascii_uppercase();
        let first = as_field(self.input_value_with_node(
            id,
            node,
            "Value",
            EvalValue::Scalar(json!(0.5)),
        )?);
        let second = as_field(self.input_value_with_node(
            id,
            node,
            "Value_001",
            EvalValue::Scalar(json!(0.5)),
        )?);
        let third = as_field(self.input_value_with_node(
            id,
            node,
            "Value_002",
            EvalValue::Scalar(json!(0.5)),
        )?);
        Ok(EvalValue::Field(Field::Math {
            operation,
            clamp: node
                .properties
                .get("use_clamp")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            first: Box::new(first),
            second: Box::new(second),
            third: Box::new(third),
        }))
    }
    fn texture_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        let mut inputs = Vec::with_capacity(10);
        let kind = match node.node_type.as_str() {
            "ShaderNodeTexWave" => {
                if !matches!(socket, "Color" | "Fac") {
                    return Err(PotError::new(
                        ErrorCode::InvalidOperation,
                        format!("unknown Wave Texture output `{socket}`"),
                    ));
                }
                let wave_type = match texture_string_property(node, "wave_type", "BANDS")? {
                    "BANDS" => WaveType::Bands,
                    "RINGS" => WaveType::Rings,
                    value => {
                        return Err(unsupported_texture_mode(
                            id,
                            "ShaderNodeTexWave",
                            "wave_type",
                            value,
                        ));
                    }
                };
                let bands_direction = if matches!(wave_type, WaveType::Bands) {
                    match texture_string_property(node, "bands_direction", "X")? {
                        "X" => WaveBandsDirection::X,
                        "Y" => WaveBandsDirection::Y,
                        "Z" => WaveBandsDirection::Z,
                        "DIAGONAL" => WaveBandsDirection::Diagonal,
                        value => {
                            return Err(unsupported_texture_mode(
                                id,
                                "ShaderNodeTexWave",
                                "bands_direction",
                                value,
                            ));
                        }
                    }
                } else {
                    WaveBandsDirection::X
                };
                let rings_direction = if matches!(wave_type, WaveType::Rings) {
                    match texture_string_property(node, "rings_direction", "X")? {
                        "X" => WaveRingsDirection::X,
                        "Y" => WaveRingsDirection::Y,
                        "Z" => WaveRingsDirection::Z,
                        "SPHERICAL" => WaveRingsDirection::Spherical,
                        value => {
                            return Err(unsupported_texture_mode(
                                id,
                                "ShaderNodeTexWave",
                                "rings_direction",
                                value,
                            ));
                        }
                    }
                } else {
                    WaveRingsDirection::X
                };
                let profile = match texture_string_property(node, "wave_profile", "SIN")? {
                    "SIN" => WaveProfile::Sine,
                    "SAW" => WaveProfile::Saw,
                    "TRI" => WaveProfile::Triangle,
                    value => {
                        return Err(unsupported_texture_mode(
                            id,
                            "ShaderNodeTexWave",
                            "wave_profile",
                            value,
                        ));
                    }
                };
                inputs.push(self.texture_vector_input(id, node)?);
                inputs.push(self.texture_input(id, node, "Scale", json!(5.0))?);
                inputs.push(self.texture_input(id, node, "Distortion", json!(0.0))?);
                inputs.push(self.texture_input(id, node, "Detail", json!(2.0))?);
                inputs.push(self.texture_input(id, node, "Detail Scale", json!(1.0))?);
                inputs.push(self.texture_input(id, node, "Detail Roughness", json!(0.5))?);
                inputs.push(self.texture_input(id, node, "Phase Offset", json!(0.0))?);
                TextureFieldKind::Wave {
                    wave_type,
                    bands_direction,
                    rings_direction,
                    profile,
                }
            }
            "ShaderNodeTexMagic" => {
                if !matches!(socket, "Color" | "Fac") {
                    return Err(PotError::new(
                        ErrorCode::InvalidOperation,
                        format!("unknown Magic Texture output `{socket}`"),
                    ));
                }
                let depth = texture_integer_property(node, "turbulence_depth", 2)?;
                inputs.push(self.texture_vector_input(id, node)?);
                inputs.push(self.texture_input(id, node, "Scale", json!(5.0))?);
                inputs.push(self.texture_input(id, node, "Distortion", json!(1.0))?);
                TextureFieldKind::Magic {
                    depth: depth.clamp(0, 10) as u8,
                }
            }
            "ShaderNodeTexBrick" => {
                if !matches!(socket, "Color" | "Fac") {
                    return Err(PotError::new(
                        ErrorCode::InvalidOperation,
                        format!("unknown Brick Texture output `{socket}`"),
                    ));
                }
                let offset = texture_float_property(node, "offset", 0.5)?;
                let offset_frequency = texture_integer_property(node, "offset_frequency", 2)?;
                let squash = texture_float_property(node, "squash", 1.0)?;
                let squash_frequency = texture_integer_property(node, "squash_frequency", 2)?;
                inputs.push(self.texture_vector_input(id, node)?);
                inputs.push(self.texture_input(id, node, "Color1", json!([0.8, 0.8, 0.8, 1.0]))?);
                inputs.push(self.texture_input(id, node, "Color2", json!([0.2, 0.2, 0.2, 1.0]))?);
                inputs.push(self.texture_input(id, node, "Mortar", json!([0.0, 0.0, 0.0, 1.0]))?);
                inputs.push(self.texture_input(id, node, "Scale", json!(5.0))?);
                inputs.push(self.texture_input(id, node, "Mortar Size", json!(0.02))?);
                inputs.push(self.texture_input(id, node, "Mortar Smooth", json!(0.1))?);
                inputs.push(self.texture_input(id, node, "Bias", json!(0.0))?);
                inputs.push(self.texture_input(id, node, "Brick Width", json!(0.5))?);
                inputs.push(self.texture_input(id, node, "Row Height", json!(0.25))?);
                TextureFieldKind::Brick {
                    offset,
                    offset_frequency,
                    squash,
                    squash_frequency,
                }
            }
            "ShaderNodeTexWhiteNoise" => {
                if !matches!(socket, "Value" | "Color") {
                    return Err(PotError::new(
                        ErrorCode::InvalidOperation,
                        format!("unknown White Noise Texture output `{socket}`"),
                    ));
                }
                let dimensions = match texture_string_property(node, "noise_dimensions", "3D")? {
                    "1D" => 1,
                    "2D" => 2,
                    "3D" => 3,
                    "4D" => 4,
                    value => {
                        return Err(unsupported_texture_mode(
                            id,
                            "ShaderNodeTexWhiteNoise",
                            "noise_dimensions",
                            value,
                        ));
                    }
                };
                match dimensions {
                    1 => inputs.push(self.texture_input(id, node, "W", json!(0.0))?),
                    2 | 3 => inputs.push(self.texture_vector_input(id, node)?),
                    4 => {
                        inputs.push(self.texture_vector_input(id, node)?);
                        inputs.push(self.texture_input(id, node, "W", json!(0.0))?);
                    }
                    _ => unreachable!(),
                }
                TextureFieldKind::WhiteNoise { dimensions }
            }
            "ShaderNodeTexGabor" => {
                if !matches!(socket, "Value" | "Phase" | "Intensity") {
                    return Err(PotError::new(
                        ErrorCode::InvalidOperation,
                        format!("unknown Gabor Texture output `{socket}`"),
                    ));
                }
                let dimensions = match texture_string_property(node, "gabor_type", "2D")? {
                    "2D" => 2,
                    "3D" => 3,
                    value => {
                        return Err(unsupported_texture_mode(
                            id,
                            "ShaderNodeTexGabor",
                            "gabor_type",
                            value,
                        ));
                    }
                };
                inputs.push(self.texture_vector_input(id, node)?);
                inputs.push(self.texture_input(id, node, "Scale", json!(5.0))?);
                inputs.push(self.texture_input(id, node, "Frequency", json!(2.0))?);
                inputs.push(self.texture_input(id, node, "Anisotropy", json!(1.0))?);
                if dimensions == 2 {
                    inputs.push(self.texture_input(
                        id,
                        node,
                        "Orientation 2D",
                        json!(std::f64::consts::FRAC_PI_4),
                    )?);
                } else {
                    inputs.push(self.texture_input(
                        id,
                        node,
                        "Orientation 3D",
                        json!([std::f64::consts::SQRT_2, std::f64::consts::SQRT_2, 0.0]),
                    )?);
                }
                TextureFieldKind::Gabor { dimensions }
            }
            _ => return Err(evaluation("unknown procedural texture field node")),
        };
        Ok(EvalValue::Field(Field::ProceduralTexture {
            kind,
            inputs,
            output: socket.to_owned(),
        }))
    }

    fn texture_input(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
        default: Value,
    ) -> Result<Field> {
        Ok(as_field(self.input_value_with_node(
            id,
            node,
            socket,
            EvalValue::Scalar(default),
        )?))
    }

    fn texture_vector_input(&mut self, id: &Id, node: &GraphNode) -> Result<Field> {
        if self
            .group
            .links
            .iter()
            .any(|link| link.to_node == *id && link.to_socket == "Vector")
        {
            self.texture_input(id, node, "Vector", json!([0.0, 0.0, 0.0]))
        } else {
            Ok(Field::Position)
        }
    }

    fn vector_math_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        let operation = node
            .properties
            .get("operation")
            .and_then(Value::as_str)
            .unwrap_or("ADD")
            .to_ascii_uppercase();
        let second_default = [0.0; 3];
        let first = as_field(self.input_value_with_node(
            id,
            node,
            "Vector",
            EvalValue::Scalar(json!([0.0, 0.0, 0.0])),
        )?);
        let second = as_field(self.input_value_with_node(
            id,
            node,
            "Vector_001",
            EvalValue::Scalar(json!(second_default)),
        )?);
        let third = as_field(self.input_value_with_node(
            id,
            node,
            "Vector_002",
            EvalValue::Scalar(json!([0.0, 0.0, 0.0])),
        )?);
        let scale = as_field(self.input_value_with_node(
            id,
            node,
            "Scale",
            EvalValue::Scalar(json!(1.0)),
        )?);
        if socket != "Vector" && socket != "Value" {
            return Err(evaluation("unknown Vector Math output"));
        }
        Ok(EvalValue::Field(Field::VectorMath {
            operation,
            first: Box::new(first),
            second: Box::new(second),
            third: Box::new(third),
            scale: Box::new(scale),
            output: socket.to_owned(),
        }))
    }
    fn field_input(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
        fallback: Value,
    ) -> Result<Field> {
        Ok(as_field(self.input_value_with_node(
            id,
            node,
            socket,
            EvalValue::Scalar(fallback),
        )?))
    }

    fn integer_math_node(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let operation = node_enum(node, "operation", "ADD");
        Ok(EvalValue::Field(Field::IntegerMath {
            operation,
            first: Box::new(self.field_input(id, node, "Value", json!(0))?),
            second: Box::new(self.field_input(id, node, "Value_001", json!(0))?),
            third: Box::new(self.field_input(id, node, "Value_002", json!(0))?),
        }))
    }

    fn boolean_math_node(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        Ok(EvalValue::Field(Field::BooleanMath {
            operation: node_enum(node, "operation", "AND"),
            first: Box::new(self.field_input(id, node, "Boolean", json!(false))?),
            second: Box::new(self.field_input(id, node, "Boolean_001", json!(false))?),
        }))
    }

    fn compare_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        if socket != "Result" {
            return Err(evaluation("unknown Compare output"));
        }
        let mut data_type = node_enum(node, "data_type", "FLOAT");
        if data_type == "INTEGER" {
            "INT".clone_into(&mut data_type);
        } else if data_type == "COLOR" {
            "RGBA".clone_into(&mut data_type);
        } else if data_type == "FLOAT_VECTOR" {
            "VECTOR".clone_into(&mut data_type);
        }
        let (first_default, second_default) = match data_type.as_str() {
            "FLOAT" => (json!(0.0), json!(0.0)),
            "INT" => (json!(0), json!(0)),
            "VECTOR" => (json!([0.0, 0.0, 0.0]), json!([0.0, 0.0, 0.0])),
            "RGBA" => (json!([0.0, 0.0, 0.0, 1.0]), json!([0.0, 0.0, 0.0, 1.0])),
            "BOOLEAN" | "BOOL" => (json!(false), json!(false)),
            "STRING" => (json!(""), json!("")),
            "OBJECT" | "IMAGE" | "COLLECTION" | "MATERIAL" | "FONT" | "SOUND" => {
                (Value::Null, Value::Null)
            }
            _ => {
                return Err(unsupported_graph_node_state(
                    node,
                    &format!("data type `{data_type}`"),
                ));
            }
        };
        let operation = node_enum(node, "operation", "GREATER_THAN");
        let mode = node_enum(node, "mode", "ELEMENT");
        let parameter_socket = match mode.as_str() {
            "DOT_PRODUCT" => Some(("C", 0.9)),
            "DIRECTION" => Some(("Angle", 0.087_266_5)),
            _ => None,
        };
        let epsilon_default = if matches!(operation.as_str(), "EQUAL" | "NOT_EQUAL")
            && matches!(data_type.as_str(), "FLOAT" | "VECTOR" | "RGBA")
        {
            0.001
        } else {
            0.0
        };
        Ok(EvalValue::Field(Field::Compare {
            operation,
            data_type,
            mode,
            first: Box::new(self.field_input(id, node, "A", first_default)?),
            second: Box::new(self.field_input(id, node, "B", second_default)?),
            parameter: Box::new(match parameter_socket {
                Some((name, default)) => self.field_input(id, node, name, json!(default))?,
                None => Field::Constant(json!(0.0)),
            }),
            epsilon: Box::new(self.field_input(id, node, "Epsilon", json!(epsilon_default))?),
        }))
    }

    fn map_range_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        let data_type = node_enum(node, "data_type", "FLOAT");
        let vector = data_type == "FLOAT_VECTOR";
        let input_socket = if vector { "Vector" } else { "Value" };
        let output_socket = if vector { "Vector" } else { "Result" };
        if !matches!(data_type.as_str(), "FLOAT" | "FLOAT_VECTOR") {
            return Err(unsupported_graph_node_state(
                node,
                &format!("data type `{data_type}`"),
            ));
        }
        if socket != output_socket {
            return Err(evaluation("unknown Map Range output for its data type"));
        }
        let (zero, one, four) = if vector {
            (
                json!([0.0, 0.0, 0.0]),
                json!([1.0, 1.0, 1.0]),
                json!([4.0, 4.0, 4.0]),
            )
        } else {
            (json!(0.0), json!(1.0), json!(4.0))
        };
        let (from_min_socket, from_max_socket, to_min_socket, to_max_socket, steps_socket) =
            if vector {
                (
                    "From_Min_FLOAT3",
                    "From_Max_FLOAT3",
                    "To_Min_FLOAT3",
                    "To_Max_FLOAT3",
                    "Steps_FLOAT3",
                )
            } else {
                ("From Min", "From Max", "To Min", "To Max", "Steps")
            };
        Ok(EvalValue::Field(Field::MapRange {
            data_type,
            interpolation: node_enum(node, "interpolation_type", "LINEAR"),
            clamp: node
                .properties
                .get("clamp")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            value: Box::new(self.field_input(
                id,
                node,
                input_socket,
                if vector { zero.clone() } else { json!(1.0) },
            )?),
            from_min: Box::new(self.field_input(id, node, from_min_socket, zero.clone())?),
            from_max: Box::new(self.field_input(id, node, from_max_socket, one.clone())?),
            to_min: Box::new(self.field_input(id, node, to_min_socket, zero)?),
            to_max: Box::new(self.field_input(id, node, to_max_socket, one)?),
            steps: Box::new(self.field_input(id, node, steps_socket, four)?),
        }))
    }

    fn clamp_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        if socket != "Result" {
            return Err(evaluation("unknown Clamp output"));
        }
        Ok(EvalValue::Field(Field::Clamp {
            mode: node_enum(node, "clamp_type", "MINMAX"),
            value: Box::new(self.field_input(id, node, "Value", json!(1.0))?),
            min: Box::new(self.field_input(id, node, "Min", json!(0.0))?),
            max: Box::new(self.field_input(id, node, "Max", json!(1.0))?),
        }))
    }

    fn mix_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        let data_type = node_enum(node, "data_type", "FLOAT");
        let factor_mode = node_enum(node, "factor_mode", "UNIFORM");
        let (factor, first, second, output, value_default) = match data_type.as_str() {
            "FLOAT" => (
                "Factor_Float",
                "A_Float",
                "B_Float",
                "Result_Float",
                json!(0.0),
            ),
            "VECTOR" => {
                let factor = if factor_mode == "NON_UNIFORM" {
                    "Factor_Vector"
                } else {
                    "Factor_Float"
                };
                (
                    factor,
                    "A_Vector",
                    "B_Vector",
                    "Result_Vector",
                    json!([0.0, 0.0, 0.0]),
                )
            }
            "RGBA" => (
                "Factor_Float",
                "A_Color",
                "B_Color",
                "Result_Color",
                json!([0.5, 0.5, 0.5, 1.0]),
            ),
            "ROTATION" => (
                "Factor_Float",
                "A_Rotation",
                "B_Rotation",
                "Result_Rotation",
                json!([0.0, 0.0, 0.0, 1.0]),
            ),
            _ => {
                return Err(unsupported_graph_node_state(
                    node,
                    &format!("data type `{data_type}`"),
                ));
            }
        };
        if socket != output {
            return Err(evaluation("unknown Mix output for its data type"));
        }
        if !matches!(factor_mode.as_str(), "UNIFORM" | "NON_UNIFORM") {
            return Err(unsupported_graph_node_state(
                node,
                &format!("factor mode `{factor_mode}`"),
            ));
        }
        let factor_default = if factor == "Factor_Vector" {
            json!([0.5, 0.5, 0.5])
        } else {
            json!(1.0)
        };
        Ok(EvalValue::Field(Field::Mix {
            node_type: "ShaderNodeMix",
            output: MixOutput::Value,
            data_type,
            factor_mode,
            blend_mode: node_enum(node, "blend_type", "MIX"),
            clamp_factor: node
                .properties
                .get("clamp_factor")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            clamp_result: node
                .properties
                .get("clamp_result")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            factor: Box::new(self.field_input(id, node, factor, factor_default)?),
            first: Box::new(self.field_input(id, node, first, value_default.clone())?),
            second: Box::new(self.field_input(id, node, second, value_default)?),
        }))
    }

    fn mix_rgb_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        let output = match socket {
            "Color" => MixOutput::Color,
            _ => return Err(evaluation("unknown Mix RGB output")),
        };
        Ok(EvalValue::Field(Field::Mix {
            node_type: "ShaderNodeMixRGB",
            output,
            data_type: "RGBA".to_owned(),
            factor_mode: "UNIFORM".to_owned(),
            blend_mode: node_enum(node, "blend_type", "MIX"),
            clamp_factor: true,
            clamp_result: node
                .properties
                .get("use_clamp")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            factor: Box::new(self.field_input(id, node, "Fac", json!(0.5))?),
            first: Box::new(self.field_input(id, node, "Color1", json!([0.5, 0.5, 0.5, 1.0]))?),
            second: Box::new(self.field_input(id, node, "Color2", json!([0.5, 0.5, 0.5, 1.0]))?),
        }))
    }

    fn color_ramp_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        if !matches!(socket, "Color" | "Alpha") {
            return Err(evaluation("unknown Color Ramp output"));
        }
        let ramp = node
            .properties
            .get("color_ramp")
            .and_then(Value::as_object)
            .ok_or_else(|| unsupported_graph_node_state(node, "missing Color Ramp state"))?;
        let elements = ramp
            .get("elements")
            .and_then(Value::as_array)
            .filter(|elements| elements.len() >= 2)
            .ok_or_else(|| evaluation("Color Ramp requires at least two elements"))?;
        let elements = elements
            .iter()
            .map(|element| {
                let position = element
                    .get("position")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| evaluation("Color Ramp element position is invalid"))?;
                let color = element
                    .get("color")
                    .and_then(json_color4)
                    .ok_or_else(|| evaluation("Color Ramp element color is invalid"))?;
                Ok((finite_f32(position, "Color Ramp position")?, color))
            })
            .collect::<Result<Vec<_>>>()?;
        let property = |name: &str, default: &str| {
            ramp.get(name)
                .and_then(Value::as_str)
                .unwrap_or(default)
                .to_ascii_uppercase()
        };
        Ok(EvalValue::Field(Field::ColorRamp {
            elements: Arc::from(elements),
            interpolation: property("interpolation", "LINEAR"),
            color_mode: property("color_mode", "RGB"),
            hue_interpolation: property("hue_interpolation", "NEAR"),
            factor: Box::new(self.field_input(id, node, "Fac", json!(0.0))?),
            output: socket.to_owned(),
        }))
    }

    fn curve_node(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
        kind: &str,
    ) -> Result<EvalValue> {
        let mapping_value = node
            .properties
            .get("mapping")
            .ok_or_else(|| unsupported_graph_node_state(node, "missing Curve Mapping state"))?;
        let mapping = Arc::new(
            crate::node_kernels::curve_mapping_from_value(mapping_value)
                .ok_or_else(|| unsupported_graph_node_state(node, "invalid Curve Mapping state"))?,
        );
        let prepared = Arc::new(crate::node_kernels::curve_mapping_prepare(&mapping));
        let (factor_socket, input_socket, output_socket, factor_default, input_default) = match kind
        {
            "FLOAT" => ("Factor", "Value", "Value", json!(1.0), json!(1.0)),
            "VECTOR" => (
                "Fac",
                "Vector",
                "Vector",
                json!(1.0),
                json!([0.0, 0.0, 0.0]),
            ),
            "RGBA" => (
                "Fac",
                "Color",
                "Color",
                json!(1.0),
                json!([1.0, 1.0, 1.0, 1.0]),
            ),
            _ => unreachable!(),
        };
        if socket != output_socket {
            return Err(evaluation("unknown Curve output"));
        }
        Ok(EvalValue::Field(Field::Curve {
            kind: kind.to_owned(),
            mapping,
            prepared,
            factor: Box::new(self.field_input(id, node, factor_socket, factor_default)?),
            value: Box::new(self.field_input(id, node, input_socket, input_default)?),
        }))
    }

    fn combine_xyz(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let x =
            as_field(self.input_value_with_node(id, node, "X", EvalValue::Scalar(json!(0.0)))?);
        let y =
            as_field(self.input_value_with_node(id, node, "Y", EvalValue::Scalar(json!(0.0)))?);
        let z =
            as_field(self.input_value_with_node(id, node, "Z", EvalValue::Scalar(json!(0.0)))?);
        Ok(EvalValue::Field(Field::CombineXYZ {
            x: Box::new(x),
            y: Box::new(y),
            z: Box::new(z),
        }))
    }

    fn separate_xyz(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        let axis = match socket {
            "X" => 0,
            "Y" => 1,
            "Z" => 2,
            _ => return Err(evaluation("unknown Separate XYZ output")),
        };
        let vector = as_field(self.input_value_with_node(
            id,
            node,
            "Vector",
            EvalValue::Scalar(json!([0.0, 0.0, 0.0])),
        )?);
        Ok(EvalValue::Field(Field::SeparateXYZ {
            vector: Box::new(vector),
            axis,
        }))
    }

    fn random_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        if socket != "Value" {
            return Err(evaluation("unknown Random Value output"));
        }
        let seed = self.integer_input(id, node, "Seed", 0)? as i32 as u32;
        let data_type = node
            .properties
            .get("data_type")
            .and_then(Value::as_str)
            .unwrap_or("FLOAT")
            .to_ascii_uppercase();
        let (min, max) = match data_type.as_str() {
            "BOOLEAN" | "BOOL" => (
                as_field(self.input_value_with_node(
                    id,
                    node,
                    "Probability",
                    EvalValue::Scalar(json!(0.5)),
                )?),
                Field::Constant(Value::Null),
            ),
            "VECTOR" | "FLOAT_VECTOR" => (
                as_field(self.input_value_with_node(
                    id,
                    node,
                    "Min",
                    EvalValue::Scalar(json!([0.0, 0.0, 0.0])),
                )?),
                as_field(self.input_value_with_node(
                    id,
                    node,
                    "Max",
                    EvalValue::Scalar(json!([1.0, 1.0, 1.0])),
                )?),
            ),
            "INT" | "INTEGER" => (
                as_field(self.input_value_with_node(
                    id,
                    node,
                    "Min",
                    EvalValue::Scalar(json!(0)),
                )?),
                as_field(self.input_value_with_node(
                    id,
                    node,
                    "Max",
                    EvalValue::Scalar(json!(100)),
                )?),
            ),
            "FLOAT" => (
                as_field(self.input_value_with_node(
                    id,
                    node,
                    "Min",
                    EvalValue::Scalar(json!(0.0)),
                )?),
                as_field(self.input_value_with_node(
                    id,
                    node,
                    "Max",
                    EvalValue::Scalar(json!(1.0)),
                )?),
            ),
            _ => {
                return Err(unsupported_graph_feature(
                    id,
                    "graph.node.random_value.data_type",
                    format!("Random Value data type `{data_type}` is not implemented"),
                ));
            }
        };
        let id_field = if let Some(link) = self
            .group
            .links
            .iter()
            .find(|link| &link.to_node == id && link.to_socket == "ID")
        {
            as_field(self.output_value(&link.from_node, &link.from_socket)?)
        } else {
            Field::IdOrIndex
        };
        Ok(EvalValue::Field(Field::Random {
            seed,
            data_type,
            id: Box::new(id_field),
            min: Box::new(min),
            max: Box::new(max),
        }))
    }

    fn scalar_input(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
        default: f64,
    ) -> Result<f64> {
        let value =
            self.input_value_with_node(id, node, socket, EvalValue::Scalar(json!(default)))?;
        match value {
            EvalValue::Scalar(value) => value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| evaluation("numeric socket input must be finite")),
            EvalValue::Field(field) => {
                field_json(&field, &self.input.cloned().unwrap_or_default(), 0, "POINT")?
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| evaluation("field input is not a finite scalar"))
            }
            EvalValue::Geometry(_) => Err(evaluation("numeric socket received geometry")),
        }
    }

    fn integer_input(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
        default: i64,
    ) -> Result<i64> {
        let value =
            self.input_value_with_node(id, node, socket, EvalValue::Scalar(json!(default)))?;
        match value {
            EvalValue::Scalar(value) => value
                .as_i64()
                .ok_or_else(|| evaluation("integer socket input must be an integer")),
            _ => Err(evaluation("integer socket received non-scalar value")),
        }
    }

    fn string_input(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
        default: &str,
    ) -> Result<String> {
        let value =
            self.input_value_with_node(id, node, socket, EvalValue::Scalar(json!(default)))?;
        match value {
            EvalValue::Scalar(value) => value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| evaluation("string socket input must be a string")),
            _ => Err(evaluation("string socket received non-scalar value")),
        }
    }

    fn vector_input(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
        default: DVec3,
        mesh: &Mesh,
    ) -> Result<DVec3> {
        let value = self.input_value_with_node(
            id,
            node,
            socket,
            EvalValue::Scalar(json!(default.to_array())),
        )?;
        match value {
            EvalValue::Scalar(value) => json_vec3(&value)
                .ok_or_else(|| evaluation("vector socket input must be a finite vec3")),
            EvalValue::Field(field) => field_vec3(&field, mesh, 0, default),
            EvalValue::Geometry(_) => Err(evaluation("vector socket received geometry")),
        }
    }

    fn quaternion_input(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
        mesh: &Mesh,
    ) -> Result<DQuat> {
        let value = self.input_value_with_node(
            id,
            node,
            socket,
            EvalValue::Scalar(json!([0.0, 0.0, 0.0, 1.0])),
        )?;
        match value {
            EvalValue::Scalar(value) => quaternion_from_json(&value),
            EvalValue::Field(field) => field_quaternion(&field, mesh, 0, "POINT"),
            EvalValue::Geometry(_) => Err(evaluation("rotation socket received geometry")),
        }
    }
    fn rotation_field_input(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<Field> {
        match self.input_value_with_node(
            id,
            node,
            socket,
            EvalValue::Scalar(json!([0.0, 0.0, 0.0, 1.0])),
        )? {
            EvalValue::Field(field) => Ok(field),
            EvalValue::Scalar(value) => Ok(Field::Constant(value)),
            EvalValue::Geometry(_) => Err(evaluation("rotation socket received geometry")),
        }
    }
}
fn quaternion_from_json(value: &Value) -> Result<DQuat> {
    if let Some(vector) = json_vec3(value) {
        return Ok(DQuat::from_euler(
            glam::EulerRot::XYZEx,
            vector.x,
            vector.y,
            vector.z,
        ));
    }
    let values = value
        .as_array()
        .filter(|values| values.len() == 4)
        .ok_or_else(|| evaluation("rotation socket input must be Euler vec3 or quaternion"))?;
    let component = |index: usize| {
        values[index]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| evaluation("rotation must be finite"))
    };
    Ok(DQuat::from_xyzw(component(0)?, component(1)?, component(2)?, component(3)?).normalize())
}

#[expect(
    clippy::too_many_lines,
    reason = "constructs all six surfaces of the subdivided geometry node cube"
)]
fn geometry_node_cube(
    size: DVec3,
    vertices_x: i64,
    vertices_y: i64,
    vertices_z: i64,
) -> Result<Mesh> {
    if !size.is_finite() {
        return Err(evaluation("cube size must be finite"));
    }
    let size = size.abs();
    let nx = segment_count(vertices_x, 2, "cube X vertices")?;
    let ny = segment_count(vertices_y, 2, "cube Y vertices")?;
    let nz = segment_count(vertices_z, 2, "cube Z vertices")?;
    let end_layer_vertices = nx
        .checked_mul(ny)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| evaluation("cube vertex count overflow"))?;
    let middle_layer_size = nx
        .checked_add(ny)
        .and_then(|count| count.checked_mul(2))
        .and_then(|count| count.checked_sub(4))
        .ok_or_else(|| evaluation("cube vertex count overflow"))?;
    let middle_layer_vertices = middle_layer_size
        .checked_mul(nz - 2)
        .ok_or_else(|| evaluation("cube vertex count overflow"))?;
    let vertex_count = end_layer_vertices
        .checked_add(middle_layer_vertices)
        .ok_or_else(|| evaluation("cube vertex count overflow"))?;
    let xy_faces = (nx - 1)
        .checked_mul(ny - 1)
        .ok_or_else(|| evaluation("cube face count overflow"))?;
    let xz_faces = (nx - 1)
        .checked_mul(nz - 1)
        .ok_or_else(|| evaluation("cube face count overflow"))?;
    let yz_faces = (ny - 1)
        .checked_mul(nz - 1)
        .ok_or_else(|| evaluation("cube face count overflow"))?;
    let face_count = xy_faces
        .checked_add(xz_faces)
        .and_then(|count| count.checked_add(yz_faces))
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| evaluation("cube face count overflow"))?;
    if vertex_count > MAX_GENERATED_POINTS || face_count > MAX_GENERATED_POINTS {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "generated primitive exceeds the vertex or face limit",
        ));
    }

    let mut positions = Vec::with_capacity(vertex_count);
    let vertex_indices = CubeSurface { nx, ny, nz };
    {
        let mut append_position = |x: usize, y: usize, z: usize| {
            positions.push(DVec3::new(
                size.x * (interpolation_factor(x, nx) - 0.5),
                size.y * (interpolation_factor(y, ny) - 0.5),
                size.z * (interpolation_factor(z, nz) - 0.5),
            ));
        };
        for z in 0..nz {
            if z == 0 || z + 1 == nz {
                for y in 0..ny {
                    for x in 0..nx {
                        append_position(x, y, z);
                    }
                }
            } else {
                for x in 0..nx {
                    append_position(x, 0, z);
                }
                for y in 1..ny - 1 {
                    append_position(0, y, z);
                    append_position(nx - 1, y, z);
                }
                for x in 0..nx {
                    append_position(x, ny - 1, z);
                }
            }
        }
    }

    let mut polygons = Vec::with_capacity(face_count);
    for y in 0..ny - 1 {
        for x in 0..nx - 1 {
            polygons.push(vec![
                cube_vertex_index(&vertex_indices, x, y, 0)?,
                cube_vertex_index(&vertex_indices, x, y + 1, 0)?,
                cube_vertex_index(&vertex_indices, x + 1, y + 1, 0)?,
                cube_vertex_index(&vertex_indices, x + 1, y, 0)?,
            ]);
            polygons.push(vec![
                cube_vertex_index(&vertex_indices, x, y, nz - 1)?,
                cube_vertex_index(&vertex_indices, x + 1, y, nz - 1)?,
                cube_vertex_index(&vertex_indices, x + 1, y + 1, nz - 1)?,
                cube_vertex_index(&vertex_indices, x, y + 1, nz - 1)?,
            ]);
        }
    }
    for z in 0..nz - 1 {
        for x in 0..nx - 1 {
            polygons.push(vec![
                cube_vertex_index(&vertex_indices, x, 0, z)?,
                cube_vertex_index(&vertex_indices, x + 1, 0, z)?,
                cube_vertex_index(&vertex_indices, x + 1, 0, z + 1)?,
                cube_vertex_index(&vertex_indices, x, 0, z + 1)?,
            ]);
            polygons.push(vec![
                cube_vertex_index(&vertex_indices, x, ny - 1, z)?,
                cube_vertex_index(&vertex_indices, x, ny - 1, z + 1)?,
                cube_vertex_index(&vertex_indices, x + 1, ny - 1, z + 1)?,
                cube_vertex_index(&vertex_indices, x + 1, ny - 1, z)?,
            ]);
        }
    }
    for z in 0..nz - 1 {
        for y in 0..ny - 1 {
            polygons.push(vec![
                cube_vertex_index(&vertex_indices, 0, y, z)?,
                cube_vertex_index(&vertex_indices, 0, y, z + 1)?,
                cube_vertex_index(&vertex_indices, 0, y + 1, z + 1)?,
                cube_vertex_index(&vertex_indices, 0, y + 1, z)?,
            ]);
            polygons.push(vec![
                cube_vertex_index(&vertex_indices, nx - 1, y, z)?,
                cube_vertex_index(&vertex_indices, nx - 1, y + 1, z)?,
                cube_vertex_index(&vertex_indices, nx - 1, y + 1, z + 1)?,
                cube_vertex_index(&vertex_indices, nx - 1, y, z + 1)?,
            ]);
        }
    }
    Mesh::from_positions_and_faces(positions, polygons)
        .map_err(|error| mesh_evaluation_error(&error))
}

struct CubeSurface {
    nx: usize,
    ny: usize,
    nz: usize,
}

fn cube_vertex_index(surface: &CubeSurface, x: usize, y: usize, z: usize) -> Result<usize> {
    let CubeSurface { nx, ny, nz } = *surface;
    if x >= nx || y >= ny || z >= nz {
        return Err(evaluation(
            "cube face references an out-of-range surface vertex",
        ));
    }
    let middle_layer_size = 2 * nx + 2 * ny - 4;
    let end_layer_size = nx * ny;
    let (layer_start, row_offset) = if z == 0 {
        (0, y * nx + x)
    } else if z + 1 == nz {
        (end_layer_size + (nz - 2) * middle_layer_size, y * nx + x)
    } else {
        let layer_start = end_layer_size + (z - 1) * middle_layer_size;
        let row_offset = if y == 0 {
            x
        } else if y + 1 == ny {
            nx + 2 * (ny - 2) + x
        } else if x == 0 {
            nx + 2 * (y - 1)
        } else if x + 1 == nx {
            nx + 2 * (y - 1) + 1
        } else {
            return Err(evaluation("cube face references a missing surface vertex"));
        };
        (layer_start, row_offset)
    };
    Ok(layer_start + row_offset)
}

fn geometry_node_uv_sphere(segments: i64, rings: i64, radius: f64) -> Result<Mesh> {
    let segments = segment_count(segments, 3, "UV sphere segments")?;
    let rings = segment_count(rings, 3, "UV sphere rings")?;
    if !radius.is_finite() {
        return Err(evaluation("UV sphere radius must be finite"));
    }
    let radius = radius.abs();
    let vertex_count = segments
        .checked_mul(rings - 1)
        .and_then(|count| count.checked_add(2))
        .ok_or_else(|| evaluation("UV sphere vertex count overflow"))?;
    let face_count = segments
        .checked_mul(rings)
        .ok_or_else(|| evaluation("UV sphere face count overflow"))?;
    if vertex_count > MAX_GENERATED_POINTS || face_count > MAX_GENERATED_POINTS {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "generated primitive exceeds the vertex or face limit",
        ));
    }

    let segments_u32 = u32::try_from(segments)
        .map_err(|_| evaluation("UV sphere segments exceed supported range"))?;
    let mut positions = Vec::with_capacity(vertex_count);
    positions.push(DVec3::new(0.0, 0.0, radius));
    for ring in 1..rings {
        let latitude = std::f64::consts::PI * interpolation_factor(ring, rings + 1);
        let z = radius * latitude.cos();
        let radial = radius * latitude.sin();
        for segment in 0..segments {
            let longitude = std::f64::consts::TAU / f64::from(segments_u32)
                + std::f64::consts::TAU * interpolation_factor(segment, segments + 1);
            positions.push(DVec3::new(
                radial * longitude.cos(),
                radial * longitude.sin(),
                z,
            ));
        }
    }
    let bottom = positions.len();
    positions.push(DVec3::new(0.0, 0.0, -radius));
    let ring_id = |ring: usize, segment: usize| 1 + ring * segments + segment % segments;
    let mut polygons = Vec::with_capacity(face_count);
    for segment in 0..segments {
        polygons.push(vec![0, ring_id(0, segment), ring_id(0, segment + 1)]);
    }
    for ring in 0..rings - 2 {
        for segment in 0..segments {
            polygons.push(vec![
                ring_id(ring, segment),
                ring_id(ring + 1, segment),
                ring_id(ring + 1, segment + 1),
                ring_id(ring, segment + 1),
            ]);
        }
    }
    for segment in 0..segments {
        polygons.push(vec![
            ring_id(rings - 2, segment),
            bottom,
            ring_id(rings - 2, segment + 1),
        ]);
    }
    Mesh::from_positions_and_faces(positions, polygons)
        .map_err(|error| mesh_evaluation_error(&error))
}

fn geometry_node_circle(vertices: i64, radius: f64) -> Result<Mesh> {
    let vertices = segment_count(vertices, 3, "circle vertices")?;
    if !radius.is_finite() {
        return Err(evaluation("circle radius must be finite"));
    }
    let radius = radius.abs();
    let vertices_u32 = u32::try_from(vertices)
        .map_err(|_| evaluation("circle vertices exceed supported range"))?;
    let positions = (0..vertices_u32)
        .map(|index| {
            let angle = std::f64::consts::TAU * f64::from(index) / f64::from(vertices_u32);
            DVec3::new(radius * angle.cos(), radius * angle.sin(), 0.0)
        })
        .collect();
    let mut mesh = Mesh::from_positions_and_faces(positions, Vec::new())
        .map_err(|error| mesh_evaluation_error(&error))?;
    for index in 0..vertices_u32 {
        mesh.insert_edge([index, (index + 1) % vertices_u32])
            .map_err(|error| mesh_evaluation_error(&error))?;
    }
    Ok(mesh)
}

fn segmented_cylinder(
    vertices: i64,
    side_segments: i64,
    fill_segments: i64,
    radius: f64,
    depth: f64,
) -> Result<Mesh> {
    let radial = segment_count(vertices, 3, "cylinder vertices")?;
    let sides = segment_count(side_segments, 1, "cylinder side segments")?;
    let fill = segment_count(fill_segments, 1, "cylinder fill segments")?;
    if !radius.is_finite() || radius < 0.0 || !depth.is_finite() {
        return Err(evaluation("cylinder radius and depth must be finite"));
    }
    if radius == 0.0 {
        let positions = if depth == 0.0 {
            vec![DVec3::ZERO]
        } else {
            vec![
                DVec3::new(0.0, 0.0, -depth * 0.5),
                DVec3::new(0.0, 0.0, depth * 0.5),
            ]
        };
        let mut mesh = Mesh::from_positions_and_faces(positions, Vec::new())
            .map_err(|error| mesh_evaluation_error(&error))?;
        if depth != 0.0 {
            mesh.insert_edge([0, 1])
                .map_err(|error| mesh_evaluation_error(&error))?;
        }
        return Ok(mesh);
    }
    let side_vertices = radial
        .checked_mul(
            sides
                .checked_add(1)
                .ok_or_else(|| evaluation("cylinder segment count overflow"))?,
        )
        .ok_or_else(|| evaluation("cylinder vertex count overflow"))?;
    let cap_vertices = if fill > 1 {
        radial
            .checked_mul(fill - 1)
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| evaluation("cylinder cap vertex count overflow"))?
    } else {
        0
    };
    ensure_generated_vertex_count(side_vertices, cap_vertices)?;
    let mut positions = Vec::with_capacity(side_vertices + cap_vertices);
    let mut polygons = Vec::with_capacity(radial * sides + 2 + radial * fill * 2);
    let mut ring_starts = Vec::with_capacity(sides + 1);
    for level in 0..=sides {
        let t = interpolation_factor(level, sides + 1);
        let z = -depth * 0.5 + depth * t;
        ring_starts.push(positions.len());
        for index in 0..radial {
            positions.push(radial_position(index, radial, radius, z)?);
        }
    }
    for side in 0..sides {
        let lower = ring_starts[side];
        let upper = ring_starts[side + 1];
        for index in 0..radial {
            let next = (index + 1) % radial;
            polygons.push(vec![
                lower + index,
                lower + next,
                upper + next,
                upper + index,
            ]);
        }
    }
    add_radial_cap(
        &mut positions,
        &mut polygons,
        ring_starts[0],
        radial,
        radius,
        -depth * 0.5,
        fill,
        false,
    )?;
    add_radial_cap(
        &mut positions,
        &mut polygons,
        *ring_starts
            .last()
            .ok_or_else(|| evaluation("cylinder has no top ring"))?,
        radial,
        radius,
        depth * 0.5,
        fill,
        true,
    )?;
    Mesh::from_positions_and_faces(positions, polygons)
        .map_err(|error| mesh_evaluation_error(&error))
}

#[expect(
    clippy::too_many_lines,
    reason = "constructs side rings, apexes, and filled caps for a cone node"
)]
fn segmented_cone(
    vertices: i64,
    side_segments: i64,
    fill_segments: i64,
    radius_bottom: f64,
    radius_top: f64,
    depth: f64,
) -> Result<Mesh> {
    let radial = segment_count(vertices, 3, "cone vertices")?;
    let sides = segment_count(side_segments, 1, "cone side segments")?;
    let fill = segment_count(fill_segments, 1, "cone fill segments")?;
    if !radius_bottom.is_finite()
        || radius_bottom < 0.0
        || !radius_top.is_finite()
        || radius_top < 0.0
        || !depth.is_finite()
    {
        return Err(evaluation("cone radii and depth must be finite"));
    }
    if radius_bottom == 0.0 && radius_top == 0.0 {
        let positions = if depth == 0.0 {
            vec![DVec3::ZERO]
        } else {
            vec![
                DVec3::new(0.0, 0.0, -depth * 0.5),
                DVec3::new(0.0, 0.0, depth * 0.5),
            ]
        };
        let mut mesh = Mesh::from_positions_and_faces(positions, Vec::new())
            .map_err(|error| mesh_evaluation_error(&error))?;
        if depth != 0.0 {
            mesh.insert_edge([0, 1])
                .map_err(|error| mesh_evaluation_error(&error))?;
        }
        return Ok(mesh);
    }
    let collapsed_rings = usize::from(radius_bottom <= 0.0) + usize::from(radius_top <= 0.0);
    let ring_count = sides + 1 - collapsed_rings;
    let side_vertices = radial
        .checked_mul(ring_count)
        .and_then(|count| count.checked_add(collapsed_rings))
        .ok_or_else(|| evaluation("cone vertex count overflow"))?;
    let active_caps = usize::from(radius_bottom > 0.0) + usize::from(radius_top > 0.0);
    let cap_vertices = if fill > 1 {
        radial
            .checked_mul(fill - 1)
            .and_then(|count| count.checked_mul(active_caps))
            .ok_or_else(|| evaluation("cone cap vertex count overflow"))?
    } else {
        0
    };
    ensure_generated_vertex_count(side_vertices, cap_vertices)?;
    let mut positions = Vec::with_capacity(side_vertices + cap_vertices);
    let mut polygons = Vec::with_capacity(radial * sides + active_caps * radial * fill);
    let mut ring_starts = Vec::with_capacity(sides + 1);
    for level in 0..=sides {
        let t = interpolation_factor(level, sides + 1);
        let radius = radius_bottom + (radius_top - radius_bottom) * t;
        let z = -depth * 0.5 + depth * t;
        if radius <= 0.0 {
            let apex = positions.len();
            positions.push(DVec3::new(0.0, 0.0, z));
            ring_starts.push((apex, true));
        } else {
            let start = positions.len();
            ring_starts.push((start, false));
            for index in 0..radial {
                positions.push(radial_position(index, radial, radius, z)?);
            }
        }
    }
    for side in 0..sides {
        let (lower, lower_apex) = ring_starts[side];
        let (upper, upper_apex) = ring_starts[side + 1];
        match (lower_apex, upper_apex) {
            (false, false) => {
                for index in 0..radial {
                    let next = (index + 1) % radial;
                    polygons.push(vec![
                        lower + index,
                        lower + next,
                        upper + next,
                        upper + index,
                    ]);
                }
            }
            (false, true) => {
                for index in 0..radial {
                    let next = (index + 1) % radial;
                    polygons.push(vec![lower + index, lower + next, upper]);
                }
            }
            (true, false) => {
                for index in 0..radial {
                    let next = (index + 1) % radial;
                    polygons.push(vec![lower, upper + next, upper + index]);
                }
            }
            (true, true) => return Err(evaluation("cone has a collapsed side segment")),
        }
    }
    if radius_bottom > 0.0 {
        let (outer, is_apex) = ring_starts[0];
        if is_apex {
            return Err(evaluation("cone bottom ring is missing"));
        }
        add_radial_cap(
            &mut positions,
            &mut polygons,
            outer,
            radial,
            radius_bottom,
            -depth * 0.5,
            fill,
            false,
        )?;
    }
    if radius_top > 0.0 {
        let (outer, is_apex) = ring_starts[sides];
        if is_apex {
            return Err(evaluation("cone top ring is missing"));
        }
        add_radial_cap(
            &mut positions,
            &mut polygons,
            outer,
            radial,
            radius_top,
            depth * 0.5,
            fill,
            true,
        )?;
    }
    Mesh::from_positions_and_faces(positions, polygons)
        .map_err(|error| mesh_evaluation_error(&error))
}

fn segment_count(value: i64, minimum: i64, name: &str) -> Result<usize> {
    if value < minimum {
        return Err(evaluation(format!("{name} must be at least {minimum}")));
    }
    let count = usize::try_from(value)
        .map_err(|_| evaluation(format!("{name} exceeds the supported range")))?;
    if count > MAX_GENERATED_POINTS {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            format!("{name} exceeds the supported limit"),
        ));
    }
    Ok(count)
}

fn ensure_generated_vertex_count(first: usize, second: usize) -> Result<()> {
    if first
        .checked_add(second)
        .is_none_or(|count| count > MAX_GENERATED_POINTS)
    {
        Err(PotError::new(
            ErrorCode::LimitExceeded,
            "generated primitive exceeds the vertex limit",
        ))
    } else {
        Ok(())
    }
}

fn radial_position(index: usize, radial_count: usize, radius: f64, z: f64) -> Result<DVec3> {
    let index =
        u32::try_from(index).map_err(|_| evaluation("radial index exceeds the supported range"))?;
    let radial_count = u32::try_from(radial_count)
        .map_err(|_| evaluation("radial count exceeds the supported range"))?;
    let angle = std::f64::consts::TAU * f64::from(index) / f64::from(radial_count);
    Ok(DVec3::new(radius * angle.cos(), radius * angle.sin(), z))
}

fn add_radial_cap(
    positions: &mut Vec<DVec3>,
    polygons: &mut Vec<Vec<usize>>,
    outer_start: usize,
    radial: usize,
    radius: f64,
    z: f64,
    fill_segments: usize,
    top: bool,
) -> Result<()> {
    if fill_segments == 1 {
        let mut face = Vec::with_capacity(radial);
        for index in 0..radial {
            face.push(outer_start + if top { index } else { radial - index - 1 });
        }
        polygons.push(face);
        return Ok(());
    }
    let first_inner = positions.len();
    for segment in 1..fill_segments {
        let segment_radius = radius * interpolation_factor(segment, fill_segments + 1);
        for index in 0..radial {
            positions.push(radial_position(index, radial, segment_radius, z)?);
        }
    }
    let innermost = first_inner + (fill_segments - 2) * radial;
    let mut face = Vec::with_capacity(radial);
    for index in 0..radial {
        face.push(innermost + if top { index } else { radial - index - 1 });
    }
    polygons.push(face);
    for segment in 1..fill_segments {
        let inner = first_inner + (segment - 1) * radial;
        let outer = if segment + 1 == fill_segments {
            outer_start
        } else {
            first_inner + segment * radial
        };
        for index in 0..radial {
            let next = (index + 1) % radial;
            polygons.push(if top {
                vec![outer + index, outer + next, inner + next, inner + index]
            } else {
                vec![outer + index, inner + index, inner + next, outer + next]
            });
        }
    }
    Ok(())
}

fn cached_simulation_state(
    signature: &str,
    zone_input: &Id,
    frame: i64,
) -> Result<Option<GeometryValue>> {
    let cache = SIMULATION_CACHE
        .lock()
        .map_err(|_| evaluation("simulation cache lock was poisoned"))?;
    Ok(cache
        .get(&(signature.to_owned(), zone_input.clone(), frame))
        .cloned())
}

fn cache_simulation_state(
    signature: &str,
    zone_input: &Id,
    frame: i64,
    state: GeometryValue,
) -> Result<()> {
    let mut cache = SIMULATION_CACHE
        .lock()
        .map_err(|_| evaluation("simulation cache lock was poisoned"))?;
    if cache.len() >= MAX_SIMULATION_CACHE_ENTRIES {
        cache.pop_first();
    }
    cache.insert((signature.to_owned(), zone_input.clone(), frame), state);
    Ok(())
}

fn node_default_input(node: &GraphNode, socket: &str) -> Option<Value> {
    default_input(&node.node_type, socket)
}

fn function_default_input(node_type: &str, socket: &str) -> Option<Value> {
    function_default_vector_input(node_type, socket)
        .or_else(|| function_default_matrix_input(node_type, socket))
}

fn function_default_vector_input(node_type: &str, socket: &str) -> Option<Value> {
    let zero_rotation = match node_type {
        "FunctionNodeAlignEulerToVector"
        | "FunctionNodeAlignRotationToVector"
        | "FunctionNodeInvertRotation"
        | "FunctionNodeRotationToAxisAngle"
        | "FunctionNodeRotationToEuler"
        | "FunctionNodeRotationToQuaternion"
        | "FunctionNodeCombineTransform"
        | "FunctionNodeRotateVector" => socket == "Rotation",
        "FunctionNodeEulerToRotation" => socket == "Euler",
        "FunctionNodeRotateRotation" | "FunctionNodeRotateEuler" => {
            matches!(socket, "Rotation" | "Rotate By")
        }
        _ => false,
    };
    if zero_rotation {
        return Some(json!([0.0, 0.0, 0.0]));
    }
    let zero_vector = match node_type {
        "FunctionNodeCombineTransform" | "GeometryNodeTransform" => socket == "Translation",
        "FunctionNodeRotateVector" | "FunctionNodeProjectPoint" | "FunctionNodeTransformPoint" => {
            socket == "Vector"
        }
        "FunctionNodeTransformDirection" => socket == "Direction",
        "GeometryNodeSetPosition" => socket == "Offset",
        "ShaderNodeVectorMath" => matches!(socket, "Vector" | "Vector_001" | "Vector_002"),
        _ => false,
    };
    if zero_vector {
        return Some(json!([0.0, 0.0, 0.0]));
    }
    let unit_z = match node_type {
        "FunctionNodeAlignEulerToVector" | "FunctionNodeAlignRotationToVector" => {
            socket == "Vector"
        }
        "FunctionNodeAxisAngleToRotation" | "FunctionNodeRotateEuler" => socket == "Axis",
        "FunctionNodeAxesToRotation" => socket == "Primary Axis",
        _ => false,
    };
    if unit_z {
        return Some(json!([0.0, 0.0, 1.0]));
    }
    let zero_scalar = match node_type {
        "FunctionNodeAxisAngleToRotation" | "FunctionNodeRotateEuler" => socket == "Angle",
        "FunctionNodeQuaternionToRotation" | "GeometryNodeCombineXYZ" => {
            matches!(socket, "X" | "Y" | "Z")
        }
        "GeometryNodeSubdivisionSurface" => matches!(socket, "Edge Crease" | "Vertex Crease"),
        _ => false,
    };
    if zero_scalar {
        return Some(json!(0.0));
    }
    let unit_scale = matches!(
        node_type,
        "FunctionNodeCombineTransform" | "GeometryNodeTransform" | "GeometryNodeInstanceOnPoints"
    ) && socket == "Scale";
    if unit_scale {
        return Some(json!([1.0, 1.0, 1.0]));
    }
    let unit_scalar = match node_type {
        "FunctionNodeAlignEulerToVector" | "FunctionNodeAlignRotationToVector" => {
            socket == "Factor"
        }
        "FunctionNodeQuaternionToRotation" => socket == "W",
        "GeometryNodeDistributePointsOnFaces" => socket == "Density Factor",
        "ShaderNodeVectorMath" => socket == "Scale",
        _ => false,
    };
    if unit_scalar {
        return Some(json!(1.0));
    }
    if node_type == "FunctionNodeAxesToRotation" && socket == "Secondary Axis" {
        return Some(json!([1.0, 0.0, 0.0]));
    }
    None
}

fn function_default_matrix_input(node_type: &str, socket: &str) -> Option<Value> {
    if node_type == "FunctionNodeCombineMatrix" {
        if matches!(
            socket,
            "Column 1 Row 1" | "Column 2 Row 2" | "Column 3 Row 3" | "Column 4 Row 4"
        ) {
            return Some(json!(1.0));
        }
        if matches!(
            socket,
            "Column 1 Row 2"
                | "Column 1 Row 3"
                | "Column 1 Row 4"
                | "Column 2 Row 1"
                | "Column 2 Row 3"
                | "Column 2 Row 4"
                | "Column 3 Row 1"
                | "Column 3 Row 2"
                | "Column 3 Row 4"
                | "Column 4 Row 1"
                | "Column 4 Row 2"
                | "Column 4 Row 3"
        ) {
            return Some(json!(0.0));
        }
    }
    let matrix_node = matches!(
        node_type,
        "FunctionNodeInvertMatrix"
            | "FunctionNodeMatrixDeterminant"
            | "FunctionNodeMatrixMultiply"
            | "FunctionNodeMatrixSVD"
            | "FunctionNodeProjectPoint"
            | "FunctionNodeSeparateMatrix"
            | "FunctionNodeSeparateTransform"
            | "FunctionNodeTransformDirection"
            | "FunctionNodeTransformPoint"
            | "FunctionNodeTransposeMatrix"
    );
    (matrix_node && matches!(socket, "Matrix" | "Matrix_001" | "Transform")).then_some(Value::Null)
}
fn default_input(node_type: &str, socket: &str) -> Option<Value> {
    if let Some(value) = function_default_input(node_type, socket) {
        return Some(value);
    }
    let value = match (node_type, socket) {
        ("ShaderNodeMixRGB", "Color1" | "Color2") => json!([0.5, 0.5, 0.5, 1.0]),
        ("GeometryNodeTransform", "Mode") => json!("Components"),
        ("GeometryNodeTransform" | "GeometryNodeInstanceOnPoints", "Rotation") => {
            json!([0.0, 0.0, 0.0, 1.0])
        }
        ("GeometryNodeTransform" | "GeometryNodeInstanceOnPoints", "Scale") => {
            json!([1.0, 1.0, 1.0])
        }
        (
            "GeometryNodeSetPosition"
            | "GeometryNodeInstanceOnPoints"
            | "GeometryNodeDistributePointsOnFaces"
            | "GeometryNodeMeshToPoints",
            "Selection",
        )
        | ("GeometryNodeRealizeInstances", "Selection" | "Realize All")
        | ("GeometryNodeSubdivisionSurface", "Limit Surface") => json!(true),
        ("GeometryNodeRealizeInstances", "Depth")
        | ("GeometryNodeDistributePointsOnFaces", "Seed")
        | ("FunctionNodeIntegerMath", "Value" | "Value_001" | "Value_002") => json!(0),
        ("GeometryNodeMeshLine", "Count") => json!(10),
        ("GeometryNodeDistributePointsOnFaces", "Density" | "Density Max") => json!(10.0),
        ("GeometryNodeSubdivisionSurface", "Quality") => json!(3),
        ("GeometryNodeSubdivisionSurface", "UV Smooth") => json!("PRESERVE_BOUNDARIES"),
        ("GeometryNodeSubdivisionSurface", "Boundary Smooth") => json!("ALL"),
        ("GeometryNodeSimulationOutput", "Skip") => json!(false),
        ("GeometryNodeRepeatInput", "Iterations") => json!(1),
        ("ShaderNodeMath", "Value" | "Value_001" | "Value_002") | ("ShaderNodeMixRGB", "Fac") => {
            json!(0.5)
        }
        _ => return None,
    };
    Some(value)
}

fn as_field(value: EvalValue) -> Field {
    match value {
        EvalValue::Field(field) => field,
        EvalValue::Scalar(value) => Field::Constant(value),
        EvalValue::Geometry(_) => Field::Constant(Value::Null),
    }
}

fn into_geometry(value: EvalValue) -> Result<GeometryValue> {
    match value {
        EvalValue::Geometry(geometry) => Ok(geometry),
        _ => Err(evaluation("geometry socket received a non-geometry value")),
    }
}

#[derive(Clone, Copy)]
enum FieldEvaluationContext<'a> {
    Mesh(&'a Mesh),
    Instance {
        attributes: &'a BTreeMap<String, InstanceAttributeColumn>,
        transform: &'a [f64; 16],
        instance_count: usize,
    },
}

fn field_json(field: &Field, mesh: &Mesh, index: usize, domain: &str) -> Result<Value> {
    field_json_in_context(field, FieldEvaluationContext::Mesh(mesh), index, domain)
}

fn field_json_instances(
    field: &Field,
    instances: &EvaluatedInstances,
    index: usize,
) -> Result<Value> {
    let transform = instances
        .transforms
        .get(index)
        .ok_or_else(|| evaluation("instance transform index is out of range"))?;
    field_json_for_instance(
        field,
        &instances.attributes,
        transform,
        index,
        instances.len(),
    )
}

fn field_json_for_instance(
    field: &Field,
    attributes: &BTreeMap<String, InstanceAttributeColumn>,
    transform: &[f64; 16],
    index: usize,
    instance_count: usize,
) -> Result<Value> {
    field_json_in_context(
        field,
        FieldEvaluationContext::Instance {
            attributes,
            transform,
            instance_count,
        },
        index,
        "INSTANCE",
    )
}

fn id_or_index_value(
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<Value> {
    match context {
        FieldEvaluationContext::Mesh(mesh) => {
            let Some(attribute) = mesh.attributes.get("id") else {
                return Ok(json!(index));
            };
            if attribute
                .get("domain")
                .and_then(Value::as_str)
                .is_some_and(|attribute_domain| attribute_domain != domain)
            {
                return Err(evaluation(
                    "default Random Value ID field requires the id attribute on its evaluated domain",
                ));
            }
            attribute
                .get("data")
                .and_then(Value::as_array)
                .and_then(|values| values.get(index))
                .cloned()
                .ok_or_else(|| evaluation("id attribute payload is malformed"))
        }
        FieldEvaluationContext::Instance {
            attributes,
            instance_count,
            ..
        } if domain == "INSTANCE" => {
            let Some(attribute) = attributes.get("id") else {
                return Ok(json!(index));
            };
            if attribute.values.len() != instance_count {
                return Err(evaluation("named instance id column has an invalid length"));
            }
            attribute
                .values
                .get(index)
                .cloned()
                .ok_or_else(|| evaluation("named instance id index is out of range"))
        }
        FieldEvaluationContext::Instance { .. } => Err(evaluation(
            "instance field context has a non-INSTANCE domain",
        )),
    }
}

fn field_json_in_context(
    field: &Field,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<Value> {
    let value = match field {
        Field::Constant(value) => value.clone(),
        Field::Position => match context {
            FieldEvaluationContext::Mesh(mesh) => {
                vector_json(domain_position(mesh, index, domain)?)
            }
            FieldEvaluationContext::Instance { transform, .. } if domain == "INSTANCE" => {
                vector_json(instance_position(transform)?)
            }
            FieldEvaluationContext::Instance { .. } => {
                return Err(evaluation(
                    "instance field context has a non-INSTANCE domain",
                ));
            }
        },
        Field::Normal => match context {
            FieldEvaluationContext::Mesh(mesh) => vector_json(normal_at(mesh, index, domain)?),
            FieldEvaluationContext::Instance { .. } => {
                return Err(unsupported_instance_field_feature(
                    "constraint.geometry_attribute.instances.normal_field",
                    "Blender does not evaluate Normal fields on the INSTANCE domain",
                ));
            }
        },
        Field::Index => json!(index),
        Field::IdOrIndex => id_or_index_value(context, index, domain)?,
        Field::Attribute { name, data_type } => match context {
            FieldEvaluationContext::Mesh(mesh) => {
                let attribute = mesh
                    .attributes
                    .get(name)
                    .ok_or_else(|| evaluation("named attribute does not exist"))?;
                if attribute
                    .get("domain")
                    .and_then(Value::as_str)
                    .is_some_and(|attribute_domain| attribute_domain != domain)
                {
                    return Err(evaluation(
                        "named attribute field domain does not match its consumer domain",
                    ));
                }
                let values = attribute
                    .get("data")
                    .and_then(Value::as_array)
                    .ok_or_else(|| evaluation("named attribute payload is malformed"))?;
                validate_named_attribute_conversion(data_type.as_deref(), attribute, values)?;
                values
                    .get(index)
                    .cloned()
                    .ok_or_else(|| evaluation("named attribute index is out of range"))?
            }
            FieldEvaluationContext::Instance {
                attributes,
                transform,
                instance_count,
            } if domain == "INSTANCE" => {
                if name == "instance_transform" {
                    if data_type.as_deref().is_some_and(|requested| {
                        !named_attribute_type_compatible(requested, "FLOAT4X4")
                    }) {
                        return Err(unsupported_instance_field_feature(
                            "graph.node.input_named_attribute.data_type",
                            "Named Attribute conversion from FLOAT4X4 is not implemented",
                        ));
                    }
                    json!(transform)
                } else {
                    let column = attributes
                        .get(name)
                        .ok_or_else(|| evaluation("named instance attribute does not exist"))?;
                    if column.values.len() != instance_count {
                        return Err(evaluation(
                            "named instance attribute column has an invalid length",
                        ));
                    }
                    if data_type.as_deref().is_some_and(|requested| {
                        !named_attribute_type_compatible(
                            requested,
                            &normalize_instance_data_type(&column.data_type),
                        )
                    }) {
                        return Err(unsupported_instance_field_feature(
                            "graph.node.input_named_attribute.data_type",
                            "Named Attribute instance attribute conversion is not implemented",
                        ));
                    }
                    column.values.get(index).cloned().ok_or_else(|| {
                        evaluation("named instance attribute index is out of range")
                    })?
                }
            }
            FieldEvaluationContext::Instance { .. } => {
                return Err(evaluation(
                    "instance field context has a non-INSTANCE domain",
                ));
            }
        },
        Field::AttributeExists(name) => match context {
            FieldEvaluationContext::Mesh(mesh) => json!(mesh.attributes.contains_key(name)),
            FieldEvaluationContext::Instance { attributes, .. } => {
                json!(name == "instance_transform" || attributes.contains_key(name))
            }
        },
        Field::Captured {
            domain: captured_domain,
            values,
        } => {
            if captured_domain != domain {
                return Err(evaluation(
                    "captured attribute domain does not match its consumer",
                ));
            }
            values
                .get(index)
                .cloned()
                .ok_or_else(|| evaluation("captured attribute index is out of range"))?
        }
        Field::ProceduralTexture {
            kind,
            inputs,
            output,
        } => procedural_texture_field_value(kind, inputs, output, context, index, domain)?,
        Field::Math {
            operation,
            clamp,
            first,
            second,
            third,
        } => {
            let a = numeric_field_in_context(first, context, index, domain, "math")? as f32;
            let b = if math_uses_second(operation) {
                numeric_field_in_context(second, context, index, domain, "math")? as f32
            } else {
                0.0
            };
            let c = if math_uses_third(operation) {
                numeric_field_in_context(third, context, index, domain, "math")? as f32
            } else {
                0.0
            };
            let result = crate::node_kernels::math(
                operation,
                a,
                b,
                c,
                *clamp,
                crate::node_kernels::MathSemantics::Geometry,
            )
            .ok_or_else(|| unsupported_graph_operation("ShaderNodeMath", operation))?;
            if !result.is_finite() {
                return Err(evaluation("math field produced a non-finite value"));
            }
            json!(f64::from(result))
        }
        Field::VectorMath {
            operation,
            first,
            second,
            third,
            scale,
            output,
        } => {
            let first = vector_field_in_context(first, context, index, domain, "vector math")?
                .to_array()
                .map(|component| component as f32);
            let second = if vector_math_uses_second(operation) {
                vector_field_in_context(second, context, index, domain, "vector math")?
                    .to_array()
                    .map(|component| component as f32)
            } else {
                [0.0; 3]
            };
            let third = if vector_math_uses_third(operation) {
                vector_field_in_context(third, context, index, domain, "vector math")?
                    .to_array()
                    .map(|component| component as f32)
            } else {
                [0.0; 3]
            };
            let scale = if matches!(operation.as_str(), "SCALE" | "REFRACT") {
                finite_f32(
                    numeric_field_in_context(scale, context, index, domain, "vector math scale")?,
                    "vector math scale",
                )?
            } else {
                1.0
            };
            let result = crate::node_kernels::vector_math(
                operation,
                output,
                first,
                second,
                third,
                scale,
                crate::node_kernels::MathSemantics::Geometry,
            )
            .ok_or_else(|| unsupported_graph_operation("ShaderNodeVectorMath", operation))?;
            match result {
                crate::node_kernels::KernelValue::Scalar(value)
                    if output == "Value" && value.is_finite() =>
                {
                    json!(f64::from(value))
                }
                crate::node_kernels::KernelValue::Vector(value)
                    if output == "Vector"
                        && value.iter().all(|component| component.is_finite()) =>
                {
                    json!(value.map(f64::from))
                }
                _ => return Err(evaluation("vector math produced an invalid output")),
            }
        }
        Field::IntegerMath {
            operation,
            first,
            second,
            third,
        } => {
            let a = integer_field_i32(first, context, index, domain)?;
            let b = if integer_math_uses_second(operation) {
                integer_field_i32(second, context, index, domain)?
            } else {
                0
            };
            let c = if operation == "MULTIPLY_ADD" {
                integer_field_i32(third, context, index, domain)?
            } else {
                0
            };
            let result = crate::node_kernels::integer_math(operation, a, b, c)
                .ok_or_else(|| unsupported_graph_operation("FunctionNodeIntegerMath", operation))?;
            json!(result)
        }
        Field::BooleanMath {
            operation,
            first,
            second,
        } => {
            let a = boolean_field_in_context(first, context, index, domain)?;
            let b = if operation == "NOT" {
                false
            } else {
                boolean_field_in_context(second, context, index, domain)?
            };
            let result = crate::node_kernels::boolean_math(operation, a, b, false)
                .ok_or_else(|| unsupported_graph_operation("FunctionNodeBooleanMath", operation))?;
            json!(result)
        }
        Field::Compare {
            operation,
            data_type,
            mode,
            first,
            second,
            parameter,
            epsilon,
        } => {
            let first_value = field_json_in_context(first, context, index, domain)?;
            let second_value = field_json_in_context(second, context, index, domain)?;
            let first_value = compare_value_from_json(data_type, &first_value)?;
            let second_value = compare_value_from_json(data_type, &second_value)?;
            let parameter =
                if data_type == "VECTOR" && matches!(mode.as_str(), "DOT_PRODUCT" | "DIRECTION") {
                    finite_f32(
                        numeric_field_in_context(
                            parameter,
                            context,
                            index,
                            domain,
                            "Compare parameter",
                        )?,
                        "Compare parameter",
                    )?
                } else {
                    0.0
                };
            let epsilon = if matches!(operation.as_str(), "EQUAL" | "NOT_EQUAL")
                && matches!(data_type.as_str(), "FLOAT" | "VECTOR" | "RGBA")
            {
                finite_f32(
                    numeric_field_in_context(epsilon, context, index, domain, "Compare epsilon")?,
                    "Compare epsilon",
                )?
            } else {
                0.0
            };
            let result = crate::node_kernels::compare(
                operation,
                mode,
                first_value,
                second_value,
                parameter,
                epsilon,
            )
            .ok_or_else(|| unsupported_graph_compare(data_type, operation, mode))?;
            json!(result)
        }
        Field::MapRange {
            data_type,
            interpolation,
            clamp,
            value,
            from_min,
            from_max,
            to_min,
            to_max,
            steps,
        } => {
            let value = kernel_value_field(data_type, value, context, index, domain, "Map Range")?;
            let from_min =
                kernel_value_field(data_type, from_min, context, index, domain, "Map Range")?;
            let from_max =
                kernel_value_field(data_type, from_max, context, index, domain, "Map Range")?;
            let to_min =
                kernel_value_field(data_type, to_min, context, index, domain, "Map Range")?;
            let to_max =
                kernel_value_field(data_type, to_max, context, index, domain, "Map Range")?;
            let steps = if interpolation == "STEPPED" {
                kernel_value_field(data_type, steps, context, index, domain, "Map Range steps")?
            } else {
                match data_type.as_str() {
                    "FLOAT" => crate::node_kernels::KernelValue::Scalar(0.0),
                    "FLOAT_VECTOR" => crate::node_kernels::KernelValue::Vector([0.0; 3]),
                    _ => return Err(evaluation("Map Range data type is invalid")),
                }
            };
            let result = crate::node_kernels::map_range(
                value,
                from_min,
                from_max,
                to_min,
                to_max,
                steps,
                interpolation,
                *clamp,
            )
            .ok_or_else(|| {
                unsupported_graph_node_state_type("ShaderNodeMapRange", interpolation)
            })?;
            kernel_value_json(result)?
        }
        Field::Clamp {
            mode,
            value,
            min,
            max,
        } => {
            let value = finite_f32(
                numeric_field_in_context(value, context, index, domain, "Clamp")?,
                "Clamp value",
            )?;
            let min = finite_f32(
                numeric_field_in_context(min, context, index, domain, "Clamp")?,
                "Clamp minimum",
            )?;
            let max = finite_f32(
                numeric_field_in_context(max, context, index, domain, "Clamp")?,
                "Clamp maximum",
            )?;
            let result = crate::node_kernels::clamp(
                crate::node_kernels::KernelValue::Scalar(value),
                crate::node_kernels::KernelValue::Scalar(min),
                crate::node_kernels::KernelValue::Scalar(max),
                mode,
            )
            .ok_or_else(|| unsupported_graph_node_state_type("ShaderNodeClamp", mode))?;
            kernel_value_json(result)?
        }
        Field::Mix {
            node_type,
            output,
            data_type,
            factor_mode,
            blend_mode,
            clamp_factor,
            clamp_result,
            factor,
            first,
            second,
        } => mix_field_json(
            node_type,
            *output,
            data_type,
            factor_mode,
            blend_mode,
            *clamp_factor,
            *clamp_result,
            factor,
            first,
            second,
            context,
            index,
            domain,
        )?,
        Field::ColorRamp {
            elements,
            interpolation,
            color_mode,
            hue_interpolation,
            factor,
            output,
        } => {
            let factor = finite_f32(
                numeric_field_in_context(factor, context, index, domain, "Color Ramp factor")?,
                "Color Ramp factor",
            )?;
            let color = crate::node_kernels::color_ramp(
                elements,
                factor,
                interpolation,
                color_mode,
                hue_interpolation,
            )
            .ok_or_else(|| {
                unsupported_graph_node_state_type(
                    "ShaderNodeValToRGB",
                    &format!("{interpolation}/{color_mode}/{hue_interpolation}"),
                )
            })?;
            if color.iter().any(|component| !component.is_finite()) {
                return Err(evaluation("Color Ramp produced a non-finite value"));
            }
            match output.as_str() {
                "Color" => json!(color.map(f64::from)),
                "Alpha" => json!(f64::from(color[3])),
                _ => return Err(evaluation("unknown Color Ramp output")),
            }
        }
        Field::Curve {
            kind,
            mapping,
            prepared,
            factor,
            value,
        } => curve_field_json(
            kind, mapping, prepared, factor, value, context, index, domain,
        )?,
        Field::CombineXYZ { x, y, z } => {
            let x = numeric_field_in_context(x, context, index, domain, "Combine XYZ")?;
            let y = numeric_field_in_context(y, context, index, domain, "Combine XYZ")?;
            let z = numeric_field_in_context(z, context, index, domain, "Combine XYZ")?;
            vector_json(DVec3::new(x, y, z))
        }
        Field::SeparateXYZ { vector, axis } => {
            let vector = vector_field_in_context(vector, context, index, domain, "Separate XYZ")?;
            json!(vector[*axis])
        }
        Field::Function {
            node_type,
            properties,
            inputs,
            output,
        } => {
            let mut values: [(&str, Value); 16] = std::array::from_fn(|_| ("", Value::Null));
            let mut count = 0;
            for (socket, field) in inputs {
                if count == values.len() {
                    return Err(evaluation("rotation or matrix node has too many inputs"));
                }
                values[count] = (
                    socket,
                    field_json_in_context(field, context, index, domain)?,
                );
                count += 1;
            }
            super::rotation_matrix::evaluate(node_type, properties, &values[..count], output)?
        }
        Field::Random {
            seed,
            data_type,
            id,
            min,
            max,
        } => {
            let id = integer_field_in_context(id, context, index, domain)? as i32 as u32;
            let minimum = field_json_in_context(min, context, index, domain)?;
            let maximum = field_json_in_context(max, context, index, domain)?;
            match data_type.as_str() {
                "VECTOR" | "FLOAT_VECTOR" => {
                    let minimum = json_vec3(&minimum)
                        .ok_or_else(|| evaluation("random vector minimum is invalid"))?
                        .as_vec3();
                    let maximum = json_vec3(&maximum)
                        .ok_or_else(|| evaluation("random vector maximum is invalid"))?
                        .as_vec3();
                    vector_json(
                        Vec3::new(
                            blender_noise_hash_to_float_3(*seed, id, 0) * (maximum.x - minimum.x)
                                + minimum.x,
                            blender_noise_hash_to_float_3(*seed, id, 1) * (maximum.y - minimum.y)
                                + minimum.y,
                            blender_noise_hash_to_float_3(*seed, id, 2) * (maximum.z - minimum.z)
                                + minimum.z,
                        )
                        .as_dvec3(),
                    )
                }
                "INT" | "INTEGER" => {
                    let mut minimum = minimum
                        .as_i64()
                        .ok_or_else(|| evaluation("random integer minimum is invalid"))?
                        as i32;
                    let mut maximum = maximum
                        .as_i64()
                        .ok_or_else(|| evaluation("random integer maximum is invalid"))?
                        as i32;
                    if minimum > maximum {
                        std::mem::swap(&mut minimum, &mut maximum);
                    }
                    let range = (maximum as u32)
                        .wrapping_sub(minimum as u32)
                        .wrapping_add(1);
                    let hash = blender_noise_hash_2(id, *seed);
                    let modulo_result = if range == 0 { hash } else { hash % range };
                    json!((minimum as u32).wrapping_add(modulo_result) as i32)
                }
                "BOOLEAN" | "BOOL" => {
                    let probability = minimum
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| evaluation("random boolean probability is invalid"))?
                        as f32;
                    json!(blender_noise_hash_to_float_2(id, *seed) <= probability)
                }
                "FLOAT" => {
                    let minimum = minimum
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| evaluation("random minimum is not numeric"))?
                        as f32;
                    let maximum = maximum
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| evaluation("random maximum is not numeric"))?
                        as f32;
                    json!(blender_noise_hash_to_float_2(*seed, id) * (maximum - minimum) + minimum)
                }
                _ => return Err(evaluation("unsupported Random Value data type")),
            }
        }
    };
    if json_is_finite(&value) {
        Ok(value)
    } else {
        Err(evaluation("field produced a non-finite value"))
    }
}
fn procedural_texture_field_value(
    kind: &TextureFieldKind,
    inputs: &[Field],
    output: &str,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<Value> {
    match kind {
        TextureFieldKind::Wave {
            wave_type,
            bands_direction,
            rings_direction,
            profile,
        } => {
            let vector = texture_field_vector(inputs, 0, context, index, domain)?;
            let value = procedural_texture::wave(
                vector,
                texture_field_number(inputs, 1, context, index, domain)?,
                texture_field_number(inputs, 2, context, index, domain)?,
                texture_field_number(inputs, 3, context, index, domain)?,
                texture_field_number(inputs, 4, context, index, domain)?,
                texture_field_number(inputs, 5, context, index, domain)?,
                texture_field_number(inputs, 6, context, index, domain)?,
                *wave_type,
                *bands_direction,
                *rings_direction,
                *profile,
            );
            match output {
                "Color" => Ok(json!([value, value, value, 1.0])),
                "Fac" => Ok(json!(value)),
                _ => Err(evaluation("unknown Wave Texture field output")),
            }
        }
        TextureFieldKind::Magic { depth } => {
            let vector = texture_field_vector(inputs, 0, context, index, domain)?;
            let (color, factor) = procedural_texture::magic(
                vector,
                texture_field_number(inputs, 1, context, index, domain)?,
                texture_field_number(inputs, 2, context, index, domain)?,
                *depth,
            );
            match output {
                "Color" => Ok(json!(color.map(f64::from))),
                "Fac" => Ok(json!(f64::from(factor))),
                _ => Err(evaluation("unknown Magic Texture field output")),
            }
        }
        TextureFieldKind::Brick {
            offset,
            offset_frequency,
            squash,
            squash_frequency,
        } => {
            let vector = texture_field_vector(inputs, 0, context, index, domain)?;
            let color1 = texture_field_color(inputs, 1, context, index, domain)?;
            let color2 = texture_field_color(inputs, 2, context, index, domain)?;
            let mortar = texture_field_color(inputs, 3, context, index, domain)?;
            let scale = texture_field_number(inputs, 4, context, index, domain)?;
            let mortar_size = texture_field_number(inputs, 5, context, index, domain)?;
            let mortar_smooth = texture_field_number(inputs, 6, context, index, domain)?;
            let bias = texture_field_number(inputs, 7, context, index, domain)?;
            let width = texture_field_number(inputs, 8, context, index, domain)?;
            let row_height = texture_field_number(inputs, 9, context, index, domain)?;
            if width == 0.0 || row_height == 0.0 {
                return Err(evaluation(
                    "Brick Texture width and row height must be nonzero",
                ));
            }
            let (color, factor) = procedural_texture::brick(
                vector,
                color1,
                color2,
                mortar,
                scale,
                mortar_size,
                mortar_smooth,
                bias,
                width,
                row_height,
                *offset,
                *offset_frequency,
                *squash,
                *squash_frequency,
            );
            match output {
                "Color" => Ok(json!(color.map(f64::from))),
                "Fac" => Ok(json!(f64::from(factor))),
                _ => Err(evaluation("unknown Brick Texture field output")),
            }
        }
        TextureFieldKind::WhiteNoise { dimensions } => {
            let (vector, w) = match dimensions {
                1 => (
                    [0.0; 3],
                    texture_field_number(inputs, 0, context, index, domain)?,
                ),
                2 | 3 => (
                    texture_field_vector(inputs, 0, context, index, domain)?,
                    0.0,
                ),
                4 => (
                    texture_field_vector(inputs, 0, context, index, domain)?,
                    texture_field_number(inputs, 1, context, index, domain)?,
                ),
                _ => return Err(evaluation("White Noise Texture dimensions are invalid")),
            };
            match output {
                "Value" => procedural_texture::white_noise_value(*dimensions, vector, w)
                    .map(|value| json!(value))
                    .ok_or_else(|| evaluation("White Noise Texture dimensions are invalid")),
                "Color" => procedural_texture::white_noise_color(*dimensions, vector, w)
                    .map(|color| json!(color.map(f64::from)))
                    .ok_or_else(|| evaluation("White Noise Texture dimensions are invalid")),
                _ => Err(evaluation("unknown White Noise Texture field output")),
            }
        }
        TextureFieldKind::Gabor { dimensions } => {
            let vector = texture_field_vector(inputs, 0, context, index, domain)?;
            let scale = texture_field_number(inputs, 1, context, index, domain)?;
            let frequency = texture_field_number(inputs, 2, context, index, domain)?;
            let anisotropy = texture_field_number(inputs, 3, context, index, domain)?;
            let values = match dimensions {
                2 => procedural_texture::gabor_2d(
                    vector,
                    scale,
                    frequency,
                    anisotropy,
                    texture_field_number(inputs, 4, context, index, domain)?,
                ),
                3 => procedural_texture::gabor_3d(
                    vector,
                    scale,
                    frequency,
                    anisotropy,
                    texture_field_vector(inputs, 4, context, index, domain)?,
                ),
                _ => return Err(evaluation("Gabor Texture dimensions are invalid")),
            };
            let value = match output {
                "Value" => values[0],
                "Phase" => values[1],
                "Intensity" => values[2],
                _ => return Err(evaluation("unknown Gabor Texture field output")),
            };
            Ok(json!(f64::from(value)))
        }
    }
}

fn texture_field_input(inputs: &[Field], index: usize) -> Result<&Field> {
    inputs
        .get(index)
        .ok_or_else(|| evaluation("procedural texture field input is missing"))
}

fn texture_field_number(
    inputs: &[Field],
    input_index: usize,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<f32> {
    let value = numeric_field_in_context(
        texture_field_input(inputs, input_index)?,
        context,
        index,
        domain,
        "procedural texture",
    )? as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(evaluation(
            "procedural texture field exceeds the float range",
        ))
    }
}

fn texture_field_vector(
    inputs: &[Field],
    input_index: usize,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<[f32; 3]> {
    let value = vector_field_in_context(
        texture_field_input(inputs, input_index)?,
        context,
        index,
        domain,
        "procedural texture",
    )?;
    let vector = [value.x as f32, value.y as f32, value.z as f32];
    if vector.iter().all(|component| component.is_finite()) {
        Ok(vector)
    } else {
        Err(evaluation(
            "procedural texture vector exceeds the float range",
        ))
    }
}

fn texture_field_color(
    inputs: &[Field],
    input_index: usize,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<[f32; 4]> {
    let value = field_json_in_context(
        texture_field_input(inputs, input_index)?,
        context,
        index,
        domain,
    )?;
    let values = value
        .as_array()
        .filter(|values| values.len() == 4)
        .ok_or_else(|| evaluation("procedural texture color input is not RGBA"))?;
    let mut color = [0.0; 4];
    for (output, value) in color.iter_mut().zip(values) {
        *output = value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| evaluation("procedural texture color input is invalid"))?
            as f32;
    }
    if color.iter().all(|component| component.is_finite()) {
        Ok(color)
    } else {
        Err(evaluation(
            "procedural texture color exceeds the float range",
        ))
    }
}

fn integer_math_uses_second(operation: &str) -> bool {
    !matches!(operation, "ABSOLUTE" | "SIGN" | "NEGATE")
}

fn math_uses_second(operation: &str) -> bool {
    !matches!(
        operation,
        "SQRT"
            | "INVERSE_SQRT"
            | "INV_SQRT"
            | "ABSOLUTE"
            | "EXPONENT"
            | "SIGN"
            | "ROUND"
            | "FLOOR"
            | "CEIL"
            | "TRUNC"
            | "FRACT"
            | "SINE"
            | "COSINE"
            | "TANGENT"
            | "ARCSINE"
            | "ASIN"
            | "ARCCOSINE"
            | "ACOS"
            | "ARCTANGENT"
            | "ATAN"
            | "SINH"
            | "COSH"
            | "TANH"
            | "DEGREES"
            | "RADIANS"
            | "FRACTION"
    )
}

fn math_uses_third(operation: &str) -> bool {
    matches!(
        operation,
        "MULTIPLY_ADD" | "COMPARE" | "SMOOTH_MIN" | "SMOOTH_MAX" | "WRAP"
    )
}

fn vector_math_uses_second(operation: &str) -> bool {
    !matches!(
        operation,
        "LENGTH"
            | "SCALE"
            | "NORMALIZE"
            | "ABSOLUTE"
            | "FLOOR"
            | "CEIL"
            | "ROUND"
            | "FRACTION"
            | "FRACT"
            | "SIGN"
            | "SINE"
            | "COSINE"
            | "TANGENT"
    )
}

fn vector_math_uses_third(operation: &str) -> bool {
    matches!(operation, "FACEFORWARD" | "WRAP" | "MULTIPLY_ADD")
}

fn node_enum(node: &GraphNode, key: &str, default: &str) -> String {
    node.properties
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_ascii_uppercase()
}

fn unsupported_graph_node_state(node: &GraphNode, state: &str) -> PotError {
    unsupported_graph_node_state_type(&node.node_type, state)
}

fn unsupported_graph_node_state_type(node_type: &str, state: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("{node_type} state `{state}` is not implemented"),
        json!({
            "feature_id": format!("graph.node.{}", node_type.to_ascii_lowercase()),
            "state": state,
        }),
    )
}

fn unsupported_graph_operation(node_type: &str, operation: &str) -> PotError {
    let feature_id = match node_type {
        "ShaderNodeMath" => format!("graph.shader_math.{operation}"),
        "ShaderNodeVectorMath" => format!("graph.shader_vector_math.{operation}"),
        "FunctionNodeIntegerMath" => format!("graph.function_integer_math.{operation}"),
        "FunctionNodeBooleanMath" => format!("graph.function_boolean_math.{operation}"),
        _ => format!(
            "graph.node.{}.{}",
            node_type.to_ascii_lowercase(),
            operation
        ),
    };
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("{node_type} operation `{operation}` is not implemented"),
        json!({"feature_id":feature_id}),
    )
}

fn unsupported_graph_compare(data_type: &str, operation: &str, mode: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("FunctionNodeCompare `{data_type}/{mode}/{operation}` is not implemented"),
        json!({"feature_id":format!("graph.function_compare.{data_type}.{mode}.{operation}")}),
    )
}

fn finite_f32(value: f64, node: &str) -> Result<f32> {
    let value = value as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(evaluation(format!(
            "{node} input is outside the finite f32 range"
        )))
    }
}

fn kernel_vec3(value: DVec3, node: &str) -> Result<[f32; 3]> {
    Ok([
        finite_f32(value.x, node)?,
        finite_f32(value.y, node)?,
        finite_f32(value.z, node)?,
    ])
}

fn json_color4(value: &Value) -> Option<[f32; 4]> {
    let values = value.as_array()?;
    if values.len() != 4 {
        return None;
    }
    let mut color = [0.0; 4];
    for (component, value) in color.iter_mut().zip(values) {
        *component = value.as_f64()? as f32;
        if !component.is_finite() {
            return None;
        }
    }
    Some(color)
}

fn integer_field_i32(
    field: &Field,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<i32> {
    let value = field_json_in_context(field, context, index, domain)?
        .as_i64()
        .ok_or_else(|| evaluation("integer math input must produce an integer"))?;
    i32::try_from(value).map_err(|_| evaluation("integer math input exceeds Blender's int range"))
}

fn boolean_field_in_context(
    field: &Field,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<bool> {
    field_json_in_context(field, context, index, domain)?
        .as_bool()
        .ok_or_else(|| evaluation("boolean math input must produce a boolean"))
}

fn compare_value_from_json<'a>(
    data_type: &str,
    value: &'a Value,
) -> Result<crate::node_kernels::CompareValue<'a>> {
    use crate::node_kernels::CompareValue;
    Ok(match data_type {
        "FLOAT" => {
            let scalar = value
                .as_f64()
                .ok_or_else(|| evaluation("Compare input must produce a number"))?
                as f32;
            if scalar.is_nan() {
                return Err(evaluation("Compare float input must not be NaN"));
            }
            CompareValue::Scalar(scalar)
        }
        "INT" | "INTEGER" => CompareValue::Integer(
            i32::try_from(
                value
                    .as_i64()
                    .ok_or_else(|| evaluation("Compare input must produce an integer"))?,
            )
            .map_err(|_| evaluation("Compare integer exceeds Blender's int range"))?,
        ),
        "VECTOR" | "FLOAT_VECTOR" => {
            let vector = json_vec3(value)
                .ok_or_else(|| evaluation("Compare input must produce a finite vector"))?;
            CompareValue::Vector(kernel_vec3(vector, "Compare")?)
        }
        "RGBA" | "COLOR" => CompareValue::Color(
            json_color4(value).ok_or_else(|| evaluation("Compare input must produce a color"))?,
        ),
        "BOOLEAN" | "BOOL" => CompareValue::Boolean(
            value
                .as_bool()
                .ok_or_else(|| evaluation("Compare input must produce a boolean"))?,
        ),
        "STRING" => CompareValue::String(
            value
                .as_str()
                .ok_or_else(|| evaluation("Compare input must produce a string"))?,
        ),
        "OBJECT" | "IMAGE" | "COLLECTION" | "MATERIAL" | "FONT" | "SOUND" => {
            let identifier = value
                .as_object()
                .and_then(|object| {
                    object
                        .get("bridge_id")
                        .or_else(|| object.get("name"))
                        .and_then(Value::as_str)
                })
                .or_else(|| value.as_str())
                .unwrap_or("");
            CompareValue::String(identifier)
        }
        _ => return Err(evaluation("Compare data type is invalid")),
    })
}

fn kernel_value_field(
    data_type: &str,
    field: &Field,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
    node: &str,
) -> Result<crate::node_kernels::KernelValue> {
    match data_type {
        "FLOAT" => Ok(crate::node_kernels::KernelValue::Scalar(finite_f32(
            numeric_field_in_context(field, context, index, domain, node)?,
            node,
        )?)),
        "FLOAT_VECTOR" => Ok(crate::node_kernels::KernelValue::Vector(kernel_vec3(
            vector_field_in_context(field, context, index, domain, node)?,
            node,
        )?)),
        _ => Err(evaluation(format!("{node} data type is invalid"))),
    }
}

fn kernel_value_json(value: crate::node_kernels::KernelValue) -> Result<Value> {
    match value {
        crate::node_kernels::KernelValue::Scalar(value) if value.is_finite() => {
            Ok(json!(f64::from(value)))
        }
        crate::node_kernels::KernelValue::Vector(value)
            if value.iter().all(|component| component.is_finite()) =>
        {
            Ok(json!(value.map(f64::from)))
        }
        _ => Err(evaluation("math kernel produced an invalid value")),
    }
}

fn mix_field_json(
    node_type: &str,
    output: MixOutput,
    data_type: &str,
    factor_mode: &str,
    blend_mode: &str,
    clamp_factor: bool,
    clamp_result: bool,
    factor: &Field,
    first: &Field,
    second: &Field,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<Value> {
    match data_type {
        "FLOAT" => {
            let factor = finite_f32(
                numeric_field_in_context(factor, context, index, domain, "Mix factor")?,
                "Mix factor",
            )?;
            let first = finite_f32(
                numeric_field_in_context(first, context, index, domain, "Mix")?,
                "Mix",
            )?;
            let second = finite_f32(
                numeric_field_in_context(second, context, index, domain, "Mix")?,
                "Mix",
            )?;
            let result =
                crate::node_kernels::mix_float("MIX", factor, first, second, clamp_factor, false)
                    .ok_or_else(|| unsupported_graph_node_state_type(node_type, "FLOAT/MIX"))?;
            if !result.is_finite() {
                return Err(evaluation("Mix produced a non-finite value"));
            }
            Ok(json!(f64::from(result)))
        }
        "VECTOR" => {
            let factor = if factor_mode == "NON_UNIFORM" {
                kernel_vec3(
                    vector_field_in_context(factor, context, index, domain, "Mix factor")?,
                    "Mix factor",
                )?
            } else {
                let factor = finite_f32(
                    numeric_field_in_context(factor, context, index, domain, "Mix factor")?,
                    "Mix factor",
                )?;
                [factor; 3]
            };
            let first = kernel_vec3(
                vector_field_in_context(first, context, index, domain, "Mix")?,
                "Mix",
            )?;
            let second = kernel_vec3(
                vector_field_in_context(second, context, index, domain, "Mix")?,
                "Mix",
            )?;
            let result =
                crate::node_kernels::mix_vector("MIX", factor, first, second, clamp_factor, false)
                    .ok_or_else(|| unsupported_graph_node_state_type(node_type, "VECTOR/MIX"))?;
            if result.iter().any(|component| !component.is_finite()) {
                return Err(evaluation("Mix produced a non-finite vector"));
            }
            Ok(json!(result.map(f64::from)))
        }
        "RGBA" => {
            let factor = finite_f32(
                numeric_field_in_context(factor, context, index, domain, "Mix factor")?,
                "Mix factor",
            )?;
            let first = json_color4(&field_json_in_context(first, context, index, domain)?)
                .ok_or_else(|| evaluation("Mix input must produce a color"))?;
            let second = json_color4(&field_json_in_context(second, context, index, domain)?)
                .ok_or_else(|| evaluation("Mix input must produce a color"))?;
            let result = crate::node_kernels::mix_color(
                blend_mode,
                factor,
                first,
                second,
                clamp_factor,
                clamp_result,
            )
            .ok_or_else(|| {
                unsupported_graph_node_state_type(
                    node_type,
                    &format!("RGBA blend mode `{blend_mode}`"),
                )
            })?;
            if result.iter().any(|component| !component.is_finite()) {
                return Err(evaluation("Mix produced a non-finite color"));
            }
            match output {
                MixOutput::Value | MixOutput::Color => Ok(json!(result.map(f64::from))),
            }
        }
        "ROTATION" => {
            let factor = finite_f32(
                numeric_field_in_context(factor, context, index, domain, "Mix factor")?,
                "Mix factor",
            )?;
            let first =
                quaternion_from_json(&field_json_in_context(first, context, index, domain)?)?
                    .to_array()
                    .map(|component| component as f32);
            let second =
                quaternion_from_json(&field_json_in_context(second, context, index, domain)?)?
                    .to_array()
                    .map(|component| component as f32);
            let result = crate::node_kernels::mix_rotation(factor, first, second, clamp_factor)
                .ok_or_else(|| evaluation("Mix produced an invalid rotation"))?;
            if result.iter().any(|component| !component.is_finite()) {
                return Err(evaluation("Mix produced a non-finite rotation"));
            }
            Ok(json!(result.map(f64::from)))
        }
        _ => Err(evaluation("Mix data type is invalid")),
    }
}

fn curve_field_json(
    kind: &str,
    mapping: &crate::color::curve::CurveMapping,
    prepared: &crate::color::curve::PreparedCurve,
    factor: &Field,
    value: &Field,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<Value> {
    let factor = finite_f32(
        numeric_field_in_context(factor, context, index, domain, "Curve factor")?,
        "Curve factor",
    )?;
    match kind {
        "FLOAT" => {
            let value = finite_f32(
                numeric_field_in_context(value, context, index, domain, "Float Curve")?,
                "Float Curve",
            )?;
            let mapped = crate::node_kernels::curve_mapping_value(mapping, prepared, 0, value)
                .ok_or_else(|| {
                    unsupported_graph_node_state_type("ShaderNodeFloatCurve", "mapping")
                })?;
            let result = crate::node_kernels::mix_float("MIX", factor, value, mapped, false, false)
                .ok_or_else(|| {
                    unsupported_graph_node_state_type("ShaderNodeFloatCurve", "factor")
                })?;
            if !result.is_finite() {
                return Err(evaluation("Float Curve produced a non-finite value"));
            }
            Ok(json!(f64::from(result)))
        }
        "VECTOR" => {
            let value = kernel_vec3(
                vector_field_in_context(value, context, index, domain, "Vector Curve")?,
                "Vector Curve",
            )?;
            let mapped = crate::node_kernels::curve_mapping_vector(prepared, value);
            let result =
                crate::node_kernels::mix_vector("MIX", [factor; 3], value, mapped, false, false)
                    .ok_or_else(|| {
                        unsupported_graph_node_state_type("ShaderNodeVectorCurve", "factor")
                    })?;
            if result.iter().any(|component| !component.is_finite()) {
                return Err(evaluation("Vector Curve produced a non-finite value"));
            }
            Ok(json!(result.map(f64::from)))
        }
        "RGBA" => {
            let value = json_color4(&field_json_in_context(value, context, index, domain)?)
                .ok_or_else(|| evaluation("RGB Curve input must produce a color"))?;
            let mapped_rgb =
                crate::node_kernels::curve_mapping_rgb(prepared, [value[0], value[1], value[2]]);
            let mapped = [mapped_rgb[0], mapped_rgb[1], mapped_rgb[2], value[3]];
            let mut result =
                crate::node_kernels::mix_color("MIX", factor, value, mapped, false, false)
                    .ok_or_else(|| {
                        unsupported_graph_node_state_type("ShaderNodeRGBCurve", "factor")
                    })?;
            result[3] = 1.0;
            if result.iter().any(|component| !component.is_finite()) {
                return Err(evaluation("RGB Curve produced a non-finite color"));
            }
            Ok(json!(result.map(f64::from)))
        }
        _ => Err(evaluation("Curve node type is invalid")),
    }
}

fn numeric_field_in_context(
    field: &Field,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
    node: &str,
) -> Result<f64> {
    field_json_in_context(field, context, index, domain)?
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| evaluation(format!("{node} input is not a finite number")))
}

fn vector_field_in_context(
    field: &Field,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
    node: &str,
) -> Result<DVec3> {
    json_vec3(&field_json_in_context(field, context, index, domain)?)
        .ok_or_else(|| evaluation(format!("{node} input is not a finite vector")))
}

fn integer_field_in_context(
    field: &Field,
    context: FieldEvaluationContext<'_>,
    index: usize,
    domain: &str,
) -> Result<i64> {
    field_json_in_context(field, context, index, domain)?
        .as_i64()
        .ok_or_else(|| evaluation("random ID field must produce an integer"))
}

fn instance_position(transform: &[f64; 16]) -> Result<DVec3> {
    if transform.iter().any(|component| !component.is_finite()) {
        return Err(evaluation("instance transform is not finite"));
    }
    let position = DMat4::from_cols_array(transform).transform_point3(DVec3::ZERO);
    if !position.is_finite() {
        return Err(evaluation("instance transform position is not finite"));
    }
    Ok(position)
}

enum PointAttributeValues<'a> {
    Array(&'a [Value]),
    NativePoint(&'a Map<String, Value>),
    VertexGroup(&'a Map<String, Value>),
    Owned(Vec<Value>),
}

struct PointAttributeColumn<'a> {
    data_type: Option<String>,
    values: PointAttributeValues<'a>,
}

fn point_domain_attribute_columns<'a>(
    mesh: &'a Mesh,
    node_id: &Id,
) -> Result<BTreeMap<String, PointAttributeColumn<'a>>> {
    let mut columns = BTreeMap::new();
    if let Some(entries) = mesh.attributes.get("blender_attributes") {
        let entries = entries
            .as_array()
            .ok_or_else(|| evaluation("Blender attribute list is malformed"))?;
        for entry in entries {
            let name = entry
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| evaluation("point attribute name is malformed"))?;
            if matches!(name, "position" | ".reference_index") || name.starts_with('.') {
                continue;
            }
            let domain = entry
                .get("domain")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !domain.eq_ignore_ascii_case("POINT") {
                return Err(unsupported_instance_feature(
                    node_id,
                    "constraint.geometry_attribute.instances.non_point_attribute",
                    format!("Instance on Points cannot adapt {domain} attribute `{name}` to POINT"),
                ));
            }
            let data_type = entry
                .get("data_type")
                .and_then(Value::as_str)
                .filter(|data_type| !data_type.is_empty())
                .map(normalize_instance_data_type)
                .ok_or_else(|| evaluation("point attribute data type is malformed"))?;
            let values = entry
                .get("values")
                .and_then(Value::as_array)
                .ok_or_else(|| evaluation("point attribute values are malformed"))?;
            if values.len() != mesh.vertices.len() || data_type.is_empty() {
                return Err(evaluation(
                    "point attribute data does not match its domain or type",
                ));
            }
            columns.insert(
                name.to_owned(),
                PointAttributeColumn {
                    data_type: Some(data_type),
                    values: PointAttributeValues::Array(values.as_slice()),
                },
            );
        }
    }

    if mesh
        .attributes
        .get("uv_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .is_some_and(|layers| !layers.is_empty())
    {
        return Err(unsupported_instance_feature(
            node_id,
            "constraint.geometry_attribute.instances.non_point_attribute",
            "Instance on Points cannot adapt CORNER UV layers to POINT",
        ));
    }

    let colors = crate::geom::color_layers::read(mesh);
    for layer in colors.layers {
        if layer.domain != crate::geom::color_layers::ColorDomain::Point {
            return Err(unsupported_instance_feature(
                node_id,
                "constraint.geometry_attribute.instances.non_point_attribute",
                format!(
                    "Instance on Points cannot adapt CORNER color layer `{}` to POINT",
                    layer.name
                ),
            ));
        }
        if layer.values.len() != mesh.vertices.len() {
            return Err(evaluation(
                "point color layer length does not match its domain",
            ));
        }
        let data_type = mesh
            .attributes
            .get("blender_metadata")
            .and_then(|metadata| metadata.get("color_data_types"))
            .and_then(|types| types.get(&layer.name))
            .and_then(Value::as_str)
            .unwrap_or("FLOAT_COLOR");
        columns.insert(
            layer.name,
            PointAttributeColumn {
                data_type: Some(normalize_instance_data_type(data_type)),
                values: PointAttributeValues::Owned(
                    layer.values.into_iter().map(|color| json!(color)).collect(),
                ),
            },
        );
    }

    for (name, attribute) in &mesh.attributes {
        if name == "blender_attributes" || name == "uv_layers" || name == "color_layers" {
            continue;
        }
        if crate::geom::attribute_keys::user_name(name).is_none() {
            continue;
        }
        let domain = attribute.get("domain").and_then(Value::as_str);
        if domain.is_some_and(|domain| {
            matches!(
                domain.to_ascii_uppercase().as_str(),
                "FACE" | "EDGE" | "CORNER"
            )
        }) {
            return Err(unsupported_instance_feature(
                node_id,
                "constraint.geometry_attribute.instances.non_point_attribute",
                format!(
                    "Instance on Points cannot adapt {} attribute `{name}` to POINT",
                    domain.unwrap_or_default()
                ),
            ));
        }
        if domain == Some("point") {
            if name.is_empty() {
                return Err(evaluation("point attribute name is empty"));
            }
            let values = attribute
                .get("values")
                .and_then(Value::as_object)
                .ok_or_else(|| evaluation("native point attribute values are malformed"))?;
            if values.len() != mesh.vertices.len() {
                return Err(evaluation(
                    "native point attribute length does not match its domain",
                ));
            }
            let data_type = attribute
                .get("type")
                .or_else(|| attribute.get("data_type"))
                .and_then(Value::as_str)
                .filter(|data_type| !data_type.is_empty())
                .map(normalize_instance_data_type)
                .ok_or_else(|| evaluation("native point attribute data type is malformed"))?;
            if data_type.is_empty() {
                return Err(evaluation("native point attribute data type is empty"));
            }
            for vertex in &mesh.vertices {
                let _ = native_point_attribute_value(values, vertex.id)?;
            }
            columns.insert(
                name.clone(),
                PointAttributeColumn {
                    data_type: Some(data_type),
                    values: PointAttributeValues::NativePoint(values),
                },
            );
            continue;
        }
        if !matches!(domain, Some("POINT" | "POINTS")) {
            continue;
        }
        if name.is_empty() {
            return Err(evaluation("point attribute name is empty"));
        }
        let values = attribute
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| evaluation("point attribute payload is malformed"))?;
        if values.len() != mesh.vertices.len() {
            return Err(evaluation(
                "point attribute length does not match its domain",
            ));
        }
        let data_type = attribute
            .get("data_type")
            .or_else(|| attribute.get("type"))
            .and_then(Value::as_str)
            .filter(|data_type| !data_type.is_empty())
            .map(normalize_instance_data_type);
        if data_type.as_ref().is_some_and(String::is_empty) {
            return Err(evaluation("point attribute data type is empty"));
        }
        columns.insert(
            name.clone(),
            PointAttributeColumn {
                data_type,
                values: PointAttributeValues::Array(values.as_slice()),
            },
        );
    }

    if let Some(groups) = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
    {
        for (name, weights) in groups {
            if name.is_empty() {
                return Err(evaluation("vertex group name is empty"));
            }
            let weights = weights
                .as_object()
                .ok_or_else(|| evaluation("vertex group weights are malformed"))?;
            columns.insert(
                name.clone(),
                PointAttributeColumn {
                    data_type: Some("FLOAT".to_owned()),
                    values: PointAttributeValues::VertexGroup(weights),
                },
            );
        }
    }
    Ok(columns)
}

fn point_attribute_value(
    column: &PointAttributeColumn<'_>,
    vertex_id: u32,
    index: usize,
) -> Result<Value> {
    match &column.values {
        PointAttributeValues::Array(values) => values
            .get(index)
            .cloned()
            .ok_or_else(|| evaluation("point attribute index is out of range")),
        PointAttributeValues::Owned(values) => values
            .get(index)
            .cloned()
            .ok_or_else(|| evaluation("point attribute index is out of range")),
        PointAttributeValues::NativePoint(values) => {
            Ok(native_point_attribute_value(values, vertex_id)?.clone())
        }
        PointAttributeValues::VertexGroup(values) => {
            Ok(vertex_group_attribute_value(values, vertex_id)
                .cloned()
                .unwrap_or(json!(0.0)))
        }
    }
}

fn native_point_attribute_value(values: &Map<String, Value>, vertex_id: u32) -> Result<&Value> {
    vertex_attribute_value(values, vertex_id)
        .ok_or_else(|| evaluation("native point attribute value is missing"))
}

fn vertex_group_attribute_value(values: &Map<String, Value>, vertex_id: u32) -> Option<&Value> {
    vertex_attribute_value(values, vertex_id)
}

fn vertex_attribute_value(values: &Map<String, Value>, vertex_id: u32) -> Option<&Value> {
    let mut digits = [0_u8; 10];
    let mut number = vertex_id;
    let mut digit_count = 0;
    loop {
        digits[digit_count] = b'0' + (number % 10) as u8;
        digit_count += 1;
        number /= 10;
        if number == 0 {
            break;
        }
    }
    let mut key = [0_u8; 11];
    key[0] = b'v';
    for (target, digit) in key[1..=digit_count]
        .iter_mut()
        .zip(digits[..digit_count].iter().rev())
    {
        *target = *digit;
    }
    let key = std::str::from_utf8(&key[..=digit_count]).ok()?;
    values.get(key)
}
fn normalize_instance_data_type(data_type: &str) -> String {
    match data_type.trim().to_ascii_uppercase().as_str() {
        "FLOAT32" => "FLOAT".to_owned(),
        "INT32" | "INTEGER" => "INT".to_owned(),
        "BOOL" => "BOOLEAN".to_owned(),
        "FLOAT3" | "VECTOR" => "FLOAT_VECTOR".to_owned(),
        "COLOR" => "FLOAT_COLOR".to_owned(),
        "ROTATION" => "QUATERNION".to_owned(),
        "MATRIX" => "FLOAT4X4".to_owned(),
        data_type => data_type.to_owned(),
    }
}

fn infer_instance_data_type(value: &Value) -> Result<String> {
    match value {
        Value::Bool(_) => Ok("BOOLEAN".to_owned()),
        Value::Number(number) if number.is_i64() || number.is_u64() => Ok("INT".to_owned()),
        Value::Number(_) => Ok("FLOAT".to_owned()),
        Value::String(_) => Ok("STRING".to_owned()),
        Value::Array(values) => match values.len() {
            2 => Ok("FLOAT2".to_owned()),
            3 => Ok("FLOAT_VECTOR".to_owned()),
            4 => Ok("FLOAT_COLOR".to_owned()),
            16 => Ok("FLOAT4X4".to_owned()),
            _ => Err(evaluation("cannot infer instance attribute data type")),
        },
        _ => Err(evaluation("cannot infer instance attribute data type")),
    }
}

#[derive(Clone, Copy)]
struct DistributionTriangle {
    positions: [Vec3; 3],
    corners: [usize; 3],
    area: f32,
}

#[derive(Clone, Copy)]
struct DistributedCandidate {
    position: Vec3,
    barycentric: Vec3,
    triangle_index: u32,
    corners: [usize; 3],
}

fn distribute_points_mesh(
    node_id: &Id,
    surface: &Mesh,
    method: &str,
    density: &Field,
    density_factor: &Field,
    selection: &Field,
    seed: i64,
    density_max: f32,
    minimum_distance: f32,
) -> Result<Mesh> {
    if !minimum_distance.is_finite() || !density_max.is_finite() {
        return Err(evaluation("point distribution settings must be finite"));
    }
    if method == "POISSON" && density_max <= 0.0 {
        return Ok(Mesh::new());
    }

    let corner_count = domain_element_count(surface, "CORNER")?;
    let mut corner_densities = vec![0.0; corner_count];
    let constant_random_density = if method == "RANDOM"
        && matches!(
            (selection, density),
            (Field::Constant(_), Field::Constant(_))
        ) {
        let value = field_json(density, surface, 0, "CORNER")?
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| evaluation("point density field must be finite and numeric"))?
            as f32;
        if !value.is_finite() {
            return Err(evaluation("point density field must be finite"));
        }
        Some(value.max(0.0))
    } else {
        None
    };
    if method == "RANDOM" && constant_random_density.is_none() {
        for (corner, density_value) in corner_densities.iter_mut().enumerate() {
            let selected = field_bool(selection, surface, corner, "CORNER", false)?;
            if selected {
                let value = field_json(density, surface, corner, "CORNER")?
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| evaluation("point density field must be finite and numeric"))?
                    as f32;
                if !value.is_finite() {
                    return Err(evaluation("point density field must be finite"));
                }
                *density_value = value.max(0.0);
            }
        }
    }

    let triangles = surface
        .tessellate()
        .map_err(|error| mesh_evaluation_error(&error))?;
    let positions_by_id: HashMap<_, _> = surface
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co.as_vec3()))
        .collect();
    let mut triangle_data = Vec::with_capacity(triangles.len());
    let mut triangle_offset = 0_usize;
    let mut corner_offset = 0_usize;
    let mut local_corner_by_vertex = HashMap::new();
    for face in &surface.faces {
        let triangle_count = face.vertices.len().saturating_sub(2);
        local_corner_by_vertex.clear();
        for (local_corner, vertex_id) in face.vertices.iter().enumerate() {
            local_corner_by_vertex.insert(*vertex_id, local_corner);
        }
        for _ in 0..triangle_count {
            let triangle = *triangles
                .get(triangle_offset)
                .ok_or_else(|| evaluation("surface triangulation result is incomplete"))?;
            let mut positions = [Vec3::ZERO; 3];
            let mut corners = [0_usize; 3];
            for vertex in 0..3 {
                let vertex_id = triangle[vertex];
                positions[vertex] = *positions_by_id
                    .get(&vertex_id)
                    .ok_or_else(|| evaluation("triangle vertex missing"))?;
                let local_corner = local_corner_by_vertex
                    .get(&vertex_id)
                    .copied()
                    .ok_or_else(|| evaluation("triangle corner missing"))?;
                corners[vertex] = corner_offset
                    .checked_add(local_corner)
                    .ok_or_else(|| evaluation("corner index exceeds supported range"))?;
            }
            let area = (positions[0] - positions[1])
                .cross(positions[1] - positions[2])
                .length()
                * 0.5;
            triangle_data.push(DistributionTriangle {
                positions,
                corners,
                area,
            });
            triangle_offset += 1;
        }
        corner_offset = corner_offset
            .checked_add(face.vertices.len())
            .ok_or_else(|| evaluation("corner count exceeds the supported range"))?;
    }
    if triangle_offset != triangles.len() || corner_offset != corner_count {
        return Err(evaluation(
            "surface triangulation result does not match face domains",
        ));
    }

    let seed = (seed as i32 as u32).wrapping_mul(5_383_843);
    let mut candidates = Vec::new();
    for (triangle_index, triangle) in triangle_data.iter().enumerate() {
        let density = if method == "POISSON" {
            density_max
        } else if let Some(constant_density) = constant_random_density {
            constant_density
        } else {
            (corner_densities[triangle.corners[0]]
                + corner_densities[triangle.corners[1]]
                + corner_densities[triangle.corners[2]])
                / 3.0
        };
        let weighted_count = triangle.area * density;
        if !weighted_count.is_finite() || weighted_count < 0.0 {
            return Err(evaluation("point density and triangle area must be finite"));
        }
        let triangle_index_u32 = u32::try_from(triangle_index)
            .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "too many surface triangles"))?;
        let triangle_seed = blender_noise_hash_2(triangle_index_u32, seed);
        let mut count_rng = BlenderRng::seed(triangle_seed);
        let count = blender_round_probabilistic(weighted_count, &mut count_rng)?;
        let next_len = candidates
            .len()
            .checked_add(count)
            .filter(|count| *count <= MAX_GENERATED_POINTS)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "distributed point count exceeds limit",
                )
            })?;
        candidates.reserve(next_len.saturating_sub(candidates.len()));

        let mut sample_rng = BlenderRng::seed(triangle_seed);
        sample_rng.skip(1);
        for _ in 0..count {
            let mut first = sample_rng.get_float();
            let mut second = sample_rng.get_float();
            if first + second > 1.0 {
                first = 1.0 - first;
                second = 1.0 - second;
            }
            let barycentric = Vec3::new(first, second, 1.0 - first - second);
            let position = triangle.positions[0] * barycentric.x
                + triangle.positions[1] * barycentric.y
                + triangle.positions[2] * barycentric.z;
            candidates.push(DistributedCandidate {
                position,
                barycentric,
                triangle_index: triangle_index_u32,
                corners: triangle.corners,
            });
        }
    }
    if method == "POISSON" {
        if candidates.is_empty() {
            return Ok(Mesh::new());
        }
        for (corner, density_value) in corner_densities.iter_mut().enumerate() {
            let selected = field_bool(selection, surface, corner, "CORNER", false)?;
            if selected {
                let value = field_json(density_factor, surface, corner, "CORNER")?
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| evaluation("Density Factor field must be finite and numeric"))?
                    as f32;
                if !value.is_finite() {
                    return Err(evaluation("Density Factor field must be finite"));
                }
                *density_value = value.max(0.0);
            }
        }
    }

    let mut elimination_mask = if method == "POISSON" && minimum_distance > 0.0 {
        eliminate_close_candidates(&candidates, minimum_distance)?
    } else {
        vec![false; candidates.len()]
    };
    if method == "POISSON" {
        for (index, candidate) in candidates.iter().enumerate() {
            if elimination_mask[index] {
                continue;
            }
            let [first, second, third] = candidate.corners.map(|corner| corner_densities[corner]);
            let probability = first * candidate.barycentric.x
                + second * candidate.barycentric.y
                + third * candidate.barycentric.z;
            if blender_noise_hash_float3_to_float(candidate.barycentric) > probability {
                elimination_mask[index] = true;
            }
        }
    }

    if method == "POISSON" {
        for index in (0..candidates.len()).rev() {
            if elimination_mask[index] {
                candidates.swap_remove(index);
            }
        }
    }
    let point_count = candidates.len();
    if point_count > 0
        && let Some(attribute_name) = unpropagated_distribute_attribute(surface)
    {
        return Err(unsupported_graph_feature(
            node_id,
            "graph.node.distribute_points.attribute_propagation",
            format!(
                "Distribute Points on Faces cannot propagate input attribute `{attribute_name}`"
            ),
        ));
    }
    let mut points = Mesh::new();
    let mut ids = Vec::with_capacity(point_count);
    let mut radii = Vec::with_capacity(point_count);
    for candidate in &candidates {
        points
            .insert_vertex(candidate.position.as_dvec3())
            .map_err(|error| mesh_evaluation_error(&error))?;
        ids.push(i64::from(blender_noise_hash_2(
            blender_noise_hash_float3(candidate.barycentric),
            candidate.triangle_index,
        ) as i32));
        radii.push(0.05);
    }
    points.attributes.insert(
        "id".to_owned(),
        json!({"domain":"POINT","data_type":"INT","data":ids}),
    );
    points.attributes.insert(
        "radius".to_owned(),
        json!({"domain":"POINT","data_type":"FLOAT","data":radii}),
    );
    Ok(points)
}

fn unpropagated_distribute_attribute(mesh: &Mesh) -> Option<String> {
    if let Some(entries) = mesh
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
    {
        for entry in entries {
            let name = entry.get("name").and_then(Value::as_str)?;
            let domain = entry
                .get("domain")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if name.starts_with('.')
                || matches!(name, "position" | "id" | "radius")
                || domain.eq_ignore_ascii_case("EDGE")
                || (crate::geom::attribute_keys::is_internal_key(name)
                    && !matches!(name, "uv_layers" | "color_layers" | "vertex_groups"))
            {
                continue;
            }
            return Some(name.to_owned());
        }
    }
    for (name, attribute) in &mesh.attributes {
        if (crate::geom::attribute_keys::is_internal_key(name)
            && !matches!(
                name.as_str(),
                "uv_layers" | "color_layers" | "vertex_groups"
            ))
            || name.starts_with('.')
            || matches!(name.as_str(), "position" | "id" | "radius")
            || attribute
                .get("domain")
                .and_then(Value::as_str)
                .is_some_and(|domain| domain.eq_ignore_ascii_case("EDGE"))
        {
            continue;
        }
        let values = attribute.get("data").or_else(|| attribute.get("values"));
        if values
            .and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty())
            || values
                .and_then(Value::as_object)
                .is_some_and(|values| !values.is_empty())
        {
            return Some(name.clone());
        }
    }
    for store in ["uv_layers", "color_layers"] {
        if mesh
            .attributes
            .get(store)
            .and_then(|store| store.get("layers"))
            .and_then(Value::as_array)
            .is_some_and(|layers| !layers.is_empty())
        {
            return Some(store.to_owned());
        }
    }
    mesh.attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .filter(|groups| !groups.is_empty())
        .map(|_| "vertex_groups".to_owned())
}

fn blender_round_probabilistic(value: f32, rng: &mut BlenderRng) -> Result<usize> {
    if !value.is_finite() || value < 0.0 || value > MAX_GENERATED_POINTS as f32 + 1.0 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "distributed point count exceeds limit",
        ));
    }
    let integer = value as usize;
    let round_up = value.fract() > rng.get_float();
    let count = integer + usize::from(round_up);
    if count > MAX_GENERATED_POINTS {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "distributed point count exceeds limit",
        ));
    }
    Ok(count)
}

fn blender_noise_hash_2(first: u32, second: u32) -> u32 {
    let seed = 0xdead_beef_u32.wrapping_add((2 << 2) + 13);
    blender_noise_hash_finalize(seed.wrapping_add(first), seed.wrapping_add(second), seed)
}

fn blender_noise_hash_to_float_2(first: u32, second: u32) -> f32 {
    blender_noise_hash_2(first, second) as f32 / u32::MAX as f32
}
fn blender_noise_hash_3(first: u32, second: u32, third: u32) -> u32 {
    let seed = 0xdead_beef_u32.wrapping_add((3 << 2) + 13);
    blender_noise_hash_finalize(
        seed.wrapping_add(first),
        seed.wrapping_add(second),
        seed.wrapping_add(third),
    )
}

fn blender_noise_hash_to_float_3(first: u32, second: u32, third: u32) -> f32 {
    blender_noise_hash_3(first, second, third) as f32 / u32::MAX as f32
}

fn blender_noise_hash_float3(value: Vec3) -> u32 {
    let seed = 0xdead_beef_u32.wrapping_add((3 << 2) + 13);
    blender_noise_hash_finalize(
        seed.wrapping_add(value.x.to_bits()),
        seed.wrapping_add(value.y.to_bits()),
        seed.wrapping_add(value.z.to_bits()),
    )
}

fn blender_noise_hash_float3_to_float(value: Vec3) -> f32 {
    blender_noise_hash_float3(value) as f32 / u32::MAX as f32
}

fn blender_noise_hash_finalize(mut first: u32, mut second: u32, mut third: u32) -> u32 {
    third ^= second;
    third = third.wrapping_sub(second.rotate_left(14));
    first ^= third;
    first = first.wrapping_sub(third.rotate_left(11));
    second ^= first;
    second = second.wrapping_sub(first.rotate_left(25));
    third ^= second;
    third = third.wrapping_sub(second.rotate_left(16));
    first ^= third;
    first = first.wrapping_sub(third.rotate_left(4));
    second ^= first;
    second = second.wrapping_sub(first.rotate_left(14));
    third ^= second;
    third.wrapping_sub(second.rotate_left(24))
}

fn eliminate_close_candidates(
    candidates: &[DistributedCandidate],
    minimum_distance: f32,
) -> Result<Vec<bool>> {
    let mut elimination_mask = vec![false; candidates.len()];
    let mut grid = HashMap::<[i64; 3], Vec<usize>>::with_capacity(candidates.len());
    let cell_size = f64::from(minimum_distance);
    let minimum_squared = minimum_distance * minimum_distance;
    for (index, candidate) in candidates.iter().enumerate() {
        let cell = poisson_cell(candidate.position.as_dvec3(), cell_size)?;
        grid.entry(cell).or_default().push(index);
    }
    for (index, candidate) in candidates.iter().enumerate() {
        if elimination_mask[index] {
            continue;
        }
        let cell = poisson_cell(candidate.position.as_dvec3(), cell_size)?;
        for x_offset in -2_i64..=2 {
            let Some(x) = cell[0].checked_add(x_offset) else {
                continue;
            };
            for y_offset in -2_i64..=2 {
                let Some(y) = cell[1].checked_add(y_offset) else {
                    continue;
                };
                for z_offset in -2_i64..=2 {
                    let Some(z) = cell[2].checked_add(z_offset) else {
                        continue;
                    };
                    if let Some(neighbors) = grid.get(&[x, y, z]) {
                        for &neighbor in neighbors {
                            if neighbor != index
                                && candidate
                                    .position
                                    .distance_squared(candidates[neighbor].position)
                                    <= minimum_squared
                            {
                                elimination_mask[neighbor] = true;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(elimination_mask)
}

fn poisson_cell(point: DVec3, cell_size: f64) -> Result<[i64; 3]> {
    if !cell_size.is_finite() || cell_size <= 0.0 {
        return Err(evaluation(
            "point distance is outside supported numeric range",
        ));
    }
    let coordinates = point / cell_size;
    if !coordinates.is_finite() {
        return Err(evaluation(
            "point distribution coordinate exceeds supported range",
        ));
    }
    Ok([
        floor_to_i64(coordinates.x)?,
        floor_to_i64(coordinates.y)?,
        floor_to_i64(coordinates.z)?,
    ])
}

fn json_is_finite(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.as_f64().is_none_or(f64::is_finite),
        Value::Array(values) => values.iter().all(json_is_finite),
        _ => true,
    }
}

fn field_vec3(field: &Field, mesh: &Mesh, index: usize, fallback: DVec3) -> Result<DVec3> {
    field_vec3_in_domain(field, mesh, index, "POINT", fallback)
}

fn field_vec3_in_domain(
    field: &Field,
    mesh: &Mesh,
    index: usize,
    domain: &str,
    fallback: DVec3,
) -> Result<DVec3> {
    let value = field_json(field, mesh, index, domain)?;
    if value.is_null() {
        Ok(fallback)
    } else {
        json_vec3(&value).ok_or_else(|| evaluation("vector field did not produce a finite vec3"))
    }
}

fn field_bool(
    field: &Field,
    mesh: &Mesh,
    index: usize,
    domain: &str,
    fallback: bool,
) -> Result<bool> {
    Ok(field_json(field, mesh, index, domain)?
        .as_bool()
        .unwrap_or(fallback))
}

fn field_vec3_for_instance(
    field: &Field,
    attributes: &BTreeMap<String, InstanceAttributeColumn>,
    transform: &[f64; 16],
    index: usize,
    instance_count: usize,
    fallback: DVec3,
) -> Result<DVec3> {
    let value = field_json_for_instance(field, attributes, transform, index, instance_count)?;
    if value.is_null() {
        Ok(fallback)
    } else {
        json_vec3(&value).ok_or_else(|| evaluation("vector field did not produce a finite vec3"))
    }
}

fn field_bool_for_instance(
    field: &Field,
    attributes: &BTreeMap<String, InstanceAttributeColumn>,
    transform: &[f64; 16],
    index: usize,
    instance_count: usize,
    fallback: bool,
) -> Result<bool> {
    Ok(
        field_json_for_instance(field, attributes, transform, index, instance_count)?
            .as_bool()
            .unwrap_or(fallback),
    )
}

fn field_bool_in_instances(
    field: &Field,
    instances: &EvaluatedInstances,
    index: usize,
    fallback: bool,
) -> Result<bool> {
    let transform = instances
        .transforms
        .get(index)
        .ok_or_else(|| evaluation("instance transform index is out of range"))?;
    field_bool_for_instance(
        field,
        &instances.attributes,
        transform,
        index,
        instances.len(),
        fallback,
    )
}

fn domain_element_count(mesh: &Mesh, domain: &str) -> Result<usize> {
    match domain {
        "POINT" | "POINTS" => Ok(mesh.vertices.len()),
        "EDGE" => Ok(mesh.edges.len()),
        "FACE" => Ok(mesh.faces.len()),
        "CORNER" => mesh.faces.iter().try_fold(0_usize, |count, face| {
            count
                .checked_add(face.vertices.len())
                .ok_or_else(|| evaluation("corner count exceeds the supported range"))
        }),
        _ => Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            format!("unsupported Mesh to Points domain `{domain}`"),
            json!({"domain":domain,"supported":["POINT","EDGE","FACE","CORNER"]}),
        )),
    }
}

fn domain_position(mesh: &Mesh, index: usize, domain: &str) -> Result<DVec3> {
    match domain {
        "POINT" | "POINTS" | "INSTANCE" => mesh
            .vertices
            .get(index)
            .map(|vertex| vertex.co)
            .ok_or_else(|| evaluation("point field index is out of range")),
        "EDGE" => {
            let edge = mesh
                .edges
                .get(index)
                .ok_or_else(|| evaluation("edge field index is out of range"))?;
            let first = mesh
                .vertex(edge.vertices[0])
                .ok_or_else(|| evaluation("edge vertex is missing"))?;
            let second = mesh
                .vertex(edge.vertices[1])
                .ok_or_else(|| evaluation("edge vertex is missing"))?;
            Ok((first.co + second.co) * 0.5)
        }
        "FACE" => {
            let face = mesh
                .faces
                .get(index)
                .ok_or_else(|| evaluation("face field index is out of range"))?;
            let count = u32::try_from(face.vertices.len())
                .map_err(|_| evaluation("face vertex count exceeds the supported range"))?;
            if count == 0 {
                return Err(evaluation("face has no vertices"));
            }
            let mut center = DVec3::ZERO;
            for vertex_id in &face.vertices {
                center += mesh
                    .vertex(*vertex_id)
                    .ok_or_else(|| evaluation("face vertex is missing"))?
                    .co;
            }
            Ok(center / f64::from(count))
        }
        "CORNER" => {
            let vertex_id = corner_vertex_id(mesh, index)?;
            mesh.vertex(vertex_id)
                .map(|vertex| vertex.co)
                .ok_or_else(|| evaluation("corner vertex is missing"))
        }
        _ => Err(evaluation(format!("unsupported field domain `{domain}`"))),
    }
}

fn corner_vertex_id(mesh: &Mesh, index: usize) -> Result<u32> {
    let mut corner_index = index;
    for face in &mesh.faces {
        if let Some(vertex_id) = face.vertices.get(corner_index) {
            return Ok(*vertex_id);
        }
        corner_index = corner_index.saturating_sub(face.vertices.len());
    }
    Err(evaluation("corner field index is out of range"))
}

fn normal_at(mesh: &Mesh, index: usize, domain: &str) -> Result<DVec3> {
    match domain {
        "FACE" => {
            let face = mesh
                .faces
                .get(index)
                .ok_or_else(|| evaluation("face normal index is out of range"))?;
            face_normal(mesh, &face.vertices)
        }
        "EDGE" => {
            let edge = mesh
                .edges
                .get(index)
                .ok_or_else(|| evaluation("edge normal index is out of range"))?;
            let first = point_normal(mesh, edge.vertices[0])?;
            let second = point_normal(mesh, edge.vertices[1])?;
            Ok((first + second).normalize_or_zero())
        }
        "CORNER" => point_normal(mesh, corner_vertex_id(mesh, index)?),
        "POINT" | "POINTS" | "INSTANCE" => {
            let vertex = mesh
                .vertices
                .get(index)
                .ok_or_else(|| evaluation("point normal index is out of range"))?;
            point_normal(mesh, vertex.id)
        }
        _ => Err(evaluation(format!(
            "unsupported normal field domain `{domain}`"
        ))),
    }
}

fn point_normal(mesh: &Mesh, vertex_id: u32) -> Result<DVec3> {
    let mut normal = DVec3::ZERO;
    for face in &mesh.faces {
        if face.vertices.contains(&vertex_id) {
            normal += face_normal(mesh, &face.vertices)?;
        }
    }
    Ok(if normal.length_squared() > 0.0 {
        normal.normalize()
    } else {
        DVec3::Z
    })
}

fn face_normal(mesh: &Mesh, vertex_ids: &[u32]) -> Result<DVec3> {
    let mut positions = vertex_ids
        .iter()
        .filter_map(|id| mesh.vertex(*id).map(|vertex| vertex.co));
    let first = positions
        .next()
        .ok_or_else(|| evaluation("face has no vertices"))?;
    let second = positions
        .next()
        .ok_or_else(|| evaluation("face has fewer than three vertices"))?;
    let third = positions
        .next()
        .ok_or_else(|| evaluation("face has fewer than three vertices"))?;
    let normal = (second - first).cross(third - first);
    Ok(if normal.length_squared() > 0.0 {
        normal.normalize()
    } else {
        DVec3::Z
    })
}

fn merge_meshes(meshes: &[Mesh]) -> Result<Mesh> {
    let vertex_count = meshes
        .iter()
        .try_fold(0_usize, |sum, mesh| sum.checked_add(mesh.vertices.len()))
        .ok_or_else(|| evaluation("joined geometry vertex count overflow"))?;
    if u32::try_from(vertex_count).is_err() {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "joined geometry exceeds vertex ID range",
        ));
    }
    let tracks_origindex = meshes.iter().any(|mesh| origindex_values(mesh).is_some());
    let mut original_vertices = tracks_origindex.then(|| Vec::with_capacity(vertex_count));
    let mut result = Mesh::new();
    for mesh in meshes {
        if let Some(output_indices) = &mut original_vertices {
            match origindex_values(mesh) {
                Some(indices) => output_indices.extend_from_slice(&indices),
                None => {
                    output_indices.extend(std::iter::repeat_n(ORIGINDEX_NONE, mesh.vertices.len()));
                }
            }
        }
        let mut mapping = HashMap::with_capacity(mesh.vertices.len());
        for vertex in &mesh.vertices {
            let id = result
                .insert_vertex(vertex.co)
                .map_err(|error| mesh_evaluation_error(&error))?;
            mapping.insert(vertex.id, id);
        }
        for face in &mesh.faces {
            let vertices = face
                .vertices
                .iter()
                .map(|id| {
                    mapping
                        .get(id)
                        .copied()
                        .ok_or_else(|| evaluation("joined face references a missing vertex"))
                })
                .collect::<Result<Vec<_>>>()?;
            result
                .insert_face(vertices, face.material_index)
                .map_err(|error| mesh_evaluation_error(&error))?;
        }
        for edge in &mesh.edges {
            let first = mapping
                .get(&edge.vertices[0])
                .copied()
                .ok_or_else(|| evaluation("joined edge references a missing vertex"))?;
            let second = mapping
                .get(&edge.vertices[1])
                .copied()
                .ok_or_else(|| evaluation("joined edge references a missing vertex"))?;
            if !result.edges.iter().any(|existing| {
                (existing.vertices[0] == first && existing.vertices[1] == second)
                    || (existing.vertices[0] == second && existing.vertices[1] == first)
            }) {
                result
                    .insert_edge([first, second])
                    .map_err(|error| mesh_evaluation_error(&error))?;
            }
        }
    }
    if let Some(vertices) = original_vertices {
        set_origindex(&mut result, &vertices);
    }
    Ok(result)
}

impl NodeBakeValue {
    fn into_eval_value(self, node_id: &Id, require_instance_attributes: bool) -> Result<EvalValue> {
        match self {
            Self::Geometry {
                mesh,
                instance_paths,
                instance_columns,
                ..
            } => {
                if require_instance_attributes
                    && instance_columns.is_none()
                    && !instance_paths.is_empty()
                {
                    return Err(legacy_instance_columns_missing(node_id));
                }
                let instance_attributes = instance_columns
                    .map(serde_json::from_value::<EvaluatedInstances>)
                    .transpose()
                    .map_err(|error| {
                        evaluation(format!("baked instance columns are invalid: {error}"))
                    })?
                    .unwrap_or_default();
                Ok(EvalValue::Geometry(GeometryValue {
                    mesh,
                    instance_paths,
                    instance_attributes,
                }))
            }
            Self::Scalar { value } => Ok(EvalValue::Scalar(value)),
            Self::Field { attribute_name } => Ok(EvalValue::Field(Field::Attribute {
                name: attribute_name,
                data_type: None,
            })),
        }
    }
}

fn read_bake_value(
    record: &NodeBakeRecord,
    frame: f64,
    socket: &str,
) -> Result<Option<NodeBakeValue>> {
    let next = record.frames.partition_point(|baked| baked.frame < frame);
    if next < record.frames.len() && same_bake_frame(record.frames[next].frame, frame) {
        return record.frames[next]
            .values
            .get(socket)
            .cloned()
            .map(Some)
            .ok_or_else(|| evaluation("Bake record is missing a socket value"));
    }
    if next == 0 {
        return record.frames[0]
            .values
            .get(socket)
            .cloned()
            .map(Some)
            .ok_or_else(|| evaluation("Bake record is missing a socket value"));
    }
    if next == record.frames.len() {
        return record.frames[next - 1]
            .values
            .get(socket)
            .cloned()
            .map(Some)
            .ok_or_else(|| evaluation("Bake record is missing a socket value"));
    }
    let previous = &record.frames[next - 1];
    let following = &record.frames[next];
    let first = previous
        .values
        .get(socket)
        .ok_or_else(|| evaluation("Bake record is missing a socket value"))?;
    let second = following
        .values
        .get(socket)
        .ok_or_else(|| evaluation("Bake record is missing a socket value"))?;
    let factor = (frame - previous.frame) / (following.frame - previous.frame);
    Ok(Some(mix_bake_values(first, second, factor)))
}

fn mix_bake_values(first: &NodeBakeValue, second: &NodeBakeValue, factor: f64) -> NodeBakeValue {
    match (first, second) {
        (
            NodeBakeValue::Geometry {
                mesh: first_mesh,
                instance_paths,
                instance_columns,
                material_refs,
            },
            NodeBakeValue::Geometry {
                mesh: second_mesh, ..
            },
        ) if same_mesh_topology(first_mesh, second_mesh) => {
            let mut mesh = first_mesh.clone();
            for (vertex, next) in mesh.vertices.iter_mut().zip(&second_mesh.vertices) {
                vertex.co = vertex.co.lerp(next.co, factor);
            }
            mix_mesh_attributes(&mut mesh, second_mesh, factor);
            NodeBakeValue::Geometry {
                mesh,
                instance_paths: instance_paths.clone(),
                instance_columns: instance_columns.clone(),
                material_refs: material_refs.clone(),
            }
        }
        (NodeBakeValue::Scalar { value: first }, NodeBakeValue::Scalar { value: second }) => {
            NodeBakeValue::Scalar {
                value: mix_json_value(first, second, factor),
            }
        }
        (
            NodeBakeValue::Field {
                attribute_name: first,
            },
            NodeBakeValue::Field {
                attribute_name: second,
            },
        ) if first == second => NodeBakeValue::Field {
            attribute_name: first.clone(),
        },
        _ => first.clone(),
    }
}

fn same_mesh_topology(first: &Mesh, second: &Mesh) -> bool {
    first
        .vertices
        .iter()
        .map(|vertex| vertex.id)
        .eq(second.vertices.iter().map(|vertex| vertex.id))
        && first.edges == second.edges
        && first.faces.len() == second.faces.len()
        && first
            .faces
            .iter()
            .zip(&second.faces)
            .all(|(first, second)| first.id == second.id && first.vertices == second.vertices)
}

fn mix_mesh_attributes(first: &mut Mesh, second: &Mesh, factor: f64) {
    for (name, first_attribute) in &mut first.attributes {
        // Blender never blends the persistent `id` attribute (mix_geometries.cc, `names.remove("id")`).
        if name == "id" {
            continue;
        }
        let Some(second_attribute) = second.attributes.get(name) else {
            continue;
        };
        if first_attribute.get("domain") != second_attribute.get("domain")
            || first_attribute.get("type") != second_attribute.get("type")
        {
            continue;
        }
        if let (Some(first_data), Some(second_data)) = (
            first_attribute
                .get_mut("data")
                .and_then(Value::as_array_mut),
            second_attribute.get("data").and_then(Value::as_array),
        ) {
            if first_data.len() == second_data.len() {
                for (first, second) in first_data.iter_mut().zip(second_data) {
                    *first = mix_json_value(first, second, factor);
                }
            }
            continue;
        }
        if !matches!(
            first_attribute.get("domain").and_then(Value::as_str),
            Some("point" | "vertices" | "POINT" | "POINTS")
        ) {
            continue;
        }
        let (Some(first_values), Some(second_values)) = (
            first_attribute
                .get_mut("values")
                .and_then(Value::as_object_mut),
            second_attribute.get("values").and_then(Value::as_object),
        ) else {
            continue;
        };
        for (key, value) in first_values {
            if let Some(next) = second_values.get(key) {
                *value = mix_json_value(value, next, factor);
            }
        }
    }
}

fn mix_json_value(first: &Value, second: &Value, factor: f64) -> Value {
    match (first, second) {
        (Value::Bool(first), Value::Bool(second)) => {
            // Blender's bool `mix2`: true once the blend reaches one half (BKE_attribute_math.hh).
            let blend = (1.0 - factor) * f64::from(u8::from(*first))
                + factor * f64::from(u8::from(*second));
            json!(blend >= 0.5)
        }
        (Value::Number(first), Value::Number(second)) => {
            if let (Some(first), Some(second)) = (first.as_i64(), second.as_i64()) {
                // Blender's int `mix2` rounds the blend, so integers never become fractions.
                let blend = (1.0 - factor) * first as f64 + factor * second as f64;
                return json!(blend.round() as i64);
            }
            let first = first.as_f64().unwrap_or(0.0);
            let second = second.as_f64().unwrap_or(0.0);
            json!(first + (second - first) * factor)
        }
        (Value::Array(first), Value::Array(second)) if first.len() == second.len() => Value::Array(
            first
                .iter()
                .zip(second)
                .map(|(first, second)| mix_json_value(first, second, factor))
                .collect(),
        ),
        _ => first.clone(),
    }
}

fn mesh_evaluation_error(error: &MeshError) -> PotError {
    PotError::new(ErrorCode::EvaluationFailed, error.to_string())
}

fn evaluation(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::EvaluationFailed, message)
}

fn subdivision_uv_smooth(value: &str) -> Result<&'static str> {
    match value {
        "NONE" | "None" => Ok("NONE"),
        "SMOOTH_ALL" | "All" => Ok("SMOOTH_ALL"),
        "PRESERVE_CORNERS" | "Keep Corners" => Ok("PRESERVE_CORNERS"),
        "PRESERVE_CORNERS_AND_JUNCTIONS" | "Keep Corners and Junctions" => {
            Ok("PRESERVE_CORNERS_AND_JUNCTIONS")
        }
        "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE" | "Keep Corners, Junctions and Concave" => {
            Ok("PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE")
        }
        "PRESERVE_BOUNDARIES" | "Keep Boundaries" => Ok("PRESERVE_BOUNDARIES"),
        _ => Err(evaluation(format!(
            "unknown Subdivision Surface UV Smooth mode `{value}`"
        ))),
    }
}

fn subdivision_boundary_smooth(value: &str) -> Result<&'static str> {
    match value {
        "ALL" | "All" => Ok("ALL"),
        "PRESERVE_CORNERS" | "Keep Corners" => Ok("PRESERVE_CORNERS"),
        _ => Err(evaluation(format!(
            "unknown Subdivision Surface Boundary Smooth mode `{value}`"
        ))),
    }
}

fn named_attribute_data_type(node_id: &Id, node: &GraphNode, feature_id: &str) -> Result<String> {
    let data_type = node
        .properties
        .get("data_type")
        .and_then(Value::as_str)
        .unwrap_or("FLOAT")
        .to_ascii_uppercase();
    if !matches!(
        data_type.as_str(),
        "FLOAT" | "FLOAT_VECTOR" | "INT" | "BOOLEAN" | "QUATERNION" | "FLOAT_COLOR"
    ) {
        return Err(unsupported_graph_feature(
            node_id,
            feature_id,
            format!("Geometry Nodes attribute data type `{data_type}` is not implemented"),
        ));
    }
    Ok(data_type)
}

fn named_attribute_zero(data_type: &str) -> Value {
    match data_type {
        "FLOAT" => json!(0.0),
        "FLOAT_VECTOR" => json!([0.0, 0.0, 0.0]),
        "INT" => json!(0),
        "BOOLEAN" => json!(false),
        "QUATERNION" => json!([0.0, 0.0, 0.0, 1.0]),
        "FLOAT_COLOR" => json!([0.0, 0.0, 0.0, 0.0]),
        _ => Value::Null,
    }
}
fn named_attribute_input_default(data_type: &str) -> Value {
    if data_type == "FLOAT_COLOR" {
        json!([0.8, 0.8, 0.8, 1.0])
    } else {
        named_attribute_zero(data_type)
    }
}

fn named_attribute_value(value: &Value, data_type: &str) -> Result<Value> {
    match data_type {
        "FLOAT" => {
            let value = value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| evaluation("FLOAT attribute value must be finite and numeric"))?
                as f32;
            if !value.is_finite() {
                return Err(evaluation("FLOAT attribute value exceeds float32 range"));
            }
            Ok(json!(value))
        }
        "FLOAT_VECTOR" => {
            let value = json_vec3(value)
                .ok_or_else(|| evaluation("FLOAT_VECTOR attribute value must be a finite vec3"))?
                .as_vec3();
            if !value.is_finite() {
                return Err(evaluation(
                    "FLOAT_VECTOR attribute value exceeds float32 range",
                ));
            }
            Ok(vector_json(value.as_dvec3()))
        }
        "INT" => value
            .as_i64()
            .and_then(|value| i32::try_from(value).ok())
            .map(|value| json!(value))
            .ok_or_else(|| evaluation("INT attribute value must be a signed 32-bit integer")),
        "BOOLEAN" => value
            .as_bool()
            .map(Value::Bool)
            .ok_or_else(|| evaluation("BOOLEAN attribute value must be boolean")),
        "FLOAT_COLOR" => {
            let value = json_color4(value)
                .ok_or_else(|| evaluation("FLOAT_COLOR attribute value must be a finite vec4"))?;
            Ok(json!(value.map(f64::from)))
        }
        "QUATERNION" => {
            let values = value
                .as_array()
                .filter(|values| values.len() == 4)
                .ok_or_else(|| {
                    evaluation("QUATERNION attribute value must be an xyzw quaternion")
                })?;
            let component = |index: usize| {
                values[index]
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .map(|value| value as f32)
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| {
                        evaluation(
                            "QUATERNION attribute value must be finite and within float32 range",
                        )
                    })
            };
            let values = [component(0)?, component(1)?, component(2)?, component(3)?];
            Ok(json!(values.map(f64::from)))
        }
        _ => Err(evaluation("unsupported Geometry Nodes attribute data type")),
    }
}
fn validate_named_attribute_conversion(
    requested_type: Option<&str>,
    attribute: &Value,
    values: &[Value],
) -> Result<()> {
    let Some(requested_type) = requested_type else {
        return Ok(());
    };
    let actual_type = attribute
        .get("data_type")
        .or_else(|| attribute.get("type"))
        .and_then(Value::as_str)
        .map(normalize_instance_data_type)
        .or_else(|| {
            values
                .first()
                .and_then(|value| infer_instance_data_type(value).ok())
        });
    if actual_type
        .as_deref()
        .is_some_and(|actual| !named_attribute_type_compatible(requested_type, actual))
    {
        return Err(unsupported_instance_field_feature(
            "graph.node.input_named_attribute.data_type",
            format!(
                "Named Attribute conversion from {} to {requested_type} is not implemented",
                actual_type.unwrap_or_default()
            ),
        ));
    }
    Ok(())
}

fn named_attribute_type_compatible(requested: &str, actual: &str) -> bool {
    requested == actual || (requested == "FLOAT" && actual == "INT")
}

fn unsupported_graph_feature(
    node_id: &Id,
    feature_id: &str,
    detail: impl Into<String>,
) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        detail,
        json!({"feature_id": feature_id, "node_id": node_id}),
    )
}
fn unsupported_instance_feature(
    node_id: &Id,
    feature_id: &str,
    detail: impl Into<String>,
) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        detail,
        json!({"feature_id": feature_id, "node_id": node_id}),
    )
}

fn legacy_instance_columns_missing(node_id: &Id) -> PotError {
    unsupported_instance_feature(
        node_id,
        "graph.bake.legacy_instance_columns_missing",
        "legacy geometry bake has instance paths but no instance attribute columns",
    )
}

fn unsupported_instance_field_feature(feature_id: &str, detail: impl Into<String>) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        detail,
        json!({"feature_id": feature_id}),
    )
}

fn rotation_json(value: &Value) -> Result<DQuat> {
    if let Some(euler) = json_vec3(value) {
        return Ok(DQuat::from_euler(
            glam::EulerRot::XYZEx,
            euler.x,
            euler.y,
            euler.z,
        ));
    }
    let values = value
        .as_array()
        .filter(|values| values.len() == 4)
        .ok_or_else(|| evaluation("rotation socket input must be Euler vec3 or quaternion"))?;
    let component = |index: usize| {
        values[index]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| evaluation("rotation quaternion must be finite"))
    };
    let x = component(0)?;
    let y = component(1)?;
    let z = component(2)?;
    let w = component(3)?;
    let length_squared = x * x + y * y + z * z + w * w;
    if !length_squared.is_finite() {
        return Err(evaluation("rotation quaternion cannot be normalized"));
    }
    if length_squared == 0.0 {
        return Ok(DQuat::IDENTITY);
    }
    let rotation = DQuat::from_xyzw(x, y, z, w).normalize();
    if rotation.to_array().iter().all(|value| value.is_finite()) {
        Ok(rotation)
    } else {
        Err(evaluation("rotation quaternion cannot be normalized"))
    }
}

fn field_quaternion(field: &Field, mesh: &Mesh, index: usize, domain: &str) -> Result<DQuat> {
    rotation_json(&field_json(field, mesh, index, domain)?)
}

fn json_vec3(value: &Value) -> Option<DVec3> {
    let values = value.as_array()?;
    if values.len() != 3 {
        return None;
    }
    let vector = DVec3::new(
        values[0].as_f64()?,
        values[1].as_f64()?,
        values[2].as_f64()?,
    );
    vector.is_finite().then_some(vector)
}

fn vector_json(value: DVec3) -> Value {
    json!(value.to_array())
}

fn property_vec3(node: &GraphNode, key: &str, default: DVec3) -> Result<DVec3> {
    node.properties.get(key).map_or(Ok(default), |value| {
        json_vec3(value)
            .ok_or_else(|| evaluation(format!("node property `{key}` must be a finite vec3")))
    })
}

fn property_f64(node: &GraphNode, key: &str, default: f64) -> Result<f64> {
    node.properties.get(key).map_or(Ok(default), |value| {
        value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| evaluation(format!("node property `{key}` must be a finite number")))
    })
}

fn property_i64(node: &GraphNode, key: &str, default: i64) -> Result<i64> {
    node.properties.get(key).map_or(Ok(default), |value| {
        value
            .as_i64()
            .ok_or_else(|| evaluation(format!("node property `{key}` must be an integer")))
    })
}
fn texture_string_property<'a>(
    node: &'a GraphNode,
    property: &str,
    default: &'a str,
) -> Result<&'a str> {
    match node.properties.get(property) {
        None => Ok(default),
        Some(value) => value
            .as_str()
            .ok_or_else(|| evaluation(format!("texture property `{property}` must be a string"))),
    }
}

fn texture_float_property(node: &GraphNode, property: &str, default: f32) -> Result<f32> {
    let value = property_f64(node, property, f64::from(default))? as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(evaluation(format!(
            "texture property `{property}` exceeds the float range"
        )))
    }
}

fn texture_integer_property(node: &GraphNode, property: &str, default: i32) -> Result<i32> {
    let value = property_i64(node, property, i64::from(default))?;
    i32::try_from(value).map_err(|_| {
        evaluation(format!(
            "texture property `{property}` exceeds the integer range"
        ))
    })
}

fn unsupported_texture_mode(
    node_id: &Id,
    node_type: &str,
    property: &str,
    value: &str,
) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("{node_type} mode `{value}` is not implemented"),
        json!({
            "feature_id": format!("graph.node.{node_type}.{property}"),
            "node_id": node_id,
            "value": value,
        }),
    )
}

#[expect(
    clippy::cast_precision_loss,
    reason = "mesh-line indices are bounded to one million and exactly representable"
)]
fn interpolation_factor(index: usize, count: usize) -> f64 {
    if count <= 1 {
        0.0
    } else {
        index as f64 / (count - 1) as f64
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the floored number was checked to fit in the i64 range"
)]
fn floor_to_i64(value: f64) -> Result<i64> {
    const I64_MIN_F64: f64 = -9_223_372_036_854_775_808.0;
    const I64_MAX_EXCLUSIVE_F64: f64 = 9_223_372_036_854_775_808.0;
    let floored = value.floor();
    if !floored.is_finite() || !(I64_MIN_F64..I64_MAX_EXCLUSIVE_F64).contains(&floored) {
        return Err(evaluation(
            "floored value is outside the signed 64-bit range",
        ));
    }
    Ok(floored as i64)
}

#[cfg(test)]
mod node_bake_tests {
    use super::{NodeBakeFrame, NodeBakeRecord, NodeBakeValue, mix_bake_values, read_bake_value};
    use crate::{
        error::{ErrorCode, PotError, Result},
        geom::{Edge, Face, IdCounters, Mesh, Vertex},
    };
    use glam::DVec3;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn invalid(message: &str) -> PotError {
        PotError::new(ErrorCode::InternalError, message)
    }

    /// A single-triangle mesh with the boundary edges a valid mesh must carry (`face boundary edge is
    /// missing` otherwise) and `next_id` counters allocated past every element ID.
    fn triangle(z: f64, vertex_count: u32, winding: [u32; 3]) -> Mesh {
        let edges = [
            [winding[0], winding[1]],
            [winding[1], winding[2]],
            [winding[2], winding[0]],
        ]
        .into_iter()
        .enumerate()
        .map(|(index, vertices)| Edge {
            id: u32::try_from(index).unwrap_or(u32::MAX),
            vertices,
        })
        .collect::<Vec<_>>();
        Mesh {
            vertices: (0..vertex_count)
                .map(|id| Vertex {
                    id,
                    co: DVec3::new(f64::from(id), 0.0, z),
                })
                .collect(),
            edges,
            faces: vec![Face {
                id: 0,
                vertices: winding.to_vec(),
                material_index: 0,
            }],
            next_id: IdCounters {
                vertex: vertex_count,
                edge: 3,
                face: 1,
            },
            ..Mesh::default()
        }
    }

    fn geometry(mesh: Mesh) -> NodeBakeValue {
        NodeBakeValue::Geometry {
            mesh,
            instance_paths: Vec::new(),
            instance_columns: None,
            material_refs: Vec::new(),
        }
    }

    fn record(frames: Vec<NodeBakeFrame>) -> NodeBakeRecord {
        NodeBakeRecord {
            storage: "inline".to_owned(),
            kind: "node_bake".to_owned(),
            version: 1,
            frames,
        }
    }

    #[test]
    fn geometry_cache_interpolates_only_matching_topology() -> Result<()> {
        let first = geometry(triangle(0.0, 3, [0, 1, 2]));
        let same_topology = geometry(triangle(2.0, 3, [0, 1, 2]));
        let NodeBakeValue::Geometry { mesh, .. } = mix_bake_values(&first, &same_topology, 0.5)
        else {
            return Err(invalid("geometry cache produced a non-geometry value"));
        };
        assert!((mesh.vertices[0].co.z - 1.0).abs() < 1.0e-12);

        let different_count = geometry(triangle(8.0, 4, [0, 1, 2]));
        assert_eq!(mix_bake_values(&first, &different_count, 0.5), first);

        let different_faces = geometry(triangle(8.0, 3, [0, 2, 1]));
        assert_eq!(mix_bake_values(&first, &different_faces, 0.5), first);
        Ok(())
    }

    #[test]
    fn scalar_bakes_interpolate_and_clamp_frame_range() -> Result<()> {
        let frames = [(1.0, 2.0), (3.0, 6.0)]
            .into_iter()
            .map(|(frame, value)| NodeBakeFrame {
                frame,
                values: BTreeMap::from([(
                    "Item_0".to_owned(),
                    NodeBakeValue::Scalar {
                        value: json!(value),
                    },
                )]),
            })
            .collect();
        let record = record(frames);
        for (frame, expected) in [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)] {
            let Some(NodeBakeValue::Scalar { value }) = read_bake_value(&record, frame, "Item_0")?
            else {
                return Err(invalid("scalar bake item is missing"));
            };
            let number = value
                .as_f64()
                .ok_or_else(|| invalid("scalar is not numeric"))?;
            assert!((number - expected).abs() < 1.0e-12);
        }
        Ok(())
    }

    #[test]
    fn typed_bake_values_and_material_references_roundtrip() -> Result<()> {
        let mut carrier = triangle(0.0, 3, [0, 1, 2]);
        carrier.attributes.insert(
            "attr".to_owned(),
            json!({"domain": "POINT", "type": "float", "data": [1.0, 2.0, 3.0]}),
        );
        let mut values: BTreeMap<String, NodeBakeValue> = [
            ("float", json!(0.75)),
            ("integer", json!(7)),
            ("boolean", json!(true)),
            ("vector", json!([1.0, 2.0, 3.0])),
            ("rotation", json!([0.0, 0.0, 0.0, 1.0])),
            ("color", json!([0.1, 0.2, 0.3, 1.0])),
            ("string", json!("hello")),
            ("material", json!("surface")),
        ]
        .into_iter()
        .map(|(socket, value)| (socket.to_owned(), NodeBakeValue::Scalar { value }))
        .collect();
        values.insert("carrier".to_owned(), geometry(carrier));
        values.insert(
            "field".to_owned(),
            NodeBakeValue::Field {
                attribute_name: "attr".to_owned(),
            },
        );
        let record = record(vec![NodeBakeFrame { frame: 1.0, values }]);
        let encoded = serde_json::to_value(&record).map_err(PotError::internal_json)?;
        crate::graph::validate_node_bake_resource(&encoded)?;
        for socket in [
            "float", "integer", "boolean", "vector", "rotation", "color", "string", "material",
        ] {
            let value = read_bake_value(&record, 1.0, socket)?
                .ok_or_else(|| invalid("bake item is present"))?;
            assert!(matches!(value, NodeBakeValue::Scalar { .. }));
        }
        let field = read_bake_value(&record, 1.0, "field")?
            .ok_or_else(|| invalid("field item is present"))?;
        assert_eq!(
            field,
            NodeBakeValue::Field {
                attribute_name: "attr".to_owned(),
            }
        );
        Ok(())
    }
}
