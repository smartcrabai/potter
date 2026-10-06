use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    sync::{LazyLock, Mutex},
};

use glam::{DMat4, DQuat, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{GridParams, IcosphereParams, Mesh, MeshError, modifiers::evaluate_modifiers},
    model::{Id, Modifier},
};

use super::{
    GraphKind, GraphNode, NodeGroup,
    sockets::{is_known_node_type, node_sockets},
};

const MAX_GENERATED_POINTS: usize = 1_000_000;
const MAX_GENERATED_POINTS_F64: f64 = 1_000_000.0;
const RANDOM_MANTISSA_RANGE: f64 = 9_007_199_254_740_992.0;
const MAX_ZONE_ITERATIONS: i64 = 100_000;
const MAX_SIMULATION_CACHE_ENTRIES: usize = 10_000;

type SimulationCache = Mutex<BTreeMap<(String, Id, i64), GeometryValue>>;

static SIMULATION_CACHE: LazyLock<SimulationCache> = LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Evaluated geometry plus stable paths for realized geometry-node instances.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphEvaluation {
    pub mesh: Mesh,
    pub instance_paths: Vec<Vec<Id>>,
}

#[derive(Clone, Debug)]
struct GeometryValue {
    mesh: Mesh,
    instance_paths: Vec<Vec<Id>>,
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
    Attribute(String),
    AttributeExists(String),
    Captured {
        domain: String,
        values: Vec<Value>,
    },
    Math {
        operation: String,
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
        seed: u64,
        data_type: String,
        id: Box<Field>,
        min: Box<Field>,
        max: Box<Field>,
    },
}

struct Evaluator<'a> {
    group: &'a NodeGroup,
    input: Option<&'a Mesh>,
    overrides: &'a Map<String, Value>,
    frame: f64,
    simulation_signature: Option<String>,
    simulation_states: HashMap<Id, GeometryValue>,
    repeat_states: HashMap<Id, GeometryValue>,
    repeat_iterations: HashMap<Id, i64>,
    active: BTreeSet<Id>,
    cache: BTreeMap<(Id, String), EvalValue>,
}

/// Evaluate a geometry node group against an optional incoming mesh.
///
/// Inputs are keyed by interface socket ID or name. Generated instances are realized
/// into `mesh` for bounds and rendering while `instance_paths` records each instance.
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
) -> Result<GraphEvaluation> {
    evaluate_at_frame(group, input, overrides, 1.0)
}

/// Evaluate a geometry node group at a scene frame.
///
/// Simulation-zone states are cached by graph inputs and integer frame.
///
/// # Errors
///
/// Returns an error for invalid graphs, unsupported graph kinds, or invalid evaluation.
pub fn evaluate_at_frame(
    group: &NodeGroup,
    input: Option<&Mesh>,
    overrides: &Map<String, Value>,
    frame: f64,
) -> Result<GraphEvaluation> {
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
    validate_graph(group)?;
    let simulation_signature = group
        .nodes
        .values()
        .any(|node| node.node_type == "GeometryNodeSimulationOutput")
        .then(|| {
            serde_json::to_value((group, input, overrides))
                .map_err(|error| {
                    PotError::new(
                        ErrorCode::InternalError,
                        format!("could not serialize graph simulation inputs: {error}"),
                    )
                })
                .and_then(|value| {
                    crate::hash::canonicalize(&value).map(|bytes| crate::hash::sha256(&bytes))
                })
        })
        .transpose()?;
    let mut output_nodes = group
        .nodes
        .iter()
        .filter(|(_, node)| node.node_type == "NodeGroupOutput");
    let output_node = output_nodes
        .next()
        .map(|(id, _)| id.clone())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "geometry node group has no NodeGroupOutput",
            )
        })?;
    if output_nodes.next().is_some() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "geometry node group has multiple NodeGroupOutput nodes",
        ));
    }
    let output_socket = if group.interface.outputs.is_empty() {
        "Geometry".to_owned()
    } else {
        let socket = group
            .interface
            .outputs
            .iter()
            .find(|socket| socket.socket_type == "geometry")
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "geometry node group has no geometry output socket",
                )
            })?;
        group
            .links
            .iter()
            .find(|link| {
                link.to_node == output_node
                    && (link.to_socket == socket.id || link.to_socket == socket.name)
            })
            .map(|link| link.to_socket.clone())
            .or_else(|| {
                group.nodes.get(&output_node).and_then(|node| {
                    if node.inputs.contains_key(&socket.id) {
                        Some(socket.id.clone())
                    } else if node.inputs.contains_key(&socket.name) {
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
        input,
        overrides,
        frame,
        simulation_signature,
        simulation_states: HashMap::new(),
        repeat_states: HashMap::new(),
        repeat_iterations: HashMap::new(),
        active: BTreeSet::new(),
        cache: BTreeMap::new(),
    };
    let value = evaluator.input_value(
        &output_node,
        &output_socket,
        EvalValue::Geometry(GeometryValue {
            mesh: input.cloned().unwrap_or_default(),
            instance_paths: Vec::new(),
        }),
    )?;
    match value {
        EvalValue::Geometry(value) => Ok(GraphEvaluation {
            mesh: value.mesh,
            instance_paths: value.instance_paths,
        }),
        _ => Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "geometry group output is not geometry",
        )),
    }
}

fn validate_graph(group: &NodeGroup) -> Result<()> {
    for (node_id, node) in &group.nodes {
        if !is_known_node_type(&node.node_type) {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("unsupported graph node type `{}`", node.node_type),
                json!({"feature_id": format!("graph.node.{}", node.node_type), "node_id": node_id}),
            ));
        }
        let (input_sockets, _) = node_sockets(group, node_id).ok_or_else(|| {
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
    super::sockets::validate_group_links(group)?;
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
        let value = result?;
        self.cache.insert(key, value.clone());
        Ok(value)
    }

    fn evaluate_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        match node.node_type.as_str() {
            "NodeGroupInput" => self.group_input_value(socket),
            "NodeGroupOutput" => self.input_value(
                id,
                socket,
                EvalValue::Geometry(GeometryValue {
                    mesh: self.input.cloned().unwrap_or_default(),
                    instance_paths: Vec::new(),
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
            "GeometryNodeRealizeInstances" => self.geometry_input(id, node, "Geometry"),
            "GeometryNodeDistributePointsOnFaces" => self.distribute_points(id, node),
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
            "ShaderNodeCombineXYZ" => self.combine_xyz(id, node),
            "ShaderNodeSeparateXYZ" => self.separate_xyz(id, node, socket),
            "FunctionNodeRandomValue" => self.random_node(id, node, socket),
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
            "Delta Time" => Ok(EvalValue::Scalar(json!(1.0 / 24.0))),
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

    fn group_input_value(&self, socket: &str) -> Result<EvalValue> {
        let interface_socket = self
            .group
            .interface
            .inputs
            .iter()
            .find(|candidate| candidate.name == socket || candidate.id == socket)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    format!("unknown group input socket `{socket}`"),
                )
            })?;
        let value = self
            .overrides
            .get(&interface_socket.id)
            .or_else(|| self.overrides.get(&interface_socket.name))
            .cloned()
            .unwrap_or_else(|| interface_socket.default.clone());
        Ok(EvalValue::Scalar(value))
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
        Ok(match default_input(&node.node_type, socket) {
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
        if let Some(value) = node.inputs.get(socket) {
            return Ok(EvalValue::Scalar(value.clone()));
        }
        Ok(match default_input(&node.node_type, socket) {
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
            }),
        )
    }

    fn geometry_input(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        let value = self.input_geometry(id, node, socket)?;
        match value {
            EvalValue::Geometry(_) => Ok(value),
            _ => Err(PotError::new(
                ErrorCode::EvaluationFailed,
                format!("input `{socket}` on `{}` is not geometry", node.node_type),
            )),
        }
    }

    fn mesh_primitive(&mut self, id: &Id, node: &GraphNode, kind: &str) -> Result<EvalValue> {
        let mesh = match kind {
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
        Ok(EvalValue::Geometry(GeometryValue {
            mesh,
            instance_paths: Vec::new(),
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
        let count = self.integer_input(id, node, "Count", property_i64(node, "count", 2)?)?;
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
        }))
    }

    fn transform_node(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let geometry = self.geometry_input(id, node, "Geometry")?;
        let mut geometry = into_geometry(geometry)?;
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
        let rotation = self.quaternion_input(id, node, "Rotation")?;
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
        for link in links {
            match self.output_value(&link.from_node, &link.from_socket)? {
                EvalValue::Geometry(mut geometry) => {
                    paths.append(&mut geometry.instance_paths);
                    geometries.push(geometry.mesh);
                }
                _ => return Err(evaluation("Join Geometry input is not geometry")),
            }
        }
        let mesh = merge_meshes(&geometries)?;
        Ok(EvalValue::Geometry(GeometryValue {
            mesh,
            instance_paths: paths,
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
            }),
        )?;
        let source_value = self.input_value_with_node(
            id,
            node,
            "Instance",
            EvalValue::Geometry(GeometryValue {
                mesh: Mesh::new(),
                instance_paths: Vec::new(),
            }),
        )?;
        let points = into_geometry(points_value)?;
        let source = into_geometry(source_value)?;
        let selection = as_field(self.input_value_with_node(
            id,
            node,
            "Selection",
            EvalValue::Scalar(json!(true)),
        )?);
        let scale = as_field(self.input_value_with_node(
            id,
            node,
            "Scale",
            EvalValue::Scalar(json!([1.0, 1.0, 1.0])),
        )?);
        let rotation = self.quaternion_input(id, node, "Rotation")?;
        let source_node = self
            .group
            .links
            .iter()
            .find(|link| &link.to_node == id && link.to_socket == "Instance")
            .map(|link| link.from_node.clone());
        let mut meshes = Vec::new();
        let mut paths = source.instance_paths;
        for (index, point) in points.mesh.vertices.iter().enumerate() {
            if !field_bool(&selection, &points.mesh, index, "POINT", false)? {
                continue;
            }
            let mut instance_mesh = source.mesh.clone();
            let local_scale = field_vec3(&scale, &points.mesh, index, DVec3::ONE)?;
            let matrix = DMat4::from_scale_rotation_translation(local_scale, rotation, point.co);
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
        }))
    }

    fn distribute_points(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let surface_value = self.input_value_with_node(
            id,
            node,
            "Mesh",
            EvalValue::Geometry(GeometryValue {
                mesh: self.input.cloned().unwrap_or_default(),
                instance_paths: Vec::new(),
            }),
        )?;
        let surface = into_geometry(surface_value)?.mesh;
        let density = as_field(self.input_value_with_node(
            id,
            node,
            "Density",
            EvalValue::Scalar(json!(1.0)),
        )?);
        let minimum_distance = self.scalar_input(id, node, "Distance Min", 0.0)?;
        let seed = self.integer_input(id, node, "Seed", 0)?;
        if !minimum_distance.is_finite() || minimum_distance < 0.0 {
            return Err(evaluation(
                "minimum point distance must be finite and non-negative",
            ));
        }
        let selection = as_field(self.input_value_with_node(
            id,
            node,
            "Selection",
            EvalValue::Scalar(json!(true)),
        )?);
        let triangles = surface
            .triangulate()
            .map_err(|error| mesh_evaluation_error(&error))?;
        let positions: HashMap<_, _> = surface
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co))
            .collect();
        let mut weighted = Vec::new();
        let mut triangle_offset = 0_usize;
        for (face_index, face) in surface.faces.iter().enumerate() {
            let triangle_count = face.vertices.len().saturating_sub(2);
            if field_bool(&selection, &surface, face_index, "FACE", false)? {
                let face_density = field_json(&density, &surface, face_index, "FACE")?
                    .as_f64()
                    .filter(|density| density.is_finite() && *density >= 0.0)
                    .ok_or_else(|| {
                        evaluation("point density field must be finite and non-negative")
                    })?;
                for _ in 0..triangle_count {
                    let triangle = *triangles
                        .get(triangle_offset)
                        .ok_or_else(|| evaluation("surface triangulation result is incomplete"))?;
                    let first = *positions
                        .get(&triangle[0])
                        .ok_or_else(|| evaluation("triangle vertex missing"))?;
                    let second = *positions
                        .get(&triangle[1])
                        .ok_or_else(|| evaluation("triangle vertex missing"))?;
                    let third = *positions
                        .get(&triangle[2])
                        .ok_or_else(|| evaluation("triangle vertex missing"))?;
                    let area = (second - first).cross(third - first).length() * 0.5;
                    let weight = area * face_density;
                    if weight > 0.0 && weight.is_finite() {
                        weighted.push((triangle, weight));
                    }
                    triangle_offset += 1;
                }
            } else {
                triangle_offset += triangle_count;
            }
        }
        if triangle_offset != triangles.len() {
            return Err(evaluation(
                "surface triangulation result does not match face domains",
            ));
        }
        let total_area: f64 = weighted.iter().map(|(_, area)| *area).sum();
        let count = total_area.round();
        if !count.is_finite() || count > MAX_GENERATED_POINTS_F64 {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "distributed point count exceeds limit",
            ));
        }
        let count = bounded_count_to_usize(count)?;
        if count == 0 || weighted.is_empty() {
            return Ok(EvalValue::Geometry(GeometryValue {
                mesh: Mesh::new(),
                instance_paths: Vec::new(),
            }));
        }
        let mut random = u64::from_ne_bytes(seed.to_ne_bytes());
        let cumulative: Vec<_> = weighted
            .iter()
            .scan(0.0, |sum, (triangle, area)| {
                *sum += area;
                Some((*sum, *triangle))
            })
            .collect();
        let mut mesh = Mesh::new();
        let mut grid: HashMap<[i64; 3], Vec<DVec3>> = HashMap::new();
        let attempt_limit = if minimum_distance > 0.0 {
            count.saturating_mul(30)
        } else {
            count
        };
        for _ in 0..attempt_limit {
            if mesh.vertices.len() >= count {
                break;
            }
            let target = next_unit(&mut random) * total_area;
            let triangle = cumulative
                .iter()
                .find(|(cumulative_area, _)| *cumulative_area >= target)
                .or_else(|| cumulative.last())
                .map(|(_, triangle)| *triangle)
                .ok_or_else(|| evaluation("surface triangulation failed"))?;
            let first = *positions
                .get(&triangle[0])
                .ok_or_else(|| evaluation("triangle vertex missing"))?;
            let second = *positions
                .get(&triangle[1])
                .ok_or_else(|| evaluation("triangle vertex missing"))?;
            let third = *positions
                .get(&triangle[2])
                .ok_or_else(|| evaluation("triangle vertex missing"))?;
            let root = next_unit(&mut random).sqrt();
            let split = next_unit(&mut random);
            let point =
                first * (1.0 - root) + second * (root * (1.0 - split)) + third * (root * split);
            if minimum_distance > 0.0 && !separated_from_neighbors(point, minimum_distance, &grid)?
            {
                continue;
            }
            mesh.insert_vertex(point)
                .map_err(|error| mesh_evaluation_error(&error))?;
            if minimum_distance > 0.0 {
                grid.entry(poisson_cell(point, minimum_distance)?)
                    .or_default()
                    .push(point);
            }
        }
        Ok(EvalValue::Geometry(GeometryValue {
            mesh,
            instance_paths: Vec::new(),
        }))
    }

    fn mesh_to_points(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let geometry = self.geometry_input(id, node, "Mesh")?;
        let geometry = into_geometry(geometry)?;
        let domain = node
            .properties
            .get("domain")
            .and_then(Value::as_str)
            .unwrap_or("POINT");
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
            }
        }
        points
            .attributes
            .insert("radius".to_owned(), json!({"domain":"POINT","data":radii}));
        Ok(EvalValue::Geometry(GeometryValue {
            mesh: points,
            instance_paths: geometry.instance_paths,
        }))
    }

    fn subdivision(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let geometry = self.geometry_input(id, node, "Mesh")?;
        let geometry = into_geometry(geometry)?;
        let level = self.integer_input(id, node, "Level", 1)?;
        if !(0..=6).contains(&level) {
            return Err(evaluation("subdivision level must be between 0 and 6"));
        }
        if level == 0 {
            return Ok(EvalValue::Geometry(geometry));
        }
        let modifier = Modifier {
            id: Id::new("graph_subdivision")
                .map_err(|error| PotError::new(ErrorCode::InternalError, error.message))?,
            modifier_type: "subdivision".to_owned(),
            name: "Graph Subdivision Surface".to_owned(),
            enabled: true,
            params: Map::from_iter([("levels".to_owned(), json!(level))]),
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        };
        let mesh = evaluate_modifiers(&geometry.mesh, &[modifier])?;
        Ok(EvalValue::Geometry(GeometryValue {
            mesh,
            instance_paths: geometry.instance_paths,
        }))
    }

    fn store_named_attribute(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
        let geometry_value = self.geometry_input(id, node, "Geometry")?;
        let mut geometry = into_geometry(geometry_value)?;
        let name = self.string_input(id, node, "Name", "")?;
        if name.is_empty() {
            return Err(evaluation("named attribute requires a non-empty name"));
        }
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
        let value = as_field(self.input_value_with_node(
            id,
            node,
            "Value",
            EvalValue::Scalar(Value::Null),
        )?);
        let previous = if let Some(attribute) = geometry.mesh.attributes.get(&name) {
            if attribute.get("domain").and_then(Value::as_str) != Some(domain) {
                return Err(evaluation(
                    "stored attribute domain differs from its existing domain",
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
                values.push(field_json(&value, &geometry.mesh, index, domain)?);
            } else {
                values.push(
                    previous
                        .and_then(|data| data.get(index))
                        .cloned()
                        .unwrap_or(Value::Null),
                );
            }
        }
        geometry
            .mesh
            .attributes
            .insert(name, json!({"domain":domain,"data":values}));
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
        Ok(EvalValue::Field(Field::Attribute(name)))
    }

    fn capture_attribute(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
        if socket != "Attribute" {
            return self.geometry_input(id, node, "Geometry");
        }
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
        let value = as_field(self.input_value_with_node(
            id,
            node,
            "Value",
            EvalValue::Scalar(Value::Null),
        )?);
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            if field_bool(&selection, &geometry.mesh, index, domain, false)? {
                values.push(field_json(&value, &geometry.mesh, index, domain)?);
            } else {
                values.push(Value::Null);
            }
        }
        Ok(EvalValue::Field(Field::Captured {
            domain: domain.to_owned(),
            values,
        }))
    }

    fn math_node(&mut self, id: &Id, node: &GraphNode) -> Result<EvalValue> {
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
            operation: node
                .properties
                .get("operation")
                .and_then(Value::as_str)
                .unwrap_or("ADD")
                .to_ascii_uppercase(),
            first: Box::new(first),
            second: Box::new(second),
            third: Box::new(third),
        }))
    }

    fn vector_math_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<EvalValue> {
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
            EvalValue::Scalar(json!([0.0, 0.0, 0.0])),
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
        let operation = node
            .properties
            .get("operation")
            .and_then(Value::as_str)
            .unwrap_or("ADD")
            .to_ascii_uppercase();
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
        let seed = self.integer_input(id, node, "Seed", 0)?;
        let data_type = node
            .properties
            .get("data_type")
            .and_then(Value::as_str)
            .unwrap_or("FLOAT")
            .to_ascii_uppercase();
        let (min_default, max_default) = match data_type.as_str() {
            "VECTOR" | "FLOAT_VECTOR" => (json!([0.0, 0.0, 0.0]), json!([1.0, 1.0, 1.0])),
            "BOOLEAN" | "BOOL" => (json!(false), json!(true)),
            "INT" | "INTEGER" => (json!(0), json!(1)),
            _ => (json!(0.0), json!(1.0)),
        };
        let id_field =
            as_field(self.input_value_with_node(id, node, "ID", EvalValue::Field(Field::Index))?);
        let min = as_field(self.input_value_with_node(
            id,
            node,
            "Min",
            EvalValue::Scalar(min_default),
        )?);
        let max = as_field(self.input_value_with_node(
            id,
            node,
            "Max",
            EvalValue::Scalar(max_default),
        )?);
        if socket != "Value" {
            return Err(evaluation("unknown Random Value output"));
        }
        Ok(EvalValue::Field(Field::Random {
            seed: u64::from_ne_bytes(seed.to_ne_bytes()),
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

    fn quaternion_input(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<DQuat> {
        let value = self.input_value_with_node(
            id,
            node,
            socket,
            EvalValue::Scalar(json!([0.0, 0.0, 0.0, 1.0])),
        )?;
        match value {
            EvalValue::Scalar(value) => {
                if let Some(vector) = json_vec3(&value) {
                    Ok(DQuat::from_euler(
                        glam::EulerRot::XYZ,
                        vector.x,
                        vector.y,
                        vector.z,
                    ))
                } else if let Some(values) = value.as_array().filter(|values| values.len() == 4) {
                    let numbers = values.iter().map(Value::as_f64).collect::<Option<Vec<_>>>();
                    let numbers = numbers
                        .filter(|values| values.iter().all(|value| value.is_finite()))
                        .ok_or_else(|| evaluation("rotation must be finite"))?;
                    Ok(
                        DQuat::from_xyzw(numbers[0], numbers[1], numbers[2], numbers[3])
                            .normalize(),
                    )
                } else {
                    Err(evaluation(
                        "rotation socket input must be Euler vec3 or quaternion",
                    ))
                }
            }
            _ => Err(evaluation("rotation socket received non-scalar value")),
        }
    }
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

fn default_input(node_type: &str, socket: &str) -> Option<Value> {
    let value = match (node_type, socket) {
        ("GeometryNodeTransform", "Translation")
        | ("GeometryNodeSetPosition", "Offset")
        | ("ShaderNodeVectorMath", "Vector" | "Vector_001" | "Vector_002") => {
            json!([0.0, 0.0, 0.0])
        }
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
        ) => json!(true),
        ("GeometryNodeDistributePointsOnFaces", "Density") | ("ShaderNodeVectorMath", "Scale") => {
            json!(1.0)
        }
        ("GeometryNodeDistributePointsOnFaces", "Seed") => json!(0),
        ("GeometryNodeSimulationOutput", "Skip") => json!(false),
        ("GeometryNodeRepeatInput", "Iterations") => json!(1),
        ("ShaderNodeMath", "Value" | "Value_001" | "Value_002") => json!(0.5),
        ("GeometryNodeCombineXYZ", "X" | "Y" | "Z") => json!(0.0),
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

fn field_json(field: &Field, mesh: &Mesh, index: usize, domain: &str) -> Result<Value> {
    let value = match field {
        Field::Constant(value) => value.clone(),
        Field::Position => vector_json(domain_position(mesh, index, domain)?),
        Field::Normal => vector_json(normal_at(mesh, index, domain)?),
        Field::Index => json!(index),
        Field::Attribute(name) => {
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
            values
                .get(index)
                .cloned()
                .ok_or_else(|| evaluation("named attribute index is out of range"))?
        }
        Field::AttributeExists(name) => json!(mesh.attributes.contains_key(name)),
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
        Field::Math {
            operation,
            first,
            second,
            third,
        } => {
            let a = numeric_field(first, mesh, index, domain, "math")?;
            let b = numeric_field(second, mesh, index, domain, "math")?;
            let c = numeric_field(third, mesh, index, domain, "math")?;
            json!(math_value(operation, a, b, c)?)
        }
        Field::VectorMath {
            operation,
            first,
            second,
            third,
            scale,
            output,
        } => {
            let first = vector_field(first, mesh, index, domain, "vector math")?;
            let second = vector_field(second, mesh, index, domain, "vector math")?;
            let third = vector_field(third, mesh, index, domain, "vector math")?;
            let scale = numeric_field(scale, mesh, index, domain, "vector math scale")?;
            vector_math_value(operation, output, first, second, third, scale)?
        }
        Field::CombineXYZ { x, y, z } => {
            let x = numeric_field(x, mesh, index, domain, "Combine XYZ")?;
            let y = numeric_field(y, mesh, index, domain, "Combine XYZ")?;
            let z = numeric_field(z, mesh, index, domain, "Combine XYZ")?;
            vector_json(DVec3::new(x, y, z))
        }
        Field::SeparateXYZ { vector, axis } => {
            let vector = vector_field(vector, mesh, index, domain, "Separate XYZ")?;
            json!(vector[*axis])
        }
        Field::Random {
            seed,
            data_type,
            id,
            min,
            max,
        } => {
            let id = integer_field(id, mesh, index, domain)?;
            let mut state = *seed ^ u64::from_ne_bytes(id.to_ne_bytes());
            let min = field_json(min, mesh, index, domain)?;
            let max = field_json(max, mesh, index, domain)?;
            if data_type == "VECTOR" || data_type == "FLOAT_VECTOR" {
                let min = json_vec3(&min)
                    .ok_or_else(|| evaluation("random vector minimum is invalid"))?;
                let max = json_vec3(&max)
                    .ok_or_else(|| evaluation("random vector maximum is invalid"))?;
                vector_json(DVec3::new(
                    min.x + next_unit(&mut state) * (max.x - min.x),
                    min.y + next_unit(&mut state) * (max.y - min.y),
                    min.z + next_unit(&mut state) * (max.z - min.z),
                ))
            } else if data_type == "INT" || data_type == "INTEGER" {
                let min = min
                    .as_i64()
                    .ok_or_else(|| evaluation("random integer minimum is invalid"))?;
                let max = max
                    .as_i64()
                    .ok_or_else(|| evaluation("random integer maximum is invalid"))?;
                json!(random_i64_inclusive(&mut state, min, max)?)
            } else if data_type == "BOOLEAN" || data_type == "BOOL" {
                let minimum = min
                    .as_bool()
                    .ok_or_else(|| evaluation("random boolean minimum is invalid"))?;
                let maximum = max
                    .as_bool()
                    .ok_or_else(|| evaluation("random boolean maximum is invalid"))?;
                if minimum == maximum {
                    json!(minimum)
                } else {
                    json!(if next_unit(&mut state) >= 0.5 {
                        maximum
                    } else {
                        minimum
                    })
                }
            } else {
                let min = min
                    .as_f64()
                    .ok_or_else(|| evaluation("random minimum is not numeric"))?;
                let max = max
                    .as_f64()
                    .ok_or_else(|| evaluation("random maximum is not numeric"))?;
                json!(min + next_unit(&mut state) * (max - min))
            }
        }
    };
    if json_is_finite(&value) {
        Ok(value)
    } else {
        Err(evaluation("field produced a non-finite value"))
    }
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

fn separated_from_neighbors(
    candidate: DVec3,
    minimum_distance: f64,
    grid: &HashMap<[i64; 3], Vec<DVec3>>,
) -> Result<bool> {
    let cell = poisson_cell(candidate, minimum_distance)?;
    let minimum_squared = minimum_distance * minimum_distance;
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
                if grid.get(&[x, y, z]).is_some_and(|points| {
                    points
                        .iter()
                        .any(|point| point.distance_squared(candidate) < minimum_squared)
                }) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

fn numeric_field(
    field: &Field,
    mesh: &Mesh,
    index: usize,
    domain: &str,
    context: &str,
) -> Result<f64> {
    field_json(field, mesh, index, domain)?
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| evaluation(format!("{context} input is not a finite number")))
}

fn vector_field(
    field: &Field,
    mesh: &Mesh,
    index: usize,
    domain: &str,
    context: &str,
) -> Result<DVec3> {
    json_vec3(&field_json(field, mesh, index, domain)?)
        .ok_or_else(|| evaluation(format!("{context} input is not a finite vector")))
}

fn integer_field(field: &Field, mesh: &Mesh, index: usize, domain: &str) -> Result<i64> {
    field_json(field, mesh, index, domain)?
        .as_i64()
        .ok_or_else(|| evaluation("random ID field must produce an integer"))
}

fn json_is_finite(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.as_f64().is_none_or(f64::is_finite),
        Value::Array(values) => values.iter().all(json_is_finite),
        _ => true,
    }
}

fn math_value(operation: &str, a: f64, b: f64, c: f64) -> Result<f64> {
    let result = match operation {
        "ADD" => a + b,
        "SUBTRACT" => a - b,
        "MULTIPLY" => a * b,
        "MULTIPLY_ADD" => a.mul_add(b, c),
        "DIVIDE" if !crate::float::equal_f64(b, 0.0) => a / b,
        "DIVIDE" => return Err(evaluation("division by zero in math field")),
        "POWER" => a.powf(b),
        "LOGARITHM" => a.log(b),
        "SQRT" => a.sqrt(),
        "INVERSE_SQRT" | "INV_SQRT" => 1.0 / a.sqrt(),
        "ABSOLUTE" => a.abs(),
        "EXPONENT" => a.exp(),
        "MINIMUM" => a.min(b),
        "MAXIMUM" => a.max(b),
        "LESS_THAN" => {
            if a < b {
                1.0
            } else {
                0.0
            }
        }
        "GREATER_THAN" => {
            if a > b {
                1.0
            } else {
                0.0
            }
        }
        "SIGN" => a.signum(),
        "COMPARE" => {
            if (a - b).abs() <= c.abs() {
                1.0
            } else {
                0.0
            }
        }
        "SMOOTH_MIN" => smooth_min(a, b, c),
        "SMOOTH_MAX" => -smooth_min(-a, -b, c),
        "ROUND" => a.round(),
        "FLOOR" => a.floor(),
        "CEIL" => a.ceil(),
        "TRUNC" => a.trunc(),
        "FRACT" => a - a.floor(),
        "MODULO" if !crate::float::equal_f64(b, 0.0) => a.rem_euclid(b),
        "FLOORED_MODULO" if !crate::float::equal_f64(b, 0.0) => a - b * (a / b).floor(),
        "MODULO" | "FLOORED_MODULO" => return Err(evaluation("modulo by zero in math field")),
        "WRAP" if !crate::float::equal_f64(b, c) => {
            let (low, high) = if b < c { (b, c) } else { (c, b) };
            low + (a - low).rem_euclid(high - low)
        }
        "WRAP" => return Err(evaluation("math wrap bounds must differ")),
        "SNAP" if !crate::float::equal_f64(b, 0.0) => (a / b).round() * b,
        "SNAP" => return Err(evaluation("math snap increment must not be zero")),
        "PINGPONG" if !crate::float::equal_f64(b, 0.0) => {
            let length = b.abs();
            length - ((a.rem_euclid(2.0 * length)) - length).abs()
        }
        "PINGPONG" => 0.0,
        "SINE" => a.sin(),
        "COSINE" => a.cos(),
        "TANGENT" => a.tan(),
        "ARCSINE" | "ASIN" => a.asin(),
        "ARCCOSINE" | "ACOS" => a.acos(),
        "ARCTANGENT" | "ATAN" => a.atan(),
        "ARCTAN2" | "ARCTAN_2" => a.atan2(b),
        "SINH" => a.sinh(),
        "COSH" => a.cosh(),
        "TANH" => a.tanh(),
        "DEGREES" => a.to_degrees(),
        "RADIANS" => a.to_radians(),
        other => {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("ShaderNodeMath operation `{other}` is not implemented"),
                json!({"feature_id":format!("graph.shader_math.{other}")}),
            ));
        }
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(evaluation("math field produced a non-finite value"))
    }
}

fn smooth_min(a: f64, b: f64, width: f64) -> f64 {
    if width <= 0.0 {
        return a.min(b);
    }
    let factor = ((width - (a - b).abs()) / width).clamp(0.0, 1.0);
    a.min(b) - factor * factor * width * 0.25
}

fn vector_math_value(
    operation: &str,
    output: &str,
    first: DVec3,
    second: DVec3,
    third: DVec3,
    scale: f64,
) -> Result<Value> {
    let scalar = match operation {
        "DOT_PRODUCT" => Some(first.dot(second)),
        "DISTANCE" => Some(first.distance(second)),
        "LENGTH" => Some(first.length()),
        _ => None,
    };
    if let Some(value) = scalar {
        if output != "Value" {
            return Err(evaluation(
                "Vector Math scalar result requested as a vector",
            ));
        }
        return if value.is_finite() {
            Ok(json!(value))
        } else {
            Err(evaluation("vector math produced a non-finite value"))
        };
    }
    if output != "Vector" {
        return Err(evaluation(
            "Vector Math vector result requested as a scalar",
        ));
    }
    let vector = match operation {
        "ADD" => first + second,
        "SUBTRACT" => first - second,
        "MULTIPLY" => first * second,
        "POWER" => DVec3::new(
            first.x.powf(second.x),
            first.y.powf(second.y),
            first.z.powf(second.z),
        ),
        "DIVIDE" => {
            if crate::float::equal_f64(second.x, 0.0)
                || crate::float::equal_f64(second.y, 0.0)
                || crate::float::equal_f64(second.z, 0.0)
            {
                return Err(evaluation("division by zero in vector math field"));
            }
            first / second
        }
        "CROSS_PRODUCT" => first.cross(second),
        "PROJECT" if second.length_squared() > 0.0 => first.project_onto(second),
        "PROJECT" => DVec3::ZERO,
        "REFLECT" => {
            let normal = second.normalize_or_zero();
            first - 2.0 * first.dot(normal) * normal
        }
        "REFRACT" => {
            let normal = second.normalize_or_zero();
            let cosine = (-first.dot(normal)).clamp(-1.0, 1.0);
            let discriminant = 1.0 - scale * scale * (1.0 - cosine * cosine);
            if discriminant < 0.0 {
                DVec3::ZERO
            } else {
                scale * first + (scale * cosine - discriminant.sqrt()) * normal
            }
        }
        "FACEFORWARD" => {
            if third.dot(second) < 0.0 {
                first
            } else {
                -first
            }
        }
        "SCALE" => first * scale,
        "NORMALIZE" => first.normalize_or_zero(),
        "ABSOLUTE" => first.abs(),
        "MINIMUM" => first.min(second),
        "MAXIMUM" => first.max(second),
        "FLOOR" => first.floor(),
        "CEIL" => first.ceil(),
        "ROUND" => first.round(),
        "FRACTION" => first - first.floor(),
        "MODULO" => {
            if crate::float::equal_f64(second.x, 0.0)
                || crate::float::equal_f64(second.y, 0.0)
                || crate::float::equal_f64(second.z, 0.0)
            {
                return Err(evaluation("modulo by zero in vector math field"));
            }
            DVec3::new(
                first.x.rem_euclid(second.x),
                first.y.rem_euclid(second.y),
                first.z.rem_euclid(second.z),
            )
        }
        "SNAP" => {
            if crate::float::equal_f64(second.x, 0.0)
                || crate::float::equal_f64(second.y, 0.0)
                || crate::float::equal_f64(second.z, 0.0)
            {
                return Err(evaluation("vector snap increment must not be zero"));
            }
            DVec3::new(
                (first.x / second.x).round() * second.x,
                (first.y / second.y).round() * second.y,
                (first.z / second.z).round() * second.z,
            )
        }
        "WRAP" => {
            let wrap = |value: f64, low: f64, high: f64| {
                let low = low.min(high);
                let range = (high - low).abs();
                if crate::float::equal_f64(range, 0.0) {
                    low
                } else {
                    low + (value - low).rem_euclid(range)
                }
            };
            DVec3::new(
                wrap(first.x, second.x, third.x),
                wrap(first.y, second.y, third.y),
                wrap(first.z, second.z, third.z),
            )
        }
        "SINE" => DVec3::new(first.x.sin(), first.y.sin(), first.z.sin()),
        "COSINE" => DVec3::new(first.x.cos(), first.y.cos(), first.z.cos()),
        "TANGENT" => DVec3::new(first.x.tan(), first.y.tan(), first.z.tan()),
        "SIGN" => first.signum(),
        "MULTIPLY_ADD" => first * second + third,
        other => {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("ShaderNodeVectorMath operation `{other}` is not implemented"),
                json!({"feature_id":format!("graph.shader_vector_math.{other}")}),
            ));
        }
    };
    if vector.is_finite() {
        Ok(vector_json(vector))
    } else {
        Err(evaluation("vector math produced a non-finite value"))
    }
}

fn random_i64_inclusive(state: &mut u64, minimum: i64, maximum: i64) -> Result<i64> {
    if minimum > maximum {
        return Err(evaluation("random integer minimum exceeds maximum"));
    }
    let range = u128::try_from(i128::from(maximum) - i128::from(minimum) + 1)
        .map_err(|_| evaluation("random integer range is invalid"))?;
    let offset = (u128::from(next_u64(state)) * range) >> 64;
    let value = i128::from(minimum)
        .checked_add(i128::try_from(offset).map_err(|_| evaluation("random integer overflow"))?)
        .ok_or_else(|| evaluation("random integer overflow"))?;
    i64::try_from(value).map_err(|_| evaluation("random integer overflow"))
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
        "POINT" | "POINTS" => mesh
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
        "POINT" | "POINTS" => {
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
    let mut result = Mesh::new();
    for mesh in meshes {
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
    Ok(result)
}

fn mesh_evaluation_error(error: &MeshError) -> PotError {
    PotError::new(ErrorCode::EvaluationFailed, error.to_string())
}

fn evaluation(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::EvaluationFailed, message)
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
    clippy::cast_sign_loss,
    reason = "the value is an integral rounded count bounded to the supported range"
)]
fn bounded_count_to_usize(count: f64) -> Result<usize> {
    if !count.is_finite() || !(0.0..=MAX_GENERATED_POINTS_F64).contains(&count) {
        return Err(evaluation(
            "distributed point count is outside the supported range",
        ));
    }
    Ok(count as usize)
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

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "a 53-bit mantissa is exactly representable in IEEE-754 binary64"
)]
fn next_unit(state: &mut u64) -> f64 {
    (next_u64(state) >> 11) as f64 / RANDOM_MANTISSA_RANGE
}
#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::next_unit;

    proptest! {
        #[test]
        fn seeded_sample_stream_is_repeatable_and_bounded(seed in any::<u64>()) {
            let mut first_state = seed;
            let mut second_state = seed;
            for _ in 0..16 {
                let first = next_unit(&mut first_state);
                let second = next_unit(&mut second_state);
                prop_assert_eq!(first.to_bits(), second.to_bits());
                prop_assert!((0.0..1.0).contains(&first));
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn kani_seeded_sample_stays_in_unit_interval() {
    let mut seed: u64 = kani::any();
    let value = next_unit(&mut seed);
    kani::assert(
        value >= 0.0 && value < 1.0,
        "seeded sample must be in [0, 1)",
    );
}
