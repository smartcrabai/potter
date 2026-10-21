use std::collections::{HashMap, HashSet};

use glam::{DQuat, DVec2, DVec3};
use serde_json::{Map, Value, json};

use crate::geom::modifiers::multires::{self, MultiresGridData};
use crate::{
    error::{PotError, Result},
    geom::{
        Face, Mesh, MeshError,
        attribute_keys::storage_key,
        clnor::{self, NormalCodeOptions, SetterInput},
        color_layers::{self, ColorDomain, ColorLayer},
        edge_key,
        triangulate::{NgonMethod, QuadMethod, triangulate_mesh},
        uv_layers::{self, UvLayer, UvStore},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Domain {
    Vertex,
    Edge,
    Face,
}

impl Domain {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "point" | "vertex" | "vertices" => Ok(Self::Vertex),
            "edge" | "edges" => Ok(Self::Edge),
            "face" | "faces" => Ok(Self::Face),
            _ => Err(invalid(format!("unknown element domain {value:?}"))),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertices",
            Self::Edge => "edges",
            Self::Face => "faces",
        }
    }

    const fn prefix(self) -> char {
        match self {
            Self::Vertex => 'v',
            Self::Edge => 'e',
            Self::Face => 'f',
        }
    }
}

#[derive(Debug)]
struct Selection {
    domain: Domain,
    ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
struct MdispSourceLoop {
    face_id: u32,
    corner: usize,
}

#[derive(Clone, Debug, Default)]
struct MdispLoopSource {
    seed: Option<MdispSourceLoop>,
    callback_faces: Vec<u32>,
}

#[derive(Default)]
struct IdChanges {
    vertices: Vec<u32>,
    edges: Vec<u32>,
    faces: Vec<u32>,
    deleted_vertices: Vec<u32>,
    deleted_edges: Vec<u32>,
    deleted_faces: Vec<u32>,
    mdisp_sources: HashMap<u32, Vec<MdispLoopSource>>,
    mdisp_radial_faces: HashMap<u32, Vec<u32>>,
    mdisp_pass_sources: HashMap<u32, Vec<MdispLoopSource>>,
}

impl IdChanges {
    fn record(&mut self, domain: Domain, id: u32) {
        match domain {
            Domain::Vertex => self.vertices.push(id),
            Domain::Edge => self.edges.push(id),
            Domain::Face => self.faces.push(id),
        }
    }

    fn delete(&mut self, domain: Domain, id: u32) {
        match domain {
            Domain::Vertex => self.deleted_vertices.push(id),
            Domain::Edge => self.deleted_edges.push(id),
            Domain::Face => self.deleted_faces.push(id),
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "vertices": self.vertices.iter().map(|id| format!("v{id}")).collect::<Vec<_>>(),
            "edges": self.edges.iter().map(|id| format!("e{id}")).collect::<Vec<_>>(),
            "faces": self.faces.iter().map(|id| format!("f{id}")).collect::<Vec<_>>(),
        })
    }

    fn deleted_to_json(&self) -> Value {
        json!({
            "vertices": self.deleted_vertices.iter().map(|id| format!("v{id}")).collect::<Vec<_>>(),
            "edges": self.deleted_edges.iter().map(|id| format!("e{id}")).collect::<Vec<_>>(),
            "faces": self.deleted_faces.iter().map(|id| format!("f{id}")).collect::<Vec<_>>(),
        })
    }
}

fn invalid(message: impl Into<String>) -> PotError {
    PotError::invalid_argument(message)
}

fn mesh_error(error: &MeshError) -> PotError {
    PotError::invalid_argument(error.to_string())
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{name} must be a JSON object")))
}
fn reject_unknown_fields(
    fields: &Map<String, Value>,
    allowed: &[&str],
    context: &str,
) -> Result<()> {
    if let Some(field) = fields
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(invalid(format!("unknown {context} field {field:?}")));
    }
    Ok(())
}

fn vector(value: &Value, name: &str) -> Result<DVec3> {
    let coordinates = value
        .as_array()
        .filter(|items| items.len() == 3)
        .ok_or_else(|| invalid(format!("{name} must be an array of three numbers")))?;
    let mut components = [0.0; 3];
    for (index, item) in coordinates.iter().enumerate() {
        components[index] = item
            .as_f64()
            .filter(|component| component.is_finite())
            .ok_or_else(|| invalid(format!("{name} components must be finite numbers")))?;
    }
    Ok(DVec3::from_array(components))
}

fn optional_vector(args: &Map<String, Value>, key: &str, default: DVec3) -> Result<DVec3> {
    args.get(key)
        .map_or(Ok(default), |value| vector(value, key))
}

fn number(args: &Map<String, Value>, key: &str, default: f64) -> Result<f64> {
    args.get(key).map_or(Ok(default), |value| {
        value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| invalid(format!("{key} must be a finite number")))
    })
}
fn positive_count(args: &Map<String, Value>, key: &str, default: u32) -> Result<u32> {
    let count = args.get(key).map_or(Ok(default), |value| {
        let raw = value
            .as_u64()
            .ok_or_else(|| invalid(format!("{key} must be a positive integer")))?;
        u32::try_from(raw).map_err(|_| invalid(format!("{key} is too large")))
    })?;
    if count == 0 {
        return Err(invalid(format!("{key} must be positive")));
    }
    Ok(count)
}

fn boolean(args: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    args.get(key).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| invalid(format!("{key} must be a boolean")))
    })
}

fn selection(mesh: &Mesh, args: &Map<String, Value>, required: bool) -> Result<Option<Selection>> {
    let Some(value) = args.get("elements") else {
        return if required {
            Err(invalid("elements is required"))
        } else {
            Ok(None)
        };
    };
    let fields = object(value, "elements")?;
    reject_unknown_fields(fields, &["domain", "ids", "selector"], "elements")?;
    let domain_name = fields
        .get("domain")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("elements.domain must be vertex, edge, or face"))?;
    let domain = Domain::parse(domain_name)?;
    let ids = match (fields.get("ids"), fields.get("selector")) {
        (Some(values), None) => parse_selection_ids(mesh, domain, values)?,
        (None, Some(selector)) => resolve_selector(mesh, domain, selector)?,
        (Some(_), Some(_)) => {
            return Err(invalid(
                "elements must specify either ids or selector, not both",
            ));
        }
        (None, None) => return Err(invalid("elements requires ids or selector")),
    };
    if ids.is_empty() && fields.contains_key("ids") {
        return Err(invalid("elements.ids must not be empty"));
    }
    Ok(Some(Selection { domain, ids }))
}

fn parse_selection_ids(mesh: &Mesh, domain: Domain, value: &Value) -> Result<Vec<u32>> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid("elements.ids must be an array of persistent IDs"))?;
    let mut ids = Vec::with_capacity(values.len());
    let mut seen = HashSet::with_capacity(values.len());
    for value in values {
        let value = value
            .as_str()
            .ok_or_else(|| invalid("element IDs must be strings"))?;
        let numeric = value
            .strip_prefix(domain.prefix())
            .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
            .ok_or_else(|| {
                invalid(format!(
                    "ID {value:?} does not match the {} domain",
                    domain.name()
                ))
            })?;
        let id = numeric
            .parse::<u32>()
            .map_err(|_| invalid(format!("element ID {value:?} is outside the u32 range")))?;
        if !seen.insert(id) {
            return Err(invalid(format!("duplicate element ID {value:?}")));
        }
        let exists = match domain {
            Domain::Vertex => mesh.vertices.iter().any(|item| item.id == id),
            Domain::Edge => mesh.edges.iter().any(|item| item.id == id),
            Domain::Face => mesh.faces.iter().any(|item| item.id == id),
        };
        if !exists {
            return Err(invalid(format!("element ID {value:?} does not exist")));
        }
        ids.push(id);
    }
    Ok(ids)
}

fn resolve_selector(mesh: &Mesh, domain: Domain, value: &Value) -> Result<Vec<u32>> {
    let fields = object(value, "elements.selector")?;
    let kind = fields
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("elements.selector.type is required"))?;
    match kind {
        "position_box" => {
            reject_unknown_fields(fields, &["type", "min", "max"], "position_box selector")?;
            let minimum = vector(
                fields
                    .get("min")
                    .ok_or_else(|| invalid("position_box selector requires min"))?,
                "selector.min",
            )?;
            let maximum = vector(
                fields
                    .get("max")
                    .ok_or_else(|| invalid("position_box selector requires max"))?,
                "selector.max",
            )?;
            if (0..3).any(|axis| minimum[axis] > maximum[axis]) {
                return Err(invalid("position_box min must not exceed max"));
            }
            Ok(element_ids(mesh, domain)
                .into_iter()
                .filter(|id| {
                    let point = element_center(mesh, domain, *id);
                    (0..3).all(|axis| point[axis] >= minimum[axis] && point[axis] <= maximum[axis])
                })
                .collect())
        }
        "position_sphere" => {
            reject_unknown_fields(
                fields,
                &["type", "center", "radius"],
                "position_sphere selector",
            )?;
            let center = vector(
                fields
                    .get("center")
                    .ok_or_else(|| invalid("position_sphere selector requires center"))?,
                "selector.center",
            )?;
            let radius = selector_number(fields, "radius")?;
            if radius < 0.0 {
                return Err(invalid("position_sphere radius must be non-negative"));
            }
            Ok(element_ids(mesh, domain)
                .into_iter()
                .filter(|id| element_center(mesh, domain, *id).distance(center) <= radius)
                .collect())
        }
        "normal_cone" => {
            reject_unknown_fields(fields, &["type", "axis", "angle"], "normal_cone selector")?;
            let axis = vector(
                fields
                    .get("axis")
                    .ok_or_else(|| invalid("normal_cone selector requires axis"))?,
                "selector.axis",
            )?;
            if axis.length_squared() <= f64::EPSILON {
                return Err(invalid("normal_cone axis must be non-zero"));
            }
            let angle = selector_number(fields, "angle")?;
            if !(0.0..=std::f64::consts::PI).contains(&angle) {
                return Err(invalid(
                    "normal_cone angle must be between zero and pi radians",
                ));
            }
            let axis = axis.normalize();
            Ok(element_ids(mesh, domain)
                .into_iter()
                .filter(|id| element_normal(mesh, domain, *id).dot(axis) >= angle.cos())
                .collect())
        }
        "attribute" => {
            reject_unknown_fields(
                fields,
                &["type", "name", "operator", "value"],
                "attribute selector",
            )?;
            let name = fields
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| invalid("attribute selector requires a non-empty name"))?;
            let operator = fields
                .get("operator")
                .and_then(Value::as_str)
                .unwrap_or("equals");
            if ![
                "equals",
                "less_than",
                "less_than_or_equal",
                "greater_than",
                "greater_than_or_equal",
            ]
            .contains(&operator)
            {
                return Err(invalid(format!(
                    "unknown attribute selector operator {operator:?}"
                )));
            }
            let expected = fields
                .get("value")
                .ok_or_else(|| invalid("attribute selector requires value"))?;
            let values = mesh
                .attributes
                .get(&*storage_key(name))
                .and_then(Value::as_object)
                .and_then(|attribute| attribute.get("values"))
                .and_then(Value::as_object)
                .ok_or_else(|| invalid(format!("attribute `{name}` has no value map")))?;
            let mut selected = Vec::new();
            for id in element_ids(mesh, domain) {
                let key = format!("{}{id}", domain.prefix());
                if values
                    .get(&key)
                    .is_some_and(|actual| attribute_matches(actual, expected, operator))
                {
                    selected.push(id);
                }
            }
            Ok(selected)
        }
        "connected" | "linked" => {
            reject_unknown_fields(fields, &["type", "ids"], "connected selector")?;
            let seeds = parse_selection_ids(
                mesh,
                domain,
                fields
                    .get("ids")
                    .ok_or_else(|| invalid("connected selector requires seed ids"))?,
            )?;
            if seeds.is_empty() {
                return Err(invalid("connected selector requires at least one seed ID"));
            }
            connected_elements(mesh, domain, &seeds)
        }
        "material_index" => {
            reject_unknown_fields(fields, &["type", "index"], "material_index selector")?;
            if domain != Domain::Face {
                return Err(invalid("material_index selector requires the face domain"));
            }
            let index = fields
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|index| u32::try_from(index).ok())
                .ok_or_else(|| invalid("material_index selector requires a u32 index"))?;
            Ok(mesh
                .faces
                .iter()
                .filter(|face| face.material_index == index)
                .map(|face| face.id)
                .collect())
        }
        _ => Err(invalid(format!("unknown element selector type {kind:?}"))),
    }
}

fn selector_number(fields: &Map<String, Value>, name: &str) -> Result<f64> {
    fields
        .get(name)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid(format!("selector.{name} must be a finite number")))
}

fn element_ids(mesh: &Mesh, domain: Domain) -> Vec<u32> {
    match domain {
        Domain::Vertex => mesh.vertices.iter().map(|item| item.id).collect(),
        Domain::Edge => mesh.edges.iter().map(|item| item.id).collect(),
        Domain::Face => mesh.faces.iter().map(|item| item.id).collect(),
    }
}

fn element_center(mesh: &Mesh, domain: Domain, id: u32) -> DVec3 {
    match domain {
        Domain::Vertex => mesh.vertex(id).map_or(DVec3::ZERO, |item| item.co),
        Domain::Edge => mesh
            .edges
            .iter()
            .find(|edge| edge.id == id)
            .and_then(|edge| {
                Some((mesh.vertex(edge.vertices[0])?.co + mesh.vertex(edge.vertices[1])?.co) * 0.5)
            })
            .unwrap_or(DVec3::ZERO),
        Domain::Face => mesh
            .faces
            .iter()
            .find(|face| face.id == id)
            .map_or(DVec3::ZERO, |face| {
                let mut center = DVec3::ZERO;
                for vertex in &face.vertices {
                    center += mesh.vertex(*vertex).map_or(DVec3::ZERO, |item| item.co);
                }
                let count = u32::try_from(face.vertices.len()).map_or(f64::INFINITY, f64::from);
                center / count
            }),
    }
}

fn element_normal(mesh: &Mesh, domain: Domain, id: u32) -> DVec3 {
    if domain == Domain::Face {
        return mesh
            .faces
            .iter()
            .find(|face| face.id == id)
            .and_then(|face| face_normal(mesh, face).ok())
            .unwrap_or(DVec3::Z);
    }
    let edge_vertices = if domain == Domain::Edge {
        mesh.edges
            .iter()
            .find(|edge| edge.id == id)
            .map(|edge| edge.vertices)
    } else {
        None
    };
    let touches_element = |vertex: &u32| match domain {
        Domain::Vertex => *vertex == id,
        Domain::Edge => edge_vertices.is_some_and(|vertices| vertices.contains(vertex)),
        Domain::Face => false,
    };
    let mut normal = DVec3::ZERO;
    for face in &mesh.faces {
        if face.vertices.iter().any(touches_element) {
            normal += face_normal(mesh, face).unwrap_or(DVec3::Z);
        }
    }
    if normal.length_squared() <= f64::EPSILON {
        DVec3::Z
    } else {
        normal.normalize()
    }
}

fn attribute_matches(actual: &Value, expected: &Value, operator: &str) -> bool {
    match operator {
        "equals" => actual == expected,
        "less_than" => actual
            .as_f64()
            .zip(expected.as_f64())
            .is_some_and(|(a, b)| a < b),
        "less_than_or_equal" => actual
            .as_f64()
            .zip(expected.as_f64())
            .is_some_and(|(a, b)| a <= b),
        "greater_than" => actual
            .as_f64()
            .zip(expected.as_f64())
            .is_some_and(|(a, b)| a > b),
        "greater_than_or_equal" => actual
            .as_f64()
            .zip(expected.as_f64())
            .is_some_and(|(a, b)| a >= b),
        _ => false,
    }
}

fn connected_elements(mesh: &Mesh, domain: Domain, seeds: &[u32]) -> Result<Vec<u32>> {
    let mut selected: HashSet<u32> = seeds.iter().copied().collect();
    let mut frontier = seeds.to_vec();
    while let Some(current) = frontier.pop() {
        let neighbors = match domain {
            Domain::Vertex => mesh
                .edges
                .iter()
                .filter(|edge| edge.vertices.contains(&current))
                .flat_map(|edge| edge.vertices)
                .collect::<Vec<_>>(),
            Domain::Edge => {
                let edge = mesh
                    .edges
                    .iter()
                    .find(|edge| edge.id == current)
                    .ok_or_else(|| invalid("connected selector references a missing edge"))?;
                mesh.edges
                    .iter()
                    .filter(|candidate| {
                        candidate.id != current
                            && candidate
                                .vertices
                                .iter()
                                .any(|vertex| edge.vertices.contains(vertex))
                    })
                    .map(|candidate| candidate.id)
                    .collect()
            }
            Domain::Face => {
                let face = mesh
                    .faces
                    .iter()
                    .find(|face| face.id == current)
                    .ok_or_else(|| invalid("connected selector references a missing face"))?;
                mesh.faces
                    .iter()
                    .filter(|candidate| {
                        candidate.id != current
                            && candidate
                                .vertices
                                .iter()
                                .filter(|vertex| face.vertices.contains(vertex))
                                .count()
                                >= 2
                    })
                    .map(|candidate| candidate.id)
                    .collect()
            }
        };
        for neighbor in neighbors {
            if selected.insert(neighbor) {
                frontier.push(neighbor);
            }
        }
    }
    let mut ids = selected.into_iter().collect::<Vec<_>>();
    ids.sort_unstable();
    Ok(ids)
}

fn required_selection(mesh: &Mesh, args: &Map<String, Value>) -> Result<Selection> {
    selection(mesh, args, true)?.ok_or_else(|| invalid("elements is required"))
}

fn insert_vertex(mesh: &mut Mesh, position: DVec3, changes: &mut IdChanges) -> Result<u32> {
    let id = mesh
        .insert_vertex(position)
        .map_err(|error| mesh_error(&error))?;
    changes.record(Domain::Vertex, id);
    Ok(id)
}

fn insert_face(
    mesh: &mut Mesh,
    vertices: Vec<u32>,
    material_index: u32,
    changes: &mut IdChanges,
) -> Result<u32> {
    let first_edge_id = mesh.next_id.edge;
    let id = mesh
        .insert_face(vertices, material_index)
        .map_err(|error| mesh_error(&error))?;
    changes.record(Domain::Face, id);
    for edge_id in first_edge_id..mesh.next_id.edge {
        changes.record(Domain::Edge, edge_id);
    }
    Ok(id)
}

fn mdisp_callback_faces(lineage: &IdChanges, face_id: u32) -> Vec<u32> {
    let Some(loops) = lineage.mdisp_sources.get(&face_id) else {
        return vec![face_id];
    };
    let mut faces = Vec::new();
    let mut seen_sequences = HashSet::new();
    for source in loops {
        let parents = if source.callback_faces.is_empty() {
            source.seed.map_or_else(Vec::new, |seed| vec![seed.face_id])
        } else {
            source.callback_faces.clone()
        };
        if seen_sequences.insert(parents.clone()) {
            faces.extend(parents);
        }
    }
    faces
}

fn mdisp_face_sources(
    lineage: &IdChanges,
    seeds: &[Option<MdispSourceLoop>],
    callback_parents: &[u32],
) -> Vec<MdispLoopSource> {
    let callback_faces = callback_parents
        .iter()
        .flat_map(|face_id| mdisp_callback_faces(lineage, *face_id))
        .collect::<Vec<_>>();
    mdisp_face_sources_from_callbacks(lineage, seeds, &callback_faces)
}

// Preserve Blender callback order; later source faces can overwrite earlier grid samples.
fn mdisp_face_sources_from_callbacks(
    lineage: &IdChanges,
    seeds: &[Option<MdispSourceLoop>],
    callback_faces: &[u32],
) -> Vec<MdispLoopSource> {
    seeds
        .iter()
        .map(|seed| {
            let seed = (*seed).and_then(|seed| {
                lineage
                    .mdisp_sources
                    .get(&seed.face_id)
                    .and_then(|loops| loops.get(seed.corner))
                    .map_or(Some(seed), |source| source.seed)
            });
            MdispLoopSource {
                seed,
                callback_faces: callback_faces.to_vec(),
            }
        })
        .collect()
}

fn insert_face_with_mdisp_sources(
    mesh: &mut Mesh,
    vertices: Vec<u32>,
    material_index: u32,
    changes: &mut IdChanges,
    sources: Vec<MdispLoopSource>,
) -> Result<u32> {
    let id = insert_face(mesh, vertices, material_index, changes)?;
    changes.mdisp_sources.insert(id, sources);
    Ok(id)
}

fn mdisp_sources_from_parent(
    lineage: &IdChanges,
    parent: &Face,
    seed_corners: &[Option<usize>],
    interpolate: bool,
) -> Vec<MdispLoopSource> {
    let seeds = seed_corners
        .iter()
        .map(|corner| {
            corner.map(|corner| MdispSourceLoop {
                face_id: parent.id,
                corner,
            })
        })
        .collect::<Vec<_>>();
    let callback_face = parent.id;
    let callbacks = if interpolate {
        std::slice::from_ref(&callback_face)
    } else {
        &[]
    };
    mdisp_face_sources(lineage, &seeds, callbacks)
}

fn source_corners(parent: &Face, target_vertices: &[u32]) -> Vec<Option<usize>> {
    target_vertices
        .iter()
        .map(|vertex_id| parent.vertices.iter().position(|id| id == vertex_id))
        .collect()
}

fn insert_face_from_parent(
    mesh: &mut Mesh,
    vertices: Vec<u32>,
    material_index: u32,
    changes: &mut IdChanges,
    parent: &Face,
    seed_corners: &[Option<usize>],
    interpolate: bool,
) -> Result<u32> {
    let sources = mdisp_sources_from_parent(changes, parent, seed_corners, interpolate);
    insert_face_with_mdisp_sources(mesh, vertices, material_index, changes, sources)
}

fn remove_face(mesh: &mut Mesh, id: u32, changes: &mut IdChanges) {
    if let Some(index) = mesh.faces.iter().position(|face| face.id == id) {
        mesh.faces.remove(index);
        changes.delete(Domain::Face, id);
    }
}

fn remove_faces(mesh: &mut Mesh, ids: &[u32], changes: &mut IdChanges) {
    for id in ids {
        remove_face(mesh, *id, changes);
    }
}

#[derive(Clone, Copy)]
enum ProportionalFalloff {
    Smooth,
    Sphere,
    Root,
    Sharp,
    Linear,
    Constant,
}

fn proportional_weights(
    mesh: &Mesh,
    seeds: &HashSet<u32>,
    radius: f64,
    falloff: ProportionalFalloff,
    connected_only: bool,
) -> HashMap<u32, f64> {
    let mut distances = HashMap::<u32, f64>::with_capacity(mesh.vertices.len());
    if connected_only {
        let mut visited = HashSet::with_capacity(mesh.vertices.len());
        for id in seeds {
            distances.insert(*id, 0.0);
        }
        loop {
            let next = mesh
                .vertices
                .iter()
                .filter(|vertex| !visited.contains(&vertex.id))
                .filter_map(|vertex| {
                    distances
                        .get(&vertex.id)
                        .map(|distance| (vertex.id, *distance))
                })
                .min_by(|left, right| left.1.total_cmp(&right.1));
            let Some((current, current_distance)) = next else {
                break;
            };
            visited.insert(current);
            for edge in mesh
                .edges
                .iter()
                .filter(|edge| edge.vertices.contains(&current))
            {
                let neighbor = if edge.vertices[0] == current {
                    edge.vertices[1]
                } else {
                    edge.vertices[0]
                };
                let neighbor_position = mesh
                    .vertex(neighbor)
                    .map_or(DVec3::ZERO, |vertex| vertex.co);
                let current_position = mesh.vertex(current).map_or(DVec3::ZERO, |vertex| vertex.co);
                let candidate = current_distance + current_position.distance(neighbor_position);
                if candidate < distances.get(&neighbor).copied().unwrap_or(f64::INFINITY) {
                    distances.insert(neighbor, candidate);
                }
            }
        }
    }
    let seed_positions = mesh
        .vertices
        .iter()
        .filter(|vertex| seeds.contains(&vertex.id))
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    mesh.vertices
        .iter()
        .filter_map(|vertex| {
            let distance = if connected_only {
                distances.get(&vertex.id).copied()
            } else {
                seed_positions
                    .iter()
                    .map(|position| vertex.co.distance(*position))
                    .min_by(f64::total_cmp)
            }?;
            if distance > radius {
                return None;
            }
            let t = (distance / radius).clamp(0.0, 1.0);
            let weight = match falloff {
                ProportionalFalloff::Smooth => 1.0 - t * t * (3.0 - 2.0 * t),
                ProportionalFalloff::Sphere => (1.0 - t * t).sqrt(),
                ProportionalFalloff::Root => 1.0 - t.sqrt(),
                ProportionalFalloff::Sharp => (1.0 - t) * (1.0 - t),
                ProportionalFalloff::Linear => 1.0 - t,
                ProportionalFalloff::Constant => 1.0,
            };
            (weight > 0.0).then_some((vertex.id, weight))
        })
        .collect()
}

fn parse_proportional(
    args: &Map<String, Value>,
) -> Result<Option<(f64, ProportionalFalloff, bool)>> {
    let Some(value) = args.get("proportional") else {
        return Ok(None);
    };
    let fields = object(value, "proportional")?;
    reject_unknown_fields(
        fields,
        &["radius", "falloff", "connected_only"],
        "proportional",
    )?;
    let radius = fields
        .get("radius")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| invalid("proportional.radius must be a positive finite number"))?;
    let falloff = match fields
        .get("falloff")
        .and_then(Value::as_str)
        .unwrap_or("smooth")
    {
        "smooth" => ProportionalFalloff::Smooth,
        "sphere" => ProportionalFalloff::Sphere,
        "root" => ProportionalFalloff::Root,
        "sharp" => ProportionalFalloff::Sharp,
        "linear" => ProportionalFalloff::Linear,
        "constant" => ProportionalFalloff::Constant,
        other => return Err(invalid(format!("unknown proportional falloff {other:?}"))),
    };
    let connected_only = fields.get("connected_only").map_or(Ok(false), |value| {
        value
            .as_bool()
            .ok_or_else(|| invalid("proportional.connected_only must be a boolean"))
    })?;
    Ok(Some((radius, falloff, connected_only)))
}

fn apply_transform(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let selected = required_selection(mesh, args)?;
    let translation = optional_vector(args, "translation", DVec3::ZERO)?;
    let rotation = optional_vector(args, "rotation", DVec3::ZERO)?;
    let scale = optional_vector(args, "scale", DVec3::ONE)?;
    let pivot = optional_vector(args, "pivot", DVec3::ZERO)?;
    if !["translation", "rotation", "scale"]
        .iter()
        .any(|key| args.contains_key(*key))
    {
        return Err(invalid(
            "transform_elements requires translation, rotation, or scale",
        ));
    }
    let mut vertex_ids = HashSet::new();
    match selected.domain {
        Domain::Vertex => vertex_ids.extend(selected.ids),
        Domain::Edge => {
            for edge in &mesh.edges {
                if selected.ids.contains(&edge.id) {
                    vertex_ids.extend(edge.vertices);
                }
            }
        }
        Domain::Face => {
            for face in &mesh.faces {
                if selected.ids.contains(&face.id) {
                    vertex_ids.extend(&face.vertices);
                }
            }
        }
    }
    let weights = parse_proportional(args)?.map(|(radius, falloff, connected_only)| {
        proportional_weights(mesh, &vertex_ids, radius, falloff, connected_only)
    });
    for vertex in &mut mesh.vertices {
        let weight = weights
            .as_ref()
            .and_then(|weights| weights.get(&vertex.id))
            .copied()
            .unwrap_or_else(|| {
                if vertex_ids.contains(&vertex.id) {
                    1.0
                } else {
                    0.0
                }
            });
        if weight <= 0.0 {
            continue;
        }
        let original = vertex.co;
        let mut position = (original - pivot) * scale;
        let (sin_x, cos_x) = rotation.x.sin_cos();
        let (sin_y, cos_y) = rotation.y.sin_cos();
        let (sin_z, cos_z) = rotation.z.sin_cos();
        position = DVec3::new(
            position.x,
            position.y * cos_x - position.z * sin_x,
            position.y * sin_x + position.z * cos_x,
        );
        position = DVec3::new(
            position.x * cos_y + position.z * sin_y,
            position.y,
            -position.x * sin_y + position.z * cos_y,
        );
        position = DVec3::new(
            position.x * cos_z - position.y * sin_z,
            position.x * sin_z + position.y * cos_z,
            position.z,
        );
        let transformed = position + pivot + translation;
        if !transformed.is_finite() {
            return Err(invalid("transform produced a non-finite vertex position"));
        }
        vertex.co = original.lerp(transformed, weight);
    }
    Ok(IdChanges::default())
}

fn selected_vertex_ids(mesh: &Mesh, selected: &Selection) -> HashSet<u32> {
    match selected.domain {
        Domain::Vertex => selected.ids.iter().copied().collect(),
        Domain::Edge => mesh
            .edges
            .iter()
            .filter(|edge| selected.ids.contains(&edge.id))
            .flat_map(|edge| edge.vertices)
            .collect(),
        Domain::Face => mesh
            .faces
            .iter()
            .filter(|face| selected.ids.contains(&face.id))
            .flat_map(|face| face.vertices.iter().copied())
            .collect(),
    }
}

fn apply_vertex_slide(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let selected = required_selection(mesh, args)?;
    if selected.domain != Domain::Vertex {
        return Err(invalid("vertex_slide requires vertex elements"));
    }
    let factor = number(args, "factor", 0.0)?;
    if !(-1.0..=1.0).contains(&factor) {
        return Err(invalid("vertex_slide factor must be between -1 and 1"));
    }
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut updates = Vec::with_capacity(selected.ids.len());
    for id in selected.ids {
        let neighbor = mesh
            .edges
            .iter()
            .filter_map(|edge| {
                if edge.vertices[0] == id {
                    Some(edge.vertices[1])
                } else if edge.vertices[1] == id {
                    Some(edge.vertices[0])
                } else {
                    None
                }
            })
            .min()
            .ok_or_else(|| invalid(format!("vertex v{id} has no adjacent edge to slide along")))?;
        let position = positions[&id].lerp(positions[&neighbor], factor);
        updates.push((id, position));
    }
    for (id, position) in updates {
        mesh.vertices
            .iter_mut()
            .find(|vertex| vertex.id == id)
            .ok_or_else(|| invalid(format!("vertex v{id} does not exist")))?
            .co = position;
    }
    Ok(IdChanges::default())
}

fn apply_edge_slide(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let selected = required_selection(mesh, args)?;
    if selected.domain != Domain::Edge {
        return Err(invalid("edge_slide requires edge elements"));
    }
    let factor = number(args, "factor", 0.5)?;
    if !(0.0..=1.0).contains(&factor) {
        return Err(invalid("edge_slide factor must be between zero and one"));
    }
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut updates: HashMap<u32, (DVec3, u32)> = HashMap::new();
    for edge in mesh
        .edges
        .iter()
        .filter(|edge| selected.ids.contains(&edge.id))
    {
        let face = mesh
            .faces
            .iter()
            .filter(|face| face.vertices.len() == 4)
            .filter(|face| {
                (0..face.vertices.len()).any(|index| {
                    edge_key(
                        face.vertices[index],
                        face.vertices[(index + 1) % face.vertices.len()],
                    ) == edge_key(edge.vertices[0], edge.vertices[1])
                })
            })
            .min_by_key(|face| face.id)
            .ok_or_else(|| {
                invalid(format!(
                    "edge e{} has no adjacent quad to slide across",
                    edge.id
                ))
            })?;
        let index = (0..face.vertices.len())
            .find(|index| {
                edge_key(face.vertices[*index], face.vertices[(*index + 1) % 4])
                    == edge_key(edge.vertices[0], edge.vertices[1])
            })
            .ok_or_else(|| {
                invalid(format!(
                    "edge e{} is not part of its adjacent face",
                    edge.id
                ))
            })?;
        let first = face.vertices[index];
        let second = face.vertices[(index + 1) % 4];
        let opposite_first = face.vertices[(index + 3) % 4];
        let opposite_second = face.vertices[(index + 2) % 4];
        for (id, opposite) in [(first, opposite_first), (second, opposite_second)] {
            let position = positions[&id].lerp(positions[&opposite], factor);
            let entry = updates.entry(id).or_insert((DVec3::ZERO, 0));
            entry.0 += position;
            entry.1 += 1;
        }
    }
    for (id, (position, count)) in updates {
        let divisor = f64::from(count);
        mesh.vertices
            .iter_mut()
            .find(|vertex| vertex.id == id)
            .ok_or_else(|| invalid(format!("vertex v{id} does not exist")))?
            .co = position / divisor;
    }
    Ok(IdChanges::default())
}

fn apply_spin_screw_faces(
    mesh: &mut Mesh,
    selected: &Selection,
    axis: DVec3,
    angle: f64,
    center: DVec3,
    distance: f64,
    steps: u32,
) -> Result<IdChanges> {
    if steps > 64 {
        return Err(invalid("spin and screw support at most 64 steps"));
    }
    let faces = mesh
        .faces
        .iter()
        .filter(|face| selected.ids.contains(&face.id))
        .cloned()
        .collect::<Vec<_>>();
    let mut changes = IdChanges::default();
    for face in faces {
        let source_positions = face
            .vertices
            .iter()
            .map(|id| {
                mesh.vertex(*id)
                    .map(|vertex| vertex.co)
                    .ok_or_else(|| invalid("face references a missing vertex"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut previous = face.vertices.clone();
        remove_face(mesh, face.id, &mut changes);
        for step in 1..=steps {
            let fraction = f64::from(step) / f64::from(steps);
            let rotation = DQuat::from_axis_angle(axis, angle * fraction);
            let translation = axis * (distance * fraction);
            let mut current = Vec::with_capacity(face.vertices.len());
            for position in &source_positions {
                let transformed = center + rotation * (*position - center) + translation;
                if !transformed.is_finite() {
                    return Err(invalid("spin or screw produced a non-finite position"));
                }
                current.push(insert_vertex(mesh, transformed, &mut changes)?);
            }
            for index in 0..face.vertices.len() {
                let next = (index + 1) % face.vertices.len();
                let source_corners = [Some(index), Some(next), Some(next), Some(index)];
                insert_face_from_parent(
                    mesh,
                    vec![
                        previous[index],
                        previous[next],
                        current[next],
                        current[index],
                    ],
                    face.material_index,
                    &mut changes,
                    &face,
                    &source_corners,
                    false,
                )?;
            }
            previous = current;
        }
        let source_corners = (0..face.vertices.len()).map(Some).collect::<Vec<_>>();
        insert_face_from_parent(
            mesh,
            previous,
            face.material_index,
            &mut changes,
            &face,
            &source_corners,
            false,
        )?;
    }
    Ok(changes)
}

fn apply_spin_or_screw(
    mesh: &mut Mesh,
    args: &Map<String, Value>,
    screw: bool,
) -> Result<IdChanges> {
    let selected = required_selection(mesh, args)?;
    if !args.contains_key("axis") || !args.contains_key("angle") {
        return Err(invalid("spin and screw require axis and angle"));
    }
    if screw && !args.contains_key("distance") {
        return Err(invalid("screw requires distance"));
    }
    let axis = optional_vector(args, "axis", DVec3::Z)?;
    if axis.length_squared() <= f64::EPSILON {
        return Err(invalid("spin axis must be non-zero"));
    }
    let axis = axis.normalize();
    let angle = number(args, "angle", 0.0)?;
    let center = optional_vector(args, "center", DVec3::ZERO)?;
    let distance = number(args, "distance", 0.0)?;
    if selected.domain == Domain::Face {
        return apply_spin_screw_faces(
            mesh,
            &selected,
            axis,
            angle,
            center,
            if screw { distance } else { 0.0 },
            positive_count(args, "steps", 9)?,
        );
    }
    let vertex_ids = selected_vertex_ids(mesh, &selected);
    let rotation = DQuat::from_axis_angle(axis, angle);
    let translation = if screw {
        axis.normalize() * distance
    } else {
        DVec3::ZERO
    };
    for vertex in &mut mesh.vertices {
        if vertex_ids.contains(&vertex.id) {
            vertex.co = center + rotation * (vertex.co - center) + translation;
            if !vertex.co.is_finite() {
                return Err(invalid("spin produced a non-finite vertex position"));
            }
        }
    }
    Ok(IdChanges::default())
}

fn apply_merge(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let selected = required_selection(mesh, args)?;
    if selected.domain != Domain::Vertex {
        return Err(invalid("merge requires vertex elements"));
    }
    if selected.ids.is_empty() {
        return Err(invalid("merge requires at least one selected vertex"));
    }
    let mode = args.get("mode").and_then(Value::as_str).unwrap_or("center");
    let destination = match mode {
        "center" => {
            let mut center = DVec3::ZERO;
            for id in &selected.ids {
                center += mesh
                    .vertex(*id)
                    .ok_or_else(|| invalid(format!("vertex v{id} does not exist")))?
                    .co;
            }
            let count = u32::try_from(selected.ids.len())
                .map_err(|_| invalid("too many selected vertices"))?;
            center / f64::from(count)
        }
        "cursor" => {
            if !args.contains_key("cursor") {
                return Err(invalid("merge at cursor requires cursor"));
            }
            optional_vector(args, "cursor", DVec3::ZERO)?
        }
        _ => return Err(invalid("merge mode must be center or cursor")),
    };
    for vertex in &mut mesh.vertices {
        if selected.ids.contains(&vertex.id) {
            vertex.co = destination;
        }
    }
    apply_weld(mesh, &selected, 0.0)
}
fn face_normal(mesh: &Mesh, face: &Face) -> Result<DVec3> {
    let mut normal = DVec3::ZERO;
    for index in 0..face.vertices.len() {
        let current = mesh
            .vertex(face.vertices[index])
            .ok_or_else(|| invalid("face references a missing vertex"))?
            .co;
        let next = mesh
            .vertex(face.vertices[(index + 1) % face.vertices.len()])
            .ok_or_else(|| invalid("face references a missing vertex"))?
            .co;
        normal += current.cross(next);
    }
    if normal.length_squared() <= f64::EPSILON {
        return Err(invalid("cannot edit a degenerate face"));
    }
    Ok(normal.normalize())
}
fn apply_bevel_edges(mesh: &mut Mesh, selected_ids: &[u32], width: f64) -> Result<IdChanges> {
    if width <= 0.0 {
        return Err(invalid("bevel width must be positive"));
    }
    let edges: Vec<_> = mesh
        .edges
        .iter()
        .filter(|edge| selected_ids.contains(&edge.id))
        .cloned()
        .collect();
    let mut directions = HashMap::with_capacity(edges.len());
    for edge in &edges {
        let mut normal = DVec3::ZERO;
        for face in &mesh.faces {
            let adjacent = face.vertices.iter().enumerate().any(|(index, id)| {
                edge_key(*id, face.vertices[(index + 1) % face.vertices.len()])
                    == edge_key(edge.vertices[0], edge.vertices[1])
            });
            if adjacent {
                normal += face_normal(mesh, face)?;
            }
        }
        if normal.length_squared() <= f64::EPSILON {
            let first = mesh
                .vertex(edge.vertices[0])
                .ok_or_else(|| invalid("edge references a missing vertex"))?
                .co;
            let second = mesh
                .vertex(edge.vertices[1])
                .ok_or_else(|| invalid("edge references a missing vertex"))?
                .co;
            let direction = second - first;
            let perpendicular_x = direction.cross(DVec3::X);
            let perpendicular = if perpendicular_x.length_squared() > f64::EPSILON {
                perpendicular_x
            } else {
                direction.cross(DVec3::Y)
            };
            if perpendicular.length_squared() <= f64::EPSILON {
                return Err(invalid("cannot bevel a zero-length edge"));
            }
            normal = perpendicular;
        }
        directions.insert(edge.id, normal.normalize());
    }
    let edge_ids: HashSet<_> = edges.iter().map(|edge| edge.id).collect();
    let mut changes = IdChanges::default();
    let lineage = IdChanges::default();
    let midpoints = split_edges(mesh, &edge_ids, &mut changes, &lineage, None, None)?;
    for edge in edges {
        let midpoint = midpoints[&edge_key(edge.vertices[0], edge.vertices[1])];
        let vertex = mesh
            .vertices
            .iter_mut()
            .find(|vertex| vertex.id == midpoint)
            .ok_or_else(|| invalid("bevel midpoint is missing"))?;
        let bevel_position = vertex.co + directions[&edge.id] * width;
        if !bevel_position.is_finite() {
            return Err(invalid("bevel produced a non-finite vertex position"));
        }
        vertex.co = bevel_position;
    }
    Ok(changes)
}

fn apply_extrude_individual(
    mesh: &mut Mesh,
    faces: &[Face],
    explicit_offset: Option<DVec3>,
    distance: f64,
) -> Result<IdChanges> {
    let mut changes = IdChanges::default();
    for face in faces {
        let offset = if let Some(offset) = explicit_offset {
            offset
        } else {
            face_normal(mesh, face)? * distance
        };
        if offset.length_squared() <= f64::EPSILON {
            return Err(invalid("extrude offset must be non-zero"));
        }
        let mut duplicated = HashMap::with_capacity(face.vertices.len());
        for id in &face.vertices {
            let position = mesh
                .vertex(*id)
                .ok_or_else(|| invalid("face references a missing vertex"))?
                .co;
            duplicated.insert(*id, insert_vertex(mesh, position + offset, &mut changes)?);
        }
        remove_face(mesh, face.id, &mut changes);
        let cap = face.vertices.iter().map(|id| duplicated[id]).collect();
        let source_corners = (0..face.vertices.len()).map(Some).collect::<Vec<_>>();
        insert_face_from_parent(
            mesh,
            cap,
            face.material_index,
            &mut changes,
            face,
            &source_corners,
            false,
        )?;
        for index in 0..face.vertices.len() {
            let next = (index + 1) % face.vertices.len();
            let first = face.vertices[index];
            let second = face.vertices[next];
            let source_corners = [Some(index), Some(next), Some(next), Some(index)];
            insert_face_from_parent(
                mesh,
                vec![first, second, duplicated[&second], duplicated[&first]],
                face.material_index,
                &mut changes,
                face,
                &source_corners,
                false,
            )?;
        }
    }
    Ok(changes)
}

fn apply_extrude(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let selected = required_selection(mesh, args)?;
    if selected.domain != Domain::Face {
        return Err(invalid("extrude requires face elements"));
    }
    let explicit_offset = args
        .get("offset")
        .map(|value| vector(value, "offset"))
        .transpose()?;
    let distance = number(args, "distance", 0.0)?;
    let faces: Vec<_> = mesh
        .faces
        .iter()
        .filter(|face| selected.ids.contains(&face.id))
        .cloned()
        .collect();
    if boolean(args, "individual", false)? {
        return apply_extrude_individual(mesh, &faces, explicit_offset, distance);
    }
    let offset = if let Some(offset) = explicit_offset {
        offset
    } else if args.contains_key("distance") {
        let mut normal = DVec3::ZERO;
        for face in &faces {
            normal += face_normal(mesh, face)?;
        }
        if normal.length_squared() <= f64::EPSILON {
            return Err(invalid(
                "selected faces do not have a stable extrusion normal",
            ));
        }
        normal.normalize() * distance
    } else {
        return Err(invalid("extrude requires offset or distance"));
    };
    if offset.length_squared() <= f64::EPSILON {
        return Err(invalid("extrude offset must be non-zero"));
    }
    let mut changes = IdChanges::default();
    let mut selected_vertices = HashMap::new();
    for face in &faces {
        for id in &face.vertices {
            if !selected_vertices.contains_key(id) {
                let position = mesh
                    .vertex(*id)
                    .ok_or_else(|| invalid("face references a missing vertex"))?
                    .co;
                let new_id = insert_vertex(mesh, position + offset, &mut changes)?;
                selected_vertices.insert(*id, new_id);
            }
        }
    }
    remove_faces(
        mesh,
        &faces.iter().map(|face| face.id).collect::<Vec<_>>(),
        &mut changes,
    );
    let mut boundary_counts: HashMap<(u32, u32), (usize, [u32; 2])> = HashMap::new();
    for face in &faces {
        for index in 0..face.vertices.len() {
            let pair = [
                face.vertices[index],
                face.vertices[(index + 1) % face.vertices.len()],
            ];
            let count = boundary_counts
                .entry(edge_key(pair[0], pair[1]))
                .or_insert((0, pair));
            count.0 += 1;
        }
    }
    for face in &faces {
        let top: Vec<_> = face
            .vertices
            .iter()
            .map(|id| selected_vertices[id])
            .collect();
        let source_corners = (0..face.vertices.len()).map(Some).collect::<Vec<_>>();
        insert_face_from_parent(
            mesh,
            top,
            face.material_index,
            &mut changes,
            face,
            &source_corners,
            false,
        )?;
    }
    for (count, pair) in boundary_counts.values() {
        if *count != 1 {
            continue;
        }
        let first = pair[0];
        let second = pair[1];
        let face = faces
            .iter()
            .find(|face| {
                face.vertices
                    .windows(2)
                    .any(|vertices| vertices == [first, second])
                    || face.vertices.first() == Some(&second)
                        && face.vertices.last() == Some(&first)
            })
            .ok_or_else(|| invalid("selected face boundary is inconsistent"))?;
        let first_corner = face
            .vertices
            .iter()
            .position(|vertex| *vertex == first)
            .ok_or_else(|| invalid("selected face boundary start is missing"))?;
        let second_corner = face
            .vertices
            .iter()
            .position(|vertex| *vertex == second)
            .ok_or_else(|| invalid("selected face boundary end is missing"))?;
        let source_corners = [
            Some(first_corner),
            Some(second_corner),
            Some(second_corner),
            Some(first_corner),
        ];
        insert_face_from_parent(
            mesh,
            vec![
                first,
                second,
                selected_vertices[&second],
                selected_vertices[&first],
            ],
            face.material_index,
            &mut changes,
            face,
            &source_corners,
            false,
        )?;
    }
    Ok(changes)
}

fn apply_inset(mesh: &mut Mesh, selected: &Selection, amount: f64) -> Result<IdChanges> {
    if selected.domain != Domain::Face {
        return Err(invalid("inset requires face elements"));
    }
    if amount <= 0.0 {
        return Err(invalid("inset amount must be positive"));
    }
    let faces: Vec<_> = mesh
        .faces
        .iter()
        .filter(|face| selected.ids.contains(&face.id))
        .cloned()
        .collect();
    let mut changes = IdChanges::default();
    for face in faces {
        let coordinates: Vec<_> = face
            .vertices
            .iter()
            .map(|id| {
                mesh.vertex(*id)
                    .map(|vertex| vertex.co)
                    .ok_or_else(|| invalid("face references a missing vertex"))
            })
            .collect::<Result<_>>()?;
        let coordinate_count = u32::try_from(coordinates.len())
            .map_err(|_| invalid("selected face has too many vertices"))?;
        let center = coordinates
            .iter()
            .copied()
            .fold(DVec3::ZERO, |sum, p| sum + p)
            / f64::from(coordinate_count);
        let mut inner = Vec::with_capacity(coordinates.len());
        for position in &coordinates {
            let toward_center = center - *position;
            let distance = toward_center.length();
            if distance <= amount {
                return Err(invalid("inset amount collapses the selected face"));
            }
            inner.push(insert_vertex(
                mesh,
                *position + toward_center * (amount / distance),
                &mut changes,
            )?);
        }
        remove_face(mesh, face.id, &mut changes);
        for index in 0..face.vertices.len() {
            let next = (index + 1) % face.vertices.len();
            let source_corners = [Some(index), Some(next), Some(next), Some(index)];
            // Inset copies loop blocks but has no MDisps or grid-paint-mask interpolation callback.
            insert_face_from_parent(
                mesh,
                vec![
                    face.vertices[index],
                    face.vertices[next],
                    inner[next],
                    inner[index],
                ],
                face.material_index,
                &mut changes,
                &face,
                &source_corners,
                false,
            )?;
        }
        let source_corners = (0..face.vertices.len()).map(Some).collect::<Vec<_>>();
        insert_face_from_parent(
            mesh,
            inner,
            face.material_index,
            &mut changes,
            &face,
            &source_corners,
            false,
        )?;
    }
    Ok(changes)
}

fn split_edges(
    mesh: &mut Mesh,
    edge_ids: &HashSet<u32>,
    changes: &mut IdChanges,
    lineage: &IdChanges,
    radial_lineage: Option<&HashMap<u32, Vec<u32>>>,
    multires_data: Option<&mut MultiresGridData>,
) -> Result<HashMap<(u32, u32), u32>> {
    let selected: Vec<_> = mesh
        .edges
        .iter()
        .filter(|edge| edge_ids.contains(&edge.id))
        .cloned()
        .collect();
    let replacement_count = u32::try_from(selected.len())
        .map_err(|_| invalid("too many edges to subdivide"))?
        .checked_mul(2)
        .ok_or_else(|| mesh_error(&MeshError::IdExhausted))?;
    if mesh.next_id.edge.checked_add(replacement_count).is_none() {
        return Err(mesh_error(&MeshError::IdExhausted));
    }
    let mut midpoints = HashMap::with_capacity(selected.len());
    for edge in &selected {
        let a = mesh
            .vertex(edge.vertices[0])
            .ok_or_else(|| invalid("edge references a missing vertex"))?
            .co;
        let b = mesh
            .vertex(edge.vertices[1])
            .ok_or_else(|| invalid("edge references a missing vertex"))?
            .co;
        let id = insert_vertex(mesh, a.midpoint(b), changes)?;
        midpoints.insert(edge_key(edge.vertices[0], edge.vertices[1]), id);
    }
    if let Some(data) = multires_data {
        let mut root_lineage = IdChanges {
            mdisp_sources: lineage.mdisp_sources.clone(),
            ..IdChanges::default()
        };
        let pass_lineage = IdChanges::default();
        for edge in &selected {
            let source_before = mesh.clone();
            let key = edge_key(edge.vertices[0], edge.vertices[1]);
            let callback_parents = radial_lineage
                .and_then(|radial_faces| radial_faces.get(&edge.id).cloned())
                .unwrap_or_else(|| bmesh_radial_face_order(&source_before, key));
            let root_callback_faces = callback_parents
                .iter()
                .flat_map(|face_id| mdisp_callback_faces(&root_lineage, *face_id))
                .collect::<Vec<_>>();
            changes.mdisp_pass_sources.clear();
            mesh.edges.retain(|candidate| candidate.id != edge.id);
            changes.delete(Domain::Edge, edge.id);
            let middle = midpoints[&key];
            let first = mesh
                .insert_edge([edge.vertices[0], middle])
                .map_err(|error| mesh_error(&error))?;
            let second = mesh
                .insert_edge([middle, edge.vertices[1]])
                .map_err(|error| mesh_error(&error))?;
            changes.record(Domain::Edge, first);
            changes.record(Domain::Edge, second);
            if radial_lineage.is_some() {
                let radial_faces = bmesh_radial_face_order_after_split(&callback_parents);
                changes
                    .mdisp_radial_faces
                    .insert(first, radial_faces.clone());
                changes.mdisp_radial_faces.insert(second, radial_faces);
            }
            for (face_index, source_face) in source_before.faces.iter().enumerate() {
                let Some(edge_index) = face_edge_index(source_face, key) else {
                    continue;
                };
                let mut vertices = Vec::with_capacity(source_face.vertices.len() + 1);
                let mut seeds = Vec::with_capacity(source_face.vertices.len() + 1);
                for corner in 0..source_face.vertices.len() {
                    vertices.push(source_face.vertices[corner]);
                    seeds.push(Some(MdispSourceLoop {
                        face_id: source_face.id,
                        corner,
                    }));
                    if corner == edge_index {
                        vertices.push(middle);
                        seeds.push(Some(MdispSourceLoop {
                            face_id: source_face.id,
                            corner: edge_index,
                        }));
                    }
                }
                let sources =
                    mdisp_face_sources_from_callbacks(&root_lineage, &seeds, &root_callback_faces);
                let pass_sources =
                    mdisp_face_sources_from_callbacks(&pass_lineage, &seeds, &callback_parents);
                root_lineage
                    .mdisp_sources
                    .insert(source_face.id, sources.clone());
                changes.mdisp_sources.insert(source_face.id, sources);
                changes
                    .mdisp_pass_sources
                    .insert(source_face.id, pass_sources);
                mesh.faces[face_index].vertices = vertices;
            }
            rebase_subdivide_stage(&source_before, mesh, data, &changes.mdisp_pass_sources)?;
        }
        changes.mdisp_pass_sources.clear();
        return Ok(midpoints);
    }
    let split_keys: HashSet<_> = midpoints.keys().copied().collect();
    for edge in &selected {
        mesh.edges.retain(|candidate| candidate.id != edge.id);
        changes.delete(Domain::Edge, edge.id);
        let middle = midpoints[&edge_key(edge.vertices[0], edge.vertices[1])];
        let first = mesh
            .insert_edge([edge.vertices[0], middle])
            .map_err(|error| mesh_error(&error))?;
        let second = mesh
            .insert_edge([middle, edge.vertices[1]])
            .map_err(|error| mesh_error(&error))?;
        changes.record(Domain::Edge, first);
        changes.record(Domain::Edge, second);
    }
    let pass_lineage = IdChanges::default();
    for face in &mut mesh.faces {
        let mut split_vertices = Vec::with_capacity(face.vertices.len() * 2);
        let mut source_corners = Vec::with_capacity(face.vertices.len() * 2);
        for index in 0..face.vertices.len() {
            let current = face.vertices[index];
            let next = face.vertices[(index + 1) % face.vertices.len()];
            split_vertices.push(current);
            source_corners.push(Some(index));
            if split_keys.contains(&edge_key(current, next)) {
                split_vertices.push(midpoints[&edge_key(current, next)]);
                source_corners.push(Some(index));
            }
        }
        if split_vertices != face.vertices {
            let seeds = source_corners
                .iter()
                .map(|corner| {
                    corner.map(|corner| MdispSourceLoop {
                        face_id: face.id,
                        corner,
                    })
                })
                .collect::<Vec<_>>();
            let callback_parent = [face.id];
            let sources = mdisp_face_sources(lineage, &seeds, &callback_parent);
            let pass_sources = mdisp_face_sources(&pass_lineage, &seeds, &callback_parent);
            changes.mdisp_sources.insert(face.id, sources);
            changes.mdisp_pass_sources.insert(face.id, pass_sources);
            face.vertices = split_vertices;
        }
    }
    Ok(midpoints)
}

// BM_face_split interpolates from its current source face, so rebase before the next split.
fn stage_subdivide_face_split(
    stage: &mut Mesh,
    data: &mut MultiresGridData,
    root_lineage: &mut IdChanges,
    source_face_id: u32,
    first_vertices: Vec<u32>,
    second_vertices: Vec<u32>,
) -> Result<(u32, u32)> {
    let source = stage.clone();
    let source_index = source
        .faces
        .iter()
        .position(|face| face.id == source_face_id)
        .ok_or_else(|| invalid("subdivide stage source face is missing"))?;
    let source_face = source.faces[source_index].clone();
    let seeds_for = |vertices: &[u32]| -> Result<Vec<Option<MdispSourceLoop>>> {
        vertices
            .iter()
            .map(|vertex_id| {
                let corner = source_face
                    .vertices
                    .iter()
                    .position(|source_vertex| source_vertex == vertex_id)
                    .ok_or_else(|| invalid("subdivide stage corner has no source loop"))?;
                Ok(Some(MdispSourceLoop {
                    face_id: source_face.id,
                    corner,
                }))
            })
            .collect()
    };
    let first_seeds = seeds_for(&first_vertices)?;
    let second_seeds = seeds_for(&second_vertices)?;
    let callback_parent = [source_face.id];
    let first_root_sources = mdisp_face_sources(root_lineage, &first_seeds, &callback_parent);
    let second_root_sources = mdisp_face_sources(root_lineage, &second_seeds, &callback_parent);
    let pass_lineage = IdChanges::default();
    let first_pass_sources = mdisp_face_sources(&pass_lineage, &first_seeds, &callback_parent);
    let second_pass_sources = mdisp_face_sources(&pass_lineage, &second_seeds, &callback_parent);

    stage.faces[source_index].vertices = first_vertices;
    let mut stage_changes = IdChanges::default();
    let second_face_id = insert_face(
        stage,
        second_vertices,
        source_face.material_index,
        &mut stage_changes,
    )?;
    root_lineage
        .mdisp_sources
        .insert(source_face.id, first_root_sources);
    root_lineage
        .mdisp_sources
        .insert(second_face_id, second_root_sources);
    let mut pass_sources = HashMap::with_capacity(2);
    pass_sources.insert(source_face.id, first_pass_sources);
    pass_sources.insert(second_face_id, second_pass_sources);
    rebase_subdivide_stage(&source, stage, data, &pass_sources)?;
    Ok((source_face.id, second_face_id))
}

fn stage_subdivide_edge_split(
    stage: &mut Mesh,
    data: &mut MultiresGridData,
    root_lineage: &mut IdChanges,
    edge_vertices: [u32; 2],
    split_vertex: u32,
    radial_faces: &[u32],
) -> Result<Vec<u32>> {
    let key = edge_key(edge_vertices[0], edge_vertices[1]);
    let source = stage.clone();
    let source_edge = source
        .edges
        .iter()
        .find(|edge| edge_key(edge.vertices[0], edge.vertices[1]) == key)
        .cloned()
        .ok_or_else(|| invalid("subdivide stage cross-edge is missing"))?;
    stage.edges.retain(|edge| edge.id != source_edge.id);
    stage
        .insert_edge([source_edge.vertices[0], split_vertex])
        .map_err(|error| mesh_error(&error))?;
    stage
        .insert_edge([split_vertex, source_edge.vertices[1]])
        .map_err(|error| mesh_error(&error))?;

    let callback_parents = radial_faces.to_vec();
    let root_callback_faces = callback_parents
        .iter()
        .flat_map(|face_id| mdisp_callback_faces(root_lineage, *face_id))
        .collect::<Vec<_>>();
    let pass_lineage = IdChanges::default();
    let mut pass_sources = HashMap::new();
    for (face_index, source_face) in source.faces.iter().enumerate() {
        let Some(edge_index) = face_edge_index(source_face, key) else {
            continue;
        };
        let mut vertices = Vec::with_capacity(source_face.vertices.len() + 1);
        let mut seeds = Vec::with_capacity(source_face.vertices.len() + 1);
        for corner in 0..source_face.vertices.len() {
            vertices.push(source_face.vertices[corner]);
            seeds.push(Some(MdispSourceLoop {
                face_id: source_face.id,
                corner,
            }));
            if corner == edge_index {
                vertices.push(split_vertex);
                seeds.push(Some(MdispSourceLoop {
                    face_id: source_face.id,
                    corner: edge_index,
                }));
            }
        }
        let root_sources =
            mdisp_face_sources_from_callbacks(root_lineage, &seeds, &root_callback_faces);
        let local_sources =
            mdisp_face_sources_from_callbacks(&pass_lineage, &seeds, &callback_parents);
        root_lineage
            .mdisp_sources
            .insert(source_face.id, root_sources);
        pass_sources.insert(source_face.id, local_sources);
        stage.faces[face_index].vertices = vertices;
    }
    rebase_subdivide_stage(&source, stage, data, &pass_sources)?;
    Ok(bmesh_radial_face_order_after_split(radial_faces))
}

fn stage_subdivide_face_batch(
    stage: &mut Mesh,
    data: &mut MultiresGridData,
    root_lineage: &mut IdChanges,
    source_face_id: u32,
    children: Vec<(Vec<u32>, Vec<Option<usize>>)>,
) -> Result<HashMap<Vec<u32>, u32>> {
    let source = stage.clone();
    let source_face = source
        .faces
        .iter()
        .find(|face| face.id == source_face_id)
        .cloned()
        .ok_or_else(|| invalid("subdivide stage source face is missing"))?;
    let mut stage_changes = IdChanges::default();
    remove_face(stage, source_face_id, &mut stage_changes);
    let pass_lineage = IdChanges::default();
    let callback_parent = [source_face.id];
    let mut pass_sources = HashMap::with_capacity(children.len());
    let mut stage_faces = HashMap::with_capacity(children.len());
    for (vertices, source_corners) in children {
        let seeds = source_corners
            .iter()
            .map(|corner| {
                corner.map(|corner| MdispSourceLoop {
                    face_id: source_face.id,
                    corner,
                })
            })
            .collect::<Vec<_>>();
        let root_sources = mdisp_face_sources(root_lineage, &seeds, &callback_parent);
        let local_sources = mdisp_face_sources(&pass_lineage, &seeds, &callback_parent);
        let stage_face_id = insert_face_with_mdisp_sources(
            stage,
            vertices.clone(),
            source_face.material_index,
            &mut stage_changes,
            root_sources.clone(),
        )?;
        root_lineage
            .mdisp_sources
            .insert(stage_face_id, root_sources);
        pass_sources.insert(stage_face_id, local_sources);
        stage_faces.insert(vertices, stage_face_id);
    }
    rebase_subdivide_stage(&source, stage, data, &pass_sources)?;
    Ok(stage_faces)
}

// Blender 5.2.2 quad_4edge_subdivide splits m0-m2, bisects it, then connects center to m3 and m1.
fn stage_quad_subdivide(
    stage: &mut Mesh,
    data: &mut MultiresGridData,
    root_lineage: &mut IdChanges,
    source_face: &Face,
    midpoints: &HashMap<(u32, u32), u32>,
    center: u32,
) -> Result<HashMap<Vec<u32>, u32>> {
    let vertices = &source_face.vertices;
    let m0 = midpoints[&edge_key(vertices[0], vertices[1])];
    let m1 = midpoints[&edge_key(vertices[1], vertices[2])];
    let m2 = midpoints[&edge_key(vertices[2], vertices[3])];
    let m3 = midpoints[&edge_key(vertices[3], vertices[0])];
    let children = [
        vec![vertices[0], m0, center, m3],
        vec![vertices[1], m1, center, m0],
        vec![vertices[2], m2, center, m1],
        vec![vertices[3], m3, center, m2],
    ];
    let left = vec![vertices[0], m0, m2, vertices[3], m3];
    let right = vec![m0, vertices[1], m1, vertices[2], m2];
    let (right_id, left_id) =
        stage_subdivide_face_split(stage, data, root_lineage, source_face.id, right, left)?;
    let radial_faces = [left_id, right_id];
    stage_subdivide_edge_split(stage, data, root_lineage, [m0, m2], center, &radial_faces)?;
    let (first_id, fourth_id) = stage_subdivide_face_split(
        stage,
        data,
        root_lineage,
        left_id,
        children[0].clone(),
        children[3].clone(),
    )?;
    let (second_id, third_id) = stage_subdivide_face_split(
        stage,
        data,
        root_lineage,
        right_id,
        children[1].clone(),
        children[2].clone(),
    )?;
    Ok(HashMap::from([
        (children[0].clone(), first_id),
        (children[1].clone(), second_id),
        (children[2].clone(), third_id),
        (children[3].clone(), fourth_id),
    ]))
}

fn reorder_multires_values<T: Clone>(
    values: Option<&[T]>,
    source_indices: &[usize],
    source_loop_count: usize,
    context: &str,
) -> Result<Option<Vec<T>>> {
    let Some(values) = values else {
        return Ok(None);
    };
    if values.len() != source_loop_count {
        return Err(invalid(format!(
            "{context} loop count does not match the staged mesh"
        )));
    }
    source_indices
        .iter()
        .map(|index| {
            values
                .get(*index)
                .cloned()
                .ok_or_else(|| invalid(format!("{context} loop mapping is invalid")))
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn reorder_multires_face_data(
    stage: &Mesh,
    target: &Mesh,
    data: &mut MultiresGridData,
    target_to_stage: &HashMap<u32, u32>,
) -> Result<()> {
    let mut stage_faces = HashMap::with_capacity(stage.faces.len());
    let mut stage_loop_starts = HashMap::with_capacity(stage.faces.len());
    let mut stage_loop_count = 0_usize;
    for face in &stage.faces {
        stage_faces.insert(face.id, face);
        stage_loop_starts.insert(face.id, stage_loop_count);
        stage_loop_count = stage_loop_count
            .checked_add(face.vertices.len())
            .ok_or_else(|| invalid("subdivide stage loop count overflows"))?;
    }
    let target_loop_count = target.faces.iter().try_fold(0_usize, |count, face| {
        count.checked_add(face.vertices.len())
    });
    let target_loop_count =
        target_loop_count.ok_or_else(|| invalid("subdivide target loop count overflows"))?;
    let mut source_indices = Vec::with_capacity(target_loop_count);
    for target_face in &target.faces {
        let stage_face_id = target_to_stage
            .get(&target_face.id)
            .copied()
            .unwrap_or(target_face.id);
        let stage_face = stage_faces
            .get(&stage_face_id)
            .ok_or_else(|| invalid("subdivide stage face mapping is missing"))?;
        if stage_face.vertices != target_face.vertices {
            return Err(invalid(
                "subdivide stage face corners do not match the target mesh",
            ));
        }
        let start = *stage_loop_starts
            .get(&stage_face_id)
            .ok_or_else(|| invalid("subdivide stage loop mapping is missing"))?;
        for corner in 0..target_face.vertices.len() {
            source_indices.push(
                start
                    .checked_add(corner)
                    .ok_or_else(|| invalid("subdivide stage loop index overflows"))?,
            );
        }
    }
    let grid_levels = reorder_multires_values(
        data.grid_levels.as_deref(),
        &source_indices,
        stage_loop_count,
        "multires grid levels",
    )?;
    let corner_grids = reorder_multires_values(
        data.corner_grids.as_deref(),
        &source_indices,
        stage_loop_count,
        "multires corner grids",
    )?;
    let paint_mask_levels = reorder_multires_values(
        data.paint_mask_levels.as_deref(),
        &source_indices,
        stage_loop_count,
        "multires paint-mask levels",
    )?;
    let paint_mask_grids = reorder_multires_values(
        data.paint_mask_grids.as_deref(),
        &source_indices,
        stage_loop_count,
        "multires paint-mask grids",
    )?;
    let hidden_grids = reorder_multires_values(
        data.hidden_grids.as_deref(),
        &source_indices,
        stage_loop_count,
        "multires hidden grids",
    )?;
    data.grid_levels = grid_levels;
    data.corner_grids = corner_grids;
    data.paint_mask_levels = paint_mask_levels;
    data.paint_mask_grids = paint_mask_grids;
    data.hidden_grids = hidden_grids;
    Ok(())
}

fn apply_subdivide_once(
    mesh: &mut Mesh,
    selected: &Selection,
    lineage: &IdChanges,
    mut multires_data: Option<&mut MultiresGridData>,
) -> Result<IdChanges> {
    let mut changes = IdChanges::default();
    match selected.domain {
        Domain::Edge => {
            let ids: HashSet<_> = selected.ids.iter().copied().collect();
            split_edges(
                mesh,
                &ids,
                &mut changes,
                lineage,
                Some(&lineage.mdisp_radial_faces),
                multires_data.as_deref_mut(),
            )?;
        }
        Domain::Face => {
            let faces: Vec<_> = mesh
                .faces
                .iter()
                .filter(|face| selected.ids.contains(&face.id))
                .cloned()
                .collect();
            let selected_edges: HashSet<_> = mesh
                .edges
                .iter()
                .filter(|edge| {
                    faces.iter().any(|face| {
                        face.vertices.contains(&edge.vertices[0])
                            && face.vertices.contains(&edge.vertices[1])
                            && face.vertices.iter().enumerate().any(|(index, id)| {
                                *id == edge.vertices[0]
                                    && face.vertices[(index + 1) % face.vertices.len()]
                                        == edge.vertices[1]
                                    || *id == edge.vertices[1]
                                        && face.vertices[(index + 1) % face.vertices.len()]
                                            == edge.vertices[0]
                            })
                    })
                })
                .map(|edge| edge.id)
                .collect();
            let midpoints = split_edges(
                mesh,
                &selected_edges,
                &mut changes,
                lineage,
                None,
                multires_data.as_deref_mut(),
            )?;
            let mut staged_mesh = multires_data.as_ref().map(|_| mesh.clone());
            let mut staged_root_lineage = IdChanges {
                mdisp_sources: changes.mdisp_sources.clone(),
                ..IdChanges::default()
            };
            let mut actual_to_stage_faces = HashMap::new();
            changes.mdisp_pass_sources.clear();
            for face in faces {
                let vertex_count = u32::try_from(face.vertices.len())
                    .map_err(|_| invalid("selected face has too many vertices"))?;
                let mut center = DVec3::ZERO;
                for id in &face.vertices {
                    center += mesh
                        .vertex(*id)
                        .ok_or_else(|| invalid("face references a missing vertex"))?
                        .co;
                }
                center /= f64::from(vertex_count);
                let center_id = insert_vertex(mesh, center, &mut changes)?;
                let source_face = mesh
                    .faces
                    .iter()
                    .find(|source| source.id == face.id)
                    .cloned()
                    .ok_or_else(|| invalid("subdivide source face is missing"))?;
                let mut children = Vec::with_capacity(face.vertices.len());
                for index in 0..face.vertices.len() {
                    let current = face.vertices[index];
                    let next = face.vertices[(index + 1) % face.vertices.len()];
                    let previous =
                        face.vertices[(index + face.vertices.len() - 1) % face.vertices.len()];
                    let next_mid = midpoints[&edge_key(current, next)];
                    let previous_mid = midpoints[&edge_key(previous, current)];
                    let current_corner = source_face
                        .vertices
                        .iter()
                        .position(|id| *id == current)
                        .ok_or_else(|| invalid("subdivide source corner is missing"))?;
                    let next_mid_corner = source_face
                        .vertices
                        .iter()
                        .position(|id| *id == next_mid)
                        .ok_or_else(|| invalid("subdivide midpoint source loop is missing"))?;
                    let previous_mid_corner = source_face
                        .vertices
                        .iter()
                        .position(|id| *id == previous_mid)
                        .ok_or_else(|| invalid("subdivide midpoint source loop is missing"))?;
                    // quad_4edge seeds center loops from the outgoing cross-edge endpoint loop.
                    let center_corner = if face.vertices.len() == 4 {
                        let edge_index = if index == 0 || index == 3 { 0 } else { 2 };
                        let center_mid = midpoints[&edge_key(
                            face.vertices[edge_index],
                            face.vertices[(edge_index + 1) % 4],
                        )];
                        Some(
                            source_face
                                .vertices
                                .iter()
                                .position(|id| *id == center_mid)
                                .ok_or_else(|| {
                                    invalid("subdivide center example loop is missing")
                                })?,
                        )
                    } else {
                        None
                    };
                    children.push((
                        vec![current, next_mid, center_id, previous_mid],
                        [
                            Some(current_corner),
                            Some(next_mid_corner),
                            center_corner,
                            Some(previous_mid_corner),
                        ]
                        .to_vec(),
                    ));
                }

                let stage_children = if let Some(stage) = staged_mesh.as_mut() {
                    if stage.vertex(center_id).is_none() {
                        let center_vertex = mesh
                            .vertex(center_id)
                            .cloned()
                            .ok_or_else(|| invalid("subdivide center vertex is missing"))?;
                        stage.vertices.push(center_vertex);
                    }
                    let data = multires_data
                        .as_deref_mut()
                        .ok_or_else(|| invalid("subdivide stage has no multires data"))?;
                    Some(if face.vertices.len() == 4 {
                        stage_quad_subdivide(
                            stage,
                            data,
                            &mut staged_root_lineage,
                            &face,
                            &midpoints,
                            center_id,
                        )?
                    } else {
                        stage_subdivide_face_batch(
                            stage,
                            data,
                            &mut staged_root_lineage,
                            source_face.id,
                            children.clone(),
                        )?
                    })
                } else {
                    None
                };

                remove_face(mesh, face.id, &mut changes);
                for (vertices, source_corners) in children {
                    let (stage_face_id, sources) = if let Some(stage_children) = &stage_children {
                        let stage_face_id = stage_children
                            .get(&vertices)
                            .copied()
                            .ok_or_else(|| invalid("subdivide stage child face is missing"))?;
                        let sources = staged_root_lineage
                            .mdisp_sources
                            .get(&stage_face_id)
                            .cloned()
                            .ok_or_else(|| {
                                invalid("subdivide stage child provenance is missing")
                            })?;
                        (Some(stage_face_id), sources)
                    } else {
                        (
                            None,
                            mdisp_sources_from_parent(
                                &changes,
                                &source_face,
                                &source_corners,
                                true,
                            ),
                        )
                    };
                    let child_id = insert_face_with_mdisp_sources(
                        mesh,
                        vertices,
                        face.material_index,
                        &mut changes,
                        sources,
                    )?;
                    if let Some(stage_face_id) = stage_face_id {
                        actual_to_stage_faces.insert(child_id, stage_face_id);
                    }
                }
            }
            if let (Some(stage), Some(data)) = (staged_mesh.as_ref(), multires_data) {
                reorder_multires_face_data(stage, mesh, data, &actual_to_stage_faces)?;
            }
        }
        Domain::Vertex => return Err(invalid("subdivide requires edge or face elements")),
    }
    Ok(changes)
}
fn merge_changes(target: &mut IdChanges, mut source: IdChanges) {
    target.vertices.append(&mut source.vertices);
    target.edges.append(&mut source.edges);
    target.faces.append(&mut source.faces);
    target.deleted_vertices.append(&mut source.deleted_vertices);
    target.deleted_edges.append(&mut source.deleted_edges);
    target.deleted_faces.append(&mut source.deleted_faces);
    target.mdisp_sources.extend(source.mdisp_sources);
    target.mdisp_radial_faces.extend(source.mdisp_radial_faces);
}

fn discard_transient_ids(created: &mut Vec<u32>, deleted: &mut Vec<u32>) {
    let created_ids: HashSet<_> = created.iter().copied().collect();
    let deleted_ids: HashSet<_> = deleted.iter().copied().collect();
    let transient: HashSet<_> = created_ids.intersection(&deleted_ids).copied().collect();
    created.retain(|id| !transient.contains(id));
    deleted.retain(|id| !transient.contains(id));
}

fn apply_subdivide(
    mesh: &mut Mesh,
    selected: &Selection,
    cuts: u32,
    mut multires_data: Option<&mut MultiresGridData>,
) -> Result<IdChanges> {
    if selected.domain == Domain::Vertex {
        return Err(invalid("subdivide requires edge or face elements"));
    }
    if !(1..=8).contains(&cuts) {
        return Err(invalid("subdivide cuts must be between one and eight"));
    }
    let mut current_ids = selected.ids.clone();
    let mut total = IdChanges::default();
    for _ in 0..cuts {
        let pass = apply_subdivide_once(
            mesh,
            &Selection {
                domain: selected.domain,
                ids: current_ids,
            },
            &total,
            multires_data.as_deref_mut(),
        )?;
        current_ids = match selected.domain {
            Domain::Edge => pass.edges.clone(),
            Domain::Face => pass.faces.clone(),
            Domain::Vertex => return Err(invalid("subdivide requires edge or face elements")),
        };
        merge_changes(&mut total, pass);
    }
    discard_transient_ids(&mut total.vertices, &mut total.deleted_vertices);
    discard_transient_ids(&mut total.edges, &mut total.deleted_edges);
    discard_transient_ids(&mut total.faces, &mut total.deleted_faces);
    Ok(total)
}

fn apply_poke(mesh: &mut Mesh, selected: &Selection) -> Result<IdChanges> {
    if selected.domain != Domain::Face {
        return Err(invalid("poke requires face elements"));
    }
    let faces: Vec<_> = mesh
        .faces
        .iter()
        .filter(|face| selected.ids.contains(&face.id))
        .cloned()
        .collect();
    let mut changes = IdChanges::default();
    for face in faces {
        let count = u32::try_from(face.vertices.len())
            .map_err(|_| invalid("selected face has too many vertices"))?;
        let mut center = DVec3::ZERO;
        for vertex_id in &face.vertices {
            center += mesh
                .vertex(*vertex_id)
                .ok_or_else(|| invalid("face references a missing vertex"))?
                .co;
        }
        let center_id = insert_vertex(mesh, center / f64::from(count), &mut changes)?;
        remove_face(mesh, face.id, &mut changes);
        for index in 0..face.vertices.len() {
            let next = (index + 1) % face.vertices.len();
            let source_corners = [Some(index), Some(next), None];
            insert_face_from_parent(
                mesh,
                vec![face.vertices[index], face.vertices[next], center_id],
                face.material_index,
                &mut changes,
                &face,
                &source_corners,
                true,
            )?;
        }
    }
    Ok(changes)
}

fn apply_triangulate(mesh: &mut Mesh, selected: &Selection) -> Result<IdChanges> {
    if selected.domain != Domain::Face {
        return Err(invalid("triangulate requires face elements"));
    }
    let faces: Vec<_> = mesh
        .faces
        .iter()
        .filter(|face| selected.ids.contains(&face.id))
        .cloned()
        .collect();
    let mut changes = IdChanges::default();
    for face in faces {
        let corners: Vec<[f32; 3]> = face
            .vertices
            .iter()
            .map(|id| {
                mesh.vertex(*id)
                    .map(|vertex| vertex.co.as_vec3().to_array())
                    .ok_or_else(|| invalid("face references a missing vertex"))
            })
            .collect::<Result<_>>()?;
        // Blender's Triangulate Faces runs `bmesh.ops.triangulate` with BEAUTY quads and n-gons.
        let polygons = triangulate_mesh(&[corners], QuadMethod::Beauty, NgonMethod::Beauty, 3);
        remove_face(mesh, face.id, &mut changes);
        for polygon in polygons {
            let mapped: Vec<u32> = polygon
                .corners
                .iter()
                .map(|&index| face.vertices[index])
                .collect();
            let source_corners = source_corners(&face, &mapped);
            insert_face_from_parent(
                mesh,
                mapped,
                face.material_index,
                &mut changes,
                &face,
                &source_corners,
                true,
            )?;
        }
    }
    Ok(changes)
}

fn apply_delete(mesh: &mut Mesh, selected: &Selection) -> IdChanges {
    let mut changes = IdChanges::default();
    match selected.domain {
        Domain::Face => remove_faces(mesh, &selected.ids, &mut changes),
        Domain::Edge => {
            let edges: Vec<_> = mesh
                .edges
                .iter()
                .filter(|edge| selected.ids.contains(&edge.id))
                .cloned()
                .collect();
            let face_ids: Vec<_> = mesh
                .faces
                .iter()
                .filter(|face| {
                    edges.iter().any(|edge| {
                        face.vertices.contains(&edge.vertices[0])
                            && face.vertices.contains(&edge.vertices[1])
                            && face.vertices.iter().enumerate().any(|(index, id)| {
                                edge_key(*id, face.vertices[(index + 1) % face.vertices.len()])
                                    == edge_key(edge.vertices[0], edge.vertices[1])
                            })
                    })
                })
                .map(|face| face.id)
                .collect();
            remove_faces(mesh, &face_ids, &mut changes);
            mesh.edges.retain(|edge| {
                if selected.ids.contains(&edge.id) {
                    changes.delete(Domain::Edge, edge.id);
                    false
                } else {
                    true
                }
            });
        }
        Domain::Vertex => {
            let selected_ids: HashSet<_> = selected.ids.iter().copied().collect();
            let face_ids: Vec<_> = mesh
                .faces
                .iter()
                .filter(|face| face.vertices.iter().any(|id| selected_ids.contains(id)))
                .map(|face| face.id)
                .collect();
            remove_faces(mesh, &face_ids, &mut changes);
            mesh.edges.retain(|edge| {
                if edge.vertices.iter().any(|id| selected_ids.contains(id)) {
                    changes.delete(Domain::Edge, edge.id);
                    false
                } else {
                    true
                }
            });
            mesh.vertices.retain(|vertex| {
                if selected_ids.contains(&vertex.id) {
                    changes.delete(Domain::Vertex, vertex.id);
                    false
                } else {
                    true
                }
            });
        }
    }
    uv_layers::prune_missing_faces(mesh);
    changes
}

fn canonical_face(vertices: &[u32]) -> Vec<u32> {
    let mut result = vertices.to_vec();
    result.sort_unstable();
    result
}

fn cleanup_after_remap(
    mesh: &mut Mesh,
    remap: &HashMap<u32, u32>,
    changes: &mut IdChanges,
) -> Result<()> {
    let source_faces = mesh.faces.clone();
    let lineage = IdChanges::default();
    for (face_index, face) in mesh.faces.iter_mut().enumerate() {
        let source_face = &source_faces[face_index];
        let mut mapped_vertices = Vec::with_capacity(source_face.vertices.len());
        for (corner, vertex_id) in source_face.vertices.iter().enumerate() {
            let mapped = remap.get(vertex_id).copied().unwrap_or(*vertex_id);
            if mapped_vertices
                .last()
                .is_none_or(|(previous, _)| *previous != mapped)
            {
                mapped_vertices.push((mapped, corner));
            }
        }
        if mapped_vertices.len() > 1
            && mapped_vertices.first().map(|(vertex, _)| vertex)
                == mapped_vertices.last().map(|(vertex, _)| vertex)
        {
            mapped_vertices.pop();
        }
        face.vertices = mapped_vertices.iter().map(|(vertex, _)| *vertex).collect();
        if source_face.vertices != face.vertices {
            let target_vertices: HashSet<_> = face.vertices.iter().copied().collect();
            let callbacks = source_faces
                .iter()
                .filter(|candidate| {
                    candidate.vertices.iter().any(|vertex_id| {
                        remap
                            .get(vertex_id)
                            .is_some_and(|mapped| target_vertices.contains(mapped))
                    })
                })
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>();
            let seeds = mapped_vertices
                .iter()
                .map(|(_, corner)| {
                    Some(MdispSourceLoop {
                        face_id: source_face.id,
                        corner: *corner,
                    })
                })
                .collect::<Vec<_>>();
            let sources = mdisp_face_sources(&lineage, &seeds, &callbacks);
            changes.mdisp_sources.insert(face.id, sources);
        }
    }
    let mut seen_faces = HashSet::new();
    mesh.faces.retain(|face| {
        if face.vertices.len() < 3
            || face.vertices.iter().copied().collect::<HashSet<_>>().len() < 3
            || !seen_faces.insert(canonical_face(&face.vertices))
        {
            changes.delete(Domain::Face, face.id);
            false
        } else {
            true
        }
    });
    let old_edges = std::mem::take(&mut mesh.edges);
    let mut retained_keys = HashSet::new();
    for mut edge in old_edges {
        for id in &mut edge.vertices {
            if let Some(replacement) = remap.get(id) {
                *id = *replacement;
            }
        }
        let key = edge_key(edge.vertices[0], edge.vertices[1]);
        if edge.vertices[0] == edge.vertices[1] || !retained_keys.insert(key) {
            changes.delete(Domain::Edge, edge.id);
        } else {
            mesh.edges.push(edge);
        }
    }
    let mut missing_edges = Vec::new();
    for face in &mesh.faces {
        for index in 0..face.vertices.len() {
            let pair = [
                face.vertices[index],
                face.vertices[(index + 1) % face.vertices.len()],
            ];
            let key = edge_key(pair[0], pair[1]);
            if !retained_keys.contains(&key) {
                retained_keys.insert(key);
                missing_edges.push(pair);
            }
        }
    }
    for pair in missing_edges {
        let id = mesh.insert_edge(pair).map_err(|error| mesh_error(&error))?;
        changes.record(Domain::Edge, id);
    }
    let removed_ids: HashSet<_> = remap.keys().copied().collect();
    mesh.vertices.retain(|vertex| {
        if removed_ids.contains(&vertex.id) {
            changes.delete(Domain::Vertex, vertex.id);
            false
        } else {
            true
        }
    });
    Ok(())
}

fn apply_weld(mesh: &mut Mesh, selected: &Selection, threshold: f64) -> Result<IdChanges> {
    if selected.domain != Domain::Vertex {
        return Err(invalid("weld requires vertex elements"));
    }
    if threshold < 0.0 {
        return Err(invalid("weld threshold must not be negative"));
    }
    let candidates: Vec<_> = mesh
        .vertices
        .iter()
        .filter(|vertex| selected.ids.contains(&vertex.id))
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut remap = HashMap::new();
    for (index, (id, position)) in candidates.iter().enumerate() {
        if remap.contains_key(id) {
            continue;
        }
        for (other_id, other_position) in candidates.iter().skip(index + 1) {
            if position.distance(*other_position) <= threshold {
                remap.insert(*other_id, *id);
            }
        }
    }
    let mut changes = IdChanges::default();
    cleanup_after_remap(mesh, &remap, &mut changes)?;
    Ok(changes)
}

fn apply_dissolve(mesh: &mut Mesh, selected: &Selection) -> Result<IdChanges> {
    if selected.domain == Domain::Face {
        return Ok(apply_delete(mesh, selected));
    }
    if selected.domain == Domain::Vertex {
        let mut remap = HashMap::new();
        for id in &selected.ids {
            let neighbor = mesh
                .edges
                .iter()
                .filter_map(|edge| {
                    if edge.vertices[0] == *id && !selected.ids.contains(&edge.vertices[1]) {
                        Some(edge.vertices[1])
                    } else if edge.vertices[1] == *id && !selected.ids.contains(&edge.vertices[0]) {
                        Some(edge.vertices[0])
                    } else {
                        None
                    }
                })
                .min()
                .ok_or_else(|| {
                    invalid("cannot dissolve a vertex without an unselected neighbor")
                })?;
            remap.insert(*id, neighbor);
        }
        let mut changes = IdChanges::default();
        cleanup_after_remap(mesh, &remap, &mut changes)?;
        return Ok(changes);
    }
    let selected_edges: Vec<_> = mesh
        .edges
        .iter()
        .filter(|edge| selected.ids.contains(&edge.id))
        .cloned()
        .collect();
    let mut changes = IdChanges::default();
    for edge in selected_edges {
        let adjacent: Vec<_> = mesh
            .faces
            .iter()
            .filter(|face| {
                face.vertices.iter().enumerate().any(|(index, id)| {
                    edge_key(*id, face.vertices[(index + 1) % face.vertices.len()])
                        == edge_key(edge.vertices[0], edge.vertices[1])
                })
            })
            .cloned()
            .collect();
        if adjacent.len() != 2 {
            return Err(invalid("dissolve edge requires exactly two adjacent faces"));
        }
        let (first, second) = (&adjacent[0], &adjacent[1]);
        let [a, b] = edge.vertices;
        let directed_edge = |face: &Face| {
            face.vertices.iter().enumerate().find_map(|(index, id)| {
                let next = face.vertices[(index + 1) % face.vertices.len()];
                (edge_key(*id, next) == edge_key(a, b)).then_some((*id, next))
            })
        };
        let boundary_path = |face: &Face, start: u32, end: u32| -> Option<Vec<u32>> {
            let start_index = face.vertices.iter().position(|id| *id == start)?;
            let mut path = Vec::with_capacity(face.vertices.len());
            for offset in 0..face.vertices.len() {
                let id = face.vertices[(start_index + offset) % face.vertices.len()];
                path.push(id);
                if id == end {
                    return Some(path);
                }
            }
            None
        };
        let first_direction = directed_edge(first)
            .ok_or_else(|| invalid("dissolve edge is not part of the first face"))?;
        let second_direction = directed_edge(second)
            .ok_or_else(|| invalid("dissolve edge is not part of the second face"))?;
        if first_direction != (second_direction.1, second_direction.0) {
            return Err(invalid("adjacent faces have incompatible edge winding"));
        }
        let mut first_path = boundary_path(first, first_direction.1, first_direction.0)
            .ok_or_else(|| invalid("could not trace the first face boundary"))?;
        let second_path = boundary_path(second, second_direction.1, second_direction.0)
            .ok_or_else(|| invalid("could not trace the second face boundary"))?;
        let mut seed_sources = first_path
            .iter()
            .map(|vertex| {
                first
                    .vertices
                    .iter()
                    .position(|id| id == vertex)
                    .map(|corner| MdispSourceLoop {
                        face_id: first.id,
                        corner,
                    })
            })
            .collect::<Vec<_>>();
        first_path.pop();
        seed_sources.pop();
        for vertex in second_path.iter().take(second_path.len().saturating_sub(1)) {
            first_path.push(*vertex);
            seed_sources.push(
                second
                    .vertices
                    .iter()
                    .position(|id| id == vertex)
                    .map(|corner| MdispSourceLoop {
                        face_id: second.id,
                        corner,
                    }),
            );
        }
        remove_faces(mesh, &[first.id, second.id], &mut changes);
        mesh.edges.retain(|item| {
            if item.id == edge.id {
                changes.delete(Domain::Edge, item.id);
                false
            } else {
                true
            }
        });
        let callbacks = [first.id, second.id];
        let sources = mdisp_face_sources(&changes, &seed_sources, &callbacks);
        insert_face_with_mdisp_sources(
            mesh,
            first_path,
            first.material_index,
            &mut changes,
            sources,
        )?;
    }
    Ok(changes)
}

fn apply_fill(mesh: &mut Mesh, selected: &Selection) -> Result<IdChanges> {
    if selected.domain != Domain::Edge {
        return Err(invalid("fill requires edge elements"));
    }
    let selected_edges: Vec<_> = mesh
        .edges
        .iter()
        .filter(|edge| selected.ids.contains(&edge.id))
        .collect();
    if selected_edges.len() < 3 {
        return Err(invalid("fill requires at least three boundary edges"));
    }
    let mut adjacency: HashMap<u32, Vec<u32>> = HashMap::new();
    for edge in selected_edges {
        adjacency
            .entry(edge.vertices[0])
            .or_default()
            .push(edge.vertices[1]);
        adjacency
            .entry(edge.vertices[1])
            .or_default()
            .push(edge.vertices[0]);
    }
    if adjacency.values().any(|neighbors| neighbors.len() != 2) {
        return Err(invalid("fill edges must form one closed boundary loop"));
    }
    let start = *adjacency
        .keys()
        .min()
        .ok_or_else(|| invalid("fill requires boundary edges"))?;
    let mut vertices = vec![start];
    let mut previous = None;
    let mut current = start;
    loop {
        let neighbors = &adjacency[&current];
        let next = if Some(neighbors[0]) == previous {
            neighbors[1]
        } else {
            neighbors[0]
        };
        if next == start {
            break;
        }
        if vertices.contains(&next) {
            return Err(invalid("fill edges contain more than one loop"));
        }
        vertices.push(next);
        previous = Some(current);
        current = next;
    }
    if vertices.len() != adjacency.len() {
        return Err(invalid("fill edges contain disconnected loops"));
    }
    let mut changes = IdChanges::default();
    // Blender's loop fill creates these loops without an example face or MDisps callback.
    let sources = vec![MdispLoopSource::default(); vertices.len()];
    insert_face_with_mdisp_sources(mesh, vertices, 0, &mut changes, sources)?;
    Ok(changes)
}

fn bisect_split_edge(
    mesh: &mut Mesh,
    key: (u32, u32),
    position: DVec3,
    changes: &mut IdChanges,
    data: &mut MultiresGridData,
) -> Result<u32> {
    let source_before = mesh.clone();
    let edge_index = mesh
        .edges
        .iter()
        .position(|edge| edge_key(edge.vertices[0], edge.vertices[1]) == key)
        .ok_or_else(|| invalid("bisect split edge is missing"))?;
    let edge = mesh.edges[edge_index].clone();
    let affected_faces = mesh
        .faces
        .iter()
        .rev()
        .filter(|face| face_edge_index(face, key).is_some())
        .cloned()
        .collect::<Vec<_>>();
    let callback_parents = bmesh_radial_face_order(&source_before, key);
    let root_callback_faces = callback_parents
        .iter()
        .flat_map(|face_id| mdisp_callback_faces(changes, *face_id))
        .collect::<Vec<_>>();
    if affected_faces.is_empty() {
        return Err(invalid("bisect split edge has no incident face"));
    }

    let intersection = insert_vertex(mesh, position, changes)?;
    mesh.edges[edge_index].vertices = [intersection, edge.vertices[1]];
    let new_edge = mesh
        .insert_edge([edge.vertices[0], intersection])
        .map_err(|error| mesh_error(&error))?;
    changes.record(Domain::Edge, new_edge);

    let pass_lineage = IdChanges::default();
    let mut pass_sources = HashMap::with_capacity(affected_faces.len());
    for source_face in affected_faces {
        let edge_corner = face_edge_index(&source_face, key)
            .ok_or_else(|| invalid("bisect incident face lost its split edge"))?;
        let mut vertices = Vec::with_capacity(source_face.vertices.len() + 1);
        for (corner, vertex_id) in source_face.vertices.iter().copied().enumerate() {
            vertices.push(vertex_id);
            if corner == edge_corner {
                vertices.push(intersection);
            }
        }
        let seed_corners = vertices
            .iter()
            .map(|vertex_id| {
                source_face
                    .vertices
                    .iter()
                    .position(|source_id| source_id == vertex_id)
                    .or_else(|| (*vertex_id == intersection).then_some(edge_corner))
            })
            .collect::<Vec<_>>();
        let seeds = seed_corners
            .iter()
            .map(|corner| {
                corner.map(|corner| MdispSourceLoop {
                    face_id: source_face.id,
                    corner,
                })
            })
            .collect::<Vec<_>>();
        let root_sources = mdisp_face_sources_from_callbacks(changes, &seeds, &root_callback_faces);
        let local_sources =
            mdisp_face_sources_from_callbacks(&pass_lineage, &seeds, &callback_parents);
        let target_face = mesh
            .faces
            .iter_mut()
            .find(|face| face.id == source_face.id)
            .ok_or_else(|| invalid("bisect incident face is missing"))?;
        target_face.vertices = vertices;
        changes.mdisp_sources.insert(source_face.id, root_sources);
        pass_sources.insert(source_face.id, local_sources);
    }
    rebase_subdivide_stage(&source_before, mesh, data, &pass_sources)?;
    Ok(intersection)
}

fn bisect_split_face(
    mesh: &mut Mesh,
    face_id: u32,
    first_vertex: u32,
    second_vertex: u32,
    changes: &mut IdChanges,
    data: &mut MultiresGridData,
) -> Result<Option<u32>> {
    let source = mesh.clone();
    let source_index = source
        .faces
        .iter()
        .position(|face| face.id == face_id)
        .ok_or_else(|| invalid("bisect source face is missing"))?;
    let source_face = source.faces[source_index].clone();
    let first_index = source_face
        .vertices
        .iter()
        .position(|vertex| *vertex == first_vertex)
        .ok_or_else(|| invalid("bisect split vertex is missing from the source face"))?;
    let second_index = source_face
        .vertices
        .iter()
        .position(|vertex| *vertex == second_vertex)
        .ok_or_else(|| invalid("bisect split vertex is missing from the source face"))?;
    if (first_index + 1) % source_face.vertices.len() == second_index
        || (second_index + 1) % source_face.vertices.len() == first_index
    {
        return Ok(None);
    }
    let face_arc = |start: usize, end: usize| {
        let mut vertices = Vec::new();
        let mut corner = start;
        loop {
            vertices.push(source_face.vertices[corner]);
            if corner == end {
                break;
            }
            corner = (corner + 1) % source_face.vertices.len();
        }
        vertices
    };
    let first_vertices = face_arc(first_index, second_index);
    let second_vertices = face_arc(second_index, first_index);
    let seeds_for = |vertices: &[u32]| -> Result<Vec<Option<MdispSourceLoop>>> {
        vertices
            .iter()
            .map(|vertex_id| {
                let corner = source_face
                    .vertices
                    .iter()
                    .position(|source_vertex| source_vertex == vertex_id)
                    .ok_or_else(|| invalid("bisect output loop has no source corner"))?;
                Ok(Some(MdispSourceLoop {
                    face_id: source_face.id,
                    corner,
                }))
            })
            .collect()
    };
    let first_seeds = seeds_for(&first_vertices)?;
    let second_seeds = seeds_for(&second_vertices)?;
    let callback_parent = [source_face.id];
    let first_root_sources = mdisp_face_sources(changes, &first_seeds, &callback_parent);
    let second_root_sources = mdisp_face_sources(changes, &second_seeds, &callback_parent);
    let pass_lineage = IdChanges::default();
    let first_pass_sources = mdisp_face_sources(&pass_lineage, &first_seeds, &callback_parent);
    let second_pass_sources = mdisp_face_sources(&pass_lineage, &second_seeds, &callback_parent);

    mesh.faces[source_index].vertices = first_vertices;
    changes
        .mdisp_sources
        .insert(source_face.id, first_root_sources);
    let second_face_id = insert_face_with_mdisp_sources(
        mesh,
        second_vertices,
        source_face.material_index,
        changes,
        second_root_sources,
    )?;
    let mut pass_sources = HashMap::with_capacity(2);
    pass_sources.insert(source_face.id, first_pass_sources);
    pass_sources.insert(second_face_id, second_pass_sources);
    rebase_subdivide_stage(&source, mesh, data, &pass_sources)?;
    Ok(Some(second_face_id))
}

// Blender 5.2.2 bisects tagged edges in mesh order, then pops each incident face LIFO.
fn apply_bisect_multires(
    mesh: &mut Mesh,
    selected_faces: &[Face],
    point: DVec3,
    plane_normal: DVec3,
    epsilon: f64,
    clear_side: Option<bool>,
    data: &mut MultiresGridData,
) -> Result<IdChanges> {
    let selected_face_ids = selected_faces
        .iter()
        .map(|face| face.id)
        .collect::<HashSet<_>>();
    let mut tagged_edge_keys = HashSet::new();
    for face in selected_faces {
        for index in 0..face.vertices.len() {
            tagged_edge_keys.insert(edge_key(
                face.vertices[index],
                face.vertices[(index + 1) % face.vertices.len()],
            ));
        }
    }

    let mut positions = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect::<HashMap<_, _>>();
    let mut distances = HashMap::with_capacity(positions.len());
    let mut directions = HashMap::with_capacity(positions.len());
    for (vertex_id, position) in &positions {
        let distance = (*position - point).dot(plane_normal);
        distances.insert(*vertex_id, distance);
        directions.insert(
            *vertex_id,
            if distance <= -epsilon {
                -1_i8
            } else {
                i8::from(distance >= epsilon)
            },
        );
    }

    let mut changes = IdChanges::default();
    let mut face_stack = Vec::new();
    let mut queued_faces = HashSet::new();
    let mut centered_vertices = HashSet::new();

    let tagged_edges = mesh.edges.clone();
    for edge in tagged_edges {
        let key = edge_key(edge.vertices[0], edge.vertices[1]);
        if !tagged_edge_keys.contains(&key) {
            continue;
        }
        let first_direction = directions[&edge.vertices[0]];
        let second_direction = directions[&edge.vertices[1]];
        if first_direction != 0 && second_direction != 0 && first_direction != second_direction {
            for face_id in bmesh_radial_face_order(mesh, key) {
                if queued_faces.insert(face_id) {
                    face_stack.push(face_id);
                }
            }
            let first_distance = distances[&edge.vertices[0]];
            let second_distance = distances[&edge.vertices[1]];
            let factor = first_distance / (first_distance - second_distance);
            let first_position = positions[&edge.vertices[0]];
            let second_position = positions[&edge.vertices[1]];
            let intersection = bisect_split_edge(
                mesh,
                key,
                first_position.lerp(second_position, factor),
                &mut changes,
                data,
            )?;
            positions.insert(intersection, first_position.lerp(second_position, factor));
            distances.insert(intersection, 0.0);
            directions.insert(intersection, 0);
        } else {
            for vertex_id in edge.vertices {
                if directions[&vertex_id] == 0 && centered_vertices.insert(vertex_id) {
                    for face in mesh.faces.iter().rev() {
                        if face.vertices.contains(&vertex_id) && queued_faces.insert(face.id) {
                            face_stack.push(face.id);
                        }
                    }
                }
            }
        }
    }

    let mut selected_descendants = selected_face_ids;
    while let Some(face_id) = face_stack.pop() {
        let Some(face) = mesh.faces.iter().find(|face| face.id == face_id).cloned() else {
            continue;
        };
        let has_positive = face.vertices.iter().any(|vertex| {
            directions
                .get(vertex)
                .is_some_and(|direction| *direction > 0)
        });
        let has_negative = face.vertices.iter().any(|vertex| {
            directions
                .get(vertex)
                .is_some_and(|direction| *direction < 0)
        });
        if !has_positive || !has_negative {
            continue;
        }

        let mut centers = Vec::new();
        let mut face_has_center_edge = false;
        for (loop_index, vertex_id) in face.vertices.iter().copied().enumerate() {
            if directions[&vertex_id] != 0 {
                continue;
            }
            let previous =
                face.vertices[(loop_index + face.vertices.len() - 1) % face.vertices.len()];
            let next = face.vertices[(loop_index + 1) % face.vertices.len()];
            let previous_direction = i32::from(directions[&previous]);
            let next_direction = i32::from(directions[&next]);
            centers.push((
                vertex_id,
                loop_index,
                previous_direction ^ next_direction == 0,
            ));
            face_has_center_edge |= directions[&previous] == 0;
        }
        if centers.len() < 2 {
            continue;
        }

        if centers.len() == 2 {
            if let Some(new_face_id) = bisect_split_face(
                mesh,
                face.id,
                centers[0].0,
                centers[1].0,
                &mut changes,
                data,
            )? && selected_descendants.contains(&face.id)
            {
                selected_descendants.insert(new_face_id);
            }
            continue;
        }

        let mut sort_direction = face_normal(mesh, &face)?
            .cross(plane_normal)
            .normalize_or_zero();
        if sort_direction.length_squared() <= f64::EPSILON {
            let first_position = positions[&centers[0].0];
            for center in centers.iter().skip(1) {
                let direction = (first_position - positions[&center.0]).normalize_or_zero();
                if direction.length_squared() > f64::EPSILON {
                    sort_direction = direction;
                }
            }
        }
        if sort_direction.length_squared() <= f64::EPSILON {
            continue;
        }
        centers.sort_by(|first, second| {
            positions[&first.0]
                .dot(sort_direction)
                .total_cmp(&positions[&second.0].dot(sort_direction))
        });

        let mut split_faces = vec![face.id];
        let mut is_inside = false;
        for pair in centers.windows(2) {
            let loop_distance = pair[0].1.abs_diff(pair[1].1);
            if face_has_center_edge
                && (loop_distance == 1 || loop_distance + 1 == face.vertices.len())
            {
                continue;
            }
            if !pair[0].2 {
                is_inside = !is_inside;
            }
            if !is_inside {
                continue;
            }
            let source_face_id = split_faces.iter().find_map(|candidate_id| {
                let candidate = mesh.faces.iter().find(|face| face.id == *candidate_id)?;
                (candidate.vertices.contains(&pair[0].0) && candidate.vertices.contains(&pair[1].0))
                    .then_some(*candidate_id)
            });
            let Some(source_face_id) = source_face_id else {
                continue;
            };
            let Some(new_face_id) = bisect_split_face(
                mesh,
                source_face_id,
                pair[0].0,
                pair[1].0,
                &mut changes,
                data,
            )?
            else {
                continue;
            };
            if selected_descendants.contains(&source_face_id) {
                selected_descendants.insert(new_face_id);
            }
            split_faces.push(new_face_id);
        }
    }

    if let Some(clear_positive) = clear_side {
        let faces_to_remove = selected_descendants
            .iter()
            .filter_map(|face_id| {
                let face = mesh.faces.iter().find(|face| face.id == *face_id)?;
                let has_positive = face
                    .vertices
                    .iter()
                    .any(|vertex_id| directions[vertex_id] > 0);
                let has_negative = face
                    .vertices
                    .iter()
                    .any(|vertex_id| directions[vertex_id] < 0);
                let wholly_cleared = if clear_positive {
                    has_positive && !has_negative
                } else {
                    has_negative && !has_positive
                };
                wholly_cleared.then_some(*face_id)
            })
            .collect::<Vec<_>>();
        if !faces_to_remove.is_empty() {
            if data.corner_grids.is_none() && data.external_filepath.is_some() {
                return Err(invalid(
                    "cannot rebase multires data while external MDisps values are unavailable",
                ));
            }
            let before_clear = mesh.clone();
            for face_id in faces_to_remove {
                remove_face(mesh, face_id, &mut changes);
            }
            reorder_multires_face_data(&before_clear, mesh, data, &HashMap::new())?;
        }
    }
    Ok(changes)
}

fn apply_bisect(
    mesh: &mut Mesh,
    args: &Map<String, Value>,
    multires_data: Option<&mut MultiresGridData>,
    snap_center: bool,
) -> Result<IdChanges> {
    let selected = selection(mesh, args, false)?;
    if selected
        .as_ref()
        .is_some_and(|selection| selection.domain != Domain::Face)
    {
        return Err(invalid("bisect elements must be faces"));
    }
    let point = optional_vector(args, "point", DVec3::ZERO)?;
    let normal = optional_vector(args, "normal", DVec3::Z)?;
    if normal.length_squared() <= f64::EPSILON {
        return Err(invalid("bisect normal must be non-zero"));
    }
    let normal = normal.normalize();
    let epsilon = number(args, "threshold", 1.0e-10)?;
    if epsilon < 0.0 {
        return Err(invalid("bisect threshold must not be negative"));
    }
    let clear_side = match args.get("clear_side") {
        None => None,
        Some(Value::String(side)) if side == "positive" => Some(true),
        Some(Value::String(side)) if side == "negative" => Some(false),
        Some(Value::String(side)) if side == "none" => None,
        Some(_) => return Err(invalid("clear_side must be positive, negative, or none")),
    };
    let selected_faces: Vec<_> = mesh
        .faces
        .iter()
        .filter(|face| {
            selected
                .as_ref()
                .is_none_or(|selection| selection.ids.contains(&face.id))
        })
        .cloned()
        .collect();
    if snap_center && multires_data.is_some() {
        let selected_vertices = selected_faces
            .iter()
            .flat_map(|face| face.vertices.iter().copied())
            .collect::<HashSet<_>>();
        for vertex in &mut mesh.vertices {
            if selected_vertices.contains(&vertex.id) {
                let distance = (vertex.co - point).dot(normal);
                if distance.abs() < epsilon {
                    vertex.co -= normal * distance;
                }
            }
        }
    }
    if let Some(data) = multires_data {
        return apply_bisect_multires(
            mesh,
            &selected_faces,
            point,
            normal,
            epsilon,
            clear_side,
            data,
        );
    }
    let mut changes = IdChanges::default();
    let mut intersections = HashMap::new();
    for face in selected_faces {
        let coords: Vec<_> = face
            .vertices
            .iter()
            .map(|id| {
                mesh.vertex(*id)
                    .map(|vertex| vertex.co)
                    .ok_or_else(|| invalid("face references a missing vertex"))
            })
            .collect::<Result<_>>()?;
        let distances: Vec<_> = coords
            .iter()
            .map(|position| (*position - point).dot(normal))
            .collect();
        let positive_side = distances.iter().any(|distance| *distance > epsilon);
        let negative_side = distances.iter().any(|distance| *distance < -epsilon);
        if clear_side == Some(true) && positive_side && !negative_side
            || clear_side == Some(false) && negative_side && !positive_side
        {
            remove_face(mesh, face.id, &mut changes);
            continue;
        }
        if !positive_side || !negative_side {
            continue;
        }
        let mut positive = Vec::new();
        let mut negative = Vec::new();
        let mut intersection_seed_corners = HashMap::new();
        for index in 0..face.vertices.len() {
            let next = (index + 1) % face.vertices.len();
            let current_distance = distances[index];
            let next_distance = distances[next];
            let current_id = face.vertices[index];
            let next_id = face.vertices[next];
            if current_distance >= -epsilon {
                positive.push(current_id);
            }
            if current_distance <= epsilon {
                negative.push(current_id);
            }
            if current_distance > epsilon && next_distance < -epsilon
                || current_distance < -epsilon && next_distance > epsilon
            {
                let key = edge_key(current_id, next_id);
                let intersection = if let Some(id) = intersections.get(&key) {
                    *id
                } else {
                    let ratio = current_distance / (current_distance - next_distance);
                    let position = coords[index].lerp(coords[next], ratio);
                    let id = insert_vertex(mesh, position, &mut changes)?;
                    intersections.insert(key, id);
                    id
                };
                intersection_seed_corners.insert(intersection, index);
                positive.push(intersection);
                negative.push(intersection);
            }
        }
        positive.dedup();
        negative.dedup();
        let seed_corners = |vertices: &[u32]| {
            vertices
                .iter()
                .map(|vertex_id| {
                    face.vertices
                        .iter()
                        .position(|source_id| source_id == vertex_id)
                        .or_else(|| intersection_seed_corners.get(vertex_id).copied())
                })
                .collect::<Vec<_>>()
        };
        remove_face(mesh, face.id, &mut changes);
        let callback_parent = [face.id];
        if clear_side != Some(true) && positive.len() >= 3 {
            let seed_corners = seed_corners(&positive);
            let seeds = seed_corners
                .iter()
                .map(|corner| {
                    corner.map(|corner| MdispSourceLoop {
                        face_id: face.id,
                        corner,
                    })
                })
                .collect::<Vec<_>>();
            let sources = mdisp_face_sources(&changes, &seeds, &callback_parent);
            insert_face_with_mdisp_sources(
                mesh,
                positive,
                face.material_index,
                &mut changes,
                sources,
            )?;
        }
        if clear_side != Some(false) && negative.len() >= 3 {
            let seed_corners = seed_corners(&negative);
            let seeds = seed_corners
                .iter()
                .map(|corner| {
                    corner.map(|corner| MdispSourceLoop {
                        face_id: face.id,
                        corner,
                    })
                })
                .collect::<Vec<_>>();
            let sources = mdisp_face_sources(&changes, &seeds, &callback_parent);
            insert_face_with_mdisp_sources(
                mesh,
                negative,
                face.material_index,
                &mut changes,
                sources,
            )?;
        }
    }
    Ok(changes)
}

fn axis_index(args: &Map<String, Value>) -> Result<usize> {
    match args.get("axis").and_then(Value::as_str).unwrap_or("x") {
        "x" | "X" => Ok(0),
        "y" | "Y" => Ok(1),
        "z" | "Z" => Ok(2),
        other => Err(invalid(format!("axis must be x, y, or z, not {other:?}"))),
    }
}

fn apply_mirror(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let axis = axis_index(args)?;
    let origin = optional_vector(args, "origin", DVec3::ZERO)?;
    let merge = boolean(args, "merge", false)?;
    let threshold = number(args, "threshold", 1.0e-6)?;
    if threshold < 0.0 {
        return Err(invalid("mirror threshold must not be negative"));
    }
    let selected = selection(mesh, args, false)?;
    let (selected_vertices, selected_edges, selected_faces) = if let Some(selection) = selected {
        let mut vertices = HashSet::new();
        let mut edges = HashSet::new();
        let mut faces = HashSet::new();
        match selection.domain {
            Domain::Vertex => vertices.extend(selection.ids),
            Domain::Edge => {
                edges.extend(&selection.ids);
                for edge in &mesh.edges {
                    if selection.ids.contains(&edge.id) {
                        vertices.extend(edge.vertices);
                    }
                }
            }
            Domain::Face => {
                faces.extend(&selection.ids);
                for face in &mesh.faces {
                    if selection.ids.contains(&face.id) {
                        vertices.extend(&face.vertices);
                    }
                }
            }
        }
        (vertices, edges, faces)
    } else {
        (
            mesh.vertices.iter().map(|vertex| vertex.id).collect(),
            mesh.edges.iter().map(|edge| edge.id).collect(),
            mesh.faces.iter().map(|face| face.id).collect(),
        )
    };
    let vertices: Vec<_> = mesh
        .vertices
        .iter()
        .filter(|vertex| selected_vertices.contains(&vertex.id))
        .cloned()
        .collect();
    let faces: Vec<_> = mesh
        .faces
        .iter()
        .filter(|face| {
            selected_faces.contains(&face.id)
                || face
                    .vertices
                    .iter()
                    .all(|id| selected_vertices.contains(id))
        })
        .cloned()
        .collect();
    let edges: Vec<_> = mesh
        .edges
        .iter()
        .filter(|edge| {
            selected_edges.contains(&edge.id)
                || selected_vertices.contains(&edge.vertices[0])
                    && selected_vertices.contains(&edge.vertices[1])
        })
        .cloned()
        .collect();
    if vertices.is_empty() {
        return Err(invalid("mirror selection contains no vertices"));
    }
    let mut changes = IdChanges::default();
    let mut reflected_ids = HashMap::with_capacity(vertices.len());
    for vertex in &vertices {
        let distance = vertex.co[axis] - origin[axis];
        let mut reflected = vertex.co;
        reflected[axis] = origin[axis] - distance;
        if merge && distance.abs() <= threshold {
            reflected_ids.insert(vertex.id, vertex.id);
        } else {
            let id = insert_vertex(mesh, reflected, &mut changes)?;
            reflected_ids.insert(vertex.id, id);
        }
    }
    for edge in edges {
        let first = reflected_ids[&edge.vertices[0]];
        let second = reflected_ids[&edge.vertices[1]];
        if first != second
            && !mesh
                .edges
                .iter()
                .any(|item| edge_key(item.vertices[0], item.vertices[1]) == edge_key(first, second))
        {
            let id = mesh
                .insert_edge([first, second])
                .map_err(|error| mesh_error(&error))?;
            changes.record(Domain::Edge, id);
        }
    }
    for face in faces {
        let mut reflected: Vec<_> = face.vertices.iter().map(|id| reflected_ids[id]).collect();
        reflected.reverse();
        if reflected.iter().copied().collect::<HashSet<_>>().len() >= 3 {
            let source_corners = (0..face.vertices.len()).rev().map(Some).collect::<Vec<_>>();
            insert_face_from_parent(
                mesh,
                reflected,
                face.material_index,
                &mut changes,
                &face,
                &source_corners,
                false,
            )?;
        }
    }
    Ok(changes)
}

fn apply_flip_normals(mesh: &mut Mesh, selected: &Selection) -> Result<IdChanges> {
    if selected.domain != Domain::Face {
        return Err(invalid("flip_normals requires face elements"));
    }
    for face in &mut mesh.faces {
        if selected.ids.contains(&face.id) {
            face.vertices.reverse();
        }
    }
    Ok(IdChanges::default())
}

fn attribute_domain(value: &Value) -> Result<(&'static str, Domain)> {
    match value.as_str() {
        Some("point" | "vertex" | "vertices") => Ok(("point", Domain::Vertex)),
        Some("edge" | "edges") => Ok(("edge", Domain::Edge)),
        Some("face" | "faces") => Ok(("face", Domain::Face)),
        Some("corner") => Ok(("corner", Domain::Face)),
        _ => Err(invalid(
            "attribute domain must be point, edge, face, or corner",
        )),
    }
}

fn attribute_type(value: &Value) -> Result<&'static str> {
    match value.as_str() {
        Some("float") => Ok("float"),
        Some("int") => Ok("int"),
        Some("float2") => Ok("float2"),
        Some("float3") => Ok("float3"),
        Some("color") => Ok("color"),
        Some("bool") => Ok("bool"),
        Some("quaternion") => Ok("quaternion"),
        _ => Err(invalid(
            "attribute type must be float, int, float2, float3, color, bool, or quaternion",
        )),
    }
}

fn default_attribute_value(attribute_type: &str) -> Value {
    match attribute_type {
        "float" => json!(0.0),
        "int" => json!(0),
        "float2" => json!([0.0, 0.0]),
        "float3" => json!([0.0, 0.0, 0.0]),
        "color" | "quaternion" => json!([0.0, 0.0, 0.0, 1.0]),
        "bool" => json!(false),
        _ => Value::Null,
    }
}

fn validate_attribute_value(attribute_type: &str, value: &Value) -> Result<Value> {
    let components = match attribute_type {
        "float" => {
            if !value.as_f64().is_some_and(f64::is_finite) {
                return Err(invalid("float attribute values must be finite numbers"));
            }
            return Ok(value.clone());
        }
        "int" => {
            let Some(number) = value
                .as_i64()
                .filter(|number| i32::try_from(*number).is_ok())
            else {
                return Err(invalid(
                    "int attribute values must fit a signed 32-bit integer",
                ));
            };
            return Ok(json!(number));
        }
        "bool" => {
            if !value.is_boolean() {
                return Err(invalid("bool attribute values must be booleans"));
            }
            return Ok(value.clone());
        }
        "float2" => 2,
        "float3" => 3,
        "color" | "quaternion" => 4,
        _ => return Err(invalid("unknown mesh attribute type")),
    };
    let values = value
        .as_array()
        .filter(|values| values.len() == components)
        .ok_or_else(|| {
            invalid(format!(
                "{attribute_type} attribute values need {components} components"
            ))
        })?;
    if attribute_type == "quaternion" {
        let mut components = [0.0; 4];
        for (index, component) in values.iter().enumerate() {
            components[index] = component
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid("quaternion components must be finite numbers"))?;
        }
        let norm = components
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm <= f64::EPSILON {
            return Err(invalid("quaternion attribute values must be non-zero"));
        }
        Ok(json!(components.map(|component| component / norm)))
    } else {
        if values
            .iter()
            .any(|component| !component.as_f64().is_some_and(f64::is_finite))
        {
            return Err(invalid(format!(
                "{attribute_type} components must be finite numbers"
            )));
        }
        Ok(value.clone())
    }
}

fn attribute_values_mut<'a>(mesh: &'a mut Mesh, name: &str) -> Result<&'a mut Map<String, Value>> {
    mesh.attributes
        .get_mut(name)
        .and_then(Value::as_object_mut)
        .and_then(|attribute| attribute.get_mut("values"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no values object")))
}

/// Reads one validated `[r, g, b, a]` value.
fn color_components(value: &Value) -> Result<[f64; 4]> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 4)
        .ok_or_else(|| invalid("color attribute values need 4 components"))?;
    let mut color = [0.0; 4];
    for (slot, component) in color.iter_mut().zip(components) {
        *slot = component
            .as_f64()
            .ok_or_else(|| invalid("color components must be finite numbers"))?;
    }
    Ok(color)
}

/// Creates a native color attribute and registers it in the color layer store, which render and
/// glTF read. The new layer goes last. It becomes the active or render layer only when the store
/// has no layer for that role.
fn create_color_layer(
    mesh: &mut Mesh,
    name: &str,
    domain: ColorDomain,
    default: &Value,
) -> Result<()> {
    let color = color_components(default)?;
    let count: usize = match domain {
        ColorDomain::Point => mesh.vertices.len(),
        ColorDomain::Corner => mesh.faces.iter().map(|face| face.vertices.len()).sum(),
    };
    color_layers::validate(mesh)?;
    let mut store = color_layers::read(mesh);
    let active_unset = store
        .active
        .as_deref()
        .and_then(|active| store.index_of(active))
        .is_none();
    let render_unset = store.render_index().is_none();
    store.layers.push(ColorLayer {
        name: name.to_owned(),
        domain,
        values: vec![color; count],
    });
    if active_unset {
        store.active = Some(name.to_owned());
    }
    if render_unset {
        store.render = Some(name.to_owned());
    }
    color_layers::write(mesh, &store);
    Ok(())
}

/// `uv_layers` is the UV layer store, not a user attribute: create, update, and delete reject it.
fn reject_reserved_attribute(name: &str) -> Result<()> {
    if name == uv_layers::ATTRIBUTE {
        return Err(invalid(format!(
            "mesh attribute `{name}` is reserved for UV layers"
        )));
    }
    Ok(())
}
fn apply_attribute_create(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid("attribute_create requires a non-empty name"))?;
    reject_reserved_attribute(name)?;
    if mesh.attributes.contains_key(&*storage_key(name)) {
        return Err(invalid(format!("mesh attribute `{name}` already exists")));
    }
    let (domain_name, _) = attribute_domain(
        args.get("domain")
            .ok_or_else(|| invalid("attribute_create requires domain"))?,
    )?;
    let attribute_type = attribute_type(
        args.get("type")
            .ok_or_else(|| invalid("attribute_create requires type"))?,
    )?;
    let default = args.get("default").map_or_else(
        || Ok(default_attribute_value(attribute_type)),
        |value| validate_attribute_value(attribute_type, value),
    )?;
    if attribute_type == "color" {
        let domain = match domain_name {
            "point" => ColorDomain::Point,
            "corner" => ColorDomain::Corner,
            _ => return Err(invalid("color attributes use the point or corner domain")),
        };
        create_color_layer(mesh, name, domain, &default)?;
        return Ok(IdChanges::default());
    }
    let mut values = Map::new();
    match domain_name {
        "point" => {
            for vertex in &mesh.vertices {
                values.insert(format!("v{}", vertex.id), default.clone());
            }
        }
        "edge" => {
            for edge in &mesh.edges {
                values.insert(format!("e{}", edge.id), default.clone());
            }
        }
        "face" => {
            for face in &mesh.faces {
                values.insert(format!("f{}", face.id), default.clone());
            }
        }
        "corner" => {
            for face in &mesh.faces {
                values.insert(
                    format!("f{}", face.id),
                    Value::Array(vec![default.clone(); face.vertices.len()]),
                );
            }
        }
        _ => return Err(invalid("unsupported attribute domain")),
    }
    mesh.attributes.insert(
        storage_key(name).into_owned(),
        json!({"domain":domain_name, "type":attribute_type, "values":values}),
    );
    Ok(IdChanges::default())
}

fn apply_attribute_update(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid("attribute_update requires a non-empty name"))?;
    reject_reserved_attribute(name)?;
    let storage = storage_key(name);
    let key: &str = &storage;
    let attribute = mesh
        .attributes
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(format!("mesh attribute `{name}` does not exist")))?;
    let stored_domain = attribute
        .get("domain")
        .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no domain")))?;
    let (domain_name, domain) = attribute_domain(stored_domain)?;
    let attribute_type = attribute_type(
        attribute
            .get("type")
            .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no type")))?,
    )?;
    let selected = required_selection(mesh, args)?;
    if selected.domain != domain {
        return Err(invalid(format!(
            "mesh attribute `{name}` uses the {domain_name} domain"
        )));
    }
    let value = validate_attribute_value(
        attribute_type,
        args.get("value")
            .ok_or_else(|| invalid("attribute_update requires value"))?,
    )?;
    let updates = selected
        .ids
        .into_iter()
        .map(|id| {
            let key = format!("{}{id}", domain.prefix());
            let updated = if domain_name == "corner" {
                let face = mesh
                    .faces
                    .iter()
                    .find(|face| face.id == id)
                    .ok_or_else(|| invalid(format!("face f{id} does not exist")))?;
                Value::Array(vec![value.clone(); face.vertices.len()])
            } else {
                value.clone()
            };
            Ok((key, updated))
        })
        .collect::<Result<Vec<_>>>()?;
    let values = attribute_values_mut(mesh, key)?;
    for (key, updated) in updates {
        values.insert(key, updated);
    }
    Ok(IdChanges::default())
}

fn apply_attribute_delete(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid("attribute_delete requires a non-empty name"))?;
    reject_reserved_attribute(name)?;
    let storage = storage_key(name);
    let key: &str = &storage;
    if !mesh.attributes.contains_key(key) {
        return Err(invalid(format!("mesh attribute `{name}` does not exist")));
    }
    let is_color = mesh.attributes[key].get("type").and_then(Value::as_str) == Some("color");
    let Some(elements) = args.get("elements") else {
        if is_color {
            // The registry owns a color layer's values, so dropping the layer from it removes them.
            color_layers::validate(mesh)?;
            let mut store = color_layers::read(mesh);
            store.layers.retain(|layer| layer.name != name);
            mesh.attributes.remove(key);
            color_layers::write(mesh, &store);
        } else {
            mesh.attributes.remove(key);
        }
        return Ok(IdChanges::default());
    };
    if is_color {
        // A color layer holds a value for every element, so selected elements reset to opaque black.
        let mut reset = args.clone();
        reset.insert("value".to_owned(), json!([0.0, 0.0, 0.0, 1.0]));
        return apply_attribute_update(mesh, &reset);
    }
    let mut selection_args = Map::new();
    selection_args.insert("elements".to_owned(), elements.clone());
    let selected = required_selection(mesh, &selection_args)?;
    let attribute_domain_name = mesh.attributes[key]
        .get("domain")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no domain")))?;
    let (_, domain) = attribute_domain(&Value::String(attribute_domain_name.to_owned()))?;
    if selected.domain != domain {
        return Err(invalid(format!(
            "mesh attribute `{name}` uses the {attribute_domain_name} domain"
        )));
    }
    let values = attribute_values_mut(mesh, key)?;
    for id in selected.ids {
        values.remove(&format!("{}{id}", domain.prefix()));
    }
    Ok(IdChanges::default())
}

fn interpolate_attribute_value(
    attribute_type: &str,
    first: &Value,
    second: &Value,
    first_weight: f64,
) -> Option<Value> {
    let blend = |a: f64, b: f64| a.mul_add(first_weight, b * (1.0 - first_weight));
    match attribute_type {
        "float" => Some(json!(blend(first.as_f64()?, second.as_f64()?))),
        "float2" | "float3" | "color" | "quaternion" => {
            let first_values = first.as_array()?;
            let second_values = second.as_array()?;
            if first_values.len() != second_values.len() {
                return None;
            }
            let mut components = first_values
                .iter()
                .zip(second_values)
                .map(|(a, b)| Some(blend(a.as_f64()?, b.as_f64()?)))
                .collect::<Option<Vec<_>>>()?;
            if attribute_type == "quaternion" {
                let length = components
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if length > f64::EPSILON {
                    for value in &mut components {
                        *value /= length;
                    }
                }
            }
            Some(json!(components))
        }
        _ => Some(first.clone()),
    }
}

fn average_attribute_values(attribute_type: &str, values: &[Value]) -> Option<Value> {
    let count = f64::from(u32::try_from(values.len()).ok()?);
    if count <= 0.0 {
        return None;
    }
    if attribute_type == "float" {
        let sum = values
            .iter()
            .try_fold(0.0, |sum, value| Some(sum + value.as_f64()?))?;
        return Some(json!(sum / count));
    }
    if !["float2", "float3", "color", "quaternion"].contains(&attribute_type) {
        return values.first().cloned();
    }
    let first = values.first()?.as_array()?;
    let mut components = vec![0.0; first.len()];
    for value in values {
        let value = value.as_array()?;
        if value.len() != components.len() {
            return None;
        }
        for (sum, component) in components.iter_mut().zip(value) {
            *sum += component.as_f64()?;
        }
    }
    for component in &mut components {
        *component /= count;
    }
    if attribute_type == "quaternion" {
        let length = components
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if length > f64::EPSILON {
            for value in &mut components {
                *value /= length;
            }
        }
    }
    Some(json!(components))
}

fn nearest_attribute_value(
    candidates: &[(DVec3, Value)],
    position: DVec3,
    attribute_type: &str,
) -> Option<Value> {
    let mut nearest: Option<(f64, &Value)> = None;
    let mut second: Option<(f64, &Value)> = None;
    for (candidate_position, value) in candidates {
        let distance = position.distance(*candidate_position);
        match nearest {
            None => nearest = Some((distance, value)),
            Some((nearest_distance, _)) if distance < nearest_distance => {
                second = nearest;
                nearest = Some((distance, value));
            }
            _ if second.is_none_or(|(second_distance, _)| distance < second_distance) => {
                second = Some((distance, value));
            }
            _ => {}
        }
    }
    let (nearest_distance, nearest_value) = nearest?;
    if nearest_distance <= f64::EPSILON {
        return Some(nearest_value.clone());
    }
    let (second_distance, second_value) = second?;
    let total = nearest_distance + second_distance;
    let nearest_weight = if total <= f64::EPSILON {
        0.5
    } else {
        second_distance / total
    };
    interpolate_attribute_value(attribute_type, nearest_value, second_value, nearest_weight)
}

fn interpolate_untyped_value(first: &Value, second: &Value, first_weight: f64) -> Option<Value> {
    let blend = |a: f64, b: f64| a.mul_add(first_weight, b * (1.0 - first_weight));
    if let (Some(first), Some(second)) = (first.as_f64(), second.as_f64()) {
        return Some(json!(blend(first, second)));
    }
    let (Some(first), Some(second)) = (first.as_array(), second.as_array()) else {
        return Some(first.clone());
    };
    if first.len() != second.len() {
        return Some(Value::Array(first.clone()));
    }
    first
        .iter()
        .zip(second)
        .map(|(first, second)| Some(json!(blend(first.as_f64()?, second.as_f64()?))))
        .collect::<Option<Vec<_>>>()
        .map(Value::Array)
        .or_else(|| Some(Value::Array(first.clone())))
}

fn nearest_untyped_attribute_value(
    candidates: &[(DVec3, Value)],
    position: DVec3,
) -> Option<Value> {
    let mut nearest: Option<(f64, &Value)> = None;
    let mut second: Option<(f64, &Value)> = None;
    for (candidate_position, value) in candidates {
        let distance = position.distance(*candidate_position);
        match nearest {
            None => nearest = Some((distance, value)),
            Some((nearest_distance, _)) if distance < nearest_distance => {
                second = nearest;
                nearest = Some((distance, value));
            }
            _ if second.is_none_or(|(second_distance, _)| distance < second_distance) => {
                second = Some((distance, value));
            }
            _ => {}
        }
    }
    let (nearest_distance, nearest_value) = nearest?;
    if nearest_distance <= f64::EPSILON || second.is_none() {
        return Some(nearest_value.clone());
    }
    let (second_distance, second_value) = second?;
    let total = nearest_distance + second_distance;
    let nearest_weight = if total <= f64::EPSILON {
        0.5
    } else {
        second_distance / total
    };
    interpolate_untyped_value(nearest_value, second_value, nearest_weight)
}

fn interpolate_legacy_attribute(
    source: &Mesh,
    target: &mut Mesh,
    name: &str,
    domain_value: &Value,
    source_values: &Map<String, Value>,
) -> Result<()> {
    let (domain_name, _) = attribute_domain(domain_value)?;
    let current_values = target
        .attributes
        .get(name)
        .and_then(Value::as_object)
        .and_then(|fields| fields.get("values"))
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no values object")))?;
    let key_prefix = match domain_name {
        "point" => 'v',
        "edge" => 'e',
        "face" | "corner" => 'f',
        _ => return Err(invalid("unsupported attribute domain")),
    };
    let ids = match domain_name {
        "point" => target
            .vertices
            .iter()
            .map(|item| item.id)
            .collect::<Vec<_>>(),
        "edge" => target.edges.iter().map(|item| item.id).collect::<Vec<_>>(),
        "face" | "corner" => target.faces.iter().map(|item| item.id).collect::<Vec<_>>(),
        _ => return Err(invalid("unsupported attribute domain")),
    };
    let alive: HashSet<String> = ids.iter().map(|id| format!("{key_prefix}{id}")).collect();
    let mut additions = Vec::new();
    for id in ids {
        let key = format!("{key_prefix}{id}");
        let has_current_value = current_values.get(&key).is_some_and(|value| {
            domain_name != "corner"
                || target
                    .faces
                    .iter()
                    .find(|face| face.id == id)
                    .is_some_and(|face| {
                        value
                            .as_array()
                            .is_some_and(|corners| corners.len() == face.vertices.len())
                    })
        });
        if has_current_value {
            continue;
        }
        let value = match domain_name {
            "point" => {
                let position = target.vertex(id).map_or(DVec3::ZERO, |vertex| vertex.co);
                let candidates = source
                    .vertices
                    .iter()
                    .filter_map(|vertex| {
                        source_values
                            .get(&format!("v{}", vertex.id))
                            .map(|value| (vertex.co, value.clone()))
                    })
                    .collect::<Vec<_>>();
                nearest_untyped_attribute_value(&candidates, position)
            }
            "edge" => {
                let edge = target
                    .edges
                    .iter()
                    .find(|edge| edge.id == id)
                    .ok_or_else(|| invalid(format!("edge e{id} does not exist")))?;
                let position = (target
                    .vertex(edge.vertices[0])
                    .map_or(DVec3::ZERO, |vertex| vertex.co)
                    + target
                        .vertex(edge.vertices[1])
                        .map_or(DVec3::ZERO, |vertex| vertex.co))
                    * 0.5;
                let candidates = source
                    .edges
                    .iter()
                    .filter_map(|edge| {
                        let value = source_values.get(&format!("e{}", edge.id))?;
                        let first = source.vertex(edge.vertices[0])?.co;
                        let second = source.vertex(edge.vertices[1])?.co;
                        Some(((first + second) * 0.5, value.clone()))
                    })
                    .collect::<Vec<_>>();
                nearest_untyped_attribute_value(&candidates, position)
            }
            "face" => {
                let face = target
                    .faces
                    .iter()
                    .find(|face| face.id == id)
                    .ok_or_else(|| invalid(format!("face f{id} does not exist")))?;
                let candidates = source
                    .faces
                    .iter()
                    .filter_map(|face| {
                        source_values.get(&format!("f{}", face.id)).map(|value| {
                            (element_center(source, Domain::Face, face.id), value.clone())
                        })
                    })
                    .collect::<Vec<_>>();
                nearest_untyped_attribute_value(
                    &candidates,
                    element_center(target, Domain::Face, face.id),
                )
            }
            "corner" => {
                let face = target
                    .faces
                    .iter()
                    .find(|face| face.id == id)
                    .ok_or_else(|| invalid(format!("face f{id} does not exist")))?;
                let corners = face
                    .vertices
                    .iter()
                    .map(|vertex_id| {
                        let position = target
                            .vertex(*vertex_id)
                            .map_or(DVec3::ZERO, |vertex| vertex.co);
                        let mut candidates = Vec::new();
                        for source_face in &source.faces {
                            let Some(values) = source_values
                                .get(&format!("f{}", source_face.id))
                                .and_then(Value::as_array)
                            else {
                                continue;
                            };
                            for (index, source_vertex) in source_face.vertices.iter().enumerate() {
                                if let (Some(vertex), Some(value)) =
                                    (source.vertex(*source_vertex), values.get(index))
                                {
                                    candidates.push((vertex.co, value.clone()));
                                }
                            }
                        }
                        nearest_untyped_attribute_value(&candidates, position)
                    })
                    .collect::<Option<Vec<_>>>();
                corners.map(Value::Array)
            }
            _ => None,
        };
        if let Some(value) = value {
            additions.push((key, value));
        }
    }
    let fields = target
        .attributes
        .get_mut(name)
        .and_then(Value::as_object_mut)
        .ok_or_else(|| invalid(format!("mesh attribute `{name}` is not an object")))?;
    let values = fields
        .get_mut("values")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no values object")))?;
    values.retain(|key, _| alive.contains(key));
    values.extend(additions);
    Ok(())
}

/// Interpolates every UV layer of `source` onto `target`, keeping layer order, names, and the
/// active and render layers. A target face copies its source face's corners when both share the
/// face ID; otherwise each corner takes the nearest source corner value. Ambiguous corners leave
/// the face without an entry.
fn interpolate_uv_layers(source: &Mesh, target: &mut Mesh) -> Result<()> {
    let source_store = uv_layers::read(source);
    if source_store.layers.is_empty() {
        return Ok(());
    }
    let source_index: HashMap<u32, usize> = source
        .faces
        .iter()
        .enumerate()
        .map(|(index, face)| (face.id, index))
        .collect();
    let mut layers = Vec::with_capacity(source_store.layers.len());
    for source_layer in &source_store.layers {
        let has_pins = source_layer
            .pinned
            .iter()
            .any(|flags| flags.as_ref().is_some_and(|flags| !flags.is_empty()));
        let mut layer = UvLayer::new(source_layer.name.clone(), vec![None; target.faces.len()]);
        for (target_index, face) in target.faces.iter().enumerate() {
            if let Some(&source_face) = source_index.get(&face.id)
                && let Some(corners) = source_layer.faces[source_face].as_ref()
                && corners.len() == face.vertices.len()
            {
                layer.faces[target_index] = Some(corners.clone());
                layer.pinned[target_index].clone_from(&source_layer.pinned[source_face]);
                continue;
            }
            let mut uv = Vec::with_capacity(face.vertices.len());
            let mut pinned = Vec::with_capacity(face.vertices.len());
            for vertex_id in &face.vertices {
                let position = target
                    .vertex(*vertex_id)
                    .ok_or_else(|| invalid("UV face references a missing vertex"))?
                    .co;
                let mut candidates = Vec::new();
                let mut nearest_pin: Option<(f64, bool)> = None;
                for (source_face_index, source_face) in source.faces.iter().enumerate() {
                    let Some(corners) = source_layer.faces[source_face_index]
                        .as_ref()
                        .filter(|corners| corners.len() == source_face.vertices.len())
                    else {
                        continue;
                    };
                    let pin_values = source_layer.pinned[source_face_index].as_ref();
                    for (index, source_vertex) in source_face.vertices.iter().enumerate() {
                        let Some(source_vertex) = source.vertex(*source_vertex) else {
                            continue;
                        };
                        let Some(corner_uv) = corners.get(index) else {
                            continue;
                        };
                        candidates.push((source_vertex.co, json!(corner_uv)));
                        if let Some(is_pinned) =
                            pin_values.and_then(|values| values.get(index)).copied()
                        {
                            let distance = position.distance(source_vertex.co);
                            if nearest_pin
                                .is_none_or(|(nearest_distance, _)| distance < nearest_distance)
                            {
                                nearest_pin = Some((distance, is_pinned));
                            }
                        }
                    }
                }
                let Some(coordinate) = nearest_untyped_attribute_value(&candidates, position)
                else {
                    uv.clear();
                    break;
                };
                uv.push(
                    serde_json::from_value::<[f64; 2]>(coordinate)
                        .map_err(|_| invalid("interpolated UV corner is not a coordinate pair"))?,
                );
                if has_pins {
                    pinned.push(nearest_pin.is_some_and(|(_, value)| value));
                }
            }
            if uv.len() != face.vertices.len() {
                continue;
            }
            layer.faces[target_index] = Some(uv);
            layer.pinned[target_index] = has_pins.then_some(pinned);
        }
        layers.push(layer);
    }
    let store = UvStore {
        layers,
        active: source_store.active,
        render: source_store.render,
    };
    uv_layers::write(target, &store);
    Ok(())
}

fn interpolate_attributes(source: &Mesh, target: &mut Mesh) -> Result<()> {
    for (name, source_attribute) in &source.attributes {
        // `custom_normal` is a `normal_code` store; `carry_custom_normals` rebuilds it from vectors.
        if name.as_str() == "custom_normal" {
            continue;
        }
        let Some(source_fields) = source_attribute.as_object() else {
            continue;
        };
        let (Some(domain_value), Some(type_value)) =
            (source_fields.get("domain"), source_fields.get("type"))
        else {
            continue;
        };
        let (domain_name, _) = attribute_domain(domain_value)?;
        let attribute_type = attribute_type(type_value)?;
        let source_values = source_fields
            .get("values")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no values object")))?;
        let current_values = target
            .attributes
            .get(name)
            .and_then(Value::as_object)
            .and_then(|fields| fields.get("values"))
            .and_then(Value::as_object)
            .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no values object")))?;
        let key_prefix = if domain_name == "point" {
            'v'
        } else if domain_name == "edge" {
            'e'
        } else {
            'f'
        };
        let ids = match domain_name {
            "point" => target
                .vertices
                .iter()
                .map(|item| item.id)
                .collect::<Vec<_>>(),
            "edge" => target.edges.iter().map(|item| item.id).collect::<Vec<_>>(),
            "face" | "corner" => target.faces.iter().map(|item| item.id).collect::<Vec<_>>(),
            _ => return Err(invalid("unsupported attribute domain")),
        };
        let alive: HashSet<String> = ids.iter().map(|id| format!("{key_prefix}{id}")).collect();
        let mut additions = Vec::new();
        for id in &ids {
            let key = format!("{key_prefix}{id}");
            if current_values.contains_key(&key) {
                continue;
            }
            let value = match domain_name {
                "point" => {
                    let position = target.vertex(*id).map_or(DVec3::ZERO, |vertex| vertex.co);
                    let face_average = source.faces.iter().find_map(|face| {
                        if element_center(source, Domain::Face, face.id).distance(position)
                            > 1.0e-12
                        {
                            return None;
                        }
                        let values = face
                            .vertices
                            .iter()
                            .map(|vertex| source_values.get(&format!("v{vertex}")).cloned())
                            .collect::<Option<Vec<_>>>()?;
                        average_attribute_values(attribute_type, &values)
                    });
                    face_average.or_else(|| {
                        let candidates = source
                            .vertices
                            .iter()
                            .filter_map(|vertex| {
                                source_values
                                    .get(&format!("v{}", vertex.id))
                                    .map(|value| (vertex.co, value.clone()))
                            })
                            .collect::<Vec<_>>();
                        nearest_attribute_value(&candidates, position, attribute_type)
                    })
                }
                "edge" => {
                    let edge = target
                        .edges
                        .iter()
                        .find(|edge| edge.id == *id)
                        .ok_or_else(|| invalid(format!("edge e{id} does not exist")))?;
                    let position = (target
                        .vertex(edge.vertices[0])
                        .map_or(DVec3::ZERO, |vertex| vertex.co)
                        + target
                            .vertex(edge.vertices[1])
                            .map_or(DVec3::ZERO, |vertex| vertex.co))
                        * 0.5;
                    let candidates = source
                        .edges
                        .iter()
                        .filter_map(|edge| {
                            let value = source_values.get(&format!("e{}", edge.id))?;
                            let first = source.vertex(edge.vertices[0])?.co;
                            let second = source.vertex(edge.vertices[1])?.co;
                            Some(((first + second) * 0.5, value.clone()))
                        })
                        .collect::<Vec<_>>();
                    nearest_attribute_value(&candidates, position, attribute_type)
                }
                "face" => {
                    let face = target
                        .faces
                        .iter()
                        .find(|face| face.id == *id)
                        .ok_or_else(|| invalid(format!("face f{id} does not exist")))?;
                    let position = element_center(target, Domain::Face, face.id);
                    let candidates = source
                        .faces
                        .iter()
                        .filter_map(|face| {
                            source_values.get(&format!("f{}", face.id)).map(|value| {
                                (element_center(source, Domain::Face, face.id), value.clone())
                            })
                        })
                        .collect::<Vec<_>>();
                    nearest_attribute_value(&candidates, position, attribute_type)
                }
                "corner" => {
                    let face = target
                        .faces
                        .iter()
                        .find(|face| face.id == *id)
                        .ok_or_else(|| invalid(format!("face f{id} does not exist")))?;
                    let corners = face
                        .vertices
                        .iter()
                        .map(|vertex_id| {
                            let position = target
                                .vertex(*vertex_id)
                                .map_or(DVec3::ZERO, |vertex| vertex.co);
                            let mut candidates = Vec::new();
                            for source_face in &source.faces {
                                let Some(values) = source_values
                                    .get(&format!("f{}", source_face.id))
                                    .and_then(Value::as_array)
                                else {
                                    continue;
                                };
                                for (index, source_vertex) in
                                    source_face.vertices.iter().enumerate()
                                {
                                    if let (Some(vertex), Some(value)) =
                                        (source.vertex(*source_vertex), values.get(index))
                                    {
                                        candidates.push((vertex.co, value.clone()));
                                    }
                                }
                            }
                            nearest_attribute_value(&candidates, position, attribute_type)
                                .ok_or_else(|| {
                                    invalid(format!(
                                        "corner attribute `{name}` has no source values"
                                    ))
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Some(Value::Array(corners))
                }
                _ => None,
            }
            .ok_or_else(|| invalid(format!("attribute `{name}` cannot be interpolated")))?;
            additions.push((key, value));
        }
        let fields = target
            .attributes
            .get_mut(name)
            .and_then(Value::as_object_mut)
            .ok_or_else(|| invalid(format!("mesh attribute `{name}` is not an object")))?;
        let values = fields
            .get_mut("values")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no values object")))?;
        values.retain(|key, _| alive.contains(key));
        values.extend(additions);
    }
    for (name, source_attribute) in &source.attributes {
        let Some(fields) = source_attribute.as_object() else {
            continue;
        };
        if fields.contains_key("type") {
            continue;
        }
        let (Some(domain), Some(values)) = (fields.get("domain"), fields.get("values")) else {
            continue;
        };
        let Some(values) = values.as_object() else {
            return Err(invalid(format!(
                "mesh attribute `{name}` has no values object"
            )));
        };
        interpolate_legacy_attribute(source, target, name, domain, values)?;
    }
    interpolate_uv_layers(source, target)?;
    Ok(())
}

fn has_interpolatable_attributes(mesh: &Mesh) -> bool {
    mesh.attributes.contains_key(uv_layers::ATTRIBUTE)
        || mesh.attributes.iter().any(|(name, attribute)| {
            name.as_str() != "custom_normal"
                && attribute.as_object().is_some_and(|fields| {
                    fields.contains_key("domain") && fields.contains_key("values")
                })
        })
}

/// Carries `custom_normal` across an edit that changes faces. Codes are relative to their fan, so
/// the source's evaluated corner vectors are decoded before the edit, keyed by face ID, and encoded
/// again on the edited mesh. A face that keeps its ID and corner count keeps its vectors; new or
/// resized faces are automatic. The edited mesh takes its storage order as its loop basis.
fn carry_custom_normals(source: &Mesh, target: &mut Mesh) -> Result<()> {
    // Remesh replaces the mesh without a custom-normal layer, and that absence is kept.
    if !target.attributes.contains_key("custom_normal") {
        return Ok(());
    }
    let positions: Vec<DVec3> = source.vertices.iter().map(|vertex| vertex.co).collect();
    let vectors = clnor::corner_custom_normals(source, &positions, NormalCodeOptions::default())?;
    // The stale layer goes first: its rows are keyed by faces the edit may have removed, and the
    // setter reads it.
    target.attributes.remove("custom_normal");
    let Some(vectors) = vectors else {
        return Ok(());
    };
    let mut source_corners = HashMap::with_capacity(source.faces.len());
    let mut start = 0;
    for face in &source.faces {
        let end = start + face.vertices.len();
        source_corners.insert(face.id, &vectors[start..end]);
        start = end;
    }
    let carried: Vec<Option<DVec3>> = target
        .faces
        .iter()
        .flat_map(|face| match source_corners.get(&face.id) {
            Some(corners) if corners.len() == face.vertices.len() => corners.to_vec(),
            _ => vec![None; face.vertices.len()],
        })
        .collect();
    clnor::encode_vectors(
        target,
        Some(carried.as_slice()),
        NormalCodeOptions::default(),
    )
}

fn is_topology_operation(operation: &str, args: &Map<String, Value>) -> bool {
    matches!(
        operation,
        "extrude"
            | "inset"
            | "bevel"
            | "subdivide"
            | "triangulate"
            | "poke"
            | "rip"
            | "merge"
            | "dissolve"
            | "weld"
            | "delete"
            | "fill"
            | "bisect"
            | "loop_cut"
            | "bridge"
            | "split"
            | "knife"
            | "symmetrize"
            | "remesh"
            | "mirror"
    ) || matches!(operation, "spin" | "screw")
        && args
            .get("elements")
            .and_then(Value::as_object)
            .and_then(|elements| elements.get("domain"))
            .and_then(Value::as_str)
            == Some("face")
}

#[derive(Clone, Copy)]
struct MdispPatch {
    quad: [DVec3; 4],
    axis_x: DVec3,
    axis_y: DVec3,
}

#[derive(Clone, Copy)]
struct MdispFaceInfo {
    normal: DVec3,
    loop_start: usize,
    corner_count: usize,
    patch_start: usize,
}

struct MdispSourceLayout {
    faces: Vec<MdispFaceInfo>,
    patches: Vec<MdispPatch>,
    face_by_id: HashMap<u32, usize>,
}

#[derive(Clone, Copy)]
struct MdispSample {
    loop_index: usize,
    uv: DVec2,
    patch: MdispPatch,
}

fn multires_loop_layout_changed(source: &Mesh, target: &Mesh) -> bool {
    source.faces.len() != target.faces.len()
        || source
            .faces
            .iter()
            .zip(&target.faces)
            .any(|(source_face, target_face)| {
                source_face.id != target_face.id || source_face.vertices != target_face.vertices
            })
}

fn multires_face_center(mesh: &Mesh, face: &Face) -> Result<DVec3> {
    let mut center = DVec3::ZERO;
    for vertex_id in &face.vertices {
        center += mesh
            .vertex(*vertex_id)
            .ok_or_else(|| invalid("multires face references a missing vertex"))?
            .co;
    }
    Ok(center / face.vertices.len() as f64)
}

fn multires_face_normal_vector(mesh: &Mesh, face: &Face) -> Result<DVec3> {
    let origin = mesh
        .vertex(face.vertices[0])
        .ok_or_else(|| invalid("multires face references a missing vertex"))?
        .co;
    let mut normal = DVec3::ZERO;
    for index in 1..face.vertices.len() - 1 {
        let current = mesh
            .vertex(face.vertices[index])
            .ok_or_else(|| invalid("multires face references a missing vertex"))?
            .co;
        let next = mesh
            .vertex(face.vertices[index + 1])
            .ok_or_else(|| invalid("multires face references a missing vertex"))?
            .co;
        normal += (current - origin).cross(next - origin);
    }
    Ok(normal)
}

fn multires_face_normal(mesh: &Mesh, face: &Face) -> Result<DVec3> {
    Ok(multires_face_normal_vector(mesh, face)?.normalize_or_zero())
}

fn multires_vertex_normals(mesh: &Mesh) -> Result<HashMap<u32, DVec3>> {
    let mut normals = HashMap::with_capacity(mesh.vertices.len());
    for face in &mesh.faces {
        for vertex_id in &face.vertices {
            normals.entry(*vertex_id).or_insert(DVec3::ZERO);
        }
    }
    for face in &mesh.faces {
        let face_normal = multires_face_normal_vector(mesh, face)?;
        for vertex_id in &face.vertices {
            let normal = normals
                .get_mut(vertex_id)
                .ok_or_else(|| invalid("multires target vertex normal mapping is invalid"))?;
            *normal += face_normal;
        }
    }
    for normal in normals.values_mut() {
        *normal = normal.normalize_or_zero();
    }
    Ok(normals)
}

fn multires_corner_patch(
    mesh: &Mesh,
    face: &Face,
    corner: usize,
    center: DVec3,
) -> Result<MdispPatch> {
    let corner_count = face.vertices.len();
    let previous = mesh
        .vertex(face.vertices[(corner + corner_count - 1) % corner_count])
        .ok_or_else(|| invalid("multires face references a missing vertex"))?
        .co;
    let current = mesh
        .vertex(face.vertices[corner])
        .ok_or_else(|| invalid("multires face references a missing vertex"))?
        .co;
    let next = mesh
        .vertex(face.vertices[(corner + 1) % corner_count])
        .ok_or_else(|| invalid("multires face references a missing vertex"))?
        .co;
    let quad = [
        center,
        (previous + current) * 0.5,
        current,
        (current + next) * 0.5,
    ];
    Ok(MdispPatch {
        axis_x: (quad[3] - quad[0]).normalize_or_zero(),
        axis_y: (quad[1] - quad[0]).normalize_or_zero(),
        quad,
    })
}

fn build_mdisp_source_layout(source: &Mesh) -> Result<MdispSourceLayout> {
    let loop_count = source.faces.iter().try_fold(0_usize, |count, face| {
        count.checked_add(face.vertices.len())
    });
    let loop_count = loop_count.ok_or_else(|| invalid("multires face corner count overflows"))?;
    let mut layout = MdispSourceLayout {
        faces: Vec::with_capacity(source.faces.len()),
        patches: Vec::with_capacity(loop_count),
        face_by_id: HashMap::with_capacity(source.faces.len()),
    };
    let mut loop_start = 0_usize;
    for (face_index, face) in source.faces.iter().enumerate() {
        let center = multires_face_center(source, face)?;
        let patch_start = layout.patches.len();
        layout.face_by_id.insert(face.id, face_index);
        for corner in 0..face.vertices.len() {
            layout
                .patches
                .push(multires_corner_patch(source, face, corner, center)?);
        }
        layout.faces.push(MdispFaceInfo {
            normal: multires_face_normal(source, face)?,
            loop_start,
            corner_count: face.vertices.len(),
            patch_start,
        });
        loop_start = loop_start
            .checked_add(face.vertices.len())
            .ok_or_else(|| invalid("multires face corner count overflows"))?;
    }
    Ok(layout)
}

fn cross_2d(first: DVec2, second: DVec2) -> f64 {
    first.x * second.y - first.y * second.x
}

fn mdisp_quad_uv(patch: MdispPatch, point: DVec3, normal: DVec3) -> Option<DVec2> {
    let normal = normal.normalize_or_zero();
    if normal.length_squared() <= f64::EPSILON {
        return None;
    }
    let quad_normal = (patch.quad[1] - patch.quad[0]).cross(patch.quad[2] - patch.quad[1])
        + (patch.quad[2] - patch.quad[1]).cross(patch.quad[3] - patch.quad[2]);
    if normal.dot(quad_normal) < -4.0e-4 {
        return None;
    }
    let axis_x = (patch.axis_x - normal * patch.axis_x.dot(normal)).normalize_or_zero();
    if axis_x.length_squared() <= f64::EPSILON {
        return None;
    }
    let axis_y = normal.cross(axis_x).normalize_or_zero();
    let project = |coordinate: DVec3| {
        let relative = coordinate - point;
        DVec2::new(relative.dot(axis_x), relative.dot(axis_y))
    };
    let q00 = project(patch.quad[0]);
    let q10 = project(patch.quad[3]);
    let q11 = project(patch.quad[2]);
    let q01 = project(patch.quad[1]);
    let linear_u = q10 - q00;
    let linear_v = q01 - q00;
    let bilinear = q00 - q10 + q11 - q01;
    let mut uv = DVec2::splat(0.5);
    for _ in 0..12 {
        let residual = q00 + linear_u * uv.x + linear_v * uv.y + bilinear * (uv.x * uv.y);
        let derivative_u = linear_u + bilinear * uv.y;
        let derivative_v = linear_v + bilinear * uv.x;
        let determinant = cross_2d(derivative_u, derivative_v);
        if determinant.abs() <= 1.0e-14 {
            return None;
        }
        let delta = DVec2::new(
            cross_2d(-residual, derivative_v) / determinant,
            cross_2d(derivative_u, -residual) / determinant,
        );
        uv += delta;
        if delta.length_squared() <= 1.0e-20 {
            break;
        }
    }
    let residual = q00 + linear_u * uv.x + linear_v * uv.y + bilinear * (uv.x * uv.y);
    let scale = q00
        .length()
        .max(q10.length())
        .max(q11.length())
        .max(q01.length())
        .max(1.0);
    if residual.length() > scale * 1.0e-6
        || uv.x < -1.0e-4
        || uv.x > 1.0 + 1.0e-4
        || uv.y < -1.0e-4
        || uv.y > 1.0 + 1.0e-4
    {
        return None;
    }
    Some(uv.clamp(DVec2::ZERO, DVec2::ONE))
}

fn source_mdisp_sample(
    layout: &MdispSourceLayout,
    face_index: usize,
    point: DVec3,
    target_normal: DVec3,
) -> Option<MdispSample> {
    let info = layout.faces[face_index];
    let normal = if target_normal.length_squared() > f64::EPSILON {
        target_normal
    } else {
        info.normal
    };
    for corner in 0..info.corner_count {
        let patch = layout.patches[info.patch_start + corner];
        if let Some(uv) = mdisp_quad_uv(patch, point, normal) {
            return Some(MdispSample {
                loop_index: info.loop_start + corner,
                uv,
                patch,
            });
        }
    }
    None
}

fn mdisp_level(data: &MultiresGridData, loop_index: usize) -> u32 {
    data.grid_levels
        .as_ref()
        .and_then(|levels| levels.get(loop_index))
        .copied()
        .unwrap_or(data.total_levels)
}

fn mdisp_grid_point(patch: MdispPatch, u: f64, v: f64) -> DVec3 {
    let left = patch.quad[0].lerp(patch.quad[1], v);
    let right = patch.quad[3].lerp(patch.quad[2], v);
    left.lerp(right, u)
}

fn sample_mdisp_vector(grid: &[[f32; 3]], side: usize, uv: DVec2) -> [f32; 3] {
    let maximum = (side - 1) as f64;
    let x = uv.x.clamp(0.0, 1.0) * maximum;
    let y = uv.y.clamp(0.0, 1.0) * maximum;
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(side - 1);
    let y1 = (y0 + 1).min(side - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    std::array::from_fn(|axis| {
        let lower = f64::from(grid[y0 * side + x0][axis]) * (1.0 - tx)
            + f64::from(grid[y0 * side + x1][axis]) * tx;
        let upper = f64::from(grid[y1 * side + x0][axis]) * (1.0 - tx)
            + f64::from(grid[y1 * side + x1][axis]) * tx;
        (lower * (1.0 - ty) + upper * ty) as f32
    })
}

fn sample_mdisp_hidden(grid: &[bool], side: usize, uv: DVec2) -> bool {
    if side == 0 {
        return false;
    }
    let maximum = (side - 1) as f64;
    let x = (uv.x.clamp(0.0, 1.0) * maximum).round() as usize;
    let y = (uv.y.clamp(0.0, 1.0) * maximum).round() as usize;
    grid[y * side + x]
}

fn rebase_mdisp_vector(
    value: [f32; 3],
    source_patch: MdispPatch,
    target_patch: MdispPatch,
) -> Result<[f32; 3]> {
    let mut displacement =
        source_patch.axis_x * f64::from(value[0]) + source_patch.axis_y * f64::from(value[1]);
    let target_normal = target_patch
        .axis_x
        .cross(target_patch.axis_y)
        .normalize_or_zero();
    if target_normal.length_squared() <= f64::EPSILON {
        return Err(invalid(
            "multires target corner has a degenerate tangent frame",
        ));
    }
    displacement -= target_normal * displacement.dot(target_normal);
    let xx = target_patch.axis_x.dot(target_patch.axis_x);
    let xy = target_patch.axis_x.dot(target_patch.axis_y);
    let yy = target_patch.axis_y.dot(target_patch.axis_y);
    let x_displacement = target_patch.axis_x.dot(displacement);
    let y_displacement = target_patch.axis_y.dot(displacement);
    let determinant = xx * yy - xy * xy;
    if determinant.abs() <= 1.0e-12 {
        return Err(invalid(
            "multires target corner has a degenerate tangent frame",
        ));
    }
    let x = (x_displacement * yy - y_displacement * xy) / determinant;
    let y = (xx * y_displacement - xy * x_displacement) / determinant;
    let rebased = [x as f32, y as f32, value[2]];
    if !rebased.iter().all(|component| component.is_finite()) {
        return Err(invalid(
            "multires rebasing produced a non-finite displacement",
        ));
    }
    Ok(rebased)
}

// Blender 5.2.2 BM_loop_interp_multires_ex maps destination samples to source quads and rebases tangents.
fn mdisp_source_loop_index(
    layout: &MdispSourceLayout,
    source_loop: MdispSourceLoop,
) -> Result<usize> {
    let face_index = layout
        .face_by_id
        .get(&source_loop.face_id)
        .copied()
        .ok_or_else(|| invalid("multires source-face provenance is invalid"))?;
    let info = layout
        .faces
        .get(face_index)
        .copied()
        .ok_or_else(|| invalid("multires source-face mapping is invalid"))?;
    if source_loop.corner >= info.corner_count {
        return Err(invalid("multires source-corner provenance is invalid"));
    }
    info.loop_start
        .checked_add(source_loop.corner)
        .ok_or_else(|| invalid("multires source loop index overflows"))
}

fn mdisp_loop_source_for_target(
    source: &Mesh,
    target_face: &Face,
    corner: usize,
    layout: &MdispSourceLayout,
    sources_by_face: &HashMap<u32, Vec<MdispLoopSource>>,
) -> Result<MdispLoopSource> {
    if let Some(sources) = sources_by_face.get(&target_face.id) {
        return sources
            .get(corner)
            .cloned()
            .ok_or_else(|| invalid("multires loop provenance does not match the target face"));
    }
    let Some(face_index) = layout.face_by_id.get(&target_face.id).copied() else {
        return Ok(MdispLoopSource::default());
    };
    let source_face = &source.faces[face_index];
    let source_corner = source_face
        .vertices
        .iter()
        .position(|vertex_id| *vertex_id == target_face.vertices[corner]);
    Ok(MdispLoopSource {
        seed: source_corner.map(|corner| MdispSourceLoop {
            face_id: source_face.id,
            corner,
        }),
        callback_faces: if source_face.vertices == target_face.vertices {
            Vec::new()
        } else {
            vec![source_face.id]
        },
    })
}

// Blender 5.2.2's loop interpolation receives its source face(s) from the topology callback.
fn interpolate_multires(
    source: &Mesh,
    target: &Mesh,
    data: &mut MultiresGridData,
    sources_by_face: &HashMap<u32, Vec<MdispLoopSource>>,
) -> Result<()> {
    let layout = build_mdisp_source_layout(source)?;
    let target_corner_count = target.faces.iter().try_fold(0_usize, |count, face| {
        count.checked_add(face.vertices.len())
    });
    let target_corner_count =
        target_corner_count.ok_or_else(|| invalid("multires target corner count overflows"))?;
    let target_vertex_normals = multires_vertex_normals(target)?;
    let fallback_level = data.total_levels;
    if data.grid_levels.is_some() && fallback_level == 0 && target_corner_count > 0 {
        return Err(invalid("multires target grids have no source resolution"));
    }
    let source_grids = data.corner_grids.as_deref();
    let source_masks = data.paint_mask_grids.as_deref();
    let source_hidden = data.hidden_grids.as_deref();
    let mut grid_levels = data
        .grid_levels
        .as_ref()
        .map(|_| Vec::with_capacity(target_corner_count));
    let mut corner_grids = source_grids.map(|_| Vec::with_capacity(target_corner_count));
    let mut paint_mask_levels = data
        .paint_mask_levels
        .as_ref()
        .map(|_| Vec::with_capacity(target_corner_count));
    let mut paint_mask_grids = source_masks.map(|_| Vec::with_capacity(target_corner_count));
    let mut hidden_grids = source_hidden.map(|_| Vec::with_capacity(target_corner_count));
    let mut callback_face_indices = Vec::new();

    for target_face in &target.faces {
        let target_center = multires_face_center(target, target_face)?;
        for corner in 0..target_face.vertices.len() {
            let provenance = mdisp_loop_source_for_target(
                source,
                target_face,
                corner,
                &layout,
                sources_by_face,
            )?;
            callback_face_indices.clear();
            for face_id in &provenance.callback_faces {
                callback_face_indices.push(
                    *layout
                        .face_by_id
                        .get(face_id)
                        .ok_or_else(|| invalid("multires callback source face is missing"))?,
                );
            }
            let copied_loop = provenance
                .seed
                .map(|source_loop| mdisp_source_loop_index(&layout, source_loop))
                .transpose()?;
            // Blender allocates an empty destination grid from the callback face's first loop.
            let callback_first_loop = callback_face_indices
                .first()
                .map(|face_index| layout.faces[*face_index].loop_start);
            let resolution_loop = copied_loop.or(callback_first_loop);
            let level = resolution_loop.map_or(fallback_level, |index| mdisp_level(data, index));
            if level == 0 && grid_levels.is_some() {
                return Err(invalid("multires target grid resolution is invalid"));
            }
            if let Some(levels) = grid_levels.as_mut() {
                levels.push(level);
            }
            let target_normal = target_vertex_normals
                .get(&target_face.vertices[corner])
                .copied()
                .ok_or_else(|| invalid("multires target vertex normal is unavailable"))?;
            let target_patch = multires_corner_patch(target, target_face, corner, target_center)?;
            let side = if corner_grids.is_some() || hidden_grids.is_some() {
                multires::grid_side(level)?
            } else {
                0
            };
            let grid_len = side
                .checked_mul(side)
                .ok_or_else(|| invalid("multires target grid size overflows"))?;
            let copied_grid = if let (Some(index), Some(grids)) = (copied_loop, source_grids) {
                Some(
                    grids
                        .get(index)
                        .ok_or_else(|| invalid("multires copied grid mapping is invalid"))?,
                )
            } else {
                None
            };
            let mut target_grid = corner_grids.as_ref().map(|_| {
                copied_grid.map_or_else(|| vec![[0.0; 3]; grid_len], std::clone::Clone::clone)
            });
            let copied_hidden = if let (Some(index), Some(grids)) = (copied_loop, source_hidden) {
                Some(
                    grids
                        .get(index)
                        .ok_or_else(|| invalid("multires copied hidden-grid mapping is invalid"))?,
                )
            } else {
                None
            };
            let mut target_hidden = hidden_grids.as_ref().map(|_| {
                copied_hidden.map_or_else(|| vec![false; grid_len], std::clone::Clone::clone)
            });
            if side > 0 {
                for y in 0..side {
                    for x in 0..side {
                        let uv =
                            DVec2::new(x as f64 / (side - 1) as f64, y as f64 / (side - 1) as f64);
                        let point = mdisp_grid_point(target_patch, uv.x, uv.y);
                        let index = y * side + x;
                        // Each source-face call takes its first matching loop; later calls can overwrite it.
                        for source_face_index in &callback_face_indices {
                            if let Some(sample) = source_mdisp_sample(
                                &layout,
                                *source_face_index,
                                point,
                                target_normal,
                            ) {
                                if let (Some(output), Some(grids)) =
                                    (target_grid.as_mut(), source_grids)
                                {
                                    let source_side =
                                        multires::grid_side(mdisp_level(data, sample.loop_index))?;
                                    let grid = grids.get(sample.loop_index).ok_or_else(|| {
                                        invalid("multires source grid mapping is invalid")
                                    })?;
                                    let value = sample_mdisp_vector(grid, source_side, sample.uv);
                                    output[index] =
                                        rebase_mdisp_vector(value, sample.patch, target_patch)?;
                                }
                                if let (Some(output), Some(grids)) =
                                    (target_hidden.as_mut(), source_hidden)
                                {
                                    let source_side =
                                        multires::grid_side(mdisp_level(data, sample.loop_index))?;
                                    let grid = grids.get(sample.loop_index).ok_or_else(|| {
                                        invalid("multires source hidden-grid mapping is invalid")
                                    })?;
                                    output[index] =
                                        sample_mdisp_hidden(grid, source_side, sample.uv);
                                }
                            }
                        }
                    }
                }
            }
            if let (Some(grids), Some(grid)) = (corner_grids.as_mut(), target_grid) {
                grids.push(grid);
            }
            if let (Some(grids), Some(grid)) = (hidden_grids.as_mut(), target_hidden) {
                grids.push(grid);
            }
            if let (Some(levels), Some(grids)) =
                (paint_mask_levels.as_mut(), paint_mask_grids.as_mut())
            {
                let (mask_level, mask_grid) = if let Some(index) = copied_loop {
                    let level = data
                        .paint_mask_levels
                        .as_deref()
                        .and_then(|levels| levels.get(index))
                        .copied()
                        .ok_or_else(|| invalid("multires copied paint-mask level is invalid"))?;
                    let grid = source_masks
                        .and_then(|grids| grids.get(index))
                        .ok_or_else(|| invalid("multires copied paint-mask mapping is invalid"))?;
                    (level, grid.clone())
                } else {
                    (0, Vec::new())
                };
                levels.push(mask_level);
                grids.push(mask_grid);
            }
        }
    }
    data.grid_levels = grid_levels;
    data.corner_grids = corner_grids;
    data.paint_mask_levels = paint_mask_levels;
    data.paint_mask_grids = paint_mask_grids;
    data.hidden_grids = hidden_grids;
    Ok(())
}

fn face_winding_reversed(source: &Face, target: &Face) -> bool {
    if source.vertices.len() != target.vertices.len() {
        return false;
    }
    let corner_count = source.vertices.len();
    (0..corner_count).any(|offset| {
        target.vertices[0] == source.vertices[offset]
            && (1..corner_count).all(|corner| {
                target.vertices[corner]
                    == source.vertices[(offset + corner_count - corner) % corner_count]
            })
    })
}

// Blender 5.2.2 BKE_mesh_mdisp_flip(..., true) transposes grids and swaps tangent components.
// The same call negates normal displacement.
fn flip_mdisp_grid(grid: &[[f32; 3]], side: usize) -> Vec<[f32; 3]> {
    let mut flipped = vec![[0.0; 3]; grid.len()];
    for y in 0..side {
        for x in 0..side {
            let value = grid[x * side + y];
            flipped[y * side + x] = [value[1], value[0], -value[2]];
        }
    }
    flipped
}

fn flip_multires(source: &Mesh, target: &Mesh, data: &mut MultiresGridData) -> Result<()> {
    let mut source_faces = HashMap::with_capacity(source.faces.len());
    let mut source_loop_start = 0_usize;
    for (face_index, face) in source.faces.iter().enumerate() {
        source_faces.insert(face.id, (face_index, source_loop_start));
        source_loop_start = source_loop_start
            .checked_add(face.vertices.len())
            .ok_or_else(|| invalid("multires face corner count overflows"))?;
    }
    let target_corner_count = target.faces.iter().try_fold(0_usize, |count, face| {
        count.checked_add(face.vertices.len())
    });
    let target_corner_count =
        target_corner_count.ok_or_else(|| invalid("multires target corner count overflows"))?;
    let mut grid_levels = data
        .grid_levels
        .as_ref()
        .map(|_| Vec::with_capacity(target_corner_count));
    let mut corner_grids = data
        .corner_grids
        .as_ref()
        .map(|_| Vec::with_capacity(target_corner_count));
    let mut paint_mask_levels = data
        .paint_mask_levels
        .as_ref()
        .map(|_| Vec::with_capacity(target_corner_count));
    let mut paint_mask_grids = data
        .paint_mask_grids
        .as_ref()
        .map(|_| Vec::with_capacity(target_corner_count));
    let mut hidden_grids = data
        .hidden_grids
        .as_ref()
        .map(|_| Vec::with_capacity(target_corner_count));

    for target_face in &target.faces {
        let (source_face_index, source_loop_start) = source_faces
            .get(&target_face.id)
            .copied()
            .ok_or_else(|| invalid("normal flip changed a multires face ID"))?;
        let source_face = &source.faces[source_face_index];
        let reversed = face_winding_reversed(source_face, target_face);
        for target_vertex in &target_face.vertices {
            let source_corner = source_face
                .vertices
                .iter()
                .position(|vertex_id| vertex_id == target_vertex)
                .ok_or_else(|| invalid("normal flip changed a multires face corner"))?;
            let source_loop = source_loop_start
                .checked_add(source_corner)
                .ok_or_else(|| invalid("multires source loop index overflows"))?;
            let level = mdisp_level(data, source_loop);
            if let Some(levels) = grid_levels.as_mut() {
                levels.push(level);
            }
            if let (Some(output), Some(grids)) =
                (corner_grids.as_mut(), data.corner_grids.as_deref())
            {
                let grid = grids
                    .get(source_loop)
                    .ok_or_else(|| invalid("multires source grid mapping is invalid"))?;
                output.push(if reversed {
                    flip_mdisp_grid(grid, multires::grid_side(level)?)
                } else {
                    grid.clone()
                });
            }
            if let (Some(output), Some(levels), Some(grids)) = (
                paint_mask_grids.as_mut(),
                data.paint_mask_levels.as_deref(),
                data.paint_mask_grids.as_deref(),
            ) {
                output.push(
                    grids
                        .get(source_loop)
                        .ok_or_else(|| invalid("multires source paint-mask mapping is invalid"))?
                        .clone(),
                );
                if let Some(output_levels) = paint_mask_levels.as_mut() {
                    output_levels.push(
                        *levels.get(source_loop).ok_or_else(|| {
                            invalid("multires source paint-mask level is invalid")
                        })?,
                    );
                }
            }
            if let (Some(output), Some(grids)) =
                (hidden_grids.as_mut(), data.hidden_grids.as_deref())
            {
                let grid = grids
                    .get(source_loop)
                    .ok_or_else(|| invalid("multires source hidden-grid mapping is invalid"))?;
                output.push(grid.clone());
            }
        }
    }
    data.grid_levels = grid_levels;
    data.corner_grids = corner_grids;
    data.paint_mask_levels = paint_mask_levels;
    data.paint_mask_grids = paint_mask_grids;
    data.hidden_grids = hidden_grids;
    Ok(())
}

fn rebase_multires(
    source: &Mesh,
    target: &Mesh,
    data: &mut MultiresGridData,
    operation: &str,
    sources_by_face: &HashMap<u32, Vec<MdispLoopSource>>,
) -> Result<()> {
    if operation == "flip_normals" {
        flip_multires(source, target, data)
    } else {
        interpolate_multires(source, target, data, sources_by_face)
    }
}

fn rebase_subdivide_stage(
    source: &Mesh,
    target: &Mesh,
    data: &mut MultiresGridData,
    sources_by_face: &HashMap<u32, Vec<MdispLoopSource>>,
) -> Result<()> {
    if !multires_loop_layout_changed(source, target) {
        return Ok(());
    }
    if data.corner_grids.is_none() && data.external_filepath.is_some() {
        return Err(invalid(
            "cannot rebase multires data while external MDisps values are unavailable",
        ));
    }
    rebase_multires(source, target, data, "subdivide", sources_by_face)
}

fn apply_shade(mesh: &mut Mesh, selected: &Selection, smooth: bool) -> Result<IdChanges> {
    if selected.domain != Domain::Face {
        return Err(invalid("shade operations require face elements"));
    }
    let attribute = mesh
        .attributes
        .entry("shade_smooth".to_owned())
        .or_insert_with(|| json!({"domain":"face","type":"bool","values":{}}));
    let fields = attribute
        .as_object_mut()
        .ok_or_else(|| invalid("shade_smooth attribute must be an object"))?;
    if fields.get("domain").and_then(Value::as_str) != Some("face")
        || fields.get("type").and_then(Value::as_str) != Some("bool")
    {
        return Err(invalid("shade_smooth must be a boolean face attribute"));
    }
    let values = fields
        .entry("values")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| invalid("shade_smooth values must be an object"))?;
    for id in &selected.ids {
        values.insert(format!("f{id}"), Value::Bool(smooth));
    }
    // Blender's shade_smooth() removes `sharp_face` once every face is smooth; the presence field goes too.
    let all_smooth = mesh
        .faces
        .iter()
        .all(|face| values.get(&format!("f{}", face.id)) == Some(&Value::Bool(true)));
    if smooth && all_smooth {
        fields.remove(crate::geom::normals::SHARP_FACE_PRESENT);
    }
    Ok(IdChanges::default())
}

fn apply_sharp_angle(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let angle = number(args, "angle", std::f64::consts::FRAC_PI_4)?;
    if !(0.0..=std::f64::consts::PI).contains(&angle) {
        return Err(invalid("sharp angle must be between zero and pi radians"));
    }
    let selected = selection(mesh, args, false)?;
    if selected
        .as_ref()
        .is_some_and(|selected| selected.domain != Domain::Edge)
    {
        return Err(invalid("sharp angle selection requires edge elements"));
    }
    let selected_ids = selected.map_or_else(
        || {
            mesh.edges
                .iter()
                .map(|edge| edge.id)
                .collect::<HashSet<_>>()
        },
        |selected| selected.ids.into_iter().collect(),
    );
    let mut sharp_values = Map::new();
    for edge in mesh
        .edges
        .iter()
        .filter(|edge| selected_ids.contains(&edge.id))
    {
        let adjacent = mesh
            .faces
            .iter()
            .filter(|face| {
                (0..face.vertices.len()).any(|index| {
                    edge_key(
                        face.vertices[index],
                        face.vertices[(index + 1) % face.vertices.len()],
                    ) == edge_key(edge.vertices[0], edge.vertices[1])
                })
            })
            .collect::<Vec<_>>();
        let sharp = if adjacent.len() < 2 {
            false
        } else {
            let first = face_normal(mesh, adjacent[0])?;
            adjacent[1..].iter().try_fold(false, |sharp, face| {
                let normal = face_normal(mesh, face)?;
                Ok::<_, PotError>(sharp || first.dot(normal).clamp(-1.0, 1.0).acos() >= angle)
            })?
        };
        sharp_values.insert(format!("e{}", edge.id), Value::Bool(sharp));
    }
    let attribute = mesh
        .attributes
        .entry("sharp".to_owned())
        .or_insert_with(|| json!({"domain":"edge","type":"bool","values":{}}));
    let fields = attribute
        .as_object_mut()
        .ok_or_else(|| invalid("sharp attribute must be an object"))?;
    if fields.get("domain").and_then(Value::as_str) != Some("edge")
        || fields.get("type").and_then(Value::as_str) != Some("bool")
    {
        return Err(invalid("sharp must be a boolean edge attribute"));
    }
    fields
        .entry("values")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| invalid("sharp values must be an object"))?
        .extend(sharp_values);
    Ok(IdChanges::default())
}

fn apply_auto_smooth(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let faces = Selection {
        domain: Domain::Face,
        ids: mesh.faces.iter().map(|face| face.id).collect(),
    };
    apply_shade(mesh, &faces, true)?;
    apply_sharp_angle(mesh, args)
}

fn apply_custom_normals(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let selected = required_selection(mesh, args)?;
    if selected.domain != Domain::Face {
        return Err(invalid("custom normals require face elements"));
    }
    let normal_values = args
        .get("normals")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("custom normals requires a face-keyed normals object"))?;
    let mut face_offsets = Vec::with_capacity(mesh.faces.len());
    let mut corner_total = 0;
    for face in &mesh.faces {
        face_offsets.push(corner_total);
        corner_total += face.vertices.len();
    }
    // Every selected corner is validated before the setter runs, so bad input changes nothing.
    let mut updates: Vec<(usize, Option<DVec3>)> = Vec::new();
    for id in selected.ids {
        let index = mesh
            .faces
            .iter()
            .position(|face| face.id == id)
            .ok_or_else(|| invalid(format!("face f{id} does not exist")))?;
        let face = &mesh.faces[index];
        let normals = normal_values
            .get(&format!("f{id}"))
            .and_then(Value::as_array)
            .filter(|normals| normals.len() == face.vertices.len())
            .ok_or_else(|| {
                invalid(format!(
                    "custom normals for face f{id} must match its corners"
                ))
            })?;
        for (corner, normal) in normals.iter().enumerate() {
            // `null` resets the corner to automatic: it takes the computed normal again.
            let entry = if normal.is_null() {
                None
            } else {
                let normal = vector(normal, "normals")?;
                if normal.length_squared() <= f64::EPSILON {
                    return Err(invalid("custom normals must be non-zero"));
                }
                Some(normal.normalize())
            };
            updates.push((face_offsets[index] + corner, entry));
        }
    }
    // Only the selected corners are re-encoded; the shared setter keeps every other code it can.
    clnor::encode_vector_updates(
        mesh,
        &updates,
        NormalCodeOptions {
            corner_order: None,
            setter_input: SetterInput::NormalizeInput,
        },
    )?;
    Ok(IdChanges::default())
}

fn apply_set_attribute(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let selected = required_selection(mesh, args)?;
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid("set_attribute requires a non-empty name"))?;
    let value = args
        .get("value")
        .cloned()
        .ok_or_else(|| invalid("set_attribute requires value"))?;
    // Shading names address Potter's native flags; every other name is a user attribute.
    let shading = matches!(name, "seam" | "sharp" | "shade_smooth");
    let storage = if shading {
        name.to_owned()
    } else {
        storage_key(name).into_owned()
    };
    if !shading
        && name != "material_index"
        && mesh
            .attributes
            .get(&storage)
            .and_then(Value::as_object)
            .is_some_and(|attribute| attribute.contains_key("type"))
    {
        return apply_attribute_update(mesh, args);
    }
    if name == "material_index" {
        if selected.domain != Domain::Face {
            return Err(invalid("material_index must be assigned to face elements"));
        }
        let raw = value
            .as_u64()
            .ok_or_else(|| invalid("material_index must be a non-negative integer"))?;
        let material_index =
            u32::try_from(raw).map_err(|_| invalid("material_index is outside the u32 range"))?;
        for face in &mut mesh.faces {
            if selected.ids.contains(&face.id) {
                face.material_index = material_index;
            }
        }
        return Ok(IdChanges::default());
    }
    match name {
        "seam" | "sharp" => {
            if selected.domain != Domain::Edge || !value.is_boolean() {
                return Err(invalid(format!(
                    "{name} requires edge elements and a boolean value"
                )));
            }
        }
        "crease" => {
            if selected.domain != Domain::Edge {
                return Err(invalid("crease requires edge elements"));
            }
            if !value
                .as_f64()
                .is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value))
            {
                return Err(invalid("crease must be a finite number from zero to one"));
            }
        }
        "shade_smooth" if selected.domain != Domain::Face || !value.is_boolean() => {
            return Err(invalid(
                "shade_smooth requires face elements and a boolean value",
            ));
        }
        _ => {}
    }
    let values = selected
        .ids
        .iter()
        .map(|id| (format!("{}{id}", selected.domain.prefix()), value.clone()))
        .collect::<Map<_, _>>();
    // Shading attributes use the singular domain names and explicit types their readers require.
    let (domain_name, kind) = match name {
        "sharp" => ("edge", Some("bool")),
        "shade_smooth" => ("face", Some("bool")),
        _ => (selected.domain.name(), None),
    };
    let attribute = mesh
        .attributes
        .entry(storage)
        .or_insert_with(|| match kind {
            Some(kind) => json!({ "domain": domain_name, "type": kind, "values": {} }),
            None => json!({ "domain": domain_name, "values": {} }),
        });
    let attribute_object = attribute
        .as_object_mut()
        .ok_or_else(|| invalid("existing mesh attribute is not a domain attribute object"))?;
    if attribute_object.get("domain").and_then(Value::as_str) != Some(domain_name) {
        return Err(invalid(
            "mesh attribute already exists in a different domain",
        ));
    }
    let existing_values = attribute_object
        .entry("values")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| invalid("mesh attribute values must be an object"))?;
    existing_values.extend(values);
    Ok(IdChanges::default())
}

fn face_edge_index(face: &Face, key: (u32, u32)) -> Option<usize> {
    (0..face.vertices.len()).find(|index| {
        edge_key(
            face.vertices[*index],
            face.vertices[(*index + 1) % face.vertices.len()],
        ) == key
    })
}

// BMesh face insertion appends each new loop after the radial head, then makes it the head.
fn bmesh_radial_face_order(mesh: &Mesh, key: (u32, u32)) -> Vec<u32> {
    let mut faces = mesh
        .faces
        .iter()
        .filter(|face| face_edge_index(face, key).is_some())
        .map(|face| face.id)
        .collect::<Vec<_>>();
    if faces.len() > 1 {
        faces.rotate_right(1);
    }
    faces
}

// Each split edge gets loops appended in the old radial traversal order; the last append is its new head.
fn bmesh_radial_face_order_after_split(radial_faces: &[u32]) -> Vec<u32> {
    let mut faces = radial_faces.to_vec();
    if faces.len() > 1 {
        faces.rotate_right(1);
    }
    faces
}

// bm_face_slice starts at the first rim loop and continues splitting the larger remainder.
fn loop_cut_remainder(
    first_side: &[u32],
    opposite_side: &[u32],
    removed: usize,
    from_start: bool,
) -> Vec<u32> {
    let intervals = first_side.len() - 1;
    if from_start {
        let mut remainder = first_side[removed..].to_vec();
        remainder.extend_from_slice(&opposite_side[..=intervals - removed]);
        remainder
    } else {
        let mut remainder = first_side[..=intervals - removed].to_vec();
        remainder.extend_from_slice(&opposite_side[removed..]);
        remainder
    }
}

fn apply_loop_cut(
    mesh: &mut Mesh,
    selected: &Selection,
    cuts: u32,
    mut multires_data: Option<&mut MultiresGridData>,
) -> Result<IdChanges> {
    if selected.domain != Domain::Edge {
        return Err(invalid("loop_cut requires edge elements"));
    }
    if !(1..=8).contains(&cuts) {
        return Err(invalid("loop_cut cuts must be between one and eight"));
    }

    let edges_by_id: HashMap<_, _> = mesh
        .edges
        .iter()
        .cloned()
        .map(|edge| (edge.id, edge))
        .collect();
    let mut ring = selected.ids.iter().copied().collect::<HashSet<_>>();
    let mut pending = selected.ids.clone();
    while let Some(edge_id) = pending.pop() {
        let edge = edges_by_id
            .get(&edge_id)
            .ok_or_else(|| invalid("loop_cut selection contains a missing edge"))?;
        let key = edge_key(edge.vertices[0], edge.vertices[1]);
        let incident: Vec<_> = mesh
            .faces
            .iter()
            .filter_map(|face| face_edge_index(face, key).map(|index| (face, index)))
            .collect();
        if incident.is_empty() || incident.len() > 2 {
            return Err(invalid(
                "loop_cut edges must have one or two adjacent quad faces",
            ));
        }
        for (face, index) in incident {
            if face.vertices.len() != 4 {
                return Err(invalid("loop_cut propagates only through quad faces"));
            }
            let opposite_key = edge_key(
                face.vertices[(index + 2) % 4],
                face.vertices[(index + 3) % 4],
            );
            let opposite_id = mesh
                .edges
                .iter()
                .find(|candidate| {
                    edge_key(candidate.vertices[0], candidate.vertices[1]) == opposite_key
                })
                .map(|candidate| candidate.id)
                .ok_or_else(|| invalid("loop_cut could not find an opposite edge"))?;
            if ring.insert(opposite_id) {
                pending.push(opposite_id);
            }
        }
    }
    let mut ordered_ring: Vec<_> = ring.iter().copied().collect();
    ordered_ring.sort_unstable();
    let ring_keys: HashSet<_> = ordered_ring
        .iter()
        .filter_map(|id| edges_by_id.get(id))
        .map(|edge| edge_key(edge.vertices[0], edge.vertices[1]))
        .collect();
    let affected: Vec<_> = mesh
        .faces
        .iter()
        .filter(|face| {
            (0..face.vertices.len()).any(|index| {
                ring_keys.contains(&edge_key(
                    face.vertices[index],
                    face.vertices[(index + 1) % face.vertices.len()],
                ))
            })
        })
        .cloned()
        .collect();
    for face in &affected {
        let cut_indices: Vec<_> = (0..face.vertices.len())
            .filter(|index| {
                ring_keys.contains(&edge_key(
                    face.vertices[*index],
                    face.vertices[(*index + 1) % face.vertices.len()],
                ))
            })
            .collect();
        if face.vertices.len() != 4
            || cut_indices.len() != 2
            || (cut_indices[0] + 2) % 4 != cut_indices[1]
                && (cut_indices[1] + 2) % 4 != cut_indices[0]
        {
            return Err(invalid(
                "loop_cut requires each affected quad to have one opposite cut-edge pair",
            ));
        }
    }

    let mut changes = IdChanges::default();
    let mut staged_mesh = multires_data.as_ref().map(|_| mesh.clone());
    let mut staged_root_lineage = IdChanges::default();
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut edge_points = HashMap::with_capacity(ordered_ring.len());
    let interval_count = cuts + 1;
    for edge_id in &ordered_ring {
        let edge = &edges_by_id[edge_id];
        let (low, high) = if edge.vertices[0] < edge.vertices[1] {
            (edge.vertices[0], edge.vertices[1])
        } else {
            (edge.vertices[1], edge.vertices[0])
        };
        let start = positions
            .get(&low)
            .copied()
            .ok_or_else(|| invalid("loop_cut edge references a missing vertex"))?;
        let end = positions
            .get(&high)
            .copied()
            .ok_or_else(|| invalid("loop_cut edge references a missing vertex"))?;
        let mut points = Vec::with_capacity(
            usize::try_from(cuts).map_err(|_| invalid("too many loop_cut subdivisions"))?,
        );
        for step in 1..=cuts {
            let fraction = f64::from(step) / f64::from(interval_count);
            points.push(insert_vertex(
                mesh,
                start.lerp(end, fraction),
                &mut changes,
            )?);
        }
        if let Some(stage) = staged_mesh.as_mut() {
            let data = multires_data
                .as_deref_mut()
                .ok_or_else(|| invalid("loop-cut stage has no multires data"))?;
            let mut active_endpoint = edge.vertices[1];
            let mut radial_faces =
                bmesh_radial_face_order(stage, edge_key(edge.vertices[0], edge.vertices[1]));
            for point_index in 0..points.len() {
                let point_index = if edge.vertices[1] == low {
                    point_index
                } else {
                    points.len() - point_index - 1
                };
                let point = points[point_index];
                let vertex = mesh
                    .vertex(point)
                    .cloned()
                    .ok_or_else(|| invalid("loop-cut stage vertex is missing"))?;
                stage.vertices.push(vertex);
                radial_faces = stage_subdivide_edge_split(
                    stage,
                    data,
                    &mut staged_root_lineage,
                    [edge.vertices[0], active_endpoint],
                    point,
                    &radial_faces,
                )?;
                active_endpoint = point;
            }
        }
        edge_points.insert((low, high), points);
    }

    let point_sequence = |start: u32, end: u32| -> Result<Vec<u32>> {
        let key = edge_key(start, end);
        let points = edge_points
            .get(&key)
            .ok_or_else(|| invalid("loop_cut face is missing a propagated edge"))?;
        let mut sequence = Vec::with_capacity(points.len() + 2);
        sequence.push(start);
        if start < end {
            sequence.extend(points.iter().copied());
        } else {
            sequence.extend(points.iter().rev().copied());
        }
        sequence.push(end);
        Ok(sequence)
    };

    let mut replacements = Vec::with_capacity(affected.len());
    for face in &affected {
        let cut_indices: Vec<_> = (0..face.vertices.len())
            .filter(|index| {
                ring_keys.contains(&edge_key(
                    face.vertices[*index],
                    face.vertices[(*index + 1) % face.vertices.len()],
                ))
            })
            .collect();
        let base = cut_indices[0];
        let rotated: Vec<_> = (0..4)
            .map(|offset| face.vertices[(base + offset) % 4])
            .collect();
        let first_side = point_sequence(rotated[0], rotated[1])?;
        let opposite_side = point_sequence(rotated[2], rotated[3])?;
        let intervals = usize::try_from(interval_count)
            .map_err(|_| invalid("too many loop_cut subdivisions"))?;
        let mut pieces = Vec::with_capacity(intervals);
        for index in 0..intervals {
            pieces.push(vec![
                first_side[index],
                first_side[index + 1],
                opposite_side[intervals - index - 1],
                opposite_side[intervals - index],
            ]);
        }
        let mut cut_point_sources = HashMap::new();
        for edge_index in 0..face.vertices.len() {
            let start = face.vertices[edge_index];
            let end = face.vertices[(edge_index + 1) % face.vertices.len()];
            if ring_keys.contains(&edge_key(start, end)) {
                for vertex_id in point_sequence(start, end)?
                    .iter()
                    .skip(1)
                    .take(intervals - 1)
                {
                    cut_point_sources.insert(*vertex_id, edge_index);
                }
            }
        }
        replacements.push((
            face.clone(),
            face.material_index,
            pieces,
            cut_point_sources,
            first_side,
            opposite_side,
            base,
        ));
    }

    let mut stage_child_faces = HashMap::<(u32, Vec<u32>), u32>::new();
    let mut actual_to_stage_faces = HashMap::new();
    if let Some(stage) = staged_mesh.as_mut() {
        let data = multires_data
            .as_deref_mut()
            .ok_or_else(|| invalid("loop-cut stage has no multires data"))?;
        for (source_face, _, pieces, _, first_side, opposite_side, base) in &replacements {
            let first_rim = (0..4)
                .find(|index| {
                    !ring_keys.contains(&edge_key(
                        source_face.vertices[*index],
                        source_face.vertices[(*index + 1) % 4],
                    ))
                })
                .ok_or_else(|| invalid("loop-cut face has no rim edge"))?;
            // The first rim edge chooses whether slicing starts at piece zero or the last piece.
            let from_start = first_rim == (base + 3) % 4;
            let mut piece_order = (0..pieces.len()).collect::<Vec<_>>();
            if !from_start {
                piece_order.reverse();
            }
            let mut remainder_id = source_face.id;
            for (step, piece_index) in piece_order.iter().enumerate() {
                let piece = pieces[*piece_index].clone();
                if step + 1 == piece_order.len() {
                    stage_child_faces.insert((source_face.id, piece), remainder_id);
                    break;
                }
                let remainder = loop_cut_remainder(first_side, opposite_side, step + 1, from_start);
                let (next_remainder_id, piece_id) = stage_subdivide_face_split(
                    stage,
                    data,
                    &mut staged_root_lineage,
                    remainder_id,
                    remainder,
                    piece.clone(),
                )?;
                stage_child_faces.insert((source_face.id, piece), piece_id);
                remainder_id = next_remainder_id;
            }
        }
    }

    let ring_ids: HashSet<_> = ordered_ring.into_iter().collect();
    mesh.edges.retain(|edge| {
        if ring_ids.contains(&edge.id) {
            changes.delete(Domain::Edge, edge.id);
            false
        } else {
            true
        }
    });
    for (face, _, _, _, _, _, _) in &replacements {
        remove_face(mesh, face.id, &mut changes);
    }
    for (source_face, material_index, pieces, cut_point_sources, _, _, _) in replacements {
        for piece in pieces {
            let source_corners = piece
                .iter()
                .map(|vertex| {
                    source_face
                        .vertices
                        .iter()
                        .position(|source_vertex| source_vertex == vertex)
                        .or_else(|| cut_point_sources.get(vertex).copied())
                        .ok_or_else(|| invalid("loop_cut face corner has no source loop"))
                })
                .collect::<Result<Vec<_>>>()?;
            let seeds = source_corners
                .iter()
                .map(|corner| {
                    Some(MdispSourceLoop {
                        face_id: source_face.id,
                        corner: *corner,
                    })
                })
                .collect::<Vec<_>>();
            let callback_parent = [source_face.id];
            let (stage_face_id, sources) = if staged_mesh.is_some() {
                let stage_face_id = stage_child_faces
                    .get(&(source_face.id, piece.clone()))
                    .copied()
                    .ok_or_else(|| invalid("loop_cut stage child face is missing"))?;
                let sources = staged_root_lineage
                    .mdisp_sources
                    .get(&stage_face_id)
                    .cloned()
                    .ok_or_else(|| invalid("loop_cut stage child provenance is missing"))?;
                (Some(stage_face_id), sources)
            } else {
                (None, mdisp_face_sources(&changes, &seeds, &callback_parent))
            };
            let face_id =
                insert_face_with_mdisp_sources(mesh, piece, material_index, &mut changes, sources)?;
            if let Some(stage_face_id) = stage_face_id {
                actual_to_stage_faces.insert(face_id, stage_face_id);
            }
        }
    }
    if let (Some(stage), Some(data)) = (staged_mesh.as_ref(), multires_data) {
        reorder_multires_face_data(stage, mesh, data, &actual_to_stage_faces)?;
    }
    Ok(changes)
}

fn trace_selected_edge_loops(mesh: &Mesh, selected: &Selection) -> Result<[Vec<u32>; 2]> {
    if selected.domain != Domain::Edge {
        return Err(invalid("bridge requires edge elements"));
    }
    let selected_ids: HashSet<_> = selected.ids.iter().copied().collect();
    let mut neighbors: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut edge_count = 0;
    for edge in &mesh.edges {
        if selected_ids.contains(&edge.id) {
            neighbors
                .entry(edge.vertices[0])
                .or_default()
                .push(edge.vertices[1]);
            neighbors
                .entry(edge.vertices[1])
                .or_default()
                .push(edge.vertices[0]);
            edge_count += 1;
            let face_uses = mesh
                .faces
                .iter()
                .filter(|face| {
                    face_edge_index(face, edge_key(edge.vertices[0], edge.vertices[1])).is_some()
                })
                .count();
            if face_uses > 1 {
                return Err(invalid("bridge edges must be boundary or loose edges"));
            }
        }
    }
    if edge_count != selected.ids.len() {
        return Err(invalid("bridge selection contains a missing edge"));
    }
    for adjacent in neighbors.values_mut() {
        adjacent.sort_unstable();
        if adjacent.len() != 2 {
            return Err(invalid("bridge selections must contain closed edge loops"));
        }
    }
    let mut unvisited: HashSet<_> = neighbors.keys().copied().collect();
    let mut loops = Vec::new();
    while let Some(start) = unvisited.iter().min().copied() {
        let mut component = HashSet::new();
        let mut pending = vec![start];
        while let Some(vertex) = pending.pop() {
            if component.insert(vertex) {
                pending.extend(neighbors[&vertex].iter().copied());
            }
        }
        unvisited.retain(|vertex| !component.contains(vertex));
        if component.len() < 3 {
            return Err(invalid("bridge loops must have at least three vertices"));
        }
        let mut ordered = vec![start];
        let mut previous = start;
        let mut current = neighbors[&start][0];
        while current != start {
            if ordered.len() >= component.len() {
                return Err(invalid("bridge edge selection is not a simple loop"));
            }
            ordered.push(current);
            let adjacent = &neighbors[&current];
            let next = if adjacent[0] == previous {
                adjacent[1]
            } else if adjacent[1] == previous {
                adjacent[0]
            } else {
                return Err(invalid("bridge edges do not form a simple loop"));
            };
            previous = current;
            current = next;
        }
        if ordered.len() != component.len() {
            return Err(invalid(
                "bridge edge selection contains disconnected cycles",
            ));
        }
        loops.push(ordered);
    }
    if loops.len() != 2 {
        return Err(invalid("bridge requires exactly two edge loops"));
    }
    let second = loops
        .pop()
        .ok_or_else(|| invalid("bridge loop is missing"))?;
    let first = loops
        .pop()
        .ok_or_else(|| invalid("bridge loop is missing"))?;
    if first.len() != second.len() {
        return Err(invalid(
            "bridge loops must have the same number of vertices",
        ));
    }
    Ok([first, second])
}

fn orient_bridge_loop(
    mesh: &Mesh,
    mut vertices: Vec<u32>,
    same_direction_as_face: bool,
) -> Result<(Vec<u32>, bool)> {
    let mut direction_matches_face = None;
    for index in 0..vertices.len() {
        let first = vertices[index];
        let second = vertices[(index + 1) % vertices.len()];
        let key = edge_key(first, second);
        if let Some((face, face_index)) = mesh
            .faces
            .iter()
            .find_map(|face| face_edge_index(face, key).map(|index| (face, index)))
        {
            let matches = face.vertices[face_index] == first;
            if direction_matches_face.is_some_and(|previous| previous != matches) {
                return Err(invalid("bridge loop has inconsistent face winding"));
            }
            direction_matches_face = Some(matches);
        }
    }
    let Some(matches) = direction_matches_face else {
        return Ok((vertices, false));
    };
    if matches != same_direction_as_face {
        let mut reversed = Vec::with_capacity(vertices.len());
        reversed.push(vertices[0]);
        reversed.extend(vertices[1..].iter().rev().copied());
        vertices = reversed;
    }
    Ok((vertices, true))
}

fn apply_bridge(mesh: &mut Mesh, selected: &Selection) -> Result<IdChanges> {
    let [first, second] = trace_selected_edge_loops(mesh, selected)?;
    let (first, _) = orient_bridge_loop(mesh, first, false)?;
    let (second, second_has_boundary) = orient_bridge_loop(mesh, second, true)?;
    let count = first.len();
    let mut aligned = second.clone();
    let mut best_distance = f64::INFINITY;
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    for reverse in [false, true] {
        if reverse && second_has_boundary {
            continue;
        }
        for shift in 0..count {
            let candidate: Vec<_> = (0..count)
                .map(|index| {
                    let offset = if reverse {
                        (shift + count - index) % count
                    } else {
                        (shift + index) % count
                    };
                    second[offset]
                })
                .collect();
            let distance = first
                .iter()
                .zip(&candidate)
                .map(|(first_id, second_id)| {
                    positions
                        .get(first_id)
                        .zip(positions.get(second_id))
                        .map_or(f64::INFINITY, |(a, b)| a.distance_squared(*b))
                })
                .sum::<f64>();
            if distance < best_distance {
                best_distance = distance;
                aligned = candidate;
            }
        }
    }

    let mut changes = IdChanges::default();
    let selected_ids: HashSet<_> = selected.ids.iter().copied().collect();
    // Blender bridge copies boundary loop blocks instead of interpolating MDisps.
    let mut boundary_loop_sources = HashMap::new();
    for edge in mesh
        .edges
        .iter()
        .filter(|edge| selected_ids.contains(&edge.id))
    {
        let key = edge_key(edge.vertices[0], edge.vertices[1]);
        for face in &mesh.faces {
            let Some(corner) = face_edge_index(face, key) else {
                continue;
            };
            let next = (corner + 1) % face.vertices.len();
            boundary_loop_sources.insert(
                (key, face.vertices[corner]),
                MdispSourceLoop {
                    face_id: face.id,
                    corner,
                },
            );
            boundary_loop_sources.insert(
                (key, face.vertices[next]),
                MdispSourceLoop {
                    face_id: face.id,
                    corner: next,
                },
            );
        }
    }
    for index in 0..count {
        let next = (index + 1) % count;
        let vertices = vec![first[index], first[next], aligned[next], aligned[index]];
        let first_key = edge_key(first[index], first[next]);
        let second_key = edge_key(aligned[index], aligned[next]);
        let seeds = [
            boundary_loop_sources
                .get(&(first_key, first[index]))
                .copied(),
            boundary_loop_sources
                .get(&(first_key, first[next]))
                .copied(),
            boundary_loop_sources
                .get(&(second_key, aligned[next]))
                .copied(),
            boundary_loop_sources
                .get(&(second_key, aligned[index]))
                .copied(),
        ];
        let sources = mdisp_face_sources(&changes, &seeds, &[]);
        insert_face_with_mdisp_sources(mesh, vertices, 0, &mut changes, sources)?;
    }
    Ok(changes)
}

fn apply_split(mesh: &mut Mesh, selected: &Selection) -> Result<IdChanges> {
    if selected.domain != Domain::Face {
        return Err(invalid("split requires face elements"));
    }
    let selected_ids: HashSet<_> = selected.ids.iter().copied().collect();
    let selected_faces: Vec<_> = mesh
        .faces
        .iter()
        .filter(|face| selected_ids.contains(&face.id))
        .cloned()
        .collect();
    let mut selected_vertices = HashSet::new();
    let mut shared_vertices = HashSet::new();
    for face in &mesh.faces {
        if selected_ids.contains(&face.id) {
            selected_vertices.extend(face.vertices.iter().copied());
        } else {
            shared_vertices.extend(face.vertices.iter().copied());
        }
    }
    selected_vertices.retain(|id| shared_vertices.contains(id));
    let duplicated: Vec<_> = mesh
        .vertices
        .iter()
        .filter(|vertex| selected_vertices.contains(&vertex.id))
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut changes = IdChanges::default();
    let mut vertex_remap = HashMap::with_capacity(duplicated.len());
    for (id, position) in duplicated {
        vertex_remap.insert(id, insert_vertex(mesh, position, &mut changes)?);
    }

    let selected_edge_keys: HashSet<_> = selected_faces
        .iter()
        .flat_map(|face| {
            (0..face.vertices.len()).map(|index| {
                edge_key(
                    face.vertices[index],
                    face.vertices[(index + 1) % face.vertices.len()],
                )
            })
        })
        .collect();
    for face in &mut mesh.faces {
        if let Some(source_face) = selected_faces.iter().find(|source| source.id == face.id) {
            for id in &mut face.vertices {
                if let Some(replacement) = vertex_remap.get(id) {
                    *id = *replacement;
                }
            }
            let source_corners = (0..source_face.vertices.len())
                .map(Some)
                .collect::<Vec<_>>();
            let sources = mdisp_sources_from_parent(&changes, source_face, &source_corners, false);
            changes.mdisp_sources.insert(face.id, sources);
        }
    }
    let used_edge_keys: HashSet<_> = mesh
        .faces
        .iter()
        .flat_map(|face| {
            (0..face.vertices.len()).map(|index| {
                edge_key(
                    face.vertices[index],
                    face.vertices[(index + 1) % face.vertices.len()],
                )
            })
        })
        .collect();
    mesh.edges.retain(|edge| {
        let key = edge_key(edge.vertices[0], edge.vertices[1]);
        if selected_edge_keys.contains(&key) && !used_edge_keys.contains(&key) {
            changes.delete(Domain::Edge, edge.id);
            false
        } else {
            true
        }
    });
    let mut present_edge_keys: HashSet<_> = mesh
        .edges
        .iter()
        .map(|edge| edge_key(edge.vertices[0], edge.vertices[1]))
        .collect();
    let mut missing_edges = Vec::new();
    for face in &mesh.faces {
        for index in 0..face.vertices.len() {
            let pair = [
                face.vertices[index],
                face.vertices[(index + 1) % face.vertices.len()],
            ];
            if present_edge_keys.insert(edge_key(pair[0], pair[1])) {
                missing_edges.push(pair);
            }
        }
    }
    for vertices in missing_edges {
        let id = mesh
            .insert_edge(vertices)
            .map_err(|error| mesh_error(&error))?;
        changes.record(Domain::Edge, id);
    }
    Ok(changes)
}

fn crossing_edge_ids(
    mesh: &Mesh,
    point: DVec3,
    normal: DVec3,
    threshold: f64,
    selected_face_ids: Option<&HashSet<u32>>,
) -> Result<HashSet<u32>> {
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let selected_edge_keys: Option<HashSet<_>> = selected_face_ids.map(|selected| {
        mesh.faces
            .iter()
            .filter(|face| selected.contains(&face.id))
            .flat_map(|face| {
                (0..face.vertices.len()).map(|index| {
                    edge_key(
                        face.vertices[index],
                        face.vertices[(index + 1) % face.vertices.len()],
                    )
                })
            })
            .collect()
    });
    let mut crossing = HashSet::new();
    for edge in &mesh.edges {
        let key = edge_key(edge.vertices[0], edge.vertices[1]);
        if selected_edge_keys
            .as_ref()
            .is_some_and(|keys| !keys.contains(&key))
        {
            continue;
        }
        let first = positions
            .get(&edge.vertices[0])
            .copied()
            .ok_or_else(|| invalid("edge references a missing vertex"))?;
        let second = positions
            .get(&edge.vertices[1])
            .copied()
            .ok_or_else(|| invalid("edge references a missing vertex"))?;
        let first_distance = (first - point).dot(normal);
        let second_distance = (second - point).dot(normal);
        if first_distance > threshold && second_distance < -threshold
            || first_distance < -threshold && second_distance > threshold
        {
            crossing.insert(edge.id);
        }
    }
    Ok(crossing)
}

fn remove_unused_edges(mesh: &mut Mesh, candidates: &HashSet<u32>, changes: &mut IdChanges) {
    let used_edge_keys: HashSet<_> = mesh
        .faces
        .iter()
        .flat_map(|face| {
            (0..face.vertices.len()).map(|index| {
                edge_key(
                    face.vertices[index],
                    face.vertices[(index + 1) % face.vertices.len()],
                )
            })
        })
        .collect();
    mesh.edges.retain(|edge| {
        let key = edge_key(edge.vertices[0], edge.vertices[1]);
        if candidates.contains(&edge.id) && !used_edge_keys.contains(&key) {
            changes.delete(Domain::Edge, edge.id);
            false
        } else {
            true
        }
    });
}

fn apply_knife(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let selected = required_selection(mesh, args)?;
    if selected.domain != Domain::Face {
        return Err(invalid("knife requires face elements"));
    }
    let (point, normal) = if let Some(plane_value) = args.get("plane") {
        if args.contains_key("point") || args.contains_key("normal") {
            return Err(invalid(
                "knife accepts either plane or top-level point and normal, not both",
            ));
        }
        let plane = object(plane_value, "plane")?;
        reject_unknown_fields(plane, &["point", "normal"], "plane")?;
        (
            plane
                .get("point")
                .ok_or_else(|| invalid("knife plane requires point"))?,
            plane
                .get("normal")
                .ok_or_else(|| invalid("knife plane requires normal"))?,
        )
    } else {
        (
            args.get("point")
                .ok_or_else(|| invalid("knife requires a plane or point and normal"))?,
            args.get("normal")
                .ok_or_else(|| invalid("knife requires a plane or point and normal"))?,
        )
    };
    let point = vector(point, "knife point")?;
    let normal = vector(normal, "knife normal")?;
    let normal_length_squared = normal.length_squared();
    if !normal_length_squared.is_finite() || normal_length_squared <= f64::EPSILON {
        return Err(invalid("knife plane normal must be non-zero"));
    }
    let normal = normal.normalize();
    let threshold = number(args, "threshold", 1.0e-10)?;
    if threshold < 0.0 {
        return Err(invalid("knife threshold must not be negative"));
    }
    let selected_face_ids: HashSet<_> = selected.ids.iter().copied().collect();
    let crossing = crossing_edge_ids(mesh, point, normal, threshold, Some(&selected_face_ids))?;
    let mut bisect_args = Map::new();
    bisect_args.insert(
        "elements".to_owned(),
        args.get("elements")
            .cloned()
            .ok_or_else(|| invalid("elements is required"))?,
    );
    bisect_args.insert("point".to_owned(), json!(point.to_array()));
    bisect_args.insert("normal".to_owned(), json!(normal.to_array()));
    bisect_args.insert("threshold".to_owned(), json!(threshold));
    let mut changes = apply_bisect(mesh, &bisect_args, None, false)?;
    remove_unused_edges(mesh, &crossing, &mut changes);
    Ok(changes)
}

fn apply_symmetrize(
    mesh: &mut Mesh,
    args: &Map<String, Value>,
    mut multires_data: Option<&mut MultiresGridData>,
) -> Result<IdChanges> {
    if !args.contains_key("axis") {
        return Err(invalid("symmetrize requires an axis"));
    }
    let axis = axis_index(args)?;
    let origin = optional_vector(args, "origin", DVec3::ZERO)?;
    let threshold = number(args, "threshold", 1.0e-6)?;
    if threshold < 0.0 {
        return Err(invalid("symmetrize threshold must not be negative"));
    }
    let positive_source = match args.get("direction").and_then(Value::as_str) {
        Some("positive_to_negative") => true,
        Some("negative_to_positive") => false,
        _ => {
            return Err(invalid(
                "symmetrize direction must be positive_to_negative or negative_to_positive",
            ));
        }
    };
    let selection = selection(mesh, args, false)?;
    if selection
        .as_ref()
        .is_some_and(|selection| selection.domain != Domain::Face)
    {
        return Err(invalid("symmetrize elements must be faces"));
    }
    let selected_ids: Option<HashSet<_>> =
        selection.map(|selection| selection.ids.into_iter().collect());
    let mut changes = IdChanges::default();
    if selected_ids.is_none() {
        let mut plane_normal = DVec3::ZERO;
        plane_normal[axis] = 1.0;
        let face_ids: HashSet<_> = mesh.faces.iter().map(|face| face.id).collect();
        let crossing = crossing_edge_ids(mesh, origin, plane_normal, threshold, Some(&face_ids))?;
        let all_crossing = crossing_edge_ids(mesh, origin, plane_normal, threshold, None)?;
        if all_crossing
            .iter()
            .any(|edge_id| !crossing.contains(edge_id))
        {
            return Err(invalid(
                "symmetrize cannot process loose edges crossing the axis plane",
            ));
        }
        let plane_args = json!({
            "point": origin.to_array(),
            "normal": plane_normal.to_array(),
            "threshold": threshold
        });
        let plane_args = object(&plane_args, "symmetrize plane")?;
        let mut plane_changes = apply_bisect(mesh, plane_args, multires_data.as_deref_mut(), true)?;
        remove_unused_edges(mesh, &crossing, &mut plane_changes);
        merge_changes(&mut changes, plane_changes);
    }
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let distance = |id: u32| -> Result<f64> {
        positions
            .get(&id)
            .map(|position| position[axis] - origin[axis])
            .ok_or_else(|| invalid("symmetrize face references a missing vertex"))
    };
    let mut source_edge_pairs = Vec::new();
    let mut source_side_vertex_ids = HashSet::new();
    let mut destination_edge_keys = HashSet::new();
    let mut destination_vertex_ids = HashSet::new();
    if selected_ids.is_none() {
        for vertex in &mesh.vertices {
            let offset = vertex.co[axis] - origin[axis];
            let side = if positive_source { offset } else { -offset };
            if side > threshold {
                source_side_vertex_ids.insert(vertex.id);
            } else if side < -threshold {
                destination_vertex_ids.insert(vertex.id);
            }
        }
        for edge in &mesh.edges {
            let first = positions
                .get(&edge.vertices[0])
                .copied()
                .ok_or_else(|| invalid("symmetrize edge references a missing vertex"))?;
            let second = positions
                .get(&edge.vertices[1])
                .copied()
                .ok_or_else(|| invalid("symmetrize edge references a missing vertex"))?;
            let first_offset = first[axis] - origin[axis];
            let second_offset = second[axis] - origin[axis];
            let first_side = if positive_source {
                first_offset
            } else {
                -first_offset
            };
            let second_side = if positive_source {
                second_offset
            } else {
                -second_offset
            };
            let source_edge = first_side >= -threshold
                && second_side >= -threshold
                && (first_side > threshold || second_side > threshold);
            let destination_edge = first_side <= threshold
                && second_side <= threshold
                && (first_side < -threshold || second_side < -threshold);
            if source_edge {
                source_edge_pairs.push(edge.vertices);
                source_side_vertex_ids.extend(edge.vertices);
            } else if destination_edge {
                destination_edge_keys.insert(edge_key(edge.vertices[0], edge.vertices[1]));
            } else if first_side > threshold && second_side < -threshold
                || first_side < -threshold && second_side > threshold
            {
                return Err(invalid(
                    "symmetrize cannot process loose edges crossing the axis plane",
                ));
            }
        }
    }
    let mut source_faces = Vec::new();
    let mut destination_faces = Vec::new();
    for face in &mesh.faces {
        let distances = face
            .vertices
            .iter()
            .map(|id| distance(*id))
            .collect::<Result<Vec<_>>>()?;
        let is_source = if positive_source {
            distances.iter().all(|value| *value >= -threshold)
        } else {
            distances.iter().all(|value| *value <= threshold)
        };
        let is_destination = if positive_source {
            distances.iter().all(|value| *value <= threshold)
        } else {
            distances.iter().all(|value| *value >= -threshold)
        };
        let has_off_plane = distances.iter().any(|value| value.abs() > threshold);
        if is_source && has_off_plane {
            if selected_ids
                .as_ref()
                .is_none_or(|ids| ids.contains(&face.id))
            {
                source_faces.push(face.clone());
            }
        } else if is_destination && has_off_plane {
            destination_faces.push(face.clone());
        } else if !distances.iter().all(|value| value.abs() <= threshold) {
            return Err(invalid(
                "symmetrize cannot process faces crossing the axis plane",
            ));
        }
    }
    if selected_ids
        .as_ref()
        .is_some_and(|ids| ids.len() != source_faces.len())
    {
        return Err(invalid(
            "symmetrize selection must contain source-side faces",
        ));
    }
    let mut source_vertex_set: HashSet<_> = source_faces
        .iter()
        .flat_map(|face| face.vertices.iter().copied())
        .collect();
    if selected_ids.is_none() {
        source_vertex_set.extend(source_side_vertex_ids);
        for edge in &source_edge_pairs {
            source_vertex_set.extend(edge.iter().copied());
        }
    }
    if source_vertex_set.is_empty() {
        return Err(invalid("symmetrize source side contains no geometry"));
    }
    let mut selected_source_vertices: Vec<_> = source_vertex_set.into_iter().collect();
    selected_source_vertices.sort_unstable();
    let reflected_face_positions: Vec<Vec<DVec3>> = source_faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| {
                    positions
                        .get(id)
                        .copied()
                        .map(|mut position| {
                            position[axis] = 2.0 * origin[axis] - position[axis];
                            position
                        })
                        .ok_or_else(|| invalid("symmetrize face references a missing vertex"))
                })
                .collect()
        })
        .collect::<Result<_>>()?;
    let selected_seam_vertices: HashSet<_> = selected_source_vertices
        .iter()
        .filter(|id| distance(**id).is_ok_and(|value| value.abs() <= threshold))
        .copied()
        .collect();
    let mut destination_ids = Vec::new();
    for face in &destination_faces {
        let geometrically_matched = reflected_face_positions.iter().any(|reflected_positions| {
            if face.vertices.len() != reflected_positions.len() {
                return false;
            }
            let mut matched = HashSet::with_capacity(reflected_positions.len());
            reflected_positions.iter().all(|position| {
                face.vertices.iter().any(|id| {
                    !matched.contains(id)
                        && positions.get(id).is_some_and(|vertex| {
                            if vertex.distance(*position) <= threshold {
                                matched.insert(*id);
                                true
                            } else {
                                false
                            }
                        })
                })
            })
        });
        if selected_ids.is_none()
            || face
                .vertices
                .iter()
                .any(|id| selected_seam_vertices.contains(id))
            || geometrically_matched
        {
            destination_ids.push(face.id);
            destination_vertex_ids.extend(face.vertices.iter().copied());
            for index in 0..face.vertices.len() {
                destination_edge_keys.insert(edge_key(
                    face.vertices[index],
                    face.vertices[(index + 1) % face.vertices.len()],
                ));
            }
        }
    }
    let multires_source = multires_data.as_ref().map(|_| mesh.clone());
    let mut multires_sources = HashMap::new();
    remove_faces(mesh, &destination_ids, &mut changes);

    let mut reflected_ids = HashMap::with_capacity(selected_source_vertices.len());
    for source_id in selected_source_vertices {
        let source_position = positions
            .get(&source_id)
            .copied()
            .ok_or_else(|| invalid("symmetrize face references a missing vertex"))?;
        let destination_id = if (source_position[axis] - origin[axis]).abs() <= threshold {
            if let Some(vertex) = mesh
                .vertices
                .iter_mut()
                .find(|vertex| vertex.id == source_id)
            {
                vertex.co[axis] = origin[axis];
            }
            source_id
        } else {
            let mut reflected_position = source_position;
            reflected_position[axis] = 2.0 * origin[axis] - reflected_position[axis];
            let existing = mesh
                .vertices
                .iter()
                .find(|vertex| vertex.co.distance(reflected_position) <= threshold)
                .map(|vertex| vertex.id);
            if let Some(id) = existing {
                id
            } else {
                insert_vertex(mesh, reflected_position, &mut changes)?
            }
        };
        reflected_ids.insert(source_id, destination_id);
    }
    let reflected_vertex_ids: HashSet<_> = reflected_ids.values().copied().collect();
    let reflected_faces = source_faces
        .iter()
        .map(|face| {
            let mut vertices: Vec<_> = face.vertices.iter().map(|id| reflected_ids[id]).collect();
            vertices.reverse();
            (vertices, face.clone())
        })
        .collect::<Vec<_>>();
    for (vertices, source_face) in reflected_faces {
        let source_corners = (0..source_face.vertices.len())
            .rev()
            .map(Some)
            .collect::<Vec<_>>();
        let local_sources = if multires_source.is_some() {
            Some(
                source_corners
                    .iter()
                    .map(|corner| {
                        let corner = (*corner).ok_or_else(|| {
                            invalid("symmetrize copied loop has no source corner")
                        })?;
                        Ok(MdispLoopSource {
                            seed: Some(MdispSourceLoop {
                                face_id: source_face.id,
                                corner,
                            }),
                            callback_faces: Vec::new(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        let mirrored_face_id = insert_face_from_parent(
            mesh,
            vertices,
            source_face.material_index,
            &mut changes,
            &source_face,
            &source_corners,
            false,
        )?;
        if let Some(sources) = local_sources {
            multires_sources.insert(mirrored_face_id, sources);
        }
    }
    let mut mirrored_edge_keys = HashSet::with_capacity(source_edge_pairs.len());
    let mut existing_edge_keys: HashSet<_> = mesh
        .edges
        .iter()
        .map(|edge| edge_key(edge.vertices[0], edge.vertices[1]))
        .collect();
    for edge in &source_edge_pairs {
        let vertices = [reflected_ids[&edge[0]], reflected_ids[&edge[1]]];
        if vertices[0] == vertices[1] {
            continue;
        }
        let key = edge_key(vertices[0], vertices[1]);
        mirrored_edge_keys.insert(key);
        if existing_edge_keys.insert(key) {
            let id = mesh
                .insert_edge(vertices)
                .map_err(|error| mesh_error(&error))?;
            changes.record(Domain::Edge, id);
        }
    }
    let mut used_edge_keys: HashSet<_> = mesh
        .faces
        .iter()
        .flat_map(|face| {
            (0..face.vertices.len()).map(|index| {
                edge_key(
                    face.vertices[index],
                    face.vertices[(index + 1) % face.vertices.len()],
                )
            })
        })
        .collect();
    used_edge_keys.extend(mirrored_edge_keys);
    mesh.edges.retain(|edge| {
        let key = edge_key(edge.vertices[0], edge.vertices[1]);
        if destination_edge_keys.contains(&key) && !used_edge_keys.contains(&key) {
            changes.delete(Domain::Edge, edge.id);
            false
        } else {
            true
        }
    });
    let referenced_vertices: HashSet<_> = mesh
        .faces
        .iter()
        .flat_map(|face| face.vertices.iter().copied())
        .chain(mesh.edges.iter().flat_map(|edge| edge.vertices))
        .collect();
    mesh.vertices.retain(|vertex| {
        if destination_vertex_ids.contains(&vertex.id)
            && !referenced_vertices.contains(&vertex.id)
            && !reflected_vertex_ids.contains(&vertex.id)
        {
            changes.delete(Domain::Vertex, vertex.id);
            false
        } else {
            true
        }
    });
    if let (Some(source), Some(data)) = (multires_source.as_ref(), multires_data) {
        rebase_subdivide_stage(source, mesh, data, &multires_sources)?;
    }
    discard_transient_ids(&mut changes.vertices, &mut changes.deleted_vertices);
    discard_transient_ids(&mut changes.edges, &mut changes.deleted_edges);
    discard_transient_ids(&mut changes.faces, &mut changes.deleted_faces);
    Ok(changes)
}

fn apply_remesh(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let voxel_size = number(args, "voxel_size", 0.0)?;
    if voxel_size <= 0.0 {
        return Err(invalid("remesh voxel_size must be positive"));
    }
    let remeshed = crate::geom::remesh::remesh(mesh, voxel_size)?;
    let mut replacement = Mesh {
        next_id: mesh.next_id,
        ..Mesh::default()
    };
    let mut changes = IdChanges::default();
    let mut vertex_remap = HashMap::with_capacity(remeshed.vertices.len());
    for vertex in &remeshed.vertices {
        let id = insert_vertex(&mut replacement, vertex.co, &mut changes)?;
        vertex_remap.insert(vertex.id, id);
    }
    for face in &remeshed.faces {
        let vertices = face
            .vertices
            .iter()
            .map(|id| {
                vertex_remap
                    .get(id)
                    .copied()
                    .ok_or_else(|| invalid("remesh output face references a missing vertex"))
            })
            .collect::<Result<Vec<_>>>()?;
        insert_face(
            &mut replacement,
            vertices,
            face.material_index,
            &mut changes,
        )?;
    }
    for vertex in &mesh.vertices {
        changes.delete(Domain::Vertex, vertex.id);
    }
    for edge in &mesh.edges {
        changes.delete(Domain::Edge, edge.id);
    }
    for face in &mesh.faces {
        changes.delete(Domain::Face, face.id);
    }
    // Voxel remesh replaces the mesh and drops loop-domain multires data.
    *mesh = replacement;
    Ok(changes)
}

/// Apply a topology or element edit to a persistent-ID mesh.
///
/// Element selections use persistent IDs or typed selectors in `elements`.
/// Created and removed IDs are returned in separate vertex, edge, and face arrays.
/// Subdivision and loop-cut `cuts` values must be between one and eight.
/// Knife accepts a nested plane or top-level `point` and `normal` and keeps both halves.
/// Symmetrize requires an explicit axis and source direction.
pub fn apply(mesh: &mut Mesh, operation: &str, args: &Value) -> Result<Value> {
    let fields = object(args, "args")?;
    let operation = operation.strip_prefix("mesh.").unwrap_or(operation);
    if (is_topology_operation(operation, fields) || operation == "flip_normals")
        && mesh.attributes.contains_key("multires_displacement")
    {
        let source = &*mesh;
        let mut staged = source.clone();
        let result = apply_with_source(&mut staged, source, operation, args)?;
        *mesh = staged;
        Ok(result)
    } else {
        apply_inner(mesh, operation, args, None)
    }
}

pub(crate) fn apply_with_source(
    mesh: &mut Mesh,
    source: &Mesh,
    operation: &str,
    args: &Value,
) -> Result<Value> {
    apply_inner(mesh, operation, args, Some(source))
}

fn apply_inner(
    mesh: &mut Mesh,
    operation: &str,
    args: &Value,
    source_mesh: Option<&Mesh>,
) -> Result<Value> {
    mesh.validate().map_err(|error| mesh_error(&error))?;
    let args = object(args, "args")?;
    let operation = operation.strip_prefix("mesh.").unwrap_or(operation);
    let allowed_fields: &[&str] = match operation {
        "transform_elements" => &[
            "elements",
            "translation",
            "rotation",
            "scale",
            "pivot",
            "proportional",
        ],
        "extrude" => &["elements", "offset", "distance", "individual"],
        "inset" => &["elements", "amount"],
        "edge_slide" | "vertex_slide" => &["elements", "factor"],
        "spin" => &["elements", "axis", "angle", "center", "steps"],
        "screw" => &["elements", "axis", "angle", "center", "distance", "steps"],
        "merge" => &["elements", "mode", "cursor"],
        "rip" | "shade_smooth" | "shade_flat" | "triangulate" | "poke" | "delete" | "dissolve"
        | "fill" | "flip_normals" | "bridge" | "split" => &["elements"],
        "auto_smooth" | "mark_sharp_by_angle" => &["elements", "angle"],
        "set_custom_normals" => &["elements", "normals"],
        "bevel" => &["elements", "width", "amount"],
        "subdivide" | "loop_cut" => &["elements", "cuts"],
        "weld" => &["elements", "threshold"],
        "bisect" => &["elements", "point", "normal", "threshold", "clear_side"],
        "knife" => &["elements", "plane", "point", "normal", "threshold"],
        "symmetrize" => &["elements", "axis", "direction", "origin", "threshold"],
        "remesh" => &["voxel_size"],
        "mirror" => &["elements", "axis", "origin", "merge", "threshold"],
        "set_attribute" => &["elements", "name", "value"],
        "attribute_create" => &["name", "domain", "type", "default"],
        "attribute_update" => &["name", "elements", "value"],
        "attribute_delete" => &["name", "elements"],
        other => {
            return Err(PotError::invalid_operation(format!(
                "unsupported mesh edit operation {other:?}"
            )));
        }
    };
    reject_unknown_fields(args, allowed_fields, "operation")?;
    let topology = is_topology_operation(operation, args);
    let multires_edit = topology || operation == "flip_normals";
    // Blender's delete (bmesh_delete.cc) removes elements and keeps the survivors' own values, so
    // it does not interpolate attributes; only the other topology edits do.
    let attribute_source_needed =
        topology && operation != "delete" && has_interpolatable_attributes(mesh);
    // Deleting faces also invalidates custom normals, so they are rebuilt as for topology edits.
    let normal_source_needed = topology && mesh.attributes.contains_key("custom_normal");
    let multires_source_needed = multires_edit
        && operation != "remesh"
        && mesh.attributes.contains_key("multires_displacement");
    let mut owned_source = None;
    if source_mesh.is_none()
        && (attribute_source_needed || normal_source_needed || multires_source_needed)
    {
        owned_source = Some(mesh.clone());
    }
    let source = source_mesh.or(owned_source.as_ref());
    let attribute_source = attribute_source_needed.then_some(source).flatten();
    let normal_source = normal_source_needed.then_some(source).flatten();
    let mut multires_data = if multires_source_needed {
        multires::read_grid_data(
            source.ok_or_else(|| invalid("multires topology edit has no source mesh"))?,
        )?
    } else {
        None
    };
    let changes = match operation {
        "transform_elements" => apply_transform(mesh, args)?,
        "extrude" => apply_extrude(mesh, args)?,
        "inset" => {
            let selected = required_selection(mesh, args)?;
            apply_inset(mesh, &selected, number(args, "amount", 0.1)?)?
        }
        "bevel" => {
            let selected = required_selection(mesh, args)?;
            let amount = if args.contains_key("width") {
                number(args, "width", 0.1)?
            } else {
                number(args, "amount", 0.1)?
            };
            match selected.domain {
                Domain::Face => apply_inset(mesh, &selected, amount)?,
                Domain::Edge => apply_bevel_edges(mesh, &selected.ids, amount)?,
                Domain::Vertex => return Err(invalid("bevel requires edge or face elements")),
            }
        }
        "subdivide" => {
            let selected = required_selection(mesh, args)?;
            apply_subdivide(
                mesh,
                &selected,
                positive_count(args, "cuts", 1)?,
                multires_data.as_mut(),
            )?
        }
        "triangulate" => apply_triangulate(mesh, &required_selection(mesh, args)?)?,
        "poke" => apply_poke(mesh, &required_selection(mesh, args)?)?,
        "edge_slide" => apply_edge_slide(mesh, args)?,
        "vertex_slide" => apply_vertex_slide(mesh, args)?,
        "spin" => apply_spin_or_screw(mesh, args, false)?,
        "screw" => apply_spin_or_screw(mesh, args, true)?,
        "merge" => apply_merge(mesh, args)?,
        "rip" | "split" => apply_split(mesh, &required_selection(mesh, args)?)?,
        "delete" => apply_delete(mesh, &required_selection(mesh, args)?),
        "dissolve" => apply_dissolve(mesh, &required_selection(mesh, args)?)?,
        "weld" => apply_weld(
            mesh,
            &required_selection(mesh, args)?,
            number(args, "threshold", 1.0e-6)?,
        )?,
        "fill" => apply_fill(mesh, &required_selection(mesh, args)?)?,
        "bisect" => apply_bisect(mesh, args, multires_data.as_mut(), false)?,
        "loop_cut" => apply_loop_cut(
            mesh,
            &required_selection(mesh, args)?,
            positive_count(args, "cuts", 1)?,
            multires_data.as_mut(),
        )?,
        "bridge" => apply_bridge(mesh, &required_selection(mesh, args)?)?,
        "knife" => apply_knife(mesh, args)?,
        "symmetrize" => apply_symmetrize(mesh, args, multires_data.as_mut())?,
        "remesh" => apply_remesh(mesh, args)?,
        "mirror" => apply_mirror(mesh, args)?,
        "flip_normals" => apply_flip_normals(mesh, &required_selection(mesh, args)?)?,
        "shade_smooth" => apply_shade(mesh, &required_selection(mesh, args)?, true)?,
        "shade_flat" => apply_shade(mesh, &required_selection(mesh, args)?, false)?,
        "auto_smooth" => apply_auto_smooth(mesh, args)?,
        "mark_sharp_by_angle" => apply_sharp_angle(mesh, args)?,
        "set_custom_normals" => apply_custom_normals(mesh, args)?,
        "set_attribute" => apply_set_attribute(mesh, args)?,
        "attribute_create" => apply_attribute_create(mesh, args)?,
        "attribute_update" => apply_attribute_update(mesh, args)?,
        "attribute_delete" => apply_attribute_delete(mesh, args)?,
        other => {
            return Err(PotError::invalid_operation(format!(
                "unsupported mesh edit operation {other:?}"
            )));
        }
    };
    if let Some(source) = attribute_source {
        interpolate_attributes(source, mesh)?;
    }
    if let Some(data) = multires_data.as_mut()
        && let Some(source) = source
        && multires_loop_layout_changed(source, mesh)
    {
        if operation != "subdivide"
            && operation != "loop_cut"
            && operation != "bisect"
            && operation != "symmetrize"
        {
            if data.corner_grids.is_none() && data.external_filepath.is_some() {
                return Err(invalid(
                    "cannot rebase multires data while external MDisps values are unavailable",
                ));
            }
            rebase_multires(source, mesh, data, operation, &changes.mdisp_sources)?;
        }
        multires::write_grid_data(mesh, data)?;
    }
    if let Some(source) = normal_source {
        carry_custom_normals(source, mesh)?;
    }
    mesh.validate().map_err(|error| mesh_error(&error))?;
    Ok(json!({
        "created": changes.to_json(),
        "deleted": changes.deleted_to_json(),
    }))
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use glam::DVec3;
    use proptest::prelude::*;
    use serde_json::{Map, json};
    use std::collections::HashSet;

    use super::{
        IdChanges, apply, bmesh_radial_face_order, bmesh_radial_face_order_after_split, split_edges,
    };
    use crate::geom::color_layers::{self, ColorDomain};
    use crate::geom::modifiers::multires;
    use crate::geom::{BoxParams, Mesh, PlaneParams};

    fn add_test_multires(mesh: &mut Mesh) {
        let mut values = Map::new();
        let mut grid_levels = Vec::new();
        let mut loop_index = 0_u32;
        for face in &mesh.faces {
            let mut corners = Vec::with_capacity(face.vertices.len());
            for _ in &face.vertices {
                let level: u32 = if loop_index.is_multiple_of(2) { 1 } else { 2 };
                let side: usize = if level == 1 { 2 } else { 3 };
                let sample = json!([loop_index + 1, loop_index + 2, 0.5]);
                corners.push(json!(vec![sample; side * side]));
                grid_levels.push(level);
                loop_index += 1;
            }
            values.insert(format!("f{}", face.id), json!(corners));
        }
        let corner_count = loop_index as usize;
        let paint_mask = (0..corner_count)
            .map(|index| {
                let level: u32 = if index % 2 == 0 { 1 } else { 2 };
                let side: usize = if level == 1 { 2 } else { 3 };
                json!({ "level": level, "values": vec![index as u32; side * side] })
            })
            .collect::<Vec<_>>();
        let hidden = (0..corner_count)
            .map(|index| {
                let side = if index % 2 == 0 { 2 } else { 3 };
                let mut grid = vec![false; side * side];
                grid[index % (side * side)] = true;
                json!(grid)
            })
            .collect::<Vec<_>>();
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({
                "total_levels": 2,
                "grid_levels": grid_levels,
                "values": values,
                "paint_mask": paint_mask,
                "hidden": hidden
            }),
        );
    }

    fn adjacent_test_quads() -> Mesh {
        Mesh::from_positions_and_faces(
            vec![
                DVec3::new(-1.0, -1.0, 0.0),
                DVec3::new(0.0, -1.0, 0.0),
                DVec3::new(1.0, -1.0, 0.0),
                DVec3::new(-1.0, 1.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 4, 3], vec![1, 2, 5, 4]],
        )
        .unwrap()
    }

    fn add_parent_test_multires(mesh: &mut Mesh) {
        let mut values = Map::new();
        let mut grid_levels = Vec::new();
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let displacement = if face_index == 0 { 1.0_f32 } else { 2.0_f32 };
            let grid = vec![[0.0, 0.0, displacement]; 4];
            values.insert(
                format!("f{}", face.id),
                json!(vec![grid; face.vertices.len()]),
            );
            grid_levels.extend(vec![1_u32; face.vertices.len()]);
        }
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({ "total_levels": 1, "grid_levels": grid_levels, "values": values }),
        );
    }

    fn assert_multires_grid_count(mesh: &Mesh, corner_count: usize) {
        let data = multires::read_grid_data(mesh).unwrap().unwrap();
        assert_eq!(data.corner_grids.as_ref().unwrap().len(), corner_count);
        assert_eq!(data.grid_levels.as_ref().unwrap().len(), corner_count);
    }

    fn assert_multires_layout(mesh: &Mesh, corner_count: usize) {
        let data = multires::read_grid_data(mesh).unwrap().unwrap();
        let grids = data.corner_grids.as_ref().unwrap();
        let levels = data.grid_levels.as_ref().unwrap();
        assert_eq!(grids.len(), corner_count);
        assert!(grids.iter().zip(levels).all(|(grid, level)| {
            let side = if *level == 1 { 2 } else { 3 };
            grid.len() == side * side
        }));
        assert!(grids.iter().flatten().any(|value| value[2].abs() == 0.5));
        let mask_grids = data.paint_mask_grids.as_ref().unwrap();
        let mask_levels = data.paint_mask_levels.as_ref().unwrap();
        assert_eq!(mask_grids.len(), corner_count);
        assert!(mask_grids.iter().zip(mask_levels).all(|(grid, level)| {
            let side = if *level == 0 {
                0
            } else if *level == 1 {
                2
            } else {
                3
            };
            grid.len() == side * side
        }));
        assert_eq!(data.hidden_grids.as_ref().unwrap().len(), corner_count);
    }

    #[test]
    fn triangulate_reinterpolates_multires_loop_grids() {
        let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
        add_test_multires(&mut mesh);
        let source_face = mesh.faces[0].clone();
        let source = multires::read_grid_data(&mesh).unwrap().unwrap();
        apply(
            &mut mesh,
            "triangulate",
            &json!({ "elements": { "domain": "face", "ids": ["f0"] } }),
        )
        .unwrap();
        assert_eq!(mesh.faces.len(), 2);
        assert_multires_layout(&mesh, 6);

        let output = multires::read_grid_data(&mesh).unwrap().unwrap();
        let source_mask_levels = source.paint_mask_levels.as_ref().unwrap();
        let source_mask_grids = source.paint_mask_grids.as_ref().unwrap();
        let output_mask_levels = output.paint_mask_levels.as_ref().unwrap();
        let output_mask_grids = output.paint_mask_grids.as_ref().unwrap();
        let mut loop_start = 0;
        for face in &mesh.faces {
            for (corner, vertex_id) in face.vertices.iter().enumerate() {
                let source_corner = source_face
                    .vertices
                    .iter()
                    .position(|source_vertex| source_vertex == vertex_id)
                    .unwrap();
                let output_loop = loop_start + corner;
                assert_eq!(
                    output_mask_levels[output_loop],
                    source_mask_levels[source_corner]
                );
                assert_eq!(
                    output_mask_grids[output_loop],
                    source_mask_grids[source_corner]
                );
            }
            loop_start += face.vertices.len();
        }
    }

    #[test]
    fn triangulate_uses_each_callback_face_for_multires() {
        let mut mesh = adjacent_test_quads();
        add_parent_test_multires(&mut mesh);
        let face_ids = mesh
            .faces
            .iter()
            .map(|face| format!("f{}", face.id))
            .collect::<Vec<_>>();

        apply(
            &mut mesh,
            "triangulate",
            &json!({ "elements": { "domain": "face", "ids": face_ids } }),
        )
        .unwrap();

        let data = multires::read_grid_data(&mesh).unwrap().unwrap();
        let grids = data.corner_grids.as_ref().unwrap();
        let mut loop_start = 0;
        for face in &mesh.faces {
            let center_x = face
                .vertices
                .iter()
                .map(|id| mesh.vertex(*id).unwrap().co.x)
                .sum::<f64>()
                / face.vertices.len() as f64;
            let expected = if center_x < 0.0 { 1.0 } else { 2.0 };
            for grid in &grids[loop_start..loop_start + face.vertices.len()] {
                assert!(grid.iter().all(|value| value[2] == expected));
            }
            loop_start += face.vertices.len();
        }
    }

    #[test]
    fn symmetrize_copies_bisected_multires_grids_to_the_mirror() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(-1.0, -1.0, 0.0),
                DVec3::new(1.0, -1.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(-1.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .unwrap();
        add_parent_test_multires(&mut mesh);

        apply(
            &mut mesh,
            "symmetrize",
            &json!({ "axis": "x", "direction": "positive_to_negative" }),
        )
        .unwrap();

        assert_eq!(mesh.faces.len(), 2);
        let data = multires::read_grid_data(&mesh).unwrap().unwrap();
        let grids = data.corner_grids.as_ref().unwrap();
        assert_eq!(grids.len(), 8);
        assert!(grids.iter().flatten().all(|value| value[2] == 1.0));
    }

    #[test]
    fn split_loop_cut_and_bisect_keep_multires_grid_layouts_valid() {
        let mut edge_mesh = adjacent_test_quads();
        add_parent_test_multires(&mut edge_mesh);
        let shared_edge = edge_mesh
            .edges
            .iter()
            .find(|edge| {
                edge.vertices
                    .iter()
                    .all(|id| edge_mesh.vertex(*id).unwrap().co.x == 0.0)
            })
            .unwrap()
            .id;
        apply(
            &mut edge_mesh,
            "subdivide",
            &json!({ "elements": { "domain": "edge", "ids": [format!("e{shared_edge}")] } }),
        )
        .unwrap();
        assert_multires_grid_count(&edge_mesh, 10);

        let mut loop_mesh = adjacent_test_quads();
        add_parent_test_multires(&mut loop_mesh);
        let boundary_edge = loop_mesh
            .edges
            .iter()
            .find(|edge| {
                edge.vertices
                    .iter()
                    .all(|id| loop_mesh.vertex(*id).unwrap().co.x == -1.0)
            })
            .unwrap()
            .id;
        apply(
            &mut loop_mesh,
            "loop_cut",
            &json!({ "elements": { "domain": "edge", "ids": [format!("e{boundary_edge}")] }, "cuts": 2 }),
        )
        .unwrap();
        assert_multires_grid_count(&loop_mesh, 24);

        let mut bisect_mesh = adjacent_test_quads();
        add_parent_test_multires(&mut bisect_mesh);
        apply(
            &mut bisect_mesh,
            "bisect",
            &json!({ "point": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0] }),
        )
        .unwrap();
        assert_multires_grid_count(&bisect_mesh, 16);
    }

    #[test]
    fn shared_edge_mdisps_follow_bmesh_radial_callback_order() {
        let mut mesh = adjacent_test_quads();
        add_parent_test_multires(&mut mesh);
        let left_face_id = mesh.faces[0].id;
        let shared_edge = mesh
            .edges
            .iter()
            .find(|edge| {
                edge.vertices
                    .iter()
                    .all(|id| mesh.vertex(*id).unwrap().co.x == 0.0)
            })
            .unwrap()
            .id;
        apply(
            &mut mesh,
            "subdivide",
            &json!({ "elements": { "domain": "edge", "ids": [format!("e{shared_edge}")] } }),
        )
        .unwrap();

        let left_face_index = mesh
            .faces
            .iter()
            .position(|face| face.id == left_face_id)
            .unwrap();
        let left_face = &mesh.faces[left_face_index];
        let midpoint = mesh
            .vertices
            .iter()
            .find(|vertex| vertex.co == DVec3::ZERO)
            .unwrap()
            .id;
        let midpoint_corner = left_face
            .vertices
            .iter()
            .position(|vertex_id| *vertex_id == midpoint)
            .unwrap();
        let loop_start = mesh.faces[..left_face_index]
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>();
        let data = multires::read_grid_data(&mesh).unwrap().unwrap();
        let grid = &data.corner_grids.as_ref().unwrap()[loop_start + midpoint_corner];
        // Level-1 sample 3 is the corner on the shared edge; F0 is the last radial callback.
        assert_eq!(grid[3][2], 1.0);
    }

    #[test]
    fn nonmanifold_edge_radial_order_rotates_across_repeated_cuts() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, -1.0),
                DVec3::new(0.0, 0.0, 1.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(-1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2], vec![1, 0, 3], vec![0, 1, 4]],
        )
        .unwrap();
        let mut values = Map::new();
        let mut grid_levels = Vec::new();
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let displacement = (face_index + 1) as f32;
            let grid = vec![[0.0, 0.0, displacement]; 4];
            values.insert(
                format!("f{}", face.id),
                json!(vec![grid; face.vertices.len()]),
            );
            grid_levels.extend(vec![1_u32; face.vertices.len()]);
        }
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({ "total_levels": 1, "grid_levels": grid_levels, "values": values }),
        );

        let face_ids = mesh.faces.iter().map(|face| face.id).collect::<Vec<_>>();
        let radial = bmesh_radial_face_order(&mesh, (0, 1));
        assert_eq!(radial, vec![face_ids[2], face_ids[0], face_ids[1]]);
        let first_radial = bmesh_radial_face_order_after_split(&radial);
        assert_eq!(first_radial, vec![face_ids[1], face_ids[2], face_ids[0]]);
        let shared_edge = mesh
            .edges
            .iter()
            .find(|edge| edge.vertices.contains(&0) && edge.vertices.contains(&1))
            .unwrap()
            .id;
        let mut first_cut = IdChanges::default();
        let initial_lineage = IdChanges::default();
        let first_edge_ids = HashSet::from([shared_edge]);
        let mut data = multires::read_grid_data(&mesh).unwrap().unwrap();
        split_edges(
            &mut mesh,
            &first_edge_ids,
            &mut first_cut,
            &initial_lineage,
            Some(&initial_lineage.mdisp_radial_faces),
            Some(&mut data),
        )
        .unwrap();

        let midpoint = mesh
            .vertices
            .iter()
            .find(|vertex| vertex.co == DVec3::ZERO)
            .unwrap()
            .id;
        let face_zero = &mesh.faces[0];
        let midpoint_corner = face_zero
            .vertices
            .iter()
            .position(|vertex_id| *vertex_id == midpoint)
            .unwrap();
        assert_eq!(
            data.corner_grids.as_ref().unwrap()[midpoint_corner][3][2],
            2.0
        );

        assert!(
            first_cut.edges.iter().all(|edge_id| {
                first_cut.mdisp_radial_faces.get(edge_id) == Some(&first_radial)
            })
        );
        let second_edge_ids = first_cut.edges.iter().copied().collect::<HashSet<_>>();
        let mut second_cut = IdChanges::default();
        split_edges(
            &mut mesh,
            &second_edge_ids,
            &mut second_cut,
            &first_cut,
            Some(&first_cut.mdisp_radial_faces),
            Some(&mut data),
        )
        .unwrap();
        let second_radial = bmesh_radial_face_order_after_split(&first_radial);
        assert_eq!(second_radial, vec![face_ids[0], face_ids[1], face_ids[2]]);
        assert!(
            second_cut.edges.iter().all(|edge_id| {
                second_cut.mdisp_radial_faces.get(edge_id) == Some(&second_radial)
            })
        );
        let loop_count = mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>();
        assert_eq!(loop_count, 18);
        assert_eq!(data.corner_grids.as_ref().unwrap().len(), loop_count);
    }

    #[test]
    fn bisect_rebases_multires_after_each_blender_edge_and_face_callback() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .unwrap();
        let peak_grid = vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let zero_grid = vec![[0.0, 0.0, 0.0]; 4];
        let mut values = Map::new();
        values.insert(
            format!("f{}", mesh.faces[0].id),
            json!([peak_grid, zero_grid.clone(), zero_grid.clone(), zero_grid]),
        );
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({ "total_levels": 1, "grid_levels": [1, 1, 1, 1], "values": values }),
        );

        apply(
            &mut mesh,
            "bisect",
            &json!({
                "point": [0.5, 0.0, 0.0],
                "normal": [1.0, 0.0, 0.0],
                "clear_side": "none"
            }),
        )
        .unwrap();

        let data = multires::read_grid_data(&mesh).unwrap().unwrap();
        let grids = data.corner_grids.as_ref().unwrap();
        let peak = grids
            .iter()
            .flatten()
            .map(|value| value[2].abs())
            .fold(0.0_f32, f32::max);
        // Per-edge and per-face Blender callbacks lower this peak versus a one-shot rebase.
        assert!(peak > 0.0);
        assert!(peak < 1.0);
    }

    #[test]
    fn dissolve_interpolates_from_both_joined_multires_faces() {
        let mut mesh = adjacent_test_quads();
        add_parent_test_multires(&mut mesh);
        let shared_vertices: Vec<_> = mesh
            .vertices
            .iter()
            .filter(|vertex| vertex.co.x == 0.0)
            .map(|vertex| vertex.id)
            .collect();
        let shared_edge = mesh
            .edges
            .iter()
            .find(|edge| {
                edge.vertices.contains(&shared_vertices[0])
                    && edge.vertices.contains(&shared_vertices[1])
            })
            .unwrap()
            .id;
        apply(
            &mut mesh,
            "dissolve",
            &json!({ "elements": { "domain": "edge", "ids": [format!("e{shared_edge}")] } }),
        )
        .unwrap();

        assert_eq!(mesh.faces.len(), 1);
        let face = &mesh.faces[0];
        let shared_corner = face
            .vertices
            .iter()
            .position(|id| {
                let position = mesh.vertex(*id).unwrap().co;
                position.x == 0.0 && position.y == -1.0
            })
            .unwrap();
        let data = multires::read_grid_data(&mesh).unwrap().unwrap();
        let grid = &data.corner_grids.as_ref().unwrap()[shared_corner];
        assert!(grid.iter().any(|value| value[2] == 1.0));
        assert!(grid.iter().any(|value| value[2] == 2.0));
    }

    #[test]
    fn fill_without_multires_callback_uses_default_grids() {
        let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
        add_test_multires(&mut mesh);
        apply(
            &mut mesh,
            "delete",
            &json!({ "elements": { "domain": "face", "ids": ["f0"] } }),
        )
        .unwrap();
        let edge_ids = mesh
            .edges
            .iter()
            .map(|edge| format!("e{}", edge.id))
            .collect::<Vec<_>>();
        apply(
            &mut mesh,
            "fill",
            &json!({ "elements": { "domain": "edge", "ids": edge_ids } }),
        )
        .unwrap();

        let data = multires::read_grid_data(&mesh).unwrap().unwrap();
        let corner_count = mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>();
        assert_eq!(data.grid_levels.as_ref().unwrap().len(), corner_count);
        assert_eq!(data.corner_grids.as_ref().unwrap().len(), corner_count);
        assert_eq!(data.paint_mask_levels.as_ref().unwrap().len(), corner_count);
        assert_eq!(data.paint_mask_grids.as_ref().unwrap().len(), corner_count);
        assert_eq!(data.hidden_grids.as_ref().unwrap().len(), corner_count);
        assert!(
            data.corner_grids
                .as_ref()
                .unwrap()
                .iter()
                .flatten()
                .all(|value| *value == [0.0; 3])
        );
        let mask_levels = data.paint_mask_levels.as_ref().unwrap();
        assert!(mask_levels.iter().all(|level| *level == 0));
        assert!(
            data.paint_mask_grids
                .as_ref()
                .unwrap()
                .iter()
                .all(Vec::is_empty)
        );
        assert!(
            data.hidden_grids
                .as_ref()
                .unwrap()
                .iter()
                .all(|grid| grid.iter().all(|hidden| !hidden))
        );
    }

    #[test]
    fn subdivide_reinterpolates_multires_loop_grids() {
        let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
        add_test_multires(&mut mesh);
        apply(
            &mut mesh,
            "subdivide",
            &json!({ "elements": { "domain": "face", "ids": ["f0"] } }),
        )
        .unwrap();
        assert_eq!(mesh.faces.len(), 4);
        assert_multires_layout(&mesh, 16);
    }

    #[test]
    fn flip_normals_rebases_grids_and_reorders_multires_loop_data() {
        let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
        add_test_multires(&mut mesh);
        let source = multires::read_grid_data(&mesh).unwrap().unwrap();
        apply(
            &mut mesh,
            "flip_normals",
            &json!({ "elements": { "domain": "face", "ids": ["f0"] } }),
        )
        .unwrap();
        let flipped = multires::read_grid_data(&mesh).unwrap().unwrap();
        let source_corner = &source.corner_grids.as_ref().unwrap()[3][0];
        assert_eq!(
            flipped.corner_grids.as_ref().unwrap()[0][0],
            [source_corner[1], source_corner[0], -source_corner[2]]
        );
        assert_eq!(
            flipped.paint_mask_grids.as_ref().unwrap()[0],
            source.paint_mask_grids.as_ref().unwrap()[3]
        );
        assert_eq!(
            flipped.hidden_grids.as_ref().unwrap()[0],
            source.hidden_grids.as_ref().unwrap()[3]
        );
        assert_multires_layout(&mesh, 4);
    }

    #[test]
    fn unavailable_external_mdisps_reject_topology_edits_without_mutation() {
        let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
        mesh.attributes.insert(
            "multires_displacement".to_owned(),
            json!({ "total_levels": 2, "external": { "filepath": "//grids.mdisps" } }),
        );
        let before = mesh.clone();
        assert!(
            apply(
                &mut mesh,
                "triangulate",
                &json!({ "elements": { "domain": "face", "ids": ["f0"] } }),
            )
            .is_err()
        );
        assert_eq!(mesh, before);
    }

    /// Blender 5.2.2 BEAUTY triangulation of each oracle case (`tests/fixtures/triangulate`, from the
    /// Triangulate modifier, which runs the same `BM_mesh_triangulate` kernel as this op). The
    /// triangles are compared as sets with each triangle rotated to its smallest corner first, so
    /// the diagonals and winding must match, while output order does not.
    #[test]
    fn triangulate_op_matches_blender_beauty_diagonals() {
        const ORACLE: &str =
            include_str!("../../../../tests/fixtures/triangulate/blender_triangulate.json");
        let oracle: serde_json::Value = serde_json::from_str(ORACLE).unwrap();
        let cases = oracle["cases"].as_array().unwrap();
        assert_eq!(
            cases.len(),
            117,
            "complete Blender triangulate oracle is required"
        );
        for case in cases {
            let kind = case["kind"].as_str().unwrap();
            let points: Vec<DVec3> = case["positions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|point| {
                    let point = point.as_array().unwrap();
                    DVec3::new(
                        point[0].as_f64().unwrap(),
                        point[1].as_f64().unwrap(),
                        point[2].as_f64().unwrap(),
                    )
                })
                .collect();
            let count = points.len();
            let mut mesh =
                Mesh::from_positions_and_faces(points, vec![(0..count).collect()]).unwrap();
            apply(
                &mut mesh,
                "triangulate",
                &json!({"elements": {"domain": "face", "ids": ["f0"]}}),
            )
            .unwrap();
            let index: std::collections::HashMap<u32, usize> = mesh
                .vertices
                .iter()
                .enumerate()
                .map(|(index, vertex)| (vertex.id, index))
                .collect();
            let actual: Vec<[usize; 3]> = mesh
                .faces
                .iter()
                .map(|face| {
                    assert_eq!(
                        face.vertices.len(),
                        3,
                        "{kind}: triangulate left a non-triangle"
                    );
                    [0, 1, 2].map(|corner| index[&face.vertices[corner]])
                })
                .collect();
            let expected: Vec<[usize; 3]> = case["triangles"]
                .as_array()
                .unwrap()
                .iter()
                .map(|triangle| {
                    let triangle = triangle.as_array().unwrap();
                    [0, 1, 2].map(|i| usize::try_from(triangle[i].as_u64().unwrap()).unwrap())
                })
                .collect();
            assert_eq!(
                canonical_triangles(&actual),
                canonical_triangles(&expected),
                "Blender BEAUTY diagonals for {kind}"
            );
        }
    }

    /// Each triangle rotated so its smallest index comes first (winding kept), sorted.
    fn canonical_triangles(triangles: &[[usize; 3]]) -> Vec<[usize; 3]> {
        let mut canonical: Vec<[usize; 3]> = triangles
            .iter()
            .map(|triangle| {
                let start = (0..3).min_by_key(|&i| triangle[i]).unwrap();
                [0, 1, 2].map(|offset| triangle[(start + offset) % 3])
            })
            .collect();
        canonical.sort_unstable();
        canonical
    }
    #[test]
    fn triangulate_replaces_selected_ngon_with_winding_preserving_triangles() {
        let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
        let result = apply(
            &mut mesh,
            "triangulate",
            &json!({ "elements": { "domain": "face", "ids": ["f0"] } }),
        )
        .unwrap();

        let triangles = mesh.tessellate().unwrap();
        assert_eq!(triangles.len(), 2);
        assert_eq!(result["created"]["faces"].as_array().unwrap().len(), 2);
        assert_eq!(result["deleted"]["faces"], json!(["f0"]));
        for [first, second, third] in triangles {
            let first = mesh.vertex(first).unwrap().co;
            let second = mesh.vertex(second).unwrap().co;
            let third = mesh.vertex(third).unwrap().co;
            assert!((second - first).cross(third - first).z > 0.0);
        }
    }

    #[test]
    fn extrude_face_builds_side_ring_and_cap() {
        let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
        let result = apply(
            &mut mesh,
            "extrude",
            &json!({
                "elements": { "domain": "face", "ids": ["f0"] },
                "offset": [0.0, 0.0, 1.0]
            }),
        )
        .unwrap();

        mesh.validate().unwrap();
        assert_eq!(
            (mesh.vertices.len(), mesh.edges.len(), mesh.faces.len()),
            (8, 12, 5)
        );
        assert_eq!(mesh.tessellate().unwrap().len(), 10);
        assert_eq!(result["created"]["faces"].as_array().unwrap().len(), 5);
        assert_eq!(result["deleted"]["faces"], json!(["f0"]));
        assert!(
            mesh.vertex(4)
                .unwrap()
                .co
                .distance(DVec3::new(-1.0, -1.0, 1.0))
                <= f64::EPSILON
        );
    }

    #[test]
    fn mirror_merge_shares_vertices_on_the_mirror_plane() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, -1.0, -1.0),
                DVec3::new(1.0, -1.0, -1.0),
                DVec3::new(1.0, 1.0, -1.0),
                DVec3::new(0.0, 1.0, -1.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .unwrap();
        let result = apply(
            &mut mesh,
            "mirror",
            &json!({ "axis": "x", "merge": true, "threshold": 1e-8 }),
        )
        .unwrap();

        mesh.validate().unwrap();
        assert_eq!(
            (mesh.vertices.len(), mesh.edges.len(), mesh.faces.len()),
            (6, 7, 2)
        );
        assert_eq!(result["created"]["faces"].as_array().unwrap().len(), 1);
        assert!(
            mesh.vertices
                .iter()
                .any(|vertex| vertex.co == DVec3::new(-1.0, 1.0, -1.0))
        );
    }

    #[test]
    fn dissolve_edge_merges_adjacent_faces() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let result = apply(
            &mut mesh,
            "dissolve",
            &json!({ "elements": { "domain": "edge", "ids": ["e0"] } }),
        )
        .unwrap();

        mesh.validate().unwrap();
        assert_eq!(
            (mesh.vertices.len(), mesh.edges.len(), mesh.faces.len()),
            (8, 11, 5)
        );
        assert_eq!(result["deleted"]["edges"], json!(["e0"]));
        assert_eq!(result["deleted"]["faces"].as_array().unwrap().len(), 2);
        assert_eq!(result["created"]["faces"].as_array().unwrap().len(), 1);
    }
    #[test]
    fn bisect_clear_side_keeps_only_the_opposite_half() {
        for clear_side in ["positive", "negative"] {
            let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
            apply(
                &mut mesh,
                "bisect",
                &json!({
                    "point": [0.0, 0.0, 0.0],
                    "normal": [1.0, 0.0, 0.0],
                    "clear_side": clear_side
                }),
            )
            .unwrap();
            mesh.validate().unwrap();
            assert_eq!(mesh.faces.len(), 1);
            for id in &mesh.faces[0].vertices {
                let x = mesh.vertex(*id).unwrap().co.x;
                if clear_side == "positive" {
                    assert!(x <= 1.0e-10);
                } else {
                    assert!(x >= -1.0e-10);
                }
            }
        }
    }

    #[test]
    fn set_attribute_updates_face_material_index() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        apply(
            &mut mesh,
            "set_attribute",
            &json!({
                "elements": { "domain": "face", "ids": ["f0"] },
                "name": "material_index",
                "value": 7
            }),
        )
        .unwrap();
        assert_eq!(mesh.faces[0].material_index, 7);
        mesh.validate().unwrap();
    }

    #[test]
    fn set_attribute_persists_edge_flags_by_persistent_id() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        apply(
            &mut mesh,
            "set_attribute",
            &json!({
                "elements": { "domain": "edge", "ids": ["e0"] },
                "name": "seam",
                "value": true
            }),
        )
        .unwrap();

        assert_eq!(mesh.attributes["seam"]["values"]["e0"], json!(true));
    }

    proptest! {
        #[test]
        fn face_subdivision_preserves_closed_cube_euler(cuts in 1_u32..=3) {
            let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
            let before_euler = mesh.vertices.len() + mesh.faces.len() - mesh.edges.len();
            let result = apply(
                &mut mesh,
                "subdivide",
                &json!({
                    "elements": { "domain": "face", "ids": ["f0"] },
                    "cuts": cuts
                }),
            );
            prop_assert!(result.is_ok());
            prop_assert!(mesh.validate().is_ok());
            prop_assert_eq!(
                mesh.vertices.len() + mesh.faces.len() - mesh.edges.len(),
                before_euler
            );
        }

        #[test]
        fn face_extrusion_preserves_closed_cube_euler(distance in 0.1_f64..=2.0) {
            let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
            let before_euler = mesh.vertices.len() + mesh.faces.len() - mesh.edges.len();
            let result = apply(
                &mut mesh,
                "extrude",
                &json!({
                    "elements": { "domain": "face", "ids": ["f0"] },
                    "offset": [0.0, 0.0, -distance]
                }),
            );
            prop_assert!(result.is_ok());
            prop_assert!(mesh.validate().is_ok());
            prop_assert_eq!(
                mesh.vertices.len() + mesh.faces.len() - mesh.edges.len(),
                before_euler
            );
        }
    }
    #[test]
    fn loop_cut_propagates_around_quad_ring_for_each_requested_cut() {
        for cuts in 1_u32..=3 {
            let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
            let result = apply(
                &mut mesh,
                "mesh.loop_cut",
                &json!({
                    "elements": { "domain": "edge", "ids": ["e0"] },
                    "cuts": cuts
                }),
            )
            .unwrap();

            mesh.validate().unwrap();
            let cut_count = usize::try_from(cuts).unwrap();
            assert_eq!(mesh.vertices.len(), 8 + 4 * cut_count);
            assert_eq!(mesh.faces.len(), 6 + 4 * cut_count);
            assert_eq!(
                result["created"]["vertices"].as_array().unwrap().len(),
                4 * cut_count
            );
            assert_eq!(
                result["created"]["faces"].as_array().unwrap().len(),
                4 * (cut_count + 1)
            );
            assert_eq!(result["deleted"]["faces"].as_array().unwrap().len(), 4);
        }
    }

    #[test]
    fn knife_keeps_both_halves_and_closed_edge_incidence() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        apply(
            &mut mesh,
            "knife",
            &json!({
                "elements": {
                    "domain": "face",
                    "ids": ["f0", "f1", "f2", "f3", "f4", "f5"]
                },
                "plane": {
                    "point": [0.0, 0.0, 0.0],
                    "normal": [1.0, 0.0, 0.0]
                }
            }),
        )
        .unwrap();

        mesh.validate().unwrap();
        assert_eq!(mesh.faces.len(), 10);
        let mut incidences = std::collections::HashMap::new();
        for face in &mesh.faces {
            for index in 0..face.vertices.len() {
                let first = face.vertices[index];
                let second = face.vertices[(index + 1) % face.vertices.len()];
                let key = if first < second {
                    (first, second)
                } else {
                    (second, first)
                };
                *incidences.entry(key).or_insert(0) += 1;
            }
        }
        assert!(incidences.values().all(|count| *count == 2));
        assert_eq!(mesh.edges.len(), incidences.len());
        for edge in &mesh.edges {
            let key = if edge.vertices[0] < edge.vertices[1] {
                (edge.vertices[0], edge.vertices[1])
            } else {
                (edge.vertices[1], edge.vertices[0])
            };
            assert_eq!(incidences.get(&key), Some(&2));
        }
        let mut flattened_mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        apply(
            &mut flattened_mesh,
            "mesh.knife",
            &json!({
                "elements": {
                    "domain": "face",
                    "ids": ["f0", "f1", "f2", "f3", "f4", "f5"]
                },
                "point": [0.0, 0.0, 0.0],
                "normal": [1.0, 0.0, 0.0],
                "threshold": 1e-10
            }),
        )
        .unwrap();
        assert_eq!(flattened_mesh.faces.len(), 10);
    }
    #[test]
    fn bridge_connects_two_compatible_edge_loops() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(-1.0, -1.0, 0.0),
                DVec3::new(1.0, -1.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(-1.0, 1.0, 0.0),
                DVec3::new(-1.0, -1.0, 1.0),
                DVec3::new(1.0, -1.0, 1.0),
                DVec3::new(1.0, 1.0, 1.0),
                DVec3::new(-1.0, 1.0, 1.0),
            ],
            vec![vec![0, 1, 2, 3], vec![4, 7, 6, 5]],
        )
        .unwrap();

        let result = apply(
            &mut mesh,
            "bridge",
            &json!({
                "elements": {
                    "domain": "edge",
                    "ids": ["e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7"]
                }
            }),
        )
        .unwrap();

        mesh.validate().unwrap();
        assert_eq!(mesh.faces.len(), 6);
        assert_eq!(result["created"]["faces"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn split_detaches_selected_faces_from_unselected_topology() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let result = apply(
            &mut mesh,
            "split",
            &json!({ "elements": { "domain": "face", "ids": ["f0"] } }),
        )
        .unwrap();

        mesh.validate().unwrap();
        assert_eq!(mesh.vertices.len(), 12);
        let selected_vertices: std::collections::HashSet<_> = mesh
            .faces
            .iter()
            .find(|face| face.id == 0)
            .unwrap()
            .vertices
            .iter()
            .copied()
            .collect();
        assert!(mesh.faces.iter().filter(|face| face.id != 0).all(|face| {
            face.vertices
                .iter()
                .all(|id| !selected_vertices.contains(id))
        }));
        assert_eq!(result["created"]["vertices"].as_array().unwrap().len(), 4);
        assert_eq!(result["created"]["edges"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn symmetrize_replaces_the_opposite_side_from_explicit_direction() {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(2.0, -1.0, 0.0),
                DVec3::new(3.0, -1.0, 0.0),
                DVec3::new(3.0, 1.0, 0.0),
                DVec3::new(2.0, 1.0, 0.0),
                DVec3::new(0.0, -1.0, 0.0),
                DVec3::new(-1.0, -1.0, 0.0),
                DVec3::new(-1.0, 1.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
        )
        .unwrap();

        let result = apply(
            &mut mesh,
            "symmetrize",
            &json!({
                "elements": { "domain": "face", "ids": ["f0"] },
                "axis": "x",
                "direction": "positive_to_negative",
                "origin": [1.0, 0.0, 0.0],
                "threshold": 1e-8
            }),
        )
        .unwrap();

        mesh.validate().unwrap();
        assert_eq!(mesh.faces.len(), 2);
        assert_eq!(result["deleted"]["faces"], json!(["f1"]));
        assert_eq!(result["created"]["faces"].as_array().unwrap().len(), 1);
        assert_eq!(result["created"]["vertices"].as_array().unwrap().len(), 0);
    }
    #[test]
    fn symmetrize_splits_crossing_faces_and_keeps_closed_topology() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        apply(
            &mut mesh,
            "symmetrize",
            &json!({
                "axis": "x",
                "direction": "positive_to_negative"
            }),
        )
        .unwrap();

        mesh.validate().unwrap();
        assert_eq!(mesh.faces.len(), 10);
        let mut incidences = std::collections::HashMap::new();
        for face in &mesh.faces {
            for index in 0..face.vertices.len() {
                let first = face.vertices[index];
                let second = face.vertices[(index + 1) % face.vertices.len()];
                let key = if first < second {
                    (first, second)
                } else {
                    (second, first)
                };
                *incidences.entry(key).or_insert(0) += 1;
            }
        }
        assert!(incidences.values().all(|count| *count == 2));
        assert_eq!(mesh.edges.len(), incidences.len());
    }
    #[test]
    fn symmetrize_reflects_loose_vertices_and_edges() {
        let mut mesh = Mesh::default();
        let first = mesh.insert_vertex(DVec3::new(1.0, 0.0, 0.0)).unwrap();
        let second = mesh.insert_vertex(DVec3::new(1.0, 1.0, 0.0)).unwrap();
        mesh.insert_edge([first, second]).unwrap();

        let result = apply(
            &mut mesh,
            "symmetrize",
            &json!({
                "axis": "x",
                "direction": "positive_to_negative"
            }),
        )
        .unwrap();

        mesh.validate().unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.edges.len(), 2);
        assert_eq!(result["created"]["vertices"].as_array().unwrap().len(), 2);
        assert_eq!(result["created"]["edges"].as_array().unwrap().len(), 1);
        assert!(
            mesh.vertices
                .iter()
                .any(|vertex| vertex.co == DVec3::new(-1.0, 0.0, 0.0))
        );
    }

    fn create_color(mesh: &mut Mesh, name: &str, domain: &str, default: &serde_json::Value) {
        apply(
            mesh,
            "attribute_create",
            &json!({ "name": name, "domain": domain, "type": "color", "default": default }),
        )
        .unwrap();
    }

    #[test]
    fn attribute_create_color_registers_a_point_layer_that_reads_back() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        create_color(&mut mesh, "Tint", "point", &json!([0.5, 0.25, 0.0, 1.0]));

        let store = color_layers::read(&mesh);
        assert_eq!(store.layers.len(), 1);
        assert_eq!(store.layers[0].name, "Tint");
        assert_eq!(store.layers[0].domain, ColorDomain::Point);
        assert_eq!(
            store.layers[0].values,
            vec![[0.5, 0.25, 0.0, 1.0]; mesh.vertices.len()]
        );
        assert_eq!(store.active.as_deref(), Some("Tint"));
        assert_eq!(store.render.as_deref(), Some("Tint"));
        assert_eq!(mesh.attributes["Tint"]["type"], json!("color"));
        assert_eq!(
            mesh.attributes[color_layers::ATTRIBUTE]["layers"],
            json!(["Tint"])
        );
        color_layers::validate(&mesh).unwrap();
    }

    #[test]
    fn attribute_create_color_corner_layer_has_one_value_per_corner() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        create_color(&mut mesh, "Paint", "corner", &json!([1.0, 1.0, 1.0, 1.0]));

        let corners: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
        let store = color_layers::read(&mesh);
        assert_eq!(store.layers[0].domain, ColorDomain::Corner);
        assert_eq!(store.layers[0].values, vec![[1.0, 1.0, 1.0, 1.0]; corners]);
        color_layers::validate(&mesh).unwrap();
    }

    #[test]
    fn attribute_create_color_appends_and_keeps_existing_designations() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        create_color(&mut mesh, "First", "point", &json!([1.0, 0.0, 0.0, 1.0]));
        create_color(&mut mesh, "Second", "corner", &json!([0.0, 1.0, 0.0, 1.0]));

        let store = color_layers::read(&mesh);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["First", "Second"]);
        assert_eq!(store.active.as_deref(), Some("First"));
        assert_eq!(store.render.as_deref(), Some("First"));
        color_layers::validate(&mesh).unwrap();
    }

    #[test]
    fn attribute_create_non_color_attribute_leaves_the_color_store_alone() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        apply(
            &mut mesh,
            "attribute_create",
            &json!({ "name": "weight", "domain": "point", "type": "float" }),
        )
        .unwrap();

        assert!(!mesh.attributes.contains_key(color_layers::ATTRIBUTE));
        assert_eq!(color_layers::read(&mesh).layers.len(), 0);
    }

    #[test]
    fn attribute_create_color_rejects_edge_and_face_domains_and_byte_color() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        for (domain, kind) in [
            ("edge", "color"),
            ("face", "color"),
            ("point", "byte_color"),
        ] {
            let result = apply(
                &mut mesh,
                "attribute_create",
                &json!({ "name": "Bad", "domain": domain, "type": kind }),
            );
            assert!(result.is_err(), "{domain} {kind} must be rejected");
        }
        assert!(!mesh.attributes.contains_key("Bad"));
        assert!(!mesh.attributes.contains_key(color_layers::ATTRIBUTE));
    }

    #[test]
    fn user_attribute_named_like_the_color_store_is_escaped_and_leaves_the_store_alone() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        create_color(&mut mesh, "Tint", "point", &json!([0.0, 0.0, 0.0, 1.0]));
        let registry = mesh.attributes[color_layers::ATTRIBUTE].clone();
        let vertex = format!("v{}", mesh.vertices[0].id);
        apply(
            &mut mesh,
            "attribute_create",
            &json!({ "name": color_layers::ATTRIBUTE, "domain": "point", "type": "float" }),
        )
        .unwrap();
        assert!(mesh.attributes.contains_key("user:color_layers"));
        apply(
            &mut mesh,
            "attribute_update",
            &json!({
                "name": color_layers::ATTRIBUTE,
                "elements": { "domain": "point", "ids": [vertex.clone()] },
                "value": 2.0
            }),
        )
        .unwrap();
        assert_eq!(
            mesh.attributes["user:color_layers"]["values"][vertex.as_str()],
            json!(2.0)
        );
        apply(
            &mut mesh,
            "attribute_delete",
            &json!({ "name": color_layers::ATTRIBUTE }),
        )
        .unwrap();
        assert!(!mesh.attributes.contains_key("user:color_layers"));
        assert_eq!(mesh.attributes[color_layers::ATTRIBUTE], registry);
        color_layers::validate(&mesh).unwrap();
    }

    #[test]
    fn attribute_update_writes_color_values_and_keeps_the_registry() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        create_color(&mut mesh, "Tint", "point", &json!([0.0, 0.0, 0.0, 1.0]));
        create_color(&mut mesh, "Paint", "corner", &json!([0.0, 0.0, 0.0, 1.0]));
        let vertex = format!("v{}", mesh.vertices[0].id);
        apply(
            &mut mesh,
            "attribute_update",
            &json!({
                "name": "Tint",
                "elements": { "domain": "point", "ids": [vertex.clone()] },
                "value": [1.0, 0.5, 0.0, 1.0]
            }),
        )
        .unwrap();
        apply(
            &mut mesh,
            "set_attribute",
            &json!({
                "name": "Paint",
                "elements": { "domain": "face", "ids": ["f0"] },
                "value": [0.25, 0.5, 0.75, 1.0]
            }),
        )
        .unwrap();

        assert_eq!(
            mesh.attributes["Tint"]["values"][&vertex],
            json!([1.0, 0.5, 0.0, 1.0])
        );
        let store = color_layers::read(&mesh);
        assert_eq!(store.layers[0].values[0], [1.0, 0.5, 0.0, 1.0]);
        assert_eq!(store.layers[0].values.len(), mesh.vertices.len());
        assert!(store.layers[1].values.contains(&[0.25, 0.5, 0.75, 1.0]));
        assert_eq!(
            mesh.attributes[color_layers::ATTRIBUTE]["layers"],
            json!(["Tint", "Paint"])
        );
        color_layers::validate(&mesh).unwrap();
    }

    #[test]
    fn attribute_delete_drops_a_color_layer_from_the_registry() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        create_color(&mut mesh, "First", "point", &json!([1.0, 0.0, 0.0, 1.0]));
        create_color(&mut mesh, "Second", "corner", &json!([0.0, 1.0, 0.0, 1.0]));

        apply(&mut mesh, "attribute_delete", &json!({ "name": "First" })).unwrap();
        let store = color_layers::read(&mesh);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["Second"]);
        assert_eq!(store.active.as_deref(), Some("Second"));
        assert_eq!(store.render.as_deref(), Some("Second"));
        assert!(!mesh.attributes.contains_key("First"));
        color_layers::validate(&mesh).unwrap();

        apply(&mut mesh, "attribute_delete", &json!({ "name": "Second" })).unwrap();
        assert!(!mesh.attributes.contains_key("Second"));
        assert!(!mesh.attributes.contains_key(color_layers::ATTRIBUTE));
        color_layers::validate(&mesh).unwrap();
    }

    #[test]
    fn attribute_delete_of_selected_color_elements_resets_them_to_opaque_black() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        create_color(&mut mesh, "Tint", "point", &json!([1.0, 1.0, 1.0, 1.0]));
        create_color(&mut mesh, "Paint", "corner", &json!([1.0, 1.0, 1.0, 1.0]));
        let vertex = format!("v{}", mesh.vertices[0].id);
        apply(
            &mut mesh,
            "attribute_delete",
            &json!({ "name": "Tint", "elements": { "domain": "point", "ids": [vertex] } }),
        )
        .unwrap();
        apply(
            &mut mesh,
            "attribute_delete",
            &json!({ "name": "Paint", "elements": { "domain": "face", "ids": ["f0"] } }),
        )
        .unwrap();

        let store = color_layers::read(&mesh);
        assert_eq!(store.layers.len(), 2);
        assert_eq!(store.layers[0].values[0], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(store.layers[0].values[1], [1.0, 1.0, 1.0, 1.0]);
        let face = mesh.faces.iter().find(|face| face.id == 0).unwrap();
        let black = store.layers[1]
            .values
            .iter()
            .filter(|color| **color == [0.0, 0.0, 0.0, 1.0])
            .count();
        assert_eq!(black, face.vertices.len());
        color_layers::validate(&mesh).unwrap();
    }
}
