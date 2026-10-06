use std::collections::{HashMap, HashSet};

use glam::{DQuat, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{PotError, Result},
    geom::{Face, Mesh, MeshError, edge_key},
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

#[derive(Default)]
struct IdChanges {
    vertices: Vec<u32>,
    edges: Vec<u32>,
    faces: Vec<u32>,
    deleted_vertices: Vec<u32>,
    deleted_edges: Vec<u32>,
    deleted_faces: Vec<u32>,
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
                .get(name)
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
                insert_face(
                    mesh,
                    vec![
                        previous[index],
                        previous[next],
                        current[next],
                        current[index],
                    ],
                    face.material_index,
                    &mut changes,
                )?;
            }
            previous = current;
        }
        insert_face(mesh, previous, face.material_index, &mut changes)?;
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
    let midpoints = split_edges(mesh, &edge_ids, &mut changes)?;
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
        insert_face(
            mesh,
            face.vertices.iter().map(|id| duplicated[id]).collect(),
            face.material_index,
            &mut changes,
        )?;
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            insert_face(
                mesh,
                vec![first, second, duplicated[&second], duplicated[&first]],
                face.material_index,
                &mut changes,
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
        insert_face(mesh, top, face.material_index, &mut changes)?;
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
        insert_face(
            mesh,
            vec![
                first,
                second,
                selected_vertices[&second],
                selected_vertices[&first],
            ],
            face.material_index,
            &mut changes,
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
            insert_face(
                mesh,
                vec![
                    face.vertices[index],
                    face.vertices[next],
                    inner[next],
                    inner[index],
                ],
                face.material_index,
                &mut changes,
            )?;
        }
        insert_face(mesh, inner, face.material_index, &mut changes)?;
    }
    Ok(changes)
}

fn split_edges(
    mesh: &mut Mesh,
    edge_ids: &HashSet<u32>,
    changes: &mut IdChanges,
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
    let split_keys: HashSet<_> = midpoints.keys().copied().collect();
    for edge in selected {
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
    for face in &mut mesh.faces {
        let mut split_vertices = Vec::with_capacity(face.vertices.len() * 2);
        for index in 0..face.vertices.len() {
            let current = face.vertices[index];
            let next = face.vertices[(index + 1) % face.vertices.len()];
            split_vertices.push(current);
            if split_keys.contains(&edge_key(current, next)) {
                split_vertices.push(midpoints[&edge_key(current, next)]);
            }
        }
        face.vertices = split_vertices;
    }
    Ok(midpoints)
}

fn apply_subdivide_once(mesh: &mut Mesh, selected: &Selection) -> Result<IdChanges> {
    let mut changes = IdChanges::default();
    match selected.domain {
        Domain::Edge => {
            let ids: HashSet<_> = selected.ids.iter().copied().collect();
            split_edges(mesh, &ids, &mut changes)?;
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
            let midpoints = split_edges(mesh, &selected_edges, &mut changes)?;
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
                remove_face(mesh, face.id, &mut changes);
                for index in 0..face.vertices.len() {
                    let current = face.vertices[index];
                    let next = face.vertices[(index + 1) % face.vertices.len()];
                    let previous =
                        face.vertices[(index + face.vertices.len() - 1) % face.vertices.len()];
                    let next_mid = midpoints[&edge_key(current, next)];
                    let previous_mid = midpoints[&edge_key(previous, current)];
                    insert_face(
                        mesh,
                        vec![current, next_mid, center_id, previous_mid],
                        face.material_index,
                        &mut changes,
                    )?;
                }
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
}

fn discard_transient_ids(created: &mut Vec<u32>, deleted: &mut Vec<u32>) {
    let created_ids: HashSet<_> = created.iter().copied().collect();
    let deleted_ids: HashSet<_> = deleted.iter().copied().collect();
    let transient: HashSet<_> = created_ids.intersection(&deleted_ids).copied().collect();
    created.retain(|id| !transient.contains(id));
    deleted.retain(|id| !transient.contains(id));
}

fn apply_subdivide(mesh: &mut Mesh, selected: &Selection, cuts: u32) -> Result<IdChanges> {
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
            insert_face(
                mesh,
                vec![
                    face.vertices[index],
                    face.vertices[(index + 1) % face.vertices.len()],
                    center_id,
                ],
                face.material_index,
                &mut changes,
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
        let positions: Vec<_> = face
            .vertices
            .iter()
            .map(|id| {
                mesh.vertex(*id)
                    .map(|vertex| vertex.co)
                    .ok_or_else(|| invalid("face references a missing vertex"))
            })
            .collect::<Result<_>>()?;
        let polygon: Vec<_> = (0..face.vertices.len()).collect();
        let temporary = Mesh::from_positions_and_faces(positions, vec![polygon])
            .map_err(|error| mesh_error(&error))?;
        let triangles = temporary
            .triangulate()
            .map_err(|error| mesh_error(&error))?;
        remove_face(mesh, face.id, &mut changes);
        for triangle in triangles {
            let mut mapped = Vec::with_capacity(3);
            for temporary_id in triangle {
                let index = usize::try_from(temporary_id)
                    .map_err(|_| invalid("temporary triangulation index is out of range"))?;
                mapped.push(
                    *face
                        .vertices
                        .get(index)
                        .ok_or_else(|| invalid("temporary triangulation index is invalid"))?,
                );
            }
            insert_face(mesh, mapped, face.material_index, &mut changes)?;
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
    for face in &mut mesh.faces {
        for id in &mut face.vertices {
            if let Some(replacement) = remap.get(id) {
                *id = *replacement;
            }
        }
        face.vertices.dedup();
        if face.vertices.len() > 1 && face.vertices.first() == face.vertices.last() {
            face.vertices.pop();
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
        first_path.pop();
        first_path.extend(second_path.iter().take(second_path.len().saturating_sub(1)));
        remove_faces(mesh, &[first.id, second.id], &mut changes);
        mesh.edges.retain(|item| {
            if item.id == edge.id {
                changes.delete(Domain::Edge, item.id);
                false
            } else {
                true
            }
        });
        insert_face(mesh, first_path, first.material_index, &mut changes)?;
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
    insert_face(mesh, vertices, 0, &mut changes)?;
    Ok(changes)
}

fn apply_bisect(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
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
                positive.push(intersection);
                negative.push(intersection);
            }
        }
        positive.dedup();
        negative.dedup();
        remove_face(mesh, face.id, &mut changes);
        if clear_side != Some(true) && positive.len() >= 3 {
            insert_face(mesh, positive, face.material_index, &mut changes)?;
        }
        if clear_side != Some(false) && negative.len() >= 3 {
            insert_face(mesh, negative, face.material_index, &mut changes)?;
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
            insert_face(mesh, reflected, face.material_index, &mut changes)?;
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
        Some("byte_color") => Ok("byte_color"),
        Some("bool") => Ok("bool"),
        Some("quaternion") => Ok("quaternion"),
        _ => Err(invalid(
            "attribute type must be float, int, float2, float3, color, byte_color, bool, or quaternion",
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
        "byte_color" => json!([0, 0, 0, 255]),
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
        "color" | "byte_color" | "quaternion" => 4,
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
    match attribute_type {
        "byte_color" => {
            let mut output = Vec::with_capacity(4);
            for component in values {
                let value = component
                    .as_u64()
                    .filter(|value| u8::try_from(*value).is_ok())
                    .ok_or_else(|| {
                        invalid("byte_color components must be integers from 0 to 255")
                    })?;
                output.push(json!(value));
            }
            Ok(Value::Array(output))
        }
        "quaternion" => {
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
        }
        _ => {
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
}

fn attribute_values_mut<'a>(mesh: &'a mut Mesh, name: &str) -> Result<&'a mut Map<String, Value>> {
    mesh.attributes
        .get_mut(name)
        .and_then(Value::as_object_mut)
        .and_then(|attribute| attribute.get_mut("values"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no values object")))
}

fn apply_attribute_create(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid("attribute_create requires a non-empty name"))?;
    if mesh.attributes.contains_key(name) {
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
        name.to_owned(),
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
    let attribute = mesh
        .attributes
        .get(name)
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
    let values = attribute_values_mut(mesh, name)?;
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
    if !mesh.attributes.contains_key(name) {
        return Err(invalid(format!("mesh attribute `{name}` does not exist")));
    }
    let Some(elements) = args.get("elements") else {
        mesh.attributes.remove(name);
        return Ok(IdChanges::default());
    };
    let mut selection_args = Map::new();
    selection_args.insert("elements".to_owned(), elements.clone());
    let selected = required_selection(mesh, &selection_args)?;
    let attribute_domain_name = mesh.attributes[name]
        .get("domain")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("mesh attribute `{name}` has no domain")))?;
    let (_, domain) = attribute_domain(&Value::String(attribute_domain_name.to_owned()))?;
    if selected.domain != domain {
        return Err(invalid(format!(
            "mesh attribute `{name}` uses the {attribute_domain_name} domain"
        )));
    }
    let values = attribute_values_mut(mesh, name)?;
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
        if current_values.contains_key(&key) {
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

fn interpolate_uv_map(source: &Mesh, target: &mut Mesh) -> Result<()> {
    let Some(source_entries) = source.attributes.get("uv_map") else {
        return Ok(());
    };
    let source_entries = source_entries
        .as_array()
        .ok_or_else(|| invalid("uv_map must be an array"))?;
    let entries_by_face: HashMap<u32, &Value> = source_entries
        .iter()
        .filter_map(|entry| Some((u32::try_from(entry.get("face_id")?.as_u64()?).ok()?, entry)))
        .collect();
    let has_pins = source_entries.iter().any(|entry| {
        entry
            .get("pinned")
            .and_then(Value::as_array)
            .is_some_and(|pinned| !pinned.is_empty())
    });
    let mut target_entries = Vec::with_capacity(target.faces.len());
    for face in &target.faces {
        if let Some(existing) = entries_by_face.get(&face.id)
            && existing
                .get("uv")
                .and_then(Value::as_array)
                .is_some_and(|uv| uv.len() == face.vertices.len())
        {
            target_entries.push((*existing).clone());
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
            for source_face in &source.faces {
                let Some(entry) = entries_by_face.get(&source_face.id) else {
                    continue;
                };
                let Some(corners) = entry
                    .get("uv")
                    .and_then(Value::as_array)
                    .filter(|corners| corners.len() == source_face.vertices.len())
                else {
                    continue;
                };
                let pin_values = entry.get("pinned").and_then(Value::as_array);
                for (index, source_vertex) in source_face.vertices.iter().enumerate() {
                    let Some(source_vertex) = source.vertex(*source_vertex) else {
                        continue;
                    };
                    let Some(corner_uv) = corners.get(index) else {
                        continue;
                    };
                    candidates.push((source_vertex.co, corner_uv.clone()));
                    if let Some(is_pinned) = pin_values
                        .and_then(|values| values.get(index))
                        .and_then(Value::as_bool)
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
            let Some(coordinate) = nearest_untyped_attribute_value(&candidates, position) else {
                uv.clear();
                break;
            };
            uv.push(coordinate);
            if has_pins {
                pinned.push(Value::Bool(nearest_pin.is_some_and(|(_, value)| value)));
            }
        }
        if uv.len() != face.vertices.len() {
            continue;
        }
        let mut entry = Map::from_iter([
            ("face_id".to_owned(), json!(face.id)),
            ("uv".to_owned(), Value::Array(uv)),
        ]);
        if has_pins {
            entry.insert("pinned".to_owned(), Value::Array(pinned));
        }
        target_entries.push(Value::Object(entry));
    }
    target
        .attributes
        .insert("uv_map".to_owned(), Value::Array(target_entries));
    Ok(())
}

fn interpolate_attributes(source: &Mesh, target: &mut Mesh) -> Result<()> {
    for (name, source_attribute) in &source.attributes {
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
    interpolate_uv_map(source, target)?;
    Ok(())
}

fn has_interpolatable_attributes(mesh: &Mesh) -> bool {
    mesh.attributes.contains_key("uv_map")
        || mesh.attributes.values().any(|attribute| {
            attribute.as_object().is_some_and(|fields| {
                fields.contains_key("domain") && fields.contains_key("values")
            })
        })
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
    let mut updates = Map::new();
    for id in selected.ids {
        let face = mesh
            .faces
            .iter()
            .find(|face| face.id == id)
            .ok_or_else(|| invalid(format!("face f{id} does not exist")))?;
        let normals = normal_values
            .get(&format!("f{id}"))
            .and_then(Value::as_array)
            .filter(|normals| normals.len() == face.vertices.len())
            .ok_or_else(|| {
                invalid(format!(
                    "custom normals for face f{id} must match its corners"
                ))
            })?;
        let mut corners = Vec::with_capacity(normals.len());
        for normal in normals {
            let normal = vector(normal, "normals")?;
            if normal.length_squared() <= f64::EPSILON {
                return Err(invalid("custom normals must be non-zero"));
            }
            corners.push(json!(normal.normalize().to_array()));
        }
        updates.insert(format!("f{id}"), Value::Array(corners));
    }
    let attribute = mesh
        .attributes
        .entry("custom_normal".to_owned())
        .or_insert_with(|| json!({"domain":"corner","type":"float3","values":{}}));
    let fields = attribute
        .as_object_mut()
        .ok_or_else(|| invalid("custom_normal attribute must be an object"))?;
    if fields.get("domain").and_then(Value::as_str) != Some("corner")
        || fields.get("type").and_then(Value::as_str) != Some("float3")
    {
        return Err(invalid("custom_normal must be a float3 corner attribute"));
    }
    fields
        .entry("values")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| invalid("custom_normal values must be an object"))?
        .extend(updates);
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
    if name != "material_index"
        && mesh
            .attributes
            .get(name)
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
        _ => {}
    }
    let values = selected
        .ids
        .iter()
        .map(|id| (format!("{}{id}", selected.domain.prefix()), value.clone()))
        .collect::<Map<_, _>>();
    let attribute = mesh
        .attributes
        .entry(name.to_owned())
        .or_insert_with(|| json!({ "domain": selected.domain.name(), "values": {} }));
    let attribute_object = attribute
        .as_object_mut()
        .ok_or_else(|| invalid("existing mesh attribute is not a domain attribute object"))?;
    if attribute_object.get("domain").and_then(Value::as_str) != Some(selected.domain.name()) {
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

fn apply_loop_cut(mesh: &mut Mesh, selected: &Selection, cuts: u32) -> Result<IdChanges> {
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
        replacements.push((face.id, face.material_index, pieces));
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
    for (face_id, _, _) in &replacements {
        remove_face(mesh, *face_id, &mut changes);
    }
    for (_, material_index, pieces) in replacements {
        for piece in pieces {
            insert_face(mesh, piece, material_index, &mut changes)?;
        }
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
    for index in 0..count {
        let next = (index + 1) % count;
        insert_face(
            mesh,
            vec![first[index], first[next], aligned[next], aligned[index]],
            0,
            &mut changes,
        )?;
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
        if selected_ids.contains(&face.id) {
            for id in &mut face.vertices {
                if let Some(replacement) = vertex_remap.get(id) {
                    *id = *replacement;
                }
            }
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
    let mut changes = apply_bisect(mesh, &bisect_args)?;
    remove_unused_edges(mesh, &crossing, &mut changes);
    Ok(changes)
}

fn apply_symmetrize(mesh: &mut Mesh, args: &Map<String, Value>) -> Result<IdChanges> {
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
        let mut plane_changes = apply_bisect(mesh, plane_args)?;
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
            (vertices, face.material_index)
        })
        .collect::<Vec<_>>();
    for (vertices, material_index) in reflected_faces {
        insert_face(mesh, vertices, material_index, &mut changes)?;
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
    let attribute_source = (is_topology_operation(operation, args)
        && has_interpolatable_attributes(mesh))
    .then(|| mesh.clone());
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
        "subdivide" => apply_subdivide(
            mesh,
            &required_selection(mesh, args)?,
            positive_count(args, "cuts", 1)?,
        )?,
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
        "bisect" => apply_bisect(mesh, args)?,
        "loop_cut" => apply_loop_cut(
            mesh,
            &required_selection(mesh, args)?,
            positive_count(args, "cuts", 1)?,
        )?,
        "bridge" => apply_bridge(mesh, &required_selection(mesh, args)?)?,
        "knife" => apply_knife(mesh, args)?,
        "symmetrize" => apply_symmetrize(mesh, args)?,
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
        interpolate_attributes(&source, mesh)?;
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
    use serde_json::json;

    use super::apply;
    use crate::geom::{BoxParams, Mesh, PlaneParams};

    #[test]
    fn triangulate_replaces_selected_ngon_with_winding_preserving_triangles() {
        let mut mesh = Mesh::plane(PlaneParams::default()).unwrap();
        let result = apply(
            &mut mesh,
            "triangulate",
            &json!({ "elements": { "domain": "face", "ids": ["f0"] } }),
        )
        .unwrap();

        let triangles = mesh.triangulate().unwrap();
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
        assert_eq!(mesh.triangulate().unwrap().len(), 10);
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
}
