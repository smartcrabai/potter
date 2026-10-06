use std::collections::{BTreeMap, HashMap, HashSet};

use glam::DVec3;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::error::{ErrorCode, PotError, Result};

use super::{Edge, Face, IdCounters, Mesh, Vertex};

const MAX_POINTS: usize = 100_000;
const MAX_ATTRIBUTES: usize = 256;
const MAX_ATTRIBUTE_VALUES: usize = 100_000;
const MAX_ATTRIBUTE_ASSOCIATIONS: usize = 1_000_000;
const MAX_SEGMENTS: u32 = 256;
const MAX_RINGS: u32 = 256;
const MAX_MESH_VERTICES: usize = 100_000;
const MAX_MESH_FACES: usize = 100_000;
const MAX_MESH_EDGES: usize = 200_000;

/// Point-cloud point with a persistent point ID, center, and non-negative sphere radius.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PointCloudPoint {
    /// Stable point ID; attribute maps refer to this as `p<ID>`.
    pub id: u32,
    /// Point center in mesh-local coordinates, in meters.
    pub position: DVec3,
    /// Non-negative radius used when realizing the point as a sphere.
    pub radius: f64,
}

/// Named point-domain attribute. `values` is sparse and keyed by persistent point ID.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PointCloudAttribute {
    /// Attribute domain. Point-cloud inputs use `"points"`.
    pub domain: String,
    /// Attribute values keyed by `p<ID>`.
    pub values: Map<String, Value>,
}

impl Default for PointCloudAttribute {
    fn default() -> Self {
        Self {
            domain: "points".to_owned(),
            values: Map::new(),
        }
    }
}

/// Serializable point-cloud geometry with stable IDs and sparse named attributes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PointCloudData {
    /// Points in storage order; IDs are persistent and need not match their indices.
    pub points: Vec<PointCloudPoint>,
    /// Named point-domain attributes, keyed by attribute name.
    pub attributes: BTreeMap<String, PointCloudAttribute>,
    /// Next assignable point ID; must exceed every ID already in `points`.
    pub next_id: u32,
}

impl PointCloudData {
    /// Validate IDs, coordinates, radii, and point-domain attribute references.
    ///
    /// # Errors
    ///
    /// Returns `INVALID_ARGUMENT` for malformed values and `LIMIT_EXCEEDED` when the
    /// point or attribute counts exceed supported bounds.
    pub fn validate(&self) -> Result<()> {
        self.validate_internal().map(|_| ())
    }

    fn validate_internal(&self) -> Result<HashMap<u32, usize>> {
        if self.points.len() > MAX_POINTS {
            return Err(limit_exceeded(
                "point cloud exceeds the maximum point count",
            ));
        }
        let mut point_indices = HashMap::with_capacity(self.points.len());
        let mut maximum_id = None;
        for (index, point) in self.points.iter().enumerate() {
            if !point.position.is_finite() {
                return Err(PotError::invalid_argument("point position must be finite"));
            }
            if !point.radius.is_finite() || point.radius < 0.0 {
                return Err(PotError::invalid_argument(
                    "point radius must be finite and non-negative",
                ));
            }
            if point_indices.insert(point.id, index).is_some() {
                return Err(PotError::invalid_argument(
                    "point cloud contains a duplicate point ID",
                ));
            }
            maximum_id = Some(maximum_id.map_or(point.id, |current: u32| current.max(point.id)));
        }
        if maximum_id.is_some_and(|maximum| self.next_id <= maximum) {
            return Err(PotError::invalid_argument(
                "point cloud next_id must exceed all allocated point IDs",
            ));
        }
        if self.attributes.len() > MAX_ATTRIBUTES {
            return Err(limit_exceeded(
                "point cloud exceeds the maximum named attribute count",
            ));
        }
        let mut value_count = 0usize;
        for (name, attribute) in &self.attributes {
            if name.is_empty() {
                return Err(PotError::invalid_argument(
                    "point attribute names must not be empty",
                ));
            }
            if attribute.domain != "points" {
                return Err(PotError::invalid_argument(
                    "point-cloud attributes must use the points domain",
                ));
            }
            value_count = value_count
                .checked_add(attribute.values.len())
                .ok_or_else(|| limit_exceeded("point attribute value count overflowed"))?;
            if value_count > MAX_ATTRIBUTE_VALUES {
                return Err(limit_exceeded(
                    "point cloud exceeds the maximum attribute value count",
                ));
            }
            for key in attribute.values.keys() {
                let id = parse_point_key(key).ok_or_else(|| {
                    PotError::invalid_argument("point attribute key must be `p<ID>`")
                })?;
                if !point_indices.contains_key(&id) {
                    return Err(PotError::invalid_argument(
                        "point attribute references a missing point ID",
                    ));
                }
            }
        }
        Ok(point_indices)
    }
}

/// Convert a point cloud to loose mesh vertices for point rendering.
///
/// Persistent point IDs become vertex IDs; named point attributes are retained in the
/// vertex domain. The reserved `point_radius` attribute carries each point's radius.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for malformed cloud data or a conflicting reserved
/// attribute name, and `LIMIT_EXCEEDED` when point or attribute counts exceed bounds.
pub fn to_mesh(cloud: &PointCloudData) -> Result<Mesh> {
    cloud.validate()?;
    if cloud.attributes.contains_key("point_radius") {
        return Err(PotError::invalid_argument(
            "`point_radius` is reserved for point-cloud radii",
        ));
    }

    let mut mesh = Mesh {
        vertices: cloud
            .points
            .iter()
            .map(|point| Vertex {
                id: point.id,
                co: point.position,
            })
            .collect(),
        edges: Vec::new(),
        faces: Vec::new(),
        attributes: Map::new(),
        next_id: IdCounters {
            vertex: cloud.next_id,
            edge: 0,
            face: 0,
        },
    };
    let mut point_radii = Map::new();
    for point in &cloud.points {
        point_radii.insert(format!("v{}", point.id), json!(point.radius));
    }
    mesh.attributes.insert(
        "point_radius".to_owned(),
        json!({"domain":"vertices","values":point_radii}),
    );
    for (name, attribute) in &cloud.attributes {
        let mut values = Map::new();
        for (point_key, value) in &attribute.values {
            let point_id = parse_point_key(point_key)
                .ok_or_else(|| PotError::invalid_argument("point attribute key must be `p<ID>`"))?;
            values.insert(format!("v{point_id}"), value.clone());
        }
        mesh.attributes
            .insert(name.clone(), json!({"domain":"vertices","values":values}));
    }
    mesh.validate().map_err(|error| {
        PotError::invalid_argument(format!(
            "generated point cloud point mesh is invalid: {error}"
        ))
    })?;
    Ok(mesh)
}

/// Realize each point as a deterministic UV sphere and transfer point attributes to vertices.
///
/// `segments` and `rings` must each be in `3..=256`. A zero-radius point becomes one
/// loose mesh vertex at its position, with no faces. Positive-radius points become
/// independent UV spheres. Point-domain values are copied to every generated vertex
/// belonging to that point; the resulting mesh attributes use the `"vertices"` domain.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for malformed cloud data or resolution below three, and
/// `LIMIT_EXCEEDED` when resolution or generated geometry/attribute counts exceed bounds.
pub fn points_to_mesh(cloud: &PointCloudData, segments: u32, rings: u32) -> Result<Mesh> {
    let point_indices = cloud.validate_internal()?;
    if segments < 3 || rings < 3 {
        return Err(PotError::invalid_argument(
            "point cloud sphere segments and rings must each be at least three",
        ));
    }
    if segments > MAX_SEGMENTS || rings > MAX_RINGS {
        return Err(limit_exceeded(
            "point cloud sphere resolution exceeds the maximum",
        ));
    }

    let segments_usize = usize::try_from(segments)
        .map_err(|_| limit_exceeded("sphere segments do not fit this platform"))?;
    let rings_usize = usize::try_from(rings)
        .map_err(|_| limit_exceeded("sphere rings do not fit this platform"))?;
    let mut vertices_per_point = Vec::with_capacity(cloud.points.len());
    let mut vertex_count = 0usize;
    let mut face_count = 0usize;
    let mut edge_bound = 0usize;
    for point in &cloud.points {
        let point_vertices = vertices_per_point_for(point.radius, segments_usize, rings_usize)?;
        vertices_per_point.push(point_vertices);
        vertex_count = checked_total(vertex_count, point_vertices, "vertex")?;
        if point.radius > 0.0 {
            face_count = checked_total(
                face_count,
                segments_usize
                    .checked_mul(rings_usize)
                    .ok_or_else(|| limit_exceeded("sphere face count overflowed"))?,
                "face",
            )?;
            let edges_for_sphere = segments_usize
                .checked_mul(
                    rings_usize
                        .checked_mul(4)
                        .and_then(|value| value.checked_sub(2))
                        .ok_or_else(|| limit_exceeded("sphere edge count overflowed"))?,
                )
                .ok_or_else(|| limit_exceeded("sphere edge count overflowed"))?;
            edge_bound = checked_total(edge_bound, edges_for_sphere, "edge")?;
        }
    }
    if vertex_count > MAX_MESH_VERTICES {
        return Err(limit_exceeded(
            "point cloud sphere conversion exceeds the maximum vertex count",
        ));
    }
    if face_count > MAX_MESH_FACES {
        return Err(limit_exceeded(
            "point cloud sphere conversion exceeds the maximum face count",
        ));
    }
    if edge_bound > MAX_MESH_EDGES {
        return Err(limit_exceeded(
            "point cloud sphere conversion exceeds the maximum edge count",
        ));
    }

    let mut association_count = 0usize;
    for attribute in cloud.attributes.values() {
        for key in attribute.values.keys() {
            let id = parse_point_key(key)
                .ok_or_else(|| PotError::invalid_argument("point attribute key must be `p<ID>`"))?;
            let point_index = *point_indices.get(&id).ok_or_else(|| {
                PotError::invalid_argument("point attribute references a missing point ID")
            })?;
            association_count = checked_total(
                association_count,
                vertices_per_point[point_index],
                "point attribute association",
            )?;
        }
    }
    if association_count > MAX_ATTRIBUTE_ASSOCIATIONS {
        return Err(limit_exceeded(
            "point cloud conversion exceeds the maximum generated attribute value count",
        ));
    }

    let mut vertices = Vec::with_capacity(vertex_count);
    let mut polygons: Vec<Vec<usize>> = Vec::with_capacity(face_count);
    let mut point_vertex_ranges = Vec::with_capacity(cloud.points.len());
    for point in &cloud.points {
        let start = vertices.len();
        if point.radius == 0.0 {
            push_vertex(&mut vertices, point.position)?;
        } else {
            append_sphere(
                point.position,
                point.radius,
                segments,
                rings,
                segments_usize,
                rings_usize,
                &mut vertices,
                &mut polygons,
            )?;
        }
        point_vertex_ranges.push(start..vertices.len());
    }

    let mut faces = Vec::with_capacity(polygons.len());
    let mut edges = Vec::with_capacity(edge_bound);
    let mut edge_keys = HashSet::with_capacity(edge_bound);
    for (index, polygon) in polygons.into_iter().enumerate() {
        let id = u32::try_from(index)
            .map_err(|_| limit_exceeded("generated face ID exceeds the supported range"))?;
        let face_vertices = polygon
            .into_iter()
            .map(|vertex_index| {
                u32::try_from(vertex_index)
                    .map_err(|_| limit_exceeded("generated vertex ID exceeds the supported range"))
            })
            .collect::<Result<Vec<_>>>()?;
        for edge_index in 0..face_vertices.len() {
            let first = face_vertices[edge_index];
            let second = face_vertices[(edge_index + 1) % face_vertices.len()];
            let key = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            if edge_keys.insert(key) {
                let edge_id = u32::try_from(edges.len())
                    .map_err(|_| limit_exceeded("generated edge ID exceeds the supported range"))?;
                edges.push(Edge {
                    id: edge_id,
                    vertices: [first, second],
                });
            }
        }
        faces.push(Face {
            id,
            vertices: face_vertices,
            material_index: 0,
        });
    }

    let mut mesh = Mesh {
        vertices,
        edges,
        faces,
        attributes: Map::new(),
        next_id: IdCounters {
            vertex: u32::try_from(vertex_count)
                .map_err(|_| limit_exceeded("generated vertex ID count exceeds u32"))?,
            edge: u32::try_from(edge_keys.len())
                .map_err(|_| limit_exceeded("generated edge ID count exceeds u32"))?,
            face: u32::try_from(face_count)
                .map_err(|_| limit_exceeded("generated face ID count exceeds u32"))?,
        },
    };
    for (name, attribute) in &cloud.attributes {
        let mut values = Map::new();
        for (key, value) in &attribute.values {
            let point_id = parse_point_key(key)
                .ok_or_else(|| PotError::invalid_argument("point attribute key must be `p<ID>`"))?;
            let point_index = *point_indices.get(&point_id).ok_or_else(|| {
                PotError::invalid_argument("point attribute references a missing point ID")
            })?;
            for vertex_index in point_vertex_ranges[point_index].clone() {
                let vertex_id = u32::try_from(vertex_index)
                    .map_err(|_| limit_exceeded("generated vertex ID exceeds u32"))?;
                values.insert(format!("v{vertex_id}"), value.clone());
            }
        }
        mesh.attributes.insert(
            name.clone(),
            json!({
                "domain": "vertices",
                "values": values,
            }),
        );
    }
    mesh.validate().map_err(|error| {
        PotError::invalid_argument(format!("generated point cloud mesh is invalid: {error}"))
    })?;
    Ok(mesh)
}

fn vertices_per_point_for(radius: f64, segments: usize, rings: usize) -> Result<usize> {
    if radius == 0.0 {
        return Ok(1);
    }
    segments
        .checked_mul(
            rings
                .checked_sub(1)
                .ok_or_else(|| PotError::invalid_argument("sphere rings must be at least three"))?,
        )
        .and_then(|count| count.checked_add(2))
        .ok_or_else(|| limit_exceeded("sphere vertex count overflowed"))
}

fn push_vertex(vertices: &mut Vec<Vertex>, co: DVec3) -> Result<usize> {
    if !co.is_finite() {
        return Err(PotError::invalid_argument(
            "point cloud sphere conversion produced a non-finite position",
        ));
    }
    let index = vertices.len();
    let id = u32::try_from(index)
        .map_err(|_| limit_exceeded("generated vertex ID exceeds the supported range"))?;
    vertices.push(Vertex { id, co });
    Ok(index)
}

fn append_sphere(
    center: DVec3,
    radius: f64,
    segments: u32,
    rings: u32,
    segments_usize: usize,
    rings_usize: usize,
    vertices: &mut Vec<Vertex>,
    polygons: &mut Vec<Vec<usize>>,
) -> Result<()> {
    let start = vertices.len();
    push_vertex(vertices, center + DVec3::Z * radius)?;
    for ring in 1..rings {
        let latitude = std::f64::consts::PI * f64::from(ring) / f64::from(rings);
        let z = radius * latitude.cos();
        let radial = radius * latitude.sin();
        for segment in 0..segments {
            let longitude = std::f64::consts::TAU * f64::from(segment) / f64::from(segments);
            push_vertex(
                vertices,
                center + DVec3::new(radial * longitude.cos(), radial * longitude.sin(), z),
            )?;
        }
    }
    let bottom = start + segments_usize * (rings_usize - 1) + 1;
    push_vertex(vertices, center - DVec3::Z * radius)?;
    let ring_vertex =
        |ring: usize, segment: usize| start + 1 + ring * segments_usize + segment % segments_usize;
    for segment in 0..segments_usize {
        polygons.push(vec![
            start,
            ring_vertex(0, segment),
            ring_vertex(0, segment + 1),
        ]);
    }
    for ring in 0..rings_usize - 2 {
        for segment in 0..segments_usize {
            polygons.push(vec![
                ring_vertex(ring, segment),
                ring_vertex(ring + 1, segment),
                ring_vertex(ring + 1, segment + 1),
                ring_vertex(ring, segment + 1),
            ]);
        }
    }
    for segment in 0..segments_usize {
        polygons.push(vec![
            ring_vertex(rings_usize - 2, segment),
            bottom,
            ring_vertex(rings_usize - 2, segment + 1),
        ]);
    }
    Ok(())
}

fn parse_point_key(key: &str) -> Option<u32> {
    let digits = key.strip_prefix('p')?;
    if digits.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        || (digits.len() > 1 && digits.starts_with('0'))
    {
        return None;
    }
    digits.parse::<u32>().ok()
}

fn checked_total(current: usize, increment: usize, kind: &str) -> Result<usize> {
    current
        .checked_add(increment)
        .ok_or_else(|| limit_exceeded(format!("generated {kind} count overflowed")))
}

fn limit_exceeded(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

#[cfg(test)]
mod tests {
    use glam::DVec3;
    use proptest::prelude::*;
    use serde_json::{Value, json};
    use std::collections::BTreeMap;

    use super::{PointCloudAttribute, PointCloudData, PointCloudPoint, points_to_mesh};

    fn cloud(points: Vec<PointCloudPoint>, values: Vec<Value>) -> PointCloudData {
        let mut attribute_values = serde_json::Map::new();
        for (point, value) in points.iter().zip(values) {
            attribute_values.insert(format!("p{}", point.id), value);
        }
        let mut attributes = BTreeMap::new();
        attributes.insert(
            "weight".to_owned(),
            PointCloudAttribute {
                domain: "points".to_owned(),
                values: attribute_values,
            },
        );
        let next_id = points
            .iter()
            .map(|point| point.id)
            .max()
            .map_or(0, |id| id + 1);
        PointCloudData {
            points,
            attributes,
            next_id,
        }
    }

    #[test]
    fn generated_sphere_bounds_follow_point_radius() -> Result<(), Box<dyn std::error::Error>> {
        let center = DVec3::new(2.0, -3.0, 5.0);
        let cloud = PointCloudData {
            points: vec![PointCloudPoint {
                id: 0,
                position: center,
                radius: 2.0,
            }],
            next_id: 1,
            ..PointCloudData::default()
        };

        let mesh = points_to_mesh(&cloud, 8, 4)?;
        let bounds = mesh
            .bounds()
            .ok_or_else(|| std::io::Error::other("generated point mesh has no bounds"))?;
        assert!((bounds.min.x - (center.x - 2.0)).abs() < 1.0e-12);
        assert!((bounds.min.y - (center.y - 2.0)).abs() < 1.0e-12);
        assert!((bounds.min.z - (center.z - 2.0)).abs() < 1.0e-12);
        assert!((bounds.max.x - (center.x + 2.0)).abs() < 1.0e-12);
        assert!((bounds.max.y - (center.y + 2.0)).abs() < 1.0e-12);
        assert!((bounds.max.z - (center.z + 2.0)).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn point_attributes_are_associated_with_every_generated_vertex()
    -> Result<(), Box<dyn std::error::Error>> {
        let cloud = cloud(
            vec![
                PointCloudPoint {
                    id: 4,
                    position: DVec3::new(-3.0, 1.0, 0.0),
                    radius: 0.5,
                },
                PointCloudPoint {
                    id: 9,
                    position: DVec3::new(4.0, -2.0, 1.0),
                    radius: 1.0,
                },
            ],
            vec![json!(17), json!(29)],
        );

        let mesh = points_to_mesh(&cloud, 6, 3)?;
        let attribute = mesh
            .attributes
            .get("weight")
            .ok_or_else(|| std::io::Error::other("generated point mesh has no weight attribute"))?;
        assert_eq!(
            attribute.get("domain").and_then(Value::as_str),
            Some("vertices")
        );
        let values = attribute
            .get("values")
            .and_then(Value::as_object)
            .ok_or_else(|| std::io::Error::other("generated point weight values are malformed"))?;
        let vertices_per_point = 6 * (3 - 1) + 2;
        for vertex in &mesh.vertices {
            let point_index = vertex.id as usize / vertices_per_point;
            let key = format!("v{}", vertex.id);
            assert_eq!(
                values.get(&key),
                Some(&json!(if point_index == 0 { 17 } else { 29 }))
            );
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn generated_sphere_meshes_contain_their_centers(
            x in -100.0_f64..100.0,
            y in -100.0_f64..100.0,
            z in -100.0_f64..100.0,
            radius in 0.01_f64..10.0,
            segments in 3_u32..12,
            rings in 3_u32..8,
        ) {
            let center = DVec3::new(x, y, z);
            let cloud = PointCloudData {
                points: vec![PointCloudPoint {
                    id: 0,
                    position: center,
                    radius,
                }],
                next_id: 1,
                ..PointCloudData::default()
            };
            let mesh = points_to_mesh(&cloud, segments, rings)
                .unwrap_or_else(|error| panic!("generated point cloud mesh is valid: {error}"));
            let bounds = mesh
                .bounds()
                .unwrap_or_else(|| panic!("generated point cloud mesh has bounds"));
            prop_assert!(bounds.contains(center));
            prop_assert!(mesh.validate().is_ok());

        }
    }
}
