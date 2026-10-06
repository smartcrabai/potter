//! Persistent polygon meshes, Blender-default primitives, triangulation, and bounds.
//!
//! The public [`Mesh`] stores stable `u32` IDs independently for vertices, edges, and
//! polygon faces. References in edges/faces use vertex IDs, never vector positions.
//! [`Mesh::validate`] checks this invariant and [`Mesh::triangulate`] returns vertex
//! IDs while preserving each polygon's winding. Primitive parameter structs implement
//! [`Default`] with Blender's standard add-primitive settings; override only the fields
//! that differ. [`Mesh::bounds`] includes loose vertices and returns `None` for an
//! empty mesh.

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
};

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

pub mod boolean;
pub mod curve;
pub mod edit;
pub mod lattice;
pub mod mesh_cache;
pub mod metaball;
pub mod modifiers;
pub mod pointcloud;
pub mod remesh;
pub mod sculpt;
pub mod text;
pub mod vdb;
pub mod volume;

/// Stable vertex record. `id` is persistent; its position in [`Mesh::vertices`] is not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vertex {
    /// Persistent ID within the vertex domain.
    pub id: u32,
    /// Position in mesh-local coordinates, in meters.
    pub co: DVec3,
}

/// Stable edge record referencing its endpoints by persistent vertex ID.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edge {
    /// Persistent ID within the edge domain.
    pub id: u32,
    /// Endpoint vertex IDs. Endpoint ordering has no geometric meaning.
    #[serde(rename = "v")]
    pub vertices: [u32; 2],
}

/// Stable polygon record. Polygon vertex order defines its winding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Face {
    /// Persistent ID within the face domain.
    pub id: u32,
    /// Ordered persistent vertex IDs; polygons have at least three vertices.
    #[serde(rename = "v")]
    pub vertices: Vec<u32>,
    /// Material slot index, zero by default.
    pub material_index: u32,
}

/// Next assignable ID in each independent element domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdCounters {
    /// Next vertex ID.
    pub vertex: u32,
    /// Next edge ID.
    pub edge: u32,
    /// Next face ID.
    pub face: u32,
}

/// Polygon mesh whose topology and element identities survive index changes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mesh {
    /// Vertices in storage order (not ID order).
    pub vertices: Vec<Vertex>,
    /// Edges in storage order (not ID order).
    pub edges: Vec<Edge>,
    /// Polygon faces in storage order (not ID order).
    pub faces: Vec<Face>,
    /// Named mesh-domain attributes, empty for the built-in primitives.
    #[serde(default)]
    pub attributes: Map<String, Value>,
    /// Independent persistent-ID allocation counters.
    pub next_id: IdCounters,
}

/// Axis-aligned bounds of mesh-local vertex positions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Aabb {
    /// Component-wise minimum.
    pub min: DVec3,
    /// Component-wise maximum.
    pub max: DVec3,
}

impl Aabb {
    /// Bounds size, equivalent to `max - min`.
    #[must_use]
    pub fn size(self) -> DVec3 {
        self.max - self.min
    }

    /// Whether the inclusive bounds contain `point`.
    #[must_use]
    pub fn contains(self, point: DVec3) -> bool {
        point.cmpge(self.min).all() && point.cmple(self.max).all()
    }
}

/// Invalid mesh data, primitive arguments, or exhausted element IDs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshError {
    /// A parameter does not satisfy its documented domain.
    InvalidParameter(&'static str),
    /// Mesh topology or stored IDs are inconsistent.
    InvalidTopology(&'static str),
    /// The next ID cannot be incremented without overflowing `u32`.
    IdExhausted,
    /// Polygon could not be triangulated (usually because it is degenerate).
    TriangulationFailed(u32),
}

impl fmt::Display for MeshError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(message) => {
                write!(formatter, "invalid primitive parameter: {message}")
            }
            Self::InvalidTopology(message) => write!(formatter, "invalid mesh topology: {message}"),
            Self::IdExhausted => formatter.write_str("mesh element ID space exhausted"),
            Self::TriangulationFailed(id) => write!(formatter, "could not triangulate face {id}"),
        }
    }
}

impl Error for MeshError {}

/// Cube with independent full side lengths; Blender's default cube has size `(2, 2, 2)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BoxParams {
    /// Positive full width, depth, and height.
    pub size: DVec3,
}

impl Default for BoxParams {
    fn default() -> Self {
        Self {
            size: DVec3::splat(2.0),
        }
    }
}

/// UV sphere parameters (Blender defaults: 32 segments, 16 rings, radius 1).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UvSphereParams {
    /// Longitudinal segments; at least 3.
    pub segments: u32,
    /// Latitude intervals from pole to pole; at least 3.
    pub ring_count: u32,
    /// Positive sphere radius.
    pub radius: f64,
}

impl Default for UvSphereParams {
    fn default() -> Self {
        Self {
            segments: 32,
            ring_count: 16,
            radius: 1.0,
        }
    }
}

/// Cap fill modes used by Blender's cylinder and cone operators.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EndFillType {
    /// Leave the ends open.
    Nothing,
    /// Fill each end with one polygon.
    #[default]
    Ngon,
    /// Fill each end with a triangle fan.
    #[serde(rename = "TRIFAN")]
    TriFan,
}

/// Cylinder parameters (Blender defaults: 32 vertices, radius 1, depth 2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CylinderParams {
    /// Vertices around each end; at least 3.
    pub vertices: u32,
    /// Positive radius.
    pub radius: f64,
    /// Positive end-to-end depth.
    pub depth: f64,
    /// End cap fill mode.
    pub end_fill_type: EndFillType,
}

impl Default for CylinderParams {
    fn default() -> Self {
        Self {
            vertices: 32,
            radius: 1.0,
            depth: 2.0,
            end_fill_type: EndFillType::Ngon,
        }
    }
}

/// Plane parameters (Blender default full side length is 2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlaneParams {
    /// Positive full side length, centered at the origin in XY.
    pub size: f64,
}

impl Default for PlaneParams {
    fn default() -> Self {
        Self { size: 2.0 }
    }
}

/// Cone/truncated-cone parameters (Blender defaults: bottom radius 1, top 0, depth 2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConeParams {
    /// Vertices around the base; at least 3.
    pub vertices: u32,
    /// Non-negative radius at `z = -depth / 2`.
    pub radius1: f64,
    /// Non-negative radius at `z = depth / 2`.
    pub radius2: f64,
    /// Positive end-to-end depth. At least one radius must be positive.
    pub depth: f64,
    /// End cap fill mode.
    pub end_fill_type: EndFillType,
}

impl Default for ConeParams {
    fn default() -> Self {
        Self {
            vertices: 32,
            radius1: 1.0,
            radius2: 0.0,
            depth: 2.0,
            end_fill_type: EndFillType::Ngon,
        }
    }
}

/// Torus radius interpretation used by Blender's add-torus operator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TorusMode {
    /// `major_radius` and `minor_radius` describe the centerline and tube radius.
    #[default]
    MajorMinor,
    /// `abso_major_rad` and `abso_minor_rad` describe exterior and interior radii.
    ExtInt,
}

/// Torus parameters (Blender defaults: 48 major, 12 minor segments, radii 1 and 0.25).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TorusParams {
    /// Main ring radius in `MAJOR_MINOR` mode.
    pub major_radius: f64,
    /// Tube radius in `MAJOR_MINOR` mode.
    pub minor_radius: f64,
    /// Exterior radius in `EXT_INT` mode.
    pub abso_major_rad: f64,
    /// Interior radius in `EXT_INT` mode.
    pub abso_minor_rad: f64,
    /// Segments around the main ring; Blender range 3–256.
    pub major_segments: u32,
    /// Segments around the tube; Blender range 3–256.
    pub minor_segments: u32,
    /// Select absolute or centerline radius interpretation.
    pub mode: TorusMode,
}

impl Default for TorusParams {
    fn default() -> Self {
        Self {
            major_radius: 1.0,
            minor_radius: 0.25,
            abso_major_rad: 1.25,
            abso_minor_rad: 0.75,
            major_segments: 48,
            minor_segments: 12,
            mode: TorusMode::MajorMinor,
        }
    }
}

/// Icosphere parameters (Blender defaults: subdivision level 2, radius 1).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IcosphereParams {
    /// Recursive subdivision level, from 1 through 8.
    pub subdivisions: u32,
    /// Positive sphere radius.
    pub radius: f64,
}

impl Default for IcosphereParams {
    fn default() -> Self {
        Self {
            subdivisions: 2,
            radius: 1.0,
        }
    }
}

/// Circle fill modes used by Blender's add-circle operator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CircleFillType {
    /// No face (Blender default).
    #[default]
    Nothing,
    /// One polygon spanning the ring.
    Ngon,
    /// A triangle fan with a center vertex.
    #[serde(rename = "TRIFAN")]
    TriFan,
}

/// Circle parameters (Blender defaults: 32 vertices, radius 1, no fill).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CircleParams {
    /// Ring vertex count; at least 3.
    pub vertices: u32,
    /// Positive radius.
    pub radius: f64,
    /// Face fill choice.
    pub fill_type: CircleFillType,
}

impl Default for CircleParams {
    fn default() -> Self {
        Self {
            vertices: 32,
            radius: 1.0,
            fill_type: CircleFillType::Nothing,
        }
    }
}

/// XY grid parameters (Blender defaults: 10 subdivisions per axis, size 2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GridParams {
    /// Full X extent; positive.
    pub size_x: f64,
    /// Full Y extent; positive.
    pub size_y: f64,
    /// Vertex subdivisions on X; at least 2.
    pub x_subdivisions: u32,
    /// Vertex subdivisions on Y; at least 2.
    pub y_subdivisions: u32,
}

impl Default for GridParams {
    fn default() -> Self {
        Self {
            size_x: 2.0,
            size_y: 2.0,
            x_subdivisions: 10,
            y_subdivisions: 10,
        }
    }
}

impl Mesh {
    /// Construct an empty mesh with all next IDs set to zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a mesh from positions and polygon vertex indices.
    ///
    /// IDs are assigned sequentially from zero in vertex, unique-edge, and face
    /// storage order. Every polygon boundary edge is included exactly once.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite positions, invalid polygon indices/topology,
    /// or exhausted element IDs.
    pub fn from_positions_and_faces(
        positions: Vec<DVec3>,
        polygons: Vec<Vec<usize>>,
    ) -> Result<Self, MeshError> {
        if positions.len() > u32::MAX as usize || polygons.len() > u32::MAX as usize {
            return Err(MeshError::IdExhausted);
        }
        let vertex_count = positions.len();
        let mut mesh = Self::new();
        mesh.vertices.reserve(vertex_count);
        for position in positions {
            mesh.insert_vertex(position)?;
        }
        let faces = polygons.into_iter().map(|polygon| {
            let mut ids = Vec::with_capacity(polygon.len());
            for index in polygon {
                if index >= vertex_count {
                    return Err(MeshError::InvalidTopology(
                        "polygon index is outside the position array",
                    ));
                }
                ids.push(u32::try_from(index).map_err(|_| MeshError::IdExhausted)?);
            }
            Ok((ids, 0))
        });
        mesh.insert_faces(faces)?;
        Ok(mesh)
    }

    /// Add a vertex and return its persistent ID.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite coordinates or exhausted vertex IDs.
    pub fn insert_vertex(&mut self, co: DVec3) -> Result<u32, MeshError> {
        if !co.is_finite() {
            return Err(MeshError::InvalidParameter(
                "vertex coordinates must be finite",
            ));
        }
        let (id, next) = allocate(self.next_id.vertex)?;
        self.vertices.push(Vertex { id, co });
        self.next_id.vertex = next;
        Ok(id)
    }

    /// Add an edge between distinct existing vertex IDs and return its persistent ID.
    ///
    /// # Errors
    ///
    /// Returns an error for identical/missing endpoints, a duplicate edge, or exhausted IDs.
    pub fn insert_edge(&mut self, vertices: [u32; 2]) -> Result<u32, MeshError> {
        if vertices[0] == vertices[1] {
            return Err(MeshError::InvalidTopology(
                "an edge must have distinct endpoints",
            ));
        }
        if !self.vertices.iter().any(|vertex| vertex.id == vertices[0])
            || !self.vertices.iter().any(|vertex| vertex.id == vertices[1])
        {
            return Err(MeshError::InvalidTopology(
                "edge references a missing vertex",
            ));
        }
        let key = edge_key(vertices[0], vertices[1]);
        if self
            .edges
            .iter()
            .any(|edge| edge_key(edge.vertices[0], edge.vertices[1]) == key)
        {
            return Err(MeshError::InvalidTopology(
                "duplicate edge between the same vertices",
            ));
        }
        let (id, next) = allocate(self.next_id.edge)?;
        self.edges.push(Edge { id, vertices });
        self.next_id.edge = next;
        Ok(id)
    }

    /// Add a polygon, creating any missing boundary edges, and return its persistent ID.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid vertices, missing/duplicate edges, or exhausted IDs.
    pub fn insert_face(
        &mut self,
        vertices: Vec<u32>,
        material_index: u32,
    ) -> Result<u32, MeshError> {
        validate_polygon_vertices(&vertices)?;
        for id in &vertices {
            if !self.vertices.iter().any(|vertex| vertex.id == *id) {
                return Err(MeshError::InvalidTopology(
                    "face references a missing vertex",
                ));
            }
        }
        let (face_id, next_face) = allocate(self.next_id.face)?;
        let mut missing = Vec::new();
        for pair in cyclic_pairs(&vertices) {
            let key = edge_key(pair[0], pair[1]);
            if !self
                .edges
                .iter()
                .any(|edge| edge_key(edge.vertices[0], edge.vertices[1]) == key)
                && !missing.contains(&key)
            {
                missing.push(key);
            }
        }
        let edge_count = u32::try_from(missing.len()).map_err(|_| MeshError::IdExhausted)?;
        let _next_edge = self
            .next_id
            .edge
            .checked_add(edge_count)
            .ok_or(MeshError::IdExhausted)?;
        self.edges.reserve(missing.len());
        for [first, second] in cyclic_pairs(&vertices) {
            let key = edge_key(first, second);
            if missing.contains(&key) {
                let (id, next) = allocate(self.next_id.edge)?;
                self.edges.push(Edge {
                    id,
                    vertices: [first, second],
                });
                self.next_id.edge = next;
                missing.retain(|candidate| *candidate != key);
            }
        }
        self.faces.push(Face {
            id: face_id,
            vertices,
            material_index,
        });
        self.next_id.face = next_face;
        Ok(face_id)
    }

    /// Add a batch of polygons while reusing temporary vertex and edge indexes.
    ///
    /// # Errors
    ///
    /// Returns an error when a polygon is invalid, references a missing vertex, or
    /// exhausts an element ID. Faces and edges inserted before an error remain present.
    pub(crate) fn insert_faces(
        &mut self,
        faces: impl IntoIterator<Item = Result<(Vec<u32>, u32), MeshError>>,
    ) -> Result<(), MeshError> {
        let faces = faces.into_iter();
        let (minimum, maximum) = faces.size_hint();
        self.faces.reserve(maximum.unwrap_or(minimum));
        let vertex_ids = self
            .vertices
            .iter()
            .map(|vertex| vertex.id)
            .collect::<HashSet<_>>();
        let mut edge_keys = self
            .edges
            .iter()
            .map(|edge| edge_key(edge.vertices[0], edge.vertices[1]))
            .collect::<HashSet<_>>();

        for face in faces {
            let (vertices, material_index) = face?;
            validate_polygon_vertices(&vertices)?;
            if vertices.iter().any(|id| !vertex_ids.contains(id)) {
                return Err(MeshError::InvalidTopology(
                    "face references a missing vertex",
                ));
            }
            let (face_id, next_face) = allocate(self.next_id.face)?;
            let new_edge_count = cyclic_pairs(&vertices)
                .filter(|pair| !edge_keys.contains(&edge_key(pair[0], pair[1])))
                .count();
            let edge_count = u32::try_from(new_edge_count).map_err(|_| MeshError::IdExhausted)?;
            let next_edge = self
                .next_id
                .edge
                .checked_add(edge_count)
                .ok_or(MeshError::IdExhausted)?;

            for [first, second] in cyclic_pairs(&vertices) {
                let key = edge_key(first, second);
                if edge_keys.insert(key) {
                    let (id, next) = allocate(self.next_id.edge)?;
                    self.edges.push(Edge {
                        id,
                        vertices: [first, second],
                    });
                    self.next_id.edge = next;
                }
            }
            debug_assert_eq!(self.next_id.edge, next_edge);
            self.faces.push(Face {
                id: face_id,
                vertices,
                material_index,
            });
            self.next_id.face = next_face;
        }
        Ok(())
    }

    /// Look up a vertex by persistent ID, independent of storage order.
    #[must_use]
    pub fn vertex(&self, id: u32) -> Option<&Vertex> {
        self.vertices.iter().find(|vertex| vertex.id == id)
    }

    /// Check finite positions, unique IDs, valid references, polygons, and ID counters.
    ///
    /// # Errors
    ///
    /// Returns an error when any stored ID, coordinate, or topology reference is invalid.
    pub fn validate(&self) -> Result<(), MeshError> {
        let vertex_ids = validate_domain(
            self.vertices.iter().map(|item| item.id),
            self.next_id.vertex,
            "vertex",
        )?;
        let _edge_ids = validate_domain(
            self.edges.iter().map(|item| item.id),
            self.next_id.edge,
            "edge",
        )?;
        let _face_ids = validate_domain(
            self.faces.iter().map(|item| item.id),
            self.next_id.face,
            "face",
        )?;
        for vertex in &self.vertices {
            if !vertex.co.is_finite() {
                return Err(MeshError::InvalidTopology(
                    "vertex coordinate is not finite",
                ));
            }
        }
        for edge in &self.edges {
            if edge.vertices[0] == edge.vertices[1] {
                return Err(MeshError::InvalidTopology(
                    "an edge has identical endpoints",
                ));
            }
            if !vertex_ids.contains(&edge.vertices[0]) || !vertex_ids.contains(&edge.vertices[1]) {
                return Err(MeshError::InvalidTopology(
                    "edge references a missing vertex",
                ));
            }
        }
        let mut seen_edges = HashSet::with_capacity(self.edges.len());
        for edge in &self.edges {
            if !seen_edges.insert(edge_key(edge.vertices[0], edge.vertices[1])) {
                return Err(MeshError::InvalidTopology(
                    "duplicate edge between the same vertices",
                ));
            }
        }
        for face in &self.faces {
            validate_polygon_vertices(&face.vertices)?;
            for id in &face.vertices {
                if !vertex_ids.contains(id) {
                    return Err(MeshError::InvalidTopology(
                        "face references a missing vertex",
                    ));
                }
            }
            for pair in cyclic_pairs(&face.vertices) {
                if !seen_edges.contains(&edge_key(pair[0], pair[1])) {
                    return Err(MeshError::InvalidTopology("face boundary edge is missing"));
                }
            }
        }
        Ok(())
    }

    /// Return bounds over all vertex coordinates, including loose vertices.
    #[must_use]
    pub fn bounds(&self) -> Option<Aabb> {
        let first = self.vertices.first()?.co;
        let (min, max) = self
            .vertices
            .iter()
            .skip(1)
            .fold((first, first), |(min, max), vertex| {
                (min.min(vertex.co), max.max(vertex.co))
            });
        Some(Aabb { min, max })
    }

    /// Triangulate polygons into vertex-ID triples, preserving face winding.
    ///
    /// Each simple, non-degenerate polygon produces exactly `n - 2` triangles.
    /// Loose vertices and edges do not appear in the result.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid mesh topology, degenerate polygons, or ID exhaustion.
    pub fn triangulate(&self) -> Result<Vec<[u32; 3]>, MeshError> {
        self.validate()?;
        let capacity = self.faces.iter().try_fold(0_usize, |capacity, face| {
            capacity
                .checked_add(face.vertices.len() - 2)
                .ok_or(MeshError::IdExhausted)
        })?;
        let positions: HashMap<_, _> = self
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co))
            .collect();
        let mut result = Vec::with_capacity(capacity);
        for face in &self.faces {
            result.extend(triangulate_face(face, &positions)?);
        }
        Ok(result)
    }

    /// Create a Blender-default or customized box mesh.
    ///
    /// # Errors
    ///
    /// Returns an error when any box dimension is not finite and positive.
    pub fn box_mesh(params: BoxParams) -> Result<Self, MeshError> {
        if !positive_vec3(params.size) {
            return Err(MeshError::InvalidParameter(
                "box dimensions must be finite and positive",
            ));
        }
        let half = params.size * 0.5;
        let positions = vec![
            DVec3::new(-half.x, -half.y, -half.z),
            DVec3::new(half.x, -half.y, -half.z),
            DVec3::new(half.x, half.y, -half.z),
            DVec3::new(-half.x, half.y, -half.z),
            DVec3::new(-half.x, -half.y, half.z),
            DVec3::new(half.x, -half.y, half.z),
            DVec3::new(half.x, half.y, half.z),
            DVec3::new(-half.x, half.y, half.z),
        ];
        Self::from_positions_and_faces(
            positions,
            vec![
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![1, 2, 6, 5],
                vec![2, 3, 7, 6],
                vec![3, 0, 4, 7],
            ],
        )
    }

    /// Create a Blender-default or customized UV sphere.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid radii/ring counts or exhausted mesh IDs.
    #[expect(
        clippy::cast_precision_loss,
        reason = "indices derive from u32 counts and are exactly representable in f64"
    )]
    pub fn uv_sphere(params: UvSphereParams) -> Result<Self, MeshError> {
        validate_radial_count(params.segments, "UV sphere segments")?;
        if params.ring_count < 3 {
            return Err(MeshError::InvalidParameter(
                "UV sphere requires at least three rings",
            ));
        }
        positive(params.radius, "UV sphere radius")?;
        let segments = usize::try_from(params.segments).map_err(|_| MeshError::IdExhausted)?;
        let rings = usize::try_from(params.ring_count).map_err(|_| MeshError::IdExhausted)?;
        let intermediate_rings = rings.checked_sub(1).ok_or(MeshError::IdExhausted)?;
        let vertex_count = segments
            .checked_mul(intermediate_rings)
            .and_then(|n| n.checked_add(2))
            .ok_or(MeshError::IdExhausted)?;
        if vertex_count > u32::MAX as usize {
            return Err(MeshError::IdExhausted);
        }
        let mut positions = Vec::with_capacity(vertex_count);
        positions.push(DVec3::new(0.0, 0.0, params.radius));
        for ring in 1..rings {
            let latitude = std::f64::consts::PI * ring as f64 / rings as f64;
            let z = params.radius * latitude.cos();
            let radial = params.radius * latitude.sin();
            for segment in 0..segments {
                let longitude = std::f64::consts::FRAC_PI_2
                    - std::f64::consts::TAU * segment as f64 / segments as f64;
                positions.push(DVec3::new(
                    radial * longitude.cos(),
                    radial * longitude.sin(),
                    z,
                ));
            }
        }
        let bottom = positions.len();
        positions.push(DVec3::new(0.0, 0.0, -params.radius));
        let ring_id = |ring: usize, segment: usize| 1 + ring * segments + segment % segments;
        let mut polygons = Vec::with_capacity(segments * rings);
        for segment in 0..segments {
            polygons.push(vec![0, ring_id(0, segment + 1), ring_id(0, segment)]);
        }
        for ring in 0..rings - 2 {
            for segment in 0..segments {
                polygons.push(vec![
                    ring_id(ring, segment),
                    ring_id(ring, segment + 1),
                    ring_id(ring + 1, segment + 1),
                    ring_id(ring + 1, segment),
                ]);
            }
        }
        for segment in 0..segments {
            polygons.push(vec![
                ring_id(rings - 2, segment),
                ring_id(rings - 2, segment + 1),
                bottom,
            ]);
        }
        Self::from_positions_and_faces(positions, polygons)
    }
    /// Create a Blender-default or customized capped cylinder.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions/vertex counts or exhausted mesh IDs.
    #[expect(
        clippy::cast_precision_loss,
        reason = "indices derive from u32 counts and are exactly representable in f64"
    )]
    pub fn cylinder(params: CylinderParams) -> Result<Self, MeshError> {
        validate_radial_count(params.vertices, "cylinder vertices")?;
        positive(params.radius, "cylinder radius")?;
        positive(params.depth, "cylinder depth")?;
        let count = params.vertices as usize;
        let fan_caps = params.end_fill_type == EndFillType::TriFan;
        let mut positions = Vec::with_capacity(count * 2 + if fan_caps { 2 } else { 0 });
        let bottom_center = if fan_caps {
            let center = positions.len();
            positions.push(DVec3::new(0.0, 0.0, -params.depth * 0.5));
            Some(center)
        } else {
            None
        };
        let top_center = if fan_caps {
            let center = positions.len();
            positions.push(DVec3::new(0.0, 0.0, params.depth * 0.5));
            Some(center)
        } else {
            None
        };
        let bottom_start = positions.len();
        for z in [-params.depth * 0.5, params.depth * 0.5] {
            for index in 0..count {
                let angle = std::f64::consts::FRAC_PI_2
                    - std::f64::consts::TAU * index as f64 / count as f64;
                positions.push(DVec3::new(
                    params.radius * angle.cos(),
                    params.radius * angle.sin(),
                    z,
                ));
            }
        }
        let top_start = bottom_start + count;
        let mut polygons = Vec::with_capacity(count * 3 + 2);
        for index in 0..count {
            let next = (index + 1) % count;
            polygons.push(vec![
                bottom_start + index,
                top_start + index,
                top_start + next,
                bottom_start + next,
            ]);
            match params.end_fill_type {
                EndFillType::Nothing | EndFillType::Ngon => {}
                EndFillType::TriFan => {
                    if let Some(center) = bottom_center {
                        polygons.push(vec![center, bottom_start + index, bottom_start + next]);
                    }
                    if let Some(center) = top_center {
                        polygons.push(vec![center, top_start + next, top_start + index]);
                    }
                }
            }
        }
        if params.end_fill_type == EndFillType::Ngon {
            polygons.push((bottom_start..top_start).collect());
            polygons.push((top_start..top_start + count).rev().collect());
        }
        Self::from_positions_and_faces(positions, polygons)
    }

    /// Create a Blender-default or customized XY plane.
    ///
    /// # Errors
    ///
    /// Returns an error when the plane size is not finite and positive.
    pub fn plane(params: PlaneParams) -> Result<Self, MeshError> {
        positive(params.size, "plane size")?;
        let half = params.size * 0.5;
        Self::from_positions_and_faces(
            vec![
                DVec3::new(-half, -half, 0.0),
                DVec3::new(half, -half, 0.0),
                DVec3::new(half, half, 0.0),
                DVec3::new(-half, half, 0.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
    }

    /// Create a Blender-default or customized cone/truncated cone with capped ends.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid radii, depth, vertex count, or exhausted mesh IDs.
    #[expect(
        clippy::cast_precision_loss,
        reason = "indices derive from u32 counts and are exactly representable in f64"
    )]
    pub fn cone(params: ConeParams) -> Result<Self, MeshError> {
        validate_radial_count(params.vertices, "cone vertices")?;
        positive(params.depth, "cone depth")?;
        nonnegative(params.radius1, "cone bottom radius")?;
        nonnegative(params.radius2, "cone top radius")?;
        if params.radius1 == 0.0 && params.radius2 == 0.0 {
            return Err(MeshError::InvalidParameter(
                "at least one cone radius must be positive",
            ));
        }
        let count = params.vertices as usize;
        let fan_caps = params.end_fill_type == EndFillType::TriFan;
        let has_bottom = params.radius1 > 0.0;
        let has_top = params.radius2 > 0.0;
        let cap_count = usize::from(fan_caps && has_bottom) + usize::from(fan_caps && has_top);
        let mut positions = Vec::with_capacity(count * 2 + cap_count + 2);
        let bottom_center = if fan_caps && has_bottom {
            let center = positions.len();
            positions.push(DVec3::new(0.0, 0.0, -params.depth * 0.5));
            Some(center)
        } else {
            None
        };
        let top_center = if fan_caps && has_top {
            let center = positions.len();
            positions.push(DVec3::new(0.0, 0.0, params.depth * 0.5));
            Some(center)
        } else {
            None
        };
        let ring = |positions: &mut Vec<DVec3>, radius: f64, z: f64| {
            let start = positions.len();
            for index in 0..count {
                let angle = std::f64::consts::FRAC_PI_2
                    - std::f64::consts::TAU * index as f64 / count as f64;
                positions.push(DVec3::new(radius * angle.cos(), radius * angle.sin(), z));
            }
            start
        };
        let bottom_start =
            has_bottom.then(|| ring(&mut positions, params.radius1, -params.depth * 0.5));
        let bottom_apex = if has_bottom {
            None
        } else {
            let apex = positions.len();
            positions.push(DVec3::new(0.0, 0.0, -params.depth * 0.5));
            Some(apex)
        };
        let top_start = has_top.then(|| ring(&mut positions, params.radius2, params.depth * 0.5));
        let top_apex = if has_top {
            None
        } else {
            let apex = positions.len();
            positions.push(DVec3::new(0.0, 0.0, params.depth * 0.5));
            Some(apex)
        };
        let mut polygons = Vec::with_capacity(count * 3 + 2);
        for index in 0..count {
            let next = (index + 1) % count;
            match (bottom_start, top_start) {
                (Some(bottom), Some(top)) => {
                    polygons.push(vec![bottom + index, top + index, top + next, bottom + next]);
                }
                (Some(bottom), None) => {
                    let apex =
                        top_apex.ok_or(MeshError::InvalidTopology("cone top apex is missing"))?;
                    polygons.push(vec![bottom + index, apex, bottom + next]);
                }
                (None, Some(top)) => {
                    let apex = bottom_apex
                        .ok_or(MeshError::InvalidTopology("cone bottom apex is missing"))?;
                    polygons.push(vec![apex, top + next, top + index]);
                }
                (None, None) => {
                    return Err(MeshError::InvalidParameter(
                        "at least one cone radius must be positive",
                    ));
                }
            }
        }
        match params.end_fill_type {
            EndFillType::Nothing => {}
            EndFillType::Ngon => {
                if let Some(bottom) = bottom_start {
                    polygons.push((bottom..bottom + count).collect());
                }
                if let Some(top) = top_start {
                    polygons.push((top..top + count).rev().collect());
                }
            }
            EndFillType::TriFan => {
                for index in 0..count {
                    let next = (index + 1) % count;
                    if let (Some(center), Some(bottom)) = (bottom_center, bottom_start) {
                        polygons.push(vec![center, bottom + index, bottom + next]);
                    }
                    if let (Some(center), Some(top)) = (top_center, top_start) {
                        polygons.push(vec![center, top + next, top + index]);
                    }
                }
            }
        }
        Self::from_positions_and_faces(positions, polygons)
    }

    /// Create a Blender-default or customized torus.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid radii/segment counts or exhausted mesh IDs.
    #[expect(
        clippy::cast_precision_loss,
        reason = "indices derive from u32 counts and are exactly representable in f64"
    )]
    pub fn torus(params: TorusParams) -> Result<Self, MeshError> {
        validate_radial_count(params.major_segments, "torus major segments")?;
        validate_radial_count(params.minor_segments, "torus minor segments")?;
        let (major_radius, minor_radius) = match params.mode {
            TorusMode::MajorMinor => {
                positive(params.major_radius, "torus major radius")?;
                positive(params.minor_radius, "torus minor radius")?;
                (params.major_radius, params.minor_radius)
            }
            TorusMode::ExtInt => {
                positive(params.abso_major_rad, "torus exterior radius")?;
                nonnegative(params.abso_minor_rad, "torus interior radius")?;
                if params.abso_minor_rad >= params.abso_major_rad {
                    return Err(MeshError::InvalidParameter(
                        "torus interior radius must be less than exterior radius",
                    ));
                }
                (
                    params.abso_major_rad.midpoint(params.abso_minor_rad),
                    (params.abso_major_rad - params.abso_minor_rad) * 0.5,
                )
            }
        };
        let major = params.major_segments as usize;
        let minor = params.minor_segments as usize;
        let count = major.checked_mul(minor).ok_or(MeshError::IdExhausted)?;
        if count > u32::MAX as usize {
            return Err(MeshError::IdExhausted);
        }
        let mut positions = Vec::with_capacity(count);
        for ring in 0..major {
            let u = std::f64::consts::TAU * ring as f64 / major as f64;
            for side in 0..minor {
                let v = std::f64::consts::TAU * side as f64 / minor as f64;
                let radial = major_radius + minor_radius * v.cos();
                positions.push(DVec3::new(
                    radial * u.cos(),
                    radial * u.sin(),
                    minor_radius * v.sin(),
                ));
            }
        }
        let mut polygons = Vec::with_capacity(count);
        for ring in 0..major {
            for side in 0..minor {
                let next_ring = (ring + 1) % major;
                let next_side = (side + 1) % minor;
                let current = ring * minor + side;
                polygons.push(vec![
                    current,
                    next_ring * minor + side,
                    next_ring * minor + next_side,
                    ring * minor + next_side,
                ]);
            }
        }
        Self::from_positions_and_faces(positions, polygons)
    }

    /// Create a Blender-default or customized recursively subdivided icosphere.
    ///
    /// # Errors
    ///
    /// Returns an error for a subdivision level outside 1–10 or exhausted mesh IDs.
    pub fn icosphere(params: IcosphereParams) -> Result<Self, MeshError> {
        if !(1..=10).contains(&params.subdivisions) {
            return Err(MeshError::InvalidParameter(
                "icosphere subdivisions must be between 1 and 10",
            ));
        }
        positive(params.radius, "icosphere radius")?;
        let radius = params.radius as f32;
        let radius_scale = radius / 200.0_f32;
        let ico_vertices = [
            [0.0_f32, 0.0, -200.0],
            [144.72, -105.144, -89.443],
            [-55.277, -170.128, -89.443],
            [-178.885, 0.0, -89.443],
            [-55.277, 170.128, -89.443],
            [144.72, 105.144, -89.443],
            [55.277, -170.128, 89.443],
            [-144.72, -105.144, 89.443],
            [-144.72, 105.144, 89.443],
            [55.277, 170.128, 89.443],
            [178.885, 0.0, 89.443],
            [0.0, 0.0, 200.0],
        ];
        let mut positions = ico_vertices
            .into_iter()
            .map(|vertex| {
                DVec3::new(
                    f64::from(vertex[0] * radius_scale),
                    f64::from(vertex[1] * radius_scale),
                    f64::from(vertex[2] * radius_scale),
                )
            })
            .collect::<Vec<_>>();
        let base_faces: Vec<[usize; 3]> = vec![
            [0, 1, 2],
            [1, 0, 5],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [1, 5, 10],
            [2, 1, 6],
            [3, 2, 7],
            [4, 3, 8],
            [5, 4, 9],
            [1, 10, 6],
            [2, 6, 7],
            [3, 7, 8],
            [4, 8, 9],
            [5, 9, 10],
            [6, 10, 11],
            [7, 6, 11],
            [8, 7, 11],
            [9, 8, 11],
            [10, 9, 11],
        ];
        if params.subdivisions == 1 {
            return Self::from_positions_and_faces(
                positions,
                base_faces
                    .into_iter()
                    .map(|face| face.into_iter().collect())
                    .collect(),
            );
        }

        // Blender subdivides the original edges once with all cuts for the
        // requested level; it does not recursively subdivide the previous mesh.
        let cuts = (1_usize << (params.subdivisions - 1)) - 1;
        let mut edge_indices = HashMap::with_capacity(base_faces.len() * 3 / 2);
        let mut edges = Vec::<([usize; 2], Vec<usize>)>::with_capacity(base_faces.len() * 3 / 2);
        for [a, b, c] in base_faces.iter().copied() {
            for [start, end] in [[c, a], [a, b], [b, c]] {
                let key = edge_key(start, end);
                if let std::collections::hash_map::Entry::Vacant(entry) = edge_indices.entry(key) {
                    let index = edges.len();
                    entry.insert(index);
                    edges.push(([start, end], Vec::with_capacity(cuts)));
                }
            }
        }
        let interior_points = base_faces.len() * cuts * (cuts - 1) / 2;
        positions.reserve(edges.len() * cuts + interior_points);
        for (endpoints, points) in &mut edges {
            subdivide_sphere_edge(
                positions[endpoints[0]],
                positions[endpoints[1]],
                cuts,
                radius,
                &mut positions,
                points,
            );
        }
        for position in &mut positions {
            let coordinates = position.to_array().map(|coordinate| coordinate as f32);
            *position =
                DVec3::from_array(normalize_sphere_point(coordinates, radius).map(f64::from));
        }

        let mut grids = vec![Vec::<Vec<usize>>::new(); base_faces.len()];
        // The subdivider defers face filling on a stack, so per-face interior
        // vertices are appended in reverse face order.
        for face_index in (0..base_faces.len()).rev() {
            let [a, b, c] = base_faces[face_index];
            let (ab_endpoints, ab_points) = &edges[edge_indices[&edge_key(a, b)]];
            let (bc_endpoints, bc_points) = &edges[edge_indices[&edge_key(b, c)]];
            let (ca_endpoints, ca_points) = &edges[edge_indices[&edge_key(c, a)]];
            let edge_point = |points: &[usize], forward: bool, index: usize| {
                if forward {
                    points[index]
                } else {
                    points[cuts - index - 1]
                }
            };
            let ab_forward = *ab_endpoints == [a, b];
            let bc_forward = *bc_endpoints == [b, c];
            let ca_forward = *ca_endpoints == [c, a];
            let mut rows = Vec::with_capacity(cuts + 2);
            rows.push(vec![c]);
            for row in 1..=cuts {
                let left = edge_point(ca_points, ca_forward, row - 1);
                let right = edge_point(bc_points, bc_forward, cuts - row);
                let mut line = Vec::with_capacity(row + 1);
                line.push(left);
                subdivide_sphere_edge(
                    positions[left],
                    positions[right],
                    row - 1,
                    radius,
                    &mut positions,
                    &mut line,
                );
                line.push(right);
                rows.push(line);
            }
            let mut bottom = Vec::with_capacity(cuts + 2);
            bottom.push(a);
            bottom.extend((0..cuts).map(|index| edge_point(ab_points, ab_forward, index)));
            bottom.push(b);
            rows.push(bottom);
            grids[face_index] = rows;
        }

        let face_count = base_faces.len() * (cuts + 1).pow(2);
        let mut polygons = Vec::with_capacity(face_count);
        let mut retained_faces = Vec::with_capacity(base_faces.len());
        let mut added_faces = Vec::with_capacity(base_faces.len());
        for rows in grids {
            let mut added = Vec::with_capacity((cuts + 1).pow(2) - 1);
            for row in 0..cuts {
                added.push(vec![rows[row + 1][0], rows[row + 1][1], rows[row][0]]);
            }
            for row in 1..=cuts {
                for column in 0..row {
                    added.push(vec![
                        rows[row][column],
                        rows[row + 1][column + 1],
                        rows[row][column + 1],
                    ]);
                    added.push(vec![
                        rows[row + 1][column + 1],
                        rows[row + 1][column + 2],
                        rows[row][column + 1],
                    ]);
                }
            }
            retained_faces.push(vec![rows[cuts + 1][0], rows[cuts + 1][1], rows[cuts][0]]);
            added_faces.push(added);
        }
        polygons.extend(retained_faces);
        for added in added_faces.into_iter().rev() {
            polygons.extend(added);
        }
        Self::from_positions_and_faces(positions, polygons)
    }

    /// Create a Blender-default or customized XY circle with optional fill.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid circle radii/counts or exhausted mesh IDs.
    #[expect(
        clippy::cast_precision_loss,
        reason = "indices derive from u32 counts and are exactly representable in f64"
    )]
    pub fn circle(params: CircleParams) -> Result<Self, MeshError> {
        validate_radial_count(params.vertices, "circle vertices")?;
        positive(params.radius, "circle radius")?;
        let count = params.vertices as usize;
        let fan = params.fill_type == CircleFillType::TriFan;
        let mut positions = Vec::with_capacity(count + usize::from(fan));
        if fan {
            positions.push(DVec3::ZERO);
        }
        let ring_start = positions.len();
        for index in 0..count {
            let angle =
                std::f64::consts::FRAC_PI_2 - std::f64::consts::TAU * index as f64 / count as f64;
            positions.push(DVec3::new(
                params.radius * angle.cos(),
                params.radius * angle.sin(),
                0.0,
            ));
        }
        let polygons = match params.fill_type {
            CircleFillType::Nothing => Vec::new(),
            CircleFillType::Ngon => vec![(ring_start..ring_start + count).rev().collect()],
            CircleFillType::TriFan => (0..count)
                .map(|index| vec![0, ring_start + (index + 1) % count, ring_start + index])
                .collect(),
        };
        let mut mesh = Self::from_positions_and_faces(positions, polygons)?;
        if params.fill_type == CircleFillType::Nothing {
            for index in 0..count {
                let first = u32::try_from(index).map_err(|_| MeshError::IdExhausted)?;
                let next =
                    u32::try_from((index + 1) % count).map_err(|_| MeshError::IdExhausted)?;
                mesh.insert_edge([first, next])?;
            }
        }
        Ok(mesh)
    }

    /// Create a Blender-default or customized subdivided XY grid.
    ///
    /// # Errors
    ///
    /// Returns an error for non-positive sizes, invalid subdivision counts, or exhausted IDs.
    #[expect(
        clippy::cast_precision_loss,
        reason = "indices derive from u32 counts and are exactly representable in f64"
    )]
    pub fn grid(params: GridParams) -> Result<Self, MeshError> {
        positive(params.size_x, "grid X size")?;
        positive(params.size_y, "grid Y size")?;
        if params.x_subdivisions < 2 || params.y_subdivisions < 2 {
            return Err(MeshError::InvalidParameter(
                "grid subdivisions must be at least two per axis",
            ));
        }
        let nx = params.x_subdivisions as usize;
        let ny = params.y_subdivisions as usize;
        let vertex_count = nx.checked_mul(ny).ok_or(MeshError::IdExhausted)?;
        if vertex_count > u32::MAX as usize {
            return Err(MeshError::IdExhausted);
        }
        let mut positions = Vec::with_capacity(vertex_count);
        for y in 0..ny {
            for x in 0..nx {
                let px = -params.size_x * 0.5 + params.size_x * x as f64 / (nx - 1) as f64;
                let py = -params.size_y * 0.5 + params.size_y * y as f64 / (ny - 1) as f64;
                positions.push(DVec3::new(px, py, 0.0));
            }
        }
        let mut polygons = Vec::with_capacity((nx - 1) * (ny - 1));
        for y in 0..ny - 1 {
            for x in 0..nx - 1 {
                let lower_left = y * nx + x;
                polygons.push(vec![
                    lower_left,
                    lower_left + 1,
                    lower_left + nx + 1,
                    lower_left + nx,
                ]);
            }
        }
        Self::from_positions_and_faces(positions, polygons)
    }
}
/// Create a Blender-style primitive by lowercase kind and optional JSON parameters.
///
/// Missing parameter fields retain their Blender defaults. Unknown fields are rejected.
///
/// # Errors
///
/// Returns an error for an unknown primitive kind, malformed parameter object, or
/// values outside the corresponding primitive's valid domain.
pub fn primitive(kind: &str, params: &serde_json::Value) -> Result<Mesh, MeshError> {
    match kind {
        "box" => {
            let params: PrimitiveBoxParams = read_params(params)?;
            Mesh::box_mesh(BoxParams {
                size: DVec3::splat(params.size),
            })
        }
        "sphere" | "uv_sphere" => Mesh::uv_sphere(read_params(params)?),
        "cylinder" => Mesh::cylinder(read_params(params)?),
        "plane" => Mesh::plane(read_params(params)?),
        "cone" => Mesh::cone(read_params(params)?),
        "torus" => Mesh::torus(read_params(params)?),
        "icosphere" => Mesh::icosphere(read_params(params)?),
        "circle" => Mesh::circle(read_params(params)?),
        "grid" => {
            let params: PrimitiveGridParams = read_params(params)?;
            Mesh::grid(GridParams {
                size_x: params.size,
                size_y: params.size,
                x_subdivisions: params
                    .x_subdivisions
                    .checked_add(1)
                    .ok_or(MeshError::IdExhausted)?,
                y_subdivisions: params
                    .y_subdivisions
                    .checked_add(1)
                    .ok_or(MeshError::IdExhausted)?,
            })
        }
        _ => Err(MeshError::InvalidParameter("unknown primitive kind")),
    }
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PrimitiveBoxParams {
    size: f64,
}

impl Default for PrimitiveBoxParams {
    fn default() -> Self {
        Self { size: 2.0 }
    }
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PrimitiveGridParams {
    x_subdivisions: u32,
    y_subdivisions: u32,
    size: f64,
}

impl Default for PrimitiveGridParams {
    fn default() -> Self {
        Self {
            x_subdivisions: 10,
            y_subdivisions: 10,
            size: 2.0,
        }
    }
}

fn read_params<T: DeserializeOwned + Default>(params: &serde_json::Value) -> Result<T, MeshError> {
    if params.is_null() {
        return Ok(T::default());
    }
    serde_json::from_value(params.clone())
        .map_err(|_| MeshError::InvalidParameter("malformed primitive parameters"))
}

fn allocate(next: u32) -> Result<(u32, u32), MeshError> {
    let following = next.checked_add(1).ok_or(MeshError::IdExhausted)?;
    Ok((next, following))
}

fn validate_domain(
    ids: impl IntoIterator<Item = u32>,
    next: u32,
    domain: &'static str,
) -> Result<HashSet<u32>, MeshError> {
    let mut unique = HashSet::new();
    let mut maximum = None;
    for id in ids {
        if !unique.insert(id) {
            return Err(MeshError::InvalidTopology(
                "duplicate persistent element ID",
            ));
        }
        maximum = Some(maximum.map_or(id, |value: u32| value.max(id)));
    }
    if maximum.is_some_and(|maximum| next <= maximum) {
        return Err(MeshError::InvalidTopology(match domain {
            "vertex" => "vertex next_id does not exceed all allocated IDs",
            "edge" => "edge next_id does not exceed all allocated IDs",
            _ => "face next_id does not exceed all allocated IDs",
        }));
    }
    Ok(unique)
}

fn validate_polygon_vertices(vertices: &[u32]) -> Result<(), MeshError> {
    if vertices.len() < 3 {
        return Err(MeshError::InvalidTopology(
            "a face needs at least three vertices",
        ));
    }
    if vertices.len() <= 8 {
        for (index, vertex) in vertices.iter().enumerate() {
            if vertices[..index].contains(vertex) {
                return Err(MeshError::InvalidTopology("a face repeats a vertex ID"));
            }
        }
    } else {
        let mut unique = HashSet::with_capacity(vertices.len());
        for vertex in vertices {
            if !unique.insert(*vertex) {
                return Err(MeshError::InvalidTopology("a face repeats a vertex ID"));
            }
        }
    }
    Ok(())
}

pub(crate) fn edge_key<T: Copy + Ord>(first: T, second: T) -> (T, T) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn cyclic_pairs(vertices: &[u32]) -> impl Iterator<Item = [u32; 2]> + '_ {
    vertices
        .iter()
        .copied()
        .zip(vertices.iter().copied().cycle().skip(1))
        .take(vertices.len())
        .map(|(a, b)| [a, b])
}

fn positive(value: f64, name: &'static str) -> Result<(), MeshError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(MeshError::InvalidParameter(name))
    }
}

fn nonnegative(value: f64, name: &'static str) -> Result<(), MeshError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(MeshError::InvalidParameter(name))
    }
}

fn positive_vec3(value: DVec3) -> bool {
    value.is_finite() && value.cmpgt(DVec3::ZERO).all()
}

fn validate_radial_count(value: u32, name: &'static str) -> Result<(), MeshError> {
    if value >= 3 {
        Ok(())
    } else {
        Err(MeshError::InvalidParameter(name))
    }
}

fn subdivide_sphere_edge(
    start: DVec3,
    end: DVec3,
    cuts: usize,
    radius: f32,
    positions: &mut Vec<DVec3>,
    indices: &mut Vec<usize>,
) {
    let mut point = start.to_array().map(|coordinate| coordinate as f32);
    let end = end.to_array().map(|coordinate| coordinate as f32);
    for cut in 0..cuts {
        let factor = 1.0_f32 / (cuts + 1 - cut) as f32;
        for axis in 0..3 {
            point[axis] += (end[axis] - point[axis]) * factor;
        }
        let sphere_point = normalize_sphere_point(point, radius);
        indices.push(positions.len());
        positions.push(DVec3::new(
            f64::from(sphere_point[0]),
            f64::from(sphere_point[1]),
            f64::from(sphere_point[2]),
        ));
    }
}

fn normalize_sphere_point(mut point: [f32; 3], radius: f32) -> [f32; 3] {
    let length = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
    for coordinate in &mut point {
        *coordinate = *coordinate / length * radius;
    }
    point
}

fn triangulate_face(
    face: &Face,
    vertex_positions: &HashMap<u32, DVec3>,
) -> Result<Vec<[u32; 3]>, MeshError> {
    let points: Vec<DVec3> = face
        .vertices
        .iter()
        .map(|id| {
            vertex_positions
                .get(id)
                .copied()
                .ok_or(MeshError::InvalidTopology(
                    "face references a missing vertex",
                ))
        })
        .collect::<Result<_, _>>()?;
    let normal = points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .fold(DVec3::ZERO, |sum, (current, next)| {
            sum + DVec3::new(
                (current.y - next.y) * (current.z + next.z),
                (current.z - next.z) * (current.x + next.x),
                (current.x - next.x) * (current.y + next.y),
            )
        });
    let absolute = normal.abs();
    let axis = if absolute.x >= absolute.y && absolute.x >= absolute.z {
        0
    } else if absolute.y >= absolute.z {
        1
    } else {
        2
    };
    let projected: Vec<DVec2> = points
        .iter()
        .map(|point| match axis {
            0 => DVec2::new(point.y, point.z),
            1 => DVec2::new(point.x, point.z),
            _ => DVec2::new(point.x, point.y),
        })
        .collect();
    let twice_area = projected
        .iter()
        .zip(projected.iter().cycle().skip(1))
        .take(projected.len())
        .map(|(a, b)| a.x * b.y - b.x * a.y)
        .sum::<f64>();
    let scale = projected
        .iter()
        .fold(1.0_f64, |scale, point| scale.max(point.abs().max_element()));
    let epsilon = f64::EPSILON * scale * scale * 32.0;
    if !twice_area.is_finite() || twice_area.abs() <= epsilon {
        return Err(MeshError::TriangulationFailed(face.id));
    }
    let orientation = twice_area.signum();
    let mut remaining: Vec<usize> = (0..face.vertices.len()).collect();
    let mut triangles = Vec::with_capacity(remaining.len() - 2);
    while remaining.len() > 3 {
        let mut ear = None;
        for cursor in 0..remaining.len() {
            let previous = remaining[(cursor + remaining.len() - 1) % remaining.len()];
            let current = remaining[cursor];
            let next = remaining[(cursor + 1) % remaining.len()];
            let a = projected[previous];
            let b = projected[current];
            let c = projected[next];
            if cross2(b - a, c - b) * orientation <= epsilon {
                continue;
            }
            if remaining
                .iter()
                .copied()
                .filter(|candidate| {
                    *candidate != previous && *candidate != current && *candidate != next
                })
                .any(|candidate| {
                    point_in_triangle_2d(projected[candidate], a, b, c, orientation, epsilon)
                })
            {
                continue;
            }
            ear = Some((
                cursor,
                [
                    face.vertices[previous],
                    face.vertices[current],
                    face.vertices[next],
                ],
            ));
            break;
        }
        let Some((cursor, triangle)) = ear else {
            return Err(MeshError::TriangulationFailed(face.id));
        };
        triangles.push(triangle);
        remaining.remove(cursor);
    }
    triangles.push([
        face.vertices[remaining[0]],
        face.vertices[remaining[1]],
        face.vertices[remaining[2]],
    ]);
    Ok(triangles)
}

fn cross2(a: DVec2, b: DVec2) -> f64 {
    a.x * b.y - a.y * b.x
}

fn point_in_triangle_2d(
    point: DVec2,
    a: DVec2,
    b: DVec2,
    c: DVec2,
    orientation: f64,
    epsilon: f64,
) -> bool {
    cross2(b - a, point - a) * orientation >= -epsilon
        && cross2(c - b, point - b) * orientation >= -epsilon
        && cross2(a - c, point - c) * orientation >= -epsilon
}

#[must_use]
pub(crate) fn point_in_triangle(point: DVec3, a: DVec3, b: DVec3, c: DVec3) -> bool {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d00 = ab.dot(ab);
    let d01 = ab.dot(ac);
    let d11 = ac.dot(ac);
    let d20 = ap.dot(ab);
    let d21 = ap.dot(ac);
    let denominator = d00 * d11 - d01 * d01;
    if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
        return false;
    }
    let v = (d11 * d20 - d01 * d21) / denominator;
    let w = (d00 * d21 - d01 * d20) / denominator;
    point_in_triangle_barycentric(v, w, 1.0e-9)
}

#[must_use]
pub(crate) fn point_in_triangle_barycentric(first: f64, second: f64, tolerance: f64) -> bool {
    first >= -tolerance && second >= -tolerance && first + second <= 1.0 + tolerance
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use glam::DVec3;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn blender_default_primitives_have_expected_topology() {
        let cube = Mesh::box_mesh(BoxParams::default()).unwrap();
        assert_eq!(
            (cube.vertices.len(), cube.edges.len(), cube.faces.len()),
            (8, 12, 6)
        );
        assert_eq!(cube.triangulate().unwrap().len(), 12);

        let sphere = Mesh::uv_sphere(UvSphereParams::default()).unwrap();
        assert_eq!((sphere.vertices.len(), sphere.faces.len()), (482, 512));
        assert_eq!(sphere.triangulate().unwrap().len(), 960);

        let cylinder = Mesh::cylinder(CylinderParams::default()).unwrap();
        assert_eq!((cylinder.vertices.len(), cylinder.faces.len()), (64, 34));
        assert_eq!(cylinder.triangulate().unwrap().len(), 124);

        let plane = Mesh::plane(PlaneParams::default()).unwrap();
        assert_eq!(
            (plane.vertices.len(), plane.edges.len(), plane.faces.len()),
            (4, 4, 1)
        );
        assert_eq!(plane.triangulate().unwrap().len(), 2);

        let cone = Mesh::cone(ConeParams::default()).unwrap();
        assert_eq!((cone.vertices.len(), cone.faces.len()), (33, 33));
        assert_eq!(cone.triangulate().unwrap().len(), 62);

        let torus = Mesh::torus(TorusParams::default()).unwrap();
        assert_eq!((torus.vertices.len(), torus.faces.len()), (576, 576));
        assert_eq!(torus.triangulate().unwrap().len(), 1_152);

        let ico = Mesh::icosphere(IcosphereParams::default()).unwrap();
        assert_eq!((ico.vertices.len(), ico.faces.len()), (42, 80));
        assert_eq!(ico.triangulate().unwrap().len(), 80);

        let circle = Mesh::circle(CircleParams::default()).unwrap();
        assert_eq!(
            (
                circle.vertices.len(),
                circle.edges.len(),
                circle.faces.len()
            ),
            (32, 32, 0)
        );

        let grid = Mesh::grid(GridParams::default()).unwrap();
        assert_eq!(
            (grid.vertices.len(), grid.edges.len(), grid.faces.len()),
            (100, 180, 81)
        );
        assert_eq!(grid.triangulate().unwrap().len(), 162);
    }

    #[test]
    fn batched_face_insertion_matches_sequential_topology() {
        let positions = vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::new(1.0, 1.0, 0.0)];
        let polygons = vec![
            (vec![0, 1, 2], 4),
            (vec![2, 1, 3], 2),
            (vec![0, 2, 3, 1], 7),
        ];
        let mut sequential = Mesh::new();
        for position in &positions {
            sequential.insert_vertex(*position).unwrap();
        }
        for (vertices, material_index) in &polygons {
            sequential
                .insert_face(vertices.clone(), *material_index)
                .unwrap();
        }

        let mut batched = Mesh::new();
        for position in positions {
            batched.insert_vertex(position).unwrap();
        }
        let faces = polygons
            .into_iter()
            .map(|(vertices, material_index)| Ok((vertices, material_index)));
        batched.insert_faces(faces).unwrap();

        assert_eq!(batched, sequential);
        assert!(batched.validate().is_ok());
    }

    #[test]
    fn triangulation_preserves_polygon_winding_for_concave_face() {
        let mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 0.0),
                DVec3::new(2.0, 2.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(0.0, 2.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3, 4]],
        )
        .unwrap();
        let triangles = mesh.triangulate().unwrap();
        assert_eq!(triangles.len(), 3);
        for [a, b, c] in triangles {
            let points = [a, b, c].map(|id| mesh.vertex(id).unwrap().co);
            assert!((points[1] - points[0]).cross(points[2] - points[0]).z > 0.0);
        }
    }

    #[test]
    fn triangulation_retains_collinear_boundary_vertices() {
        let mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 0.0),
                DVec3::new(2.0, 2.0, 0.0),
                DVec3::new(0.0, 2.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3, 4]],
        )
        .unwrap();
        assert_eq!(mesh.triangulate().unwrap().len(), 3);
    }

    #[test]
    fn primitive_parameters_reject_invalid_values() {
        assert!(
            Mesh::uv_sphere(UvSphereParams {
                segments: 2,
                ..UvSphereParams::default()
            })
            .is_err()
        );
        assert!(Mesh::box_mesh(BoxParams { size: DVec3::ZERO }).is_err());
        assert!(
            Mesh::torus(TorusParams {
                minor_radius: 0.0,
                ..TorusParams::default()
            })
            .is_err()
        );
    }

    #[test]
    fn primitive_dispatch_and_mesh_json_follow_the_public_contract() {
        let circle = primitive("circle", &serde_json::json!({"vertices": 5})).unwrap();
        let sphere_alias = primitive("sphere", &serde_json::Value::Null).unwrap();
        let uv_sphere = primitive("uv_sphere", &serde_json::Value::Null).unwrap();
        assert_eq!(sphere_alias, uv_sphere);
        assert_eq!((circle.vertices.len(), circle.edges.len()), (5, 5));
        assert!(primitive("unknown", &serde_json::Value::Null).is_err());

        let plane = Mesh::plane(PlaneParams::default()).unwrap();
        let value = serde_json::to_value(&plane).unwrap();
        assert_eq!(value["edges"][0]["v"].as_array().unwrap().len(), 2);
        assert_eq!(value["faces"][0]["v"].as_array().unwrap().len(), 4);
        assert!(value["edges"][0].get("vertices").is_none());
        let decoded: Mesh = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, plane);
    }

    #[test]
    fn cone_supports_zero_radius_at_either_end() {
        let inverted = Mesh::cone(ConeParams {
            radius1: 0.0,
            radius2: 1.0,
            ..ConeParams::default()
        })
        .unwrap();
        assert_eq!((inverted.vertices.len(), inverted.faces.len()), (33, 33));
        assert_eq!(inverted.triangulate().unwrap().len(), 62);
        let truncated = Mesh::cone(ConeParams {
            radius1: 0.5,
            radius2: 1.0,
            ..ConeParams::default()
        })
        .unwrap();
        assert_eq!(truncated.vertices.len(), 64);
        assert_eq!(truncated.triangulate().unwrap().len(), 124);
    }

    fn triangle_mesh() -> Mesh {
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])
            .unwrap()
    }

    #[test]
    fn mesh_mutators_reject_invalid_ids_and_keep_valid_topology() {
        let mut mesh = Mesh::new();
        let first = mesh.insert_vertex(DVec3::ZERO).unwrap();
        let second = mesh.insert_vertex(DVec3::X).unwrap();
        let third = mesh.insert_vertex(DVec3::Y).unwrap();
        assert!(mesh.insert_vertex(DVec3::splat(f64::NAN)).is_err());
        assert!(mesh.insert_edge([first, first]).is_err());
        assert!(mesh.insert_edge([first, 99]).is_err());
        assert_eq!(mesh.insert_edge([first, second]).unwrap(), 0);
        assert!(mesh.insert_edge([second, first]).is_err());
        assert!(mesh.insert_face(vec![first, second], 0).is_err());
        assert!(mesh.insert_face(vec![first, second, first], 0).is_err());
        assert!(mesh.insert_face(vec![first, second, 99], 0).is_err());
        assert_eq!(mesh.insert_face(vec![first, second, third], 0).unwrap(), 0);
        assert!(mesh.validate().is_ok());
    }

    #[test]
    fn mesh_validation_rejects_corrupt_ids_coordinates_and_references() {
        let valid = triangle_mesh();

        let mut duplicate_vertex = valid.clone();
        duplicate_vertex.vertices[1].id = duplicate_vertex.vertices[0].id;
        assert!(duplicate_vertex.validate().is_err());

        let mut exhausted_vertex_counter = valid.clone();
        exhausted_vertex_counter.next_id.vertex = 2;
        assert!(exhausted_vertex_counter.validate().is_err());

        let mut non_finite_vertex = valid.clone();
        non_finite_vertex.vertices[0].co.x = f64::NAN;
        assert!(non_finite_vertex.validate().is_err());

        let mut identical_edge = valid.clone();
        identical_edge.edges[0].vertices[1] = identical_edge.edges[0].vertices[0];
        assert!(identical_edge.validate().is_err());

        let mut missing_edge_vertex = valid.clone();
        missing_edge_vertex.edges[0].vertices[1] = 99;
        assert!(missing_edge_vertex.validate().is_err());

        let mut duplicate_edge = valid.clone();
        duplicate_edge.edges[1].vertices = duplicate_edge.edges[0].vertices;
        assert!(duplicate_edge.validate().is_err());

        let mut missing_boundary_edge = valid.clone();
        missing_boundary_edge.edges.pop();
        assert!(missing_boundary_edge.validate().is_err());

        let mut repeated_face_vertex = valid.clone();
        repeated_face_vertex.faces[0].vertices[1] = repeated_face_vertex.faces[0].vertices[0];
        assert!(repeated_face_vertex.validate().is_err());

        let mut missing_face_vertex = valid.clone();
        missing_face_vertex.faces[0].vertices[2] = 99;
        assert!(missing_face_vertex.validate().is_err());
    }

    #[test]
    fn bounds_and_triangulation_cover_empty_and_degenerate_meshes() {
        let empty = Mesh::new();
        assert!(empty.bounds().is_none());
        assert!(empty.triangulate().unwrap().is_empty());

        let collinear = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::new(2.0, 0.0, 0.0)],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        assert_eq!(
            collinear.triangulate(),
            Err(MeshError::TriangulationFailed(0))
        );
    }

    #[test]
    fn aabb_contains_uses_inclusive_bounds_on_every_axis() {
        let bounds = Aabb {
            min: DVec3::new(-1.0, 2.0, 0.0),
            max: DVec3::new(3.0, 5.0, 4.0),
        };
        assert!(bounds.contains(bounds.min));
        assert!(bounds.contains(bounds.max));
        assert!(!bounds.contains(DVec3::new(-1.1, 3.0, 2.0)));
        assert!(!bounds.contains(DVec3::new(0.0, 5.1, 2.0)));
        assert!(!bounds.contains(DVec3::new(0.0, 3.0, 4.1)));
    }

    proptest! {
        #[test]
        fn aabb_contains_matches_inclusive_reference(
            minimum in prop::array::uniform3(-1.0e6_f64..1.0e6),
            size in prop::array::uniform3(0.0_f64..1.0e6),
            point in prop::array::uniform3(-2.0e6_f64..2.0e6),
        ) {
            let min = DVec3::from_array(minimum);
            let max = min + DVec3::from_array(size);
            let point = DVec3::from_array(point);
            let expected = point.x >= min.x && point.x <= max.x
                && point.y >= min.y && point.y <= max.y
                && point.z >= min.z && point.z <= max.z;
            prop_assert_eq!(Aabb { min, max }.contains(point), expected);
        }
    }
    proptest! {
        #[test]
        fn bounds_match_brute_force_min_max(points in prop::collection::vec(
            ( -1.0e6_f64..1.0e6, -1.0e6_f64..1.0e6, -1.0e6_f64..1.0e6), 1..100
        )) {
            let positions: Vec<_> = points.into_iter().map(|(x, y, z)| DVec3::new(x, y, z)).collect();
            let mesh = Mesh::from_positions_and_faces(positions.clone(), vec![]).unwrap();
            let bounds = mesh.bounds().unwrap();
            let mut min = positions[0];
            let mut max = positions[0];
            for point in &positions[1..] {
                min = min.min(*point);
                max = max.max(*point);
            }
            prop_assert_eq!(bounds.min, min);
            prop_assert_eq!(bounds.max, max);
        }
        #[test]
        fn triangulation_count_matches_convex_polygon_size(vertex_count in 3_u32..100) {
            let count = usize::try_from(vertex_count).unwrap();
            let positions: Vec<_> = (0..vertex_count)
                .map(|index| {
                    let angle = std::f64::consts::TAU * f64::from(index) / f64::from(vertex_count);
                    DVec3::new(angle.cos(), angle.sin(), 0.0)
                })
                .collect();
            let polygon = (0..count).collect();
            let mesh = Mesh::from_positions_and_faces(positions, vec![polygon]).unwrap();
            prop_assert_eq!(mesh.triangulate().unwrap().len(), count - 2);
        }
    }
}
