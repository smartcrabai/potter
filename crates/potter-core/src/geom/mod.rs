// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/bmesh/operators/bmo_primitive.cc, source/blender/bmesh/operators/bmo_subdivide.cc, SPDX-FileCopyrightText: 2023 Blender Authors, GPL-2.0-or-later.

//! Persistent polygon meshes, Blender-default primitives, triangulation, and bounds.
//!
//! The public [`Mesh`] stores stable `u32` IDs independently for vertices, edges, and
//! polygon faces. References in edges/faces use vertex IDs, never vector positions.
//! [`Mesh::validate`] checks this invariant and [`Mesh::tessellate`] returns vertex
//! IDs while preserving each polygon's winding. Primitive parameter structs implement
//! [`Default`] with Blender's standard add-primitive settings; override only the fields
//! that differ. [`Mesh::bounds`] includes loose vertices and returns `None` for an
//! empty mesh.

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
};

use glam::DVec3;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

pub(crate) mod attribute_keys;
pub(crate) mod blender_matrix;
pub mod boolean;
pub(crate) mod clnor;
pub(crate) mod color_layers;
pub(crate) mod concat;
pub mod curve;
pub mod edit;
pub mod lattice;
pub mod mesh_cache;
pub mod metaball;
pub mod modifiers;
pub(crate) mod multires_external;
pub(crate) mod normals;
pub mod pointcloud;
mod primitive_order;
pub(crate) mod provenance;
pub mod remesh;
pub mod sculpt;
mod sculpt_dabs;
pub mod text;
pub(crate) mod triangulate;
pub(crate) mod uv_layers;
mod uv_sphere;
pub mod vdb;
pub mod volume;

/// Evaluated per-corner normals of `mesh`, in flattened `mesh.faces` corner order. Authored
/// custom-normal codes decode on their current fans; Auto corners and meshes without custom normals
/// take the computed shading. This is the evaluator the renderer and exporters read, not a second
/// implementation.
///
/// # Errors
///
/// Returns the errors of the shared corner-normal evaluator for malformed custom-normal state.
pub fn corner_normal_vectors(mesh: &Mesh) -> crate::error::Result<Vec<DVec3>> {
    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    normals::corner_normals(mesh, &positions)
}

#[cfg(test)]
mod mirror_oracle;
#[cfg(test)]
mod shading_oracle;

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
}

impl fmt::Display for MeshError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(message) => {
                write!(formatter, "invalid primitive parameter: {message}")
            }
            Self::InvalidTopology(message) => write!(formatter, "invalid mesh topology: {message}"),
            Self::IdExhausted => formatter.write_str("mesh element ID space exhausted"),
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

/// The largest UV sphere ring count: `uv_sphere::topology` computes `2 * rings * 8` in an `i32`.
const MAX_UV_SPHERE_RINGS: u64 = (i32::MAX / 16) as u64;

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

    /// Triangulate every face as Blender's `Mesh::corner_tris` does (`mesh_tessellate.cc`): a
    /// quad splits along its 0-2 diagonal, flipped to 1-3 when that split folds; any other face
    /// uses Blender's clip ear-clipper (`BLI_polyfill_calc`), which accepts self-intersecting and
    /// zero-area input. Each face yields `n - 2` triangles in its winding; loose vertices and edges
    /// do not appear. Render, preview, exports, and modifiers that need Blender's looptris use this
    /// one implementation, so their diagonals match Blender.
    ///
    /// # Errors
    ///
    /// Returns an error only for invalid mesh topology or ID exhaustion.
    pub fn tessellate(&self) -> Result<Vec<[u32; 3]>, MeshError> {
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
            result.extend(tessellate_face(face, &positions)?);
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
        primitive_order::cube(params.size * 0.5)
    }

    /// Create a Blender-default or customized UV sphere in Blender 5.2.2 storage order.
    ///
    /// Vertex order and positions match Blender bit for bit; see [`uv_sphere`] for the edge
    /// and face order.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid radii/ring counts or exhausted mesh IDs.
    pub fn uv_sphere(params: UvSphereParams) -> Result<Self, MeshError> {
        validate_radial_count(params.segments, "UV sphere segments")?;
        if params.ring_count < 3 {
            return Err(MeshError::InvalidParameter(
                "UV sphere requires at least three rings",
            ));
        }
        positive(params.radius, "UV sphere radius")?;
        // Faces are segments x rings and vertices segments x (rings - 1) + 2; each takes a u32 ID.
        // `uv_sphere::topology` multiplies the ring count by 16 inside an i32, so rings are bounded.
        let (segments, rings) = (u64::from(params.segments), u64::from(params.ring_count));
        let faces = segments * rings;
        let vertices = segments * (rings - 1) + 2;
        let edges = faces + vertices - 2;
        if [faces, vertices, edges]
            .iter()
            .any(|count| *count > u64::from(u32::MAX))
            || rings > MAX_UV_SPHERE_RINGS
        {
            return Err(MeshError::IdExhausted);
        }
        uv_sphere::build(params)
    }
    /// Create a Blender-default or customized capped cylinder.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions/vertex counts or exhausted mesh IDs.
    pub fn cylinder(params: CylinderParams) -> Result<Self, MeshError> {
        validate_radial_count(params.vertices, "cylinder vertices")?;
        positive(params.radius, "cylinder radius")?;
        positive(params.depth, "cylinder depth")?;
        primitive_order::cone(
            params.vertices as usize,
            params.radius,
            params.radius,
            params.depth,
            params.end_fill_type,
        )
    }

    /// Create a Blender-default or customized XY plane.
    ///
    /// # Errors
    ///
    /// Returns an error when the plane size is not finite and positive.
    pub fn plane(params: PlaneParams) -> Result<Self, MeshError> {
        positive(params.size, "plane size")?;
        primitive_order::grid(2, 2, params.size, params.size)
    }

    /// Create a Blender-default or customized cone/truncated cone with capped ends.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid radii, depth, vertex count, or exhausted mesh IDs.
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
        primitive_order::cone(
            params.vertices as usize,
            params.radius1,
            params.radius2,
            params.depth,
            params.end_fill_type,
        )
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
        primitive_order::torus(positions, &polygons)
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
            let polygons: Vec<Vec<usize>> = base_faces.iter().map(|face| face.to_vec()).collect();
            return primitive_order::polygons(positions, &polygons);
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

        // Blender's subdivide_edges (bmo_subdivide.cc): each original edge keeps its slot, now
        // holding its last cut point and end vertex. Then each edge's first half and cut edges are
        // appended in edge order. Then each face's grid is filled, last face first (the operator's
        // stack), with the diagonal and its splits, and the grid's connecting edges.
        let mut edge_pairs: Vec<[usize; 2]> =
            Vec::with_capacity(edges.len() * (cuts + 1) + base_faces.len() * 3 * cuts * cuts);
        for (endpoints, points) in &edges {
            edge_pairs.push([points[cuts - 1], endpoints[1]]);
        }
        for (endpoints, points) in &edges {
            let mut previous = endpoints[0];
            for &point in points {
                edge_pairs.push([previous, point]);
                previous = point;
            }
        }
        for rows in grids.iter().rev() {
            for row in 0..cuts {
                let diagonal = edge_pairs.len();
                edge_pairs.push([rows[row + 1][0], rows[row + 1][row + 1]]);
                for split in 0..row {
                    edge_pairs.push([rows[row + 1][split], rows[row + 1][split + 1]]);
                }
                if row > 0 {
                    edge_pairs[diagonal] = [rows[row + 1][row], rows[row + 1][row + 1]];
                }
            }
            for row in 1..=cuts {
                for column in 0..row {
                    edge_pairs.push([rows[row][column], rows[row + 1][column + 1]]);
                    edge_pairs.push([rows[row][column + 1], rows[row + 1][column + 1]]);
                }
            }
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
        primitive_order::explicit_edges(positions, &edge_pairs, &polygons)
    }

    /// Create a Blender-default or customized XY circle with optional fill.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid circle radii/counts or exhausted mesh IDs.
    pub fn circle(params: CircleParams) -> Result<Self, MeshError> {
        validate_radial_count(params.vertices, "circle vertices")?;
        positive(params.radius, "circle radius")?;
        primitive_order::circle(params.vertices as usize, params.radius, params.fill_type)
    }

    /// Create a Blender-default or customized subdivided XY grid.
    ///
    /// # Errors
    ///
    /// Returns an error for non-positive sizes, invalid subdivision counts, or exhausted IDs.
    pub fn grid(params: GridParams) -> Result<Self, MeshError> {
        positive(params.size_x, "grid X size")?;
        positive(params.size_y, "grid Y size")?;
        if params.x_subdivisions < 2 || params.y_subdivisions < 2 {
            return Err(MeshError::InvalidParameter(
                "grid subdivisions must be at least two per axis",
            ));
        }
        // Vertices, edges and faces each take a u32 ID; refuse before anything is allocated.
        let (nx, ny) = (
            u64::from(params.x_subdivisions),
            u64::from(params.y_subdivisions),
        );
        let vertices = nx * ny;
        let edges = nx * (ny - 1) + ny * (nx - 1);
        let faces = (nx - 1) * (ny - 1);
        if [vertices, edges, faces]
            .iter()
            .any(|count| *count > u64::from(u32::MAX))
        {
            return Err(MeshError::IdExhausted);
        }
        primitive_order::grid(
            params.x_subdivisions as usize,
            params.y_subdivisions as usize,
            params.size_x,
            params.size_y,
        )
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

/// One face's triangles as `mesh_tessellate.cc` lays them out (see [`Mesh::tessellate`]).
fn tessellate_face(
    face: &Face,
    vertex_positions: &HashMap<u32, DVec3>,
) -> Result<Vec<[u32; 3]>, MeshError> {
    let corners = face
        .vertices
        .iter()
        .map(|id| {
            let point = vertex_positions.get(id).ok_or(MeshError::InvalidTopology(
                "face references a missing vertex",
            ))?;
            Ok([point.x as f32, point.y as f32, point.z as f32])
        })
        .collect::<Result<Vec<[f32; 3]>, MeshError>>()?;
    Ok(triangulate::loop_triangles(&corners)
        .into_iter()
        .map(|[a, b, c]| [face.vertices[a], face.vertices[b], face.vertices[c]])
        .collect())
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
        assert_eq!(cube.tessellate().unwrap().len(), 12);

        let sphere = Mesh::uv_sphere(UvSphereParams::default()).unwrap();
        assert_eq!((sphere.vertices.len(), sphere.faces.len()), (482, 512));
        assert_eq!(sphere.tessellate().unwrap().len(), 960);

        let cylinder = Mesh::cylinder(CylinderParams::default()).unwrap();
        assert_eq!((cylinder.vertices.len(), cylinder.faces.len()), (64, 34));
        assert_eq!(cylinder.tessellate().unwrap().len(), 124);

        let plane = Mesh::plane(PlaneParams::default()).unwrap();
        assert_eq!(
            (plane.vertices.len(), plane.edges.len(), plane.faces.len()),
            (4, 4, 1)
        );
        assert_eq!(plane.tessellate().unwrap().len(), 2);

        let cone = Mesh::cone(ConeParams::default()).unwrap();
        assert_eq!((cone.vertices.len(), cone.faces.len()), (33, 33));
        assert_eq!(cone.tessellate().unwrap().len(), 62);

        let torus = Mesh::torus(TorusParams::default()).unwrap();
        assert_eq!((torus.vertices.len(), torus.faces.len()), (576, 576));
        assert_eq!(torus.tessellate().unwrap().len(), 1_152);

        let ico = Mesh::icosphere(IcosphereParams::default()).unwrap();
        assert_eq!((ico.vertices.len(), ico.faces.len()), (42, 80));
        assert_eq!(ico.tessellate().unwrap().len(), 80);

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
        assert_eq!(grid.tessellate().unwrap().len(), 162);
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
        let triangles = mesh.tessellate().unwrap();
        assert_eq!(triangles.len(), 3);
        for [a, b, c] in triangles {
            let points = [a, b, c].map(|id| mesh.vertex(id).unwrap().co);
            assert!((points[1] - points[0]).cross(points[2] - points[0]).z > 0.0);
        }
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
    fn oversized_grid_and_uv_sphere_are_refused_before_allocation() {
        // 100000 x 100000 vertices exceed u32 IDs; the check runs before any allocation.
        let grid = Mesh::grid(GridParams {
            size_x: 2.0,
            size_y: 2.0,
            x_subdivisions: 100_000,
            y_subdivisions: 100_000,
        });
        assert!(matches!(grid, Err(MeshError::IdExhausted)));
        // Ring counts past the i32 bound of the UV sphere's angle arithmetic are refused too.
        let sphere = Mesh::uv_sphere(UvSphereParams {
            segments: 3,
            ring_count: u32::MAX,
            radius: 1.0,
        });
        assert!(matches!(sphere, Err(MeshError::IdExhausted)));
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
        assert_eq!(inverted.tessellate().unwrap().len(), 62);
        let truncated = Mesh::cone(ConeParams {
            radius1: 0.5,
            radius2: 1.0,
            ..ConeParams::default()
        })
        .unwrap();
        assert_eq!(truncated.vertices.len(), 64);
        assert_eq!(truncated.tessellate().unwrap().len(), 124);
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
        assert!(
            empty.tessellate().unwrap().is_empty(),
            "an empty mesh has no triangles"
        );

        let collinear = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::new(2.0, 0.0, 0.0)],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        assert_eq!(collinear.tessellate().unwrap(), vec![[0, 1, 2]]);
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
            prop_assert_eq!(mesh.tessellate().unwrap().len(), count - 2);
        }
    }
}

#[cfg(kani)]
mod kani_verification {
    use super::{cyclic_pairs, edge_key, validate_radial_count};

    #[kani::proof]
    fn edge_key_is_canonical_and_direction_independent() {
        let first: u32 = kani::any();
        let second: u32 = kani::any();
        let key = edge_key(first, second);

        kani::assert(
            key.0 == first.min(second),
            "edge keys store the smaller ID first",
        );
        kani::assert(
            key.1 == first.max(second),
            "edge keys store the larger ID second",
        );
        kani::assert(
            key == edge_key(second, first),
            "reversing endpoints preserves the edge key",
        );
    }

    #[kani::proof]
    fn radial_count_validation_matches_the_minimum_topology() {
        let count: u32 = kani::any();

        kani::assert(
            validate_radial_count(count, "radial count").is_ok() == (count >= 3),
            "radial counts below three are rejected and all larger counts are accepted",
        );
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn cyclic_pairs_cover_each_edge_and_wrap_once() {
        let vertices: [u32; 4] = kani::any();
        let mut pairs = cyclic_pairs(&vertices);

        for index in 0..vertices.len() {
            kani::assert(
                pairs.next() == Some([vertices[index], vertices[(index + 1) % vertices.len()]]),
                "cyclic pairs preserve vertex order and wrap the final edge",
            );
        }
        kani::assert(
            pairs.next().is_none(),
            "each polygon edge is emitted exactly once",
        );
    }
}
