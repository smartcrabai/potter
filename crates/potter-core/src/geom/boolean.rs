//! Boolean operations on polygon meshes.
//!
//! `EXACT` (the default solver) is Blender's Mesh Arrangements, ported under `boolean/`: exact
//! rational intersection, classification of the intersected triangles, and assembly of the output
//! polygons with Blender's attribute rules.
//!
//! `FLOAT` runs the BSP boundary representation below: plane-side classification with exact
//! `orient3d` predicates outside a small, scale-relative coplanar tolerance, f64 intersections, and
//! post-hoc coplanar merging. It is a shortcut until `PR17b` ports Blender's float path.
//!
//! Operand and material semantics follow Blender's `non_float_boolean_mesh`: an unset Object
//! operand leaves the mesh unchanged, and `material_remaps` gives the INDEX and TRANSFER slot maps.

mod cdt;
mod classify;
mod dissolve;
mod exact;
mod imesh;
mod input;
mod intersect;
mod output;
#[cfg(test)]
mod tests;
mod triangulate;

use std::collections::{BTreeMap, HashMap, HashSet};

use glam::{DMat4, DVec3};
use robust::{Coord3D, orient3d};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh, ORIGINDEX_ATTRIBUTE,
        provenance::{Origin, mean_value_corners, remap_attributes},
    },
    model::Id,
};

use classify::{BoolOp, boolean_tris};
use dissolve::polymesh_from_tris;
use imesh::Arena;
use input::{Operand, meshes_to_tris};
use output::{OutputMaterials, polys_to_mesh};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Operation {
    Union,
    Difference,
    Intersect,
}

impl Operation {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_uppercase().as_str() {
            "UNION" => Ok(Self::Union),
            "DIFFERENCE" => Ok(Self::Difference),
            "INTERSECT" => Ok(Self::Intersect),
            _ => Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                "boolean operation must be UNION, DIFFERENCE, or INTERSECT",
                serde_json::json!({"parameter":"operation","pointer":"/params/operation"}),
            )),
        }
    }
}

/// Blender `material_mode`: how operand material indices map to output material indices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MaterialMode {
    /// Each operand slot takes the subject slot holding the same material, or keeps its index.
    Index,
    /// The subject's slots come first, then each operand's new materials are appended to one pool.
    Transfer,
}

/// One operand of the N-ary EXACT Boolean, resolved by the evaluator in the subject's space.
#[derive(Clone, Debug, PartialEq)]
pub struct BooleanOperand {
    /// The operand's mesh in its object space.
    pub mesh: Mesh,
    /// Object space to subject space (Blender `world_to_object * operand_world`).
    pub to_subject: DMat4,
    /// Output material index for each local material index, from [`material_remaps`]. An empty
    /// list keeps every index.
    pub material_remap: Vec<u32>,
}

/// Blender `non_float_boolean_mesh` for the EXACT solver: `subject` is operand 0 with the identity
/// transform, and `operands` follow in order. One N-ary call, so a collection is never folded
/// pairwise. An empty `operands` runs the solver on the subject alone, which removes its
/// self-intersections (a collection with no members).
///
/// # Errors
/// Returns `INVALID_ARGUMENT` for an invalid operand mesh, and the errors of the exact pipeline
/// stages.
pub(crate) fn exact_boolean(
    subject: &Mesh,
    operands: &[BooleanOperand],
    operation: Operation,
    use_self: bool,
    hole_tolerant: bool,
) -> Result<Mesh> {
    let mut joined = Vec::with_capacity(operands.len() + 1);
    joined.push(Operand {
        mesh: subject,
        transform: DMat4::IDENTITY,
    });
    joined.extend(operands.iter().map(|operand| Operand {
        mesh: &operand.mesh,
        transform: operand.to_subject,
    }));
    // The subject keeps its own indices (empty remap); each operand uses its table.
    let mut remap = vec![Vec::new()];
    remap.extend(
        operands
            .iter()
            .map(|operand| operand.material_remap.clone()),
    );

    // Blender `mesh_id_for_face`: a triangle's `orig` is a global face index, mapped to the
    // operand whose face range holds it.
    let mut face_starts = Vec::with_capacity(joined.len());
    let mut face_total = 0_usize;
    for operand in &joined {
        face_starts.push(face_total);
        face_total += operand.mesh.faces.len();
    }
    let shape_of = |orig: i32| -> usize {
        usize::try_from(orig).map_or(0, |face| {
            face_starts
                .partition_point(|&start| start <= face)
                .saturating_sub(1)
        })
    };

    let mut arena = Arena::new();
    let tris = meshes_to_tris(&joined, &mut arena)?;
    let intersected = boolean_tris(
        &mut arena,
        &tris,
        joined.len(),
        &shape_of,
        bool_op(operation),
        use_self,
        hole_tolerant,
    )?;
    let polys = polymesh_from_tris(&arena, &intersected);
    polys_to_mesh(&polys, &arena, &joined, &OutputMaterials { remap: &remap })
}

/// Blender `BoolOpType` for an operation.
fn bool_op(operation: Operation) -> BoolOp {
    match operation {
        Operation::Union => BoolOp::Union,
        Operation::Difference => BoolOp::Difference,
        Operation::Intersect => BoolOp::Intersect,
    }
}

/// Blender `material_remaps` (`non_float_boolean_mesh`): the output slot index of each operand's
/// local slot, and the output node's slot list.
///
/// `subject` and every entry of `operands` list one object's material slots by material identity,
/// in slot order; `None` is an empty slot. `remaps` has one entry per operand, subject first. The
/// subject's entry is empty, so its indices stay. `pool` is the output node's slot list.
///
/// - INDEX: `pool` is the subject's slots. An operand slot maps to the subject slot at its index
///   when that slot holds the same material. Otherwise it maps to the first subject slot holding
///   the material (`BKE_object_material_remap_calc`, material.cc:1307-1351, and `Map::add`, which
///   keeps the first key, BLI_map.hh:289-295). Otherwise it keeps its local index when the subject
///   has that slot, and maps to 0 when it does not. An operand with no slots maps to `[0]`
///   (MOD_boolean.cc:366-375).
/// - TRANSFER: `pool` is the subject's distinct materials in first-occurrence order (Blender's
///   `VectorSet`, MOD_boolean.cc:422-430), or `[None]` when the subject has no slots. Each operand
///   slot maps to the pool index of its material, appending a material the pool lacks. An empty
///   operand slot maps to the pooled `None`. Blender writes `-1` there (`MOD_boolean.cc:387` and
///   `geometry_mesh_boolean.cc:276-280`), which `u32` cannot hold: both mean no material.
///
/// As in Blender, the subject keeps its own indices, so with duplicate subject slots a face index
/// can point past the end of `pool`.
#[must_use]
pub(crate) fn material_remaps<'a>(
    mode: MaterialMode,
    subject: &[Option<&'a Id>],
    operands: &[Vec<Option<&'a Id>>],
) -> (Vec<Vec<u32>>, Vec<Option<Id>>) {
    let mut pool: Vec<Option<Id>> = match mode {
        MaterialMode::Index => subject.iter().map(|&slot| slot.cloned()).collect(),
        MaterialMode::Transfer => transfer_pool(subject),
    };
    let mut remaps = vec![Vec::new()];
    for slots in operands {
        remaps.push(match mode {
            MaterialMode::Index => index_remap(subject, slots),
            MaterialMode::Transfer => transfer_remap(&mut pool, slots),
        });
    }
    (remaps, pool)
}

/// The TRANSFER pool: the subject's distinct materials in first-occurrence order, or one empty slot
/// when the subject has none.
fn transfer_pool(subject: &[Option<&Id>]) -> Vec<Option<Id>> {
    if subject.is_empty() {
        return vec![None];
    }
    let mut pool: Vec<Option<Id>> = Vec::with_capacity(subject.len());
    for &slot in subject {
        if !pool.iter().any(|pooled| pooled.as_ref() == slot) {
            pool.push(slot.cloned());
        }
    }
    pool
}

/// The INDEX remap of one operand (Blender `BKE_object_material_remap_calc`, see
/// [`material_remaps`]). An operand with no slots maps to `[0]`.
fn index_remap(subject: &[Option<&Id>], slots: &[Option<&Id>]) -> Vec<u32> {
    if slots.is_empty() {
        return vec![0];
    }
    slots
        .iter()
        .enumerate()
        .map(|(local, &slot)| {
            let index = if subject.get(local).copied() == Some(slot) {
                local
            } else if let Some(first) = subject.iter().position(|&held| held == slot) {
                first
            } else if local < subject.len() {
                local
            } else {
                0
            };
            slot_index(index)
        })
        .collect()
}

/// The TRANSFER remap of one operand: each slot's index in `pool`, appending the materials the
/// pool lacks (Blender `VectorSet::index_of_or_add`).
fn transfer_remap(pool: &mut Vec<Option<Id>>, slots: &[Option<&Id>]) -> Vec<u32> {
    slots
        .iter()
        .map(|&slot| {
            let index = pool
                .iter()
                .position(|pooled| pooled.as_ref() == slot)
                .unwrap_or_else(|| {
                    pool.push(slot.cloned());
                    pool.len() - 1
                });
            slot_index(index)
        })
        .collect()
}

/// Output material indices are slot numbers, which always fit in `u32`.
fn slot_index(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

#[derive(Clone, Debug)]
struct Polygon {
    vertices: Vec<DVec3>,
    material_index: u32,
    source_face: Option<usize>,
}

impl Polygon {
    fn plane(&self) -> Option<Plane> {
        let first = *self.vertices.first()?;
        for second_index in 1..self.vertices.len().saturating_sub(1) {
            let second = self.vertices[second_index];
            let third = self.vertices[second_index + 1];
            let normal = (second - first).cross(third - first);
            if normal.length_squared() > f64::MIN_POSITIVE {
                return Some(Plane {
                    a: first,
                    b: second,
                    c: third,
                    normal: normal.normalize(),
                });
            }
        }
        None
    }

    fn invert(&mut self) {
        self.vertices.reverse();
    }
}
#[derive(Clone, Debug)]
struct OutputPolygon {
    vertices: Vec<usize>,
    material_index: u32,
    source_face: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct Plane {
    a: DVec3,
    b: DVec3,
    c: DVec3,
    normal: DVec3,
}

impl Plane {
    fn invert(&mut self) {
        std::mem::swap(&mut self.a, &mut self.c);
        self.normal = -self.normal;
    }

    fn side(&self, point: DVec3, epsilon: f64) -> i8 {
        if self.distance(point).abs() <= epsilon {
            return 0;
        }
        let exact = orient3d(coord(self.a), coord(self.b), coord(self.c), coord(point));
        if exact == 0.0 {
            0
        } else if exact < 0.0 {
            1
        } else {
            -1
        }
    }

    fn distance(&self, point: DVec3) -> f64 {
        self.normal.dot(point - self.a)
    }

    fn split(
        &self,
        polygon: Polygon,
        epsilon: f64,
        coplanar_front: &mut Vec<Polygon>,
        coplanar_back: &mut Vec<Polygon>,
        front: &mut Vec<Polygon>,
        back: &mut Vec<Polygon>,
    ) {
        let sides: Vec<i8> = polygon
            .vertices
            .iter()
            .map(|point| self.side(*point, epsilon))
            .collect();
        let polygon_type = sides.iter().fold(0_i8, |kind, side| {
            kind | match side.cmp(&0) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => 2,
                std::cmp::Ordering::Equal => 0,
            }
        });
        match polygon_type {
            0 => {
                if self
                    .normal
                    .dot(polygon.plane().map_or(self.normal, |plane| plane.normal))
                    > 0.0
                {
                    coplanar_front.push(polygon);
                } else {
                    coplanar_back.push(polygon);
                }
            }
            1 => front.push(polygon),
            2 => back.push(polygon),
            _ => {
                let mut front_vertices = Vec::with_capacity(polygon.vertices.len() + 1);
                let mut back_vertices = Vec::with_capacity(polygon.vertices.len() + 1);
                for index in 0..polygon.vertices.len() {
                    let next = (index + 1) % polygon.vertices.len();
                    let first = polygon.vertices[index];
                    let second = polygon.vertices[next];
                    let first_side = sides[index];
                    let second_side = sides[next];
                    if first_side >= 0 {
                        front_vertices.push(first);
                    }
                    if first_side <= 0 {
                        back_vertices.push(first);
                    }
                    if first_side * second_side < 0 {
                        let first_distance = self.distance(first);
                        let second_distance = self.distance(second);
                        let denominator = first_distance - second_distance;
                        if denominator != 0.0 {
                            let t = (first_distance / denominator).clamp(0.0, 1.0);
                            let intersection = first + (second - first) * t;
                            front_vertices.push(intersection);
                            back_vertices.push(intersection);
                        }
                    }
                }
                if let Some(vertices) = clean_vertices(front_vertices, epsilon) {
                    front.push(Polygon {
                        vertices,
                        material_index: polygon.material_index,
                        source_face: polygon.source_face,
                    });
                }
                if let Some(vertices) = clean_vertices(back_vertices, epsilon) {
                    back.push(Polygon {
                        vertices,
                        material_index: polygon.material_index,
                        source_face: polygon.source_face,
                    });
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
struct BspNode {
    plane: Option<Plane>,
    polygons: Vec<Polygon>,
    front: Option<Box<Self>>,
    back: Option<Box<Self>>,
}

// Polygon splitting can produce deep, unbalanced trees; iterative traversals avoid exhausting the caller's stack.
impl BspNode {
    fn new(polygons: Vec<Polygon>, epsilon: f64) -> Self {
        let mut node = Self::default();
        node.build(polygons, epsilon);
        node
    }

    fn invert(&mut self) {
        let mut pending = vec![self];
        while let Some(node) = pending.pop() {
            for polygon in &mut node.polygons {
                polygon.invert();
            }
            if let Some(plane) = &mut node.plane {
                plane.invert();
            }
            std::mem::swap(&mut node.front, &mut node.back);
            if let Some(front) = node.front.as_deref_mut() {
                pending.push(front);
            }
            if let Some(back) = node.back.as_deref_mut() {
                pending.push(back);
            }
        }
    }

    fn clip_polygons(&self, polygons: Vec<Polygon>, epsilon: f64) -> Vec<Polygon> {
        enum Task<'a> {
            Visit {
                node: &'a BspNode,
                polygons: Vec<Polygon>,
            },
            Combine {
                has_front_child: bool,
                has_back_child: bool,
                front_direct: Option<Vec<Polygon>>,
                back_direct: Option<Vec<Polygon>>,
            },
        }

        let mut pending = vec![Task::Visit {
            node: self,
            polygons,
        }];
        let mut results = Vec::<Vec<Polygon>>::new();
        while let Some(task) = pending.pop() {
            match task {
                Task::Visit { node, polygons } => {
                    let Some(plane) = node.plane else {
                        results.push(polygons);
                        continue;
                    };
                    let mut front = Vec::new();
                    let mut back = Vec::new();
                    let mut coplanar_front = Vec::new();
                    let mut coplanar_back = Vec::new();
                    for polygon in polygons {
                        plane.split(
                            polygon,
                            epsilon,
                            &mut coplanar_front,
                            &mut coplanar_back,
                            &mut front,
                            &mut back,
                        );
                    }
                    front.extend(coplanar_front);
                    back.extend(coplanar_back);

                    let (front_direct, front_visit) = match node.front.as_deref() {
                        Some(child) => (None, Some((child, front))),
                        None => (Some(front), None),
                    };
                    let (back_direct, back_visit) = match node.back.as_deref() {
                        Some(child) => (None, Some((child, back))),
                        None => (Some(Vec::new()), None),
                    };
                    pending.push(Task::Combine {
                        has_front_child: front_visit.is_some(),
                        has_back_child: back_visit.is_some(),
                        front_direct,
                        back_direct,
                    });
                    if let Some((child, polygons)) = back_visit {
                        pending.push(Task::Visit {
                            node: child,
                            polygons,
                        });
                    }
                    if let Some((child, polygons)) = front_visit {
                        pending.push(Task::Visit {
                            node: child,
                            polygons,
                        });
                    }
                }
                Task::Combine {
                    has_front_child,
                    has_back_child,
                    front_direct,
                    back_direct,
                } => {
                    let mut back = if has_back_child {
                        results.pop().unwrap_or_default()
                    } else {
                        back_direct.unwrap_or_default()
                    };
                    let mut front = if has_front_child {
                        results.pop().unwrap_or_default()
                    } else {
                        front_direct.unwrap_or_default()
                    };
                    front.append(&mut back);
                    results.push(front);
                }
            }
        }
        results.pop().unwrap_or_default()
    }

    fn clip_to(&mut self, other: &Self, epsilon: f64) {
        let mut pending = vec![self];
        while let Some(node) = pending.pop() {
            node.polygons = other.clip_polygons(std::mem::take(&mut node.polygons), epsilon);
            if let Some(back) = node.back.as_deref_mut() {
                pending.push(back);
            }
            if let Some(front) = node.front.as_deref_mut() {
                pending.push(front);
            }
        }
    }

    fn all_polygons(&self) -> Vec<Polygon> {
        let mut polygons = Vec::new();
        let mut pending = vec![self];
        while let Some(node) = pending.pop() {
            polygons.extend(node.polygons.iter().cloned());
            if let Some(back) = node.back.as_deref() {
                pending.push(back);
            }
            if let Some(front) = node.front.as_deref() {
                pending.push(front);
            }
        }
        polygons
    }

    fn build(&mut self, polygons: Vec<Polygon>, epsilon: f64) {
        let mut pending = vec![(self, polygons)];
        while let Some((node, polygons)) = pending.pop() {
            if polygons.is_empty() {
                continue;
            }
            if node.plane.is_none() {
                node.plane = polygons.iter().find_map(Polygon::plane);
            }
            let Some(plane) = node.plane else {
                continue;
            };
            let mut front_polygons = Vec::new();
            let mut back_polygons = Vec::new();
            let mut coplanar_front = Vec::new();
            let mut coplanar_back = Vec::new();
            for polygon in polygons {
                plane.split(
                    polygon,
                    epsilon,
                    &mut coplanar_front,
                    &mut coplanar_back,
                    &mut front_polygons,
                    &mut back_polygons,
                );
            }
            node.polygons.extend(coplanar_front);
            node.polygons.extend(coplanar_back);
            if !back_polygons.is_empty() {
                let child = node.back.get_or_insert_with(|| Box::new(Self::default()));
                pending.push((child.as_mut(), back_polygons));
            }
            if !front_polygons.is_empty() {
                let child = node.front.get_or_insert_with(|| Box::new(Self::default()));
                pending.push((child.as_mut(), front_polygons));
            }
        }
    }
}

/// Computes a watertight polygon mesh Boolean using adaptive exact orientation tests.
///
/// `operation` is Blender's `UNION`, `DIFFERENCE`, or `INTERSECT` string. The operand
/// mesh is supplied separately so callers can resolve Blender object or collection operands
/// in their scene context.
///
/// # Errors
/// Returns `INVALID_ARGUMENT` for invalid meshes, degenerate output, or an unknown operation.
pub fn boolean_mesh(source: &Mesh, operand: &Mesh, operation: &str) -> Result<Mesh> {
    boolean_mesh_op(source, operand, Operation::parse(operation)?)
}

/// The FLOAT-solver BSP Boolean with the operation already parsed (see [`boolean_mesh`]).
///
/// shortcut: FLOAT is the pre-PR17 BSP until `PR17b` ports Blender's float path. EXACT never calls it.
/// Blender's exact Boolean path omits point CD_ORIGINDEX from its output, including shortcuts.
pub(crate) fn boolean_mesh_op(source: &Mesh, operand: &Mesh, operation: Operation) -> Result<Mesh> {
    let mut result = boolean_mesh_op_solve(source, operand, operation)?;
    result.attributes.remove(ORIGINDEX_ATTRIBUTE);
    Ok(result)
}

fn boolean_mesh_op_solve(source: &Mesh, operand: &Mesh, operation: Operation) -> Result<Mesh> {
    source.validate().map_err(|error| {
        PotError::invalid_argument(format!("invalid boolean source mesh: {error}"))
    })?;
    operand.validate().map_err(|error| {
        PotError::invalid_argument(format!("invalid boolean operand mesh: {error}"))
    })?;
    if source.faces.is_empty() {
        return if operation == Operation::Union {
            let mut result = operand.clone();
            clear_element_keyed_attributes(&mut result);
            Ok(result)
        } else {
            Ok(Mesh::new())
        };
    }
    if operand.faces.is_empty() {
        return if operation == Operation::Intersect {
            Ok(Mesh::new())
        } else {
            Ok(source.clone())
        };
    }
    validate_closed_mesh(source, "source")?;
    validate_closed_mesh(operand, "operand")?;
    if operation == Operation::Difference {
        let components = face_components(operand);
        if components.len() > 1 {
            let mut result = source.clone();
            for component_faces in components {
                let component = mesh_for_faces(operand, &component_faces)?;
                result = boolean_mesh_validated(&result, &component, operation)?;
                if result.faces.is_empty() {
                    break;
                }
            }
            return Ok(result);
        }
    }
    boolean_mesh_validated(source, operand, operation)
}

fn boolean_mesh_validated(source: &Mesh, operand: &Mesh, operation: Operation) -> Result<Mesh> {
    let mut source_polygons = mesh_polygons(source, false)?;
    let mut operand_polygons = mesh_polygons(operand, true)?;
    let epsilon = weld_tolerance(source, operand);
    let mut source_node = BspNode::new(std::mem::take(&mut source_polygons), epsilon);
    let mut operand_node = BspNode::new(std::mem::take(&mut operand_polygons), epsilon);
    match operation {
        Operation::Union => {
            source_node.clip_to(&operand_node, epsilon);
            operand_node.clip_to(&source_node, epsilon);
            operand_node.invert();
            operand_node.clip_to(&source_node, epsilon);
            operand_node.invert();
            source_node.build(operand_node.all_polygons(), epsilon);
            polygons_to_mesh(source_node.all_polygons(), source, epsilon)
        }
        Operation::Difference => {
            source_node.invert();
            source_node.clip_to(&operand_node, epsilon);
            operand_node.clip_to(&source_node, epsilon);
            operand_node.invert();
            operand_node.clip_to(&source_node, epsilon);
            operand_node.invert();
            source_node.build(operand_node.all_polygons(), epsilon);
            source_node.invert();
            polygons_to_mesh(source_node.all_polygons(), source, epsilon)
        }
        Operation::Intersect => {
            source_node.invert();
            operand_node.clip_to(&source_node, epsilon);
            operand_node.invert();
            source_node.clip_to(&operand_node, epsilon);
            operand_node.clip_to(&source_node, epsilon);
            source_node.build(operand_node.all_polygons(), epsilon);
            source_node.invert();
            polygons_to_mesh(source_node.all_polygons(), source, epsilon)
        }
    }
}

fn face_components(mesh: &Mesh) -> Vec<Vec<usize>> {
    let mut edge_faces = HashMap::<(u32, u32), Vec<usize>>::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            edge_faces
                .entry((first.min(second), first.max(second)))
                .or_default()
                .push(face_index);
        }
    }
    let mut adjacency = vec![Vec::new(); mesh.faces.len()];
    for owners in edge_faces.values() {
        if let [first, second] = owners.as_slice() {
            adjacency[*first].push(*second);
            adjacency[*second].push(*first);
        }
    }
    let mut visited = vec![false; mesh.faces.len()];
    let mut components = Vec::new();
    for start in 0..mesh.faces.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut pending = vec![start];
        let mut component = Vec::new();
        while let Some(face) = pending.pop() {
            component.push(face);
            for neighbor in &adjacency[face] {
                if !visited[*neighbor] {
                    visited[*neighbor] = true;
                    pending.push(*neighbor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

fn mesh_for_faces(mesh: &Mesh, faces: &[usize]) -> Result<Mesh> {
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut component = Mesh::new();
    let mut vertex_ids = HashMap::<u32, u32>::new();
    for face_index in faces {
        let face = &mesh.faces[*face_index];
        let mut vertices = Vec::with_capacity(face.vertices.len());
        for vertex_id in &face.vertices {
            let component_id = if let Some(component_id) = vertex_ids.get(vertex_id) {
                *component_id
            } else {
                let position = positions.get(vertex_id).copied().ok_or_else(|| {
                    PotError::invalid_argument("boolean component references a missing vertex")
                })?;
                let component_id = component.insert_vertex(position).map_err(|error| {
                    PotError::invalid_argument(format!("boolean component vertex failed: {error}"))
                })?;
                vertex_ids.insert(*vertex_id, component_id);
                component_id
            };
            vertices.push(component_id);
        }
        component
            .insert_face(vertices, face.material_index)
            .map_err(|error| {
                PotError::invalid_argument(format!("boolean component face failed: {error}"))
            })?;
    }
    Ok(component)
}
/// Clear B-only keys when the empty-source shortcut has no A provenance to remap.
fn clear_element_keyed_attributes(mesh: &mut Mesh) {
    for key in [
        "vertex_groups",
        "crease_edge",
        "blender_edge_flags",
        "skin_radii",
        "skin_roots",
        "blender_skin_vertices",
    ] {
        mesh.attributes.remove(key);
    }
    if let Some(entries) = mesh
        .attributes
        .get_mut("blender_attributes")
        .and_then(serde_json::Value::as_array_mut)
    {
        entries.retain(|entry| {
            !matches!(
                entry.get("domain").and_then(serde_json::Value::as_str),
                Some("POINT" | "EDGE" | "FACE" | "CORNER")
            )
        });
    }
}

fn validate_closed_mesh(mesh: &Mesh, role: &str) -> Result<()> {
    let mut incidences = HashMap::<(u32, u32), (usize, i32)>::new();
    let mut referenced_vertices = HashSet::new();
    for face in &mesh.faces {
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            referenced_vertices.insert(first);
            let direction = if first < second { 1 } else { -1 };
            let entry = incidences
                .entry((first.min(second), first.max(second)))
                .or_default();
            entry.0 += 1;
            entry.1 += direction;
        }
    }
    if referenced_vertices.len() != mesh.vertices.len()
        || incidences
            .values()
            .any(|(count, direction)| *count != 2 || *direction != 0)
    {
        return Err(PotError::invalid_argument(format!(
            "boolean {role} mesh must be a closed consistently oriented 2-manifold"
        )));
    }
    let face_edges: HashSet<_> = incidences.keys().copied().collect();
    if mesh.edges.iter().any(|edge| {
        !face_edges.contains(&(
            edge.vertices[0].min(edge.vertices[1]),
            edge.vertices[0].max(edge.vertices[1]),
        ))
    }) {
        return Err(PotError::invalid_argument(format!(
            "boolean {role} mesh cannot contain loose edges"
        )));
    }
    let volume = signed_volume(mesh)?;
    let extent = mesh
        .bounds()
        .map_or(0.0, |bounds| bounds.size().max_element());
    if !volume.is_finite() || volume.abs() <= extent.powi(3) * 1.0e-14 {
        return Err(PotError::invalid_argument(format!(
            "boolean {role} mesh must enclose a non-zero volume"
        )));
    }
    Ok(())
}

fn signed_volume(mesh: &Mesh) -> Result<f64> {
    let positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let origin = mesh
        .vertices
        .first()
        .map_or(DVec3::ZERO, |vertex| vertex.co);
    let mut volume = 0.0;
    for face in &mesh.faces {
        let first = positions.get(&face.vertices[0]).copied().ok_or_else(|| {
            PotError::invalid_argument("boolean face references a missing vertex")
        })? - origin;
        for index in 1..face.vertices.len() - 1 {
            let second = positions
                .get(&face.vertices[index])
                .copied()
                .ok_or_else(|| {
                    PotError::invalid_argument("boolean face references a missing vertex")
                })?
                - origin;
            let third = positions
                .get(&face.vertices[index + 1])
                .copied()
                .ok_or_else(|| {
                    PotError::invalid_argument("boolean face references a missing vertex")
                })?
                - origin;
            volume += first.dot(second.cross(third)) / 6.0;
        }
    }
    Ok(volume)
}

fn mesh_polygons(mesh: &Mesh, has_operand_b: bool) -> Result<Vec<Polygon>> {
    let positions: HashMap<u32, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let triangles = mesh.triangulate().map_err(|error| {
        PotError::invalid_argument(format!("cannot triangulate boolean mesh: {error}"))
    })?;
    let mut cursor = 0;
    let mut result = Vec::with_capacity(triangles.len());
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let triangle_count = face.vertices.len() - 2;
        for triangle in &triangles[cursor..cursor + triangle_count] {
            let vertices = triangle
                .iter()
                .map(|id| {
                    positions.get(id).copied().ok_or_else(|| {
                        PotError::invalid_argument("boolean face references a missing vertex")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if (vertices[1] - vertices[0])
                .cross(vertices[2] - vertices[0])
                .length_squared()
                <= f64::MIN_POSITIVE
            {
                return Err(PotError::invalid_argument(
                    "boolean mesh contains a degenerate triangle",
                ));
            }
            result.push(Polygon {
                vertices,
                material_index: face.material_index,
                source_face: (!has_operand_b).then_some(face_index),
            });
        }
        cursor += triangle_count;
    }
    if signed_volume(mesh)? < 0.0 {
        for polygon in &mut result {
            polygon.invert();
        }
    }
    Ok(result)
}

fn polygons_to_mesh(polygons: Vec<Polygon>, template: &Mesh, epsilon: f64) -> Result<Mesh> {
    let source_vertex_index: HashMap<u32, usize> = template
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut points = Vec::new();
    let mut point_origins = Vec::new();
    let mut faces = Vec::<OutputPolygon>::new();
    for polygon in polygons {
        let Some(vertices) = clean_vertices(polygon.vertices, epsilon) else {
            continue;
        };
        let source_face_vertices = polygon
            .source_face
            .and_then(|face_index| template.faces.get(face_index))
            .map(|face| face.vertices.as_slice());
        let mut source_vertex_origins = Vec::with_capacity(vertices.len());
        let indices = vertices
            .into_iter()
            .map(|point| {
                let source_vertex =
                    source_face_vertices
                        .into_iter()
                        .flatten()
                        .find_map(|vertex_id| {
                            let index = source_vertex_index.get(vertex_id).copied()?;
                            (template.vertices[index].co == point).then_some(index)
                        });
                source_vertex_origins.push(source_vertex);
                find_or_insert_point(point, &mut points, &mut point_origins, epsilon)
            })
            .collect::<Vec<_>>();
        if indices.len() >= 3 && polygon_area(&indices, &points) > epsilon * epsilon {
            for (index, source_vertex) in indices.iter().zip(source_vertex_origins) {
                if point_origins[*index].is_none() {
                    point_origins[*index] = source_vertex;
                }
            }
            faces.push(OutputPolygon {
                vertices: indices,
                material_index: polygon.material_index,
                source_face: polygon.source_face,
            });
        }
    }
    merge_coplanar_faces(&mut faces, &points, epsilon);
    let active_points: HashSet<usize> = faces
        .iter()
        .flat_map(|polygon| polygon.vertices.iter().copied())
        .collect();
    split_t_junctions(&points, &mut faces, &active_points, epsilon);
    let used_points: HashSet<usize> = faces
        .iter()
        .flat_map(|polygon| polygon.vertices.iter().copied())
        .collect();
    let mut point_remap = vec![usize::MAX; points.len()];
    let mut compact_points = Vec::with_capacity(used_points.len());
    let mut compact_vertex_origins = Vec::with_capacity(used_points.len());
    for (index, point) in points.into_iter().enumerate() {
        if used_points.contains(&index) {
            point_remap[index] = compact_points.len();
            compact_points.push(point);
            compact_vertex_origins.push(point_origins[index]);
        }
    }
    for polygon in &mut faces {
        for index in &mut polygon.vertices {
            *index = point_remap[*index];
        }
    }
    let mut mesh = Mesh::new();
    for point in compact_points {
        mesh.insert_vertex(point).map_err(|error| {
            PotError::invalid_argument(format!("boolean output vertex failed: {error}"))
        })?;
    }
    let mut output_face_origins = Vec::with_capacity(faces.len());
    for polygon in faces {
        let vertices = polygon
            .vertices
            .into_iter()
            .filter_map(|index| mesh.vertices.get(index).map(|vertex| vertex.id))
            .collect::<Vec<_>>();
        if vertices.len() >= 3 {
            mesh.insert_face(vertices, polygon.material_index)
                .map_err(|error| {
                    PotError::invalid_argument(format!("boolean output face failed: {error}"))
                })?;
            output_face_origins.push(polygon.source_face);
        }
    }
    // Blender interpolates each loop from its operand-A source face at the loop's position, so cut
    // vertices take the face's UVs by mean value; untouched vertices map to their own corner.
    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let vertex_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let face_vertices: Vec<Vec<usize>> = mesh
        .faces
        .iter()
        .map(|face| face.vertices.iter().map(|id| vertex_index[id]).collect())
        .collect();
    let corners = mean_value_corners(
        template,
        &positions,
        face_vertices
            .iter()
            .zip(&output_face_origins)
            .map(|(vertices, source)| (vertices.as_slice(), *source)),
    );
    let origin = Origin {
        corners: Some(corners),
        ..Origin::copies(compact_vertex_origins, output_face_origins, false)
    };
    remap_attributes(template, &mut mesh, &origin)?;
    mesh.validate().map_err(|error| {
        PotError::invalid_argument(format!("boolean output is invalid: {error}"))
    })?;
    if !mesh.faces.is_empty() {
        validate_closed_mesh(&mesh, "output")?;
    }
    Ok(mesh)
}

fn find_or_insert_point(
    point: DVec3,
    points: &mut Vec<DVec3>,
    point_origins: &mut Vec<Option<usize>>,
    tolerance: f64,
) -> usize {
    if let Some((index, _)) = points
        .iter()
        .enumerate()
        .find(|(_, candidate)| candidate.distance_squared(point) <= tolerance * tolerance)
    {
        index
    } else {
        points.push(point);
        point_origins.push(None);
        points.len() - 1
    }
}

fn merge_coplanar_faces(faces: &mut Vec<OutputPolygon>, points: &[DVec3], tolerance: f64) {
    loop {
        let mut shared_edges = BTreeMap::<(usize, usize), Vec<(usize, usize)>>::new();
        for (face_index, polygon) in faces.iter().enumerate() {
            let face = &polygon.vertices;
            for edge_index in 0..face.len() {
                let first = face[edge_index];
                let second = face[(edge_index + 1) % face.len()];
                shared_edges
                    .entry((first.min(second), first.max(second)))
                    .or_default()
                    .push((face_index, edge_index));
            }
        }
        let candidate = shared_edges.values().find_map(|owners| {
            if owners.len() != 2 || owners[0].0 == owners[1].0 {
                return None;
            }
            let (first_index, first_edge) = owners[0];
            let (second_index, second_edge) = owners[1];
            let first = &faces[first_index];
            let second = &faces[second_index];
            if first.material_index != second.material_index
                || first.vertices[first_edge]
                    != second.vertices[(second_edge + 1) % second.vertices.len()]
                || first.vertices[(first_edge + 1) % first.vertices.len()]
                    != second.vertices[second_edge]
                || !faces_are_coplanar(&first.vertices, &second.vertices, points, tolerance)
            {
                return None;
            }
            let merged_face =
                merge_face_loops(&first.vertices, first_edge, &second.vertices, second_edge);
            let merged_face = clean_face_loop(merged_face, points, tolerance)?;
            (polygon_area(&merged_face, points) > tolerance * tolerance).then_some((
                first_index,
                second_index,
                merged_face,
            ))
        });
        let Some((first_index, second_index, merged_face)) = candidate else {
            break;
        };
        let first = &faces[first_index];
        let second = &faces[second_index];
        let source_face = (first.source_face == second.source_face)
            .then_some(first.source_face)
            .flatten();
        let material_index = first.material_index;
        faces[first_index] = OutputPolygon {
            vertices: merged_face,
            material_index,
            source_face,
        };
        faces.remove(second_index);
    }
}

fn faces_are_coplanar(first: &[usize], second: &[usize], points: &[DVec3], tolerance: f64) -> bool {
    let Some((normal, origin)) = face_plane(first, points) else {
        return false;
    };
    let Some((other_normal, _)) = face_plane(second, points) else {
        return false;
    };
    if normal.distance_squared(other_normal) > 1.0e-20 {
        return false;
    }
    second
        .iter()
        .all(|index| normal.dot(points[*index] - origin).abs() <= tolerance * 8.0)
}

fn face_plane(face: &[usize], points: &[DVec3]) -> Option<(DVec3, DVec3)> {
    let origin = *points.get(*face.first()?)?;
    for index in 1..face.len() - 1 {
        let second = *points.get(face[index])?;
        let third = *points.get(face[index + 1])?;
        let normal = (second - origin).cross(third - origin);
        if normal.length_squared() > f64::MIN_POSITIVE {
            return Some((normal.normalize(), origin));
        }
    }
    None
}

fn merge_face_loops(
    first: &[usize],
    first_edge: usize,
    second: &[usize],
    second_edge: usize,
) -> Vec<usize> {
    let mut merged = Vec::with_capacity(first.len() + second.len() - 2);
    let mut index = (first_edge + 1) % first.len();
    loop {
        merged.push(first[index]);
        if index == first_edge {
            break;
        }
        index = (index + 1) % first.len();
    }
    index = (second_edge + 2) % second.len();
    while index != second_edge {
        merged.push(second[index]);
        index = (index + 1) % second.len();
    }
    merged
}

fn clean_face_loop(mut face: Vec<usize>, points: &[DVec3], tolerance: f64) -> Option<Vec<usize>> {
    face.dedup();
    if face.first() == face.last() {
        face.pop();
    }
    let mut changed = true;
    while changed && face.len() >= 3 {
        changed = false;
        for index in 0..face.len() {
            let previous = points[face[(index + face.len() - 1) % face.len()]];
            let current = points[face[index]];
            let next = points[face[(index + 1) % face.len()]];
            let scale = previous.distance(current) + current.distance(next);
            if (current - previous).cross(next - current).length() <= tolerance * scale {
                face.remove(index);
                changed = true;
                break;
            }
        }
    }
    (face.len() >= 3).then_some(face)
}

fn split_t_junctions(
    points: &[DVec3],
    faces: &mut [OutputPolygon],
    active_points: &HashSet<usize>,
    tolerance: f64,
) {
    for polygon in faces {
        let original = std::mem::take(&mut polygon.vertices);
        let mut split = Vec::new();
        for index in 0..original.len() {
            let first_index = original[index];
            let second_index = original[(index + 1) % original.len()];
            let first = points[first_index];
            let second = points[second_index];
            split.push(first_index);
            let edge = second - first;
            let length_squared = edge.length_squared();
            if length_squared <= tolerance * tolerance {
                continue;
            }
            let mut interior = points
                .iter()
                .enumerate()
                .filter_map(|(point_index, point)| {
                    if point_index == first_index
                        || point_index == second_index
                        || !active_points.contains(&point_index)
                    {
                        return None;
                    }
                    let parameter = (*point - first).dot(edge) / length_squared;
                    if parameter <= 0.0 || parameter >= 1.0 {
                        return None;
                    }
                    let projection = first + edge * parameter;
                    (projection.distance_squared(*point) <= tolerance * tolerance)
                        .then_some((parameter, point_index))
                })
                .collect::<Vec<_>>();
            interior.sort_by(|left, right| left.0.total_cmp(&right.0));
            split.extend(interior.into_iter().map(|(_, point_index)| point_index));
        }
        polygon.vertices = split;
    }
}

fn clean_vertices(vertices: Vec<DVec3>, tolerance: f64) -> Option<Vec<DVec3>> {
    let mut result = Vec::with_capacity(vertices.len());
    for vertex in vertices {
        if !result
            .iter()
            .any(|known: &DVec3| known.distance_squared(vertex) <= tolerance * tolerance)
        {
            result.push(vertex);
        }
    }
    let mut changed = true;
    while changed && result.len() >= 3 {
        changed = false;
        for index in 0..result.len() {
            let previous = result[(index + result.len() - 1) % result.len()];
            let current = result[index];
            let next = result[(index + 1) % result.len()];
            let scale = previous.distance(current) + current.distance(next);
            if (current - previous).cross(next - current).length() <= tolerance * scale {
                result.remove(index);
                changed = true;
                break;
            }
        }
    }
    (result.len() >= 3).then_some(result)
}

fn polygon_area(indices: &[usize], points: &[DVec3]) -> f64 {
    (1..indices.len() - 1)
        .map(|index| {
            (points[indices[index]] - points[indices[0]])
                .cross(points[indices[index + 1]] - points[indices[0]])
                .length()
                * 0.5
        })
        .sum()
}

fn weld_tolerance(source: &Mesh, operand: &Mesh) -> f64 {
    let extent = source
        .bounds()
        .into_iter()
        .chain(operand.bounds())
        .map(|bounds| bounds.size().max_element())
        .fold(0.0_f64, f64::max);
    let magnitude = source
        .vertices
        .iter()
        .chain(&operand.vertices)
        .map(|vertex| vertex.co.abs().max_element())
        .fold(0.0_f64, f64::max);
    (extent * 1.0e-10)
        .max(magnitude * f64::EPSILON * 2.0)
        .max(f64::MIN_POSITIVE)
}

fn coord(point: DVec3) -> Coord3D<f64> {
    Coord3D {
        x: point.x,
        y: point.y,
        z: point.z,
    }
}
