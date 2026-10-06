//! Exact-predicate polygon Boolean operations using a BSP boundary representation.
//!
//! Plane-side classification uses adaptive exact `orient3d` predicates outside a small,
//! scale-relative coplanar tolerance that absorbs transform roundoff on planar faces.
//! The same tolerance reunites coordinates introduced by floating-point edge interpolation.

use std::collections::{BTreeMap, HashMap, HashSet};

use glam::DVec3;
use robust::{Coord3D, orient3d};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Operation {
    Union,
    Difference,
    Intersect,
}

impl Operation {
    fn parse(value: &str) -> Result<Self> {
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

#[derive(Clone, Debug)]
struct Polygon {
    vertices: Vec<DVec3>,
    material_index: u32,
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
                    });
                }
                if let Some(vertices) = clean_vertices(back_vertices, epsilon) {
                    back.push(Polygon {
                        vertices,
                        material_index: polygon.material_index,
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
    source.validate().map_err(|error| {
        PotError::invalid_argument(format!("invalid boolean source mesh: {error}"))
    })?;
    operand.validate().map_err(|error| {
        PotError::invalid_argument(format!("invalid boolean operand mesh: {error}"))
    })?;
    let operation = Operation::parse(operation)?;
    if source.faces.is_empty() {
        return if operation == Operation::Union {
            Ok(operand.clone())
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
    let mut source_polygons = mesh_polygons(source)?;
    let mut operand_polygons = mesh_polygons(operand)?;
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
    component.attributes.clone_from(&mesh.attributes);
    Ok(component)
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

fn mesh_polygons(mesh: &Mesh) -> Result<Vec<Polygon>> {
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
    for face in &mesh.faces {
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
    let mut points = Vec::new();
    let mut faces = Vec::<(Vec<usize>, u32)>::new();
    for polygon in polygons {
        let Some(vertices) = clean_vertices(polygon.vertices, epsilon) else {
            continue;
        };
        let indices = vertices
            .into_iter()
            .map(|point| find_or_insert_point(point, &mut points, epsilon))
            .collect::<Vec<_>>();
        if indices.len() >= 3 && polygon_area(&indices, &points) > epsilon * epsilon {
            faces.push((indices, polygon.material_index));
        }
    }
    merge_coplanar_faces(&mut faces, &points, epsilon);
    let active_points: HashSet<usize> = faces
        .iter()
        .flat_map(|(face, _)| face.iter().copied())
        .collect();
    split_t_junctions(&points, &mut faces, &active_points, epsilon);
    let used_points: HashSet<usize> = faces
        .iter()
        .flat_map(|(face, _)| face.iter().copied())
        .collect();
    let mut point_remap = vec![usize::MAX; points.len()];
    let mut compact_points = Vec::with_capacity(used_points.len());
    for (index, point) in points.into_iter().enumerate() {
        if used_points.contains(&index) {
            point_remap[index] = compact_points.len();
            compact_points.push(point);
        }
    }
    for (face, _) in &mut faces {
        for index in face {
            *index = point_remap[*index];
        }
    }
    let points = compact_points;
    let mut mesh = Mesh::new();
    for point in points {
        mesh.insert_vertex(point).map_err(|error| {
            PotError::invalid_argument(format!("boolean output vertex failed: {error}"))
        })?;
    }
    for (indices, material_index) in faces {
        let vertices = indices
            .into_iter()
            .filter_map(|index| mesh.vertices.get(index).map(|vertex| vertex.id))
            .collect::<Vec<_>>();
        if vertices.len() >= 3 {
            mesh.insert_face(vertices, material_index)
                .map_err(|error| {
                    PotError::invalid_argument(format!("boolean output face failed: {error}"))
                })?;
        }
    }
    mesh.attributes.clone_from(&template.attributes);
    mesh.validate().map_err(|error| {
        PotError::invalid_argument(format!("boolean output is invalid: {error}"))
    })?;
    if !mesh.faces.is_empty() {
        validate_closed_mesh(&mesh, "output")?;
    }
    Ok(mesh)
}

fn find_or_insert_point(point: DVec3, points: &mut Vec<DVec3>, tolerance: f64) -> usize {
    if let Some((index, _)) = points
        .iter()
        .enumerate()
        .find(|(_, candidate)| candidate.distance_squared(point) <= tolerance * tolerance)
    {
        index
    } else {
        points.push(point);
        points.len() - 1
    }
}

fn merge_coplanar_faces(faces: &mut Vec<(Vec<usize>, u32)>, points: &[DVec3], tolerance: f64) {
    loop {
        let mut shared_edges = BTreeMap::<(usize, usize), Vec<(usize, usize)>>::new();
        for (face_index, (face, _)) in faces.iter().enumerate() {
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
            let (first, first_material) = &faces[first_index];
            let (second, second_material) = &faces[second_index];
            if first_material != second_material
                || first[first_edge] != second[(second_edge + 1) % second.len()]
                || first[(first_edge + 1) % first.len()] != second[second_edge]
                || !faces_are_coplanar(first, second, points, tolerance)
            {
                return None;
            }
            let merged_face = merge_face_loops(first, first_edge, second, second_edge);
            let merged_face = clean_face_loop(merged_face, points, tolerance)?;
            (polygon_area(&merged_face, points) > tolerance * tolerance).then_some((
                first_index,
                *first_material,
                second_index,
                merged_face,
            ))
        });
        let Some((first_index, material_index, second_index, merged_face)) = candidate else {
            break;
        };
        faces[first_index] = (merged_face, material_index);
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
    faces: &mut [(Vec<usize>, u32)],
    active_points: &HashSet<usize>,
    tolerance: f64,
) {
    for (face, _) in faces {
        let original = std::mem::take(face);
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
        *face = split;
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
