use glam::DVec3;
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Face, Mesh},
    model::Id,
};

use super::{ChangeKind, Engine, check_fields, resolve_node_targets};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "uv.unwrap" => unwrap(engine, operation),
        "uv.transform" => transform(engine, operation),
        "uv.pin" => pin(engine, operation),
        "uv.pack" => pack(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported UV operation `{name}`"),
            "/op",
        )),
    }
}

fn unwrap(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "elements", "scope", "method"],
        &["target"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    let method = operation
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("box");
    if !["box", "smart", "cube", "angle_based", "conformal"].contains(&method) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "method must be box, smart, cube, angle_based, or conformal",
            "/method",
        ));
    }
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let selected = selected_faces(mesh, elements.as_ref())?;
        let mut uv_map = read_uv_map(mesh)?;
        if matches!(method, "angle_based" | "conformal") {
            unwrap_advanced(mesh, &selected, &mut uv_map, method)?;
        } else {
            for face in mesh.faces.iter().filter(|face| selected.contains(&face.id)) {
                let corners = project_face(mesh, face, method)?;
                let pinned = uv_map
                    .get(&face.id)
                    .and_then(|face_uv| face_uv.pinned.clone())
                    .filter(|pinned| pinned.len() == corners.len());
                uv_map.insert(
                    face.id,
                    UvFace {
                        uv: corners,
                        pinned,
                    },
                );
            }
        }
        write_uv_map(mesh, &uv_map);
        Ok(())
    })
}

fn transform(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "elements",
            "scope",
            "translation",
            "scale",
            "rotation",
        ],
        &["target"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    let translation = read_vec2(
        engine,
        operation.get("translation"),
        [0.0, 0.0],
        "/translation",
    )?;
    let scale = read_vec2(engine, operation.get("scale"), [1.0, 1.0], "/scale")?;
    let rotation = read_number(engine, operation.get("rotation"), 0.0, "/rotation")?.to_radians();
    let (sin, cos) = rotation.sin_cos();
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let selected = selected_faces(mesh, elements.as_ref())?;
        let mut uv_map = read_uv_map(mesh)?;
        if uv_map.is_empty() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "uv.transform requires an existing UV map",
            ));
        }
        for (face_id, face_uv) in &mut uv_map {
            if !selected.contains(face_id) {
                continue;
            }
            for uv in &mut face_uv.uv {
                let u = uv[0] * scale[0];
                let v = uv[1] * scale[1];
                uv[0] = u.mul_add(cos, -v * sin) + translation[0];
                uv[1] = u.mul_add(sin, v * cos) + translation[1];
            }
        }
        write_uv_map(mesh, &uv_map);
        Ok(())
    })
}

fn pin(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "elements", "scope", "pinned"],
        &["target"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    let pinned = match operation.get("pinned") {
        None => true,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "pinned must be a boolean",
                "/pinned",
            ));
        }
    };
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let selected = selected_faces(mesh, elements.as_ref())?;
        let mut uv_map = read_uv_map(mesh)?;
        if uv_map.is_empty() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "uv.pin requires an existing UV map",
            ));
        }
        for face_id in selected {
            let face = mesh
                .faces
                .iter()
                .find(|face| face.id == face_id)
                .ok_or_else(|| {
                    PotError::new(ErrorCode::TargetNotFound, "selected UV face was not found")
                })?;
            let face_uv = uv_map.get_mut(&face_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    format!("uv.pin requires UV coordinates for face f{face_id}"),
                )
            })?;
            if face_uv.uv.len() != face.vertices.len() {
                return Err(PotError::new(
                    ErrorCode::InvalidOperation,
                    format!("UV corner count does not match face f{face_id}"),
                ));
            }
            face_uv.pinned = Some(vec![pinned; face_uv.uv.len()]);
        }
        write_uv_map(mesh, &uv_map);
        Ok(())
    })
}

fn pack(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "elements", "scope", "margin"],
        &["target"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    let margin = read_number(engine, operation.get("margin"), 0.0, "/margin")?;
    if !(0.0..=0.5).contains(&margin) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "margin must be between 0 and 0.5",
            "/margin",
        ));
    }
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let selected = selected_faces(mesh, elements.as_ref())?;
        let mut uv_map = read_uv_map(mesh)?;
        if uv_map.is_empty() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "uv.pack requires an existing UV map",
            ));
        }
        pack_selected_uvs(&mut uv_map, &selected, margin)?;
        write_uv_map(mesh, &uv_map);
        Ok(())
    })
}

fn selected_targets(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<(Vec<Id>, Option<Value>)> {
    let target = operation.get("target").ok_or_else(|| {
        engine.error(ErrorCode::InvalidOperation, "target is required", "/target")
    })?;
    let Some(target_object) = target.as_object() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target must be an object",
            "/target",
        ));
    };
    let mut target_without_elements = target_object.clone();
    let nested_elements = target_without_elements.remove("elements");
    if operation.contains_key("elements") && nested_elements.is_some() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "elements must appear in either target or operation",
            "/elements",
        ));
    }
    let elements = operation.get("elements").cloned().or(nested_elements);
    let targets = resolve_node_targets(engine, &Value::Object(target_without_elements), true)?;
    Ok((targets, elements))
}

fn edit_targets(
    engine: &mut Engine<'_>,
    targets: Vec<Id>,
    scope_value: Option<&Value>,
    mut edit: impl FnMut(&mut Mesh) -> Result<()>,
) -> Result<bool> {
    let scope = super::parse_scope(
        engine,
        scope_value,
        "/scope",
        "scope must be a string",
        "scope must be shared or single_user",
        false,
    )?;
    let mut node_data = Vec::with_capacity(targets.len());
    for target in &targets {
        let data_id = engine
            .doc
            .nodes
            .get(target)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("node `{target}` has no geometry data"),
                    "/target",
                )
            })?;
        node_data.push((target.clone(), data_id));
    }
    let shared = node_data
        .iter()
        .any(|(_, data_id)| super::data_user_count(engine, data_id) > 1);
    if shared && scope.is_none() {
        return Err(engine.error(
            ErrorCode::SharedDataRequiresScope,
            "shared geometry edits require scope",
            "/scope",
        ));
    }
    if scope == Some("single_user") {
        for (target, data_id) in &node_data {
            if super::data_user_count(engine, data_id) <= 1 {
                continue;
            }
            let copy = engine
                .doc
                .data_blocks
                .get(data_id)
                .cloned()
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::TargetNotFound,
                        format!("data block `{data_id}` was not found"),
                    )
                })?;
            let new_id = super::unique_data_id(
                engine,
                target,
                "_mesh_single",
                super::DataIdCollisionHandling::ReserveSuffix,
                "data-block ID suffix exhausted",
            )?;
            engine.doc.data_blocks.insert(new_id.clone(), copy);
            if let Some(node) = engine.doc.nodes.get_mut(target) {
                node.data = Some(new_id.clone());
            }
            engine.mark("data_blocks", &new_id, ChangeKind::Created);
            engine.mark("nodes", target, ChangeKind::Updated);
            engine.map_id("data_blocks", data_id, &new_id);
        }
    }
    let mut changed = false;
    let mut edited_data = Vec::new();
    for target in targets {
        let data_id = engine
            .doc
            .nodes
            .get(&target)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    format!("node `{target}` has no geometry data"),
                )
            })?;
        if scope == Some("shared") && edited_data.contains(&data_id) {
            continue;
        }
        let (data_changed, had_descriptor) = {
            let data = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::TargetNotFound,
                    format!("data block `{data_id}` was not found"),
                )
            })?;
            let mesh = data.mesh.as_mut().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    format!("data block `{data_id}` has no mesh"),
                )
            })?;
            let before = mesh.clone();
            let had_descriptor = data.descriptor.is_some();
            edit(mesh)?;
            let changed = *mesh != before;
            if changed {
                data.descriptor = None;
            }
            (changed, had_descriptor)
        };
        if data_changed {
            if had_descriptor {
                engine
                    .id_mappings
                    .entry("mesh_descriptor_dropped".to_owned())
                    .or_insert_with(|| json!({}))[data_id.as_str()] = Value::Bool(true);
            }
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            super::clear_primitive_metadata(engine, &data_id);
            changed = true;
        }
        edited_data.push(data_id);
    }
    Ok(changed)
}

fn selected_faces(mesh: &Mesh, elements: Option<&Value>) -> Result<Vec<u32>> {
    let Some(elements) = elements else {
        return Ok(mesh.faces.iter().map(|face| face.id).collect());
    };
    let object = elements
        .as_object()
        .ok_or_else(|| PotError::new(ErrorCode::InvalidOperation, "elements must be an object"))?;
    if object.get("domain").and_then(Value::as_str) != Some("face") {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV elements domain must be face",
        ));
    }
    let ids = object.get("ids").and_then(Value::as_array).ok_or_else(|| {
        PotError::new(ErrorCode::InvalidOperation, "elements.ids must be an array")
    })?;
    let mut selected = Vec::with_capacity(ids.len());
    for value in ids {
        let text = value.as_str().ok_or_else(|| {
            PotError::new(ErrorCode::InvalidOperation, "element IDs must be strings")
        })?;
        let Some(id_text) = text.strip_prefix('f') else {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "face element IDs must start with f",
            ));
        };
        let id = id_text
            .parse::<u32>()
            .map_err(|_| PotError::new(ErrorCode::InvalidOperation, "invalid face element ID"))?;
        if !mesh.faces.iter().any(|face| face.id == id) {
            return Err(PotError::new(
                ErrorCode::TargetNotFound,
                format!("face element `{text}` was not found"),
            ));
        }
        if !selected.contains(&id) {
            selected.push(id);
        }
    }
    Ok(selected)
}

#[derive(Clone, Copy)]
struct UnwrapTriangle {
    vertices: [usize; 3],
}

struct UnwrapChart {
    face_indices: Vec<usize>,
    corner_indices: std::collections::HashMap<(usize, usize), usize>,
    vertex_ids: Vec<u32>,
    positions: Vec<DVec3>,
    triangles: Vec<UnwrapTriangle>,
}

fn unwrap_advanced(mesh: &Mesh, selected: &[u32], uv_map: &mut UvMap, method: &str) -> Result<()> {
    let selected_set = selected
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let faces = mesh
        .faces
        .iter()
        .filter(|face| selected_set.contains(&face.id))
        .collect::<Vec<_>>();
    if faces.is_empty() {
        return Ok(());
    }

    // Marked seam edges form chart boundaries; unmarked selected edges join faces.
    let seam_edges = seam_edge_keys(mesh)?;
    let mut edge_faces = std::collections::BTreeMap::<(u32, u32), Vec<usize>>::new();
    for (face_index, face) in faces.iter().enumerate() {
        if face.vertices.len() < 3 {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "UV unwrap encountered a face with fewer than three vertices",
            ));
        }
        for corner in 0..face.vertices.len() {
            let edge = crate::geom::edge_key(
                face.vertices[corner],
                face.vertices[(corner + 1) % face.vertices.len()],
            );
            edge_faces.entry(edge).or_default().push(face_index);
        }
    }
    let mut neighbors = vec![Vec::new(); faces.len()];
    for (edge, incident_faces) in edge_faces {
        if seam_edges.contains(&edge) {
            continue;
        }
        for &left in &incident_faces {
            for &right in &incident_faces {
                if left != right && !neighbors[left].contains(&right) {
                    neighbors[left].push(right);
                }
            }
        }
    }

    let mut visited = vec![false; faces.len()];
    for seed in 0..faces.len() {
        if visited[seed] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::from([seed]);
        visited[seed] = true;
        while let Some(face_index) = queue.pop_front() {
            component.push(face_index);
            for &neighbor in &neighbors[face_index] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        let chart = build_unwrap_chart(mesh, &faces, component, &seam_edges)?;
        let initial = unfold_chart(&chart)?;
        let pins = chart_pins(&faces, &chart, uv_map)?;
        let coordinates = if method == "conformal" || !pins.is_empty() {
            let anchors = chart_anchors(&chart, &pins, &initial)?;
            solve_lscm(&chart, &anchors)?
        } else {
            initial
        };

        for &face_index in &chart.face_indices {
            let face = faces[face_index];
            let corners = face
                .vertices
                .iter()
                .enumerate()
                .map(|(corner, _)| coordinates[chart.corner_indices[&(face_index, corner)]])
                .collect::<Vec<_>>();
            let pinned = uv_map
                .get(&face.id)
                .and_then(|face_uv| face_uv.pinned.clone());
            uv_map.insert(
                face.id,
                UvFace {
                    uv: corners,
                    pinned,
                },
            );
        }
    }
    Ok(())
}

fn seam_edge_keys(mesh: &Mesh) -> Result<std::collections::HashSet<(u32, u32)>> {
    let Some(attribute) = mesh.attributes.get("seam") else {
        return Ok(std::collections::HashSet::new());
    };
    if attribute.get("domain").and_then(Value::as_str) != Some("edge") {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "seam attribute must use the edge domain",
        ));
    }
    let values = attribute
        .get("values")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "seam attribute values must be an object",
            )
        })?;
    for value in values.values() {
        if !value.is_boolean() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "seam edge values must be boolean",
            ));
        }
    }
    let marked = mesh
        .edges
        .iter()
        .filter_map(|edge| {
            values
                .get(&format!("e{}", edge.id))
                .and_then(Value::as_bool)
                .filter(|is_seam| *is_seam)
                .map(|_| crate::geom::edge_key(edge.vertices[0], edge.vertices[1]))
        })
        .collect();
    Ok(marked)
}

fn corner_component_root(parents: &mut [usize], index: usize) -> usize {
    let mut root = index;
    while parents[root] != root {
        root = parents[root];
    }
    let mut current = index;
    while parents[current] != current {
        let next = parents[current];
        parents[current] = root;
        current = next;
    }
    root
}

fn join_corner_components(parents: &mut [usize], left: usize, right: usize) {
    let left_root = corner_component_root(parents, left);
    let right_root = corner_component_root(parents, right);
    if left_root != right_root {
        let (root, child) = if left_root < right_root {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        parents[child] = root;
    }
}

fn build_unwrap_chart(
    mesh: &Mesh,
    faces: &[&Face],
    face_indices: Vec<usize>,
    seam_edges: &std::collections::HashSet<(u32, u32)>,
) -> Result<UnwrapChart> {
    let mut corner_offsets = std::collections::HashMap::new();
    let mut edge_faces = std::collections::BTreeMap::<(u32, u32), Vec<usize>>::new();
    for &face_index in &face_indices {
        let face = faces[face_index];
        if face.vertices.len() < 3 {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "UV unwrap encountered a face with fewer than three vertices",
            ));
        }
        for (corner, &vertex_id) in face.vertices.iter().enumerate() {
            corner_offsets.insert((face_index, corner), corner_offsets.len());
            edge_faces
                .entry(crate::geom::edge_key(
                    vertex_id,
                    face.vertices[(corner + 1) % face.vertices.len()],
                ))
                .or_default()
                .push(face_index);
        }
    }

    let mut parents = (0..corner_offsets.len()).collect::<Vec<_>>();
    for (edge, incident_faces) in edge_faces {
        if seam_edges.contains(&edge) {
            continue;
        }
        for left_index in 0..incident_faces.len() {
            for right_index in left_index + 1..incident_faces.len() {
                let left_face_index = incident_faces[left_index];
                let right_face_index = incident_faces[right_index];
                for vertex_id in [edge.0, edge.1] {
                    let left_corner = faces[left_face_index]
                        .vertices
                        .iter()
                        .position(|candidate| *candidate == vertex_id)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "UV edge references a missing face corner",
                            )
                        })?;
                    let right_corner = faces[right_face_index]
                        .vertices
                        .iter()
                        .position(|candidate| *candidate == vertex_id)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "UV edge references a missing face corner",
                            )
                        })?;
                    join_corner_components(
                        &mut parents,
                        corner_offsets[&(left_face_index, left_corner)],
                        corner_offsets[&(right_face_index, right_corner)],
                    );
                }
            }
        }
    }

    let mut root_indices = std::collections::HashMap::new();
    let mut corner_indices = std::collections::HashMap::new();
    let mut vertex_ids = Vec::new();
    let mut positions = Vec::new();
    for &face_index in &face_indices {
        for (corner, &vertex_id) in faces[face_index].vertices.iter().enumerate() {
            let root = corner_component_root(&mut parents, corner_offsets[&(face_index, corner)]);
            let local_index = if let Some(index) = root_indices.get(&root) {
                *index
            } else {
                let position = mesh
                    .vertex(vertex_id)
                    .map(|vertex| vertex.co)
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "face references a missing vertex")
                    })?;
                let index = vertex_ids.len();
                vertex_ids.push(vertex_id);
                positions.push(position);
                root_indices.insert(root, index);
                index
            };
            corner_indices.insert((face_index, corner), local_index);
        }
    }

    let mut triangles = Vec::new();
    for &face_index in &face_indices {
        let face = faces[face_index];
        let local_face = (0..face.vertices.len())
            .map(|corner| corner_indices[&(face_index, corner)])
            .collect::<Vec<_>>();
        for corner in 1..local_face.len() - 1 {
            triangles.push(UnwrapTriangle {
                vertices: [local_face[0], local_face[corner], local_face[corner + 1]],
            });
        }
    }
    Ok(UnwrapChart {
        face_indices,
        corner_indices,
        vertex_ids,
        positions,
        triangles,
    })
}

// Intrinsically unfold each triangle across a spanning traversal of its island.
fn unfold_chart(chart: &UnwrapChart) -> Result<Vec<[f64; 2]>> {
    let Some(seed) = chart.triangles.first() else {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap requires at least one triangle",
        ));
    };
    let mut coordinates = vec![None; chart.vertex_ids.len()];
    place_seed_triangle(chart, seed, &mut coordinates, [0.0, 0.0])?;

    let mut edge_triangles = std::collections::BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (triangle_index, triangle) in chart.triangles.iter().enumerate() {
        for [left, right] in [
            [triangle.vertices[0], triangle.vertices[1]],
            [triangle.vertices[1], triangle.vertices[2]],
            [triangle.vertices[2], triangle.vertices[0]],
        ] {
            let edge = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            edge_triangles.entry(edge).or_default().push(triangle_index);
        }
    }
    let mut queue = std::collections::VecDeque::from([0usize]);
    let mut visited = vec![false; chart.triangles.len()];
    visited[0] = true;
    while let Some(triangle_index) = queue.pop_front() {
        let triangle = chart.triangles[triangle_index];
        for [left, right] in [
            [triangle.vertices[0], triangle.vertices[1]],
            [triangle.vertices[1], triangle.vertices[2]],
            [triangle.vertices[2], triangle.vertices[0]],
        ] {
            let edge = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            let Some(adjacent) = edge_triangles.get(&edge) else {
                continue;
            };
            for &neighbor_index in adjacent {
                if neighbor_index == triangle_index || visited[neighbor_index] {
                    continue;
                }
                let neighbor = chart.triangles[neighbor_index];
                let third = neighbor
                    .vertices
                    .iter()
                    .copied()
                    .find(|vertex| *vertex != left && *vertex != right)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "UV triangulation contains a repeated vertex",
                        )
                    })?;
                if coordinates[third].is_none() {
                    let current_third = triangle
                        .vertices
                        .iter()
                        .copied()
                        .find(|vertex| *vertex != left && *vertex != right)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "UV triangulation contains a repeated vertex",
                            )
                        })?;
                    place_unfolded_vertex(
                        chart,
                        left,
                        right,
                        current_third,
                        third,
                        &mut coordinates,
                    )?;
                }
                visited[neighbor_index] = true;
                queue.push_back(neighbor_index);
            }
        }
    }
    if coordinates.iter().any(Option::is_none) || visited.iter().any(|visited| !visited) {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV island has disconnected triangulation",
        ));
    }
    Ok(coordinates.into_iter().map(Option::unwrap).collect())
}

fn place_seed_triangle(
    chart: &UnwrapChart,
    triangle: &UnwrapTriangle,
    coordinates: &mut [Option<[f64; 2]>],
    origin: [f64; 2],
) -> Result<()> {
    let [first, second, third] = triangle.vertices;
    let edge = chart.positions[second] - chart.positions[first];
    let edge_length = edge.length();
    let to_third = chart.positions[third] - chart.positions[first];
    let cross_length = edge.cross(to_third).length();
    if !edge_length.is_finite() || !cross_length.is_finite() || edge_length <= 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap encountered a degenerate triangle",
        ));
    }
    let x = to_third.dot(edge / edge_length);
    let height = cross_length / edge_length;
    if !height.is_finite() || height <= 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap encountered a degenerate triangle",
        ));
    }
    coordinates[first] = Some(origin);
    coordinates[second] = Some([origin[0] + edge_length, origin[1]]);
    coordinates[third] = Some([origin[0] + x, origin[1] + height]);
    Ok(())
}

fn place_unfolded_vertex(
    chart: &UnwrapChart,
    left: usize,
    right: usize,
    current_third: usize,
    next_third: usize,
    coordinates: &mut [Option<[f64; 2]>],
) -> Result<()> {
    let a = coordinates[left].ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap lost an edge endpoint",
        )
    })?;
    let b = coordinates[right].ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap lost an edge endpoint",
        )
    })?;
    let c = coordinates[current_third].ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap lost a neighboring triangle",
        )
    })?;
    let edge = chart.positions[right] - chart.positions[left];
    let edge_length = edge.length();

    let to_next = chart.positions[next_third] - chart.positions[left];
    let cross_length = edge.cross(to_next).length();
    if !edge_length.is_finite() || !cross_length.is_finite() || edge_length <= 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap encountered a degenerate triangle",
        ));
    }
    let x = to_next.dot(edge / edge_length);
    let height = cross_length / edge_length;
    if !height.is_finite() || height <= 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap encountered a degenerate triangle",
        ));
    }
    let uv_edge = [b[0] - a[0], b[1] - a[1]];
    let uv_length = uv_edge[0].hypot(uv_edge[1]);
    let side = uv_edge[0] * (c[1] - a[1]) - uv_edge[1] * (c[0] - a[0]);
    if uv_length <= 0.0 || !uv_length.is_finite() || side == 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap encountered a collapsed triangle edge",
        ));
    }
    let unit = [uv_edge[0] / uv_length, uv_edge[1] / uv_length];
    let perpendicular = if side > 0.0 {
        [unit[1], -unit[0]]
    } else {
        [-unit[1], unit[0]]
    };
    coordinates[next_third] = Some([
        a[0] + unit[0] * (x * uv_length / edge_length)
            + perpendicular[0] * (height * uv_length / edge_length),
        a[1] + unit[1] * (x * uv_length / edge_length)
            + perpendicular[1] * (height * uv_length / edge_length),
    ]);

    Ok(())
}

fn chart_pins(
    faces: &[&Face],
    chart: &UnwrapChart,
    uv_map: &UvMap,
) -> Result<std::collections::BTreeMap<usize, [f64; 2]>> {
    let mut pins = std::collections::BTreeMap::<usize, [f64; 2]>::new();
    for &face_index in &chart.face_indices {
        let face = faces[face_index];
        let Some(face_uv) = uv_map.get(&face.id) else {
            continue;
        };
        if face_uv.uv.len() != face.vertices.len() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                format!("UV corner count does not match face f{}", face.id),
            ));
        }
        let Some(pinned) = &face_uv.pinned else {
            continue;
        };
        if pinned.len() != face.vertices.len() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                format!("UV pin count does not match face f{}", face.id),
            ));
        }
        for (corner, (&vertex_id, &is_pinned)) in
            face.vertices.iter().zip(pinned.iter()).enumerate()
        {
            if !is_pinned {
                continue;
            }
            let vertex_index = chart.corner_indices[&(face_index, corner)];
            let coordinate = face_uv.uv[corner];
            if let Some(previous) = pins.get(&vertex_index) {
                let scale = previous[0]
                    .abs()
                    .max(previous[1].abs())
                    .max(coordinate[0].abs())
                    .max(coordinate[1].abs())
                    .max(1.0);
                if (previous[0] - coordinate[0]).hypot(previous[1] - coordinate[1])
                    > scale * 1.0e-10
                {
                    return Err(PotError::new(
                        ErrorCode::InvalidOperation,
                        format!("pinned UVs disagree at vertex v{vertex_id}"),
                    ));
                }
            } else {
                pins.insert(vertex_index, coordinate);
            }
        }
    }
    Ok(pins)
}

fn chart_anchors(
    chart: &UnwrapChart,
    pins: &std::collections::BTreeMap<usize, [f64; 2]>,
    initial: &[[f64; 2]],
) -> Result<std::collections::BTreeMap<usize, [f64; 2]>> {
    let mut anchors = pins.clone();
    if anchors.len() >= 2 {
        return Ok(anchors);
    }
    if chart.positions.len() < 2 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap requires at least two distinct vertices",
        ));
    }
    if let Some((&fixed_index, &fixed_uv)) = anchors.iter().next() {
        let other = (0..chart.positions.len())
            .filter(|index| *index != fixed_index)
            .max_by(|left, right| {
                chart.positions[*left]
                    .distance_squared(chart.positions[fixed_index])
                    .total_cmp(
                        &chart.positions[*right].distance_squared(chart.positions[fixed_index]),
                    )
            })
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    "UV unwrap could not place an anchor",
                )
            })?;
        anchors.insert(
            other,
            [
                fixed_uv[0] + initial[other][0] - initial[fixed_index][0],
                fixed_uv[1] + initial[other][1] - initial[fixed_index][1],
            ],
        );
    } else {
        let mut pair = (0, 1);
        let mut farthest = 0.0;
        for left in 0..chart.positions.len() {
            for right in left + 1..chart.positions.len() {
                let distance = chart.positions[left].distance_squared(chart.positions[right]);
                if distance > farthest {
                    farthest = distance;
                    pair = (left, right);
                }
            }
        }
        if farthest <= 0.0 || !farthest.is_finite() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "UV unwrap requires distinct anchor vertices",
            ));
        }
        anchors.insert(pair.0, initial[pair.0]);
        anchors.insert(pair.1, initial[pair.1]);
    }
    Ok(anchors)
}

struct LinearRow {
    terms: [(usize, f64); 6],
    term_count: usize,
    rhs: f64,
}

fn solve_lscm(
    chart: &UnwrapChart,
    anchors: &std::collections::BTreeMap<usize, [f64; 2]>,
) -> Result<Vec<[f64; 2]>> {
    let mut fixed = vec![None; chart.vertex_ids.len() * 2];
    for (&vertex_index, uv) in anchors {
        fixed[2 * vertex_index] = Some(uv[0]);
        fixed[2 * vertex_index + 1] = Some(uv[1]);
    }
    let mut free_indices = vec![None; fixed.len()];
    let mut free_count = 0;
    for (dof, value) in fixed.iter().enumerate() {
        if value.is_none() {
            free_indices[dof] = Some(free_count);
            free_count += 1;
        }
    }

    // Weighted Cauchy-Riemann residuals define the LSCM least-squares system.
    let mut rows = Vec::with_capacity(chart.triangles.len() * 2);
    for triangle in &chart.triangles {
        let [first, second, third] = triangle.vertices;
        let edge = chart.positions[second] - chart.positions[first];
        let edge_length = edge.length();
        let to_third = chart.positions[third] - chart.positions[first];
        let x = to_third.dot(edge / edge_length);
        let y = edge.cross(to_third).length() / edge_length;
        let area = edge_length * y * 0.5;
        if !edge_length.is_finite() || !y.is_finite() || area <= 0.0 {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "conformal unwrap encountered a degenerate triangle",
            ));
        }
        let gradients = [
            [-1.0 / edge_length, (x - edge_length) / (edge_length * y)],
            [1.0 / edge_length, -x / (edge_length * y)],
            [0.0, 1.0 / y],
        ];
        let weight = area.sqrt();
        let mut first_row = [(0usize, 0.0); 6];
        let mut second_row = [(0usize, 0.0); 6];
        for corner in 0..3 {
            let vertex_index = triangle.vertices[corner];
            first_row[2 * corner] = (2 * vertex_index, gradients[corner][0] * weight);
            first_row[2 * corner + 1] = (2 * vertex_index + 1, -gradients[corner][1] * weight);
            second_row[2 * corner] = (2 * vertex_index, gradients[corner][1] * weight);
            second_row[2 * corner + 1] = (2 * vertex_index + 1, gradients[corner][0] * weight);
        }
        rows.push(reduce_lscm_row(first_row, &fixed, &free_indices));
        rows.push(reduce_lscm_row(second_row, &fixed, &free_indices));
    }
    let mut diagonal = vec![0.0; free_count];
    let mut right_hand_side = vec![0.0; free_count];
    for row in &rows {
        for &(index, coefficient) in &row.terms[..row.term_count] {
            diagonal[index] += coefficient * coefficient;
            right_hand_side[index] += coefficient * row.rhs;
        }
    }
    if diagonal
        .iter()
        .any(|value| *value <= 0.0 || !value.is_finite())
    {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "conformal unwrap has unconstrained UV coordinates",
        ));
    }
    let solution = conjugate_gradient(&rows, &diagonal, &right_hand_side)?;
    let mut coordinates = vec![[0.0; 2]; chart.vertex_ids.len()];
    for (vertex_index, coordinate) in coordinates.iter_mut().enumerate() {
        for (axis, value) in coordinate.iter_mut().enumerate() {
            let dof = 2 * vertex_index + axis;
            *value = if let Some(value) = fixed[dof] {
                value
            } else {
                solution[free_indices[dof].ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InvalidOperation,
                        "conformal unwrap lost a UV value",
                    )
                })?]
            };
        }
    }
    if coordinates
        .iter()
        .flatten()
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "conformal unwrap produced non-finite UV coordinates",
        ));
    }
    Ok(coordinates)
}

fn reduce_lscm_row(
    coefficients: [(usize, f64); 6],
    fixed: &[Option<f64>],
    free_indices: &[Option<usize>],
) -> LinearRow {
    let mut row = LinearRow {
        terms: [(0usize, 0.0); 6],
        term_count: 0,
        rhs: 0.0,
    };
    for (dof, coefficient) in coefficients {
        if let Some(value) = fixed[dof] {
            row.rhs -= coefficient * value;
        } else if let Some(index) = free_indices[dof] {
            row.terms[row.term_count] = (index, coefficient);
            row.term_count += 1;
        }
    }
    row
}

fn conjugate_gradient(
    rows: &[LinearRow],
    diagonal: &[f64],
    right_hand_side: &[f64],
) -> Result<Vec<f64>> {
    let mut solution = vec![0.0; diagonal.len()];
    if diagonal.is_empty() {
        return Ok(solution);
    }
    let mut residual = right_hand_side.to_vec();
    let mut preconditioned = residual
        .iter()
        .zip(diagonal)
        .map(|(value, diagonal)| value / diagonal)
        .collect::<Vec<_>>();
    let mut direction = preconditioned.clone();
    let mut residual_preconditioned = dot(&residual, &preconditioned);
    let rhs_norm = dot(right_hand_side, right_hand_side).sqrt();
    let tolerance = rhs_norm.max(1.0) * 1.0e-11;
    let max_iterations = diagonal.len().saturating_mul(4).clamp(64, 20_000);
    let mut product = vec![0.0; diagonal.len()];
    for _ in 0..max_iterations {
        if dot(&residual, &residual).sqrt() <= tolerance {
            return Ok(solution);
        }
        normal_matrix_product(rows, &direction, &mut product);
        let denominator = dot(&direction, &product);
        if denominator <= 0.0 || !denominator.is_finite() {
            break;
        }
        let alpha = residual_preconditioned / denominator;
        for index in 0..solution.len() {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
            preconditioned[index] = residual[index] / diagonal[index];
        }
        let next_residual_preconditioned = dot(&residual, &preconditioned);
        if !next_residual_preconditioned.is_finite() {
            break;
        }
        let beta = next_residual_preconditioned / residual_preconditioned;
        for index in 0..direction.len() {
            direction[index] = preconditioned[index] + beta * direction[index];
        }
        residual_preconditioned = next_residual_preconditioned;
    }
    if dot(&residual, &residual).sqrt() <= tolerance * 10.0 {
        Ok(solution)
    } else {
        Err(PotError::new(
            ErrorCode::InvalidOperation,
            "conformal unwrap did not converge",
        ))
    }
}

fn normal_matrix_product(rows: &[LinearRow], vector: &[f64], output: &mut [f64]) {
    output.fill(0.0);
    for row in rows {
        let projected = row.terms[..row.term_count]
            .iter()
            .map(|(index, coefficient)| coefficient * vector[*index])
            .sum::<f64>();
        for &(index, coefficient) in &row.terms[..row.term_count] {
            output[index] += coefficient * projected;
        }
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
fn project_face(mesh: &Mesh, face: &Face, method: &str) -> Result<Vec<[f64; 2]>> {
    let points = face
        .vertices
        .iter()
        .map(|id| {
            mesh.vertex(*id).map(|vertex| vertex.co).ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "face references a missing vertex")
            })
        })
        .collect::<Result<Vec<DVec3>>>()?;
    let normal = points
        .windows(3)
        .find_map(|pair| {
            let cross = (pair[1] - pair[0]).cross(pair[2] - pair[0]);
            (cross.length_squared() > f64::EPSILON).then_some(cross.normalize())
        })
        .unwrap_or(DVec3::Z);
    let axis = if normal.x.abs() > normal.y.abs() && normal.x.abs() > normal.z.abs() {
        0
    } else if normal.y.abs() > normal.z.abs() {
        1
    } else {
        2
    };
    let mut projected = points
        .into_iter()
        .map(|point| match axis {
            0 => [point.y, point.z],
            1 => [point.x, point.z],
            _ => [point.x, point.y],
        })
        .collect::<Vec<_>>();
    if method == "smart" {
        normalize_uv(&mut projected);
    }
    Ok(projected)
}

fn normalize_uv(coords: &mut [[f64; 2]]) {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for uv in coords.iter() {
        for axis in 0..2 {
            min[axis] = min[axis].min(uv[axis]);
            max[axis] = max[axis].max(uv[axis]);
        }
    }
    for uv in coords {
        for axis in 0..2 {
            let extent = max[axis] - min[axis];
            uv[axis] = if extent > f64::EPSILON {
                (uv[axis] - min[axis]) / extent
            } else {
                0.5
            };
        }
    }
}

#[derive(Clone, Debug)]
struct UvFace {
    uv: Vec<[f64; 2]>,
    pinned: Option<Vec<bool>>,
}

type UvMap = std::collections::BTreeMap<u32, UvFace>;

fn read_uv_map(mesh: &Mesh) -> Result<UvMap> {
    let mut result = UvMap::new();
    let entries = match mesh.attributes.get("uv_map") {
        None => return Ok(result),
        Some(Value::Array(entries)) => entries,
        Some(_) => {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "uv_map must be an array",
            ));
        }
    };
    for entry in entries {
        let face_id = entry
            .get("face_id")
            .and_then(Value::as_u64)
            .and_then(|id| u32::try_from(id).ok())
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "uv_map contains an invalid face ID",
                )
            })?;
        let coords = entry.get("uv").and_then(Value::as_array).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "uv_map entry has no corner array")
        })?;
        let mut corners = Vec::with_capacity(coords.len());
        for coord in coords {
            let pair = coord
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "UV corner must contain two coordinates",
                    )
                })?;
            let u = pair[0].as_f64().ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "UV coordinate must be numeric")
            })?;
            let v = pair[1].as_f64().ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "UV coordinate must be numeric")
            })?;
            corners.push([u, v]);
        }
        let pinned = entry
            .get("pinned")
            .map(|value| {
                let values = value.as_array().ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "UV pin data must be an array")
                })?;
                if values.len() != corners.len() {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "UV pin count must match the corner count",
                    ));
                }
                values
                    .iter()
                    .map(|value| {
                        value.as_bool().ok_or_else(|| {
                            PotError::new(ErrorCode::SceneInvalid, "UV pin values must be boolean")
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        result.insert(
            face_id,
            UvFace {
                uv: corners,
                pinned,
            },
        );
    }
    Ok(result)
}

fn write_uv_map(mesh: &mut Mesh, uv_map: &UvMap) {
    let entries = uv_map
        .iter()
        .map(|(face_id, face_uv)| {
            let mut entry = json!({"face_id":face_id,"uv":face_uv.uv});
            if let Some(pinned) = &face_uv.pinned {
                entry["pinned"] = json!(pinned);
            }
            entry
        })
        .collect::<Vec<_>>();
    mesh.attributes
        .insert("uv_map".to_owned(), Value::Array(entries));
}

#[derive(Clone, Copy)]
struct UvIsland {
    face_id: u32,
    min: [f64; 2],
    width: f64,
    height: f64,
}

const UV_PACK_MIN_EXTENT: f64 = 1.0e-12;

fn pack_selected_uvs(uv_map: &mut UvMap, selected: &[u32], margin: f64) -> Result<()> {
    if selected.is_empty() {
        return Ok(());
    }
    let mut islands = Vec::with_capacity(selected.len());
    for face_id in selected {
        let face_uv = uv_map.get(face_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                format!("uv.pack requires UV coordinates for face f{face_id}"),
            )
        })?;
        if face_uv.uv.is_empty() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                format!("UV face f{face_id} has no corners"),
            ));
        }
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for uv in &face_uv.uv {
            for axis in 0..2 {
                if !uv[axis].is_finite() {
                    return Err(PotError::new(
                        ErrorCode::InvalidOperation,
                        "UV coordinates must be finite to pack",
                    ));
                }
                min[axis] = min[axis].min(uv[axis]);
                max[axis] = max[axis].max(uv[axis]);
            }
        }
        let width = max[0] - min[0];
        let height = max[1] - min[1];
        if !width.is_finite() || !height.is_finite() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "UV island bounds are too large to pack",
            ));
        }
        islands.push(UvIsland {
            face_id: *face_id,
            min,
            width,
            height,
        });
    }
    islands.sort_by(|left, right| {
        right
            .width
            .max(right.height)
            .total_cmp(&left.width.max(left.height))
            .then_with(|| left.face_id.cmp(&right.face_id))
    });

    let mut placements = Vec::with_capacity(islands.len());
    if !try_shelf_pack(&islands, 1.0, margin, &mut placements) {
        if !try_shelf_pack(&islands, 0.0, margin, &mut placements) {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "UV islands cannot fit in the unit square with this margin",
            ));
        }
        let mut best = placements.clone();
        let (mut low, mut high) = (0.0, 1.0);
        for _ in 0..64 {
            let scale = f64::midpoint(low, high);
            if try_shelf_pack(&islands, scale, margin, &mut placements) {
                low = scale;
                best.clone_from(&placements);
            } else {
                high = scale;
            }
        }
        placements = best;
        let scale = low * (1.0 - 1.0e-12);
        for (island, placement) in islands.iter().zip(placements) {
            let face_uv = uv_map.get_mut(&island.face_id).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "packed UV face disappeared")
            })?;
            for uv in &mut face_uv.uv {
                uv[0] = placement[0] + (uv[0] - island.min[0]) * scale;
                uv[1] = placement[1] + (uv[1] - island.min[1]) * scale;
            }
        }
        return Ok(());
    }

    for (island, placement) in islands.iter().zip(placements) {
        let face_uv = uv_map
            .get_mut(&island.face_id)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "packed UV face disappeared"))?;
        for uv in &mut face_uv.uv {
            uv[0] = placement[0] + (uv[0] - island.min[0]);
            uv[1] = placement[1] + (uv[1] - island.min[1]);
        }
    }
    Ok(())
}

fn try_shelf_pack(
    islands: &[UvIsland],
    scale: f64,
    margin: f64,
    placements: &mut Vec<[f64; 2]>,
) -> bool {
    placements.clear();
    let (mut x, mut y, mut row_height): (f64, f64, f64) = (0.0, 0.0, 0.0);
    for island in islands {
        let width = (island.width * scale).max(UV_PACK_MIN_EXTENT);
        let height = (island.height * scale).max(UV_PACK_MIN_EXTENT);
        if x > 0.0 && x + width > 1.0 {
            y += row_height + margin;
            x = 0.0;
            row_height = 0.0;
        }
        if x + width > 1.0 || y + height > 1.0 {
            placements.clear();
            return false;
        }
        placements.push([x, y]);
        x += width + margin;
        row_height = row_height.max(height);
    }
    true
}

fn read_vec2(
    engine: &Engine<'_>,
    value: Option<&Value>,
    default: [f64; 2],
    field: &str,
) -> Result<[f64; 2]> {
    let Some(value) = value else {
        return Ok(default);
    };
    let array = value
        .as_array()
        .filter(|array| array.len() == 2)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must contain two numbers"),
                field,
            )
        })?;
    let x = array[0].as_f64().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must contain numbers"),
            field,
        )
    })?;
    let y = array[1].as_f64().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must contain numbers"),
            field,
        )
    })?;
    if !x.is_finite() || !y.is_finite() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be finite"),
            field,
        ));
    }
    Ok([x, y])
}

fn read_number(
    engine: &Engine<'_>,
    value: Option<&Value>,
    default: f64,
    field: &str,
) -> Result<f64> {
    let value = value.map_or(Ok(default), |value| {
        value.as_f64().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must be numeric"),
                field,
            )
        })
    })?;
    if !value.is_finite() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be finite"),
            field,
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]
    use proptest::prelude::*;

    use super::*;

    fn document_with_uvs(rectangles: &[(f64, f64)]) -> crate::model::SceneDoc {
        let initial = crate::model::SceneDoc::new("scene-test".to_owned());
        let outcome = super::super::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": initial.revision,
                "operations": [
                    {"op":"node.create","id":"body","kind":"box","params":{"size":1.0}}
                ]
            }),
        )
        .unwrap();
        let mut doc = outcome.doc;
        let data_id = Id::new("body_mesh").unwrap();
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .unwrap()
            .mesh
            .as_mut()
            .unwrap();
        let entries = mesh
            .faces
            .iter()
            .zip(rectangles.iter().copied())
            .enumerate()
            .map(|(index, (face, (width, height)))| {
                let offset = f64::from(u32::try_from(index).unwrap()) * 2.0;
                json!({
                    "face_id": face.id,
                    "uv": [
                        [offset, 0.0],
                        [offset + width, 0.0],
                        [offset + width, height],
                        [offset, height]
                    ]
                })
            })
            .collect::<Vec<_>>();
        mesh.attributes.insert("uv_map".to_owned(), json!(entries));
        doc
    }

    fn document_with_planar_grid() -> crate::model::SceneDoc {
        let mut doc = document_with_uvs(&[(1.0, 1.0); 6]);
        let data_id = Id::new("body_mesh").unwrap();
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .unwrap()
            .mesh
            .as_mut()
            .unwrap();
        mesh.vertices = (0..9_u32)
            .map(|id| crate::geom::Vertex {
                id,
                co: DVec3::new(f64::from(id % 3), f64::from(id / 3), 0.0),
            })
            .collect();
        mesh.faces = (0..2_u32)
            .flat_map(|y| {
                (0..2_u32).map(move |x| {
                    let lower_left = y * 3 + x;
                    crate::geom::Face {
                        id: y * 2 + x,
                        vertices: vec![lower_left, lower_left + 1, lower_left + 4, lower_left + 3],
                        material_index: 0,
                    }
                })
            })
            .collect();
        let mut edge_vertices = std::collections::BTreeSet::new();
        for face in &mesh.faces {
            for corner in 0..face.vertices.len() {
                edge_vertices.insert(crate::geom::edge_key(
                    face.vertices[corner],
                    face.vertices[(corner + 1) % face.vertices.len()],
                ));
            }
        }
        mesh.edges = edge_vertices
            .into_iter()
            .enumerate()
            .map(|(index, (left, right))| crate::geom::Edge {
                id: u32::try_from(index).unwrap(),
                vertices: [left, right],
            })
            .collect();
        mesh.next_id = crate::geom::IdCounters {
            vertex: 9,
            edge: u32::try_from(mesh.edges.len()).unwrap(),
            face: 4,
        };
        mesh.attributes.clear();
        doc
    }

    fn run_uv(doc: &mut crate::model::SceneDoc, operation: Value) -> Result<bool> {
        let Value::Object(object) = operation else {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "UV test operation must be an object",
            ));
        };
        let name = object.get("op").and_then(Value::as_str).ok_or_else(|| {
            PotError::new(ErrorCode::InvalidOperation, "UV test operation requires op")
        })?;
        let mut engine = super::super::Engine {
            doc,
            changes: super::super::ChangeSet::default(),
            id_mappings: Map::new(),
            operation_index: 0,
            evaluation_frame: 1.0,
            asset_root: None,
            pending_assets: std::collections::BTreeMap::new(),
        };
        apply(&mut engine, name, &object)
    }

    fn uv_entries(doc: &crate::model::SceneDoc) -> Vec<Value> {
        let data_id = Id::new("body_mesh").unwrap();
        doc.data_blocks[&data_id].mesh.as_ref().unwrap().attributes["uv_map"]
            .as_array()
            .unwrap()
            .clone()
    }

    #[test]
    fn advanced_unwrap_reproduces_a_planar_grid_up_to_similarity() {
        for method in ["angle_based", "conformal"] {
            let mut doc = document_with_planar_grid();
            run_uv(
                &mut doc,
                json!({"op":"uv.unwrap","target":{"id":"body"},"method":method}),
            )
            .unwrap();

            let entries = uv_entries(&doc);
            let data_id = Id::new("body_mesh").unwrap();
            let mesh = doc.data_blocks[&data_id].mesh.as_ref().unwrap();
            let mut uv_by_vertex: Vec<Option<[f64; 2]>> = vec![None; mesh.vertices.len()];
            for face in &mesh.faces {
                let entry = entries
                    .iter()
                    .find(|entry| entry["face_id"].as_u64() == Some(u64::from(face.id)))
                    .unwrap();
                for (corner, &vertex_id) in face.vertices.iter().enumerate() {
                    let uv = [
                        entry["uv"][corner][0].as_f64().unwrap(),
                        entry["uv"][corner][1].as_f64().unwrap(),
                    ];
                    let index = usize::try_from(vertex_id).unwrap();
                    if let Some(previous) = uv_by_vertex[index] {
                        assert!(
                            (previous[0] - uv[0]).hypot(previous[1] - uv[1]) < 1.0e-8,
                            "{method} did not join unseamed face corners at v{vertex_id}"
                        );
                    } else {
                        uv_by_vertex[index] = Some(uv);
                    }
                }
            }

            let uv_by_vertex = uv_by_vertex
                .into_iter()
                .map(Option::unwrap)
                .collect::<Vec<_>>();
            let scale = (uv_by_vertex[0][0] - uv_by_vertex[1][0])
                .hypot(uv_by_vertex[0][1] - uv_by_vertex[1][1]);
            assert!(scale > 0.0 && scale.is_finite());
            for left in 0..uv_by_vertex.len() {
                for right in left + 1..uv_by_vertex.len() {
                    let left_xy = [
                        f64::from(u32::try_from(left).unwrap() % 3),
                        f64::from(u32::try_from(left).unwrap() / 3),
                    ];
                    let right_xy = [
                        f64::from(u32::try_from(right).unwrap() % 3),
                        f64::from(u32::try_from(right).unwrap() / 3),
                    ];
                    let expected =
                        (left_xy[0] - right_xy[0]).hypot(left_xy[1] - right_xy[1]) * scale;
                    let actual = (uv_by_vertex[left][0] - uv_by_vertex[right][0])
                        .hypot(uv_by_vertex[left][1] - uv_by_vertex[right][1]);
                    assert!(
                        (actual - expected).abs() <= scale * 1.0e-7,
                        "{method} changed grid distance {left}-{right}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn advanced_unwrap_preserves_pins_and_splits_seams() {
        let mut doc = document_with_planar_grid();
        let data_id = Id::new("body_mesh").unwrap();
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .unwrap()
            .mesh
            .as_mut()
            .unwrap();
        let entries = mesh
            .faces
            .iter()
            .map(|face| {
                let mut uv = face
                    .vertices
                    .iter()
                    .map(|vertex_id| {
                        let point = mesh.vertex(*vertex_id).unwrap().co;
                        [point.x, point.y]
                    })
                    .collect::<Vec<_>>();
                let mut entry = json!({"face_id":face.id,"uv":uv});
                if face.id == 0 {
                    uv[0] = [-3.0, 5.0];
                    entry["uv"] = json!(uv);
                    entry["pinned"] = json!([true, false, false, false]);
                }
                entry
            })
            .collect::<Vec<_>>();
        let seam_edge_id = mesh
            .edges
            .iter()
            .find(|edge| crate::geom::edge_key(edge.vertices[0], edge.vertices[1]) == (1, 4))
            .unwrap()
            .id;
        let mut seam_values = serde_json::Map::new();
        seam_values.insert(format!("e{seam_edge_id}"), json!(true));
        mesh.attributes.insert("uv_map".to_owned(), json!(entries));
        mesh.attributes.insert(
            "seam".to_owned(),
            json!({"domain":"edge","values":seam_values}),
        );
        run_uv(
            &mut doc,
            json!({"op":"uv.unwrap","target":{"id":"body"},"method":"angle_based"}),
        )
        .unwrap();

        let entries = uv_entries(&doc);
        let pinned_entry = entries
            .iter()
            .find(|entry| entry["face_id"].as_u64() == Some(0))
            .unwrap();
        assert_eq!(pinned_entry["uv"][0], json!([-3.0, 5.0]));
        assert_eq!(pinned_entry["pinned"][0], json!(true));

        let mesh = doc.data_blocks[&data_id].mesh.as_ref().unwrap();
        let left_face = mesh.faces.iter().find(|face| face.id == 0).unwrap();
        let right_face = mesh.faces.iter().find(|face| face.id == 1).unwrap();
        let left_entry = entries
            .iter()
            .find(|entry| entry["face_id"].as_u64() == Some(0))
            .unwrap();
        let right_entry = entries
            .iter()
            .find(|entry| entry["face_id"].as_u64() == Some(1))
            .unwrap();
        let split = [1, 4].into_iter().any(|vertex_id| {
            let left_corner = left_face
                .vertices
                .iter()
                .position(|candidate| *candidate == vertex_id)
                .unwrap();
            let right_corner = right_face
                .vertices
                .iter()
                .position(|candidate| *candidate == vertex_id)
                .unwrap();
            left_entry["uv"][left_corner] != right_entry["uv"][right_corner]
        });
        assert!(split, "marked seam did not split its shared UV vertices");
    }

    proptest! {
        #[test]
        fn packed_uv_faces_stay_in_unit_square_and_have_disjoint_bounds(
            rectangles in prop::collection::vec((0.01_f64..1.0, 0.01_f64..1.0), 1..=6),
            margin in 0.0_f64..0.08,
        ) {
            let mut doc = document_with_uvs(&rectangles);
            let selected_ids = uv_entries(&doc)
                .iter()
                .map(|entry| format!("f{}", entry["face_id"].as_u64().unwrap()))
                .collect::<Vec<_>>();
            run_uv(&mut doc, json!({
                "op":"uv.pack",
                "target":{"id":"body"},
                "elements":{"domain":"face","ids":selected_ids},
                "margin":margin
            })).unwrap();

            let packed = uv_entries(&doc);
            let mut bounds = Vec::with_capacity(packed.len());
            for entry in &packed {
                let corners = entry["uv"].as_array().unwrap();
                prop_assert_eq!(corners.len(), 4);
                let mut min = [f64::INFINITY; 2];
                let mut max = [f64::NEG_INFINITY; 2];
                let mut uv = [[0.0; 2]; 4];
                for (index, corner) in corners.iter().enumerate() {
                    for axis in 0..2 {
                        let value = corner[axis].as_f64().unwrap();
                        prop_assert!((0.0..=1.0).contains(&value));
                        uv[index][axis] = value;
                        min[axis] = min[axis].min(value);
                        max[axis] = max[axis].max(value);
                    }
                }
                prop_assert_eq!(uv[0][1], uv[1][1]);
                prop_assert_eq!(uv[1][0], uv[2][0]);
                prop_assert_eq!(uv[2][1], uv[3][1]);
                prop_assert_eq!(uv[3][0], uv[0][0]);
                bounds.push((min, max));
            }

            for left in 0..bounds.len() {
                for right in left + 1..bounds.len() {
                    let (left_min, left_max) = bounds[left];
                    let (right_min, right_max) = bounds[right];
                    let separated = left_max[0] + margin <= right_min[0] + 1.0e-12
                        || right_max[0] + margin <= left_min[0] + 1.0e-12
                        || left_max[1] + margin <= right_min[1] + 1.0e-12
                        || right_max[1] + margin <= left_min[1] + 1.0e-12;
                    prop_assert!(separated, "packed face bounds overlap or violate margin");
                }
            }
        }
    }

    #[test]
    fn pins_survive_face_transforms_and_unwrap() {
        let mut doc = document_with_uvs(&[(1.0, 1.0); 6]);
        let first_face = {
            let data_id = Id::new("body_mesh").unwrap();
            doc.data_blocks[&data_id].mesh.as_ref().unwrap().faces[0].id
        };
        let selector = json!({"domain":"face","ids":[format!("f{first_face}")]});
        run_uv(
            &mut doc,
            json!({
                "op":"uv.pin",
                "target":{"id":"body"},
                "elements":selector,
                "pinned":true
            }),
        )
        .unwrap();

        assert!(
            uv_entries(&doc)
                .iter()
                .filter(|entry| entry["face_id"].as_u64() != Some(u64::from(first_face)))
                .all(|entry| entry.get("pinned").is_none())
        );
        run_uv(
            &mut doc,
            json!({
                "op":"uv.transform",
                "target":{"id":"body"},
                "elements":selector,
                "translation":[0.25,-0.5]
            }),
        )
        .unwrap();
        run_uv(
            &mut doc,
            json!({
                "op":"uv.unwrap",
                "target":{"id":"body"},
                "elements":selector,
                "method":"box"
            }),
        )
        .unwrap();

        let pinned = uv_entries(&doc)
            .into_iter()
            .find(|entry| entry["face_id"].as_u64() == Some(u64::from(first_face)))
            .unwrap()["pinned"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(pinned, vec![json!(true); 4]);

        run_uv(
            &mut doc,
            json!({
                "op":"uv.pin",
                "target":{"id":"body"},
                "elements":selector,
                "pinned":false
            }),
        )
        .unwrap();
        let unpinned = uv_entries(&doc)
            .into_iter()
            .find(|entry| entry["face_id"].as_u64() == Some(u64::from(first_face)))
            .unwrap()["pinned"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(unpinned, vec![json!(false); 4]);
    }
}
