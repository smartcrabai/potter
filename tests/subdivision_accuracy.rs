use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    error::Error,
};

use glam::DVec3;
use potter_core::{
    geom::{Mesh, modifiers::evaluate_modifiers},
    model::{Id, Modifier, ModifierRuntime},
};
use serde_json::{Map, Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ORACLE: &str = include_str!("fixtures/pr16_subdivision/oracle.json");
// Blender f32 output is rounded to seven fixture decimals; generation measured a maximum
// serialization error of 5.000000003e-8. These bounds cover f32/OpenSubdiv arithmetic only.
const POSITION_TOLERANCE: f64 = 3.0e-6;
const UV_TOLERANCE: f64 = 2.0e-6;
const NORMAL_TOLERANCE: f64 = 2.0e-6;
const CREASE_TOLERANCE: f64 = 1.0e-6;
// Non-manifold splits put several Blender vertices at one position; vertices closer than this
// are one position, and each position must have the same multiplicity on both sides.
const CLUSTER_TOLERANCE: f64 = 1.0e-4;

fn number(value: &Value) -> TestResult<f64> {
    value.as_f64().ok_or_else(|| "expected a number".into())
}

fn indices(value: &Value) -> TestResult<Vec<usize>> {
    value
        .as_array()
        .ok_or("expected an index array")?
        .iter()
        .map(|item| -> TestResult<usize> {
            let index = item.as_u64().ok_or("index must be an unsigned integer")?;
            Ok(usize::try_from(index)?)
        })
        .collect()
}

fn point(value: &Value) -> TestResult<DVec3> {
    let values = value.as_array().ok_or("point must be an array")?;
    if values.len() != 3 {
        return Err("point must have three coordinates".into());
    }
    Ok(DVec3::new(
        number(&values[0])?,
        number(&values[1])?,
        number(&values[2])?,
    ))
}

fn uv(value: &Value) -> TestResult<[f64; 2]> {
    let values = value.as_array().ok_or("UV must be an array")?;
    if values.len() != 2 {
        return Err("UV must have two coordinates".into());
    }
    Ok([number(&values[0])?, number(&values[1])?])
}

fn rotated<T: Clone>(values: &[T], start: usize) -> Vec<T> {
    (0..values.len())
        .map(|index| values[(start + index) % values.len()].clone())
        .collect()
}

/// Rotates a cycle to its lexicographically smallest form. Returns the rotated cycle and the
/// rotation applied, so per-corner data can be rotated the same way.
fn canonical_cycle(keys: &[usize]) -> (Vec<usize>, usize) {
    let len = keys.len();
    let start = (0..len)
        .min_by(|first, second| {
            (0..len)
                .map(|offset| keys[(first + offset) % len])
                .cmp((0..len).map(|offset| keys[(second + offset) % len]))
        })
        .unwrap_or(0);
    (rotated(keys, start), start)
}

fn edge_key(first: usize, second: usize) -> (usize, usize) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

/// One output or oracle face, with vertices as indices into its shape's point list.
#[derive(Clone)]
struct FaceRecord {
    vertices: Vec<usize>,
    smooth: bool,
    uv: BTreeMap<String, Vec<[f64; 2]>>,
    normals: Option<Vec<DVec3>>,
}

struct EdgeRecord {
    vertices: [usize; 2],
    seam: bool,
    crease: f64,
}

/// A mesh compared by position: only point coordinates and index relations are read.
struct Shape {
    points: Vec<DVec3>,
    faces: Vec<FaceRecord>,
    edges: Vec<EdgeRecord>,
    has_custom_normals: bool,
}

fn oracle_shape(expected: &Value) -> TestResult<Shape> {
    let points = expected["vertices"]
        .as_array()
        .ok_or("oracle positions missing")?
        .iter()
        .map(point)
        .collect::<TestResult<Vec<_>>>()?;
    let faces = expected["faces"]
        .as_array()
        .ok_or("oracle faces missing")?
        .iter()
        .map(|face| -> TestResult<FaceRecord> {
            let mut uv_layers = BTreeMap::new();
            for (layer, corners) in face["uvs"].as_object().ok_or("oracle UV map missing")? {
                let values = corners
                    .as_array()
                    .ok_or("oracle UV corners missing")?
                    .iter()
                    .map(uv)
                    .collect::<TestResult<Vec<_>>>()?;
                uv_layers.insert(layer.clone(), values);
            }
            let normals = match face.get("normals") {
                Some(values) => Some(
                    values
                        .as_array()
                        .ok_or("oracle custom normals missing")?
                        .iter()
                        .map(point)
                        .collect::<TestResult<Vec<_>>>()?,
                ),
                None => None,
            };
            Ok(FaceRecord {
                vertices: indices(&face["v"])?,
                smooth: face["smooth"]
                    .as_bool()
                    .ok_or("oracle smooth flag missing")?,
                uv: uv_layers,
                normals,
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    let edges = expected["edges"]
        .as_array()
        .ok_or("oracle edges missing")?
        .iter()
        .map(|edge| -> TestResult<EdgeRecord> {
            let pair = indices(&edge["v"])?;
            if pair.len() != 2 {
                return Err("oracle edge must reference two vertices".into());
            }
            Ok(EdgeRecord {
                vertices: [pair[0], pair[1]],
                seam: edge["seam"].as_bool().ok_or("oracle seam flag missing")?,
                crease: number(&edge["crease"])?,
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Shape {
        points,
        faces,
        edges,
        has_custom_normals: expected["has_custom_normals"]
            .as_bool()
            .ok_or("oracle custom-normal flag missing")?,
    })
}

fn potter_shape(mesh: &Mesh) -> TestResult<Shape> {
    let vertex_index: BTreeMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let points = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let uv_layers = mesh
        .attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
        .ok_or("output UV layers missing")?
        .iter()
        .map(|layer| -> TestResult<(String, Vec<[f64; 2]>)> {
            let name = layer["name"]
                .as_str()
                .ok_or("output UV layer name missing")?;
            let values = layer["values"]
                .as_array()
                .ok_or("output UV values missing")?
                .iter()
                .map(uv)
                .collect::<TestResult<Vec<_>>>()?;
            Ok((name.to_owned(), values))
        })
        .collect::<TestResult<Vec<_>>>()?;
    let smooth = mesh
        .attributes
        .get("blender_polygon_smooth")
        .and_then(Value::as_array)
        .ok_or("output polygon smooth flags missing")?;
    if smooth.len() != mesh.faces.len() {
        return Err("output smooth flag count differs from face count".into());
    }
    let custom = mesh
        .attributes
        .get("custom_normal")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let mut loop_offset = 0;
    let mut faces = Vec::with_capacity(mesh.faces.len());
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let vertices = face
            .vertices
            .iter()
            .map(|id| {
                vertex_index
                    .get(id)
                    .copied()
                    .ok_or("output face references a missing vertex")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let end = loop_offset + vertices.len();
        let mut uv_map = BTreeMap::new();
        for (name, values) in &uv_layers {
            if end > values.len() {
                return Err(format!("UV layer {name} is shorter than face loops").into());
            }
            uv_map.insert(name.clone(), values[loop_offset..end].to_vec());
        }
        let normals = match custom {
            Some(values) => Some(
                values
                    .get(&format!("f{}", face.id))
                    .and_then(Value::as_array)
                    .ok_or("custom normals missing for output face")?
                    .iter()
                    .map(point)
                    .collect::<TestResult<Vec<_>>>()?,
            ),
            None => None,
        };
        faces.push(FaceRecord {
            vertices,
            smooth: smooth[face_index]
                .as_bool()
                .ok_or("output smooth flag is not boolean")?,
            uv: uv_map,
            normals,
        });
        loop_offset = end;
    }
    for (name, values) in &uv_layers {
        if loop_offset != values.len() {
            return Err(format!(
                "UV layer {name} has {} loops, faces have {loop_offset}",
                values.len()
            )
            .into());
        }
    }
    let flags = mesh
        .attributes
        .get("blender_edge_flags")
        .and_then(Value::as_array)
        .ok_or("output edge flags missing")?;
    if flags.len() != mesh.edges.len() {
        return Err("output edge flag count differs from edge count".into());
    }
    let crease_values = mesh
        .attributes
        .get("crease_edge")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let mut edges = Vec::with_capacity(mesh.edges.len());
    for (edge_index, edge) in mesh.edges.iter().enumerate() {
        let mut endpoints = [0; 2];
        for (slot, id) in edge.vertices.iter().enumerate() {
            endpoints[slot] = *vertex_index
                .get(id)
                .ok_or("output edge references a missing vertex")?;
        }
        let crease = crease_values
            .and_then(|values| values.get(&format!("e{}", edge.id)))
            .map(number)
            .transpose()?
            .unwrap_or(0.0);
        edges.push(EdgeRecord {
            vertices: endpoints,
            seam: flags[edge_index]["seam"]
                .as_bool()
                .ok_or("output seam flag is not boolean")?,
            crease,
        });
    }
    Ok(Shape {
        points,
        faces,
        edges,
        has_custom_normals: custom.is_some(),
    })
}

fn input_mesh(spec: &Value) -> TestResult<Mesh> {
    let vertices = spec["vertices"]
        .as_array()
        .ok_or("mesh vertices missing")?
        .iter()
        .map(point)
        .collect::<TestResult<Vec<_>>>()?;
    let faces = spec["faces"]
        .as_array()
        .ok_or("mesh faces missing")?
        .iter()
        .map(indices)
        .collect::<TestResult<Vec<_>>>()?;
    let mut mesh = Mesh::from_positions_and_faces(vertices, faces)?;
    for loose in spec["loose_edges"]
        .as_array()
        .ok_or("mesh loose edges missing")?
    {
        let pair = indices(loose)?;
        if pair.len() != 2 {
            return Err("loose edge must reference two vertices".into());
        }
        mesh.insert_edge([u32::try_from(pair[0])?, u32::try_from(pair[1])?])?;
    }

    let seams = spec["seams"]
        .as_array()
        .ok_or("mesh seams missing")?
        .iter()
        .map(indices)
        .map(|edge| {
            let edge = edge?;
            if edge.len() != 2 {
                return Err("seam must reference two vertices".into());
            }
            Ok(edge_key(edge[0], edge[1]))
        })
        .collect::<TestResult<BTreeSet<_>>>()?;
    let crease_pairs = spec["creases"]
        .as_array()
        .ok_or("mesh creases missing")?
        .iter()
        .map(|item| -> TestResult<((usize, usize), f64)> {
            let values = item.as_array().ok_or("crease must be an array")?;
            if values.len() != 3 {
                return Err("crease must have two endpoints and a value".into());
            }
            let first = usize::try_from(values[0].as_u64().ok_or("crease endpoint missing")?)?;
            let second = usize::try_from(values[1].as_u64().ok_or("crease endpoint missing")?)?;
            Ok((edge_key(first, second), number(&values[2])?))
        })
        .collect::<TestResult<BTreeMap<_, _>>>()?;
    let mut flags = Vec::with_capacity(mesh.edges.len());
    let mut crease_values = Map::new();
    for edge in &mesh.edges {
        let endpoints = edge_key(edge.vertices[0] as usize, edge.vertices[1] as usize);
        flags.push(json!({"sharp": false, "seam": seams.contains(&endpoints)}));
        if let Some(value) = crease_pairs.get(&endpoints) {
            crease_values.insert(format!("e{}", edge.id), json!(value));
        }
    }
    mesh.attributes
        .insert("blender_edge_flags".to_owned(), Value::Array(flags));
    mesh.attributes
        .insert("blender_polygon_smooth".to_owned(), spec["smooth"].clone());
    mesh.attributes
        .insert("blender_uv_layers".to_owned(), spec["uv_layers"].clone());
    mesh.attributes.insert(
        "crease_edge".to_owned(),
        json!({"domain": "edge", "type": "float", "values": crease_values}),
    );

    let source_normals = spec["custom_normals"]
        .as_array()
        .ok_or("mesh custom normals missing")?;
    let mut corner = 0;
    let mut normal_values = Map::new();
    for face in &mesh.faces {
        let end = corner + face.vertices.len();
        if end > source_normals.len() {
            return Err("source custom normals do not cover every corner".into());
        }
        normal_values.insert(
            format!("f{}", face.id),
            Value::Array(source_normals[corner..end].to_vec()),
        );
        corner = end;
    }
    if corner != source_normals.len() {
        return Err("source custom normals contain extra corners".into());
    }
    mesh.attributes.insert(
        "custom_normal".to_owned(),
        json!({"domain": "corner", "type": "float3", "values": normal_values}),
    );
    Ok(mesh)
}

/// Groups vertices whose positions coincide. Returns each vertex's cluster, each cluster's
/// representative position, and each cluster's multiplicity.
fn cluster_positions(points: &[DVec3]) -> (Vec<usize>, Vec<DVec3>, Vec<usize>) {
    let cell = |point: DVec3| {
        (
            (point.x / CLUSTER_TOLERANCE).floor() as i64,
            (point.y / CLUSTER_TOLERANCE).floor() as i64,
            (point.z / CLUSTER_TOLERANCE).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut representatives: Vec<DVec3> = Vec::new();
    let mut multiplicities: Vec<usize> = Vec::new();
    let mut assignment = Vec::with_capacity(points.len());
    for point in points {
        let (x, y, z) = cell(*point);
        let mut found = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(clusters) = grid.get(&(x + dx, y + dy, z + dz)) {
                        for &cluster in clusters {
                            if representatives[cluster].distance(*point) <= CLUSTER_TOLERANCE {
                                found = Some(cluster);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        let cluster = match found {
            Some(cluster) => {
                multiplicities[cluster] += 1;
                cluster
            }
            None => {
                representatives.push(*point);
                multiplicities.push(1);
                let cluster = representatives.len() - 1;
                grid.entry((x, y, z)).or_default().push(cluster);
                cluster
            }
        };
        assignment.push(cluster);
    }
    (assignment, representatives, multiplicities)
}

#[derive(Default)]
struct Errors {
    position: f64,
    uv: f64,
    normal: f64,
    crease: f64,
}

/// Compares two shapes by position. Faces match by their canonical position-cluster cycle, edges
/// by their endpoint clusters, so the comparison does not depend on output vertex order.
fn compare_shapes(actual: &Shape, expected: &Shape) -> Result<Errors, String> {
    if actual.points.len() != expected.points.len() {
        return Err(format!(
            "vertex count {} != {}",
            actual.points.len(),
            expected.points.len()
        ));
    }
    if actual.faces.len() != expected.faces.len() {
        return Err(format!(
            "face count {} != {}",
            actual.faces.len(),
            expected.faces.len()
        ));
    }
    if actual.edges.len() != expected.edges.len() {
        return Err(format!(
            "edge count {} != {}",
            actual.edges.len(),
            expected.edges.len()
        ));
    }
    if actual.has_custom_normals != expected.has_custom_normals {
        return Err("custom-normal layer presence differs".to_owned());
    }
    let (actual_clusters, actual_reps, actual_counts) = cluster_positions(&actual.points);
    let (expected_clusters, expected_reps, expected_counts) = cluster_positions(&expected.points);
    if actual_reps.len() != expected_reps.len() {
        return Err(format!(
            "distinct positions {} != {}",
            actual_reps.len(),
            expected_reps.len()
        ));
    }
    let mut used = vec![false; expected_reps.len()];
    let mut to_expected = vec![0; actual_reps.len()];
    for (cluster, representative) in actual_reps.iter().enumerate() {
        let (target, error) = expected_reps
            .iter()
            .enumerate()
            .filter(|(index, _)| !used[*index])
            .map(|(index, other)| (index, other.distance(*representative)))
            .min_by(|first, second| first.1.total_cmp(&second.1))
            .ok_or("no Blender position left to match")?;
        if error > POSITION_TOLERANCE {
            return Err(format!(
                "output position {representative:?} has no Blender position within {POSITION_TOLERANCE:.1e} (nearest {error:.9e})"
            ));
        }
        if actual_counts[cluster] != expected_counts[target] {
            return Err(format!(
                "position {representative:?} occurs {} times, Blender {} times",
                actual_counts[cluster], expected_counts[target]
            ));
        }
        used[target] = true;
        to_expected[cluster] = target;
    }
    let mut errors = Errors::default();
    for (index, point) in actual.points.iter().enumerate() {
        let target = expected_reps[to_expected[actual_clusters[index]]];
        errors.position = errors.position.max(point.distance(target));
    }
    if errors.position > POSITION_TOLERANCE {
        return Err(format!(
            "max position error {:.9e} exceeds {POSITION_TOLERANCE:.9e}",
            errors.position
        ));
    }

    let mut actual_faces: BTreeMap<Vec<usize>, Vec<(&FaceRecord, usize)>> = BTreeMap::new();
    for face in &actual.faces {
        let keys: Vec<usize> = face
            .vertices
            .iter()
            .map(|vertex| to_expected[actual_clusters[*vertex]])
            .collect();
        let (cycle, rotation) = canonical_cycle(&keys);
        actual_faces
            .entry(cycle)
            .or_default()
            .push((face, rotation));
    }
    let mut expected_faces: BTreeMap<Vec<usize>, Vec<(&FaceRecord, usize)>> = BTreeMap::new();
    for face in &expected.faces {
        let keys: Vec<usize> = face
            .vertices
            .iter()
            .map(|vertex| expected_clusters[*vertex])
            .collect();
        let (cycle, rotation) = canonical_cycle(&keys);
        expected_faces
            .entry(cycle)
            .or_default()
            .push((face, rotation));
    }
    if actual_faces.keys().collect::<Vec<_>>() != expected_faces.keys().collect::<Vec<_>>() {
        return Err("canonical oriented face topology differs".to_owned());
    }
    for (cycle, expected_list) in &expected_faces {
        let actual_list = &actual_faces[cycle];
        if actual_list.len() != expected_list.len() {
            return Err(format!(
                "face {cycle:?} occurs {} times, Blender {}",
                actual_list.len(),
                expected_list.len()
            ));
        }
        for ((actual_face, actual_rotation), (expected_face, expected_rotation)) in
            actual_list.iter().zip(expected_list)
        {
            if actual_face.smooth != expected_face.smooth {
                return Err(format!("face {cycle:?} polygon interpolation flag differs"));
            }
            if actual_face.uv.keys().collect::<Vec<_>>()
                != expected_face.uv.keys().collect::<Vec<_>>()
            {
                return Err(format!("face {cycle:?} UV layer set differs"));
            }
            for (layer, expected_values) in &expected_face.uv {
                let actual_values = rotated(&actual_face.uv[layer], *actual_rotation);
                let expected_values = rotated(expected_values, *expected_rotation);
                for (actual_uv, expected_uv) in actual_values.iter().zip(&expected_values) {
                    errors.uv = errors
                        .uv
                        .max((actual_uv[0] - expected_uv[0]).abs())
                        .max((actual_uv[1] - expected_uv[1]).abs());
                }
            }
            match (&actual_face.normals, &expected_face.normals) {
                (Some(actual_normals), Some(expected_normals)) => {
                    let actual_normals = rotated(actual_normals, *actual_rotation);
                    let expected_normals = rotated(expected_normals, *expected_rotation);
                    for (actual_normal, expected_normal) in
                        actual_normals.iter().zip(&expected_normals)
                    {
                        errors.normal = errors.normal.max(actual_normal.distance(*expected_normal));
                    }
                }
                (None, None) => {}
                _ => return Err(format!("face {cycle:?} custom-normal presence differs")),
            }
        }
    }
    if errors.uv > UV_TOLERANCE {
        return Err(format!(
            "max corner UV error {:.9e} exceeds {UV_TOLERANCE:.9e}",
            errors.uv
        ));
    }
    if errors.normal > NORMAL_TOLERANCE {
        return Err(format!(
            "max custom-normal error {:.9e} exceeds {NORMAL_TOLERANCE:.9e}",
            errors.normal
        ));
    }

    let mut actual_edges: BTreeMap<(usize, usize), Vec<(bool, f64)>> = BTreeMap::new();
    for edge in &actual.edges {
        let first = to_expected[actual_clusters[edge.vertices[0]]];
        let second = to_expected[actual_clusters[edge.vertices[1]]];
        actual_edges
            .entry(edge_key(first, second))
            .or_default()
            .push((edge.seam, edge.crease));
    }
    let mut expected_edges: BTreeMap<(usize, usize), Vec<(bool, f64)>> = BTreeMap::new();
    for edge in &expected.edges {
        expected_edges
            .entry(edge_key(
                expected_clusters[edge.vertices[0]],
                expected_clusters[edge.vertices[1]],
            ))
            .or_default()
            .push((edge.seam, edge.crease));
    }
    if actual_edges.keys().collect::<Vec<_>>() != expected_edges.keys().collect::<Vec<_>>() {
        return Err("canonical edge topology differs".to_owned());
    }
    for (key, expected_list) in &expected_edges {
        let mut actual_list = actual_edges[key].clone();
        let mut expected_list = expected_list.clone();
        actual_list.sort_by(|first, second| {
            first
                .partial_cmp(second)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        expected_list.sort_by(|first, second| {
            first
                .partial_cmp(second)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if actual_list.len() != expected_list.len() {
            return Err(format!(
                "edge {key:?} occurs {} times, Blender {}",
                actual_list.len(),
                expected_list.len()
            ));
        }
        for ((actual_seam, actual_crease), (expected_seam, expected_crease)) in
            actual_list.iter().zip(&expected_list)
        {
            if actual_seam != expected_seam {
                return Err(format!("seam flag differs on edge {key:?}"));
            }
            errors.crease = errors.crease.max((actual_crease - expected_crease).abs());
        }
    }
    if errors.crease > CREASE_TOLERANCE {
        return Err(format!(
            "max propagated crease error {:.9e} exceeds {CREASE_TOLERANCE:.9e}",
            errors.crease
        ));
    }
    Ok(errors)
}

fn compare_case(case: &Value, spec: &Value) -> TestResult<Errors> {
    let name = case["name"].as_str().ok_or("case name missing")?;
    let params: Map<String, Value> = serde_json::from_value(case["params"].clone())?;
    let mesh = input_mesh(spec)?;
    let modifier = Modifier {
        id: Id::new("subdivision_oracle")?,
        modifier_type: "subdivision".to_owned(),
        name: "Subdivision oracle".to_owned(),
        enabled: true,
        params,
        binding_data: None,
        runtime: ModifierRuntime::default(),
    };
    let actual = evaluate_modifiers(&mesh, &[modifier])?;
    let actual = potter_shape(&actual)?;
    let expected = oracle_shape(&case["expected"])?;
    compare_shapes(&actual, &expected).map_err(|message| format!("{name}: {message}").into())
}

#[test]
fn subdivision_surface_matches_blender_522_across_the_modifier_matrix() -> TestResult {
    let oracle: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    assert_eq!(oracle["round_digits"], 7);
    let max_serialization_error = oracle["max_serialization_rounding_error"]
        .as_str()
        .ok_or("fixture rounding metadata missing")?
        .parse::<f64>()?;
    assert!(
        max_serialization_error <= 5.1e-8,
        "fixture max serialization rounding error {max_serialization_error:.9e} exceeds the seven-decimal bound"
    );
    let meshes = oracle["meshes"]
        .as_object()
        .ok_or("oracle mesh inputs missing")?;
    let cases = oracle["cases"].as_array().ok_or("oracle cases missing")?;
    assert_matrix_coverage(cases, meshes)?;

    let mut failures = Vec::new();
    let mut worst = Errors::default();
    for case in cases {
        let case_name = case["name"].as_str().ok_or("case name missing")?;
        let mesh_name = case["mesh"].as_str().ok_or("case mesh missing")?;
        let spec = meshes
            .get(mesh_name)
            .ok_or("case mesh not present in oracle")?;
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compare_case(case, spec)));
        match outcome {
            Ok(Ok(errors)) => {
                worst.position = worst.position.max(errors.position);
                worst.uv = worst.uv.max(errors.uv);
                worst.normal = worst.normal.max(errors.normal);
                worst.crease = worst.crease.max(errors.crease);
            }
            Ok(Err(error)) => failures.push(format!("{case_name}: {error}")),
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|text| (*text).to_owned()))
                    .unwrap_or_default();
                failures.push(format!("{case_name}: {message}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} oracle cases failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
    eprintln!(
        "subdivision corpus: {} cases; worst position {:.3e}, UV {:.3e}, normal {:.3e}, crease {:.3e}",
        cases.len(),
        worst.position,
        worst.uv,
        worst.normal,
        worst.crease
    );
    Ok(())
}

/// Measures which settings, mesh classes, and crease inputs the corpus exercises. A missing
/// combination fails the test, so the corpus cannot silently narrow.
fn assert_matrix_coverage(cases: &[Value], meshes: &Map<String, Value>) -> TestResult {
    let mut levels = BTreeSet::new();
    let mut qualities = BTreeSet::new();
    let mut uv_modes = BTreeSet::new();
    let mut boundary_modes = BTreeSet::new();
    let mut uv_boundary_pairs = BTreeSet::new();
    let mut limit_settings = BTreeSet::new();
    let mut subdivision_types = BTreeSet::new();
    let mut crease_settings = BTreeSet::new();
    let mut crease_dependent_uv_modes = BTreeSet::new();
    let mut custom_normal_settings = BTreeSet::new();
    let mut used_meshes = BTreeSet::new();
    let mut crease_values = BTreeSet::new();
    let mut cube_above_quality = false;
    for case in cases {
        let params = &case["params"];
        let level = params["levels"].as_u64().ok_or("levels missing")?;
        let quality = params["quality"].as_u64().ok_or("quality missing")?;
        let uv = params["uv_smooth"]
            .as_str()
            .ok_or("uv_smooth missing")?
            .to_owned();
        let boundary = params["boundary_smooth"]
            .as_str()
            .ok_or("boundary rule missing")?
            .to_owned();
        let mesh = case["mesh"].as_str().ok_or("case mesh missing")?;
        let limit = params["use_limit_surface"]
            .as_bool()
            .ok_or("limit flag missing")?;
        levels.insert(level);
        qualities.insert(quality);
        uv_boundary_pairs.insert((uv.clone(), boundary.clone()));
        if uv == "PRESERVE_CORNERS_AND_JUNCTIONS" || uv == "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE"
        {
            if params["use_creases"].as_bool() == Some(true) {
                crease_dependent_uv_modes.insert(uv.clone());
            }
        }
        uv_modes.insert(uv);
        boundary_modes.insert(boundary);
        limit_settings.insert(limit);
        subdivision_types.insert(
            params["subdivision_type"]
                .as_str()
                .ok_or("type missing")?
                .to_owned(),
        );
        crease_settings.insert(
            params["use_creases"]
                .as_bool()
                .ok_or("crease flag missing")?,
        );
        custom_normal_settings.insert(
            params["use_custom_normals"]
                .as_bool()
                .ok_or("custom-normal flag missing")?,
        );
        cube_above_quality |= mesh == "cube" && limit && level > quality;
        used_meshes.insert(mesh.to_owned());
        let mesh_spec = meshes.get(mesh).ok_or("case mesh not present in oracle")?;
        for crease in mesh_spec["creases"]
            .as_array()
            .ok_or("crease inputs missing")?
        {
            crease_values.insert((number(&crease[2])? * 100.0).round() as i64);
        }
    }
    assert_eq!(levels, BTreeSet::from([0, 1, 2, 3, 4, 5]), "levels");
    assert_eq!(
        qualities,
        BTreeSet::from([1, 2, 3, 4, 5, 6, 10]),
        "qualities"
    );
    let all_uv_modes = [
        "NONE",
        "PRESERVE_CORNERS",
        "PRESERVE_CORNERS_AND_JUNCTIONS",
        "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE",
        "PRESERVE_BOUNDARIES",
        "SMOOTH_ALL",
    ];
    assert_eq!(
        uv_modes,
        all_uv_modes.iter().map(|mode| (*mode).to_owned()).collect(),
        "uv_smooth modes"
    );
    assert_eq!(
        boundary_modes,
        BTreeSet::from(["ALL".to_owned(), "PRESERVE_CORNERS".to_owned()]),
        "boundary_smooth modes"
    );
    assert_eq!(
        uv_boundary_pairs.len(),
        all_uv_modes.len() * 2,
        "uv_smooth x boundary_smooth pairs"
    );
    assert_eq!(
        limit_settings,
        BTreeSet::from([false, true]),
        "use_limit_surface"
    );
    assert_eq!(
        subdivision_types,
        BTreeSet::from(["CATMULL_CLARK".to_owned(), "SIMPLE".to_owned()]),
        "subdivision types"
    );
    assert_eq!(
        crease_settings,
        BTreeSet::from([false, true]),
        "use_creases"
    );
    assert_eq!(
        custom_normal_settings,
        BTreeSet::from([false, true]),
        "use_custom_normals"
    );
    assert_eq!(
        crease_values,
        BTreeSet::from([10, 25, 30, 50, 60, 75, 80, 90, 100]),
        "crease input values"
    );
    assert_eq!(
        crease_dependent_uv_modes,
        BTreeSet::from([
            "PRESERVE_CORNERS_AND_JUNCTIONS".to_owned(),
            "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE".to_owned(),
        ]),
        "dependent UV corner-sharpness modes with creases"
    );
    assert!(
        cube_above_quality,
        "a limit-surface cube must run with levels above quality"
    );
    for mesh in [
        "quad",
        "cube",
        "cube_flat",
        "ngon",
        "nonmanifold",
        "mixed",
        "mixed_consistent",
        "dependent_uv_plus1",
        "dependent_uv_plus2",
        "loose",
        "curved_grid",
    ] {
        assert!(used_meshes.contains(mesh), "{mesh} must be in the corpus");
    }
    for mesh in ["curved_grid", "cube", "ngon", "nonmanifold", "mixed"] {
        let seams = meshes[mesh]["seams"]
            .as_array()
            .ok_or("seam input missing")?;
        assert!(!seams.is_empty(), "{mesh} must carry seam edges");
    }
    let loose_edges = meshes["loose"]["loose_edges"]
        .as_array()
        .ok_or("loose edges missing")?;
    assert!(
        !loose_edges.is_empty(),
        "loose mesh must carry a loose edge"
    );
    let flat = meshes["cube_flat"]["smooth"]
        .as_array()
        .ok_or("smooth flags missing")?;
    assert!(
        flat.iter().any(|flag| flag.as_bool() == Some(false)),
        "flat faces must be in the corpus"
    );
    Ok(())
}
