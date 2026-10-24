use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use glam::DVec3;
use serde_json::Value;
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;

const BUILDER: &str = include_str!("fixtures/curve_taper_fill/build_oracle.py");
const OPERATIONS: &str = include_str!("fixtures/curve_taper_fill/operations.json");
// At the fixture's 4.48-unit maximum coordinate, one float32 ULP is about 4.8e-7.
const TAPER_SURFACE_TOLERANCE: f64 = 5.0e-7;
// Surface area uses the same relative scale for Blender's float32 vertex-coordinate rounding.
const TAPER_AREA_RELATIVE_TOLERANCE: f64 = 5.0e-7;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Debug)]
struct Geometry {
    positions: Vec<DVec3>,
    faces: Vec<Vec<usize>>,
}

#[derive(Debug)]
struct SurfaceMetrics {
    area: f64,
    area_vector: DVec3,
    area_centroid: DVec3,
    minimum: DVec3,
    maximum: DVec3,
}

impl Geometry {
    fn from_json(value: &Value) -> TestResult<Self> {
        let positions = value["positions"]
            .as_array()
            .ok_or("mesh positions are missing")?
            .iter()
            .map(|point| {
                let coordinates = point
                    .as_array()
                    .filter(|coordinates| coordinates.len() == 3)
                    .ok_or("mesh position must have three coordinates")?;
                let coordinate = |index: usize| {
                    coordinates[index]
                        .as_f64()
                        .ok_or("mesh coordinate is not a number")
                };
                Ok(DVec3::new(coordinate(0)?, coordinate(1)?, coordinate(2)?))
            })
            .collect::<TestResult<Vec<_>>>()?;
        let faces = value["faces"]
            .as_array()
            .ok_or("mesh faces are missing")?
            .iter()
            .map(|face| {
                face.as_array()
                    .filter(|indices| indices.len() >= 3)
                    .ok_or("mesh face must have at least three vertices")?
                    .iter()
                    .map(|index| {
                        index
                            .as_u64()
                            .and_then(|index| usize::try_from(index).ok())
                            .ok_or_else(|| "mesh face index is invalid".into())
                    })
                    .collect::<TestResult<Vec<_>>>()
            })
            .collect::<TestResult<Vec<_>>>()?;
        if positions.is_empty() || faces.is_empty() {
            return Err("evaluated curve mesh has no surface geometry".into());
        }
        if faces
            .iter()
            .flatten()
            .any(|index| *index >= positions.len())
        {
            return Err("mesh face references a missing vertex".into());
        }
        Ok(Self { positions, faces })
    }

    fn triangles(&self) -> Vec<[DVec3; 3]> {
        let mut triangles = Vec::new();
        for face in &self.faces {
            let first = self.positions[face[0]];
            for index in 1..face.len() - 1 {
                triangles.push([
                    first,
                    self.positions[face[index]],
                    self.positions[face[index + 1]],
                ]);
            }
        }
        triangles
    }

    fn metrics(&self) -> SurfaceMetrics {
        let triangles = self.triangles();
        let mut area = 0.0;
        let mut area_vector = DVec3::ZERO;
        let mut weighted_centroid = DVec3::ZERO;
        for [a, b, c] in triangles {
            let oriented_area = (b - a).cross(c - a) * 0.5;
            let triangle_area = oriented_area.length();
            area += triangle_area;
            area_vector += oriented_area;
            weighted_centroid += (a + b + c) * (triangle_area / 3.0);
        }
        let minimum = self
            .positions
            .iter()
            .copied()
            .fold(DVec3::splat(f64::INFINITY), DVec3::min);
        let maximum = self
            .positions
            .iter()
            .copied()
            .fold(DVec3::splat(f64::NEG_INFINITY), DVec3::max);
        SurfaceMetrics {
            area,
            area_vector,
            area_centroid: weighted_centroid / area,
            minimum,
            maximum,
        }
    }
}

fn checked(mut command: Command, label: &str) -> TestResult<Output> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    Ok(output)
}

fn pot_json(configure: impl FnOnce(&mut Command)) -> TestResult<Value> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    configure(&mut command);
    command.arg("--json");
    let output = checked(command, "pot")?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn run_blender(blender: &Path, script: &Path, operations: &Path, root: &Path) -> TestResult {
    let mut command = Command::new(blender);
    command
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "3",
        ])
        .arg("--python")
        .arg(script)
        .arg("--")
        .arg(operations)
        .arg(root);
    checked(command, "Blender 5.2.2 oracle generation")?;
    Ok(())
}

fn inspected_item<'a>(response: &'a Value, id: &str) -> TestResult<&'a Value> {
    response["result"]["items"]
        .as_array()
        .ok_or("pot inspect did not return items")?
        .iter()
        .find(|item| item["id"].as_str() == Some(id))
        .ok_or_else(|| format!("pot inspect did not return ID {id}").into())
}

fn item_geometry<'a>(response: &'a Value, id: &str) -> TestResult<&'a Value> {
    let item = inspected_item(response, id)?;
    if !item["evaluation_error"].is_null() {
        return Err(format!(
            "node {id} has an evaluation error: {}",
            item["evaluation_error"]
        )
        .into());
    }
    let geometry = &item["evaluated_geometry"];
    if geometry.is_null() {
        return Err(format!("node {id} has no evaluated mesh").into());
    }
    Ok(geometry)
}
fn expected_geometry<'a>(oracle: &'a Value, id: &str) -> TestResult<&'a Value> {
    oracle["objects"]
        .get(id)
        .ok_or_else(|| format!("Blender oracle has no mesh for {id}").into())
}

fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label} differs: Potter={actual}, Blender={expected}, tolerance={tolerance}"
    );
}

fn surface_samples(mesh: &Geometry) -> Vec<DVec3> {
    let mut samples = mesh.positions.clone();
    samples.extend(
        mesh.triangles()
            .into_iter()
            .map(|[a, b, c]| (a + b + c) / 3.0),
    );
    samples
}

fn point_segment_distance_squared(point: DVec3, start: DVec3, end: DVec3) -> f64 {
    let segment = end - start;
    let length_squared = segment.length_squared();
    if length_squared <= f64::EPSILON {
        return point.distance_squared(start);
    }
    let factor = ((point - start).dot(segment) / length_squared).clamp(0.0, 1.0);
    point.distance_squared(start + segment * factor)
}

fn point_triangle_distance_squared(point: DVec3, [a, b, c]: [DVec3; 3]) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let normal = ab.cross(ac);
    let normal_length_squared = normal.length_squared();
    if normal_length_squared <= f64::EPSILON {
        return point_segment_distance_squared(point, a, b)
            .min(point_segment_distance_squared(point, b, c))
            .min(point_segment_distance_squared(point, c, a));
    }

    let projection = point - normal * ((point - a).dot(normal) / normal_length_squared);
    let relative = projection - a;
    let d00 = ab.dot(ab);
    let d01 = ab.dot(ac);
    let d11 = ac.dot(ac);
    let d20 = relative.dot(ab);
    let d21 = relative.dot(ac);
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() <= f64::EPSILON {
        return point_segment_distance_squared(point, a, b)
            .min(point_segment_distance_squared(point, b, c))
            .min(point_segment_distance_squared(point, c, a));
    }
    let second = (d11 * d20 - d01 * d21) / denominator;
    let third = (d00 * d21 - d01 * d20) / denominator;
    if second >= 0.0 && third >= 0.0 && second + third <= 1.0 {
        point.distance_squared(projection)
    } else {
        point_segment_distance_squared(point, a, b)
            .min(point_segment_distance_squared(point, b, c))
            .min(point_segment_distance_squared(point, c, a))
    }
}

fn directed_surface_distance(samples: &[DVec3], target: &Geometry) -> f64 {
    let triangles = target.triangles();
    samples
        .iter()
        .map(|point| {
            triangles
                .iter()
                .map(|triangle| point_triangle_distance_squared(*point, *triangle))
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .fold(0.0, f64::max)
}

fn assert_surface_equivalent(
    label: &str,
    potter: &Geometry,
    blender: &Geometry,
    relative_area_tolerance: f64,
    position_tolerance: f64,
) {
    let actual = potter.metrics();
    let expected = blender.metrics();
    let area_tolerance = expected.area * relative_area_tolerance;
    assert_close(
        &format!("{label} surface area"),
        actual.area,
        expected.area,
        area_tolerance,
    );
    for axis in 0..3 {
        assert_close(
            &format!("{label} minimum bound {axis}"),
            actual.minimum[axis],
            expected.minimum[axis],
            position_tolerance,
        );
        assert_close(
            &format!("{label} maximum bound {axis}"),
            actual.maximum[axis],
            expected.maximum[axis],
            position_tolerance,
        );
    }
    assert!(
        actual.area_centroid.distance(expected.area_centroid) <= position_tolerance,
        "{label} area centroid differs: Potter={:?}, Blender={:?}",
        actual.area_centroid,
        expected.area_centroid,
    );
    // A closed bevel has zero net oriented area, so float32 rounding leaves no stable winding direction.
    let actual_area_vector_length = actual.area_vector.length();
    let expected_area_vector_length = expected.area_vector.length();
    if actual_area_vector_length > area_tolerance || expected_area_vector_length > area_tolerance {
        assert!(
            actual_area_vector_length > area_tolerance
                && expected_area_vector_length > area_tolerance,
            "{label} has no stable oriented-area vector"
        );
        let actual_winding = actual.area_vector / actual_area_vector_length;
        let expected_winding = expected.area_vector / expected_area_vector_length;
        assert!(
            actual_winding.dot(expected_winding) > 0.99,
            "{label} winding differs: Potter={actual_winding:?}, Blender={expected_winding:?}"
        );
    }

    let hausdorff = directed_surface_distance(&surface_samples(potter), blender)
        .max(directed_surface_distance(&surface_samples(blender), potter));
    assert!(
        hausdorff <= position_tolerance,
        "{label} surface geometry differs: Hausdorff distance {hausdorff} exceeds {position_tolerance}"
    );
}

fn assert_vertex_geometry_matches(
    label: &str,
    potter: &Geometry,
    blender: &Geometry,
    position_tolerance: f64,
) {
    assert_eq!(
        potter.positions.len(),
        blender.positions.len(),
        "{label} vertex count"
    );
    assert_eq!(potter.faces, blender.faces, "{label} face topology");
    for (index, (actual, expected)) in potter.positions.iter().zip(&blender.positions).enumerate() {
        let distance = actual.distance(*expected);
        assert!(
            distance <= position_tolerance,
            "{label} vertex {index} differs: Potter={actual:?}, Blender={expected:?}, distance={distance}"
        );
    }
}

fn assert_nurbs_endpoint_rings_match(
    label: &str,
    potter: &Geometry,
    blender: &Geometry,
    position_tolerance: f64,
) {
    const RING_VERTEX_COUNT: usize = 8;
    assert!(
        potter.positions.len() >= 2 * RING_VERTEX_COUNT
            && blender.positions.len() >= 2 * RING_VERTEX_COUNT,
        "{label} is missing an endpoint ring"
    );
    for (end, actual_start, expected_start) in [
        ("first", 0, 0),
        (
            "last",
            potter.positions.len() - RING_VERTEX_COUNT,
            blender.positions.len() - RING_VERTEX_COUNT,
        ),
    ] {
        let actual_center = potter.positions[actual_start..actual_start + RING_VERTEX_COUNT]
            .iter()
            .copied()
            .fold(DVec3::ZERO, |center, position| center + position)
            / RING_VERTEX_COUNT as f64;
        let expected_center = blender.positions[expected_start..expected_start + RING_VERTEX_COUNT]
            .iter()
            .copied()
            .fold(DVec3::ZERO, |center, position| center + position)
            / RING_VERTEX_COUNT as f64;
        for offset in 0..RING_VERTEX_COUNT {
            let actual_index = actual_start + offset;
            let expected_index = expected_start + offset;
            let actual = potter.positions[actual_index] - actual_center;
            let expected = blender.positions[expected_index] - expected_center;
            let distance = actual.distance(expected);
            assert!(
                distance <= position_tolerance,
                "{label} {end} ring vertex {offset} differs (Potter index {actual_index}, Blender index {expected_index}): Potter={actual:?}, Blender={expected:?}, distance={distance}"
            );
        }
    }
}
fn root(parent: &mut [usize], index: usize) -> usize {
    if parent[index] != index {
        let next = parent[index];
        parent[index] = root(parent, next);
    }
    parent[index]
}

fn join(parent: &mut [usize], first: usize, second: usize) {
    let first_root = root(parent, first);
    let second_root = root(parent, second);
    parent[first_root] = second_root;
}

fn face_components(mesh: &Geometry) -> usize {
    let mut parent: Vec<usize> = (0..mesh.faces.len()).collect();
    let mut first_face_by_vertex = HashMap::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for vertex in face {
            if let Some(other_face) = first_face_by_vertex.insert(*vertex, face_index) {
                join(&mut parent, face_index, other_face);
            }
        }
    }
    (0..mesh.faces.len())
        .map(|index| root(&mut parent, index))
        .collect::<HashSet<_>>()
        .len()
}

fn contains_xy(mesh: &Geometry, point: [f64; 2]) -> bool {
    mesh.triangles().into_iter().any(|[a, b, c]| {
        let edge_ab = [b.x - a.x, b.y - a.y];
        let edge_ac = [c.x - a.x, c.y - a.y];
        let point_a = [point[0] - a.x, point[1] - a.y];
        let denominator = edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0];
        if denominator.abs() <= f64::EPSILON {
            return false;
        }
        let second = (point_a[0] * edge_ac[1] - point_a[1] * edge_ac[0]) / denominator;
        let third = (edge_ab[0] * point_a[1] - edge_ab[1] * point_a[0]) / denominator;
        second >= -1.0e-9 && third >= -1.0e-9 && second + third <= 1.0 + 1.0e-9
    })
}

fn assert_fill_regions(mesh: &Geometry, label: &str) {
    assert_eq!(
        face_components(mesh),
        3,
        "{label} must have three disconnected filled regions"
    );
    for (point, expected) in [
        ([1.0, 1.0], true),
        ([5.0, 4.0], false),
        ([13.0, 1.0], true),
        ([17.5, 1.5], true),
        ([11.0, 1.0], false),
        ([15.0, 1.0], false),
        ([-1.0, 1.0], false),
    ] {
        assert_eq!(
            contains_xy(mesh, point),
            expected,
            "{label} fill membership at {point:?}"
        );
    }
}

#[test]
fn curve_taper_and_multi_contour_fill_match_blender_522_through_cli() -> TestResult {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping curve taper/fill oracle: Blender is unavailable");
        return Ok(());
    };

    let root = tempdir()?;
    let builder_path = root.path().join("build_oracle.py");
    let operations_path = root.path().join("operations.json");
    fs::write(&builder_path, BUILDER)?;
    fs::write(&operations_path, OPERATIONS)?;
    let oracle_root = root.path().join("blender_oracle");
    run_blender(&blender, &builder_path, &operations_path, &oracle_root)?;
    assert!(
        oracle_root.join("source.blend").is_file(),
        "Blender did not save its source scene"
    );
    let oracle: Value = serde_json::from_slice(&fs::read(oracle_root.join("expected.json"))?)?;
    assert!(
        oracle["blender_version"]
            .as_str()
            .is_some_and(|version| version.starts_with("5.2.2")),
        "oracle must be generated by Blender 5.2.2, found {}",
        oracle["blender_version"]
    );

    let batch_path = root.path().join("batch.json");
    let batch = serde_json::json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": serde_json::from_str::<Value>(OPERATIONS)?,
    });
    fs::write(&batch_path, serde_json::to_vec(&batch)?)?;
    let project: PathBuf = root.path().join("project");
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    pot_json(|command| {
        command
            .arg("apply")
            .arg(&project)
            .arg("--file")
            .arg(&batch_path);
    })?;
    let host_response = pot_json(|command| {
        command
            .arg("inspect")
            .arg(&project)
            .args(["--id", "taper_multiply"]);
    })?;
    let host = inspected_item(&host_response, "taper_multiply")?;
    let data_id = host["data"]
        .as_str()
        .ok_or("taper host has no curve data-block ID")?;
    let data_response = pot_json(|command| {
        command.arg("inspect").arg(&project).args(["--id", data_id]);
    })?;
    let data_block = inspected_item(&data_response, data_id)?;
    assert_eq!(data_block["type"].as_str(), Some("data_block"));
    let curve = &data_block["curve"];
    assert_eq!(curve["taper"].as_str(), Some("taper_profile_data"));
    assert_eq!(curve["taper_radius_mode"].as_str(), Some("multiply"));
    assert_eq!(curve["use_map_taper"].as_bool(), Some(true));
    assert_eq!(curve["fill_mode"].as_str(), Some("none"));
    assert_eq!(curve["fill_solver"].as_str(), Some("sweep_line"));
    assert_eq!(curve["fill_rule"].as_str(), Some("even_odd"));
    assert_eq!(curve["resolution_u"].as_u64(), Some(12));

    let filled_node_response = pot_json(|command| {
        command
            .arg("inspect")
            .arg(&project)
            .args(["--id", "filled"]);
    })?;
    let filled_node = inspected_item(&filled_node_response, "filled")?;
    let filled_data_id = filled_node["data"]
        .as_str()
        .ok_or("filled curve node has no curve data-block ID")?;
    let filled_data_response = pot_json(|command| {
        command
            .arg("inspect")
            .arg(&project)
            .args(["--id", filled_data_id]);
    })?;
    let filled_curve = &inspected_item(&filled_data_response, filled_data_id)?["curve"];
    assert_eq!(filled_curve["fill_solver"].as_str(), Some("sweep_line"));
    assert_eq!(filled_curve["fill_rule"].as_str(), Some("nonzero"));
    assert_eq!(filled_curve["resolution_u"].as_u64(), Some(6));

    let mut cli_meshes = HashMap::new();
    for id in [
        "taper_override",
        "taper_multiply",
        "taper_add",
        "taper_2d_contours",
        "bevelled_taper_host",
        "filled",
        "nurbs_endpoint_on",
        "nurbs_endpoint_off",
    ] {
        let response = pot_json(|command| {
            command.arg("inspect").arg(&project).args(["--id", id]);
        })?;
        cli_meshes.insert(id, Geometry::from_json(item_geometry(&response, id)?)?);
    }

    let mut taper_areas = Vec::new();
    for id in ["taper_override", "taper_multiply", "taper_add"] {
        let blender_mesh = Geometry::from_json(expected_geometry(&oracle, id)?)?;
        taper_areas.push(blender_mesh.metrics().area);
        assert_surface_equivalent(
            &format!("taper radius mode {id}"),
            &cli_meshes[id],
            &blender_mesh,
            TAPER_AREA_RELATIVE_TOLERANCE,
            TAPER_SURFACE_TOLERANCE,
        );
    }
    assert!(
        (taper_areas[0] - taper_areas[1]).abs() > 1.0e-4
            && (taper_areas[1] - taper_areas[2]).abs() > 1.0e-4
            && (taper_areas[0] - taper_areas[2]).abs() > 1.0e-4,
        "Blender taper-mode fixtures must produce distinct surface areas: {taper_areas:?}"
    );

    let taper_2d = &cli_meshes["taper_2d_contours"];
    let blender_taper_2d = Geometry::from_json(expected_geometry(&oracle, "taper_2d_contours")?)?;
    assert_surface_equivalent(
        "2D cyclic taper outer contour and hole",
        taper_2d,
        &blender_taper_2d,
        TAPER_AREA_RELATIVE_TOLERANCE,
        TAPER_SURFACE_TOLERANCE,
    );
    assert_vertex_geometry_matches(
        "2D cyclic taper outer contour and hole",
        taper_2d,
        &blender_taper_2d,
        TAPER_SURFACE_TOLERANCE,
    );

    let bevelled_taper_host = &cli_meshes["bevelled_taper_host"];
    let blender_bevelled_taper =
        Geometry::from_json(expected_geometry(&oracle, "bevelled_taper_host")?)?;
    assert_surface_equivalent(
        "bevelled taper profile ring order",
        bevelled_taper_host,
        &blender_bevelled_taper,
        TAPER_AREA_RELATIVE_TOLERANCE,
        TAPER_SURFACE_TOLERANCE,
    );

    for id in ["nurbs_endpoint_on", "nurbs_endpoint_off"] {
        let blender_nurbs = Geometry::from_json(expected_geometry(&oracle, id)?)?;
        assert_nurbs_endpoint_rings_match(
            &format!("open 2D NURBS bevel endpoint direction {id}"),
            &cli_meshes[id],
            &blender_nurbs,
            TAPER_SURFACE_TOLERANCE,
        );
    }

    let filled = &cli_meshes["filled"];
    let blender_filled = Geometry::from_json(expected_geometry(&oracle, "filled")?)?;
    assert_surface_equivalent(
        "multi-contour fill",
        filled,
        &blender_filled,
        1.0e-8,
        1.0e-6,
    );
    assert_close(
        "Blender multi-contour filled area",
        blender_filled.metrics().area,
        77.0,
        1.0e-6,
    );
    assert_fill_regions(&blender_filled, "Blender");
    assert_fill_regions(filled, "Potter CLI");
    Ok(())
}
