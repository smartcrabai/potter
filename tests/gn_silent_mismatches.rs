use std::{
    collections::BTreeMap,
    error::Error,
    fs, io,
    path::Path,
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use serde_json::Value;
use tempfile::tempdir;

const BLENDER_REQUIRED: &str =
    "Blender 5.2.2 is required for the Geometry Nodes silent-mismatch oracle";
const POSITION_TOLERANCE: f64 = 1.0e-5;
const OBJECT_NAMES: [&str; 14] = [
    "GNPoissonInstances",
    "GNCreasedSubdivision",
    "GNSimulationDeltaTime",
    "GNPoissonCandidates",
    "GNRandomValueFloat",
    "GNRandomValueInt",
    "GNRandomValueBoolean",
    "GNRandomValueVector",
    "GNRandomValuePoissonID",
    "GNRandomSelectionFalse",
    "GNSmoothRandomDistribution",
    "GNZeroCreaseThenSubdivision",
    "GNStoreReadInt",
    "GNStoreReadBoolean",
];

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn run_ok(command: &mut Command, operation: &str) -> TestResult<Output> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "{operation} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
        .into());
    }
    Ok(output)
}

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn parse_vertex(values: &[Value]) -> TestResult<[f64; 3]> {
    if values.len() != 3 {
        return Err(format!("expected a 3D vertex, got {} values", values.len()).into());
    }
    Ok([
        values[0].as_f64().ok_or("vertex x is not numeric")?,
        values[1].as_f64().ok_or("vertex y is not numeric")?,
        values[2].as_f64().ok_or("vertex z is not numeric")?,
    ])
}

fn oracle_vertices(oracle: &Value, name: &str) -> TestResult<Vec<[f64; 3]>> {
    oracle["objects"][name]
        .as_array()
        .ok_or_else(|| io::Error::other(format!("Blender oracle is missing `{name}` vertices")))?
        .iter()
        .map(|vertex| {
            let values = vertex.as_array().ok_or_else(|| {
                io::Error::other(format!("`{name}` oracle vertex is not an array"))
            })?;
            parse_vertex(values)
        })
        .collect()
}

fn obj_vertices(text: &str) -> TestResult<BTreeMap<String, Vec<[f64; 3]>>> {
    let mut objects = BTreeMap::<String, Vec<[f64; 3]>>::new();
    let mut current = None;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("o" | "g") => {
                let name = fields.collect::<Vec<_>>().join(" ");
                if name.is_empty() || objects.insert(name.clone(), Vec::new()).is_some() {
                    return Err(
                        format!("OBJ has an empty or duplicate object name `{name}`").into(),
                    );
                }
                current = Some(name);
            }
            Some("v") => {
                let name = current
                    .as_ref()
                    .ok_or("OBJ vertex appeared before its object declaration")?;
                let coordinates = fields
                    .map(str::parse::<f64>)
                    .collect::<Result<Vec<_>, _>>()?;
                if coordinates.len() != 3 {
                    return Err(format!("OBJ vertex has {} coordinates", coordinates.len()).into());
                }
                // OBJ is Y-up; restore Blender's Z-up evaluation coordinates.
                objects
                    .get_mut(name)
                    .ok_or("OBJ object disappeared")?
                    .push([coordinates[0], -coordinates[2], coordinates[1]]);
            }
            _ => {}
        }
    }
    Ok(objects)
}

fn compare_vertices(name: &str, actual: &[[f64; 3]], expected: &[[f64; 3]]) -> TestResult {
    if actual.len() != expected.len() {
        return Err(format!(
            "{name}: Blender has {} vertices, Potter OBJ has {}",
            expected.len(),
            actual.len()
        )
        .into());
    }
    let mut matched = vec![false; actual.len()];
    for (expected_index, expected_vertex) in expected.iter().enumerate() {
        let mut closest: Option<(usize, f64)> = None;
        let mut best_match: Option<(usize, f64)> = None;
        for (actual_index, actual_vertex) in actual.iter().enumerate() {
            if matched[actual_index] {
                continue;
            }
            let error = (0..3)
                .map(|axis| (actual_vertex[axis] - expected_vertex[axis]).abs())
                .fold(0.0_f64, f64::max);
            if closest.is_none_or(|(_, closest_error)| error < closest_error) {
                closest = Some((actual_index, error));
            }
            if error <= POSITION_TOLERANCE
                && best_match.is_none_or(|(_, best_error)| error < best_error)
            {
                best_match = Some((actual_index, error));
            }
        }
        let Some((actual_index, _)) = best_match else {
            let Some((actual_index, error)) = closest else {
                return Err(format!(
                    "{name}: no unmatched Potter vertex for Blender vertex {expected_index}"
                )
                .into());
            };
            return Err(format!(
                "{name} vertex {expected_index} differs by {error:.3e}: Potter {:?}, Blender {expected_vertex:?}",
                actual[actual_index]
            )
            .into());
        };
        matched[actual_index] = true;
    }
    Ok(())
}

fn compare_vertices_ordered(name: &str, actual: &[[f64; 3]], expected: &[[f64; 3]]) -> TestResult {
    if actual.len() != expected.len() {
        return Err(format!(
            "{name}: Blender has {} vertices, Potter OBJ has {}",
            expected.len(),
            actual.len()
        )
        .into());
    }
    for (index, (actual_vertex, expected_vertex)) in actual.iter().zip(expected).enumerate() {
        let error = (0..3)
            .map(|axis| (actual_vertex[axis] - expected_vertex[axis]).abs())
            .fold(0.0_f64, f64::max);
        if error > POSITION_TOLERANCE {
            return Err(format!(
                "{name} vertex {index} differs by {error:.3e}: Potter {actual_vertex:?}, Blender {expected_vertex:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn sorted_object_names(objects: &BTreeMap<String, Vec<[f64; 3]>>) -> Vec<&str> {
    objects.keys().map(String::as_str).collect()
}

fn coordinate_mean_z(vertices: &[[f64; 3]]) -> f64 {
    vertices.iter().map(|vertex| vertex[2]).sum::<f64>() / vertices.len() as f64
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one end-to-end oracle test exercises Blender generation and both pot CLI exchanges"
)]
fn geometry_nodes_silent_mismatches_match_blender_522_obj_oracle() -> TestResult {
    let blender = blender_executable().ok_or(BLENDER_REQUIRED)?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gn_silent_mismatches/generate.py");
    let temp = tempdir()?;
    let output_dir = fs::canonicalize(temp.path())?;
    run_ok(
        Command::new(&blender)
            .args([
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--threads",
                "1",
                "--python-exit-code",
                "1",
                "--python",
            ])
            .arg(&fixture)
            .arg("--")
            .arg(&output_dir),
        "Blender 5.2.2 GN oracle generation",
    )?;

    let oracle_path = output_dir.join("oracle.json");
    let blend_path = output_dir.join("gn_silent_mismatches.blend");
    if !blend_path.is_file() {
        return Err("Blender generator did not write the .blend scene".into());
    }
    let oracle: Value = serde_json::from_slice(&fs::read(oracle_path)?)?;
    let blender_version = oracle["blender_version"]
        .as_str()
        .ok_or("Blender oracle version is missing")?;
    if !blender_version.starts_with("5.2.2") {
        return Err(format!("expected Blender 5.2.2, got {blender_version}").into());
    }
    let fps_base = oracle["fps_base"]
        .as_f64()
        .ok_or("Blender oracle frame rate base is missing")?;
    if oracle["frame"] != 2 || oracle["fps"] != 30 || (fps_base - 1.001).abs() > 1.0e-6 {
        return Err(format!(
            "Blender oracle does not use frame 2 at 30 fps with fps_base 1.001: {fps_base}"
        )
        .into());
    }
    if oracle["poisson"]["method"] != "POISSON"
        || oracle["poisson"]["density"] != 23.0
        || oracle["poisson"]["density_factor"] != 0.35
        || oracle["poisson"]["density_max"] != 6.0
        || oracle["poisson"]["distance_min"] != 0.45
    {
        return Err("Blender oracle no longer exercises non-default Poisson inputs".into());
    }
    if oracle["realize"]["depth"] != 5
        || oracle["realize"]["realize_all"] != true
        || oracle["realize"]["selection"] != true
    {
        return Err(
            "Blender oracle does not exercise Realize Instances Depth with Realize All enabled"
                .into(),
        );
    }
    let point_count = oracle["poisson"]["point_count"]
        .as_u64()
        .ok_or("Blender Poisson point count is missing")?;
    if !(1..40).contains(&point_count) {
        return Err(format!("unexpected Blender Poisson point count {point_count}").into());
    }
    let poisson = oracle_vertices(&oracle, OBJECT_NAMES[0])?;
    if poisson.len() != point_count as usize * 8 {
        return Err(format!(
            "Poisson instances produced {} vertices for {point_count} points",
            poisson.len()
        )
        .into());
    }
    if oracle["poisson_candidates"]["method"] != "POISSON"
        || oracle["poisson_candidates"]["density_factor"] != 1.0
        || oracle["poisson_candidates"]["density_max"] != 6.0
        || oracle["poisson_candidates"]["distance_min"] != 0.0
    {
        return Err(
            "Blender oracle does not exercise unfiltered POISSON candidate sampling".into(),
        );
    }
    let candidate_count = oracle["poisson_candidates"]["point_count"]
        .as_u64()
        .ok_or("Blender POISSON candidate count is missing")?;
    if candidate_count != 24 {
        return Err(format!("expected 24 POISSON candidates, got {candidate_count}").into());
    }
    let poisson_candidates = oracle_vertices(&oracle, OBJECT_NAMES[3])?;
    if poisson_candidates.len() != candidate_count as usize * 8 {
        return Err("Blender POISSON candidate sample geometry is incomplete".into());
    }

    let random_values = &oracle["random_values"];
    if random_values["float"]["data_type"] != "FLOAT"
        || random_values["float"]["minimum"] != -1.25
        || random_values["float"]["maximum"] != 0.75
        || random_values["float"]["seed"] != 29
        || random_values["int"]["data_type"] != "INT"
        || random_values["int"]["minimum"] != -7
        || random_values["int"]["maximum"] != 12
        || random_values["int"]["seed"] != 29
        || random_values["int"]["attribute_name"] != "oracle_random_int"
        || random_values["boolean"]["data_type"] != "BOOLEAN"
        || random_values["boolean"]["probability"] != 0.35
        || random_values["boolean"]["seed"] != 29
        || random_values["vector"]["data_type"] != "FLOAT_VECTOR"
        || random_values["vector"]["minimum"] != serde_json::json!([-0.75, -1.25, -0.5])
        || random_values["vector"]["maximum"] != serde_json::json!([0.75, 1.25, 0.5])
        || random_values["vector"]["seed"] != 29
    {
        return Err("Blender oracle no longer exercises non-default Random Value inputs".into());
    }
    for kind in ["float", "int", "boolean", "vector"] {
        if random_values[kind]["mesh_line_count"] != 8 {
            return Err(
                format!("Blender Random Value {kind} does not use Mesh Line Count 8").into(),
            );
        }
    }
    for name in OBJECT_NAMES[4..8].iter().copied() {
        if oracle_vertices(&oracle, name)?.len() != 8 {
            return Err(format!("{name} Blender oracle should contain eight vertices").into());
        }
    }
    let random_int_oracle = random_values["int"]["values"]
        .as_array()
        .ok_or("Blender Random Value INT oracle values are missing")?;
    if random_int_oracle.len() != 8
        || random_int_oracle
            .iter()
            .any(|value| !matches!(value.as_i64(), Some(value) if (-7..=12).contains(&value)))
    {
        return Err("Blender Random Value INT oracle values are invalid".into());
    }
    if oracle["random_poisson_id"]["method"] != "POISSON"
        || oracle["random_poisson_id"]["density"] != 23.0
        || oracle["random_poisson_id"]["density_factor"] != 1.0
        || oracle["random_poisson_id"]["density_max"] != 6.0
        || oracle["random_poisson_id"]["distance_min"] != 0.0
        || oracle["random_poisson_id"]["distribute_seed"] != 17
        || oracle["random_poisson_id"]["data_type"] != "FLOAT_VECTOR"
        || oracle["random_poisson_id"]["minimum"] != serde_json::json!([-0.3, -0.4, -0.5])
        || oracle["random_poisson_id"]["maximum"] != serde_json::json!([0.3, 0.4, 0.5])
        || oracle["random_poisson_id"]["random_seed"] != 29
        || oracle["random_poisson_id"]["id_linked"] != false
    {
        return Err(
            "Blender oracle does not exercise Random Value's unlinked ID after POISSON".into(),
        );
    }
    let random_poisson_count = oracle["random_poisson_id"]["point_count"]
        .as_u64()
        .ok_or("Blender Random Value POISSON point count is missing")?;
    if random_poisson_count == 0
        || oracle_vertices(&oracle, OBJECT_NAMES[8])?.len() != random_poisson_count as usize * 8
    {
        return Err("Random Value POISSON oracle did not retain its scattered points".into());
    }
    if oracle["random_selection"]["method"] != "RANDOM"
        || oracle["random_selection"]["density"] != 8.0
        || oracle["random_selection"]["selection"] != false
    {
        return Err(
            "Blender oracle does not exercise RANDOM distribution with Selection=false".into(),
        );
    }
    let random_selection_count = oracle["random_selection"]["point_count"]
        .as_u64()
        .ok_or("Blender RANDOM point count is missing")?;
    if random_selection_count == 0
        || oracle_vertices(&oracle, OBJECT_NAMES[9])?.len() != random_selection_count as usize * 8
    {
        return Err("Blender RANDOM distribution did not scatter with Selection=false".into());
    }
    let smooth_random = &oracle["smooth_random"];
    if smooth_random["method"] != "RANDOM"
        || smooth_random["density"] != 2.0
        || smooth_random["selection"] != true
        || smooth_random["smooth_shading"] != true
        || smooth_random["has_custom_normals"] != true
    {
        return Err(
            "Blender oracle does not exercise RANDOM distribution on a smooth custom-normal mesh"
                .into(),
        );
    }
    let smooth_point_count = smooth_random["point_count"]
        .as_u64()
        .ok_or("Blender smooth RANDOM point count is missing")?;
    let smooth_random_vertices = oracle_vertices(&oracle, OBJECT_NAMES[10])?;
    if smooth_point_count == 0 || smooth_random_vertices.len() != smooth_point_count as usize * 8 {
        return Err("smooth RANDOM distribution did not retain its scattered points".into());
    }
    let zero_crease_subdivision = &oracle["zero_crease_subdivision"];
    let input_edge_crease_max = zero_crease_subdivision["input_edge_crease_max"]
        .as_f64()
        .ok_or("Blender zero-crease subdivision input crease is missing")?;
    if zero_crease_subdivision["node_level"] != 1
        || zero_crease_subdivision["edge_crease_default"] != 0.0
        || zero_crease_subdivision["vertex_crease_default"] != 0.0
        || input_edge_crease_max <= 0.0
        || zero_crease_subdivision["modifier_level"] != 1
        || zero_crease_subdivision["modifier_uses_creases"] != true
    {
        return Err(
            "Blender oracle does not exercise creased input with default-zero GN crease fields and a crease-aware Subdivision modifier"
                .into(),
        );
    }
    if oracle_vertices(&oracle, OBJECT_NAMES[11])?.len() <= 8 {
        return Err("zero-crease Geometry Nodes subdivision did not add geometry".into());
    }
    let named_attributes = &oracle["named_attributes"];
    for (kind, data_type, name) in [
        ("int", "INT", OBJECT_NAMES[12]),
        ("boolean", "BOOLEAN", OBJECT_NAMES[13]),
    ] {
        if named_attributes[kind]["data_type"] != data_type
            || named_attributes[kind]["attribute_name"] != "roundtrip_value"
            || named_attributes[kind]["domain"] != "POINT"
            || named_attributes[kind]["mesh_line_count"] != 4
            || (kind == "int" && named_attributes[kind]["copy_attribute_name"] != "roundtrip_copy")
        {
            return Err(format!(
                "Blender oracle does not exercise same-type {data_type} store/read"
            )
            .into());
        }
        if oracle_vertices(&oracle, name)?.len() != 4 {
            return Err(format!(
                "Blender {data_type} store/read output should contain four vertices"
            )
            .into());
        }
    }
    let named_int_oracle = named_attributes["int"]["values"]
        .as_array()
        .ok_or("Blender named INT store/read values are missing")?;
    if named_int_oracle.len() != 4
        || named_int_oracle
            .iter()
            .any(|value| !matches!(value.as_i64(), Some(value) if (0..4).contains(&value)))
    {
        return Err("Blender named INT store/read oracle values are invalid".into());
    }
    let stored_boolean = oracle_vertices(&oracle, OBJECT_NAMES[13])?;
    if !stored_boolean
        .iter()
        .any(|vertex| vertex[2].abs() > POSITION_TOLERANCE)
    {
        return Err(
            "BOOLEAN Store Named Attribute -> Named Attribute did not affect positions".into(),
        );
    }
    if oracle["subdivision"]["edge_crease"] == 0.0 || oracle["subdivision"]["vertex_crease"] == 0.0
    {
        return Err("Blender oracle does not exercise both subdivision crease inputs".into());
    }
    let subdivision = oracle_vertices(&oracle, OBJECT_NAMES[1])?;
    if subdivision.len() <= 8 {
        return Err("Blender subdivision oracle did not add subdivided vertices".into());
    }
    let simulation = oracle_vertices(&oracle, OBJECT_NAMES[2])?;
    if simulation.len() != 3 || !(0.01..0.1).contains(&coordinate_mean_z(&simulation)) {
        return Err(format!(
            "Delta Time did not translate the frame-2 mesh at 30 fps: {simulation:?}"
        )
        .into());
    }

    let project = output_dir.join("project");
    run_ok(pot().arg("init").arg(&project).arg("--json"), "pot init")?;
    run_ok(
        pot()
            .arg("import")
            .arg(&project)
            .args(["--file"])
            .arg(&blend_path)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender)
            .arg("--json"),
        "pot import of Blender GN scene",
    )?;
    let project_scene_path = project.join("scene.json");
    let imported_scene: Value = serde_json::from_slice(&fs::read(&project_scene_path)?)?;
    let nodes = imported_scene["nodes"]
        .as_object()
        .ok_or("imported Blender nodes are missing")?;
    let base_revision = imported_scene["revision"]
        .as_u64()
        .ok_or("imported scene revision is missing")?;
    let mut apply_operations = Vec::with_capacity(2);
    let mut random_int_data_id = None;
    let mut named_int_data_id = None;
    for object_name in [OBJECT_NAMES[5], OBJECT_NAMES[12]] {
        let (node_id, node) = nodes
            .iter()
            .find(|(_, node)| node["name"].as_str() == Some(object_name))
            .ok_or_else(|| format!("imported Blender object {object_name} is missing"))?;
        let modifier_id = node["modifiers"]
            .as_array()
            .and_then(|modifiers| modifiers.first())
            .and_then(|modifier| modifier["id"].as_str())
            .ok_or_else(|| format!("{object_name} Geometry Nodes modifier is missing"))?;
        let data_id = node["data"]
            .as_str()
            .ok_or_else(|| format!("{object_name} mesh data ID is missing"))?;
        apply_operations.push(serde_json::json!({
            "op": "modifier.apply",
            "target": {"id": node_id},
            "modifier_id": modifier_id,
        }));
        if object_name == OBJECT_NAMES[5] {
            random_int_data_id = Some(data_id.to_owned());
        } else {
            named_int_data_id = Some(data_id.to_owned());
        }
    }
    let random_int_data_id =
        random_int_data_id.ok_or("Random Value INT mesh data ID is missing")?;
    let named_int_data_id =
        named_int_data_id.ok_or("Named Attribute INT mesh data ID is missing")?;
    let apply_batch_path = output_dir.join("apply_integer_attributes.json");
    fs::write(
        &apply_batch_path,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "base_revision": base_revision,
            "operations": apply_operations,
        }))?,
    )?;
    run_ok(
        pot()
            .arg("apply")
            .arg(&project)
            .arg("--file")
            .arg(&apply_batch_path)
            .arg("--json"),
        "pot apply of INT Geometry Nodes attributes",
    )?;
    let applied_scene: Value = serde_json::from_slice(&fs::read(&project_scene_path)?)?;
    let random_attribute = &applied_scene["data_blocks"][random_int_data_id.as_str()]["mesh"]["attributes"]
        ["oracle_random_int"];
    if random_attribute["domain"] != "POINT" || random_attribute["data_type"] != "INT" {
        return Err("Potter did not persist the Random Value INT point attribute".into());
    }
    let applied_random_ints = random_attribute["data"]
        .as_array()
        .ok_or("Potter Random Value INT attribute data is missing")?;
    if applied_random_ints.len() != random_int_oracle.len() {
        return Err("Potter Random Value INT attribute has the wrong point count".into());
    }
    for (index, (actual, expected)) in applied_random_ints
        .iter()
        .zip(random_int_oracle)
        .enumerate()
    {
        let actual = actual
            .as_i64()
            .ok_or("Potter Random Value INT attribute contains a non-integer")?;
        let expected = expected
            .as_i64()
            .ok_or("Blender Random Value INT oracle contains a non-integer")?;
        if actual != expected {
            return Err(format!(
                "Random Value INT point {index} differs: Potter={actual}, Blender={expected}"
            )
            .into());
        }
    }
    let named_attribute = &applied_scene["data_blocks"][named_int_data_id.as_str()]["mesh"]["attributes"]
        ["roundtrip_copy"];
    if named_attribute["domain"] != "POINT" || named_attribute["data_type"] != "INT" {
        return Err("Potter did not persist the same-type Named Attribute copy".into());
    }
    let applied_named_ints = named_attribute["data"]
        .as_array()
        .ok_or("Potter Named Attribute INT data is missing")?;
    if applied_named_ints.len() != named_int_oracle.len() {
        return Err("Potter Named Attribute INT data has the wrong point count".into());
    }
    for (index, (actual, expected)) in applied_named_ints.iter().zip(named_int_oracle).enumerate() {
        let actual = actual
            .as_i64()
            .ok_or("Potter Named Attribute INT data contains a non-integer")?;
        let expected = expected
            .as_i64()
            .ok_or("Blender Named Attribute INT oracle contains a non-integer")?;
        if actual != expected {
            return Err(format!(
                "Named Attribute INT point {index} differs: Potter={actual}, Blender={expected}"
            )
            .into());
        }
    }

    let obj_path = output_dir.join("potter.obj");
    run_ok(
        pot()
            .arg("export")
            .arg(&project)
            .args(["--format", "obj", "--frame", "2", "--allow-lossy", "--out"])
            .arg(&obj_path)
            .arg("--json"),
        "pot OBJ export",
    )?;

    let actual = obj_vertices(&fs::read_to_string(obj_path)?)?;
    let mut expected_names = OBJECT_NAMES.to_vec();
    expected_names.sort_unstable();
    if sorted_object_names(&actual) != expected_names {
        return Err(format!(
            "OBJ objects were {:?}, expected {:?}",
            sorted_object_names(&actual),
            expected_names
        )
        .into());
    }
    for name in [
        OBJECT_NAMES[0],
        OBJECT_NAMES[3],
        OBJECT_NAMES[4],
        OBJECT_NAMES[5],
        OBJECT_NAMES[6],
        OBJECT_NAMES[7],
        OBJECT_NAMES[8],
        OBJECT_NAMES[9],
        OBJECT_NAMES[10],
        OBJECT_NAMES[12],
        OBJECT_NAMES[13],
    ] {
        let expected = oracle_vertices(&oracle, name)?;
        compare_vertices_ordered(name, &actual[name], &expected)?;
    }
    for name in [OBJECT_NAMES[1], OBJECT_NAMES[2], OBJECT_NAMES[11]] {
        let expected = oracle_vertices(&oracle, name)?;
        compare_vertices(name, &actual[name], &expected)?;
    }
    eprintln!(
        "GN Blender 5.2.2 CLI oracle: POISSON {} selected / {} candidates; exported vertices {:?}; frame-2 mean Z {:.9} at {} fps / fps_base {}",
        point_count,
        candidate_count,
        OBJECT_NAMES.map(|name| actual[name].len()),
        coordinate_mean_z(&actual[OBJECT_NAMES[2]]),
        oracle["fps"],
        oracle["fps_base"],
    );
    Ok(())
}
