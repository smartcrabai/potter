use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_ok(command: &mut Command) -> Result<std::process::Output, Box<dyn Error>> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn init(scene: &Path) -> Result<(), Box<dyn Error>> {
    run_ok(pot().args(["init"]).arg(scene).arg("--json"))?;
    Ok(())
}

fn apply(scene: &Path, operations: &Value) -> Result<(), Box<dyn Error>> {
    let document_path = scene.join("scene.json");
    let document: Value = serde_json::from_slice(&fs::read(&document_path)?)?;
    let batch = scene.with_extension("alembic-cache-ops.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": document["revision"],
            "operations": operations,
        }))?,
    )?;
    run_ok(
        pot()
            .args(["apply"])
            .arg(scene)
            .args(["--file"])
            .arg(batch)
            .arg("--json"),
    )?;
    Ok(())
}
fn apply_output(scene: &Path, operations: &Value) -> Result<std::process::Output, Box<dyn Error>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let batch = scene.with_extension("sequence-unpack-ops.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":document["revision"],
            "operations":operations
        }))?,
    )?;
    Ok(pot()
        .args(["apply"])
        .arg(scene)
        .args(["--file"])
        .arg(batch)
        .arg("--json")
        .output()?)
}
fn assert_failed_sequence_unpack_is_atomic(
    directory: &Path,
    name: &str,
    source_document: &Value,
    expected_code: &str,
) -> Result<(), Box<dyn Error>> {
    let scene = directory.join(name);
    init(&scene)?;
    let scene_path = scene.join("scene.json");
    fs::write(&scene_path, serde_json::to_vec_pretty(source_document)?)?;
    let pattern = scene.join("unpacked.####.abc");
    let existing = scene.join("unpacked.0001.abc");
    let sentinel = b"existing sequence frame must not be overwritten";
    fs::write(&existing, sentinel)?;
    let absent = scene.join("unpacked.0002.abc");
    assert!(!absent.exists());

    let output = apply_output(
        &scene,
        &json!([{"op":"resource.unpack","id":"packed_sequence","path":pattern}]),
    )?;
    assert!(
        !output.status.success(),
        "invalid packed sequence was unpacked"
    );
    let error: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(error["error"]["code"], expected_code);
    let actual = fs::read(&existing)?;
    assert_eq!(actual.as_slice(), &sentinel[..]);
    assert!(!absent.exists(), "failed unpack created a later frame");
    for frame in 3..=4 {
        assert!(
            !scene.join(format!("unpacked.{frame:04}.abc")).exists(),
            "failed unpack created frame {frame}"
        );
    }
    let after: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    assert_eq!(&after, source_document);
    Ok(())
}

fn export_animated_cache(scene: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    init(scene)?;
    apply(
        scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"plane", "params":{}},
            {"op":"modifier.create", "target":{"id":"body"}, "id":"wave", "type":"wave", "params":{"speed":0.25,"height":0.5,"width":1.5,"narrowness":1.5}},
            {"op":"action.create", "id":"samples", "name":"Samples"},
            {"op":"node.update", "target":{"id":"body"}, "set":{"action":"samples"}},
            {"op":"keyframe.insert", "target":{"id":"body"}, "path":"transform.translation", "index":0, "frame":1.0, "value":0.0},
            {"op":"keyframe.insert", "target":{"id":"body"}, "path":"transform.translation", "index":0, "frame":3.0, "value":0.0}
        ]),
    )?;
    run_ok(
        pot()
            .args(["export"])
            .arg(scene)
            .args(["--format", "alembic", "--allow-lossy", "--out"])
            .arg(output)
            .args(["--overwrite", "--json"]),
    )?;
    Ok(())
}

fn add_external_cache_modifier(
    scene: &Path,
    resource_path: &Path,
    expected_hash: &str,
) -> Result<(), Box<dyn Error>> {
    let document_path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&document_path)?)?;
    document["resources"]["cache"] = json!({
        "uri": resource_path,
        "hash": expected_hash,
        "kind": "alembic",
        "packed": false,
    });
    fs::write(&document_path, serde_json::to_vec_pretty(&document)?)?;
    apply(
        scene,
        &json!([{
            "op":"modifier.create",
            "target":{"id":"body"},
            "id":"cache",
            "type":"mesh_sequence_cache",
            "params":{
                "resource":"cache",
                "object_path":"/root/body/geometry",
                "read_data":["VERT","POLY","UV","COLOR"],
                "use_vertex_interpolation":true
            }
        }]),
    )?;
    Ok(())
}

fn inspect(scene: &Path, frame: f64) -> Result<std::process::Output, Box<dyn Error>> {
    Ok(pot()
        .args(["inspect"])
        .arg(scene)
        .args(["--id", "body", "--frame"])
        .arg(frame.to_string())
        .arg("--json")
        .output()?)
}

fn bounds(scene: &Path, frame: f64) -> Result<Value, Box<dyn Error>> {
    let output = run_ok(
        pot()
            .args(["inspect"])
            .arg(scene)
            .args(["--id", "body", "--frame"])
            .arg(frame.to_string())
            .arg("--json"),
    )?;
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    let bounds = envelope
        .get("result")
        .and_then(|result| result.get("items"))
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("bounds"))
        .ok_or("inspect response is missing evaluated bounds")?;
    Ok(bounds.clone())
}

fn mesh_positions(scene: &Path, frame: f64) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let project = potter_core::store::Project::open(scene)?;
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        project.doc(),
        &potter_core::eval::EvaluationContext {
            frame: Some(frame),
            ..potter_core::eval::EvaluationContext::default()
        },
        Some(project.path()),
        Some(project.cache_directory()),
    )?;
    let mesh = snapshot
        .meshes
        .values()
        .next()
        .ok_or("evaluated mesh is missing")?;
    Ok(mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect())
}

fn assert_positions_close(actual: &[[f64; 3]], expected: &[[f64; 3]], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
        assert!((actual - expected).abs() <= tolerance);
    }
}

fn evaluated_mesh(scene: &Path, frame: f64) -> Result<potter_core::geom::Mesh, Box<dyn Error>> {
    let project = potter_core::store::Project::open(scene)?;
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        project.doc(),
        &potter_core::eval::EvaluationContext {
            frame: Some(frame),
            ..potter_core::eval::EvaluationContext::default()
        },
        Some(project.path()),
        Some(project.cache_directory()),
    )?;
    Ok(snapshot
        .meshes
        .values()
        .next()
        .ok_or("evaluated mesh is missing")?
        .clone())
}

fn assert_numeric_json_close(actual: &Value, expected: &Value, tolerance: f64) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual.as_f64().unwrap_or(f64::NAN);
            let expected = expected.as_f64().unwrap_or(f64::NAN);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} differs from {expected} by more than {tolerance}"
            );
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert_numeric_json_close(actual, expected, tolerance);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

fn assert_bounds_close(actual: &Value, expected: &Value) -> Result<(), Box<dyn Error>> {
    for side in ["min", "max"] {
        let actual_values = actual
            .get(side)
            .and_then(Value::as_array)
            .ok_or("inspect bounds are missing a coordinate array")?;
        let expected_values = expected
            .get(side)
            .and_then(Value::as_array)
            .ok_or("expected bounds are missing a coordinate array")?;
        for (actual, expected) in actual_values.iter().zip(expected_values) {
            let actual = actual
                .as_f64()
                .ok_or("actual bounds coordinate is not numeric")?;
            let expected = expected
                .as_f64()
                .ok_or("expected bounds coordinate is not numeric")?;
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }
    Ok(())
}

fn midpoint_position(first: [f64; 3], last: [f64; 3]) -> [f64; 3] {
    let [first_x, first_y, first_z] = first;
    let [last_x, last_y, last_z] = last;
    [
        first_x + (last_x - first_x) * 0.5,
        first_y + (last_y - first_y) * 0.5,
        first_z + (last_z - first_z) * 0.5,
    ]
}

#[test]
fn mesh_sequence_cache_samples_and_interpolates_exported_alembic_meshes()
-> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let source = temporary.path().join("source.pot");
    let target = temporary.path().join("target.pot");
    let archive = temporary.path().join("animated.abc");
    export_animated_cache(&source, &archive)?;
    init(&target)?;
    apply(
        &target,
        &json!([
            {"op":"node.create", "id":"body", "kind":"plane", "params":{}},
            {"op":"resource.pack", "id":"cache", "uri":archive, "kind":"alembic"},
            {"op":"modifier.create", "target":{"id":"body"}, "id":"cache", "type":"mesh_sequence_cache", "params":{"resource":"cache", "object_path":"/root/body/geometry", "read_data":["VERT","POLY","UV","COLOR"], "use_vertex_interpolation":true}}
        ]),
    )?;
    let archive_hash = potter_core::hash::sha256(&fs::read(&archive)?);
    let hash_hex = archive_hash.strip_prefix("sha256:").unwrap_or_default();
    let asset_uri = format!("assets/sha256/{hash_hex}/animated.abc");
    let asset_path = target.join(&asset_uri);
    fs::create_dir_all(asset_path.parent().ok_or("asset path has no parent")?)?;
    fs::copy(&archive, &asset_path)?;
    let scene_path = target.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    document["resources"]["cache"]["uri"] = json!(asset_uri);
    document["resources"]["cache"]["packed"] = json!(false);
    if let Some(resource) = document["resources"]["cache"].as_object_mut() {
        resource.remove("bytes");
    }
    fs::write(scene_path, serde_json::to_vec_pretty(&document)?)?;

    for frame in [1.0, 3.0] {
        assert_positions_close(
            &mesh_positions(&target, frame)?,
            &mesh_positions(&source, frame)?,
            1.0e-6,
        );
        assert_bounds_close(&bounds(&target, frame)?, &bounds(&source, frame)?)?;
    }
    let first = mesh_positions(&target, 1.0)?;
    let last = mesh_positions(&target, 3.0)?;
    let expected_middle = first
        .iter()
        .zip(&last)
        .map(|(first, last)| midpoint_position(*first, *last))
        .collect::<Vec<_>>();
    assert_positions_close(&mesh_positions(&target, 2.0)?, &expected_middle, 1.0e-12);
    Ok(())
}

fn set_receiver_mesh(
    scene: &Path,
    positions: &[[f64; 3]],
    faces: &[Vec<usize>],
    seeded_uv: bool,
) -> Result<(), Box<dyn Error>> {
    let scene_path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    let data_id = document["nodes"]["body"]["data"]
        .as_str()
        .ok_or("oracle receiver mesh data block is missing")?
        .to_owned();
    let mut mesh = potter_core::geom::Mesh::from_positions_and_faces(
        positions
            .iter()
            .map(|position| glam::DVec3::from_array(*position))
            .collect(),
        faces.to_vec(),
    )?;
    for (name, domain, prefix, ids) in [
        (
            "sharp",
            "edge",
            "e",
            mesh.edges.iter().map(|edge| edge.id).collect::<Vec<_>>(),
        ),
        (
            "shade_smooth",
            "face",
            "f",
            mesh.faces.iter().map(|face| face.id).collect::<Vec<_>>(),
        ),
    ] {
        let values = ids
            .into_iter()
            .map(|id| (format!("{prefix}{id}"), json!(false)))
            .collect::<serde_json::Map<_, _>>();
        mesh.attributes.insert(
            name.to_owned(),
            json!({"domain":domain,"type":"bool","values":values}),
        );
    }
    if seeded_uv {
        let rows = mesh
            .faces
            .iter()
            .enumerate()
            .map(|(face_index, face)| {
                let uv = (0..face.vertices.len())
                    .map(|corner| json!([20 + face_index, 10 + corner]))
                    .collect::<Vec<_>>();
                json!({"face_id":face.id,"uv":uv})
            })
            .collect::<Vec<_>>();
        mesh.attributes.insert(
            "uv_layers".to_owned(),
            json!({
                "layers":[{"name":"UVMap","faces":rows}],
                "active":"UVMap",
                "render":"UVMap",
            }),
        );
    }
    document["data_blocks"][data_id.as_str()]["descriptor"] = Value::Null;
    document["data_blocks"][data_id.as_str()]["mesh"] = serde_json::to_value(mesh)?;
    fs::write(&scene_path, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

fn oracle_receiver_geometry(case_id: &str) -> (Vec<[f64; 3]>, Vec<Vec<usize>>) {
    if matches!(
        case_id,
        "topology_changed_cube_input"
            | "topology_changed_cube_input_vert_only"
            | "topology_face_change_full_read"
            | "topology_face_change_vert_only"
    ) {
        (
            vec![
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            vec![
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![1, 2, 6, 5],
                vec![2, 3, 7, 6],
                vec![3, 0, 4, 7],
            ],
        )
    } else {
        (
            vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [-0.5, 1.5, 0.5],
                [1.5, 2.5, -0.5],
            ],
            vec![vec![0, 1, 2, 3], vec![0, 3, 4], vec![3, 2, 5, 4]],
        )
    }
}

fn add_sequence_resource(
    scene: &Path,
    fixture_dir: &Path,
    cache_file: &Value,
) -> Result<(), Box<dyn Error>> {
    let filename = Path::new(
        cache_file["filepath"]
            .as_str()
            .ok_or("Blender oracle sequence filepath is missing")?,
    )
    .file_name()
    .and_then(|name| name.to_str())
    .ok_or("Blender oracle sequence filename is missing")?;
    let extension_start = filename.rfind('.').unwrap_or(filename.len());
    let stem = &filename[..extension_start];
    let frame_start = stem
        .as_bytes()
        .iter()
        .rposition(|byte| !byte.is_ascii_digit())
        .map_or(0, |index| index + 1);
    if frame_start == stem.len() {
        return Err("Blender oracle sequence path must end in a numbered filename".into());
    }
    let frame_width = extension_start - frame_start;
    let first_frame = stem[frame_start..].parse::<i32>()?;
    let prefix = &filename[..frame_start];
    let suffix = &filename[extension_start..];
    let pattern_name = format!("{prefix}{}{suffix}", "#".repeat(frame_width));
    let sequence_pattern = fixture_dir.join(pattern_name);
    let mut sequence_frames = serde_json::Map::new();
    let mut frame = first_frame;
    loop {
        let source = fixture_dir.join(format!("{prefix}{frame:0frame_width$}{suffix}"));
        if !source.is_file() {
            break;
        }
        let digest = potter_core::hash::sha256(&fs::read(&source)?);
        sequence_frames.insert(
            frame.to_string(),
            json!({
                "uri":source,
                "hash":digest,
                "kind":"alembic",
                "packed":false,
            }),
        );
        frame += 1;
    }
    let end = frame - 1;
    if sequence_frames.is_empty() {
        return Err("Blender oracle sequence has no numbered frame files".into());
    }
    let scene_path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    document["resources"]["oracle_cache"] = json!({
        "uri":sequence_pattern,
        "kind":"alembic",
        "packed":false,
        "cache_file":cache_file,
        "sequence_start":first_frame,
        "sequence_end":end,
        "sequence_frames":sequence_frames
    });
    fs::write(&scene_path, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

fn oracle_cache_params(oracle_case: &Value) -> Result<Value, Box<dyn Error>> {
    let cache_file = &oracle_case["cache_file"];
    let mut params = oracle_case["modifier"]
        .as_object()
        .ok_or("Blender oracle modifier settings are missing")?
        .clone();
    params.insert("resource".to_owned(), json!("oracle_cache"));
    for field in [
        "frame_offset",
        "is_sequence",
        "velocity_name",
        "velocity_unit",
    ] {
        params.insert(field.to_owned(), cache_file[field].clone());
    }
    params.insert(
        "override_frame".to_owned(),
        if cache_file["override_frame"] == true {
            cache_file["frame"].clone()
        } else {
            Value::Null
        },
    );
    Ok(Value::Object(params))
}

fn oracle_actual_attribute_values(
    mesh: &potter_core::geom::Mesh,
    attribute: &Value,
) -> Result<Value, Box<dyn Error>> {
    let values = attribute["values"]
        .as_object()
        .ok_or("evaluated cache attribute has no indexed values")?;
    let domain = attribute["domain"]
        .as_str()
        .ok_or("evaluated cache attribute has no domain")?;
    let mut ordered = Vec::new();
    match domain {
        "point" => {
            for vertex in &mesh.vertices {
                ordered.push(values[&format!("v{}", vertex.id)].clone());
            }
        }
        "edge" => {
            for edge in &mesh.edges {
                ordered.push(values[&format!("e{}", edge.id)].clone());
            }
        }
        "face" => {
            for face in &mesh.faces {
                ordered.push(values[&format!("f{}", face.id)].clone());
            }
        }
        "corner" => {
            for face in &mesh.faces {
                ordered.extend(
                    values[&format!("f{}", face.id)]
                        .as_array()
                        .ok_or("evaluated corner attribute face row is missing")?
                        .iter()
                        .cloned(),
                );
            }
        }
        _ => return Err("evaluated cache attribute domain is unsupported".into()),
    }
    Ok(Value::Array(ordered))
}

/// The Potter attribute that stores an Alembic cache attribute. Velocity is kept as `alembic_velocity`.
fn potter_attribute_name(name: &str) -> &str {
    match name {
        "velocity" => "alembic_velocity",
        other => other,
    }
}

fn assert_oracle_attribute(
    mesh: &potter_core::geom::Mesh,
    name: &str,
    expected: &Value,
) -> Result<(), Box<dyn Error>> {
    let actual = if name == "CornerCol" {
        let layer_names = mesh
            .attributes
            .get("color_layers")
            .and_then(|registry| registry.get("layers"))
            .and_then(Value::as_array)
            .ok_or("canonical color-layer registry is missing")?;
        let stored_name = layer_names
            .iter()
            .filter_map(Value::as_str)
            .find(|stored_name| *stored_name == name)
            .ok_or_else(|| format!("canonical color layer {name} is not registered"))?;
        mesh.attributes.get(stored_name)
    } else {
        mesh.attributes.get(potter_attribute_name(name))
    }
    .ok_or_else(|| format!("evaluated cache attribute {name} is missing"))?;
    let expected_domain = match expected["domain"].as_str() {
        Some("POINT") => "point",
        Some("EDGE") => "edge",
        Some("FACE") => "face",
        Some("CORNER") => "corner",
        _ => return Err(format!("oracle attribute {name} has no supported domain").into()),
    };
    assert_eq!(actual["domain"], expected_domain, "attribute {name} domain");
    if name == "custom_normal" {
        assert_eq!(actual["type"], "normal_code", "attribute {name} encoding");
        let actual_codes = oracle_actual_attribute_values(mesh, actual)?;
        let actual_codes = actual_codes
            .as_array()
            .ok_or("custom_normal codes are not an array")?;
        let expected_codes = expected["values"]
            .as_array()
            .ok_or("Blender custom_normal codes are not an array")?;
        if actual_codes.len() != expected_codes.len() {
            return Err(format!(
                "custom_normal has {} corners, Blender has {}",
                actual_codes.len(),
                expected_codes.len()
            )
            .into());
        }
        for (corner, (actual_code, expected_code)) in
            actual_codes.iter().zip(expected_codes).enumerate()
        {
            let same = |value: &Value| {
                value
                    .as_array()
                    .and_then(|pair| Some([pair.first()?.as_f64()?, pair.get(1)?.as_f64()?]))
            };
            if same(actual_code) != same(expected_code) {
                return Err(format!(
                    "custom_normal corner {corner}: Potter stores {actual_code}, Blender stores {expected_code}"
                )
                .into());
            }
        }
    } else {
        let expected_type = match expected["type"].as_str() {
            Some("BOOLEAN") => "bool",
            Some("INT") => "int",
            Some("FLOAT_VECTOR") => "float3",
            Some("BYTE_COLOR") => "color",
            _ => return Err(format!("oracle attribute {name} has no supported type").into()),
        };
        assert_eq!(actual["type"], expected_type, "attribute {name} type");
        assert_numeric_json_close(
            &oracle_actual_attribute_values(mesh, actual)?,
            &expected["values"],
            1.0e-6,
        );
    }
    Ok(())
}

fn is_oracle_uv_attribute(attribute: &Value) -> bool {
    attribute["domain"] == "CORNER" && attribute["type"] == "FLOAT2"
}

/// A UV pair within the tolerance of Blender's stored value (Potter stores V flipped, as the caller passes it).
fn uv_close(actual: &Value, u: f64, v: f64) -> bool {
    actual.as_array().is_some_and(|pair| {
        pair.len() == 2
            && [u, v].iter().zip(pair).all(|(expected, actual)| {
                actual
                    .as_f64()
                    .is_some_and(|actual| (actual - expected).abs() <= 1.0e-6)
            })
    })
}

/// The UV layers of an evaluated mesh: names in store order, and the active and render layers.
fn uv_layer_summary(mesh: &potter_core::geom::Mesh) -> String {
    let Some(store) = mesh.attributes.get("uv_layers") else {
        return "no UV store".to_owned();
    };
    let names = store["layers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|layer| layer["name"].as_str())
        .collect::<Vec<_>>();
    format!(
        "layers {names:?}, active {}, render {}",
        store["active"], store["render"]
    )
}

/// The UV layers of the cache file that the oracle case reads, with the layer a read without polygons
/// leaves unwritten. Reports the reason instead of failing, since it only annotates another failure.
fn uv_cache_summary(fixture_dir: &Path, oracle_case: &Value) -> String {
    let archive = oracle_case["cache_file"]["filepath"]
        .as_str()
        .and_then(|path| Path::new(path).file_name())
        .map(|name| fixture_dir.join(name));
    let (Some(archive), Some(object_path)) =
        (archive, oracle_case["modifier"]["object_path"].as_str())
    else {
        return "cache unavailable: no archive or object path".to_owned();
    };
    let summary = fs::read(&archive)
        .map_err(|error| error.to_string())
        .and_then(|bytes| {
            potter_core::exchange::alembic::mesh_cache_uv_layers(&bytes, object_path)
                .map_err(|error| error.to_string())
        });
    match summary {
        Ok((names, primary)) => {
            format!("cache layers {names:?}, unwritten without polygons {primary:?}")
        }
        Err(error) => format!("cache unavailable: {error}"),
    }
}

fn assert_oracle_uv_layers(
    mesh: &potter_core::geom::Mesh,
    expected: &Value,
) -> Result<(), Box<dyn Error>> {
    let expected_attributes = expected["attributes"]
        .as_object()
        .ok_or("Blender oracle attributes are missing")?;
    let expected_layers = expected_attributes
        .iter()
        .filter(|(_, attribute)| is_oracle_uv_attribute(attribute))
        .collect::<Vec<_>>();
    let actual_layers = mesh
        .attributes
        .get("uv_layers")
        .and_then(|registry| registry.get("layers"))
        .and_then(Value::as_array);
    let mut actual_names = actual_layers
        .into_iter()
        .flatten()
        .filter_map(|layer| layer["name"].as_str())
        .collect::<Vec<_>>();
    let mut expected_names = expected_layers
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    actual_names.sort_unstable();
    expected_names.sort_unstable();
    assert_eq!(actual_names, expected_names, "canonical UV layer names");

    let actual_layers: &[Value] = actual_layers.map_or(&[], Vec::as_slice);
    let expected_positions: Vec<[f64; 3]> = serde_json::from_value(expected["positions"].clone())?;
    let actual_positions = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect::<std::collections::BTreeMap<_, _>>();
    let expected_faces: Vec<Vec<usize>> = serde_json::from_value(expected["faces"].clone())?;
    for (name, oracle_layer) in expected_layers {
        let layer = actual_layers
            .iter()
            .find(|layer| layer["name"].as_str() == Some(name.as_str()))
            .ok_or_else(|| format!("canonical UV layer {name} is missing"))?;
        let stored_faces = layer["faces"]
            .as_array()
            .ok_or_else(|| format!("canonical UV layer {name} face rows are missing"))?;
        let oracle_values = oracle_layer["values"]
            .as_array()
            .ok_or_else(|| format!("Blender oracle UV layer {name} values are missing"))?;
        let mut corner_offset = 0;
        for expected_face in &expected_faces {
            let corner_count = expected_face.len();
            let oracle_row = oracle_values
                .get(corner_offset..corner_offset + corner_count)
                .ok_or_else(|| format!("Blender oracle UV layer {name} row is truncated"))?;
            let expected_face_positions = expected_face
                .iter()
                .map(|index| expected_positions[*index])
                .collect::<Vec<_>>();
            let actual_face =
                mesh.faces
                    .iter()
                    .find(|face| {
                        face.vertices.len() == corner_count
                            && face.vertices.iter().all(|id| {
                                let actual = actual_positions[id];
                                expected_face_positions.iter().any(|expected| {
                                    actual.to_array().iter().zip(expected).all(
                                        |(actual, expected)| (actual - expected).abs() <= 1.0e-6,
                                    )
                                })
                            })
                    })
                    .ok_or_else(|| format!("evaluated face for UV layer {name} is missing"))?;
            let actual_row = stored_faces
                .iter()
                .find(|entry| entry["face_id"].as_u64() == Some(u64::from(actual_face.id)))
                .and_then(|entry| entry["uv"].as_array())
                .ok_or_else(|| format!("canonical UV layer {name} face values are missing"))?;
            assert_eq!(actual_row.len(), corner_count);
            for (oracle_uv, oracle_position) in oracle_row
                .iter()
                .rev()
                .zip(expected_face_positions.iter().rev())
            {
                let actual_corner = actual_face
                    .vertices
                    .iter()
                    .position(|id| {
                        actual_positions[id]
                            .to_array()
                            .iter()
                            .zip(oracle_position)
                            .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-6)
                    })
                    .ok_or_else(|| format!("UV layer {name} corner position is missing"))?;
                let u = oracle_uv[0]
                    .as_f64()
                    .ok_or_else(|| format!("oracle UV layer {name} U is not numeric"))?;
                let v = oracle_uv[1]
                    .as_f64()
                    .ok_or_else(|| format!("oracle UV layer {name} V is not numeric"))?;
                if !uv_close(&actual_row[actual_corner], u, 1.0 - v) {
                    return Err(format!(
                        "UV layer {name} on face {}: actual {} expected [{u}, {}]",
                        actual_face.id,
                        actual_row[actual_corner],
                        1.0 - v
                    )
                    .into());
                }
            }
            corner_offset += corner_count;
        }
        assert_eq!(corner_offset, oracle_values.len());
    }
    Ok(())
}

fn flat_corner_normals(mesh: &potter_core::geom::Mesh) -> Vec<[f64; 3]> {
    let positions = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect::<std::collections::BTreeMap<_, _>>();
    mesh.faces
        .iter()
        .flat_map(|face| {
            let points = face
                .vertices
                .iter()
                .map(|id| positions[id])
                .collect::<Vec<_>>();
            let normal = if points.len() == 4 {
                (points[0] - points[2]).cross(points[1] - points[3])
            } else {
                points
                    .iter()
                    .zip(points.iter().cycle().skip(1))
                    .fold(glam::DVec3::ZERO, |normal, (first, second)| {
                        normal + first.cross(*second)
                    })
            }
            .normalize_or_zero()
            .to_array();
            std::iter::repeat_n(normal, face.vertices.len())
        })
        .collect()
}

fn assert_oracle_builtin_attribute(
    mesh: &potter_core::geom::Mesh,
    name: &str,
    expected: &Value,
) -> Result<(), Box<dyn Error>> {
    match name {
        "sharp_edge" => {
            assert_eq!(expected["domain"], "EDGE");
            assert_eq!(expected["type"], "BOOLEAN");
            let sharp_values = mesh
                .attributes
                .get("sharp")
                .and_then(|attribute| attribute.get("values"))
                .and_then(Value::as_object);
            let actual = mesh
                .edges
                .iter()
                .map(|edge| {
                    sharp_values
                        .and_then(|values| values.get(&format!("e{}", edge.id)))
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>();
            let expected_values: Vec<bool> = serde_json::from_value(expected["values"].clone())?;
            assert_eq!(actual, expected_values, "canonical edge sharp flags");
        }
        "sharp_face" => {
            assert_eq!(expected["domain"], "FACE");
            assert_eq!(expected["type"], "BOOLEAN");
            let smooth_values = mesh
                .attributes
                .get("shade_smooth")
                .and_then(|attribute| attribute.get("values"))
                .and_then(Value::as_object);
            let actual = mesh
                .faces
                .iter()
                .map(|face| {
                    !smooth_values
                        .and_then(|values| values.get(&format!("f{}", face.id)))
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>();
            let expected_values: Vec<bool> = serde_json::from_value(expected["values"].clone())?;
            assert_eq!(actual, expected_values, "canonical sharp_face semantics");
        }
        "material_index" => {
            assert_eq!(expected["domain"], "FACE");
            assert_eq!(expected["type"], "INT");
            let actual = mesh
                .faces
                .iter()
                .map(|face| face.material_index)
                .collect::<Vec<_>>();
            let expected_values: Vec<u32> = serde_json::from_value(expected["values"].clone())?;
            assert_eq!(actual, expected_values, "canonical face material indices");
        }
        _ => return Err(format!("oracle built-in attribute {name} is unsupported").into()),
    }
    Ok(())
}

fn assert_oracle_mesh(
    mesh: &potter_core::geom::Mesh,
    expected: &Value,
) -> Result<(), Box<dyn Error>> {
    let expected_positions: Vec<[f64; 3]> = serde_json::from_value(expected["positions"].clone())?;
    assert_positions_close(
        &mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co.to_array())
            .collect::<Vec<_>>(),
        &expected_positions,
        1.0e-6,
    );
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    let actual_faces = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| vertex_indices[id])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual_faces,
        serde_json::from_value::<Vec<Vec<usize>>>(expected["faces"].clone())?,
        "face count, winding, or corner order differs at frame {}",
        expected["frame"]
    );
    let smooth_values = mesh
        .attributes
        .get("shade_smooth")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let actual_smooth_faces = mesh
        .faces
        .iter()
        .map(|face| {
            smooth_values
                .and_then(|values| values.get(&format!("f{}", face.id)))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    let expected_smooth_faces: Vec<bool> =
        serde_json::from_value(expected["smooth_faces"].clone())?;
    assert_eq!(actual_smooth_faces, expected_smooth_faces);
    let has_custom_normals = expected["has_custom_normals"]
        .as_bool()
        .ok_or("Blender oracle custom-normal flag is missing")?;
    if mesh.attributes.contains_key("custom_normal") != has_custom_normals {
        return Err(format!("custom_normal presence must be {has_custom_normals}").into());
    }

    let expected_attributes = expected["attributes"]
        .as_object()
        .ok_or("Blender oracle attributes are missing")?;
    let mut actual_attribute_names = mesh
        .attributes
        .keys()
        .filter(|name| {
            !name.starts_with('.')
                && !matches!(
                    name.as_str(),
                    "position" | "color_layers" | "uv_layers" | "shade_smooth" | "sharp"
                )
                && !expected_attributes
                    .get(name.as_str())
                    .is_some_and(is_oracle_uv_attribute)
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut expected_attribute_names = expected_attributes
        .iter()
        .filter(|(name, attribute)| {
            !is_oracle_uv_attribute(attribute)
                && !matches!(
                    name.as_str(),
                    "material_index" | "sharp_edge" | "sharp_face"
                )
        })
        .map(|(name, _)| potter_attribute_name(name).to_owned())
        .collect::<Vec<_>>();
    actual_attribute_names.sort();
    expected_attribute_names.sort();
    assert_eq!(actual_attribute_names, expected_attribute_names);
    for (name, attribute) in expected_attributes {
        if is_oracle_uv_attribute(attribute) {
            continue;
        }
        if matches!(
            name.as_str(),
            "material_index" | "sharp_edge" | "sharp_face"
        ) {
            assert_oracle_builtin_attribute(mesh, name, attribute)?;
        } else {
            assert_oracle_attribute(mesh, name, attribute)?;
        }
    }
    assert_oracle_uv_layers(mesh, expected)?;
    if !has_custom_normals {
        assert_numeric_json_close(
            &serde_json::to_value(flat_corner_normals(mesh))?,
            &expected["corner_normals"],
            1.0e-5,
        );
    }
    Ok(())
}

fn add_oracle_cache(
    scene: &Path,
    fixture_dir: &Path,
    oracle_case: &Value,
) -> Result<(), Box<dyn Error>> {
    let cache_file = &oracle_case["cache_file"];
    if cache_file["is_sequence"] == true {
        add_sequence_resource(scene, fixture_dir, cache_file)?;
    } else {
        let archive = fixture_dir.join(
            Path::new(
                cache_file["filepath"]
                    .as_str()
                    .ok_or("Blender oracle cache filepath is missing")?,
            )
            .file_name()
            .ok_or("Blender oracle cache filename is missing")?,
        );
        apply(
            scene,
            &json!([{"op":"resource.pack","id":"oracle_cache","uri":archive,"kind":"alembic"}]),
        )?;
        let scene_path = scene.join("scene.json");
        let mut document: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
        document["resources"]["oracle_cache"]["cache_file"] = cache_file.clone();
        fs::write(&scene_path, serde_json::to_vec_pretty(&document)?)?;
    }
    apply(
        scene,
        &json!([{
            "op":"modifier.create",
            "target":{"id":"body"},
            "id":"oracle_cache_modifier",
            "type":"mesh_sequence_cache",
            "params":oracle_cache_params(oracle_case)?
        }]),
    )?;
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(
        document["resources"]["oracle_cache"]["cache_file"], *cache_file,
        "CacheFile settings were not persisted for {}",
        oracle_case["id"]
    );
    let params = document["nodes"]["body"]["modifiers"]
        .as_array()
        .and_then(|modifiers| {
            modifiers
                .iter()
                .find(|modifier| modifier["id"] == "oracle_cache_modifier")
        })
        .and_then(|modifier| modifier.get("params"))
        .ok_or("persisted oracle cache modifier is missing")?;
    let expected_params = oracle_cache_params(oracle_case)?;
    for field in [
        "resource",
        "object_path",
        "read_data",
        "use_vertex_interpolation",
        "velocity_scale",
        "velocity_name",
        "velocity_unit",
        "frame_offset",
        "override_frame",
        "is_sequence",
    ] {
        assert_eq!(params[field], expected_params[field], "persisted {field}");
    }
    Ok(())
}

fn assert_inspect_matches_oracle(
    scene: &Path,
    expected: &Value,
    missing_sequence: bool,
) -> Result<(), Box<dyn Error>> {
    let frame = expected["frame"]
        .as_f64()
        .ok_or("Blender oracle frame is not numeric")?;
    let output = run_ok(
        pot()
            .args(["inspect"])
            .arg(scene)
            .args(["--id", "body", "--frame"])
            .arg(frame.to_string())
            .arg("--json"),
    )?;
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    let item = envelope["result"]["items"]
        .as_array()
        .and_then(|items| items.first())
        .ok_or("CLI inspect omitted the oracle receiver")?;
    assert!(item.get("error").is_none_or(Value::is_null));
    let geometry = item
        .get("evaluated_geometry")
        .ok_or("CLI inspect omitted evaluated geometry")?;
    assert_numeric_json_close(&geometry["positions"], &expected["positions"], 1.0e-6);
    assert_eq!(geometry["faces"], expected["faces"]);
    let expected_warnings = if missing_sequence {
        json!([{
            "code":"DEPENDENCY_MISSING",
            "message":"Alembic sequence frame is unavailable; Mesh Sequence Cache was skipped",
            "details":{
                "modifier_id":"oracle_cache_modifier",
                "node_id":"body",
                "resource_id":"oracle_cache",
                "frame":5,
                "reason":"frame is outside the registered source sequence"
            }
        }])
    } else {
        json!([])
    };
    assert_eq!(envelope["result"]["warnings"], expected_warnings);
    Ok(())
}

fn assert_blender_corner_normals(
    blender: &Path,
    blend: &Path,
    directory: &Path,
    oracle_case: &Value,
) -> Result<(), Box<dyn Error>> {
    let frames = oracle_case["frames"]
        .as_array()
        .ok_or("sequence oracle frames are missing")?
        .iter()
        .map(|sample| {
            let frame = sample
                .get("frame")
                .ok_or("sequence oracle frame is missing")?;
            Ok(json!({"frame":frame,"key":frame.to_string()}))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let output_path = directory.join("sequence_blender_corner_normals.json");
    let output_path_text = output_path
        .to_str()
        .ok_or("Blender normal output path is not UTF-8")?;
    let script = format!(
        r#"import bpy, json
samples = {frames}
output_path = {output_path}
obj = bpy.data.objects.get("OracleReceiver")
if obj is None:
    raise RuntimeError("OracleReceiver is missing after .blend reopen")
result = {{}}
scene = bpy.context.scene
for sample in samples:
    frame = float(sample["frame"])
    scene.frame_set(frame)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        result[sample["key"]] = [
            [float(component) for component in normal.vector]
            for normal in mesh.corner_normals
        ]
    finally:
        evaluated.to_mesh_clear()
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"#,
        frames = serde_json::to_string(&frames)?,
        output_path = serde_json::to_string(output_path_text)?,
    );
    let output = Command::new(blender)
        .args(["--background"])
        .arg(blend)
        .args(["--python-expr", &script])
        .output()?;
    assert!(
        output.status.success(),
        "Blender corner-normal readback failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Value = serde_json::from_slice(&fs::read(output_path)?)?;
    let expected_frames = oracle_case["frames"]
        .as_array()
        .ok_or("sequence oracle frames are missing")?;
    for expected in expected_frames {
        let key = expected["frame"].to_string();
        let actual_normals = actual
            .get(key.as_str())
            .ok_or("Blender corner-normal readback omitted a frame")?;
        assert_numeric_json_close(actual_normals, &expected["corner_normals"], 1.0e-5);
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "the sequence round-trip verifies resource metadata and reopened evaluation together"
)]
fn round_trip_sequence_cache_blend(
    scene: &Path,
    directory: &Path,
    blender: &Path,
    oracle_case: &Value,
) -> Result<(), Box<dyn Error>> {
    let blender_path = blender
        .to_str()
        .ok_or("Blender executable path is not UTF-8")?;
    let blend = directory.join("sequence_cache_roundtrip.blend");
    run_ok(
        pot()
            .args(["export"])
            .arg(scene)
            .args(["--format", "blend", "--out"])
            .arg(&blend)
            .args(["--blender", blender_path, "--json"]),
    )?;
    assert_blender_corner_normals(blender, &blend, directory, oracle_case)?;

    let reopened = directory.join("sequence_cache_reopened.pot");
    init(&reopened)?;
    let imported = run_ok(
        pot()
            .args(["import"])
            .arg(&reopened)
            .args(["--file"])
            .arg(&blend)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
                blender_path,
                "--json",
            ]),
    )?;
    let import_result: Value = serde_json::from_slice(&imported.stdout)?;
    assert_eq!(import_result["result"]["losses"], json!([]));
    let receiver_id = import_result["result"]["id_mappings"]["Object:OracleReceiver"]
        .as_str()
        .ok_or("reopened Blender cache receiver ID mapping is missing")?;
    let project = potter_core::store::Project::open(&reopened)?;
    let node = project
        .doc()
        .nodes
        .get(&potter_core::model::Id::new(receiver_id.to_owned())?)
        .ok_or("reopened Blender cache receiver is missing")?;
    let modifier = node
        .modifiers
        .iter()
        .find(|modifier| modifier.modifier_type == "mesh_sequence_cache")
        .ok_or("reopened Blender CacheFile modifier is missing")?;
    let expected_params = oracle_cache_params(oracle_case)?;
    for field in [
        "object_path",
        "read_data",
        "use_vertex_interpolation",
        "velocity_scale",
        "velocity_name",
        "velocity_unit",
        "frame_offset",
        "override_frame",
        "is_sequence",
    ] {
        assert_eq!(
            modifier.params[field], expected_params[field],
            "reopened {field}"
        );
    }
    let resource_id = modifier.params["resource"]
        .as_str()
        .ok_or("reopened Blender cache resource ID is missing")?;
    let resource = project
        .doc()
        .resources
        .get(&potter_core::model::Id::new(resource_id.to_owned())?)
        .ok_or("reopened Blender cache resource is missing")?;
    assert_eq!(resource["kind"], "alembic_cache");
    let reopened_settings = resource
        .get("cache_file")
        .ok_or("reopened Blender CacheFile settings are missing")?;
    for field in [
        "is_sequence",
        "override_frame",
        "frame",
        "frame_offset",
        "forward_axis",
        "up_axis",
        "scale",
        "velocity_name",
        "velocity_unit",
    ] {
        assert_eq!(
            reopened_settings[field], oracle_case["cache_file"][field],
            "reopened CacheFile {field}"
        );
    }
    let source_cache_path = oracle_case["cache_file"]["filepath"]
        .as_str()
        .ok_or("oracle CacheFile path is missing")?;
    let resource_uri = resource["uri"]
        .as_str()
        .ok_or("reopened CacheFile resource URI is missing")?;
    let resource_name = Path::new(resource_uri)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("reopened CacheFile resource filename is missing")?;
    let source_name = Path::new(source_cache_path)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("oracle CacheFile filename is missing")?;
    assert!(
        resource_name == source_name || resource_name == "seq.####.abc",
        "reopened CacheFile URI does not name its sequence source: {resource_uri}"
    );
    let sequence_frames = resource["sequence_frames"]
        .as_object()
        .ok_or("reopened Blender CacheFile sequence manifest is missing")?;
    assert_eq!(sequence_frames.len(), 4);
    for frame in 1..=4 {
        let frame_path = sequence_frames[&frame.to_string()]["uri"]
            .as_str()
            .ok_or("reopened sequence frame path is missing")?;
        let expected_filename = format!("seq.{frame:04}.abc");
        assert_eq!(
            Path::new(frame_path)
                .file_name()
                .and_then(|name| name.to_str()),
            Some(expected_filename.as_str())
        );
        assert!(
            Path::new(frame_path).is_file(),
            "reopened sequence frame is missing"
        );
    }
    drop(project);

    for expected in oracle_case["frames"]
        .as_array()
        .ok_or("sequence oracle frames are missing")?
    {
        let frame = expected["frame"]
            .as_f64()
            .ok_or("sequence oracle frame is not numeric")?;
        assert_oracle_mesh(&evaluated_mesh(&reopened, frame)?, expected)?;
        let inspected = run_ok(
            pot()
                .args(["inspect"])
                .arg(&reopened)
                .args(["--id", receiver_id, "--frame"])
                .arg(frame.to_string())
                .arg("--json"),
        )?;
        let response: Value = serde_json::from_slice(&inspected.stdout)?;
        let geometry = &response["result"]["items"][0]["evaluated_geometry"];
        assert_numeric_json_close(&geometry["positions"], &expected["positions"], 1.0e-6);
        assert_eq!(geometry["faces"], expected["faces"]);
        assert_eq!(response["result"]["warnings"], json!([]));
    }
    Ok(())
}
#[test]
fn blender_oracle_alembic_cache_matches_all_cases_and_mesh_attributes() -> Result<(), Box<dyn Error>>
{
    let temporary = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/alembic_cache");
    let oracle: Value = serde_json::from_str(include_str!("fixtures/alembic_cache/oracle.json"))?;
    let cases = oracle["cases"]
        .as_array()
        .ok_or("Blender Alembic oracle cases are missing")?;
    let mut case_ids = cases
        .iter()
        .map(|case| case["id"].as_str().ok_or("Blender oracle case has no id"))
        .collect::<Result<Vec<_>, &str>>()?;
    case_ids.sort_unstable();
    let mut expected_case_ids = [
        "axis_settings_mesh_data_unchanged",
        "default_same_topology",
        "frame_offset_fractional",
        "frame_offset_positive",
        "is_sequence_missing_file",
        "is_sequence_per_frame_files",
        "override_frame_fractional_interpolated",
        "override_frame_fractional_no_interpolation",
        "sequence_same_size_reordered_connectivity",
        "nonsequence_animated_same_size_reordered_connectivity",
        "read_attributes_only",
        "read_no_color",
        "read_poly_only",
        "read_uv_only",
        "read_vert_only",
        "scale_two_ignored",
        "sequence_offset_ignored",
        "sequence_override_frame",
        "topology_changed_cube_input",
        "topology_changed_cube_input_vert_only",
        "topology_face_change_full_read",
        "topology_face_change_vert_only",
        "velocity_name_empty",
        "velocity_name_missing",
        "velocity_scale_zero",
        "velocity_unit_frame",
        "velocity_unit_second",
    ];
    expected_case_ids.sort_unstable();
    assert_eq!(case_ids, expected_case_ids.to_vec());
    for oracle_case in cases {
        assert_eq!(oracle_case["modifier_error"], Value::Null);
        let case_id = oracle_case["id"]
            .as_str()
            .ok_or("Blender oracle case has no id")?;
        let target = temporary.path().join(format!("{case_id}.pot"));
        init(&target)?;
        apply(
            &target,
            &json!([{"op":"node.create","id":"body","name":"OracleReceiver","kind":"box","params":{}}]),
        )?;
        let (positions, faces) = oracle_receiver_geometry(case_id);
        if case_id == "nonsequence_animated_same_size_reordered_connectivity" {
            assert_eq!(oracle_case["cache_file"]["is_sequence"], false);
            assert_eq!(oracle_case["modifier"]["read_data"], json!(["VERT"]));
        }
        set_receiver_mesh(
            &target,
            &positions,
            &faces,
            case_id == "sequence_same_size_reordered_connectivity"
                || case_id == "nonsequence_animated_same_size_reordered_connectivity",
        )?;
        let receiver_uv = uv_layer_summary(&evaluated_mesh(&target, 0.0)?);
        add_oracle_cache(&target, &fixture_dir, oracle_case)?;
        let frames = oracle_case["frames"]
            .as_array()
            .ok_or("Blender oracle frames are missing")?;
        for expected in frames {
            assert!(expected.get("error").is_none_or(Value::is_null));
            let missing_sequence = case_id == "is_sequence_missing_file";
            if case_id == "nonsequence_animated_same_size_reordered_connectivity" {
                let uv_map = &expected["attributes"]["UVMap"];
                assert_eq!(uv_map["domain"], "CORNER");
                assert_eq!(uv_map["type"], "FLOAT2");
                let values: Vec<[f64; 2]> = serde_json::from_value(uv_map["values"].clone())?;
                assert_eq!(values.len(), 11);
                let input_uv: Vec<[f64; 2]> = serde_json::from_value(json!([
                    [20.0, 10.0],
                    [20.0, 11.0],
                    [20.0, 12.0],
                    [20.0, 13.0],
                    [21.0, 10.0],
                    [21.0, 11.0],
                    [21.0, 12.0],
                    [22.0, 10.0],
                    [22.0, 11.0],
                    [22.0, 12.0],
                    [22.0, 13.0]
                ]))?;
                if expected["frame"].as_i64() == Some(1) {
                    assert_eq!(values, input_uv);
                } else {
                    assert_ne!(values, input_uv);
                }
            }
            let actual = evaluated_mesh(
                &target,
                expected["frame"]
                    .as_f64()
                    .ok_or("Blender oracle frame is not numeric")?,
            )?;
            assert_oracle_mesh(&actual, expected).map_err(|error| {
                format!(
                    "{case_id} frame {}: {error}\n  receiver before modifier: {receiver_uv}\n  evaluated: {}\n  {}",
                    expected["frame"],
                    uv_layer_summary(&actual),
                    uv_cache_summary(&fixture_dir, oracle_case),
                )
            })?;
            assert_inspect_matches_oracle(&target, expected, missing_sequence)?;
        }
        if case_id == "is_sequence_per_frame_files"
            && let Some(blender) = blender_executable()
        {
            round_trip_sequence_cache_blend(&target, temporary.path(), &blender, oracle_case)?;
        }
        if case_id == "default_same_topology" {
            let exported = temporary.path().join("evaluated_cache.abc");
            run_ok(
                pot()
                    .args([
                        "export",
                        target.to_str().ok_or("oracle scene path is not UTF-8")?,
                        "--format",
                        "alembic",
                        "--frame",
                        "2",
                        "--out",
                    ])
                    .arg(&exported)
                    .args(["--overwrite", "--allow-lossy", "--json"]),
            )?;
            assert!(exported.is_file(), "CLI Alembic export was not created");
            let preview_dir = temporary.path().join("cache_preview");
            let preview_output = run_ok(
                pot()
                    .args(["preview"])
                    .arg(&target)
                    .args([
                        "--views", "iso", "--size", "64", "--mode", "normal", "--out",
                    ])
                    .arg(&preview_dir)
                    .args(["--overwrite", "--json"]),
            )?;
            let preview: Value = serde_json::from_slice(&preview_output.stdout)?;
            assert_eq!(preview["result"]["warnings"], json!([]));
            assert!(
                preview_dir.join("iso.png").is_file(),
                "CLI normal preview was not created"
            );
        }
    }
    Ok(())
}

#[test]
fn blender_file_sequence_uses_frame_files_and_validates_each_hash() -> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let target = temporary.path().join("sequence.pot");
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/alembic_cache");
    let source_archive = fixture_dir.join("animated.abc");
    init(&target)?;
    run_ok(
        pot()
            .arg("import")
            .arg(&target)
            .arg("--file")
            .arg(&source_archive)
            .args([
                "--format",
                "alembic",
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--allow-lossy",
                "--json",
            ]),
    )?;
    let project = potter_core::store::Project::open(&target)?;
    let target_id = project
        .doc()
        .nodes
        .iter()
        .find_map(|(id, node)| (node.name == "Animated").then(|| id.clone()))
        .ok_or("oracle Alembic mesh node is missing")?;
    drop(project);

    let mut sequence_frames = serde_json::Map::new();
    for frame in 1..=4 {
        let source = fixture_dir.join(format!("seq.{frame:04}.abc"));
        let destination = temporary.path().join(format!("seq.{frame:04}.abc"));
        fs::copy(&source, &destination)?;
        let digest = potter_core::hash::sha256(&fs::read(&destination)?);
        sequence_frames.insert(
            frame.to_string(),
            json!({
                "uri":destination,
                "hash":digest,
                "kind":"alembic",
                "packed":false,
            }),
        );
    }
    fs::copy(&source_archive, temporary.path().join("seq.0005.abc"))?;
    let sequence_path = temporary.path().join("seq.####.abc");
    let mut document: Value = serde_json::from_slice(&fs::read(target.join("scene.json"))?)?;
    document["resources"]["oracle_sequence"] = json!({
        "uri": sequence_path,
        "kind": "alembic",
        "packed": false,
        "sequence_start": 1,
        "sequence_end": 4,
        "sequence_frames": sequence_frames,
    });
    fs::write(
        target.join("scene.json"),
        serde_json::to_vec_pretty(&document)?,
    )?;
    let receiver_positions = mesh_positions(&target, 5.0)?;
    apply(
        &target,
        &json!([{
            "op":"modifier.create",
            "target":{"id":target_id},
            "id":"oracle_sequence_modifier",
            "type":"mesh_sequence_cache",
            "params":{
                "resource":"oracle_sequence",
                "object_path":"/Animated/Animated",
                "is_sequence":true,
                "frame_offset":10.0
            }
        }]),
    )?;
    let oracle: Value = serde_json::from_str(include_str!("fixtures/alembic_cache/oracle.json"))?;
    let sequence_case = oracle["cases"]
        .as_array()
        .and_then(|cases| {
            cases
                .iter()
                .find(|case| case["id"] == "is_sequence_per_frame_files")
        })
        .ok_or("Blender sequence oracle case is missing")?;
    for frame in [1.0, 2.0, 3.0, 4.0] {
        let expected = sequence_case["frames"]
            .as_array()
            .and_then(|frames| {
                frames
                    .iter()
                    .find(|sample| sample["frame"].as_f64() == Some(frame))
            })
            .ok_or("Blender sequence frame is missing")?;
        assert_positions_close(
            &mesh_positions(&target, frame)?,
            &serde_json::from_value::<Vec<[f64; 3]>>(expected["positions"].clone())?,
            1.0e-6,
        );
    }
    // Blender leaves the receiver unchanged on a missing frame (oracle case is_sequence_missing_file
    // has no modifier error), so the evaluated mesh equals the receiver captured before the modifier.
    assert_positions_close(&mesh_positions(&target, 5.0)?, &receiver_positions, 1.0e-6);
    // A modifier warning carries no node_id, so only an unfiltered inspect reports it.
    let output = run_ok(
        pot()
            .args(["inspect"])
            .arg(&target)
            .args(["--frame", "5", "--json"]),
    )?;
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert!(
        response["result"]["warnings"]
            .as_array()
            .is_some_and(|warnings| warnings.iter().any(|warning| {
                warning["details"]["modifier_id"] == "oracle_sequence_modifier"
                    && warning["details"]["frame"] == 5
            })),
        "missing sequence frame did not report a warning"
    );

    let changed_path = temporary.path().join("seq.0002.abc");
    let mut changed = fs::read(&changed_path)?;
    let last = changed.len() - 1;
    changed[last] ^= 1;
    fs::write(changed_path, changed)?;
    let project = potter_core::store::Project::open(&target)?;
    let Err(error) = potter_core::eval::Snapshot::evaluate_with_cache(
        project.doc(),
        &potter_core::eval::EvaluationContext {
            frame: Some(2.0),
            ..potter_core::eval::EvaluationContext::default()
        },
        Some(project.path()),
        Some(project.cache_directory()),
    ) else {
        return Err("changed sequence frame must be rejected".into());
    };
    assert_eq!(error.code, potter_core::error::ErrorCode::AssetChanged);
    Ok(())
}

#[test]
fn resource_pack_sequence_is_self_contained_after_reopen() -> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let target = temporary.path().join("packed_sequence.pot");
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/alembic_cache");
    let source_archive = fixture_dir.join("animated.abc");
    init(&target)?;
    run_ok(
        pot()
            .arg("import")
            .arg(&target)
            .arg("--file")
            .arg(&source_archive)
            .args([
                "--format",
                "alembic",
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--allow-lossy",
                "--json",
            ]),
    )?;
    let project = potter_core::store::Project::open(&target)?;
    let target_id = project
        .doc()
        .nodes
        .iter()
        .find_map(|(id, node)| (node.name == "Animated").then(|| id.clone()))
        .ok_or("oracle Alembic mesh node is missing")?;
    drop(project);
    let sequence_pattern = fixture_dir.join("seq.####.abc");
    apply(
        &target,
        &json!([
            {
                "op":"resource.pack",
                "id":"packed_sequence",
                "uri":sequence_pattern,
                "kind":"alembic",
                "sequence":{"start":1,"end":4}
            },
            {
                "op":"modifier.create",
                "target":{"id":target_id},
                "id":"packed_sequence_modifier",
                "type":"mesh_sequence_cache",
                "params":{
                    "resource":"packed_sequence",
                    "object_path":"/Animated/Animated",
                    "is_sequence":true
                }
            }
        ]),
    )?;
    let oracle: Value = serde_json::from_str(include_str!("fixtures/alembic_cache/oracle.json"))?;
    let sequence_case = oracle["cases"]
        .as_array()
        .and_then(|cases| {
            cases
                .iter()
                .find(|case| case["id"] == "is_sequence_per_frame_files")
        })
        .ok_or("Blender sequence oracle case is missing")?;
    for frame in [1.0, 2.0, 3.0, 4.0] {
        let expected = sequence_case["frames"]
            .as_array()
            .and_then(|frames| {
                frames
                    .iter()
                    .find(|sample| sample["frame"].as_f64() == Some(frame))
            })
            .ok_or("Blender sequence frame is missing")?;
        assert_positions_close(
            &mesh_positions(&target, frame)?,
            &serde_json::from_value::<Vec<[f64; 3]>>(expected["positions"].clone())?,
            1.0e-6,
        );
    }
    let project = potter_core::store::Project::open(&target)?;
    let packed = &project.doc().resources[&potter_core::model::Id::new("packed_sequence")?];
    assert_eq!(packed["packed"], true);
    let frames = packed["sequence_frames"]
        .as_object()
        .ok_or("packed sequence frame records are missing")?;
    assert_eq!(frames.len(), 4);
    assert!(
        frames
            .values()
            .all(|record| record["packed"] == true && record["bytes"].is_array())
    );
    let mut corrupted = project.doc().clone();
    let bytes = corrupted
        .resources
        .get_mut(&potter_core::model::Id::new("packed_sequence")?)
        .ok_or("packed sequence resource missing")?
        .get_mut("sequence_frames")
        .and_then(Value::as_object_mut)
        .and_then(|frames| frames.get_mut("2"))
        .and_then(Value::as_object_mut)
        .and_then(|record| record.get_mut("bytes"))
        .and_then(Value::as_array_mut)
        .ok_or("packed sequence frame bytes are missing")?;
    bytes[0] = json!(256);
    let Err(error) = potter_core::eval::Snapshot::evaluate_with_cache(
        &corrupted,
        &potter_core::eval::EvaluationContext {
            frame: Some(2.0),
            ..potter_core::eval::EvaluationContext::default()
        },
        Some(project.path()),
        Some(project.cache_directory()),
    ) else {
        return Err("corrupt packed sequence payload must fail closed".into());
    };
    assert_eq!(error.code, potter_core::error::ErrorCode::SceneInvalid);
    let mut changed_packed = project.doc().clone();
    let valid_bytes = changed_packed
        .resources
        .get_mut(&potter_core::model::Id::new("packed_sequence")?)
        .ok_or("packed sequence resource missing")?
        .get_mut("sequence_frames")
        .and_then(Value::as_object_mut)
        .and_then(|frames| frames.get_mut("2"))
        .and_then(Value::as_object_mut)
        .and_then(|record| record.get_mut("bytes"))
        .and_then(Value::as_array_mut)
        .ok_or("packed sequence frame bytes are missing")?;
    let first_byte = valid_bytes[0]
        .as_u64()
        .ok_or("packed sequence first byte is not an unsigned integer")?;
    valid_bytes[0] = json!(first_byte ^ 1);
    let Err(error) = potter_core::eval::Snapshot::evaluate_with_cache(
        &changed_packed,
        &potter_core::eval::EvaluationContext {
            frame: Some(2.0),
            ..potter_core::eval::EvaluationContext::default()
        },
        Some(project.path()),
        Some(project.cache_directory()),
    ) else {
        return Err("a valid but changed packed frame must fail its stored hash".into());
    };
    assert_eq!(error.code, potter_core::error::ErrorCode::AssetChanged);
    let changed_packed_document = serde_json::to_value(&changed_packed)?;
    let mut noncanonical_document: Value = serde_json::to_value(project.doc())?;
    let frame_records = noncanonical_document["resources"]["packed_sequence"]["sequence_frames"]
        .as_object_mut()
        .ok_or("packed sequence frame records are missing")?;
    let frame_two = frame_records
        .remove("2")
        .ok_or("packed sequence frame 2 is missing")?;
    frame_records.insert("02".to_owned(), frame_two);
    drop(project);

    assert_failed_sequence_unpack_is_atomic(
        temporary.path(),
        "unpack_changed_hash",
        &changed_packed_document,
        "ASSET_CHANGED",
    )?;
    assert_failed_sequence_unpack_is_atomic(
        temporary.path(),
        "unpack_noncanonical_key",
        &noncanonical_document,
        "SCENE_INVALID",
    )?;

    let unpack_pattern = temporary.path().join("unpacked.####.abc");
    apply(
        &target,
        &json!([{"op":"resource.unpack","id":"packed_sequence","path":unpack_pattern}]),
    )?;
    for frame in 1..=4 {
        assert!(
            temporary
                .path()
                .join(format!("unpacked.{frame:04}.abc"))
                .is_file()
        );
    }
    for frame in [1.0, 2.0, 3.0, 4.0] {
        let expected = sequence_case["frames"]
            .as_array()
            .and_then(|frames| {
                frames
                    .iter()
                    .find(|sample| sample["frame"].as_f64() == Some(frame))
            })
            .ok_or("Blender sequence frame is missing")?;
        assert_positions_close(
            &mesh_positions(&target, frame)?,
            &serde_json::from_value::<Vec<[f64; 3]>>(expected["positions"].clone())?,
            1.0e-6,
        );
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn resource_pack_sequence_rejects_out_of_directory_symlink_without_commit()
-> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let target = temporary.path().join("symlink_sequence.pot");
    init(&target)?;

    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/alembic_cache");
    let sequence_root = temporary.path().join("sequence");
    let outside_root = temporary.path().join("outside");
    fs::create_dir_all(&sequence_root)?;
    fs::create_dir_all(&outside_root)?;
    fs::copy(
        fixture_dir.join("seq.0001.abc"),
        sequence_root.join("seq.0001.abc"),
    )?;
    let outside_frame = outside_root.join("seq.0002.abc");
    fs::copy(fixture_dir.join("seq.0002.abc"), &outside_frame)?;
    std::os::unix::fs::symlink(&outside_frame, sequence_root.join("seq.0002.abc"))?;

    let scene_path = target.join("scene.json");
    let before: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    let batch = target.with_extension("sequence-symlink-ops.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":before["revision"],
            "operations":[{
                "op":"resource.pack",
                "id":"escaped_sequence",
                "uri":sequence_root.join("seq.####.abc"),
                "kind":"alembic",
                "sequence":{"start":1,"end":2}
            }]
        }))?,
    )?;
    let output = pot()
        .args(["apply"])
        .arg(&target)
        .args(["--file"])
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        !output.status.success(),
        "resource.pack accepted a sequence frame symlink outside its directory"
    );
    let error: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(error["error"]["code"], "INVALID_ARGUMENT");
    assert_eq!(error["error"]["details"]["frame"], 2);
    let after: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    assert_eq!(after["revision"], before["revision"]);
    assert!(
        after["resources"].get("escaped_sequence").is_none(),
        "failed sequence packing committed a resource"
    );
    Ok(())
}

#[test]
fn blender_topology_change_forces_full_cache_read_with_vertex_only_flag()
-> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let target = temporary.path().join("topology.pot");
    let archive =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/alembic_cache/topo.abc");
    init(&target)?;
    run_ok(
        pot()
            .arg("import")
            .arg(&target)
            .arg("--file")
            .arg(&archive)
            .args([
                "--format",
                "alembic",
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--allow-lossy",
                "--json",
            ]),
    )?;
    let project = potter_core::store::Project::open(&target)?;
    let target_id = project
        .doc()
        .nodes
        .iter()
        .find_map(|(id, node)| (node.name == "TopoSource").then(|| id.clone()))
        .ok_or("topology oracle mesh node is missing")?;
    drop(project);
    apply(
        &target,
        &json!([
            {"op":"resource.pack","id":"topology_cache","uri":archive,"kind":"alembic"},
            {
                "op":"modifier.create",
                "target":{"id":target_id},
                "id":"topology_cache_modifier",
                "type":"mesh_sequence_cache",
                "params":{
                    "resource":"topology_cache",
                    "object_path":"/TopoSource/TopoSource",
                    "read_data":["VERT"]
                }
            }
        ]),
    )?;
    assert_eq!(evaluated_mesh(&target, 1.0)?.faces.len(), 6);
    assert_eq!(evaluated_mesh(&target, 3.0)?.faces.len(), 12);
    Ok(())
}

#[test]
fn mesh_sequence_cache_reports_missing_and_changed_external_files() -> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let source = temporary.path().join("source.pot");
    let archive = temporary.path().join("animated.abc");
    export_animated_cache(&source, &archive)?;
    let original = fs::read(&archive)?;
    let expected_hash = potter_core::hash::sha256(&original);

    let missing_scene = temporary.path().join("missing.pot");
    init(&missing_scene)?;
    apply(
        &missing_scene,
        &json!([{"op":"node.create", "id":"body", "kind":"plane", "params":{}}]),
    )?;
    let absent = temporary.path().join("missing.abc");
    add_external_cache_modifier(&missing_scene, &absent, &expected_hash)?;
    let output = inspect(&missing_scene, 1.0)?;
    assert_eq!(output.status.code(), Some(3));
    let error: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(error["error"]["code"], "DEPENDENCY_MISSING");

    let changed_file = temporary.path().join("changed.abc");
    let mut changed = original;
    let last = changed.len() - 1;
    changed[last] ^= 1;
    fs::write(&changed_file, changed)?;
    let changed_scene = temporary.path().join("changed.pot");
    init(&changed_scene)?;
    apply(
        &changed_scene,
        &json!([{"op":"node.create", "id":"body", "kind":"plane", "params":{}}]),
    )?;
    add_external_cache_modifier(&changed_scene, &changed_file, &expected_hash)?;
    let output = inspect(&changed_scene, 1.0)?;
    assert_eq!(output.status.code(), Some(4));
    let error: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(error["error"]["code"], "ASSET_CHANGED");
    Ok(())
}

#[test]
fn mesh_sequence_cache_replaces_topology_for_vertex_only_reads() -> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let source = temporary.path().join("source.pot");
    let archive = temporary.path().join("animated.abc");
    export_animated_cache(&source, &archive)?;
    let target = temporary.path().join("target.pot");
    init(&target)?;
    apply(
        &target,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{"size":2.0}},
            {"op":"resource.pack", "id":"cache", "uri":archive, "kind":"alembic"},
            {"op":"modifier.create", "target":{"id":"body"}, "id":"cache", "type":"mesh_sequence_cache", "params":{"resource":"cache", "object_path":"/root/body/geometry", "read_data":["VERT"]}}
        ]),
    )?;
    assert_positions_close(
        &mesh_positions(&target, 1.0)?,
        &mesh_positions(&source, 1.0)?,
        1.0e-6,
    );
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "cache modifier and resource operation schemas share one oracle contract test"
)]
#[test]
fn mesh_sequence_cache_and_resource_schema_describe_parameters() -> Result<(), Box<dyn Error>> {
    let schema = potter_core::schema::schema("operations", Some("modifier.create"))?;
    let cache_params = schema
        .get("allOf")
        .and_then(Value::as_array)
        .and_then(|conditions| {
            conditions.iter().find(|condition| {
                condition
                    .get("if")
                    .and_then(|condition| condition.get("properties"))
                    .and_then(|properties| properties.get("type"))
                    .and_then(|type_schema| type_schema.get("const"))
                    .and_then(Value::as_str)
                    == Some("mesh_sequence_cache")
            })
        })
        .and_then(|condition| condition.get("then"))
        .and_then(|then| then.get("properties"))
        .and_then(|properties| properties.get("params"))
        .ok_or("modifier schema is missing conditional cache parameters")?;
    assert_eq!(cache_params.get("required"), Some(&json!(["object_path"])));
    assert_eq!(
        cache_params
            .get("properties")
            .and_then(|properties| properties.get("read_data"))
            .and_then(|read_data| read_data.get("items"))
            .and_then(|items| items.get("enum")),
        Some(&json!(["VERT", "POLY", "UV", "COLOR", "ATTRIBUTES"]))
    );
    for field in [
        "resource",
        "object_path",
        "read_data",
        "frame_offset",
        "is_sequence",
        "velocity_name",
        "velocity_unit",
        "override_frame",
        "velocity_scale",
        "use_vertex_interpolation",
    ] {
        let field_present = cache_params
            .get("properties")
            .and_then(|properties| properties.get(field))
            .is_some();
        assert!(field_present);
    }
    let pack_schema = potter_core::schema::schema("operations", Some("resource.pack"))?;
    let pack_required = pack_schema["required"]
        .as_array()
        .ok_or("resource.pack schema is missing required fields")?;
    for optional in ["kind", "sequence"] {
        assert!(
            !pack_required
                .iter()
                .any(|field| field.as_str() == Some(optional))
        );
    }
    assert_eq!(pack_schema["properties"]["kind"]["type"], "string");
    assert_eq!(pack_schema["properties"]["kind"]["default"], "binary");
    assert_eq!(pack_schema["properties"]["sequence"]["type"], "object");
    assert_eq!(
        pack_schema["properties"]["sequence"]["required"],
        json!(["start", "end"])
    );
    assert_eq!(
        pack_schema["properties"]["sequence"]["additionalProperties"],
        false
    );
    for bound in ["start", "end"] {
        assert_eq!(
            pack_schema["properties"]["sequence"]["properties"][bound]["type"],
            "integer"
        );
        assert_eq!(
            pack_schema["properties"]["sequence"]["properties"][bound]["minimum"],
            json!(i32::MIN)
        );
        assert_eq!(
            pack_schema["properties"]["sequence"]["properties"][bound]["maximum"],
            json!(i32::MAX)
        );
    }
    assert_eq!(
        pack_schema["allOf"][0]["if"]["required"],
        json!(["sequence"])
    );
    assert_eq!(pack_schema["allOf"][0]["then"]["required"], json!(["kind"]));
    assert_eq!(
        pack_schema["allOf"][0]["then"]["properties"]["kind"]["const"],
        "alembic"
    );

    let unpack_schema = potter_core::schema::schema("operations", Some("resource.unpack"))?;
    let unpack_required = unpack_schema["required"]
        .as_array()
        .ok_or("resource.unpack schema is missing required fields")?;
    assert!(
        !unpack_required
            .iter()
            .any(|field| field.as_str() == Some("path"))
    );
    assert_eq!(unpack_schema["properties"]["path"]["type"], "string");
    Ok(())
}
