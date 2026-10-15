//! PLY and STL shading normals, matching Blender 5.2.2 defaults: PLY stores one normal per vertex
//! (the normalized sum of its corner normals) and omits normals for flat meshes; STL stores one
//! facet normal per triangle, and Blender ignores stored facet normals on import.

#[path = "common/exchange_normals.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared fixture helpers"
)]
mod exchange_normals;

use std::{fs, path::Path, process::Output};

use exchange_normals::{
    BLENDER_NORMAL_TOLERANCE, NORMAL_TOLERANCE, TestResult, export, export_strict, init, near, pot,
    reference_corners, write_fixture_scene,
};
use serde_json::{Value, json};
use tempfile::tempdir;

fn envelope(output: &Output) -> TestResult<Value> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Losses from a success envelope, or from the details of an `UNREPRESENTABLE_FEATURE` error.
fn losses(envelope: &Value) -> Vec<Value> {
    envelope["error"]["details"]["result"]["losses"]
        .as_array()
        .or_else(|| envelope["result"]["losses"].as_array())
        .cloned()
        .unwrap_or_default()
}

fn has_feature(losses: &[Value], feature: &str) -> bool {
    losses.iter().any(|loss| loss["feature_id"] == feature)
}

fn apply_operations(scene: &Path, operations: &Value) -> TestResult {
    init(scene)?;
    let batch = scene.with_extension("batch.json");
    fs::write(
        &batch,
        json!({"schema_version": 1, "base_revision": 0, "operations": operations}).to_string(),
    )?;
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(&batch)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

/// One unshaded box: every corner equals its face normal.
fn flat_box_operations() -> Value {
    json!([{"op": "node.create", "id": "cube", "kind": "box", "params": {"size": 2.0}}])
}

/// One box with every face smooth, so vertex normals differ from face normals.
fn smooth_box_operations() -> Value {
    json!([
        {"op": "node.create", "id": "cube", "kind": "box", "params": {"size": 2.0}},
        {
            "op": "mesh.shade_smooth", "target": {"id": "cube"},
            "elements": {"domain": "face", "ids": ["f0", "f1", "f2", "f3", "f4", "f5"]}
        }
    ])
}

fn export_with(scene: &Path, format: &str, out: &Path, allow_lossy: bool) -> TestResult<Output> {
    let mut command = pot();
    command
        .arg("export")
        .arg(scene)
        .args(["--format", format, "--out"])
        .arg(out)
        .args(["--overwrite", "--json"]);
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    Ok(command.output()?)
}

fn import_file(scene: &Path, file: &Path, format: &str, allow_lossy: bool) -> TestResult<Output> {
    let mut command = pot();
    command
        .arg("import")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .args([
            "--format",
            format,
            "--base-revision",
            "0",
            "--mode",
            "replace",
            "--json",
        ]);
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    Ok(command.output()?)
}

fn mesh_attribute_names(scene: &Path) -> TestResult<Vec<String>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let blocks = document["data_blocks"]
        .as_object()
        .ok_or("scene has no data_blocks")?;
    let mut names = Vec::new();
    for block in blocks.values() {
        if let Some(attributes) = block["mesh"]["attributes"].as_object() {
            names.extend(attributes.keys().cloned());
        }
    }
    Ok(names)
}

/// Decoded binary little-endian PLY vertex element. Faces are not decoded.
struct Ply {
    has_normals: bool,
    vertices: Vec<Vertex>,
}

#[derive(Clone, Copy)]
struct Vertex {
    position: [f64; 3],
    /// Zero when the file carries no normal properties.
    normal: [f64; 3],
}

fn scalar_size(kind: &str) -> TestResult<usize> {
    match kind {
        "char" | "uchar" | "int8" | "uint8" => Ok(1),
        "short" | "ushort" | "int16" | "uint16" => Ok(2),
        "int" | "uint" | "float" | "int32" | "uint32" | "float32" => Ok(4),
        "double" | "float64" => Ok(8),
        other => Err(format!("unsupported PLY scalar type {other}").into()),
    }
}

fn read_float(bytes: &[u8], offset: usize, kind: &str) -> TestResult<f64> {
    if kind != "float" {
        return Err(format!("test decoder expects float vertex properties, found {kind}").into());
    }
    let raw: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or("PLY vertex data is truncated")?
        .try_into()?;
    Ok(f64::from(f32::from_le_bytes(raw)))
}

fn parse_ply(bytes: &[u8]) -> TestResult<Ply> {
    let marker = b"end_header\n";
    let body = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or("PLY header has no end_header")?
        + marker.len();
    let header = std::str::from_utf8(&bytes[..body])?;
    let mut element = String::new();
    let mut vertex_count = 0_usize;
    let mut properties = Vec::<(String, String)>::new();
    for line in header.lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        match fields.as_slice() {
            ["format", format, ..] => {
                assert_eq!(
                    *format, "binary_little_endian",
                    "test decodes binary PLY only"
                );
            }
            ["element", name, count] => {
                (*name).clone_into(&mut element);
                if element == "vertex" {
                    vertex_count = count.parse::<usize>()?;
                }
            }
            ["property", kind, name] if element == "vertex" => {
                properties.push(((*kind).to_owned(), (*name).to_owned()));
            }
            _ => {}
        }
    }
    let has_normals = ["nx", "ny", "nz"]
        .iter()
        .all(|wanted| properties.iter().any(|(_, name)| name.as_str() == *wanted));
    let mut stride = 0_usize;
    for (kind, _) in &properties {
        stride += scalar_size(kind)?;
    }
    if bytes.len() < body + vertex_count * stride {
        return Err("PLY vertex data is truncated".into());
    }
    let mut vertices = Vec::with_capacity(vertex_count);
    for index in 0..vertex_count {
        let mut offset = body + index * stride;
        let mut vertex = Vertex {
            position: [0.0; 3],
            normal: [0.0; 3],
        };
        for (kind, name) in &properties {
            let slot = match name.as_str() {
                "x" => Some((false, 0)),
                "y" => Some((false, 1)),
                "z" => Some((false, 2)),
                "nx" => Some((true, 0)),
                "ny" => Some((true, 1)),
                "nz" => Some((true, 2)),
                _ => None,
            };
            if let Some((is_normal, axis)) = slot {
                let value = read_float(bytes, offset, kind)?;
                if is_normal {
                    vertex.normal[axis] = value;
                } else {
                    vertex.position[axis] = value;
                }
            }
            offset += scalar_size(kind)?;
        }
        vertices.push(vertex);
    }
    Ok(Ply {
        has_normals,
        vertices,
    })
}

fn normalized(vector: [f64; 3]) -> [f64; 3] {
    let length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    vector.map(|value| value / length)
}

fn max_error(left: [f64; 3], right: [f64; 3]) -> f64 {
    (0..3)
        .map(|axis| (left[axis] - right[axis]).abs())
        .fold(0.0_f64, f64::max)
}

/// Binary STL with one triangle (0,0,0), (1,0,0), (0,1,0): counter-clockwise about +Z, so its
/// winding normal is (0,0,1). `stored` is the facet normal written in the file.
fn stl_single_triangle(stored: [f32; 3]) -> Vec<u8> {
    let mut bytes = vec![0_u8; 80];
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    let vertices = [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    for value in stored.iter().chain(vertices.iter().flatten()) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    bytes
}

/// Diagonal scale of the fixture object that owns a world position: `NonUni` at y = 5 with scale
/// (2, 1, 0.5), `Mirror` at x = 5 with scale (-1, 1, 1), `Mixed` at the origin.
fn fixture_scale(position: [f64; 3]) -> [f64; 3] {
    if position[1] >= 3.0 {
        [2.0, 1.0, 0.5]
    } else if position[0] >= 3.0 {
        [-1.0, 1.0, 1.0]
    } else {
        [1.0, 1.0, 1.0]
    }
}

#[test]
fn ply_fixture_vertex_normals_are_blender_per_vertex_sums() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    let file = export(&scene, "ply", &directory.path().join("fixture.ply"))?;
    let ply = parse_ply(&fs::read(&file)?)?;
    assert!(ply.has_normals, "shaded fixture must write nx ny nz");
    assert_eq!(ply.vertices.len(), 24, "one PLY vertex per object vertex");

    let reference = reference_corners()?;
    for vertex in &ply.vertices {
        // Blender sums the object-local corner normals and transforms the sum by N = M^-T. A world
        // corner normal n_w is proportional to M^-T n_local, so n_local is proportional to M^T n_w.
        let mut sum = [0.0_f64; 3];
        let mut corners = 0;
        for corner in reference
            .iter()
            .filter(|corner| near(corner.corner, vertex.position))
        {
            corners += 1;
            let scale = fixture_scale(corner.corner);
            let local = normalized([
                corner.normal[0] * scale[0],
                corner.normal[1] * scale[1],
                corner.normal[2] * scale[2],
            ]);
            for (total, value) in sum.iter_mut().zip(local) {
                *total += value;
            }
        }
        assert!(
            corners > 0,
            "PLY vertex {:?} matches no Blender corner",
            vertex.position
        );
        let scale = fixture_scale(vertex.position);
        let expected = normalized([sum[0] / scale[0], sum[1] / scale[1], sum[2] / scale[2]]);
        // Blender's corner normals are int16-quantized, so its per-vertex sums can differ from
        // exact sums by more than NORMAL_TOLERANCE. Compare with the Blender-side bound.
        let error = max_error(vertex.normal, expected);
        assert!(
            error <= BLENDER_NORMAL_TOLERANCE,
            "vertex {:?}: normal {:?} differs from Blender's sum {:?} (error {error})",
            vertex.position,
            vertex.normal,
            expected
        );
    }
    Ok(())
}

#[test]
fn flat_only_ply_writes_no_normals_and_imports_flat() -> TestResult {
    let directory = tempdir()?;
    let flat_scene = directory.path().join("flat");
    apply_operations(&flat_scene, &flat_box_operations())?;
    let flat_file = directory.path().join("flat.ply");
    let output = export_with(&flat_scene, "ply", &flat_file, false)?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(envelope(&output)?["result"]["losses"], json!([]));
    let flat = parse_ply(&fs::read(&flat_file)?)?;
    assert!(
        !flat.has_normals,
        "flat-only meshes must not write normal properties"
    );

    let imported = directory.path().join("imported-flat");
    init(&imported)?;
    let import = import_file(&imported, &flat_file, "ply", false)?;
    assert!(
        import.status.success(),
        "{}",
        String::from_utf8_lossy(&import.stdout)
    );
    assert_eq!(envelope(&import)?["result"]["losses"], json!([]));
    assert!(
        !mesh_attribute_names(&imported)?
            .iter()
            .any(|name| name == "shade_smooth"),
        "a flat PLY must import as flat"
    );

    // Control: a smooth box writes normals and imports smooth, so the check above can fail.
    let smooth_scene = directory.path().join("smooth");
    apply_operations(&smooth_scene, &smooth_box_operations())?;
    let smooth_file = directory.path().join("smooth.ply");
    let output = export_with(&smooth_scene, "ply", &smooth_file, true)?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(parse_ply(&fs::read(&smooth_file)?)?.has_normals);
    let smooth_import = directory.path().join("imported-smooth");
    init(&smooth_import)?;
    let import = import_file(&smooth_import, &smooth_file, "ply", false)?;
    assert!(
        import.status.success(),
        "{}",
        String::from_utf8_lossy(&import.stdout)
    );
    assert!(
        mesh_attribute_names(&smooth_import)?
            .iter()
            .any(|name| name == "shade_smooth"),
        "a PLY with vertex normals must import as smooth"
    );
    Ok(())
}

#[test]
fn ply_flat_mesh_beside_a_shaded_mesh_reports_shading_loss() -> TestResult {
    // One PLY file carries normals for every mesh once any mesh is shaded, so the flat box loses too.
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let mut operations = flat_box_operations()
        .as_array()
        .cloned()
        .unwrap_or_default();
    operations.extend(
        smooth_box_operations()
            .as_array()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|mut op| {
                if op["id"] == "cube" {
                    op["id"] = json!("smooth");
                }
                if op["target"]["id"] == "cube" {
                    op["target"]["id"] = json!("smooth");
                }
                op
            }),
    );
    apply_operations(&scene, &json!(operations))?;
    let refused = export_strict(&scene, "ply", &directory.path().join("mixed.ply"))?;
    assert_eq!(refused.status.code(), Some(4));
    assert!(has_feature(&losses(&envelope(&refused)?), "mesh.shading"));
    Ok(())
}

#[test]
fn fixture_ply_export_is_refused_without_allow_lossy() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;

    let refused_path = directory.path().join("refused.ply");
    let refused = export_strict(&scene, "ply", &refused_path)?;
    assert_eq!(refused.status.code(), Some(4));
    let body = envelope(&refused)?;
    assert_eq!(body["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert!(has_feature(&losses(&body), "mesh.shading"));
    assert!(!refused_path.exists(), "refused export must not write");

    let allowed_path = directory.path().join("allowed.ply");
    let allowed = export_with(&scene, "ply", &allowed_path, true)?;
    assert!(
        allowed.status.success(),
        "{}",
        String::from_utf8_lossy(&allowed.stdout)
    );
    assert!(has_feature(&losses(&envelope(&allowed)?), "mesh.shading"));
    assert!(allowed_path.exists());
    Ok(())
}

#[test]
fn fixture_stl_export_is_refused_without_allow_lossy() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;

    let refused_path = directory.path().join("refused.stl");
    let refused = export_strict(&scene, "stl", &refused_path)?;
    assert_eq!(refused.status.code(), Some(4));
    let body = envelope(&refused)?;
    assert_eq!(body["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert!(has_feature(&losses(&body), "mesh.shading"));
    assert!(!refused_path.exists(), "refused export must not write");
    Ok(())
}

#[test]
fn flat_box_stl_export_reports_no_loss() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("flat");
    apply_operations(&scene, &flat_box_operations())?;
    let output = export_strict(&scene, "stl", &directory.path().join("flat.stl"))?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(envelope(&output)?["result"]["losses"], json!([]));
    Ok(())
}

#[test]
fn stl_import_reports_facet_normal_that_contradicts_winding() -> TestResult {
    let directory = tempdir()?;
    let file = directory.path().join("flipped.stl");
    fs::write(&file, stl_single_triangle([0.0, 0.0, -1.0]))?;

    let refused_project = directory.path().join("refused");
    init(&refused_project)?;
    let refused = import_file(&refused_project, &file, "stl", false)?;
    assert_eq!(refused.status.code(), Some(4));
    assert!(has_feature(&losses(&envelope(&refused)?), "mesh.normals"));

    let project = directory.path().join("imported");
    init(&project)?;
    let imported = import_file(&project, &file, "stl", true)?;
    assert!(
        imported.status.success(),
        "{}",
        String::from_utf8_lossy(&imported.stdout)
    );
    assert!(has_feature(&losses(&envelope(&imported)?), "mesh.normals"));
    Ok(())
}

#[test]
fn stl_import_accepts_agreeing_or_zero_facet_normals() -> TestResult {
    for stored in [[0.0_f32, 0.0, 1.0], [0.0, 0.0, 0.0]] {
        let directory = tempdir()?;
        let file = directory.path().join("triangle.stl");
        fs::write(&file, stl_single_triangle(stored))?;
        let project = directory.path().join("imported");
        init(&project)?;
        let imported = import_file(&project, &file, "stl", false)?;
        assert!(
            imported.status.success(),
            "stored {stored:?}: {}",
            String::from_utf8_lossy(&imported.stdout)
        );
        assert_eq!(
            envelope(&imported)?["result"]["losses"],
            json!([]),
            "stored {stored:?}"
        );
    }
    Ok(())
}

#[test]
fn ply_round_trip_keeps_per_vertex_normals() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    let first_file = export(&scene, "ply", &directory.path().join("first.ply"))?;
    let first = parse_ply(&fs::read(&first_file)?)?;

    let imported = directory.path().join("imported");
    init(&imported)?;
    let import = import_file(&imported, &first_file, "ply", false)?;
    assert!(
        import.status.success(),
        "{}",
        String::from_utf8_lossy(&import.stdout)
    );
    assert_eq!(envelope(&import)?["result"]["losses"], json!([]));

    let second_file = directory.path().join("second.ply");
    let second_export = export_strict(&imported, "ply", &second_file)?;
    assert!(
        second_export.status.success(),
        "re-export of an imported PLY must be lossless: {}",
        String::from_utf8_lossy(&second_export.stdout)
    );
    let second = parse_ply(&fs::read(&second_file)?)?;
    assert!(second.has_normals);
    assert_eq!(second.vertices.len(), first.vertices.len());
    for vertex in &first.vertices {
        let other = second
            .vertices
            .iter()
            .find(|candidate| near(candidate.position, vertex.position))
            .ok_or("re-exported PLY lost a vertex")?;
        let error = max_error(vertex.normal, other.normal);
        assert!(
            error <= NORMAL_TOLERANCE,
            "vertex {:?}: normal {:?} became {:?} after round trip (error {error})",
            vertex.position,
            vertex.normal,
            other.normal
        );
    }
    Ok(())
}
