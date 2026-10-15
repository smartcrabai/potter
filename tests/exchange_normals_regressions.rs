//! Regressions for degenerate and singular geometry. A collapsed face (zero area, coincident
//! vertices) and a zero-scale node must export to every format without a new error, and the
//! normals written for the collapsed face must match Blender 5.2.2.
//!
//! Construction: `mesh.transform_elements` flattens a plane onto a line, so its single face has
//! zero area with coincident vertices. The expected values come from Blender 5.2.2 exporting the
//! same four-vertex face (`(-1,0,0), (1,0,0), (1,0,0), (-1,0,0)`): every corner and vertex normal
//! is the unit Z normal, in both flat and smooth shading. Blender writes zero OBJ normals for a
//! zero-scale object, which the zero-scale checks below assert.

#[path = "common/exchange_normals.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared fixture helpers"
)]
mod exchange_normals;

use std::{fs, path::Path};

use exchange_normals::{TestResult, pot};
use serde_json::{Value, json};
use tempfile::tempdir;

/// Every format `pot export` accepts for shading, with its file extension.
const FORMATS: [(&str, &str); 10] = [
    ("obj", "obj"),
    ("usda", "usda"),
    ("usdc", "usdc"),
    ("fbx", "fbx"),
    ("fbx-binary", "fbx"),
    ("ply", "ply"),
    ("stl", "stl"),
    ("alembic", "abc"),
    ("gltf", "gltf"),
    ("glb", "glb"),
];

/// Two operations: a 2x2 plane at `translation` whose four vertices are flattened onto the X axis.
fn collapsed_plane(id: &str, translation: [f64; 3]) -> Vec<Value> {
    vec![
        json!({
            "op": "node.create", "id": id, "kind": "plane", "params": {"size": 2.0},
            "transform": {"translation": translation}
        }),
        json!({
            "op": "mesh.transform_elements",
            "target": {"id": id, "elements": {"domain": "vertex", "ids": ["v0", "v1", "v2", "v3"]}},
            "scale": [1.0, 0.0, 1.0], "pivot": [0.0, 0.0, 0.0]
        }),
    ]
}

/// Writes a scene with a collapsed flat plane and a zero-scale flat box. With `shaded`, it also
/// adds smooth copies of both, so the scene carries per-corner shading.
fn write_scene(scene: &Path, shaded: bool) -> TestResult {
    let init = pot().arg("init").arg(scene).arg("--json").output()?;
    assert!(init.status.success(), "pot init failed");
    let mut operations = collapsed_plane("collapsed_flat", [0.0, 0.0, 0.0]);
    operations.push(json!({
        "op": "node.create", "id": "zscale_flat", "kind": "box", "params": {"size": 2.0},
        "transform": {"translation": [5.0, 0.0, 0.0], "scale": [0.0, 1.0, 1.0]}
    }));
    if shaded {
        operations.extend(collapsed_plane("collapsed_smooth", [0.0, 3.0, 0.0]));
        operations.push(json!({
            "op": "mesh.shade_smooth", "target": {"id": "collapsed_smooth"},
            "elements": {"domain": "face", "ids": ["f0"]}
        }));
        operations.push(json!({
            "op": "node.create", "id": "zscale_smooth", "kind": "box", "params": {"size": 2.0},
            "transform": {"translation": [5.0, 5.0, 0.0], "scale": [0.0, 1.0, 1.0]}
        }));
        operations.push(json!({
            "op": "mesh.shade_smooth", "target": {"id": "zscale_smooth"},
            "elements": {"domain": "face", "ids": ["f0", "f1", "f2", "f3", "f4", "f5"]}
        }));
    }
    let batch = json!({"schema_version": 1, "base_revision": 0, "operations": operations});
    let batch_file = scene.with_extension("json");
    fs::write(&batch_file, batch.to_string())?;
    let apply = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(&batch_file)
        .arg("--json")
        .output()?;
    assert!(
        apply.status.success(),
        "pot apply failed: {}",
        String::from_utf8_lossy(&apply.stdout)
    );
    Ok(())
}

/// Exports `scene` as `format` and returns the envelope. Lossy exports need `allow_lossy`.
fn export(scene: &Path, format: &str, out: &Path, allow_lossy: bool) -> TestResult<Value> {
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
    let output = command.output()?;
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    assert!(
        output.status.success(),
        "pot export {format} failed: {envelope}"
    );
    Ok(envelope)
}

fn losses(envelope: &Value) -> Vec<Value> {
    envelope["result"]["losses"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// Corner normals of the named OBJ object, in face order, as written in OBJ axes.
fn obj_corner_normals(text: &str, name: &str) -> TestResult<Vec<[f64; 3]>> {
    let mut vn = Vec::<[f64; 3]>::new();
    let mut current = String::new();
    let mut corners = Vec::new();
    for line in text.lines() {
        let mut words = line.split_whitespace();
        match words.next() {
            Some("o") => {
                current = words.collect::<Vec<_>>().join(" ");
                // Potter numbers OBJ normals across the whole file, so the list is never reset.
            }
            Some("vn") => {
                let values = words
                    .map(str::parse::<f64>)
                    .collect::<Result<Vec<_>, _>>()?;
                let [x, y, z] = values[..] else {
                    return Err("OBJ vn needs three components".into());
                };
                vn.push([x, y, z]);
            }
            Some("f") if current == name => {
                for corner in words {
                    let index: usize = corner
                        .split("//")
                        .nth(1)
                        .ok_or("OBJ face corner has no normal index")?
                        .parse()?;
                    corners.push(*vn.get(index - 1).ok_or("OBJ normal index out of range")?);
                }
            }
            _ => {}
        }
    }
    Ok(corners)
}

/// `(position, normal)` per vertex of a binary little-endian PLY file whose vertices carry float
/// positions and normals.
fn ply_vertices(bytes: &[u8]) -> TestResult<Vec<([f64; 3], [f64; 3])>> {
    const END_HEADER: &[u8] = b"end_header\n";
    let end = bytes
        .windows(END_HEADER.len())
        .position(|window| window == END_HEADER)
        .ok_or("PLY header is missing end_header")?
        + END_HEADER.len();
    let header = std::str::from_utf8(&bytes[..end])?;
    let mut count = 0_usize;
    let mut properties = Vec::<(String, usize)>::new();
    let mut in_vertices = false;
    for line in header.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["element", "vertex", n] => {
                count = n.parse()?;
                in_vertices = true;
            }
            ["element", _, _] => in_vertices = false,
            ["property", kind, name] if in_vertices => {
                let size = match *kind {
                    "float" => 4,
                    "uchar" => 1,
                    other => return Err(format!("unexpected PLY property type {other}").into()),
                };
                properties.push(((*name).to_owned(), size));
            }
            _ => {}
        }
    }
    let stride: usize = properties.iter().map(|(_, size)| size).sum();
    let offset_of = |name: &str| -> TestResult<usize> {
        let mut offset = 0;
        for (property, size) in &properties {
            if property == name {
                return Ok(offset);
            }
            offset += size;
        }
        Err(format!("PLY is missing property {name}").into())
    };
    let offsets = ["x", "y", "z", "nx", "ny", "nz"]
        .into_iter()
        .map(offset_of)
        .collect::<TestResult<Vec<_>>>()?;
    let body = &bytes[end..];
    let mut vertices = Vec::with_capacity(count);
    for index in 0..count {
        let base = index * stride;
        let read = |offset: usize| -> TestResult<f64> {
            let start = base + offset;
            let raw: [u8; 4] = body
                .get(start..start + 4)
                .ok_or("PLY body is truncated")?
                .try_into()?;
            Ok(f64::from(f32::from_le_bytes(raw)))
        };
        let position = [read(offsets[0])?, read(offsets[1])?, read(offsets[2])?];
        let normal = [read(offsets[3])?, read(offsets[4])?, read(offsets[5])?];
        vertices.push((position, normal));
    }
    Ok(vertices)
}

/// `(point count, normals)` of each USDA `Mesh` block, in file order.
fn usda_point_normals(text: &str) -> Vec<(usize, Vec<[f64; 3]>)> {
    let tuples = |list: &str| -> Vec<[f64; 3]> {
        list.split('(')
            .skip(1)
            .filter_map(|chunk| {
                let inner = chunk.split(')').next()?;
                let parts = inner
                    .split(',')
                    .map(|part| part.trim().parse::<f64>().ok())
                    .collect::<Option<Vec<_>>>()?;
                match parts[..] {
                    [x, y, z] => Some([x, y, z]),
                    _ => None,
                }
            })
            .collect()
    };
    let mut blocks = Vec::new();
    let mut points = 0;
    for line in text.lines() {
        if let Some((_, list)) = line.split_once("point3f[] points = [") {
            points = tuples(list).len();
        } else if let Some((_, list)) = line.split_once("normal3f[] normals = [") {
            let list = list.split("] (").next().unwrap_or(list);
            blocks.push((points, tuples(list)));
        }
    }
    blocks
}

fn assert_vec_near(actual: [f64; 3], expected: [f64; 3], context: &str) {
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= 1.0e-4,
            "{context}: expected {expected:?}, got {actual:?}"
        );
    }
}

#[test]
fn flat_degenerate_scene_exports_to_every_format_without_losses() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("flat.pot");
    write_scene(&scene, false)?;
    for (format, extension) in FORMATS {
        let out = directory.path().join(format!("{format}.{extension}"));
        let envelope = export(&scene, format, &out, false)?;
        assert!(out.exists(), "{format} wrote no file");
        assert_eq!(
            losses(&envelope),
            Vec::<Value>::new(),
            "{format} reported losses"
        );
    }
    Ok(())
}

#[test]
fn shaded_degenerate_scene_exports_to_every_format_with_only_shading_losses() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("shaded.pot");
    write_scene(&scene, true)?;
    for (format, extension) in FORMATS {
        let out = directory.path().join(format!("{format}.{extension}"));
        let envelope = export(&scene, format, &out, true)?;
        assert!(out.exists(), "{format} wrote no file");
        let reported = losses(&envelope);
        for loss in &reported {
            assert_eq!(
                loss["feature_id"], "mesh.shading",
                "{format} reported an unexpected loss: {loss}"
            );
        }
        if format == "stl" {
            assert!(
                !reported.is_empty(),
                "STL cannot hold smooth shading and must report it"
            );
        }
    }
    Ok(())
}

#[test]
fn collapsed_face_normals_match_blender_oracle() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("shaded.pot");
    write_scene(&scene, true)?;

    // OBJ: Blender's Z is stored as (0, 1, 0) in OBJ's Y-up axes.
    let obj = directory.path().join("out.obj");
    export(&scene, "obj", &obj, true)?;
    let text = fs::read_to_string(&obj)?;
    for name in ["collapsed_flat", "collapsed_smooth"] {
        let corners = obj_corner_normals(&text, name)?;
        assert_eq!(corners.len(), 4, "{name} has four corners");
        for corner in corners {
            assert_vec_near(corner, [0.0, 1.0, 0.0], name);
        }
    }
    // Blender 5.2.2 writes zero normals for every corner of a zero-scale object in OBJ.
    for name in ["zscale_flat", "zscale_smooth"] {
        for corner in obj_corner_normals(&text, name)? {
            assert_vec_near(corner, [0.0, 0.0, 0.0], name);
        }
    }

    // PLY: the smooth collapsed vertices sit at y = 3 and must have Blender's vertex normal Z.
    let ply = directory.path().join("out.ply");
    export(&scene, "ply", &ply, true)?;
    let vertices = ply_vertices(&fs::read(&ply)?)?;
    let collapsed: Vec<_> = vertices
        .iter()
        .filter(|(position, _)| {
            (position[1] - 3.0).abs() <= 1.0e-6 && (position[0].abs() - 1.0).abs() <= 1.0e-6
        })
        .collect();
    assert_eq!(collapsed.len(), 4, "four collapsed smooth PLY vertices");
    for (_, normal) in collapsed {
        assert_vec_near(*normal, [0.0, 0.0, 1.0], "PLY collapsed vertex");
    }

    // USDA: each four-point mesh is a collapsed face, and its corner normals are Blender's Z.
    let usda = directory.path().join("out.usda");
    export(&scene, "usda", &usda, true)?;
    let blocks = usda_point_normals(&fs::read_to_string(&usda)?);
    let collapsed_blocks: Vec<_> = blocks.iter().filter(|(points, _)| *points == 4).collect();
    assert_eq!(collapsed_blocks.len(), 2, "two collapsed USD meshes");
    for (_, normals) in collapsed_blocks {
        assert_eq!(normals.len(), 4, "four USD corners");
        for normal in normals {
            assert_vec_near(*normal, [0.0, 0.0, 1.0], "USD collapsed corner");
        }
    }
    Ok(())
}
