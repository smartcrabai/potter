//! Shading carried through mirror and subdivision must reach the glTF `NORMAL` accessor.
//!
//! The input is the `chain_mirror_subsurf` fixture in `tests/fixtures/pr4_shading`: a unit cube
//! with smooth flags, sharp edges and stored custom normals. Blender 5.2.2 evaluated
//! `mirror(use_axis X)` then `subdivision(levels 1, use_custom_normals)` on it, and its per-corner
//! normals are the reference. The scene is rebuilt through the CLI from the fixture's own input
//! values, and the exported GLB is matched to Blender's output corners by position.
//!
//! Potter's output geometry does not match Blender's yet. Only 4 faces on each side of the mirror
//! plane (the faces at x = +-0.82) have glTF triangles at the same positions, so only those 32
//! corners are compared. Every other Blender face is uncompared and reported. The compared face
//! set and corner count are pinned, so a change in either direction fails the test.

use std::{collections::HashMap, fs, path::Path};

use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/exchange_normals.rs"]
#[expect(
    dead_code,
    reason = "this test binary uses only the CLI helpers and the quantized normal tolerance"
)]
mod exchange_normals;

use exchange_normals::{BLENDER_NORMAL_TOLERANCE, TestResult, export_strict, init, pot};

/// Blender's output corners of the fixture (44 faces, 176 corners).
const BLENDER_CORNERS: usize = 176;
/// Position tolerance for matching glTF vertices and fixture corners. Blender's output is f32.
const POSITION_TOLERANCE: f64 = 1.0e-4;
/// Potter's box primitive spans [-0.5, 0.5]; the fixture cube spans [0, 1].
const BOX_TO_FIXTURE_OFFSET: f64 = 0.5;

fn fixture() -> TestResult<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pr4_shading/chain_mirror_subsurf.json");
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

/// Applies `operations` to the project at `scene`, using the revision the project currently has.
fn apply(scene: &Path, operations: &Value) -> TestResult {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let batch = scene.with_extension("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": document["revision"],
            "operations": operations,
        }))?,
    )?;
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "pot apply failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

/// The mesh data block of the project at `scene`.
fn project_mesh(scene: &Path) -> TestResult<Value> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    document["data_blocks"]
        .as_object()
        .ok_or("data_blocks is missing")?
        .values()
        .find(|block| block["mesh"].is_object())
        .map(|block| block["mesh"].clone())
        .ok_or_else(|| "project has no mesh data".into())
}

fn vec3(value: &Value) -> TestResult<[f64; 3]> {
    let [x, y, z] = value.as_array().ok_or("vector is not an array")?.as_slice() else {
        return Err("vector must have three components".into());
    };
    Ok([
        x.as_f64().ok_or("component")?,
        y.as_f64().ok_or("component")?,
        z.as_f64().ok_or("component")?,
    ])
}

fn index(value: &Value) -> TestResult<usize> {
    Ok(usize::try_from(
        value.as_u64().ok_or("index must be an unsigned integer")?,
    )?)
}

fn fixture_vertex(position: [f64; 3], vertices: &[[f64; 3]]) -> TestResult<usize> {
    vertices
        .iter()
        .position(|candidate| (0..3).all(|axis| (candidate[axis] - position[axis]).abs() <= 1.0e-6))
        .ok_or_else(|| format!("no fixture vertex at {position:?}").into())
}

/// Operations that rebuild the fixture's input on a box primitive, then add its stack.
///
/// The box's vertex, face and edge IDs are read from the project, and each is matched to a
/// fixture element by position. A face keeps its fixture smooth flag, a fixture sharp edge is
/// marked, and every face corner receives the stored custom normal of its fixture corner.
fn chain_operations(fixture: &Value, mesh: &Value) -> TestResult<Value> {
    let input = &fixture["input"];
    let vertices = input["vertices"]
        .as_array()
        .ok_or("fixture vertices")?
        .iter()
        .map(vec3)
        .collect::<TestResult<Vec<_>>>()?;
    let faces = input["faces"]
        .as_array()
        .ok_or("fixture faces")?
        .iter()
        .map(|face| {
            face.as_array()
                .ok_or("fixture face")?
                .iter()
                .map(|index| Ok(usize::try_from(index.as_u64().ok_or("index")?)?))
                .collect::<TestResult<Vec<_>>>()
        })
        .collect::<TestResult<Vec<_>>>()?;

    let sharp_pairs = input["sharp_edges"]
        .as_array()
        .ok_or("fixture sharp edges")?
        .iter()
        .map(|pair| Ok((index(&pair[0])?, index(&pair[1])?)))
        .collect::<TestResult<Vec<(usize, usize)>>>()?;

    // Potter vertex ID -> fixture vertex index, through the position shift to the fixture cube.
    let mut vertex_of = HashMap::new();
    for vertex in mesh["vertices"].as_array().ok_or("mesh vertices")? {
        let mut position = vec3(&vertex["co"])?;
        for coordinate in &mut position {
            *coordinate += BOX_TO_FIXTURE_OFFSET;
        }
        let id = vertex["id"].as_u64().ok_or("vertex id")?;
        vertex_of.insert(id, fixture_vertex(position, &vertices)?);
    }
    let all_vertices: Vec<String> = vertex_of.keys().map(|id| format!("v{id}")).collect();

    let mut smooth_ids = Vec::new();
    let mut custom_normals = serde_json::Map::new();
    let mut face_ids = Vec::new();
    for face in mesh["faces"].as_array().ok_or("mesh faces")? {
        let id = face["id"].as_u64().ok_or("face id")?;
        let loop_vertices = face["v"]
            .as_array()
            .ok_or("face vertices")?
            .iter()
            .map(|vertex| {
                let id = vertex.as_u64().ok_or("vertex")?;
                Ok(vertex_of
                    .get(&id)
                    .copied()
                    .ok_or("face vertex is not a box vertex")?)
            })
            .collect::<TestResult<Vec<usize>>>()?;
        let fixture_face = faces
            .iter()
            .position(|candidate| {
                let mut left = candidate.clone();
                let mut right = loop_vertices.clone();
                left.sort_unstable();
                right.sort_unstable();
                left == right
            })
            .ok_or("no fixture face has this vertex set")?;
        let fixture_corners = &faces[fixture_face];
        let corner_normals = loop_vertices
            .iter()
            .map(|vertex| {
                let corner = fixture_corners
                    .iter()
                    .position(|candidate| candidate == vertex)
                    .ok_or("fixture corner")?;
                Ok(json!(input["custom_normals"][fixture_face][corner]))
            })
            .collect::<TestResult<Vec<_>>>()?;
        custom_normals.insert(format!("f{id}"), Value::Array(corner_normals));
        face_ids.push(format!("f{id}"));
        if fixture["input"]["face_smooth"][fixture_face] == json!(true) {
            smooth_ids.push(format!("f{id}"));
        }
    }

    let mut sharp_ids = Vec::new();
    for edge in mesh["edges"].as_array().ok_or("mesh edges")? {
        let ends = [0, 1].map(|end| {
            edge["v"][end]
                .as_u64()
                .and_then(|id| vertex_of.get(&id).copied())
        });
        let [Some(a), Some(b)] = ends else {
            return Err("edge vertex is missing".into());
        };
        if sharp_pairs
            .iter()
            .any(|&(x, y)| (x, y) == (a, b) || (x, y) == (b, a))
        {
            sharp_ids.push(format!("e{}", edge["id"].as_u64().ok_or("edge id")?));
        }
    }

    let mut operations = vec![json!({
        "op": "mesh.transform_elements", "target": {"id": "cube"},
        "elements": {"domain": "vertex", "ids": all_vertices},
        "translation": [BOX_TO_FIXTURE_OFFSET, BOX_TO_FIXTURE_OFFSET, BOX_TO_FIXTURE_OFFSET]
    })];
    if !smooth_ids.is_empty() {
        operations.push(json!({
            "op": "mesh.shade_smooth", "target": {"id": "cube"},
            "elements": {"domain": "face", "ids": smooth_ids}
        }));
    }
    operations.push(json!({
        "op": "mesh.mark_sharp_by_angle", "target": {"id": "cube"},
        "elements": {"domain": "edge", "ids": sharp_ids}, "angle": 0.1
    }));
    operations.push(json!({
        "op": "mesh.set_custom_normals", "target": {"id": "cube"},
        "elements": {"domain": "face", "ids": face_ids}, "normals": custom_normals
    }));
    operations.push(json!({
        "op": "modifier.create", "target": {"id": "cube"}, "id": "mirror_x", "type": "mirror",
        "params": {"use_axis": [true, false, false]}
    }));
    operations.push(json!({
        "op": "modifier.create", "target": {"id": "cube"}, "id": "subdivision", "type": "subdivision",
        "params": {
            "levels": 1, "render_levels": 1, "subdivision_type": "CATMULL_CLARK",
            "use_custom_normals": true
        }
    }));
    Ok(Value::Array(operations))
}

/// Parsed GLB: the JSON chunk and the BIN chunk.
struct Glb {
    document: Value,
    binary: Vec<u8>,
}

fn parse_glb(bytes: &[u8]) -> TestResult<Glb> {
    if bytes.get(..4) != Some(b"glTF") {
        return Err("output is not a GLB".into());
    }
    let json_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    if u32::from_le_bytes(bytes[16..20].try_into()?) != 0x4E4F_534A {
        return Err("first GLB chunk is not JSON".into());
    }
    let document = serde_json::from_slice(&bytes[20..20 + json_length])?;
    let binary_header = 20 + json_length;
    let binary_length = usize::try_from(u32::from_le_bytes(
        bytes[binary_header..binary_header + 4].try_into()?,
    ))?;
    let start = binary_header + 8;
    Ok(Glb {
        document,
        binary: bytes[start..start + binary_length].to_vec(),
    })
}

/// Reads a float VEC3 accessor, honoring the buffer view offset and stride.
fn read_vec3(glb: &Glb, accessor: &Value) -> TestResult<Vec<[f64; 3]>> {
    let index = usize::try_from(accessor.as_u64().ok_or("accessor index")?)?;
    let accessor = &glb.document["accessors"][index];
    assert_eq!(accessor["componentType"], 5126, "accessor must be float32");
    assert_eq!(accessor["type"], "VEC3", "accessor must be VEC3");
    let count = usize::try_from(accessor["count"].as_u64().ok_or("count")?)?;
    let view = &glb.document["bufferViews"]
        [usize::try_from(accessor["bufferView"].as_u64().ok_or("bufferView")?)?];
    let base = usize::try_from(view["byteOffset"].as_u64().unwrap_or(0))?
        + usize::try_from(accessor["byteOffset"].as_u64().unwrap_or(0))?;
    let stride = usize::try_from(view["byteStride"].as_u64().unwrap_or(12))?;
    (0..count)
        .map(|vertex| {
            let at = base + vertex * stride;
            let mut value = [0.0_f64; 3];
            for (axis, slot) in value.iter_mut().enumerate() {
                let bytes = glb
                    .binary
                    .get(at + axis * 4..at + axis * 4 + 4)
                    .ok_or("accessor overruns BIN")?;
                *slot = f64::from(f32::from_le_bytes(bytes.try_into()?));
            }
            Ok(value)
        })
        .collect()
}

/// Reads an index accessor of any unsigned integer component type.
fn read_indices(glb: &Glb, accessor: &Value) -> TestResult<Vec<usize>> {
    let index = usize::try_from(accessor.as_u64().ok_or("indices accessor")?)?;
    let accessor = &glb.document["accessors"][index];
    let count = usize::try_from(accessor["count"].as_u64().ok_or("count")?)?;
    let view = &glb.document["bufferViews"]
        [usize::try_from(accessor["bufferView"].as_u64().ok_or("bufferView")?)?];
    let base = usize::try_from(view["byteOffset"].as_u64().unwrap_or(0))?
        + usize::try_from(accessor["byteOffset"].as_u64().unwrap_or(0))?;
    let width = match accessor["componentType"].as_u64() {
        Some(5121) => 1,
        Some(5123) => 2,
        Some(5125) => 4,
        other => return Err(format!("unsupported index component {other:?}").into()),
    };
    (0..count)
        .map(|position| {
            let at = base + position * width;
            let bytes = glb.binary.get(at..at + width).ok_or("index overruns BIN")?;
            let mut padded = [0_u8; 4];
            padded[..width].copy_from_slice(bytes);
            Ok(usize::try_from(u32::from_le_bytes(padded))?)
        })
        .collect()
}

/// Blender's output face: corner positions and per-corner normals, in loop order.
struct BlenderFace {
    corners: Vec<[f64; 3]>,
    normals: Vec<[f64; 3]>,
}

fn blender_faces(fixture: &Value) -> TestResult<Vec<BlenderFace>> {
    fixture["output"]["faces"]
        .as_array()
        .ok_or("output faces")?
        .iter()
        .map(|face| {
            Ok(BlenderFace {
                corners: face["corners"]
                    .as_array()
                    .ok_or("corners")?
                    .iter()
                    .map(vec3)
                    .collect::<TestResult<_>>()?,
                normals: face["normals"]
                    .as_array()
                    .ok_or("normals")?
                    .iter()
                    .map(vec3)
                    .collect::<TestResult<_>>()?,
            })
        })
        .collect()
}

fn same_position(left: [f64; 3], right: [f64; 3]) -> bool {
    (0..3).all(|axis| (left[axis] - right[axis]).abs() <= POSITION_TOLERANCE)
}

#[test]
fn chain_mirror_subsurf_glb_normals_match_blender() -> TestResult {
    let fixture = fixture()?;
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply(
        &scene,
        &json!([{"op": "node.create", "id": "cube", "kind": "box", "params": {"size": 1.0}}]),
    )?;
    let operations = chain_operations(&fixture, &project_mesh(&scene)?)?;
    apply(&scene, &operations)?;

    let output = directory.path().join("chain.glb");
    let result = export_strict(&scene, "glb", &output)?;
    assert!(
        result.status.success(),
        "pot export glb refused: {}",
        String::from_utf8_lossy(&result.stdout)
    );
    let glb = parse_glb(&fs::read(&output)?)?;
    let primitive = &glb.document["meshes"][0]["primitives"][0];
    // glTF is Y-up. Potter's `axis_from_gltf` maps (x, y, z) to (x, -z, y); the fixture is Potter's.
    let to_potter = |point: [f64; 3]| [point[0], -point[2], point[1]];
    let positions: Vec<[f64; 3]> = read_vec3(&glb, &primitive["attributes"]["POSITION"])?
        .into_iter()
        .map(to_potter)
        .collect();
    let normals: Vec<[f64; 3]> = read_vec3(&glb, &primitive["attributes"]["NORMAL"])?
        .into_iter()
        .map(to_potter)
        .collect();
    let indices = match primitive.get("indices") {
        Some(accessor) => read_indices(&glb, accessor)?,
        None => (0..positions.len()).collect(),
    };
    assert_eq!(
        indices.len() % 3,
        0,
        "glTF primitive is not a triangle list"
    );

    let faces = blender_faces(&fixture)?;
    let corner_total: usize = faces.iter().map(|face| face.corners.len()).sum();
    assert_eq!(corner_total, BLENDER_CORNERS, "fixture corner count");

    // For each Blender corner, the normals of glTF triangle corners it covers.
    let mut covered: Vec<Vec<Vec<[f64; 3]>>> = faces
        .iter()
        .map(|face| vec![Vec::new(); face.corners.len()])
        .collect();
    let mut orphan_triangles = 0_usize;
    for triangle in indices.chunks(3) {
        let points: Vec<[f64; 3]> = triangle.iter().map(|&index| positions[index]).collect();
        let triangle_normals: Vec<[f64; 3]> =
            triangle.iter().map(|&index| normals[index]).collect();
        let owners: Vec<usize> = faces
            .iter()
            .enumerate()
            .filter(|(_, face)| {
                points.iter().all(|point| {
                    face.corners
                        .iter()
                        .any(|corner| same_position(*corner, *point))
                })
            })
            .map(|(index, _)| index)
            .collect();
        let [owner] = owners[..] else {
            orphan_triangles += 1;
            continue;
        };
        for (point, normal) in points.iter().zip(triangle_normals) {
            let corner = faces[owner]
                .corners
                .iter()
                .position(|corner| same_position(*corner, *point))
                .ok_or("owner corner")?;
            covered[owner][corner].push(normal);
        }
    }
    let mut compared_faces = Vec::new();
    let mut compared = 0_usize;
    let mut worst = 0.0_f64;
    for (index, (face, face_covered)) in faces.iter().zip(&covered).enumerate() {
        let mut face_compared = false;
        for (expected, observed) in face.normals.iter().zip(face_covered) {
            if observed.is_empty() {
                continue;
            }
            compared += 1;
            face_compared = true;
            for normal in observed {
                for axis in 0..3 {
                    worst = worst.max((normal[axis] - expected[axis]).abs());
                }
            }
        }
        if face_compared {
            compared_faces.push(index);
        }
    }
    eprintln!(
        "chain_mirror_subsurf: compared {compared} of {corner_total} Blender corners on faces \
         {compared_faces:?}, max normal component error {worst:.3e}, orphan glTF triangles \
         {orphan_triangles}"
    );
    assert_eq!(
        compared_faces.len(),
        faces.len(),
        "every Blender face must have glTF triangles"
    );
    assert_eq!(orphan_triangles, 0, "glTF triangles without a Blender face");
    assert_eq!(
        compared, BLENDER_CORNERS,
        "every Blender corner must have glTF triangle corners"
    );
    assert!(
        worst <= BLENDER_NORMAL_TOLERANCE,
        "glTF NORMAL differs from Blender by {worst:e}"
    );
    Ok(())
}
