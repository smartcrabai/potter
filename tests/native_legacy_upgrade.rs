//! PR9 legacy upgrade chain (1 → 2 → 3) on real project files.
//!
//! * `schema_v1_fixture`: the committed schema-1 project upgrades on apply, with native shading
//!   keys and no retired keys afterwards.
//! * `schema_v2_scene`: a schema-2 scene written by hand from the Blender oracle's explicit
//!   numbers. It uses the old keys: `blender_polygon_smooth`, `blender_edge_flags`, colors and
//!   generic attributes inside `blender_attributes`, a paint color `color` keyed `f<face>:v<vertex>`,
//!   and `custom_normal` rows of explicit vectors. Its history record is written by the test and
//!   its digest is verified before upgrade. Load, apply, undo and redo must keep the same semantic
//!   data as a native import of the same oracle, and must keep the authored normals.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use native_oracle::{
    FLOAT_TOLERANCE, StoredCorner, TestResult, apply_operations, attribute, build_oracle,
    import_blend, mesh_of_node, node_id, normal_code_rows, pot, pot_json, read_scene, value_at,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

/// The legacy scene's authored normals: every corner of every face carries its fan's vector, so the
/// vectors agree within each smooth fan (see `fan_roots`).
/// The one legacy corner whose vector is a literal zero; it must upgrade to `explicit_zero`. It lies on
/// flat face 0, a fan of its own, so the zero agrees with no neighbour.
const LEGACY_EXPLICIT_ZERO: (usize, usize) = (0, 0);

fn fixture_v1() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/schema_v1_project")
}

fn copy_tree(from: &Path, to: &Path) -> TestResult {
    if from.is_dir() {
        fs::create_dir_all(to)?;
        for entry in fs::read_dir(from)? {
            let entry = entry?;
            copy_tree(&entry.path(), &to.join(entry.file_name()))?;
        }
    } else {
        fs::copy(from, to)?;
    }
    Ok(())
}

fn read_json(path: &Path) -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn revision_of(project: &Path) -> TestResult<u64> {
    let inspected = pot_json(|command| {
        command.arg("inspect").arg(project);
    })?;
    inspected["scene"]["revision"]
        .as_u64()
        .ok_or_else(|| "scene revision is missing".into())
}

fn history(project: &Path, direction: &str, base_revision: u64) -> TestResult<Value> {
    pot_json(|command| {
        command
            .arg(direction)
            .arg(project)
            .arg("--base-revision")
            .arg(base_revision.to_string());
    })
}

/// A JSON-tolerant comparison: numbers within `FLOAT_TOLERANCE`, everything else exactly.
fn json_close(actual: &Value, expected: &Value, label: &str) {
    match (actual, expected) {
        (Value::Number(_), Value::Number(_)) => {
            let a = actual.as_f64().unwrap_or(f64::NAN);
            let e = expected.as_f64().unwrap_or(f64::NAN);
            assert!(
                (a - e).abs() <= FLOAT_TOLERANCE,
                "{label}: {a} differs from {e}"
            );
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{label}: length");
            for (index, (a, e)) in a.iter().zip(e).enumerate() {
                json_close(a, e, &format!("{label}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            let a_keys: BTreeSet<&String> = a.keys().collect();
            let e_keys: BTreeSet<&String> = e.keys().collect();
            assert_eq!(a_keys, e_keys, "{label}: keys");
            for (key, e) in e {
                json_close(&a[key], e, &format!("{label}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{label}"),
    }
}

/// Blender's loop-normal fans of the oracle: one root per corner, indexed `starts[face] + corner` with
/// the corners in `faces` order. Corners join across every edge that is not sharp when both faces are
/// smooth; a flat face's corners stand alone. Blender v5.2.2 `source/blender/blenkernel/intern/
/// mesh_normals.cc`: the custom-normal setter marks an edge sharp where one fan's vectors disagree
/// (`LNOR_SPACE_TRIGO_THRESHOLD`, line 591; the in-fan comparison and marking at lines 1505-1515; the
/// closing comparison at lines 1532-1538). The authored vectors must therefore agree within each fan,
/// or the upgrade adds sharp edges the document never stored.
fn fan_roots(expected: &Value) -> TestResult<Vec<usize>> {
    let oracle = &expected["oracle"];
    let faces = oracle["faces"].as_array().ok_or("faces are missing")?;
    let mut sharp = BTreeMap::new();
    for edge in oracle["edges"].as_array().ok_or("edges are missing")? {
        sharp.insert(
            vertex_pair(&edge["v"])?,
            edge["sharp"].as_bool() == Some(true),
        );
    }
    let mut starts = Vec::with_capacity(faces.len());
    let mut total = 0;
    for face in faces {
        starts.push(total);
        total += vertex_list(&face["v"])?.len();
    }
    // The faces using each undirected edge.
    let mut users: BTreeMap<(u64, u64), Vec<usize>> = BTreeMap::new();
    for (index, face) in faces.iter().enumerate() {
        let vertices = vertex_list(&face["v"])?;
        for position in 0..vertices.len() {
            let next = vertices[(position + 1) % vertices.len()];
            users
                .entry(edge_key(vertices[position], next))
                .or_default()
                .push(index);
        }
    }
    let corner_at = |face: usize, vertex: u64| -> TestResult<usize> {
        let vertices = vertex_list(&faces[face]["v"])?;
        let position = vertices
            .iter()
            .position(|&candidate| candidate == vertex)
            .ok_or("face does not use the vertex")?;
        Ok(starts[face] + position)
    };
    let mut parent: Vec<usize> = (0..total).collect();
    for (key, using) in &users {
        if sharp.get(key).copied().unwrap_or(false) || using.len() != 2 {
            continue;
        }
        if !using
            .iter()
            .all(|&face| faces[face]["smooth"].as_bool() == Some(true))
        {
            continue;
        }
        for vertex in [key.0, key.1] {
            let first = find(&mut parent, corner_at(using[0], vertex)?);
            let second = find(&mut parent, corner_at(using[1], vertex)?);
            parent[first] = second;
        }
    }
    Ok((0..total).map(|corner| find(&mut parent, corner)).collect())
}

fn find(parent: &mut [usize], mut corner: usize) -> usize {
    while parent[corner] != corner {
        parent[corner] = parent[parent[corner]];
        corner = parent[corner];
    }
    corner
}

fn vertex_list(value: &Value) -> TestResult<Vec<u64>> {
    let items = value.as_array().ok_or("face vertices are missing")?;
    Ok(items
        .iter()
        .map(|item| item.as_u64().ok_or("vertex ID is not a number"))
        .collect::<Result<Vec<u64>, &str>>()?)
}

fn vertex_pair(value: &Value) -> TestResult<(u64, u64)> {
    match vertex_list(value)?.as_slice() {
        [first, second] => Ok(edge_key(*first, *second)),
        _ => Err("edge is not a vertex pair".into()),
    }
}

fn edge_key(first: u64, second: u64) -> (u64, u64) {
    (first.min(second), first.max(second))
}

/// The authored vector of the fan with root `fan`: a unit vector, distinct for each fan.
fn fan_vector(fan: usize) -> [f64; 3] {
    let index = fan as f64;
    let raw = [0.3 + 0.05 * (index % 5.0), -0.2 + 0.03 * (index % 7.0), 0.9];
    let length = raw.iter().map(|c| c * c).sum::<f64>().sqrt();
    [raw[0] / length, raw[1] / length, raw[2] / length]
}

/// The legacy vector of one corner: its fan's vector, or a literal zero at `LEGACY_EXPLICIT_ZERO`.
fn legacy_vector(face: usize, corner: usize, fan: usize) -> [f64; 3] {
    if (face, corner) == LEGACY_EXPLICIT_ZERO {
        return [0.0; 3];
    }
    fan_vector(fan)
}

fn paint_value(face: usize, corner: usize) -> [f64; 4] {
    let step = (face * 3 + corner) % 9;
    [
        0.1 + 0.01 * step as f64,
        0.4 - 0.02 * step as f64,
        0.25 + 0.03 * step as f64,
        1.0 - 0.05 * step as f64,
    ]
}

/// The schema-2 mesh of the oracle's `Oracle` object, written from its explicit numbers.
fn legacy_mesh(expected: &Value) -> TestResult<Value> {
    let oracle = &expected["oracle"];
    let vertices = oracle["vertices"]
        .as_array()
        .ok_or("vertices are missing")?;
    let edges = oracle["edges"].as_array().ok_or("edges are missing")?;
    let faces = oracle["faces"].as_array().ok_or("faces are missing")?;
    let face_sizes: Vec<usize> = faces
        .iter()
        .map(|face| face["v"].as_array().map_or(0, Vec::len))
        .collect();
    let mut starts = Vec::new();
    let mut next = 0;
    for size in &face_sizes {
        starts.push(next);
        next += size;
    }

    let mut blender_attributes = Vec::new();
    let color_order = oracle["color_layers"]["order"]
        .as_array()
        .ok_or("color order is missing")?;
    for name in color_order {
        let name = name.as_str().ok_or("color name is not a string")?;
        let layer = &oracle["color_layers"]["layers"][name];
        blender_attributes.push(json!({
            "name": name,
            "domain": layer["domain"].as_str().unwrap_or_default().to_uppercase(),
            "data_type": layer["source"],
            "values": layer["values"],
        }));
    }
    for (name, spec) in oracle["generic"].as_object().ok_or("generic is missing")? {
        let data_type = match spec["type"].as_str().unwrap_or_default() {
            "float" => "FLOAT",
            "int" => "INT",
            "bool" => "BOOLEAN",
            _ => "FLOAT_VECTOR",
        };
        blender_attributes.push(json!({
            "name": name,
            "domain": spec["domain"].as_str().unwrap_or_default().to_uppercase(),
            "data_type": data_type,
            "values": spec["values"],
        }));
    }

    let mut uv = Map::new();
    let uv_names = oracle["uv_layers"]["names"]
        .as_array()
        .ok_or("UV names are missing")?;
    let mut uv_layers = Vec::new();
    for name in uv_names {
        let name = name.as_str().ok_or("UV name is not a string")?;
        let values = oracle["uv_layers"]["values"][name]
            .as_array()
            .ok_or("UV values are missing")?;
        let mut stored_faces = Vec::new();
        for (face, start) in starts.iter().enumerate() {
            let uv_values: Vec<Value> = (0..face_sizes[face])
                .map(|corner| {
                    let source = &values[start + corner];
                    json!([source[0], 1.0 - source[1].as_f64().unwrap_or_default()])
                })
                .collect();
            stored_faces.push(json!({
                "face_id": face,
                "uv": uv_values,
                "pinned": [false, false, false, false],
            }));
        }
        uv_layers.push(json!({"name": name, "faces": stored_faces}));
    }
    uv.insert("layers".to_owned(), Value::Array(uv_layers));
    uv.insert("active".to_owned(), oracle["uv_layers"]["active"].clone());
    uv.insert("render".to_owned(), oracle["uv_layers"]["render"].clone());

    let roots = fan_roots(expected)?;
    let mut normal_rows = Map::new();
    for (face, &face_size) in face_sizes.iter().enumerate() {
        let row: Vec<Value> = (0..face_size)
            .map(|corner| json!(legacy_vector(face, corner, roots[starts[face] + corner])))
            .collect();
        normal_rows.insert(format!("f{face}"), Value::Array(row));
    }

    let mut paint = Map::new();
    for (face, size) in face_sizes.iter().enumerate() {
        let vertices = faces[face]["v"]
            .as_array()
            .ok_or("face vertices are missing")?;
        for (corner, vertex) in vertices.iter().take(*size).enumerate() {
            let vertex = vertex.as_u64().ok_or("face vertex is missing")?;
            paint.insert(
                format!("f{face}:v{vertex}"),
                json!(paint_value(face, corner)),
            );
        }
    }

    Ok(json!({
        "vertices": vertices
            .iter()
            .enumerate()
            .map(|(id, vertex)| json!({"id": id, "co": vertex}))
            .collect::<Vec<_>>(),
        "edges": edges
            .iter()
            .enumerate()
            .map(|(id, edge)| json!({"id": id, "v": edge["v"]}))
            .collect::<Vec<_>>(),
        "faces": faces
            .iter()
            .enumerate()
            .map(|(id, face)| json!({"id": id, "v": face["v"], "material_index": 0}))
            .collect::<Vec<_>>(),
        "next_id": {"vertex": vertices.len(), "edge": edges.len(), "face": faces.len()},
        "attributes": {
            "blender_polygon_smooth": faces.iter().map(|face| face["smooth"].clone()).collect::<Vec<_>>(),
            "blender_edge_flags": edges
                .iter()
                .map(|edge| json!({"sharp": edge["sharp"], "seam": edge["seam"]}))
                .collect::<Vec<_>>(),
            "blender_attributes": blender_attributes,
            "uv_layers": Value::Object(uv),
            "custom_normal": {"domain": "corner", "type": "float3", "values": normal_rows},
            "color": {"domain": "corners", "values": paint},
        },
    }))
}

/// Builds the schema-2 project at `root/legacy`: the committed v1 fixture's scene skeleton with
/// one mesh node, and a history record whose snapshot is the legacy document.
fn write_legacy_project(root: &Path, expected: &Value) -> TestResult<PathBuf> {
    let project = root.join("legacy");
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    let mut doc = read_json(&fixture_v1().join("scene.json"))?;
    let mesh = legacy_mesh(expected)?;
    let mut node = doc["nodes"]["cube"].clone();
    node["name"] = json!("Oracle");
    node["data"] = json!("oracle_mesh");
    node["materials"] = json!([]);
    doc["schema_version"] = json!(2);
    doc["revision"] = json!(1);
    doc["nodes"] = json!({"oracle": node});
    doc["data_blocks"] = json!({"oracle_mesh": {"type": "mesh", "descriptor": null, "mesh": mesh}});
    doc["collections"]["collection_root"]["objects"] = json!(["oracle"]);
    doc["materials"] = json!({});
    doc["images"] = json!({});
    doc["node_groups"] = json!({});

    let mut snapshot = doc.clone();
    snapshot["history"] = json!({"head": null, "undo": [], "redo": []});
    let record = json!({
        "record_version": 1,
        "parent": null,
        "kind": "init",
        "revision": 1,
        "scene_hash": format!("sha256:{}", hex::encode(Sha256::digest(serde_json::to_vec(&snapshot)?))),
        "operations": [],
        "changes": {},
        "target": null,
        "snapshot": snapshot,
    });
    let record_bytes = serde_json::to_vec(&record)?;
    let digest = hex::encode(Sha256::digest(&record_bytes));
    let blob_dir = project.join("history/sha256").join(&digest);
    fs::create_dir_all(&blob_dir)?;
    fs::write(blob_dir.join("blob.json"), &record_bytes)?;
    doc["history"] = json!({"head": format!("sha256:{digest}"), "undo": [], "redo": []});
    fs::write(project.join("scene.json"), serde_json::to_vec_pretty(&doc)?)?;
    Ok(project)
}

/// The mesh of the legacy node with ID `oracle`. The apply renames the node, so it is read by ID.
fn stored_legacy_mesh(project: &Path) -> TestResult<Value> {
    let scene = read_scene(project)?;
    let data = scene["nodes"]["oracle"]["data"]
        .as_str()
        .ok_or("legacy node has no data block")?;
    Ok(scene["data_blocks"][data]["mesh"].clone())
}

/// The storage position of each persistent ID in one element list (`vertices`, `edges`, or `faces`).
fn id_positions(mesh: &Value, list: &str) -> TestResult<BTreeMap<u64, usize>> {
    mesh[list]
        .as_array()
        .ok_or_else(|| format!("mesh {list} are missing"))?
        .iter()
        .enumerate()
        .map(|(index, element)| -> TestResult<(u64, usize)> {
            let id = element["id"].as_u64().ok_or("mesh element has no id")?;
            Ok((id, index))
        })
        .collect()
}

/// An attribute with its per-element values re-keyed by storage position: `f<id>` faces, `e<id>`
/// edges, and `v<id>` vertices become `f<index>`, `e<index>`, and `v<index>`. The legacy document and
/// the native `.blend` import number their elements differently (the importer is 1-based), so the two
/// meshes are compared element by element, in the order the oracle lists them.
fn positional_attribute(mesh: &Value, name: &str) -> TestResult<Value> {
    let attribute = attribute(mesh, name)?;
    let Some(values) = attribute["values"].as_object() else {
        return Ok(attribute.clone());
    };
    let faces = id_positions(mesh, "faces")?;
    let edges = id_positions(mesh, "edges")?;
    let vertices = id_positions(mesh, "vertices")?;
    let mut keyed = serde_json::Map::new();
    for (key, value) in values {
        let (prefix, table, id) = if let Some(id) = key.strip_prefix('f') {
            ('f', &faces, id)
        } else if let Some(id) = key.strip_prefix('e') {
            ('e', &edges, id)
        } else if let Some(id) = key.strip_prefix('v') {
            ('v', &vertices, id)
        } else {
            return Err(format!("{name} key {key} is not an element ID").into());
        };
        let id: u64 = id
            .parse()
            .map_err(|_| format!("{name} key {key} is not an element ID"))?;
        let index = table
            .get(&id)
            .ok_or_else(|| format!("{name} key {key} names no element"))?;
        keyed.insert(format!("{prefix}{index}"), value.clone());
    }
    let mut positional = attribute.clone();
    positional["values"] = Value::Object(keyed);
    Ok(positional)
}

/// Checks the native store of a legacy-derived mesh against a native import of the oracle.
fn assert_native_semantics(legacy: &Value, native: &Value) -> TestResult {
    for name in ["shade_smooth", "sharp", "seam"] {
        json_close(
            &positional_attribute(legacy, name)?,
            &positional_attribute(native, name)?,
            name,
        );
    }
    assert!(
        legacy["attributes"].get("blender_polygon_smooth").is_none(),
        "blender_polygon_smooth must be upgraded away"
    );
    assert!(
        legacy["attributes"].get("blender_edge_flags").is_none(),
        "blender_edge_flags must be upgraded away"
    );
    let legacy_store = attribute(legacy, "color_layers")?;
    let native_store = attribute(native, "color_layers")?;
    let native_layers = native_store["layers"]
        .as_array()
        .ok_or("native color layers are missing")?;
    let legacy_layers = legacy_store["layers"]
        .as_array()
        .ok_or("legacy color layers are missing")?;
    // The legacy paint layer `color` is registered after the Blender layers.
    assert_eq!(
        legacy_layers.get(..native_layers.len()),
        Some(native_layers.as_slice()),
        "legacy color layers keep the Blender order"
    );
    for layer in native_layers {
        let name = layer.as_str().ok_or("layer name is not a string")?;
        json_close(
            &positional_attribute(legacy, name)?,
            &positional_attribute(native, name)?,
            name,
        );
    }
    // Only the Blender layers carry a source encoding; the legacy paint layer has none.
    let legacy_encodings = &attribute(legacy, "blender_metadata")?["color_data_types"];
    let native_encodings = &attribute(native, "blender_metadata")?["color_data_types"];
    for layer in native_layers {
        let name = layer.as_str().ok_or("layer name is not a string")?;
        json_close(&legacy_encodings[name], &native_encodings[name], name);
    }
    Ok(())
}

/// The upgraded store must be a `normal_code` layer: every corner holds an authored code pair, and the
/// literal-zero legacy vector holds `explicit_zero`. An Auto (`null`) corner would mean an authored
/// normal was lost on upgrade.
fn assert_upgraded_normals(mesh: &Value) -> TestResult {
    let rows = normal_code_rows(mesh)?;
    for (face, row) in rows.iter().enumerate() {
        for (corner, stored) in row.iter().enumerate() {
            if (face, corner) == LEGACY_EXPLICIT_ZERO {
                assert_eq!(
                    *stored,
                    StoredCorner::ExplicitZero,
                    "face {face} corner {corner} lost its explicit zero"
                );
            } else {
                assert!(
                    matches!(stored, StoredCorner::Code(_)),
                    "face {face} corner {corner} lost its authored normal: {stored:?}"
                );
            }
        }
    }
    Ok(())
}

fn assert_paint_layer(mesh: &Value, expected: &Value) -> TestResult {
    let faces = expected["oracle"]["faces"]
        .as_array()
        .ok_or("faces are missing")?;
    let layer = attribute(mesh, "color")?;
    assert_eq!(layer["domain"], "corner", "paint layer domain");
    for (face, record) in faces.iter().enumerate() {
        let corners = record["v"].as_array().map_or(0, Vec::len);
        let row = value_at(layer, &format!("f{face}"))?;
        for corner in 0..corners {
            let wanted = paint_value(face, corner);
            for (channel, value) in row[corner]
                .as_array()
                .ok_or("paint value is not an array")?
                .iter()
                .enumerate()
            {
                assert!(
                    (value.as_f64().unwrap_or(f64::NAN) - wanted[channel]).abs() <= 1.0e-12,
                    "paint face {face} corner {corner} channel {channel}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn schema_v1_fixture_upgrades_on_apply_without_retired_keys() -> TestResult {
    let directory = TempDir::new()?;
    let scene = directory.path().join("scene");
    copy_tree(&fixture_v1(), &scene)?;
    fs::create_dir_all(scene.join(".potter/tmp"))?;
    let stored = fs::read(scene.join("scene.json"))?;
    let inspected = pot_json(|command| {
        command.arg("inspect").arg(&scene);
    })?;
    assert_eq!(inspected["ok"], true, "{inspected}");
    assert_eq!(
        fs::read(scene.join("scene.json"))?,
        stored,
        "opening a schema 1 project must not rewrite it"
    );
    let node = node_id(&read_scene(&scene)?, "Cube")?;
    let applied = apply_operations(
        &scene,
        directory.path(),
        json!([{"op": "node.update", "target": {"id": node}, "set": {"name": "Cube renamed"}}]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let document = read_json(&scene.join("scene.json"))?;
    assert_eq!(
        document["schema_version"], 6,
        "apply writes the current schema"
    );
    let mesh = stored_oracle_or_cube(&document)?;
    for retired in ["blender_polygon_smooth", "blender_edge_flags"] {
        assert!(
            mesh["attributes"].get(retired).is_none(),
            "{retired} survived the upgrade"
        );
    }
    assert!(
        mesh["attributes"].get("uv_layers").is_some(),
        "UV layers were not upgraded"
    );
    Ok(())
}

fn stored_oracle_or_cube(document: &Value) -> TestResult<Value> {
    document["data_blocks"]
        .as_object()
        .ok_or("data blocks are missing")?
        .values()
        .find(|block| block["mesh"].is_object())
        .map(|block| block["mesh"].clone())
        .ok_or_else(|| "the project has no mesh".into())
}

/// The stored mesh of the imported `Oracle` node. The scene holds other meshes (flat ones among
/// them, which carry no `shade_smooth` key), so the node is looked up by name.
fn stored_oracle(project: &Path) -> TestResult<Value> {
    Ok(mesh_of_node(&read_scene(project)?, "Oracle")?.clone())
}

#[test]
fn schema_v2_scene_keeps_native_semantics_through_apply_undo_and_redo() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = TempDir::new()?;
    let (blend, expected) = build_oracle(&blender, root.path())?;
    let (native_project, _) = import_blend(&blender, root.path(), "native", &blend)?;
    let native = stored_oracle(&native_project)?;

    let legacy_root = root.path().join("legacy-root");
    fs::create_dir_all(&legacy_root)?;
    let legacy = write_legacy_project(&legacy_root, &expected)?;
    let before = fs::read(legacy.join("scene.json"))?;
    let inspected = pot_json(|command| {
        command.arg("inspect").arg(&legacy);
    })?;
    assert_eq!(inspected["ok"], true, "{inspected}");
    assert_eq!(
        fs::read(legacy.join("scene.json"))?,
        before,
        "opening a schema 2 scene must not rewrite it"
    );

    let applied = apply_operations(
        &legacy,
        &legacy_root,
        json!([{"op": "node.update", "target": {"id": node_id(&read_scene(&legacy)?, "Oracle")?}, "set": {"name": "Legacy Oracle"}}]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    assert_eq!(read_json(&legacy.join("scene.json"))?["schema_version"], 6);
    let upgraded = stored_legacy_mesh(&legacy)?;
    assert_native_semantics(&upgraded, &native)?;
    assert_upgraded_normals(&upgraded)?;
    assert_paint_layer(&upgraded, &expected)?;

    let revision = revision_of(&legacy)?;
    let undone = history(&legacy, "undo", revision)?;
    assert_eq!(undone["ok"], true, "{undone}");
    let after_undo = stored_legacy_mesh(&legacy)?;
    assert_native_semantics(&after_undo, &native)?;
    assert_upgraded_normals(&after_undo)?;
    assert_paint_layer(&after_undo, &expected)?;

    let revision = revision_of(&legacy)?;
    let redone = history(&legacy, "redo", revision)?;
    assert_eq!(redone["ok"], true, "{redone}");
    let after_redo = stored_legacy_mesh(&legacy)?;
    assert_native_semantics(&after_redo, &native)?;
    assert_upgraded_normals(&after_redo)?;
    assert_paint_layer(&after_redo, &expected)?;
    Ok(())
}

#[test]
fn schema_v2_history_record_is_verified_before_it_is_upgraded() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = TempDir::new()?;
    let (_, expected) = build_oracle(&blender, root.path())?;
    let legacy = write_legacy_project(root.path(), &expected)?;
    // Apply once so the legacy init record is an undo target. `undo` must read and digest-check
    // this record before it can parse or upgrade its schema-2 snapshot.
    let applied = apply_operations(
        &legacy,
        root.path(),
        json!([{
            "op": "node.update",
            "target": {"id": "oracle"},
            "set": {"name": "Applied Oracle"}
        }]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let scene = read_json(&legacy.join("scene.json"))?;
    let revision = scene["revision"].as_u64().ok_or("revision is missing")?;
    let undo_target = scene["history"]["undo"]
        .as_array()
        .and_then(|undo| undo.last())
        .and_then(Value::as_str)
        .ok_or("apply did not add an undo record")?;
    let digest = undo_target
        .strip_prefix("sha256:")
        .ok_or("undo target is not a sha256 digest")?;
    let blob = legacy.join("history/sha256").join(digest).join("blob.json");
    let mut bytes = fs::read(&blob)?;
    bytes.push(b'\n');
    fs::write(&blob, bytes)?;
    let output = pot()
        .arg("undo")
        .arg(&legacy)
        .arg("--base-revision")
        .arg(revision.to_string())
        .arg("--json")
        .output()?;
    let refused: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(refused["ok"], false, "{refused}");
    assert_eq!(refused["error"]["code"], "INTERNAL_ERROR", "{refused}");
    Ok(())
}
