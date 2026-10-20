#![expect(clippy::unwrap_used, reason = "integration test fixtures")]

use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn json(output: &std::process::Output) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn make_box(scene: &Path) -> Result<(), Box<dyn Error>> {
    let init = pot()
        .args(["init", scene.to_str().unwrap(), "--json"])
        .output()?;
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stdout)
    );
    let operations = scene.with_extension("operations.json");
    fs::write(&operations, include_str!("fixtures/first.json"))?;
    let applied = pot()
        .args(["apply", scene.to_str().unwrap(), "--file"])
        .arg(&operations)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    Ok(())
}

fn export(scene: &Path, format: &str, file: &Path) -> Result<Value, Box<dyn Error>> {
    let mut command = pot();
    command
        .args([
            "export",
            scene.to_str().unwrap(),
            "--format",
            format,
            "--out",
        ])
        .arg(file);
    if matches!(format, "obj" | "stl" | "ply" | "pdf" | "svg") {
        command.arg("--allow-lossy");
    }
    let output = command.arg("--json").output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    json(&output)
}

fn import_replace(scene: &Path, file: &Path, format: &str) -> Result<Value, Box<dyn Error>> {
    let output = pot()
        .args(["import", scene.to_str().unwrap(), "--file"])
        .arg(file)
        .args([
            "--format",
            format,
            "--base-revision",
            "0",
            "--mode",
            "replace",
            "--json",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    json(&output)
}

fn bounds(scene: &Path) -> Result<Value, Box<dyn Error>> {
    let output = pot()
        .args(["inspect", scene.to_str().unwrap(), "--json"])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let response = json(&output)?;
    response["result"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| !item["bounds"].is_null()))
        .map(|item| item["bounds"].clone())
        .ok_or_else(|| "scene has no evaluated bounds".into())
}

fn assert_box_bounds(scene: &Path) -> Result<(), Box<dyn Error>> {
    let actual = bounds(scene)?;
    let expected = [[-0.5, -0.3, 0.0], [0.5, 0.3, 0.8]];
    for (index, key) in ["min", "max"].into_iter().enumerate() {
        for axis in 0..3 {
            let value = actual[key][axis]
                .as_f64()
                .ok_or("missing bound component")?;
            assert!(
                (value - expected[index][axis]).abs() <= 1.0e-6,
                "{key}[{axis}] = {value}"
            );
        }
    }
    Ok(())
}

fn vertex_count(scene: &Path) -> Result<usize, Box<dyn Error>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    Ok(document["data_blocks"]
        .as_object()
        .ok_or("missing data_blocks")?
        .values()
        .filter_map(|block| block["mesh"]["vertices"].as_array())
        .map(Vec::len)
        .sum())
}

#[test]
fn glb_structure_and_box_round_trip_preserve_bounds() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let source = directory.path().join("source");
    let imported = directory.path().join("imported-glb");
    let output_path = directory.path().join("box.glb");
    make_box(&source)?;
    let report = export(&source, "glb", &output_path)?;
    let bytes = fs::read(&output_path)?;
    assert_eq!(&bytes[..4], b"glTF");
    assert_eq!(bytes.len() % 4, 0);
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into()?),
        u32::try_from(bytes.len())?
    );
    let json_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    assert_eq!(json_length % 4, 0);
    assert_eq!(u32::from_le_bytes(bytes[16..20].try_into()?), 0x4E4F_534A);
    let binary_header = 20 + json_length;
    let binary_length = usize::try_from(u32::from_le_bytes(
        bytes[binary_header..binary_header + 4].try_into()?,
    ))?;
    assert_eq!(&bytes[binary_header + 4..binary_header + 8], b"BIN\0");
    assert_eq!(binary_length % 4, 0);
    assert_eq!(binary_header + 8 + binary_length, bytes.len());
    let document: Value = serde_json::from_slice(&bytes[20..20 + json_length])?;
    let position = &document["accessors"][0];
    assert_eq!(position["componentType"], 5126);
    assert_eq!(position["type"], "VEC3");
    assert!(position["min"].as_array().is_some());
    assert!(position["max"].as_array().is_some());
    let minimum = position["min"]
        .as_array()
        .ok_or("accessor min is missing")?;
    let maximum = position["max"]
        .as_array()
        .ok_or("accessor max is missing")?;
    for (axis, expected) in [-0.5, -0.5, -0.5].into_iter().enumerate() {
        assert!((minimum[axis].as_f64().ok_or("invalid accessor min")? - expected).abs() <= 1.0e-6);
    }
    for (axis, expected) in [0.5, 0.5, 0.5].into_iter().enumerate() {
        assert!((maximum[axis].as_f64().ok_or("invalid accessor max")? - expected).abs() <= 1.0e-6);
    }
    for view in document["bufferViews"]
        .as_array()
        .ok_or("missing bufferViews")?
    {
        assert_eq!(view["byteOffset"].as_u64().unwrap_or(0) % 4, 0);
    }
    assert_eq!(report["result"]["format"], "glb");
    pot()
        .args(["init", imported.to_str().unwrap(), "--json"])
        .output()?;
    let report = import_replace(&imported, &output_path, "glb")?;
    assert_eq!(report["result"]["committed"], true);
    assert_eq!(report["result"]["candidate_revision"], 1);
    let source_blob = report["result"]["resources"][0]["path"]
        .as_str()
        .ok_or("missing source compatibility blob path")?;
    assert!(imported.join(source_blob).is_file());
    assert_box_bounds(&imported)?;
    Ok(())
}

#[test]
fn gltf_json_uses_bin_sidecar_and_round_trips_box() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let source = directory.path().join("source");
    let imported = directory.path().join("imported");
    let gltf = directory.path().join("box.gltf");
    make_box(&source)?;
    let report = export(&source, "gltf", &gltf)?;
    let sidecar = directory.path().join("box.bin");
    assert!(sidecar.is_file());
    assert!(
        report["result"]["files"]
            .as_array()
            .is_some_and(|files| files.len() == 2)
    );
    let document: Value = serde_json::from_slice(&fs::read(&gltf)?)?;
    assert_eq!(document["buffers"][0]["uri"], "box.bin");
    pot()
        .args(["init", imported.to_str().unwrap(), "--json"])
        .output()?;
    import_replace(&imported, &gltf, "gltf")?;
    assert_box_bounds(&imported)?;
    Ok(())
}

#[test]
fn obj_stl_and_ply_round_trips_preserve_vertex_counts_and_bounds() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let source = directory.path().join("source");
    make_box(&source)?;
    let expected_vertices = vertex_count(&source)?;
    for format in ["obj", "stl", "ply"] {
        let imported = directory.path().join(format!("imported-{format}"));
        let output_path = directory.path().join(format!("box.{format}"));
        export(&source, format, &output_path)?;
        pot()
            .args(["init", imported.to_str().unwrap(), "--json"])
            .output()?;
        import_replace(&imported, &output_path, format)?;
        // Blender's PLY export writes one vertex per unique (vertex, UV) pair (`export_uv` defaults
        // on), and its importer keeps them (`merge_verts` defaults off). A cube's default UVs give 14
        // pairs for its 8 vertices (`box_size_2` in the primitive UV oracle). OBJ and STL write 8.
        let expected = if format == "ply" {
            14
        } else {
            expected_vertices
        };
        assert_eq!(vertex_count(&imported)?, expected, "{format}");
        assert_box_bounds(&imported)?;
    }
    Ok(())
}

#[test]
fn usda_usdz_and_ascii_fbx_round_trip_ids_bounds_and_material_color() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let source = directory.path().join("source");
    make_box(&source)?;
    for format in ["usda", "usdz", "fbx"] {
        let output_path = directory.path().join(format!("box.{format}"));
        export(&source, format, &output_path)?;
        let bytes = fs::read(&output_path)?;
        if format == "usda" {
            assert!(bytes.starts_with(b"#usda 1.0"));
        } else if format == "usdz" {
            assert_eq!(&bytes[..4], b"PK\x03\x04");
            let name_length = usize::from(u16::from_le_bytes(bytes[26..28].try_into()?));
            let extra_length = usize::from(u16::from_le_bytes(bytes[28..30].try_into()?));
            assert_eq!((30 + name_length + extra_length) % 64, 0);
        } else {
            assert!(bytes.starts_with(b"; FBX 7.4.0 project file"));
        }
        let imported = directory.path().join(format!("imported-{format}"));
        pot()
            .args(["init", imported.to_str().unwrap(), "--json"])
            .output()?;
        import_replace(&imported, &output_path, format)?;
        assert_box_bounds(&imported)?;
        let document: Value = serde_json::from_slice(&fs::read(imported.join("scene.json"))?)?;
        assert!(
            document["nodes"].get("body").is_some(),
            "{format} node ID was not preserved"
        );
        assert_eq!(
            document["materials"]["clay"]["base_color"],
            serde_json::json!([0.65, 0.32, 0.18, 1.0]),
            "{format}"
        );
    }
    Ok(())
}

#[test]
fn append_remaps_colliding_ids_and_dry_run_does_not_commit() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let source = directory.path().join("source");
    let target = directory.path().join("target");
    let output_path = directory.path().join("box.glb");
    make_box(&source)?;
    make_box(&target)?;
    export(&source, "glb", &output_path)?;
    let dry = pot()
        .args(["import", target.to_str().unwrap(), "--file"])
        .arg(&output_path)
        .args([
            "--format",
            "glb",
            "--base-revision",
            "1",
            "--mode",
            "append",
            "--dry-run",
            "--json",
        ])
        .output()?;
    assert!(
        dry.status.success(),
        "{}",
        String::from_utf8_lossy(&dry.stdout)
    );
    assert_eq!(json(&dry)?["result"]["committed"], false);
    assert_eq!(json(&dry)?["scene"]["revision"], 1);
    assert_eq!(json(&dry)?["result"]["candidate_revision"], 2);
    assert_eq!(vertex_count(&target)?, 8);

    let committed = pot()
        .args(["import", target.to_str().unwrap(), "--file"])
        .arg(&output_path)
        .args([
            "--format",
            "glb",
            "--base-revision",
            "1",
            "--mode",
            "append",
            "--json",
        ])
        .output()?;
    assert!(
        committed.status.success(),
        "{}",
        String::from_utf8_lossy(&committed.stdout)
    );
    let result = json(&committed)?;
    assert_eq!(result["scene"]["revision"], 2);
    let mapped = result["result"]["id_mappings"]["nodes"]["body"]
        .as_str()
        .ok_or("append collision mapping missing")?;
    assert_ne!(mapped, "body");
    let document: Value = serde_json::from_slice(&fs::read(target.join("scene.json"))?)?;
    assert!(document["nodes"].get(mapped).is_some());
    // The exporter splits the box's vertices per normal; the importer welds those copies back to 8.
    assert_eq!(vertex_count(&target)?, 8 + 8);
    Ok(())
}

#[test]
fn export_rejects_existing_output_and_import_rejects_extension_mismatch()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let existing = directory.path().join("existing.glb");
    let wrong_extension = directory.path().join("wrong.obj");
    make_box(&scene)?;
    fs::write(&existing, b"keep")?;
    let exists = pot()
        .args([
            "export",
            scene.to_str().unwrap(),
            "--format",
            "glb",
            "--out",
        ])
        .arg(&existing)
        .arg("--json")
        .output()?;
    assert_eq!(exists.status.code(), Some(5));
    assert_eq!(json(&exists)?["error"]["code"], "OUTPUT_EXISTS");
    assert_eq!(fs::read(&existing)?, b"keep");
    fs::write(&wrong_extension, b"not an OBJ")?;
    let mismatch = pot()
        .args(["import", scene.to_str().unwrap(), "--file"])
        .arg(&wrong_extension)
        .args(["--format", "stl", "--base-revision", "1", "--json"])
        .output()?;
    assert_eq!(mismatch.status.code(), Some(2));
    assert_eq!(json(&mismatch)?["error"]["code"], "INVALID_ARGUMENT");
    Ok(())
}

#[test]
fn lossy_export_requires_explicit_permission_and_reports_losses() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let output_path = directory.path().join("scene.stl");
    make_box(&scene)?;
    let output = pot()
        .args([
            "export",
            scene.to_str().unwrap(),
            "--format",
            "stl",
            "--out",
        ])
        .arg(&output_path)
        .arg("--json")
        .output()?;
    assert_eq!(output.status.code(), Some(4));
    let result = json(&output)?;
    assert_eq!(result["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert!(
        !result["result"]["losses"]
            .as_array()
            .ok_or("losses missing")?
            .is_empty(),
        "STL export should report losses"
    );
    assert!(!output_path.exists());
    Ok(())
}

#[test]
fn ascii_fbx_rejects_binary_header_honestly() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    make_box(&scene)?;
    let binary_fbx = directory.path().join("binary.fbx");
    fs::write(&binary_fbx, b"Kaydara FBX Binary  \0\x1a\0")?;
    let imported = pot()
        .args(["import", scene.to_str().unwrap(), "--file"])
        .arg(&binary_fbx)
        .args(["--format", "fbx", "--base-revision", "1", "--json"])
        .output()?;
    assert_eq!(imported.status.code(), Some(4));
    assert_eq!(json(&imported)?["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        json(&imported)?["error"]["details"]["feature_id"],
        "format.fbx_binary"
    );
    Ok(())
}

#[test]
fn export_view_is_restricted_to_svg_and_pdf() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let output_path = directory.path().join("view.glb");
    make_box(&scene)?;
    let output = pot()
        .args([
            "export",
            scene.to_str().unwrap(),
            "--format",
            "glb",
            "--out",
        ])
        .arg(&output_path)
        .args(["--view", "top", "--json"])
        .output()?;
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(json(&output)?["error"]["code"], "INVALID_ARGUMENT");
    assert!(!output_path.exists());
    Ok(())
}

#[test]
fn glb_emissive_strength_is_omitted_when_inert_and_round_trips_through_extras()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let source = directory.path().join("source");
    let imported = directory.path().join("imported");
    let output_path = directory.path().join("emissive.glb");
    pot()
        .args(["init", source.to_str().unwrap(), "--json"])
        .output()?;
    let operations = directory.path().join("emissive.operations.json");
    fs::write(
        &operations,
        json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op": "material.create", "id": "inert", "name": "Inert", "emission_strength": 2.5},
                {"op": "material.create", "id": "unit", "name": "Unit", "emission_color": [0.5, 0.25, 0.125], "emission_strength": 1.0},
                {"op": "material.create", "id": "glow", "name": "Glow", "emission_color": [0.5, 0.25, 0.125], "emission_strength": 2.5},
                {"op": "node.create", "id": "inert_box", "kind": "box", "params": {"size": 0.5}, "material": "inert"},
                {"op": "node.create", "id": "unit_box", "kind": "box", "params": {"size": 0.5}, "material": "unit"},
                {"op": "node.create", "id": "glow_box", "kind": "box", "params": {"size": 0.5}, "material": "glow"}
            ]
        })
        .to_string(),
    )?;
    let applied = pot()
        .args(["apply", source.to_str().unwrap(), "--file"])
        .arg(&operations)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    export(&source, "glb", &output_path)?;
    let bytes = fs::read(&output_path)?;
    let json_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    let document: Value = serde_json::from_slice(&bytes[20..20 + json_length])?;
    let named = |name: &str| -> Result<Value, Box<dyn Error>> {
        document["materials"]
            .as_array()
            .and_then(|materials| materials.iter().find(|material| material["name"] == name))
            .cloned()
            .ok_or_else(|| format!("missing material {name}").into())
    };
    let inert = named("Inert")?;
    assert!(inert["extensions"]["KHR_materials_emissive_strength"].is_null());
    assert_eq!(inert["extras"]["potter"]["emission_strength"], 2.5);
    assert!(named("Unit")?["extensions"]["KHR_materials_emissive_strength"].is_null());
    assert_eq!(
        named("Glow")?["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"],
        1.25
    );
    assert!(document["extensionsUsed"].as_array().is_some_and(|used| {
        used.iter()
            .any(|name| name == "KHR_materials_emissive_strength")
    }));

    pot()
        .args(["init", imported.to_str().unwrap(), "--json"])
        .output()?;
    import_replace(&imported, &output_path, "glb")?;
    let scene: Value = serde_json::from_slice(&fs::read(imported.join("scene.json"))?)?;
    assert_eq!(scene["materials"]["inert"]["emission_strength"], 2.5);
    assert_eq!(scene["materials"]["unit"]["emission_strength"], 1.0);
    assert_eq!(scene["materials"]["glow"]["emission_strength"], 2.5);
    Ok(())
}

fn rigged_shape_key_scene(scene: &Path, posed: bool) -> Result<(), Box<dyn Error>> {
    let init = pot()
        .args(["init", scene.to_str().unwrap(), "--json"])
        .output()?;
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stdout)
    );
    let weights: Vec<Value> = (0..8)
        .map(|vertex_id| json!({"vertex_id": vertex_id, "weight": 1.0}))
        .collect();
    let mut operations = vec![
        json!({"op":"node.create","id":"body","kind":"box","params":{"size":2.0}}),
        json!({"op":"node.create","id":"rig","name":"Rig","kind":"armature"}),
        json!({"op":"bone.create","target":{"id":"rig"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,0.0,1.0]}),
        json!({"op":"vertex_group.create","target":{"id":"body"},"id":"root_group","name":"Root"}),
        json!({"op":"vertex_group.assign","target":{"id":"body"},"group_id":"root_group","weights":weights}),
        json!({"op":"modifier.create","target":{"id":"body"},"id":"armature","type":"armature","params":{"object":"rig","use_vertex_groups":true}}),
        json!({"op":"shape_key.create","target":{"id":"body"},"id":"lift","name":"Lift","positions":{"0":[-1.0,-1.0,1.5]}}),
    ];
    if posed {
        let half_turn = std::f64::consts::FRAC_1_SQRT_2;
        operations.push(json!({"op":"pose.set","target":{"id":"rig"},"bone_id":"root","set":{"rotation":[0.0,0.0,half_turn,half_turn]}}));
    }
    let operations_path = scene.with_extension("operations.json");
    fs::write(
        &operations_path,
        serde_json::to_vec(
            &json!({"schema_version": 1, "base_revision": 0, "operations": operations}),
        )?,
    )?;
    let applied = pot()
        .args(["apply", scene.to_str().unwrap(), "--file"])
        .arg(&operations_path)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    Ok(())
}

#[test]
fn glb_refuses_posed_armature_under_shape_keys_and_exports_at_rest() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let rest = directory.path().join("rest");
    rigged_shape_key_scene(&rest, false)?;
    export(&rest, "glb", &directory.path().join("rest.glb"))?;

    let posed = directory.path().join("posed");
    rigged_shape_key_scene(&posed, true)?;
    let output = pot()
        .args([
            "export",
            posed.to_str().unwrap(),
            "--format",
            "glb",
            "--out",
        ])
        .arg(directory.path().join("posed.glb"))
        .arg("--json")
        .output()?;
    assert!(
        !output.status.success(),
        "a posed armature under shape keys must not export as glb"
    );
    let failure = json(&output)?;
    assert_eq!(failure["ok"], false);
    assert_eq!(failure["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        failure["error"]["details"]["feature_id"],
        "rig.skin_shape_key_pose"
    );
    Ok(())
}
