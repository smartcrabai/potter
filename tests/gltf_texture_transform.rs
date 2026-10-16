//! glTF texture slots that Potter's texture fields cannot hold import as material graphs, as Blender's
//! glTF importer builds them: transformed textures through Mapping, emissive and occlusion textures
//! into the Principled BSDF and the `glTF Material Output` group, and normal map scale as Strength.

use std::{error::Error, fs, path::Path, path::PathBuf, process::Command};

use serde_json::{Value, json};
use tempfile::TempDir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// A 1x1 RGB PNG (200, 100, 50).
const PNG: [u8; 69] = [
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0,
    0, 0, 144, 119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 56, 145, 98, 4, 0, 3, 86,
    1, 95, 232, 23, 132, 82, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

/// Positions (3 vertices) then `TEXCOORD_0` (3 vertices), little-endian f32.
const BUFFER: [u8; 60] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128,
    63,
];

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

/// A glTF triangle whose only material is `material`, written into `directory`.
fn write_gltf(directory: &Path, material: &Value) -> TestResult<PathBuf> {
    fs::write(directory.join("image.png"), PNG)?;
    fs::write(directory.join("buffer.bin"), BUFFER)?;
    let document = json!({
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{ "primitives": [{
            "attributes": { "POSITION": 0, "TEXCOORD_0": 1 },
            "material": 0
        }] }],
        "materials": [material],
        "textures": [{ "source": 0 }],
        "images": [{ "uri": "image.png", "mimeType": "image/png" }],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
              "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] },
            { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2" }
        ],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 24 }
        ],
        "buffers": [{ "uri": "buffer.bin", "byteLength": 60 }],
        "extensionsUsed": ["KHR_texture_transform"]
    });
    let path = directory.join("textured.gltf");
    fs::write(&path, serde_json::to_vec(&document)?)?;
    Ok(path)
}

/// A material whose base color texture carries `transform`.
fn transformed_material(transform: &Value) -> Value {
    json!({
        "name": "Textured",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "baseColorTexture": {
                "index": 0,
                "extensions": { "KHR_texture_transform": transform.clone() }
            }
        }
    })
}

/// An imported triangle: its directory (kept alive), its project and the scene JSON.
struct Imported {
    directory: TempDir,
    project: PathBuf,
    scene: Value,
}

/// Imports `material` as a glTF without `--allow-lossy`: any loss fails the import.
fn import(material: &Value) -> TestResult<Imported> {
    let directory = TempDir::new()?;
    let gltf = write_gltf(directory.path(), material)?;
    let project = directory.path().join("scene");
    let init = pot().arg("init").arg(&project).arg("--json").output()?;
    assert!(init.status.success(), "pot init failed");
    let output = pot()
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(&gltf)
        .args(["--format", "gltf", "--base-revision", "0", "--json"])
        .output()?;
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(envelope["ok"], true, "import failed: {envelope}");
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    Ok(Imported {
        directory,
        project,
        scene,
    })
}

/// The shader graph of the only material of `scene`.
fn only_graph(scene: &Value) -> TestResult<&Value> {
    let material = scene["materials"]
        .as_object()
        .and_then(|materials| materials.values().next())
        .ok_or("no material")?;
    let graph_id = material["node_tree"]
        .as_str()
        .ok_or("material has no shader graph")?;
    Ok(&scene["node_groups"][graph_id])
}

/// The node of `graph` with type `node_type` and its ID.
fn node_of<'a>(graph: &'a Value, node_type: &str) -> TestResult<(&'a str, &'a Value)> {
    graph["nodes"]
        .as_object()
        .and_then(|nodes| nodes.iter().find(|(_, node)| node["type"] == node_type))
        .map(|(id, node)| (id.as_str(), node))
        .ok_or_else(|| format!("no {node_type} node").into())
}

fn has_link(graph: &Value, from: &str, from_socket: &str, to: &str, to_socket: &str) -> bool {
    graph["links"].as_array().is_some_and(|links| {
        links.iter().any(|link| {
            link["from_node"] == from
                && link["from_socket"] == from_socket
                && link["to_node"] == to
                && link["to_socket"] == to_socket
        })
    })
}

#[test]
fn transformed_base_color_imports_as_a_mapped_image_texture_graph() -> TestResult {
    let imported = import(&transformed_material(
        &json!({ "offset": [0.25, 0.5], "rotation": 0.5, "scale": [2.0, 3.0] }),
    ))?;
    let scene = &imported.scene;
    let material = scene["materials"]
        .as_object()
        .and_then(|materials| materials.values().next())
        .ok_or("no material")?;
    assert!(
        material["base_color_texture"].is_null(),
        "the graph holds the texture"
    );
    let graph = only_graph(scene)?;
    let (mapping_id, mapping) = node_of(graph, "ShaderNodeMapping")?;
    // Blender's glTF importer: Location = (ox + sy sin r, 1 - oy - sy cos r), Rotation z = r, Scale = (sx, sy, 1).
    let (sin, cos) = 0.5_f64.sin_cos();
    let location = mapping["inputs"]["Location"]
        .as_array()
        .ok_or("no Location")?;
    assert!((location[0].as_f64().ok_or("x")? - (0.25 + 3.0 * sin)).abs() < 1e-12);
    assert!((location[1].as_f64().ok_or("y")? - (1.0 - 0.5 - 3.0 * cos)).abs() < 1e-12);
    assert_eq!(mapping["inputs"]["Rotation"][2], json!(0.5));
    assert_eq!(mapping["inputs"]["Scale"], json!([2.0, 3.0, 1.0]));
    assert_eq!(mapping["properties"]["vector_type"], "POINT");
    let (image_id, _) = node_of(graph, "ShaderNodeTexImage")?;
    let (coord_id, _) = node_of(graph, "ShaderNodeTexCoord")?;
    assert!(has_link(graph, coord_id, "UV", mapping_id, "Vector"));
    assert!(has_link(graph, mapping_id, "Vector", image_id, "Vector"));
    assert!(has_link(
        graph,
        image_id,
        "Color",
        "principled",
        "Base Color"
    ));
    Ok(())
}

#[test]
fn transformed_base_color_round_trips_through_glb_export() -> TestResult {
    let transform = json!({ "offset": [0.25, 0.5], "rotation": 0.5, "scale": [2.0, 3.0] });
    let imported = import(&transformed_material(&transform))?;
    let out = imported.directory.path().join("out.glb");
    let export = pot()
        .arg("export")
        .arg(&imported.project)
        .args(["--format", "glb", "--out"])
        .arg(&out)
        .arg("--json")
        .output()?;
    let envelope: Value = serde_json::from_slice(&export.stdout)?;
    assert_eq!(envelope["ok"], true, "glb export failed: {envelope}");
    let bytes = fs::read(&out)?;
    let json_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    let document: Value = serde_json::from_slice(&bytes[20..20 + json_length])?;
    let exported = &document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["extensions"]
        ["KHR_texture_transform"];
    assert_eq!(exported["offset"], transform["offset"], "offset survives");
    assert_eq!(exported["scale"], transform["scale"], "scale survives");
    let rotation = exported["rotation"].as_f64().ok_or("no rotation")?;
    assert!((rotation - 0.5).abs() < 1e-12, "rotation survives");
    Ok(())
}

#[test]
fn emissive_and_occlusion_textures_import_as_graph_nodes_without_loss() -> TestResult {
    let imported = import(&json!({
        "name": "Lit",
        "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0] },
        "emissiveFactor": [1.0, 1.0, 1.0],
        "emissiveTexture": { "index": 0 },
        "occlusionTexture": {
            "index": 0,
            "extensions": { "KHR_texture_transform": { "offset": [0.5, 0.0] } }
        }
    }))?;
    let graph = only_graph(&imported.scene)?;
    let (split_id, _) = node_of(graph, "ShaderNodeSeparateColor")?;
    let (group_id, group) = node_of(graph, "ShaderNodeGroup")?;
    assert_eq!(
        group["properties"]["node_tree"]["name"],
        "glTF Material Output"
    );
    assert!(
        has_link(graph, split_id, "Red", group_id, "Occlusion"),
        "occlusion's red channel feeds the group's Occlusion input"
    );
    let output_group = &imported.scene["node_groups"]["gltf_material_output"];
    assert_eq!(output_group["name"], "glTF Material Output");
    assert_eq!(output_group["interface"]["inputs"][0]["id"], "Occlusion");
    let (image_id, _) = node_of(graph, "ShaderNodeTexImage")?;
    assert!(
        has_link(graph, image_id, "Color", "principled", "Emission Color"),
        "a white emissive factor leaves the texture color unmultiplied"
    );
    Ok(())
}

#[test]
fn normal_texture_scale_becomes_the_normal_map_strength() -> TestResult {
    let imported = import(&json!({
        "name": "Bumpy",
        "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0] },
        "normalTexture": { "index": 0, "scale": 0.5 }
    }))?;
    let graph = only_graph(&imported.scene)?;
    let (normal_id, normal_map) = node_of(graph, "ShaderNodeNormalMap")?;
    assert_eq!(normal_map["inputs"]["Strength"], json!(0.5));
    assert!(has_link(graph, normal_id, "Normal", "principled", "Normal"));
    Ok(())
}

#[test]
fn occlusion_strength_other_than_one_is_reported_as_a_loss() -> TestResult {
    let directory = TempDir::new()?;
    let gltf = write_gltf(
        directory.path(),
        &json!({
            "name": "Occluded",
            "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0] },
            "occlusionTexture": { "index": 0, "strength": 0.5 }
        }),
    )?;
    let project = directory.path().join("scene");
    pot().arg("init").arg(&project).arg("--json").output()?;
    let output = pot()
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(&gltf)
        .args(["--format", "gltf", "--base-revision", "0", "--json"])
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !output.status.success(),
        "a strength loss refuses the import"
    );
    assert!(stdout.contains("material.occlusion_strength"), "{stdout}");
    Ok(())
}

#[test]
fn emissive_and_occlusion_graphs_export_back_to_their_gltf_slots() -> TestResult {
    let imported = import(&json!({
        "name": "Lit",
        "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0] },
        "emissiveFactor": [1.0, 1.0, 1.0],
        "emissiveTexture": { "index": 0 },
        "occlusionTexture": { "index": 0 }
    }))?;
    let out = imported.directory.path().join("out.glb");
    let export = pot()
        .arg("export")
        .arg(&imported.project)
        .args(["--format", "glb", "--out"])
        .arg(&out)
        .arg("--json")
        .output()?;
    let envelope: Value = serde_json::from_slice(&export.stdout)?;
    assert_eq!(envelope["ok"], true, "glb export failed: {envelope}");
    let bytes = fs::read(&out)?;
    let json_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    let document: Value = serde_json::from_slice(&bytes[20..20 + json_length])?;
    let material = &document["materials"][0];
    assert!(
        material["emissiveTexture"].is_object(),
        "emissive texture is exported"
    );
    assert!(
        material["occlusionTexture"].is_object(),
        "occlusion texture is exported"
    );
    Ok(())
}

/// The glTF JSON chunk of a GLB file.
fn glb_document(path: &Path) -> TestResult<Value> {
    let bytes = fs::read(path)?;
    let json_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    Ok(serde_json::from_slice(&bytes[20..20 + json_length])?)
}

/// The `pot export --format glb` of the project into `out`.
fn export_glb(project: &Path, out: &Path) -> TestResult<Value> {
    let export = pot()
        .arg("export")
        .arg(project)
        .args(["--format", "glb", "--out"])
        .arg(out)
        .arg("--json")
        .output()?;
    let envelope: Value = serde_json::from_slice(&export.stdout)?;
    assert_eq!(envelope["ok"], true, "glb export failed: {envelope}");
    glb_document(out)
}

/// The scene revision stored in `project`.
fn revision(project: &Path) -> TestResult<u64> {
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    scene["revision"]
        .as_u64()
        .ok_or_else(|| "scene has no revision".into())
}

/// Runs `pot import` of `gltf` into `project` and returns the JSON envelope; `allow_lossy` adds `--allow-lossy`.
fn run_import(
    project: &Path,
    gltf: &Path,
    base_revision: u64,
    allow_lossy: bool,
) -> TestResult<Value> {
    let mut command = pot();
    command
        .arg("import")
        .arg(project)
        .arg("--file")
        .arg(gltf)
        .args(["--format", "gltf", "--mode", "append", "--base-revision"])
        .arg(base_revision.to_string());
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    let output = command.arg("--json").output()?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Asserts that every number in `value` is within 1e-9 of the matching number in `expected`.
fn assert_close(value: &Value, expected: &[f64]) {
    let actual = value
        .as_array()
        .unwrap_or_else(|| panic!("expected an array, got {value}"));
    assert_eq!(actual.len(), expected.len(), "{value}");
    for (actual, expected) in actual.iter().zip(expected) {
        let actual = actual.as_f64().unwrap_or(f64::NAN);
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }
}

#[test]
fn transformed_normal_texture_keeps_its_scale_and_round_trips() -> TestResult {
    let imported = import(&json!({
        "name": "Bumpy",
        "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0] },
        "normalTexture": {
            "index": 0,
            "scale": 0.5,
            "extensions": {
                "KHR_texture_transform": { "offset": [0.1, 0.0], "scale": [2.0, 2.0] }
            }
        }
    }))?;
    let graph = only_graph(&imported.scene)?;
    let (map_id, _) = node_of(graph, "ShaderNodeMapping")?;
    let (tex_id, _) = node_of(graph, "ShaderNodeTexImage")?;
    let (normal_id, normal_map) = node_of(graph, "ShaderNodeNormalMap")?;
    assert_eq!(
        normal_id, "normal_strength",
        "the Normal Map node has its own name"
    );
    assert_eq!(normal_map["inputs"]["Strength"], json!(0.5));
    assert!(has_link(graph, map_id, "Vector", tex_id, "Vector"));
    assert!(has_link(graph, tex_id, "Color", normal_id, "Color"));
    assert!(has_link(graph, normal_id, "Normal", "principled", "Normal"));

    let out = imported.directory.path().join("out.glb");
    let document = export_glb(&imported.project, &out)?;
    let normal = &document["materials"][0]["normalTexture"];
    assert_eq!(normal["scale"], json!(0.5), "normal scale survives");
    let transform = &normal["extensions"]["KHR_texture_transform"];
    assert_close(&transform["offset"], &[0.1, 0.0]);
    assert_close(&transform["scale"], &[2.0, 2.0]);
    Ok(())
}

#[test]
fn appending_the_same_gltf_twice_keeps_each_copys_graph_and_image() -> TestResult {
    let directory = TempDir::new()?;
    let gltf = write_gltf(
        directory.path(),
        &transformed_material(&json!({ "offset": [0.25, 0.0] })),
    )?;
    let project = directory.path().join("scene");
    let init = pot().arg("init").arg(&project).arg("--json").output()?;
    assert!(init.status.success(), "pot init failed");
    let first = run_import(&project, &gltf, 0, false)?;
    assert_eq!(first["ok"], true, "first import failed: {first}");
    let second = run_import(&project, &gltf, revision(&project)?, false)?;
    assert_eq!(second["ok"], true, "second import failed: {second}");

    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let materials = scene["materials"]
        .as_object()
        .ok_or("no materials")?
        .values()
        .collect::<Vec<_>>();
    assert_eq!(materials.len(), 2);
    let graph_ids = materials
        .iter()
        .map(|material| material["node_tree"].as_str().map(str::to_owned))
        .collect::<Option<std::collections::BTreeSet<_>>>()
        .ok_or("a material has no shader graph")?;
    assert_eq!(graph_ids.len(), 2, "each copy keeps its own graph");
    let mut images = Vec::new();
    for graph_id in &graph_ids {
        let (_, node) = node_of(&scene["node_groups"][graph_id], "ShaderNodeTexImage")?;
        images.push(
            node["properties"]["image"]
                .as_str()
                .ok_or("no image")?
                .to_owned(),
        );
    }
    assert_ne!(images[0], images[1], "each copy samples its own image");
    for image in &images {
        assert!(
            scene["images"][image].is_object(),
            "{image} is in the scene"
        );
    }
    Ok(())
}

#[test]
fn emissive_texture_on_another_uv_set_is_a_loss_not_a_failure() -> TestResult {
    let directory = TempDir::new()?;
    let gltf = write_gltf(
        directory.path(),
        &json!({
            "name": "Lit",
            "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0] },
            "emissiveFactor": [1.0, 1.0, 1.0],
            "emissiveTexture": { "index": 0, "texCoord": 1 }
        }),
    )?;
    let project = directory.path().join("scene");
    pot().arg("init").arg(&project).arg("--json").output()?;
    let refused = run_import(&project, &gltf, 0, false)?;
    assert_eq!(
        refused["ok"], false,
        "the loss is refused without --allow-lossy"
    );
    assert!(
        refused.to_string().contains("material.texture_uv_set"),
        "{refused}"
    );
    let accepted = run_import(&project, &gltf, 0, true)?;
    assert_eq!(accepted["ok"], true, "{accepted}");
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let material = scene["materials"]
        .as_object()
        .and_then(|materials| materials.values().next())
        .ok_or("no material")?;
    assert!(
        material["node_tree"].is_null(),
        "the skipped texture needs no graph"
    );
    Ok(())
}

#[test]
fn transform_on_another_uv_set_is_dropped_with_a_loss() -> TestResult {
    let directory = TempDir::new()?;
    let gltf = write_gltf(
        directory.path(),
        &transformed_material(&json!({ "offset": [0.5, 0.0], "texCoord": 1 })),
    )?;
    let project = directory.path().join("scene");
    pot().arg("init").arg(&project).arg("--json").output()?;
    let refused = run_import(&project, &gltf, 0, false)?;
    assert_eq!(
        refused["ok"], false,
        "the loss is refused without --allow-lossy"
    );
    assert!(
        refused.to_string().contains("material.texture_uv_set"),
        "{refused}"
    );
    let accepted = run_import(&project, &gltf, 0, true)?;
    assert_eq!(accepted["ok"], true, "{accepted}");
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let material = scene["materials"]
        .as_object()
        .and_then(|materials| materials.values().next())
        .ok_or("no material")?;
    assert!(
        material["node_tree"].is_null(),
        "the untransformed texture stays a field"
    );
    assert!(
        !material["base_color_texture"].is_null(),
        "the texture is kept"
    );
    Ok(())
}
