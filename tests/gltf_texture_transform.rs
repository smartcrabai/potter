//! A glTF base color texture with `KHR_texture_transform` imports as a material graph: Image Texture
//! fed by Mapping (Blender's Location, Rotation and Scale for the transform) fed by Texture Coordinate.

use std::{error::Error, fs, path::Path, process::Command};

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

fn cli_json(output: &std::process::Output) -> TestResult<Value> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Writes a triangle whose base color texture carries `transform` and returns the glTF path.
fn write_gltf(directory: &Path, transform: &Value) -> TestResult<std::path::PathBuf> {
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
        "materials": [{
            "name": "Textured",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "baseColorTexture": {
                    "index": 0,
                    "extensions": { "KHR_texture_transform": transform.clone() }
                }
            }
        }],
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

fn import_project(transform: &Value) -> TestResult<Value> {
    let directory = TempDir::new()?;
    let gltf = write_gltf(directory.path(), transform)?;
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
    let envelope = cli_json(&output)?;
    assert_eq!(envelope["ok"], true, "import failed: {envelope}");
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    Ok(scene)
}

#[test]
fn transformed_base_color_imports_as_a_mapped_image_texture_graph() -> TestResult {
    let scene =
        import_project(&json!({ "offset": [0.25, 0.5], "rotation": 0.5, "scale": [2.0, 3.0] }))?;
    let material = scene["materials"]
        .as_object()
        .and_then(|materials| materials.values().next())
        .ok_or("no material")?;
    let graph_id = material["node_tree"]
        .as_str()
        .ok_or("material has no shader graph")?;
    assert!(
        material["base_color_texture"].is_null(),
        "the graph holds the texture"
    );
    let graph = &scene["node_groups"][graph_id];
    let nodes = &graph["nodes"];
    let mapping = nodes
        .as_object()
        .and_then(|nodes| {
            nodes
                .values()
                .find(|node| node["type"] == "ShaderNodeMapping")
        })
        .ok_or("no Mapping node")?;
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
    let links = graph["links"].as_array().ok_or("no links")?;
    let has_link = |from: &str, from_socket: &str, to: &str, to_socket: &str| {
        links.iter().any(|link| {
            link["from_node"] == from
                && link["from_socket"] == from_socket
                && link["to_node"] == to
                && link["to_socket"] == to_socket
        })
    };
    let image = nodes
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["type"] == "ShaderNodeTexImage")
        })
        .map(|(id, _)| id.clone())
        .ok_or("no Image Texture node")?;
    let coord = nodes
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["type"] == "ShaderNodeTexCoord")
        })
        .map(|(id, _)| id.clone())
        .ok_or("no Texture Coordinate node")?;
    let mapping_id = nodes
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["type"] == "ShaderNodeMapping")
        })
        .map(|(id, _)| id.clone())
        .ok_or("no Mapping id")?;
    assert!(
        has_link(&coord, "UV", &mapping_id, "Vector"),
        "coordinates feed Mapping"
    );
    assert!(
        has_link(&mapping_id, "Vector", &image, "Vector"),
        "Mapping feeds the texture"
    );
    assert!(
        has_link(&image, "Color", "principled", "Base Color"),
        "the texture feeds Base Color"
    );
    Ok(())
}
