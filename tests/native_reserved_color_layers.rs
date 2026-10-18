//! Reserved Blender color-layer names must stay user-facing without colliding with Potter's
//! internal vertex-weight and color-layer stores. The Blender oracle also exercises render and
//! glTF selection through a shared material.

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
    fs,
    path::{Path, PathBuf},
};

use native_oracle::{
    FLOAT_TOLERANCE, TestResult, apply_operations, assert_close, assert_floats, assert_rgba,
    checked, dump_blend, dumped_object, export, export_raw, floats, import_blend, loss_features,
    losses, mesh_of_node, node_id, number, pot, read_scene, vec4, write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const LAYER_KEYS: [(&str, &str); 2] = [
    ("vertex_groups", "user:vertex_groups"),
    ("color_layers", "user:color_layers"),
];
const FORBIDDEN_LOSSES: [&str; 3] = [
    "mesh.color_attribute",
    "mesh.attribute",
    "material.shader_graph.vertex_color",
];
const RENDER_TOLERANCE: f64 = 0.12;

fn build_reserved_oracle(blender: &Path, root: &Path) -> TestResult<(PathBuf, Value)> {
    let script = write_script(
        root,
        "build_reserved_color_oracle.py",
        include_str!("fixtures/native_oracle/build_reserved_color_oracle.py"),
    )?;
    let output = root.join("reserved_color_oracle");
    native_oracle::run_blender(blender, &script, None, &output)?;
    let expected = serde_json::from_slice(&fs::read(output.join("expected.json"))?)?;
    Ok((output.join("reserved_color_oracle.blend"), expected))
}

fn forbid_color_losses(report: &Value, label: &str) {
    for feature in loss_features(report) {
        assert!(
            !FORBIDDEN_LOSSES.contains(&feature.as_str()),
            "{label} reported {feature} for represented reserved color data: {report}"
        );
    }
}

fn forbid_gltf_color_losses(report: &Value) {
    const FEATURES: [&str; 2] = ["mesh.color_attribute", "material.shader_graph.vertex_color"];
    for feature in loss_features(report) {
        assert!(
            !FEATURES.contains(&feature.as_str()),
            "glTF export reported {feature} for representable color data: {report}"
        );
    }
}

fn check_imported_colors(mesh: &Value, expected: &Value) -> TestResult {
    let store = &mesh["attributes"]["color_layers"];
    let order: Vec<String> = expected["color_layer_order"]
        .as_array()
        .ok_or("expected color layer order is missing")?
        .iter()
        .map(|name| {
            name.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "expected color layer name is not a string".into())
        })
        .collect::<TestResult<_>>()?;
    assert_eq!(store["layers"], json!(order), "color layer names/order");
    let layers = expected["color_layers"]
        .as_array()
        .ok_or("expected color layer values are missing")?;
    let active = layers
        .get(
            expected["color_active_index"]
                .as_u64()
                .ok_or("active index missing")? as usize,
        )
        .ok_or("active color layer index is out of range")?;
    let render = layers
        .get(
            expected["color_render_index"]
                .as_u64()
                .ok_or("render index missing")? as usize,
        )
        .ok_or("render color layer index is out of range")?;
    assert_eq!(store["active"], active["name"]);
    assert_eq!(store["render"], render["name"]);

    for (name, storage_key) in LAYER_KEYS {
        let Some(layer_expected) = layers
            .iter()
            .find(|layer| layer["name"].as_str() == Some(name))
        else {
            assert_eq!(
                name, "color_layers",
                "only the registry-name collision is optional"
            );
            continue;
        };
        let attribute = mesh["attributes"]
            .get(storage_key)
            .ok_or_else(|| format!("color layer {name} is missing under {storage_key}"))?;
        assert_eq!(attribute["type"], "color", "{name} storage type");
        let values = layer_expected["values"]
            .as_array()
            .ok_or("expected color values are missing")?;
        match layer_expected["domain"].as_str() {
            Some("POINT") => {
                assert_eq!(attribute["domain"], "point", "{name} domain");
                let vertices = mesh["vertices"]
                    .as_array()
                    .ok_or("mesh vertices are missing")?;
                assert_eq!(vertices.len(), values.len(), "{name} point count");
                for (index, (vertex, expected_value)) in vertices.iter().zip(values).enumerate() {
                    let vertex_id = vertex["id"].as_u64().ok_or("vertex id is missing")?;
                    let vertex_key = format!("v{vertex_id}");
                    let actual = attribute["values"]
                        .get(vertex_key.as_str())
                        .ok_or_else(|| format!("{name} vertex {index} value is missing"))?;
                    assert_rgba(
                        vec4(actual)?,
                        vec4(expected_value)?,
                        FLOAT_TOLERANCE,
                        &format!("{name} vertex {index}"),
                    );
                }
            }
            Some("CORNER") => {
                assert_eq!(attribute["domain"], "corner", "{name} domain");
                let face_id = mesh["faces"][0]["id"]
                    .as_u64()
                    .ok_or("face id is missing")?;
                let face_key = format!("f{face_id}");
                let actual = attribute["values"][face_key.as_str()]
                    .as_array()
                    .ok_or("corner color row is missing")?;
                assert_eq!(actual.len(), values.len(), "{name} corner count");
                for (index, (actual, expected_value)) in actual.iter().zip(values).enumerate() {
                    assert_rgba(
                        vec4(actual)?,
                        vec4(expected_value)?,
                        FLOAT_TOLERANCE,
                        &format!("{name} corner {index}"),
                    );
                }
            }
            other => return Err(format!("unexpected Blender domain {other:?}").into()),
        }
    }

    let raw_vertex_groups = &mesh["attributes"]["vertex_groups"];
    if !raw_vertex_groups.is_null() {
        assert!(
            raw_vertex_groups.is_object(),
            "native vertex weights must be stored separately"
        );
        assert_ne!(
            raw_vertex_groups["type"], "color",
            "raw vertex_groups holds weights, not colors"
        );
    }
    assert!(
        mesh["attributes"]["user:vertex_groups"]["values"].is_object(),
        "the reserved color layer must have its own escaped storage key"
    );
    let blender_attributes = &mesh["attributes"]["blender_attributes"];
    assert!(
        !has_named_layer(blender_attributes, &["vertex_groups", "color_layers"]),
        "native colors must not be duplicated in blender_attributes"
    );
    Ok(())
}

fn has_named_layer(value: &Value, names: &[&str]) -> bool {
    match value {
        Value::Array(items) => items.iter().any(|item| has_named_layer(item, names)),
        Value::Object(fields) => {
            fields
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| names.contains(&name))
                || fields.iter().any(|(key, item)| {
                    names.contains(&key.as_str()) || has_named_layer(item, names)
                })
        }
        _ => false,
    }
}

fn check_imported_weights(scene: &Value, name: &str, expected: &Value) -> TestResult {
    let mesh = mesh_of_node(scene, name)?;
    let node = node_id(scene, name)?;
    let data_id = scene["nodes"][&node]["data"]
        .as_str()
        .ok_or("mesh node has no data block")?;
    let data = &scene["data_blocks"][data_id];
    let groups = data["vertex_groups"]
        .as_array()
        .ok_or("native vertex groups are missing")?;
    let expected_weights = expected["vertex_groups"]["Hip"]
        .as_object()
        .ok_or("expected Hip weights are missing")?;
    let group = groups
        .iter()
        .find(|group| group["name"].as_str() == Some("Hip"))
        .ok_or("native vertex group Hip is missing")?;
    let group_id = group["id"].as_str().ok_or("Hip group id is missing")?;
    let vertices = mesh["vertices"]
        .as_array()
        .ok_or("mesh vertices are missing")?;
    assert_eq!(vertices.len(), expected_weights.len(), "Hip member count");
    for (index, vertex) in vertices.iter().enumerate() {
        let expected_weight = number(&expected_weights[&index.to_string()])?;
        let vertex_id = vertex["id"].as_u64().ok_or("vertex id is missing")?;
        let key = vertex_id.to_string();
        assert_close(
            number(&data["vertex_weights"][&key][group_id])?,
            expected_weight,
            FLOAT_TOLERANCE,
            &format!("{name} data-block Hip vertex {index}"),
        );
        assert_close(
            number(&mesh["attributes"]["vertex_groups"]["Hip"][format!("v{vertex_id}")])?,
            expected_weight,
            FLOAT_TOLERANCE,
            &format!("{name} mesh Hip vertex {index}"),
        );
    }
    Ok(())
}

fn data_slot_material<'a>(scene: &'a Value, node: &str, label: &str) -> TestResult<&'a str> {
    let node_value = &scene["nodes"][node];
    let link = node_value["material_links"][0]
        .as_str()
        .ok_or_else(|| format!("{label} material slot link is missing"))?;
    let slots = if link == "data" {
        let data = node_value["data"]
            .as_str()
            .ok_or_else(|| format!("{label} object data is missing"))?;
        &scene["data_blocks"][data]["materials"]
    } else {
        &node_value["materials"]
    };
    slots[0]
        .as_str()
        .ok_or_else(|| format!("{label} material assignment is missing").into())
}

fn check_imported_material(scene: &Value, expected: &Value) -> TestResult<String> {
    let names = expected["material"]["shared_by"]
        .as_array()
        .ok_or("expected shared material users are missing")?;
    let left = node_id(
        scene,
        names[0].as_str().ok_or("left object name is missing")?,
    )?;
    let right = node_id(
        scene,
        names[1].as_str().ok_or("right object name is missing")?,
    )?;
    // The shared material is linked to each mesh data-block (build_reserved_color_oracle.py appends it
    // to `obj.data.materials`), so the slot lives on the data, not on the object.
    let left_material = data_slot_material(scene, &left, "left")?;
    let right_material = data_slot_material(scene, &right, "right")?;
    assert_eq!(
        left_material, right_material,
        "objects must share one material"
    );

    let material = &scene["materials"][left_material];
    assert_eq!(material["name"], expected["material"]["name"]);
    let graph_id = material["node_tree"]
        .as_str()
        .ok_or("shared material graph is missing")?;
    let graph = &scene["node_groups"][graph_id];
    let graph_nodes = graph["nodes"]
        .as_object()
        .ok_or("material graph nodes are missing")?;
    let vertex_node = graph_nodes
        .values()
        .find(|node| node["type"] == "ShaderNodeVertexColor")
        .ok_or("material has no Vertex Color node")?;
    assert_eq!(
        vertex_node["properties"]["layer_name"], expected["material"]["vertex_color_layer"],
        "material layer selector is the raw user-facing name"
    );
    let multiply = graph_nodes
        .values()
        .find(|node| node["type"] == "ShaderNodeMix")
        .ok_or("material has no multiply node")?;
    assert_eq!(
        multiply["properties"]["blend_type"],
        expected["material"]["vertex_color_blend"]
    );
    assert_eq!(
        multiply["properties"]["data_type"],
        expected["material"]["blend_data_type"]
    );
    assert_eq!(multiply["properties"]["blend_type"], "MULTIPLY");
    assert_eq!(multiply["properties"]["data_type"], "RGBA");
    let factor_socket = expected["material"]["base_factor_socket"]
        .as_str()
        .ok_or("Blender oracle factor socket is missing")?;
    assert_floats(
        &floats(&multiply["inputs"][factor_socket])?,
        &floats(&expected["material"]["base_factor"])?,
        FLOAT_TOLERANCE,
        "Principled base factor",
    );
    Ok(left_material.to_owned())
}

fn pixel_at(project: &Path, image_id: &str, uv: [f64; 2]) -> TestResult<[f64; 4]> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][image_id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    Ok(sample(&data, uv, 0))
}

fn check_blend_colors(object: &Value, expected: &Value) -> TestResult {
    let expected_layers = expected["color_layers"]
        .as_array()
        .ok_or("expected Blender layers are missing")?;
    let actual_layers = object["color_layers"]
        .as_array()
        .ok_or("exported Blender color layers are missing")?;
    let names: Vec<&str> = actual_layers
        .iter()
        .map(|layer| layer["name"].as_str().unwrap_or_default())
        .collect();
    let expected_names: Vec<&str> = expected_layers
        .iter()
        .map(|layer| layer["name"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(names, expected_names, "Blender color layer order");
    assert_eq!(
        object["color_active"], expected["color_active_index"],
        "active color index"
    );
    assert_eq!(
        object["color_render"], expected["color_render_index"],
        "render color index"
    );
    for (actual, expected_layer) in actual_layers.iter().zip(expected_layers) {
        assert_eq!(actual["domain"], expected_layer["domain"]);
        assert_eq!(actual["data_type"], expected_layer["data_type"]);
        let values = actual["values"]
            .as_array()
            .ok_or("exported Blender color values are missing")?;
        let wanted = expected_layer["values"]
            .as_array()
            .ok_or("expected Blender color values are missing")?;
        assert_eq!(values.len(), wanted.len(), "{} value count", actual["name"]);
        for (index, (actual, expected)) in values.iter().zip(wanted).enumerate() {
            assert_rgba(
                vec4(actual)?,
                vec4(expected)?,
                FLOAT_TOLERANCE,
                &format!("{} exported Blender value {index}", actual["name"]),
            );
        }
    }
    Ok(())
}

fn check_blend_weights(object: &Value, expected: &Value) -> TestResult {
    let weights = expected["vertex_groups"]["Hip"]
        .as_object()
        .ok_or("expected Hip weights are missing")?;
    let group = object["vertex_groups"]
        .as_array()
        .ok_or("exported Blender groups are missing")?
        .iter()
        .find(|group| group["name"] == "Hip")
        .ok_or("exported Blender group Hip is missing")?;
    assert_eq!(group["weights"].as_object(), Some(weights));
    for (vertex, expected_weight) in weights {
        assert_close(
            number(&group["weights"][vertex])?,
            number(expected_weight)?,
            FLOAT_TOLERANCE,
            &format!("exported Hip vertex {vertex}"),
        );
    }
    Ok(())
}

struct Glb {
    json: Value,
    binary: Vec<u8>,
}

fn read_u32(bytes: &[u8], offset: usize) -> TestResult<u32> {
    let raw: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or("truncated GLB header")?
        .try_into()?;
    Ok(u32::from_le_bytes(raw))
}

fn parse_glb(bytes: &[u8]) -> TestResult<Glb> {
    if bytes.get(0..4) != Some(b"glTF".as_slice()) {
        return Err("output is not a GLB".into());
    }
    let json_len = read_u32(bytes, 12)? as usize;
    let json = serde_json::from_slice(
        bytes
            .get(20..20 + json_len)
            .ok_or("truncated GLB JSON chunk")?,
    )?;
    let bin_header = 20 + json_len;
    let bin_len = read_u32(bytes, bin_header)? as usize;
    let binary = bytes
        .get(bin_header + 8..bin_header + 8 + bin_len)
        .ok_or("truncated GLB BIN chunk")?
        .to_vec();
    Ok(Glb { json, binary })
}

fn color_accessor(glb: &Glb, index: u64) -> TestResult<Vec<[f64; 4]>> {
    let accessor = &glb.json["accessors"][index as usize];
    if accessor["componentType"] != 5126 || accessor["type"] != "VEC4" {
        return Err("COLOR accessor is not a float32 VEC4".into());
    }
    let count = accessor["count"].as_u64().ok_or("COLOR count is missing")? as usize;
    let view_index = accessor["bufferView"]
        .as_u64()
        .ok_or("COLOR buffer view is missing")? as usize;
    let view = &glb.json["bufferViews"][view_index];
    let base = view["byteOffset"].as_u64().unwrap_or(0) as usize
        + accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
    let stride = view["byteStride"].as_u64().unwrap_or(16) as usize;
    let mut values = Vec::with_capacity(count);
    for item in 0..count {
        let mut color = [0.0; 4];
        for (channel, component) in color.iter_mut().enumerate() {
            let offset = base + item * stride + channel * 4;
            *component = f64::from(f32::from_le_bytes(
                glb.binary
                    .get(offset..offset + 4)
                    .ok_or("COLOR accessor exceeds BIN chunk")?
                    .try_into()?,
            ));
        }
        values.push(color);
    }
    Ok(values)
}

fn check_glb_colors(glb: &Glb, expected: &Value) -> TestResult {
    let mut colors_by_semantic: [Vec<[f64; 4]>; 2] = [Vec::new(), Vec::new()];
    for mesh in glb.json["meshes"]
        .as_array()
        .ok_or("GLB meshes are missing")?
    {
        for primitive in mesh["primitives"]
            .as_array()
            .ok_or("GLB primitive list is missing")?
        {
            for (index, semantic) in ["COLOR_0", "COLOR_1"].iter().enumerate() {
                if let Some(accessor) = primitive["attributes"][*semantic].as_u64() {
                    colors_by_semantic[index].extend(color_accessor(glb, accessor)?);
                }
            }
        }
    }

    let expected_meshes = expected["meshes"]
        .as_array()
        .ok_or("expected meshes are missing")?;
    let layers = expected_meshes[0]["color_layers"]
        .as_array()
        .ok_or("expected color layers are missing")?;
    assert_eq!(layers[0]["name"], "vertex_groups");
    for (semantic_index, expected_layer) in layers.iter().enumerate() {
        let name = expected_layer["name"]
            .as_str()
            .ok_or("color layer name is missing")?;
        let actual = &colors_by_semantic[semantic_index];
        assert!(!actual.is_empty(), "GLB {name} color accessor is missing");
        for mesh in expected_meshes {
            let layer = mesh["color_layers"]
                .as_array()
                .ok_or("expected color layers are missing")?
                .iter()
                .find(|layer| layer["name"].as_str() == Some(name))
                .ok_or_else(|| format!("expected color layer {name} is missing"))?;
            for (value_index, expected_value) in layer["values"]
                .as_array()
                .ok_or("expected color values are missing")?
                .iter()
                .enumerate()
            {
                let wanted = vec4(expected_value)?;
                assert!(
                    actual.iter().any(|color| {
                        color
                            .iter()
                            .zip(wanted)
                            .all(|(actual, wanted)| (*actual - wanted).abs() <= FLOAT_TOLERANCE)
                    }),
                    "{} value {value_index} is missing from GLB {semantic_index}",
                    mesh["name"]
                );
            }
        }
    }
    Ok(())
}

fn check_glb_material(glb: &Glb, expected: &Value) -> TestResult {
    let materials = glb.json["materials"]
        .as_array()
        .ok_or("GLB materials are missing")?;
    let material = materials
        .iter()
        .find(|material| material["name"] == expected["material"]["name"])
        .ok_or("shared material is missing from GLB")?;
    assert_floats(
        &floats(&material["pbrMetallicRoughness"]["baseColorFactor"])?,
        &floats(&expected["material"]["base_factor"])?,
        FLOAT_TOLERANCE,
        "GLB base color factor",
    );
    Ok(())
}

#[test]
fn reserved_color_layer_names_coexist_with_weights_material_render_and_exports() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, expected) = build_reserved_oracle(&blender, root.path())?;
    let (project, import_report) = import_blend(&blender, root.path(), "reserved-project", &blend)?;
    forbid_color_losses(&import_report, "Blender import");

    let scene = read_scene(&project)?;
    for expected_mesh in expected["meshes"]
        .as_array()
        .ok_or("expected meshes are missing")?
    {
        let name = expected_mesh["name"]
            .as_str()
            .ok_or("mesh name is missing")?;
        let mesh = mesh_of_node(&scene, name)?;
        check_imported_colors(mesh, expected_mesh)?;
        if expected_mesh["vertex_groups"]["Hip"].is_object() {
            check_imported_weights(&scene, name, expected_mesh)?;
        }
    }
    check_imported_material(&scene, &expected)?;

    let frames = root.path().join("rendered");
    let mut render = pot();
    render
        .arg("render")
        .arg(&project)
        .args(["--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    let render_output = checked(&mut render, "rendering reserved vertex color")?;
    let render_report: Value = serde_json::from_slice(&render_output.stdout)?;
    assert_eq!(render_report["ok"], true, "{render_report}");
    let rendered_exr = frames.join("frame_0001.exr");
    assert!(
        fs::metadata(&rendered_exr)?.len() > 0,
        "rendered EXR is empty"
    );
    let loaded = apply_operations(
        &project,
        root.path(),
        json!([{"op":"image.load","id":"reserved_render","path":rendered_exr.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    for name in ["ReservedLeft", "ReservedRight"] {
        let wanted = vec4(&expected["render_probe"]["pixels"][name])?;
        let uv: [f64; 2] = expected["render_probe"]["uv"][name]
            .as_array()
            .ok_or("render probe UV is missing")?
            .iter()
            .map(number)
            .collect::<TestResult<Vec<_>>>()?
            .try_into()
            .map_err(|_| "render probe UV is not a pair")?;
        let actual = pixel_at(&project, "reserved_render", uv)?;
        assert!(
            actual[..3].iter().any(|channel| *channel > 0.01),
            "{name} pixel is black"
        );
        assert_rgba(
            actual,
            wanted,
            RENDER_TOLERANCE,
            &format!("{name} rendered pixel"),
        );
    }

    let exported_blend = root.path().join("reserved_roundtrip.blend");
    let blend_report = export(&blender, &project, "blend", &exported_blend)?;
    assert!(
        losses(&blend_report).is_empty(),
        "blend export reported losses: {blend_report}"
    );
    let dump = dump_blend(&blender, root.path(), &exported_blend, "reserved-roundtrip")?;
    let expected_objects = expected["meshes"]
        .as_array()
        .ok_or("expected meshes are missing")?;
    for expected_mesh in expected_objects {
        let name = expected_mesh["name"]
            .as_str()
            .ok_or("mesh name is missing")?;
        let object = dumped_object(&dump, name)?;
        check_blend_colors(object, expected_mesh)?;
        if expected_mesh["vertex_groups"]["Hip"].is_object() {
            check_blend_weights(object, expected_mesh)?;
        }
        let materials = object["materials"]
            .as_array()
            .ok_or("exported Blender material slots are missing")?;
        let nodes = materials[0]["nodes"]
            .as_array()
            .ok_or("exported Blender material nodes are missing")?;
        let vertex_node = nodes
            .iter()
            .find(|node| node["type"] == "ShaderNodeVertexColor")
            .ok_or("exported Blender material lost its Vertex Color node")?;
        assert_eq!(
            vertex_node["layer_name"],
            expected["material"]["vertex_color_layer"]
        );
        let multiply = nodes
            .iter()
            .find(|node| node["type"] == "ShaderNodeMix")
            .ok_or("exported Blender material lost its multiply node")?;
        let principled = nodes
            .iter()
            .find(|node| node["type"] == "ShaderNodeBsdfPrincipled")
            .ok_or("exported Blender material lost its Principled node")?;
        assert!(
            materials[0]["links"].as_array().is_some_and(|links| {
                links.iter().any(|link| {
                    link["from_node"] == multiply["name"]
                        && link["to_node"] == principled["name"]
                        && link["to_socket"] == "Base Color"
                })
            }),
            "exported multiply no longer feeds Principled Base Color"
        );
    }

    let glb_path = root.path().join("reserved_colors.glb");
    let glb_output = export_raw(&blender, &project, "glb", &glb_path, true)?;
    assert!(
        glb_output.status.success(),
        "glTF export failed: {}",
        String::from_utf8_lossy(&glb_output.stdout)
    );
    let glb_report: Value = serde_json::from_slice(&glb_output.stdout)?;
    forbid_gltf_color_losses(&glb_report);
    let glb = parse_glb(&fs::read(glb_path)?)?;
    check_glb_material(&glb, &expected)?;
    check_glb_colors(&glb, &expected)?;
    Ok(())
}
