//! PR9 native mesh data against Blender 5.2.2.
//!
//! The oracle `.blend` (built by `tests/fixtures/native_oracle/build_oracle.py`) holds mixed
//! flat/smooth faces, sharp and seam edges, four color layers (POINT and CORNER, FLOAT and BYTE,
//! with alpha and an HDR value), three UV layers with active and render different, user generic
//! attributes on every domain, and two vertex groups. The first test checks that `pot import`
//! stores each of these natively and that no retired key remains. The second checks that
//! `pot export --format blend` gives Blender the same flags, colors, names, active and render
//! indices, generic values, UV layers, and groups.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;

use native_oracle::{
    BYTE_TOLERANCE, FLOAT_TOLERANCE, TestResult, assert_close, assert_floats, assert_rgba, attribute,
    blender, build_oracle, dump_blend, dumped_object, export, face_loop_starts, floats, import_blend,
    loss_features, losses, mesh_of_node, node_id, number, oracle_section, read_scene, value_at,
    vec3, vec4,
};
use serde_json::Value;
use tempfile::tempdir;

/// Features that a `.blend` import must never report as lost for the oracle.
const UNEXPECTED_IMPORT_LOSSES: [&str; 4] = [
    "mesh.attribute",
    "mesh.color_attribute",
    "mesh.uv_layer",
    "material.shader_graph.vertex_color",
];

/// Blender-only `blender_attributes` entries that the native store may keep: selection, hide, and
/// sculpt state. Mirrors and topology are recomputed, so they must not appear here.
const INTERNAL_PREFIXES: [&str; 4] = [".select_", ".uv_select_", ".hide_", ".sculpt_"];

fn face_sizes(oracle: &Value) -> TestResult<Vec<usize>> {
    oracle["faces"]
        .as_array()
        .ok_or("expected faces are missing")?
        .iter()
        .map(|face| {
            face["v"]
                .as_array()
                .map(Vec::len)
                .ok_or_else(|| "expected face has no vertices".into())
        })
        .collect()
}

fn string_list(value: &Value) -> TestResult<Vec<String>> {
    value
        .as_array()
        .ok_or("value is not an array")?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "value is not a string".into())
        })
        .collect()
}

/// The color layer names of the oracle in Blender order.
fn expected_names(oracle: &Value) -> TestResult<Vec<String>> {
    string_list(&oracle["color_layers"]["order"])
}

fn expected_color(oracle: &Value, name: &str) -> TestResult<Value> {
    oracle["color_layers"]["layers"]
        .get(name)
        .cloned()
        .ok_or_else(|| format!("oracle has no color layer {name}").into())
}

fn check_shading(mesh: &Value, oracle: &Value) -> TestResult {
    let attributes = &mesh["attributes"];
    assert!(
        attributes.get("blender_polygon_smooth").is_none(),
        "blender_polygon_smooth is a retired key"
    );
    assert!(
        attributes.get("blender_edge_flags").is_none(),
        "blender_edge_flags is a retired key"
    );
    let shade = attribute(mesh, "shade_smooth")?;
    assert_eq!(shade["domain"], "face");
    assert_eq!(shade["type"], "bool");
    let sharp = attribute(mesh, "sharp")?;
    assert_eq!(sharp["domain"], "edge");
    assert_eq!(sharp["type"], "bool");
    let seam = attribute(mesh, "seam")?;
    assert_eq!(seam["domain"], "edge");
    assert_eq!(seam["type"], "bool");
    for (index, face) in oracle["faces"]
        .as_array()
        .ok_or("expected faces are missing")?
        .iter()
        .enumerate()
    {
        assert_eq!(
            value_at(shade, &format!("f{index}"))?,
            &face["smooth"],
            "face {index} shade_smooth"
        );
    }
    for (index, edge) in oracle["edges"]
        .as_array()
        .ok_or("expected edges are missing")?
        .iter()
        .enumerate()
    {
        assert_eq!(
            value_at(sharp, &format!("e{index}"))?,
            &edge["sharp"],
            "edge {index} sharp"
        );
        assert_eq!(
            value_at(seam, &format!("e{index}"))?,
            &edge["seam"],
            "edge {index} seam"
        );
    }
    Ok(())
}

fn check_internals(mesh: &Value) -> TestResult {
    let Some(entries) = mesh["attributes"]["blender_attributes"].as_array() else {
        return Ok(());
    };
    for entry in entries {
        let name = entry["name"]
            .as_str()
            .ok_or("blender attribute has no name")?;
        assert!(
            INTERNAL_PREFIXES.iter().any(|prefix| name.starts_with(prefix)),
            "{name} must be native or dropped, not kept as a Blender-only attribute"
        );
    }
    Ok(())
}

fn check_colors(mesh: &Value, oracle: &Value) -> TestResult {
    let store = attribute(mesh, "color_layers")?;
    let names = expected_names(oracle)?;
    assert_eq!(string_list(&store["layers"])?, names, "layer order");
    assert_eq!(store["active"], "Glow", "active color layer");
    assert_eq!(store["render"], "Trim", "render color layer");
    let metadata = attribute(mesh, "blender_metadata")?;
    let sizes = face_sizes(oracle)?;
    let starts = face_loop_starts(&sizes);
    for name in &names {
        let expected = expected_color(oracle, name)?;
        let attr = attribute(mesh, name)?;
        let source = expected["source"].as_str().ok_or("source is missing")?;
        let tolerance = if source == "BYTE_COLOR" {
            BYTE_TOLERANCE
        } else {
            FLOAT_TOLERANCE
        };
        assert_eq!(attr["type"], "color", "{name} type");
        assert_eq!(
            metadata["color_data_types"][name], source,
            "{name} source encoding"
        );
        let values = expected["values"]
            .as_array()
            .ok_or("expected color values are missing")?;
        match expected["domain"].as_str() {
            Some("point") => {
                assert_eq!(attr["domain"], "point", "{name} domain");
                for (index, value) in values.iter().enumerate() {
                    assert_rgba(
                        vec4(value_at(attr, &format!("v{index}"))?)?,
                        vec4(value)?,
                        tolerance,
                        &format!("{name} v{index}"),
                    );
                }
            }
            Some("corner") => {
                assert_eq!(attr["domain"], "corner", "{name} domain");
                for (face, size) in sizes.iter().enumerate() {
                    let row = value_at(attr, &format!("f{face}"))?
                        .as_array()
                        .ok_or("corner row is not an array")?;
                    assert_eq!(row.len(), *size, "{name} face {face} corner count");
                    for corner in 0..*size {
                        assert_rgba(
                            vec4(&row[corner])?,
                            vec4(&values[starts[face] + corner])?,
                            tolerance,
                            &format!("{name} face {face} corner {corner}"),
                        );
                    }
                }
            }
            other => return Err(format!("unexpected domain {other:?}").into()),
        }
    }
    Ok(())
}

/// Expected generic attribute type name (Potter convention) to Blender's data type.
fn blender_data_type(potter: &str) -> &'static str {
    match potter {
        "float" => "FLOAT",
        "int" => "INT",
        "bool" => "BOOLEAN",
        _ => "FLOAT_VECTOR",
    }
}

fn check_value(actual: &Value, expected: &Value, kind: &str, label: &str) -> TestResult {
    match kind {
        "bool" => assert_eq!(actual.as_bool(), expected.as_bool(), "{label}"),
        "int" => assert_eq!(actual.as_i64(), expected.as_i64(), "{label}"),
        "float3" => assert_floats(
            &vec3(actual)?,
            &vec3(expected)?,
            FLOAT_TOLERANCE,
            label,
        ),
        _ => assert_close(number(actual)?, number(expected)?, FLOAT_TOLERANCE, label),
    }
    Ok(())
}

fn check_generics(mesh: &Value, oracle: &Value) -> TestResult {
    let sizes = face_sizes(oracle)?;
    let starts = face_loop_starts(&sizes);
    let generic = oracle["generic"]
        .as_object()
        .ok_or("expected generic attributes are missing")?;
    for (name, spec) in generic {
        let attr = attribute(mesh, name)?;
        let domain = spec["domain"].as_str().ok_or("domain is missing")?;
        let kind = spec["type"].as_str().ok_or("type is missing")?;
        assert_eq!(attr["domain"], domain, "{name} domain");
        assert_eq!(attr["type"], kind, "{name} type");
        let values = spec["values"].as_array().ok_or("values are missing")?;
        match domain {
            "corner" => {
                for (face, size) in sizes.iter().enumerate() {
                    let row = value_at(attr, &format!("f{face}"))?
                        .as_array()
                        .ok_or("corner row is not an array")?;
                    assert_eq!(row.len(), *size, "{name} face {face} corner count");
                    for corner in 0..*size {
                        check_value(
                            &row[corner],
                            &values[starts[face] + corner],
                            kind,
                            &format!("{name} face {face} corner {corner}"),
                        )?;
                    }
                }
            }
            "point" | "edge" | "face" => {
                let prefix = match domain {
                    "point" => "v",
                    "edge" => "e",
                    _ => "f",
                };
                for (index, expected) in values.iter().enumerate() {
                    check_value(
                        value_at(attr, &format!("{prefix}{index}"))?,
                        expected,
                        kind,
                        &format!("{name}[{index}]"),
                    )?;
                }
            }
            other => return Err(format!("unexpected domain {other}").into()),
        }
    }
    Ok(())
}

fn check_uv(mesh: &Value, oracle: &Value) -> TestResult {
    let store = attribute(mesh, "uv_layers")?;
    let uv = &oracle["uv_layers"];
    let names = string_list(&uv["names"])?;
    let layer_names: Vec<String> = store["layers"]
        .as_array()
        .ok_or("stored UV layers are missing")?
        .iter()
        .map(|layer| layer["name"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert_eq!(layer_names, names, "UV layer order");
    assert_eq!(store["active"], uv["active"], "UV active layer");
    assert_eq!(store["render"], uv["render"], "UV render layer");
    let starts = face_loop_starts(&face_sizes(oracle)?);
    for name in &names {
        let layer = store["layers"]
            .as_array()
            .ok_or("stored UV layers are missing")?
            .iter()
            .find(|layer| layer["name"] == name.as_str())
            .ok_or_else(|| format!("UV layer {name} is missing"))?;
        let expected = uv["values"][name]
            .as_array()
            .ok_or("expected UV values are missing")?;
        for (face, start) in starts.iter().enumerate() {
            let entry = layer["faces"]
                .as_array()
                .ok_or("stored UV faces are missing")?
                .iter()
                .find(|entry| entry["face_id"].as_u64() == Some(face as u64))
                .ok_or_else(|| format!("UV layer {name} has no face {face}"))?;
            let corners = entry["uv"].as_array().ok_or("stored UV corners are missing")?;
            for (corner, stored) in corners.iter().enumerate() {
                let expected_uv = floats(&expected[start + corner])?;
                let stored = floats(stored)?;
                // Storage is glTF's V-down convention: stored v = 1 - Blender v.
                assert_close(stored[0], expected_uv[0], FLOAT_TOLERANCE, name);
                assert_close(stored[1], 1.0 - expected_uv[1], FLOAT_TOLERANCE, name);
            }
        }
    }
    Ok(())
}

fn check_groups(scene: &Value) -> TestResult {
    let id = node_id(scene, "Oracle")?;
    let data = scene["nodes"][&id]["data"]
        .as_str()
        .ok_or("oracle node has no data block")?;
    let names: Vec<&str> = scene["data_blocks"][data]["vertex_groups"]
        .as_array()
        .ok_or("vertex groups are missing")?
        .iter()
        .filter_map(|group| group["name"].as_str())
        .collect();
    assert!(names.contains(&"Top"), "vertex group Top is missing");
    assert!(names.contains(&"Side"), "vertex group Side is missing");
    Ok(())
}

#[test]
fn import_stores_shading_colors_generics_uv_and_groups_natively() -> TestResult {
    let Some(blender) = blender() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, expected) = build_oracle(&blender, root.path())?;
    let (project, report) = import_blend(&blender, root.path(), "project", &blend)?;
    for feature in loss_features(&report) {
        assert!(
            !UNEXPECTED_IMPORT_LOSSES.contains(&feature.as_str()),
            "import reported {feature} for representable oracle data"
        );
    }
    let scene = read_scene(&project)?;
    let oracle = oracle_section(&expected);
    let mesh = mesh_of_node(&scene, "Oracle")?;
    check_shading(mesh, oracle)?;
    check_internals(mesh)?;
    check_colors(mesh, oracle)?;
    check_generics(mesh, oracle)?;
    check_uv(mesh, oracle)?;
    check_groups(&scene)?;
    Ok(())
}

fn check_flags_dumped(object: &Value, oracle: &Value) -> TestResult {
    for (index, face) in oracle["faces"]
        .as_array()
        .ok_or("expected faces are missing")?
        .iter()
        .enumerate()
    {
        assert_eq!(object["faces"][index]["smooth"], face["smooth"], "face {index}");
        assert_eq!(object["faces"][index]["v"], face["v"], "face {index} corners");
    }
    for (index, edge) in oracle["edges"]
        .as_array()
        .ok_or("expected edges are missing")?
        .iter()
        .enumerate()
    {
        assert_eq!(object["edges"][index]["sharp"], edge["sharp"], "edge {index} sharp");
        assert_eq!(object["edges"][index]["seam"], edge["seam"], "edge {index} seam");
    }
    for (index, vertex) in oracle["vertices"]
        .as_array()
        .ok_or("expected vertices are missing")?
        .iter()
        .enumerate()
    {
        assert_floats(
            &vec3(&object["vertices"][index])?,
            &vec3(vertex)?,
            FLOAT_TOLERANCE,
            &format!("vertex {index}"),
        );
    }
    Ok(())
}

fn check_colors_dumped(object: &Value, oracle: &Value) -> TestResult {
    let names = expected_names(oracle)?;
    let layers = object["color_layers"]
        .as_array()
        .ok_or("Blender color layers are missing")?;
    let order: Vec<String> = layers
        .iter()
        .map(|layer| layer["name"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert_eq!(order, names, "Blender color layer order");
    assert_eq!(object["color_active"], 1, "active color index");
    assert_eq!(object["color_render"], 3, "render color index");
    for layer in layers {
        let name = layer["name"].as_str().unwrap_or_default();
        let expected = expected_color(oracle, name)?;
        let source = expected["source"].as_str().ok_or("source is missing")?;
        assert_eq!(layer["data_type"], source, "{name} Blender data type");
        let tolerance = if source == "BYTE_COLOR" {
            BYTE_TOLERANCE
        } else {
            FLOAT_TOLERANCE
        };
        let values = expected["values"]
            .as_array()
            .ok_or("expected color values are missing")?;
        let actual = layer["values"].as_array().ok_or("Blender values are missing")?;
        assert_eq!(actual.len(), values.len(), "{name} element count");
        for (index, value) in values.iter().enumerate() {
            assert_rgba(
                vec4(&actual[index])?,
                vec4(value)?,
                tolerance,
                &format!("{name} element {index}"),
            );
        }
    }
    Ok(())
}

fn check_generics_dumped(object: &Value, oracle: &Value) -> TestResult {
    let attributes = object["attributes"]
        .as_array()
        .ok_or("Blender attributes are missing")?;
    let generic = oracle["generic"]
        .as_object()
        .ok_or("expected generic attributes are missing")?;
    for (name, spec) in generic {
        let attr = attributes
            .iter()
            .find(|attr| attr["name"].as_str() == Some(name.as_str()))
            .ok_or_else(|| format!("Blender has no attribute {name}"))?;
        let domain = spec["domain"].as_str().unwrap_or_default().to_uppercase();
        let kind = spec["type"].as_str().unwrap_or_default();
        assert_eq!(attr["domain"], domain.as_str(), "{name} Blender domain");
        assert_eq!(attr["data_type"], blender_data_type(kind), "{name} Blender type");
        let values = spec["values"].as_array().ok_or("values are missing")?;
        let actual = attr["values"].as_array().ok_or("Blender values are missing")?;
        assert_eq!(actual.len(), values.len(), "{name} element count");
        for (index, expected) in values.iter().enumerate() {
            check_value(&actual[index], expected, kind, &format!("Blender {name}[{index}]"))?;
        }
    }
    Ok(())
}

fn check_uv_dumped(object: &Value, oracle: &Value) -> TestResult {
    let uv = &oracle["uv_layers"];
    assert_eq!(object["uv_active"], 2, "Blender active UV index");
    assert_eq!(object["uv_render"], 1, "Blender render UV index");
    let layers = object["uv_layers"].as_array().ok_or("Blender UV layers are missing")?;
    let names: Vec<String> = layers
        .iter()
        .map(|layer| layer["name"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert_eq!(names, string_list(&uv["names"])?, "Blender UV order");
    for layer in layers {
        let name = layer["name"].as_str().unwrap_or_default();
        let expected = uv["values"][name].as_array().ok_or("expected UV values are missing")?;
        let actual = layer["values"].as_array().ok_or("Blender UV values are missing")?;
        for (loop_index, value) in expected.iter().enumerate() {
            assert_floats(
                &floats(&actual[loop_index])?,
                &floats(value)?,
                FLOAT_TOLERANCE,
                &format!("{name} loop {loop_index}"),
            );
        }
    }
    Ok(())
}

fn check_groups_dumped(object: &Value, oracle: &Value) -> TestResult {
    let groups = object["vertex_groups"]
        .as_array()
        .ok_or("Blender vertex groups are missing")?;
    for (name, members) in oracle["vertex_groups"]
        .as_object()
        .ok_or("expected groups are missing")?
    {
        let group = groups
            .iter()
            .find(|group| group["name"].as_str() == Some(name.as_str()))
            .ok_or_else(|| format!("Blender has no group {name}"))?;
        let expected = members.as_object().ok_or("group members are missing")?;
        assert_eq!(
            group["weights"].as_object().map(serde_json::Map::len),
            Some(expected.len()),
            "{name} member count"
        );
        for (vertex, weight) in expected {
            assert_close(
                number(&group["weights"][vertex])?,
                number(weight)?,
                FLOAT_TOLERANCE,
                &format!("{name} vertex {vertex}"),
            );
        }
    }
    Ok(())
}

#[test]
fn blend_export_gives_blender_the_same_flags_colors_generics_uv_and_groups() -> TestResult {
    let Some(blender) = blender() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, expected) = build_oracle(&blender, root.path())?;
    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let exported_blend = root.path().join("roundtrip.blend");
    let exported = export(&blender, &project, "blend", &exported_blend)?;
    assert!(
        losses(&exported).is_empty(),
        "blend export must not lose PR9 mesh data: {exported}"
    );
    let dump = dump_blend(&blender, root.path(), &exported_blend, "roundtrip")?;
    let object = dumped_object(&dump, "Oracle")?;
    let oracle = oracle_section(&expected);
    check_flags_dumped(object, oracle)?;
    check_colors_dumped(object, oracle)?;
    check_generics_dumped(object, oracle)?;
    check_uv_dumped(object, oracle)?;
    check_groups_dumped(object, oracle)?;
    Ok(())
}
