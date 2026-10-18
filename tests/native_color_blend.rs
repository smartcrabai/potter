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
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use native_oracle::{
    BYTE_TOLERANCE, FLOAT_TOLERANCE, TestResult, apply_operations, assert_close, assert_floats,
    assert_rgba, attribute, build_oracle, dump_blend, dumped_object, export, export_raw,
    face_loop_starts, floats, import_blend, loss_features, losses, mesh_of_node, node_id, number,
    oracle_section, pot, read_scene, run_blender_arguments, value_at, vec3, vec4, write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use std::{fs, path::Path};
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
            value_at(shade, &format!("f{}", index + 1))?,
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
            value_at(sharp, &format!("e{}", index + 1))?,
            &edge["sharp"],
            "edge {index} sharp"
        );
        assert_eq!(
            value_at(seam, &format!("e{}", index + 1))?,
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
            INTERNAL_PREFIXES
                .iter()
                .any(|prefix| name.starts_with(prefix)),
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
                        vec4(value_at(attr, &format!("v{}", index + 1))?)?,
                        vec4(value)?,
                        tolerance,
                        &format!("{name} v{index}"),
                    );
                }
            }
            Some("corner") => {
                assert_eq!(attr["domain"], "corner", "{name} domain");
                for (face, size) in sizes.iter().enumerate() {
                    let row = value_at(attr, &format!("f{}", face + 1))?
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
        "float3" => assert_floats(&vec3(actual)?, &vec3(expected)?, FLOAT_TOLERANCE, label),
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
                    let row = value_at(attr, &format!("f{}", face + 1))?
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
                        value_at(attr, &format!("{prefix}{}", index + 1))?,
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
    let faces = mesh["faces"]
        .as_array()
        .ok_or("imported faces are missing")?;
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
        for (face_index, start) in starts.iter().enumerate() {
            let face_id = faces[face_index]["id"]
                .as_u64()
                .ok_or("imported face ID is missing")?;
            let entry = layer["faces"]
                .as_array()
                .ok_or("stored UV faces are missing")?
                .iter()
                .find(|entry| entry["face_id"].as_u64() == Some(face_id))
                .ok_or_else(|| format!("UV layer {name} has no face {face_id}"))?;
            let corners = entry["uv"]
                .as_array()
                .ok_or("stored UV corners are missing")?;
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
    let Some(blender) = blender_checked::blender_executable() else {
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
    let mut preview_command = pot();
    preview_command
        .arg("preview")
        .arg(&project)
        .args(["--views", "iso", "--size", "64", "--json"]);
    let preview_output =
        native_oracle::checked(&mut preview_command, "previewing imported mesh data")?;
    let preview: Value = serde_json::from_slice(&preview_output.stdout)?;
    assert_eq!(preview["ok"], true, "{preview}");
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

#[test]
fn glb_hdr_color_layers_export_raw_and_restore_from_extras() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, expected) = build_oracle(&blender, root.path())?;
    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let oracle = oracle_section(&expected);
    let layers = oracle["color_layers"]["layers"]
        .as_object()
        .ok_or("oracle color layers are missing")?;
    let (hdr_name, hdr_layer) = layers
        .iter()
        .find(|(_, layer)| {
            layer["values"].as_array().is_some_and(|values| {
                values.iter().any(|value| {
                    value.as_array().is_some_and(|components| {
                        components.iter().any(|component| {
                            component
                                .as_f64()
                                .is_some_and(|component| !(0.0..=1.0).contains(&component))
                        })
                    })
                })
            })
        })
        .ok_or("oracle has no HDR color value")?;
    assert_eq!(hdr_layer["domain"], "corner");
    let hdr_name = hdr_name.clone();
    let hdr_layer = hdr_layer.clone();
    let hdr_values = hdr_layer["values"]
        .as_array()
        .ok_or("oracle HDR values are missing")?;
    let sizes = face_sizes(oracle)?;
    let starts = face_loop_starts(&sizes);

    let strict_path = root.path().join("strict.glb");
    let strict = export_raw(&blender, &project, "glb", &strict_path, false)?;
    assert!(
        !strict.status.success(),
        "strict HDR export must be refused"
    );
    let strict_report: Value = serde_json::from_slice(&strict.stdout)?;
    let strict_losses = strict_report["error"]["details"]["result"]["losses"]
        .as_array()
        .ok_or("strict GLB loss details are missing")?;
    assert!(
        !strict_losses.iter().any(|loss| {
            loss["feature_id"] == "mesh.color_attribute"
                && loss["reason"]
                    .as_str()
                    .is_some_and(|reason| reason.contains(&hdr_name))
        }),
        "COLOR_1 is unconstrained by glTF; strict export must not name {hdr_name}: {strict_report}"
    );

    let glb = root.path().join("allow-lossy.glb");
    let allowed = export_raw(&blender, &project, "glb", &glb, true)?;
    assert!(
        allowed.status.success(),
        "allow-lossy GLB export failed: {}",
        String::from_utf8_lossy(&allowed.stdout)
    );
    let allowed_report: Value = serde_json::from_slice(&allowed.stdout)?;
    assert!(
        !losses(&allowed_report).iter().any(|loss| {
            loss["feature_id"] == "mesh.color_attribute"
                && loss["reason"]
                    .as_str()
                    .is_some_and(|reason| reason.contains(&hdr_name))
        }),
        "COLOR_1 is unconstrained by glTF; {hdr_name} must not be a color loss"
    );
    let bytes = fs::read(&glb)?;
    assert!(bytes.len() >= 20, "GLB container is too short");
    assert_eq!(&bytes[0..4], b"glTF");
    let json_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    assert_eq!(&bytes[16..20], b"JSON");
    let json_end = 20_usize
        .checked_add(json_length)
        .filter(|end| *end <= bytes.len())
        .ok_or("GLB JSON chunk is outside the file")?;
    let document: Value = serde_json::from_slice(&bytes[20..json_end])?;
    let mesh_values = document["meshes"]
        .as_array()
        .ok_or("GLB meshes are missing")?;
    let color_mesh_index = mesh_values
        .iter()
        .position(|mesh| {
            mesh["extras"]["potter"]["color_attributes"]["layers"]
                .as_array()
                .is_some_and(|layers| {
                    layers
                        .iter()
                        .any(|layer| layer["name"].as_str() == Some(hdr_name.as_str()))
                })
                && mesh["primitives"].as_array().is_some_and(|primitives| {
                    primitives
                        .iter()
                        .any(|primitive| primitive["attributes"]["COLOR_0"].is_u64())
                })
        })
        .ok_or("GLB mesh with raw HDR color metadata is missing")?;
    let color_mesh = &mesh_values[color_mesh_index];
    let primitives = color_mesh["primitives"]
        .as_array()
        .ok_or("GLB primitives are missing")?;
    let colored = primitives
        .iter()
        .find(|primitive| primitive["attributes"]["COLOR_0"].is_u64())
        .ok_or("GLB has no selected COLOR_0")?;
    for set in 0..4 {
        assert!(colored["attributes"][format!("COLOR_{set}")].is_u64());
    }
    assert!(
        colored["attributes"]["COLOR_4"].is_null(),
        "COLOR_n accessors must be dense"
    );
    assert_eq!(colored["extras"]["potter"]["color_layer"], json!("Paint"));
    let exported_hdr_layer = color_mesh["extras"]["potter"]["color_attributes"]["layers"]
        .as_array()
        .ok_or("raw Potter color metadata is missing")?
        .iter()
        .find(|layer| layer["name"].as_str() == Some(hdr_name.as_str()))
        .ok_or("raw HDR layer is missing from Potter extras")?;
    let exported_hdr_rows = exported_hdr_layer["values"]
        .as_array()
        .ok_or("raw HDR corner values are missing")?;
    assert_eq!(exported_hdr_rows.len(), sizes.len());
    for (face_index, row) in exported_hdr_rows.iter().enumerate() {
        let row = row.as_array().ok_or("raw HDR face row is missing")?;
        let expected_row = &hdr_values[starts[face_index]..starts[face_index] + sizes[face_index]];
        assert_eq!(row.len(), expected_row.len());
        for (corner, (actual, expected)) in row.iter().zip(expected_row).enumerate() {
            assert_rgba(
                vec4(actual)?,
                vec4(expected)?,
                FLOAT_TOLERANCE,
                &format!("raw {hdr_name} face {face_index} corner {corner}"),
            );
        }
    }

    let imported_project = root.path().join("glb-import");
    let mut init = pot();
    init.arg("init").arg(&imported_project).arg("--json");
    native_oracle::checked(&mut init, "initializing GLB import")?;
    let mut import = pot();
    import
        .arg("import")
        .arg(&imported_project)
        .arg("--file")
        .arg(&glb)
        .args([
            "--format",
            "glb",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--allow-lossy",
            "--json",
        ]);
    native_oracle::checked(&mut import, "importing lossy GLB")?;
    let scene = read_scene(&imported_project)?;
    let mesh = mesh_of_node(&scene, "Oracle")?;
    let restored = attribute(mesh, &hdr_name)?;
    assert_eq!(restored["domain"], hdr_layer["domain"]);
    let registry = attribute(mesh, "color_layers")?;
    assert_eq!(string_list(&registry["layers"])?, expected_names(oracle)?);
    assert_eq!(registry["active"], "Glow");
    assert_eq!(registry["render"], "Trim");
    let faces = mesh["faces"]
        .as_array()
        .ok_or("imported faces are missing")?;
    assert_eq!(faces.len(), sizes.len());
    for (face_index, face) in faces.iter().enumerate() {
        let face_id = face["id"].as_u64().ok_or("imported face ID is missing")?;
        let actual = value_at(restored, &format!("f{face_id}"))?
            .as_array()
            .ok_or("restored HDR face values are missing")?;
        let expected = &hdr_values[starts[face_index]..starts[face_index] + sizes[face_index]];
        assert_eq!(actual.len(), expected.len());
        for (corner, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_rgba(
                vec4(actual)?,
                vec4(expected)?,
                FLOAT_TOLERANCE,
                &format!("{hdr_name} face {face_index} corner {corner}"),
            );
        }
    }
    Ok(())
}

fn check_flags_dumped(object: &Value, oracle: &Value) -> TestResult {
    for (index, face) in oracle["faces"]
        .as_array()
        .ok_or("expected faces are missing")?
        .iter()
        .enumerate()
    {
        assert_eq!(
            object["faces"][index]["smooth"], face["smooth"],
            "face {index}"
        );
        assert_eq!(
            object["faces"][index]["v"], face["v"],
            "face {index} corners"
        );
    }
    for (index, edge) in oracle["edges"]
        .as_array()
        .ok_or("expected edges are missing")?
        .iter()
        .enumerate()
    {
        assert_eq!(
            object["edges"][index]["sharp"], edge["sharp"],
            "edge {index} sharp"
        );
        assert_eq!(
            object["edges"][index]["seam"], edge["seam"],
            "edge {index} seam"
        );
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
        let actual = layer["values"]
            .as_array()
            .ok_or("Blender values are missing")?;
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
        assert_eq!(
            attr["data_type"],
            blender_data_type(kind),
            "{name} Blender type"
        );
        let values = spec["values"].as_array().ok_or("values are missing")?;
        let actual = attr["values"]
            .as_array()
            .ok_or("Blender values are missing")?;
        assert_eq!(actual.len(), values.len(), "{name} element count");
        for (index, expected) in values.iter().enumerate() {
            check_value(
                &actual[index],
                expected,
                kind,
                &format!("Blender {name}[{index}]"),
            )?;
        }
    }
    Ok(())
}

fn check_uv_dumped(object: &Value, oracle: &Value) -> TestResult {
    let uv = &oracle["uv_layers"];
    assert_eq!(object["uv_active"], uv["active"], "Blender active UV layer");
    assert_eq!(object["uv_render"], uv["render"], "Blender render UV layer");
    let layers = object["uv_layers"]
        .as_array()
        .ok_or("Blender UV layers are missing")?;
    let names: Vec<String> = layers
        .iter()
        .map(|layer| layer["name"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert_eq!(names, string_list(&uv["names"])?, "Blender UV order");
    for layer in layers {
        let name = layer["name"].as_str().unwrap_or_default();
        let expected = uv["values"][name]
            .as_array()
            .ok_or("expected UV values are missing")?;
        let actual = layer["values"]
            .as_array()
            .ok_or("Blender UV values are missing")?;
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
    let Some(blender) = blender_checked::blender_executable() else {
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
/// Blender renders the source `.blend` directly; only the named-color emitter remains visible.
/// Its centre pixel is read from the linear `OpenEXR`, independently of Potter's project/image path.
const EMITTER_SOURCE_RENDER: &str = r#"
import json
import sys

import bpy

out_json, out_exr = sys.argv[sys.argv.index("--") + 1 :]
scene = bpy.context.scene
emitter = bpy.data.objects.get("Emitter")
if emitter is None:
    raise SystemExit("source .blend has no Emitter")
for obj in scene.objects:
    obj.hide_render = obj is not emitter

camera_data = bpy.data.cameras.new("EmitterOracleCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("EmitterOracleCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 6.0)
scene.camera = camera
scene.world = None

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 8
scene.cycles.max_bounces = 0
scene.cycles.use_denoising = False
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 0.01
scene.render.resolution_x = 16
scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "Standard"
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.color_mode = "RGBA"
scene.render.filepath = out_exr
bpy.ops.render.render(write_still=True)

image = bpy.data.images.load(out_exr, check_existing=False)
if tuple(image.size) != (16, 16):
    raise SystemExit("source render has unexpected dimensions: %r" % (tuple(image.size),))
pixels = list(image.pixels)
offset = ((16 // 2) * 16 + 16 // 2) * 4
with open(out_json, "w", encoding="utf-8") as handle:
    json.dump({"pixel": pixels[offset : offset + 4]}, handle)
"#;

fn blender_emitter_center_pixel(
    blender: &Path,
    root: &Path,
    source_blend: &Path,
) -> TestResult<[f64; 4]> {
    let script = write_script(root, "render_emitter_source.py", EMITTER_SOURCE_RENDER)?;
    let json_path = root.join("emitter_source_pixel.json");
    let exr_path = root.join("emitter_source.exr");
    run_blender_arguments(
        blender,
        &script,
        Some(source_blend),
        &[&json_path, &exr_path],
    )?;
    let render: Value = serde_json::from_slice(&fs::read(json_path)?)?;
    vec4(&render["pixel"])
}

fn render_center_pixel(project: &Path, image_id: &str) -> TestResult<[f64; 4]> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][image_id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    Ok(sample(&data, [0.5, 0.5], 0))
}

#[test]
fn imported_emitter_color_renders_like_blender_after_blend_roundtrip() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (source_blend, expected) = build_oracle(&blender, root.path())?;

    let expected_ink = vec4(&expected["emitter"]["value"])?;
    assert_eq!(expected["emitter"]["layer"], "Ink");
    assert_eq!(expected_ink, [0.0625, 0.25, 0.5, 1.0]);
    assert_eq!(
        expected["emitter"]["camera"],
        json!({
            "location": [0.0, 0.0, 6.0],
            "projection": "orthographic",
            "ortho_scale": 4.0
        })
    );
    assert_eq!(expected["emitter"]["pixel"], json!([0.5, 0.5]));

    let blender_pixel = blender_emitter_center_pixel(&blender, root.path(), &source_blend)?;
    assert_rgba(
        blender_pixel,
        expected_ink,
        FLOAT_TOLERANCE,
        "Blender source-render Emitter centre",
    );

    let (project, import_report) = import_blend(&blender, root.path(), "project", &source_blend)?;
    for feature in loss_features(&import_report) {
        assert!(
            !UNEXPECTED_IMPORT_LOSSES.contains(&feature.as_str()),
            "native import lost representable data: {feature}"
        );
    }
    let exported_blend = root.path().join("roundtrip.blend");
    let export_report = export(&blender, &project, "blend", &exported_blend)?;
    assert!(
        losses(&export_report).is_empty(),
        "blend export lost data before the native render: {export_report}"
    );
    let (roundtrip, reimport_report) =
        import_blend(&blender, root.path(), "roundtrip", &exported_blend)?;
    for feature in loss_features(&reimport_report) {
        assert!(
            !UNEXPECTED_IMPORT_LOSSES.contains(&feature.as_str()),
            "roundtrip import lost representable data: {feature}"
        );
    }

    // Confirm the re-imported native layer and named shader input feeding the actual CLI render.
    let scene = read_scene(&roundtrip)?;
    let emitter = mesh_of_node(&scene, "Emitter")?;
    let ink = attribute(emitter, "Ink")?;
    assert_eq!(ink["domain"], "corner");
    assert_eq!(ink["type"], "color");
    let face_id = emitter["faces"]
        .as_array()
        .and_then(|faces| faces.first())
        .and_then(|face| face["id"].as_u64())
        .ok_or("Emitter's first face has no id")?;
    let corners = value_at(ink, &format!("f{face_id}"))?
        .as_array()
        .ok_or("Emitter Ink face values are not an array")?;
    assert_eq!(corners.len(), 4);
    for (index, corner) in corners.iter().enumerate() {
        assert_rgba(
            vec4(corner)?,
            expected_ink,
            FLOAT_TOLERANCE,
            &format!("re-imported Ink corner {index}"),
        );
    }
    let emitter_id = node_id(&scene, "Emitter")?;
    let emitter = &scene["nodes"][&emitter_id];
    // The emitter's material is linked to its mesh data (build_oracle.py), so its slot lives on the data-block.
    let object_linked = emitter["material_links"][0].as_str() == Some("object");
    let slots = if object_linked {
        &emitter["materials"]
    } else {
        let data = emitter["data"].as_str().ok_or("Emitter data is missing")?;
        &scene["data_blocks"][data]["materials"]
    };
    let material_id = slots[0]
        .as_str()
        .ok_or("Emitter has no material assignment")?;
    let graph_id = scene["materials"][material_id]["node_tree"]
        .as_str()
        .ok_or("Emitter material has no node tree")?;
    let graph_nodes = scene["node_groups"][graph_id]["nodes"]
        .as_object()
        .ok_or("Emitter material nodes are missing")?;
    let vertex_color = graph_nodes
        .values()
        .find(|node| node["type"] == "ShaderNodeVertexColor")
        .ok_or("Emitter material has no Vertex Color node")?;
    assert_eq!(vertex_color["properties"]["layer_name"], "Ink");

    let active_scene = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &roundtrip,
        root.path(),
        json!([
            {"op":"camera.create","id":"emitter_camera","transform":{"translation":[0.0,0.0,6.0]},"projection":"orthographic","ortho_scale":4.0},
            {"op":"scene.update","target":{"id":active_scene},"set":{"camera":"emitter_camera"}},
            {"op":"render.update","target":{"id":active_scene},"set":{"resolution_x":16,"resolution_y":16,"samples":8,"seed":0,"max_bounces":0}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    let frames = root.path().join("native-render");
    let mut render = pot();
    render
        .arg("render")
        .arg(&roundtrip)
        .args(["--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    let rendered = native_oracle::checked(&mut render, "rendering re-imported native Ink")?;
    let stdout = String::from_utf8_lossy(&rendered.stdout);
    assert!(
        !stdout.contains("color_layer_missing"),
        "the named Ink layer exists: {stdout}"
    );
    let exr = frames.join("frame_0001.exr");
    assert!(fs::metadata(&exr)?.len() > 0, "native EXR render is empty");
    let loaded = apply_operations(
        &roundtrip,
        root.path(),
        json!([{"op":"image.load","id":"emitter_render","path":exr.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");

    let potter_pixel = render_center_pixel(&roundtrip, "emitter_render")?;
    assert_rgba(
        potter_pixel,
        blender_pixel,
        FLOAT_TOLERANCE,
        "Potter CLI render versus Blender source-render centre",
    );
    assert_rgba(
        potter_pixel,
        expected_ink,
        FLOAT_TOLERANCE,
        "Potter CLI render versus independently authored linear Ink",
    );
    Ok(())
}
