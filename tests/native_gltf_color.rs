//! PR9 glTF color export and import against the Blender oracle.
//!
//! `COLOR_k` accessors must hold the stored linear values bit for bit. A primitive is compared as
//! a set of per-corner tuples (position and every `COLOR_k` as float32 bits), so glTF's own vertex
//! splitting is free but each corner's color must sit with its own position. The oracle's
//! material multiplies the layer `Paint` into its base color, so `COLOR_0` must be `Paint`, and
//! the other layers follow in layer order. The emitter's material reads `Ink`, the only layer.
//! Strict export refuses the generic attributes; `--allow-lossy` writes the file and reports them.

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

use std::{collections::BTreeSet, fs};

use native_oracle::{
    TestResult, attribute, blender, build_oracle, export_raw, import_blend, loss_features,
    mesh_of_node, pot, read_scene, value_at, vec3, vec4,
};
use serde_json::Value;
use tempfile::tempdir;

/// Blender-convention layer order of the oracle's `Oracle` mesh, with the material's layer first.
const ORACLE_COLUMNS: [&str; 4] = ["Paint", "Glow", "Dirt", "Trim"];

struct Glb {
    json: Value,
    binary: Vec<u8>,
}

fn parse_glb(bytes: &[u8]) -> TestResult<Glb> {
    if bytes.get(0..4) != Some(b"glTF".as_slice()) {
        return Err("output is not a GLB".into());
    }
    let json_length = u32_at(bytes, 12)? as usize;
    let json = serde_json::from_slice(
        bytes
            .get(20..20 + json_length)
            .ok_or("GLB JSON chunk is truncated")?,
    )?;
    let binary_header = 20 + json_length;
    let binary_length = u32_at(bytes, binary_header)? as usize;
    let binary = bytes
        .get(binary_header + 8..binary_header + 8 + binary_length)
        .ok_or("GLB BIN chunk is truncated")?
        .to_vec();
    Ok(Glb { json, binary })
}

fn u32_at(bytes: &[u8], offset: usize) -> TestResult<u32> {
    let raw: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or("GLB header is truncated")?
        .try_into()?;
    Ok(u32::from_le_bytes(raw))
}

/// The float32 bit patterns of every element of a float accessor, as `width` words per element.
fn accessor_bits(glb: &Glb, index: u64, width: usize) -> TestResult<Vec<Vec<u32>>> {
    let accessor = &glb.json["accessors"][usize::try_from(index)?];
    if accessor["componentType"] != 5126 {
        return Err("accessor must be float32".into());
    }
    let count = usize::try_from(
        accessor["count"]
            .as_u64()
            .ok_or("accessor count is missing")?,
    )?;
    let view = &glb.json["bufferViews"][usize::try_from(
        accessor["bufferView"]
            .as_u64()
            .ok_or("accessor has no bufferView")?,
    )?];
    let view_offset = usize::try_from(view["byteOffset"].as_u64().unwrap_or(0))?;
    let accessor_offset = usize::try_from(accessor["byteOffset"].as_u64().unwrap_or(0))?;
    let stride = match view["byteStride"].as_u64() {
        Some(stride) => usize::try_from(stride)?,
        None => width * 4,
    };
    let mut rows = Vec::with_capacity(count);
    for element in 0..count {
        let mut row = Vec::with_capacity(width);
        for component in 0..width {
            let offset = view_offset + accessor_offset + element * stride + component * 4;
            let raw: [u8; 4] = glb
                .binary
                .get(offset..offset + 4)
                .ok_or("accessor reads past the BIN chunk")?
                .try_into()?;
            row.push(u32::from_le_bytes(raw));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn bits(values: &[f64]) -> Vec<u32> {
    values
        .iter()
        .map(|value| (*value as f32).to_bits())
        .collect()
}

/// Per-corner tuples of a stored mesh: position, then each named layer's value at that corner.
fn expanded_corners(mesh: &Value, layers: &[&str]) -> TestResult<BTreeSet<Vec<u32>>> {
    let vertices = mesh["vertices"]
        .as_array()
        .ok_or("mesh vertices are missing")?;
    let mut corners = BTreeSet::new();
    for face in mesh["faces"].as_array().ok_or("mesh faces are missing")? {
        let face_id = face["id"].as_u64().ok_or("face id is missing")?;
        let refs = face["v"].as_array().ok_or("face vertices are missing")?;
        for (corner, vertex_ref) in refs.iter().enumerate() {
            let vertex_id = vertex_ref.as_u64().ok_or("vertex reference is missing")?;
            let position = vec3(&vertices[usize::try_from(vertex_id)?]["co"])?;
            let mut key = bits(&position);
            for layer in layers {
                let attr = attribute(mesh, layer)?;
                let color = match attr["domain"].as_str() {
                    Some("point") => vec4(value_at(attr, &format!("v{vertex_id}"))?)?,
                    Some("corner") => {
                        let row = value_at(attr, &format!("f{face_id}"))?;
                        vec4(&row[corner])?
                    }
                    other => return Err(format!("layer {layer} has domain {other:?}").into()),
                };
                key.extend(bits(&color));
            }
            corners.insert(key);
        }
    }
    Ok(corners)
}

/// Per-corner tuples of a glTF primitive, from its `POSITION` and `COLOR_k` accessors.
fn primitive_corners(glb: &Glb, primitive: &Value) -> TestResult<BTreeSet<Vec<u32>>> {
    let attributes = &primitive["attributes"];
    let position = accessor_bits(
        glb,
        attributes["POSITION"]
            .as_u64()
            .ok_or("POSITION is missing")?,
        3,
    )?;
    let mut columns = vec![position];
    let mut color = 0;
    while let Some(index) = attributes[format!("COLOR_{color}")].as_u64() {
        columns.push(accessor_bits(glb, index, 4)?);
        color += 1;
    }
    let count = columns[0].len();
    let mut corners = BTreeSet::new();
    for element in 0..count {
        let mut key = Vec::new();
        for column in &columns {
            if column.len() != count {
                return Err("COLOR accessor count differs from POSITION".into());
            }
            key.extend(&column[element]);
        }
        corners.insert(key);
    }
    Ok(corners)
}

/// Every primitive of every mesh in the GLB, as the set of its per-corner tuples.
fn glb_primitives(glb: &Glb) -> TestResult<Vec<BTreeSet<Vec<u32>>>> {
    let mut sets = Vec::new();
    for mesh in glb.json["meshes"].as_array().ok_or("GLB has no meshes")? {
        for primitive in mesh["primitives"]
            .as_array()
            .ok_or("mesh has no primitives")?
        {
            sets.push(primitive_corners(glb, primitive)?);
        }
    }
    Ok(sets)
}

fn color_columns(mesh: &Value) -> TestResult<Vec<String>> {
    attribute(mesh, "color_layers")?["layers"]
        .as_array()
        .ok_or("color layer list is missing")?
        .iter()
        .map(|name| {
            name.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "layer name is not a string".into())
        })
        .collect()
}

#[test]
fn glb_color_accessors_hold_stored_linear_values_and_strict_export_reports_generics() -> TestResult
{
    let Some(blender) = blender() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, _) = build_oracle(&blender, root.path())?;
    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;

    let strict_path = root.path().join("strict.glb");
    let strict = export_raw(&blender, &project, "glb", &strict_path, false)?;
    assert!(
        !strict.status.success(),
        "strict glTF export must refuse generic attributes"
    );
    let strict_text = String::from_utf8_lossy(&strict.stdout);
    assert!(
        strict_text.contains("mesh.attribute"),
        "strict refusal must name mesh.attribute: {strict_text}"
    );
    assert!(
        !strict_text.contains("mesh.color_attribute"),
        "representable color layers must not be losses: {strict_text}"
    );

    let out = root.path().join("oracle.glb");
    let lossy = export_raw(&blender, &project, "glb", &out, true)?;
    assert!(lossy.status.success(), "--allow-lossy must write the GLB");
    let report: Value = serde_json::from_slice(&lossy.stdout)?;
    let features = loss_features(&report);
    assert!(
        features.iter().any(|feature| feature == "mesh.attribute"),
        "generic attributes must be reported as losses: {features:?}"
    );
    assert!(
        !features
            .iter()
            .any(|feature| feature == "mesh.color_attribute"),
        "color layers are representable in glTF: {features:?}"
    );
    assert!(
        !features
            .iter()
            .any(|feature| feature == "material.shader_graph.vertex_color"),
        "the multiplied Vertex Color is representable: {features:?}"
    );

    let glb = parse_glb(&fs::read(&out)?)?;
    let oracle = mesh_of_node(&scene, "Oracle")?;
    assert_eq!(color_columns(oracle)?, ORACLE_COLUMNS, "stored color order");
    let expected = expanded_corners(oracle, &ORACLE_COLUMNS)?;
    let primitives = glb_primitives(&glb)?;
    assert!(
        primitives.iter().any(|set| *set == expected),
        "no primitive holds the Oracle corners with COLOR_0..COLOR_3 = {ORACLE_COLUMNS:?}"
    );

    let emitter = mesh_of_node(&scene, "Emitter")?;
    let emitter_expected = expanded_corners(emitter, &["Ink"])?;
    assert!(
        primitives.iter().any(|set| *set == emitter_expected),
        "the emitter's material reads Ink, which must be its COLOR_0"
    );
    Ok(())
}

#[test]
fn glb_colors_import_back_into_color_layers_per_corner() -> TestResult {
    let Some(blender) = blender() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, _) = build_oracle(&blender, root.path())?;
    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;
    let expected = expanded_corners(mesh_of_node(&scene, "Oracle")?, &ORACLE_COLUMNS)?;

    let out = root.path().join("oracle.glb");
    export_raw(&blender, &project, "glb", &out, true)?;
    let back = root.path().join("back");
    pot().arg("init").arg(&back).output()?;
    let imported = pot()
        .arg("import")
        .arg(&back)
        .arg("--file")
        .arg(&out)
        .args([
            "--format",
            "glb",
            "--base-revision",
            "0",
            "--mode",
            "replace",
            "--json",
        ])
        .output()?;
    assert!(imported.status.success(), "glTF import must succeed");
    let back_scene = read_scene(&back)?;
    let blocks = back_scene["data_blocks"]
        .as_object()
        .ok_or("imported data blocks are missing")?;
    let mesh = blocks
        .values()
        .filter_map(|block| block.get("mesh"))
        .find(|mesh| {
            mesh["attributes"]["color_layers"]["layers"]
                .as_array()
                .is_some_and(|layers| layers.len() == ORACLE_COLUMNS.len())
        })
        .ok_or("imported glTF has no color layers")?;
    let names = color_columns(mesh)?;
    assert_eq!(names.len(), ORACLE_COLUMNS.len(), "imported layer count");
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    assert_eq!(
        expanded_corners(mesh, &refs)?,
        expected,
        "imported corners must match the exported per-corner colors"
    );
    Ok(())
}
