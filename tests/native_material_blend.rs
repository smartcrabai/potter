//! PR9 textured field material through `.blend`, checked by Blender's own rendering.
//!
//! The oracle's `TexturePlane` carries an asymmetric 4 x 4 PNG (every row and every column holds
//! distinct colors; the last row is blue). Its UV layer `Base` is flipped vertically and `Light`
//! is the identity; `Base` is active and `Light` is the render layer. The field material samples
//! `Base` for its base color, and the roughness texture uses the render layer (`uv_layer: null`).
//!
//! Round trips alone cannot show the orientation: two flips would cancel. So the proof is done in
//! Blender: `render_texture.py` renders the reopened file with the texture as the only light, a
//! box filter, and one texel per render pixel. The render's rows must equal the expected values
//! computed from the PNG's file rows under the Base flip, and must differ from the Light layer's
//! rows. Both the Blender-authored reference (`reference.blend`) and Potter's export must pass.

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
    FLOAT_TOLERANCE, TestResult, apply_operations, assert_close, assert_floats, build_oracle,
    dump_blend, dumped_object, export, f32_ulp_tolerance, import_blend, losses, node_id, number,
    read_scene, render_texture, run_blender, vec3, write_script,
};
use serde_json::{Value, json};
use std::{fs, path::Path};
use tempfile::tempdir;

/// Render tolerance (linear). The box-filtered Cycles render matches the file rows to about 1e-6;
/// the bound leaves room for other drivers. The Light layer's rows differ by 0.78 or more.
const RENDER_TOLERANCE: f64 = 0.002;

type Grid = Vec<Vec<[f64; 3]>>;

fn rows(value: &Value) -> TestResult<Grid> {
    let mut grid = Vec::new();
    for row in value.as_array().ok_or("rows are not an array")? {
        let mut pixels = Vec::new();
        for pixel in row.as_array().ok_or("row is not an array")? {
            pixels.push(vec3(pixel)?);
        }
        grid.push(pixels);
    }
    Ok(grid)
}

fn assert_grid(actual: &Value, wanted: &Value, tolerance: f64, label: &str) -> TestResult {
    let actual = rows(actual)?;
    let wanted = rows(wanted)?;
    assert_eq!(actual.len(), wanted.len(), "{label}: row count");
    for (y, (actual_row, wanted_row)) in actual.iter().zip(&wanted).enumerate() {
        assert_eq!(actual_row.len(), wanted_row.len(), "{label}: row {y} width");
        for (x, (a, w)) in actual_row.iter().zip(wanted_row).enumerate() {
            assert_floats(a, w, tolerance, &format!("{label} pixel ({x}, {y})"));
        }
    }
    Ok(())
}

/// Stored f32 pixels that Potter's export must match, in f32 ulps of each expected value.
///
/// The chain is: decode the 8-bit byte to linear in f64 (`color::srgb_to_linear`), re-encode in f64
/// (`linear_to_srgb_unclamped`, `exchange/blend.rs`), store as f32 in Blender's float buffer. The f64
/// round trip is exact to about 1e-8, and the f32 store adds at most half an ulp, so the stored value
/// is within one ulp of f32(200/255). Two ulps leaves margin for one more rounding. A larger
/// difference is a product or store defect, not a tolerance to widen.
const STORED_PIXEL_ULPS: f64 = 2.0;

/// Compares each channel of `actual` with `wanted` within `ulps` f32 ulps of the wanted value.
fn assert_stored_grid(actual: &Value, wanted: &Value, ulps: f64, label: &str) -> TestResult {
    let actual = rows(actual)?;
    let wanted = rows(wanted)?;
    assert_eq!(actual.len(), wanted.len(), "{label}: row count");
    for (y, (actual_row, wanted_row)) in actual.iter().zip(&wanted).enumerate() {
        assert_eq!(actual_row.len(), wanted_row.len(), "{label}: row {y} width");
        for (x, (a, w)) in actual_row.iter().zip(wanted_row).enumerate() {
            for (channel, (a, w)) in a.iter().zip(w).enumerate() {
                assert_close(
                    *a,
                    *w,
                    f32_ulp_tolerance(*w, ulps),
                    &format!("{label} pixel ({x}, {y}) channel {channel}"),
                );
            }
        }
    }
    Ok(())
}

/// The largest per-channel difference between two grids of the same size.
fn grid_distance(a: &Grid, b: &Grid) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(row_a, row_b)| row_a.iter().zip(row_b))
        .flat_map(|(pa, pb)| pa.iter().zip(pb).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

/// Checks one Blender render of the texture plane (`render_texture.py` output). `potter_export`
/// marks Potter's export, whose packed PNG reopens as a FILE image with its raw encoded bytes, so
/// `image.pixels` holds the stored sRGB values unconverted. Those are compared directly within f32
/// ulps (`STORED_PIXEL_ULPS`). The Blender reference is compared raw at `FLOAT_TOLERANCE`.
fn check_render(result: &Value, expected: &Value, label: &str, potter_export: bool) -> TestResult {
    let texture = &expected["texture"];
    assert_eq!(
        result["image_size"],
        json!([texture["width"], texture["height"]]),
        "{label}: image size"
    );
    assert_eq!(result["interpolation"], "Closest", "{label}: interpolation");
    assert_eq!(
        result["uv_map"], "Base",
        "{label}: texture must sample the Base layer"
    );
    // The file's PNG, read by Blender (`image.pixels`, bottom row first): no flip anywhere.
    if potter_export {
        assert_stored_grid(
            &result["image_pixels_bottom_up"],
            &texture["blender_pixels_bottom_up"],
            STORED_PIXEL_ULPS,
            &format!("{label} image pixels"),
        )?;
    } else {
        assert_grid(
            &result["image_pixels_bottom_up"],
            &texture["blender_pixels_bottom_up"],
            FLOAT_TOLERANCE,
            &format!("{label} image pixels"),
        )?;
    }
    // The render: one texel per pixel, Base flipped.
    let rendered = rows(&result["render_bottom_up"])?;
    assert_grid(
        &result["render_bottom_up"],
        &texture["render_bottom_up_base_layer"],
        RENDER_TOLERANCE,
        &format!("{label} render"),
    )?;
    let light = rows(&texture["render_bottom_up_light_layer"])?;
    assert!(
        grid_distance(&rendered, &light) > 0.1,
        "{label}: the render must not match the Light layer's orientation"
    );
    Ok(())
}

/// The link that feeds `to_socket` of the node named `to_node`, from a dumped material.
fn link_into<'a>(material: &'a Value, to_node: &str, to_socket: &str) -> TestResult<&'a Value> {
    material["links"]
        .as_array()
        .ok_or("dumped links are missing")?
        .iter()
        .find(|link| {
            link["to_node"].as_str() == Some(to_node)
                && link["to_socket"].as_str() == Some(to_socket)
        })
        .ok_or_else(|| format!("nothing feeds {to_node}.{to_socket}").into())
}

/// The node that feeds `to_socket` of the node named `to_node`, from a dumped material.
fn feeding<'a>(material: &'a Value, to_node: &str, to_socket: &str) -> TestResult<&'a Value> {
    let link = link_into(material, to_node, to_socket)?;
    let source = link["from_node"].as_str().ok_or("link has no source")?;
    material["nodes"]
        .as_array()
        .ok_or("dumped nodes are missing")?
        .iter()
        .find(|node| node["name"].as_str() == Some(source))
        .ok_or_else(|| format!("node {source} is missing").into())
}

/// The Blender-side structure of the field material. The field keeps Potter's default factors (base
/// color 0.6, roughness 0.8), which multiply the texture as Blender's glTF importer builds them
/// (`material_graph.rs` `multiply_color`, `multiply_channel`). The base texture reads `Base` into a
/// `MixRGB` MULTIPLY whose Color2 is the texture, feeding Base Color. The roughness texture reads the
/// render layer `Light` through its Red channel and a Math MULTIPLY, feeding Roughness. Factors are
/// unlinked constants, which the dump does not record; the links prove the chains.
fn check_material_links(object: &Value) -> TestResult {
    let material = object["materials"]
        .as_array()
        .and_then(|materials| materials.first())
        .ok_or("TexturePlane has no material")?;
    let principled = material["nodes"]
        .as_array()
        .ok_or("dumped nodes are missing")?
        .iter()
        .find(|node| node["type"] == "ShaderNodeBsdfPrincipled")
        .ok_or("no Principled node")?["name"]
        .as_str()
        .ok_or("node has no name")?
        .to_owned();

    let base_mix = feeding(material, &principled, "Base Color")?;
    assert_eq!(
        base_mix["type"], "ShaderNodeMixRGB",
        "base color is the texture times its factor"
    );
    let base = feeding(
        material,
        base_mix["name"].as_str().unwrap_or_default(),
        "Color2",
    )?;
    assert_eq!(base["type"], "ShaderNodeTexImage", "base texture");
    assert_eq!(base["interpolation"], "Closest", "base color interpolation");
    let base_uv = feeding(
        material,
        base["name"].as_str().unwrap_or_default(),
        "Vector",
    )?;
    assert_eq!(
        base_uv["type"], "ShaderNodeUVMap",
        "base texture has a UV Map"
    );
    assert_eq!(base_uv["uv_map"], "Base", "base texture layer selection");

    // Potter's scalar fields read the texture's Red channel (`material_graph.rs` `channel_texture`).
    let rough_factor = feeding(material, &principled, "Roughness")?;
    assert_eq!(
        rough_factor["type"], "ShaderNodeMath",
        "roughness is the channel times its factor"
    );
    let rough_split = feeding(
        material,
        rough_factor["name"].as_str().unwrap_or_default(),
        "Value",
    )?;
    assert_eq!(
        rough_split["type"], "ShaderNodeSeparateColor",
        "roughness is a separated texture channel"
    );
    let rough_link = link_into(
        material,
        rough_factor["name"].as_str().unwrap_or_default(),
        "Value",
    )?;
    assert_eq!(
        rough_link["from_socket"], "Red",
        "roughness reads the texture's Red channel"
    );
    let rough = feeding(
        material,
        rough_split["name"].as_str().unwrap_or_default(),
        "Color",
    )?;
    assert_eq!(
        rough["type"], "ShaderNodeTexImage",
        "roughness is a texture"
    );
    let rough_uv = feeding(
        material,
        rough["name"].as_str().unwrap_or_default(),
        "Vector",
    )?;
    assert_eq!(
        rough_uv["uv_map"], "Light",
        "uv_layer null resolves to the render layer"
    );
    Ok(())
}

#[test]
fn textured_plane_renders_the_base_layer_with_its_file_orientation() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, expected) = build_oracle(&blender, root.path())?;
    let oracle_dir = root.path().join("oracle");

    // The reference: Blender's own nodes, rendered by the same script.
    let reference = render_texture(
        &blender,
        root.path(),
        &oracle_dir.join("reference.blend"),
        "reference",
    )?;
    check_render(&reference, &expected, "reference", false)?;

    // Potter: import the oracle, author the field material, export, and render the export.
    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let plane = node_id(&read_scene(&project)?, "TexturePlane")?;
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "image.load", "id": "texture", "path": oracle_dir.join("texture.png").to_string_lossy()},
            {"op": "image.create", "id": "rough", "width": 4, "height": 4, "colorspace": "non_color", "fill_color": [0.5, 0.5, 0.5, 1.0]},
            {"op": "material.create", "id": "field", "base_color_texture": {"image": "texture", "uv_layer": "Base", "interpolation": "closest"}, "roughness_texture": {"image": "rough", "uv_layer": null, "interpolation": "linear"}},
            {"op": "node.update", "target": {"id": plane}, "set": {"materials": ["field"]}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    let exported_blend = root.path().join("field.blend");
    let exported = export(&blender, &project, "blend", &exported_blend)?;
    assert!(
        losses(&exported).is_empty(),
        "texture references must not be lost on .blend export: {exported}"
    );
    let dump = dump_blend(&blender, root.path(), &exported_blend, "field")?;
    check_material_links(dumped_object(&dump, "TexturePlane")?)?;

    let potter = render_texture(&blender, root.path(), &exported_blend, "potter")?;
    check_render(&potter, &expected, "potter", true)?;
    Ok(())
}

const MATERIAL_ALPHA_SCRIPT: &str = include_str!("fixtures/native_oracle/material_alpha.py");
const AUTHOR_ALPHA_SCRIPT: &str = include_str!("fixtures/native_oracle/author_alpha_materials.py");

/// Blender's record of every material in `blend` (`material_alpha.py` output).
fn blender_materials(blender: &Path, root: &Path, blend: &Path, name: &str) -> TestResult<Value> {
    let script = write_script(root, "material_alpha.py", MATERIAL_ALPHA_SCRIPT)?;
    let out = root.join(format!("{name}.json"));
    run_blender(blender, &script, Some(blend), &out)?;
    Ok(serde_json::from_slice(&fs::read(out)?)?)
}

/// The Blender record of the material named `name` in a `blender_materials` dump.
fn blender_material<'a>(dump: &'a Value, name: &str) -> TestResult<&'a Value> {
    dump["materials"]
        .as_array()
        .ok_or("material dump is not an array")?
        .iter()
        .find(|material| material["name"].as_str() == Some(name))
        .ok_or_else(|| format!("Blender has no material named {name}").into())
}

/// The Potter material named `name` in a scene document.
fn potter_material<'a>(scene: &'a Value, name: &str) -> TestResult<&'a Value> {
    scene["materials"]
        .as_object()
        .ok_or("scene materials are not an object")?
        .values()
        .find(|material| material["name"].as_str() == Some(name))
        .ok_or_else(|| format!("no material named {name}").into())
}

#[test]
fn alpha_mode_and_threshold_survive_authoring_through_blend() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, _) = build_oracle(&blender, root.path())?;
    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let plane = node_id(&read_scene(&project)?, "TexturePlane")?;
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "material.create", "id": "blend", "name": "Blend Glass",
             "base_color": [0.2, 0.4, 0.6, 0.6], "alpha_mode": "blend", "alpha_threshold": 0.23},
            {"op": "material.create", "id": "clip", "name": "Clip Leaf",
             "base_color": [0.8, 0.1, 0.1, 0.8], "alpha_mode": "clip", "alpha_threshold": 0.42},
            {"op": "material.create", "id": "opaque", "name": "Opaque Paint",
             "base_color": [0.5, 0.5, 0.5, 0.3], "alpha_mode": "opaque", "alpha_threshold": 0.7},
            {"op": "node.update", "target": {"id": plane},
             "set": {"materials": ["blend", "clip", "opaque"]}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    let exported_blend = root.path().join("alpha.blend");
    let exported = export(&blender, &project, "blend", &exported_blend)?;
    assert!(
        losses(&exported).is_empty(),
        "material alpha must not be lost on .blend export: {exported}"
    );

    // Blender's own state: render method, threshold, Principled Alpha, and the marker.
    let dump = blender_materials(&blender, root.path(), &exported_blend, "alpha")?;
    let glass = blender_material(&dump, "Blend Glass")?;
    assert_eq!(glass["surface_render_method"], "BLENDED");
    assert_close(
        number(&glass["alpha_threshold"])?,
        0.23,
        FLOAT_TOLERANCE,
        "blend threshold",
    );
    assert_close(
        number(&glass["principled_alpha"])?,
        0.6,
        FLOAT_TOLERANCE,
        "blend alpha",
    );
    assert_close(
        number(&glass["principled_base_alpha"])?,
        0.6,
        FLOAT_TOLERANCE,
        "blend base",
    );
    assert_eq!(glass["alpha_mode_property"], "blend");

    let clip = blender_material(&dump, "Clip Leaf")?;
    assert_eq!(clip["surface_render_method"], "DITHERED");
    assert_close(
        number(&clip["alpha_threshold"])?,
        0.42,
        FLOAT_TOLERANCE,
        "clip threshold",
    );
    assert_close(
        number(&clip["principled_alpha"])?,
        0.8,
        FLOAT_TOLERANCE,
        "clip alpha",
    );
    assert_eq!(clip["alpha_mode_property"], "clip");

    let opaque = blender_material(&dump, "Opaque Paint")?;
    assert_eq!(opaque["surface_render_method"], "DITHERED");
    assert_close(
        number(&opaque["alpha_threshold"])?,
        0.7,
        FLOAT_TOLERANCE,
        "opaque threshold",
    );
    assert_close(
        number(&opaque["principled_alpha"])?,
        1.0,
        FLOAT_TOLERANCE,
        "opaque alpha",
    );
    assert_close(
        number(&opaque["principled_base_alpha"])?,
        0.3,
        FLOAT_TOLERANCE,
        "opaque base",
    );
    assert_eq!(opaque["alpha_mode_property"], "opaque");

    // Potter's view after reopening the exported file: modes, thresholds, and base alpha.
    let (reimported, _) = import_blend(&blender, root.path(), "reimport", &exported_blend)?;
    let scene = read_scene(&reimported)?;
    let glass = potter_material(&scene, "Blend Glass")?;
    assert_eq!(glass["alpha_mode"], "blend");
    assert_close(
        number(&glass["alpha_threshold"])?,
        0.23,
        FLOAT_TOLERANCE,
        "reimport blend",
    );
    assert_close(
        number(&glass["base_color"][3])?,
        0.6,
        FLOAT_TOLERANCE,
        "reimport blend base",
    );
    let clip = potter_material(&scene, "Clip Leaf")?;
    assert_eq!(clip["alpha_mode"], "clip");
    assert_close(
        number(&clip["alpha_threshold"])?,
        0.42,
        FLOAT_TOLERANCE,
        "reimport clip",
    );
    let opaque = potter_material(&scene, "Opaque Paint")?;
    assert_eq!(opaque["alpha_mode"], "opaque");
    assert_close(
        number(&opaque["alpha_threshold"])?,
        0.7,
        FLOAT_TOLERANCE,
        "reimport opaque",
    );
    assert_close(
        number(&opaque["base_color"][3])?,
        0.3,
        FLOAT_TOLERANCE,
        "reimport opaque base",
    );
    Ok(())
}

#[test]
fn blender_authored_materials_infer_alpha_mode_from_render_state() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(
        root.path(),
        "author_alpha_materials.py",
        AUTHOR_ALPHA_SCRIPT,
    )?;
    let authored = root.path().join("authored.blend");
    run_blender(&blender, &script, None, &authored)?;
    let (project, _) = import_blend(&blender, root.path(), "authored_project", &authored)?;
    let scene = read_scene(&project)?;
    assert_eq!(
        potter_material(&scene, "Authored Blended")?["alpha_mode"],
        "blend"
    );
    let dithered = potter_material(&scene, "Authored Dithered Alpha")?;
    // Blender 5.2.2 never thresholds a DITHERED material's alpha. A black plane with Principled Alpha 0.4 shows a white
    // world at 0.6 in Cycles and EEVEE, under DITHERED and BLENDED alike, so the graph carries the alpha and Potter
    // leaves it to the graph as `opaque` (`blend_bridge.py` `_alpha_mode_of`).
    assert_eq!(dithered["alpha_mode"], "opaque");
    // The threshold is Blender's own default (0.5), not derived from the inferred mode.
    assert_close(
        number(&dithered["alpha_threshold"])?,
        0.5,
        FLOAT_TOLERANCE,
        "authored threshold",
    );
    assert_eq!(
        potter_material(&scene, "Authored Dithered Opaque")?["alpha_mode"],
        "opaque"
    );
    Ok(())
}
