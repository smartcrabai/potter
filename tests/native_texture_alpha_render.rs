//! Blender-gated oracle for image alpha multiplied by a field material's base alpha.
//!
//! The test renders the imported `TexturePlane` through Potter's CLI with transparent film, reads the
//! EXR back through Potter's image loader, and checks that .blend export/import retains the field
//! texture and alpha state. Blender's reopened node graph is checked for the same numeric product.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "the shared native oracle module exposes helpers not used by this test"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{error::Error, fs, path::Path};

use native_oracle::{
    FLOAT_TOLERANCE, TestResult, apply_operations, assert_close, build_oracle, export,
    import_blend, losses, node_id, number, pot, read_scene, run_blender, write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const IMAGE_ALPHA: f64 = 0.5;
const BASE_ALPHA: f64 = 0.6;
const RENDER_SAMPLES: u32 = 4096;
// For Bernoulli alpha coverage, the standard error at the expected alpha is sqrt(.3 * .7 / 4096)
// ~= .0072; 0.04 is over five standard errors, yet far below the errors from either missing factor.
const ALPHA_TOLERANCE: f64 = 0.04;

const BLENDER_ALPHA_GRAPH_SCRIPT: &str = r#"
import bpy
import json
import sys

output_path = sys.argv[sys.argv.index("--") + 1]
material = next(item for item in bpy.data.materials if item.name == "Texture Alpha Field")
shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
texture = next(node for node in material.node_tree.nodes if node.type == "TEX_IMAGE")
alpha_factor = next(
    node for node in material.node_tree.nodes
    if node.type == "MATH" and node.operation == "MULTIPLY"
)

def linked(from_node, from_socket, to_node, to_socket):
    return any(
        link.from_node == from_node
        and link.from_socket.name == from_socket
        and link.to_node == to_node
        and link.to_socket.name == to_socket
        for link in material.node_tree.links
    )

image = texture.image
image_alpha = float(image.pixels[3])
multiply_factor = float(alpha_factor.inputs[1].default_value)
with open(output_path, "w", encoding="utf-8") as output:
    json.dump({
        "surface_render_method": material.surface_render_method,
        "alpha_threshold": float(material.alpha_threshold),
        "alpha_mode": material.get("potter.alpha_mode"),
        "texture_image": image.name,
        "image_alpha": image_alpha,
        "texture_alpha_link": linked(texture, "Alpha", alpha_factor, "Value"),
        "multiply_operation": alpha_factor.operation,
        "multiply_factor": multiply_factor,
        "alpha_product": image_alpha * multiply_factor,
        "principled_alpha_default": float(shader.inputs["Alpha"].default_value),
        "principled_alpha_link": linked(alpha_factor, "Value", shader, "Alpha"),
    }, output)
"#;

fn center_pixel(project: &Path, image_id: &str) -> Result<[f64; 4], Box<dyn Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][image_id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    Ok(sample(&data, [0.5, 0.5], 0))
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one end-to-end oracle covers Potter rendering and the .blend field round trip"
)]
fn field_image_alpha_times_base_alpha_renders_and_round_trips_through_blend() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        eprintln!("skipping texture alpha render oracle: Blender is unavailable");
        return Ok(());
    };

    let root = tempdir()?;
    let (oracle_blend, _) = build_oracle(&blender, root.path())?;
    let (project, import_report) = import_blend(&blender, root.path(), "project", &oracle_blend)?;
    assert!(
        losses(&import_report).is_empty(),
        "oracle import lost data: {import_report}"
    );
    let imported_scene = read_scene(&project)?;
    let plane = node_id(&imported_scene, "TexturePlane")?;
    let active_scene = imported_scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();

    let authored = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "image.create", "id": "alpha_texture", "name": "Alpha Texture",
             "width": 1, "height": 1, "colorspace": "linear",
             "fill_color": [1.0, 1.0, 1.0, IMAGE_ALPHA]},
            {"op": "material.create", "id": "alpha_material", "name": "Texture Alpha Field",
             "base_color": [1.0, 1.0, 1.0, BASE_ALPHA], "alpha_mode": "blend",
             "alpha_threshold": 0.23,
             "base_color_texture": {"image": "alpha_texture", "uv_layer": "Base", "interpolation": "closest"}},
            {"op": "node.update", "target": {"id": plane}, "set": {"materials": ["alpha_material"]}},
            {"op": "camera.create", "id": "alpha_camera",
             "transform": {"translation": [8.0, 0.0, 6.0]},
             "projection": "orthographic", "ortho_scale": 2.0},
            {"op": "scene.update", "target": {"id": active_scene}, "set": {"camera": "alpha_camera"}},
            {"op": "render.update", "target": {"id": active_scene},
             "set": {"resolution_x": 4, "resolution_y": 4, "resolution_percentage": 100,
                     "samples": RENDER_SAMPLES, "seed": 0, "max_bounces": 1,
                     "film_transparent": true, "engine": "path"}}
        ]),
    )?;
    assert_eq!(
        authored["ok"], true,
        "field material setup failed: {authored}"
    );

    // The sample point is inside TexturePlane; all color channels are immaterial to this alpha proof.
    let exr_dir = root.path().join("rendered");
    let mut render = pot();
    render
        .arg("render")
        .arg(&project)
        .args(["--format", "exr", "--out"])
        .arg(&exr_dir)
        .arg("--json");
    let output = native_oracle::checked(&mut render, "Potter transparent alpha render")?;
    let render_report: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        render_report["ok"], true,
        "render did not report success: {render_report}"
    );
    let exr = exr_dir.join("frame_0001.exr");
    assert!(fs::metadata(&exr)?.len() > 0, "Potter wrote an empty EXR");
    let loaded = apply_operations(
        &project,
        root.path(),
        json!([{"op": "image.load", "id": "alpha_render", "path": exr.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "could not load Potter's EXR: {loaded}");
    let actual = center_pixel(&project, "alpha_render")?[3];
    let expected = IMAGE_ALPHA * BASE_ALPHA;
    assert!(
        (actual - expected).abs() <= ALPHA_TOLERANCE,
        "rendered EXR alpha {actual} differs from image alpha {IMAGE_ALPHA} * base alpha {BASE_ALPHA} = {expected}"
    );

    let scene = read_scene(&project)?;
    let field = &scene["materials"]["alpha_material"];
    assert_eq!(field["alpha_mode"], "blend");
    assert_eq!(field["base_color_texture"]["image"], "alpha_texture");
    assert_eq!(field["base_color_texture"]["uv_layer"], "Base");
    assert_eq!(field["base_color_texture"]["interpolation"], "closest");
    assert_close(
        number(&field["base_color"][3])?,
        BASE_ALPHA,
        FLOAT_TOLERANCE,
        "authored base alpha",
    );
    assert_close(
        number(&field["alpha_threshold"])?,
        0.23,
        FLOAT_TOLERANCE,
        "authored alpha threshold",
    );

    let exported_blend = root.path().join("texture-alpha.blend");
    let export_report = export(&blender, &project, "blend", &exported_blend)?;
    assert!(
        losses(&export_report).is_empty(),
        "field texture/alpha export lost data: {export_report}"
    );

    let inspect_script = write_script(
        root.path(),
        "inspect_alpha_graph.py",
        BLENDER_ALPHA_GRAPH_SCRIPT,
    )?;
    let blender_dump_path = root.path().join("blender_alpha_graph.json");
    run_blender(
        &blender,
        &inspect_script,
        Some(&exported_blend),
        &blender_dump_path,
    )?;
    let blender_graph: Value = serde_json::from_slice(&fs::read(&blender_dump_path)?)?;
    assert_eq!(blender_graph["surface_render_method"], "BLENDED");
    assert_eq!(blender_graph["alpha_mode"], "blend");
    assert_close(
        number(&blender_graph["alpha_threshold"])?,
        0.23,
        FLOAT_TOLERANCE,
        "Blender alpha threshold",
    );
    assert_eq!(blender_graph["texture_image"], "Alpha Texture");
    assert_close(
        number(&blender_graph["image_alpha"])?,
        IMAGE_ALPHA,
        FLOAT_TOLERANCE,
        "Blender texture alpha",
    );
    assert_eq!(blender_graph["texture_alpha_link"], true);
    assert_eq!(blender_graph["multiply_operation"], "MULTIPLY");
    assert_close(
        number(&blender_graph["multiply_factor"])?,
        BASE_ALPHA,
        FLOAT_TOLERANCE,
        "Blender base alpha factor",
    );
    assert_close(
        number(&blender_graph["alpha_product"])?,
        expected,
        FLOAT_TOLERANCE,
        "Blender image alpha times base alpha",
    );
    assert_close(
        number(&blender_graph["principled_alpha_default"])?,
        BASE_ALPHA,
        FLOAT_TOLERANCE,
        "Blender Principled Alpha default",
    );
    assert_eq!(blender_graph["principled_alpha_link"], true);

    let (reimported, reimport_report) =
        import_blend(&blender, root.path(), "reimported", &exported_blend)?;
    assert!(
        losses(&reimport_report).is_empty(),
        "field texture/alpha reimport lost data: {reimport_report}"
    );
    let reimported_scene = read_scene(&reimported)?;
    let field = &reimported_scene["materials"]["alpha_material"];
    assert_eq!(field["alpha_mode"], "blend");
    assert_close(
        number(&field["alpha_threshold"])?,
        0.23,
        FLOAT_TOLERANCE,
        "reimported alpha threshold",
    );
    assert_close(
        number(&field["base_color"][3])?,
        BASE_ALPHA,
        FLOAT_TOLERANCE,
        "reimported base alpha",
    );
    // Reimport keeps the texture as a linked graph node, not as a `base_color_texture` field, so the
    // alpha texture is found through the image texture node that reads it.
    let graph = field_graph(&reimported_scene, field)?;
    let (texture_node, texture) = image_texture(graph, "alpha_texture")?;
    assert!(
        reimported_scene["images"].get("alpha_texture").is_some(),
        "reimported texture image is absent"
    );
    assert_close(
        center_pixel(&reimported, "alpha_texture")?[3],
        IMAGE_ALPHA,
        FLOAT_TOLERANCE,
        "reimported image alpha",
    );
    assert_eq!(texture["properties"]["interpolation"], "Closest");
    let uv_link = graph["links"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|link| link["to_node"] == texture_node && link["to_socket"] == "Vector")
        .ok_or("reimported texture Vector input is not linked")?;
    let uv_node = &graph["nodes"][uv_link["from_node"].as_str().unwrap_or_default()];
    assert_eq!(uv_node["type"], "ShaderNodeUVMap");
    assert_eq!(uv_node["properties"]["uv_map"], "Base");
    let (principled_node, _) = node_of_type(graph, "ShaderNodeBsdfPrincipled")?;
    assert!(
        feeds(graph, texture_node, "Color", principled_node, "Base Color"),
        "reimported texture colour does not reach Base Color"
    );
    assert!(
        feeds(graph, texture_node, "Alpha", principled_node, "Alpha"),
        "reimported texture alpha does not reach Principled Alpha"
    );
    Ok(())
}

/// The shader graph of the reopened material `field`, from the scene's `node_groups`.
fn field_graph<'a>(scene: &'a Value, field: &Value) -> TestResult<&'a Value> {
    let graph_id = field["node_tree"]
        .as_str()
        .ok_or("reimported field material has no shader graph")?;
    Ok(&scene["node_groups"][graph_id])
}

/// The first node of `graph` with type `node_type`, and its ID.
fn node_of_type<'a>(graph: &'a Value, node_type: &str) -> TestResult<(&'a str, &'a Value)> {
    graph["nodes"]
        .as_object()
        .and_then(|nodes| nodes.iter().find(|(_, node)| node["type"] == node_type))
        .map(|(id, node)| (id.as_str(), node))
        .ok_or_else(|| format!("no {node_type} node").into())
}

/// The image texture node of `graph` that reads the image `image`, and its ID.
fn image_texture<'a>(graph: &'a Value, image: &str) -> TestResult<(&'a str, &'a Value)> {
    graph["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes.iter().find(|(_, node)| {
                node["type"] == "ShaderNodeTexImage" && node["properties"]["image"] == image
            })
        })
        .map(|(id, node)| (id.as_str(), node))
        .ok_or_else(|| format!("no image texture node reads {image}").into())
}

/// Whether `source`'s `source_socket` feeds `target`'s `target_socket`, following links upstream
/// through any intermediate nodes.
fn feeds(
    graph: &Value,
    source: &str,
    source_socket: &str,
    target: &str,
    target_socket: &str,
) -> bool {
    let links = graph["links"].as_array().map_or(&[][..], Vec::as_slice);
    let mut pending = vec![(target.to_owned(), Some(target_socket.to_owned()))];
    let mut visited: Vec<String> = Vec::new();
    while let Some((node, socket)) = pending.pop() {
        if socket.is_none() {
            if visited.contains(&node) {
                continue;
            }
            visited.push(node.clone());
        }
        for link in links.iter().filter(|link| {
            link["to_node"] == node.as_str()
                && socket
                    .as_deref()
                    .is_none_or(|name| link["to_socket"] == name)
        }) {
            let from = link["from_node"].as_str().unwrap_or_default();
            let from_socket = link["from_socket"].as_str().unwrap_or_default();
            if from == source && from_socket == source_socket {
                return true;
            }
            pending.push((from.to_owned(), None));
        }
    }
    false
}
