//! Cycles hair-curve shadow transparency against the committed Blender 5.2.2 oracle.
//!
//! The source curve is camera-hidden and casts a deterministic zero-angle SUN shadow onto a diffuse
//! receiver. Cycles bakes `ATTR_STD_SHADOW_TRANSPARENCY` from its tinted Transparent/Diffuse material
//! mix; the test checks that graph and curve inputs survive Potter import and compares the shadowed
//! and unshadowed receiver patches to Blender's linear EXR.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{collections::BTreeSet, path::Path};

use blender_checked::blender_executable;
use native_oracle::{
    FLOAT_TOLERANCE, TestResult, apply_operations, checked, import_blend, losses, node_id, pot,
    read_scene,
};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde::Deserialize;
use serde_json::json;
use tempfile::tempdir;

const FIXTURES: &str = "tests/fixtures/native_oracle/hair_shadow_transparency";
const RESOLUTION: [u32; 2] = [64, 64];
const SAMPLES: u32 = 64;
const SEED: u32 = 29;
const CAPTURE: &str =
    include_str!("fixtures/native_oracle/hair_shadow_transparency/hair_shadow_transparency.json");

#[derive(Deserialize)]
struct Capture {
    blender_version: String,
    blender_build_hash: String,
    engine: String,
    resolution: [u32; 2],
    samples: u32,
    seed: u32,
    pixel_filter: String,
    filter_width: f64,
    view_transform: String,
    linear_exr: String,
    cases: Vec<CaptureCase>,
    inputs: CaptureInputs,
}

#[derive(Deserialize)]
struct CaptureCase {
    name: String,
    probe: [u32; 3],
    linear_rgba: [f64; 4],
}

#[derive(Deserialize)]
struct CaptureInputs {
    attribute_source: String,
    attribute_name: String,
    expected_baked_shadow_transparency: f64,
    curve: CurveInputs,
    hair_material: HairMaterialInputs,
    receiver: ReceiverInputs,
    sun: SunInputs,
    probes: std::collections::BTreeMap<String, ProbeInput>,
}

#[derive(Deserialize)]
struct CurveInputs {
    name: String,
    spline_type: String,
    key_positions: [[f64; 3]; 4],
    key_radii: Vec<f64>,
    visible_camera: bool,
    visible_diffuse: bool,
    visible_glossy: bool,
    visible_transmission: bool,
    visible_volume_scatter: bool,
    visible_shadow: bool,
}

#[derive(Deserialize)]
struct HairMaterialInputs {
    nodes: Vec<String>,
    transparent_color: [f64; 4],
    mix_factor: f64,
    transparent_weight: f64,
    opaque_closure: String,
    dynamic_hair_model: Option<String>,
}

#[derive(Deserialize)]
struct ReceiverInputs {
    diffuse_color: [f64; 4],
}

#[derive(Deserialize)]
struct SunInputs {
    r#type: String,
    energy: f64,
    angle: f64,
    rotation_euler: [f64; 3],
}

#[derive(Deserialize)]
struct ProbeInput {
    pixel: [u32; 2],
    radius: u32,
}

#[test]
fn catmull_rom_hair_shadow_transparency_matches_blender_capture() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping hair-curve shadow parity; no executable Blender was found");
        return Ok(());
    };
    let capture: Capture = serde_json::from_str(CAPTURE)?;
    check_capture_inputs(&capture)?;

    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES);
    let (project, imported) = import_blend(
        &blender,
        root.path(),
        "hair_shadow_transparency",
        &fixture_dir.join("hair_shadow_transparency.blend"),
    )?;
    assert!(
        losses(&imported).is_empty(),
        "hair shadow source import reported losses: {imported}"
    );
    check_imported_curve(&read_scene(&project)?, &capture.inputs)?;
    configure_render(&project, root.path())?;

    let source = load_image_file(
        &project,
        root.path(),
        "blender_hair_shadow",
        &fixture_dir.join(&capture.linear_exr),
    )?;
    assert_eq!([source.width, source.height], RESOLUTION);
    let rendered = render_frame(&project, root.path(), "pot_hair_shadow")?;
    assert_eq!([rendered.width, rendered.height], RESOLUTION);

    let mut by_name = std::collections::BTreeMap::new();
    for case in &capture.cases {
        by_name.insert(case.name.as_str(), case);
        let source_patch = patch_mean(&source, case.probe);
        let actual_patch = patch_mean(&rendered, case.probe);
        for channel in 0..4 {
            assert_close(
                source_patch[channel],
                case.linear_rgba[channel],
                FLOAT_TOLERANCE,
                &format!("Blender EXR case {} channel {channel}", case.name),
            );
            assert_close(
                actual_patch[channel],
                case.linear_rgba[channel],
                FLOAT_TOLERANCE,
                &format!("Potter case {} channel {channel}", case.name),
            );
        }
    }
    let shadow = by_name["shadow_patch"].linear_rgba;
    let lit = by_name["lit_patch"].linear_rgba;
    for channel in 0..3 {
        assert!(
            shadow[channel] > 0.0 && shadow[channel] < lit[channel],
            "the mixed transparent/opaque hair closure must partially attenuate lit receiver channel {channel}: shadow={shadow:?}, lit={lit:?}"
        );
    }
    assert_eq!(shadow[3], 1.0);
    assert_eq!(lit[3], 1.0);
    Ok(())
}

fn check_capture_inputs(capture: &Capture) -> TestResult {
    assert_eq!(capture.blender_version, "5.2.2 LTS");
    assert_eq!(capture.blender_build_hash, "d13f752e3b9c");
    assert_eq!(capture.engine, "CYCLES");
    assert_eq!(capture.resolution, RESOLUTION);
    assert_eq!(capture.samples, SAMPLES);
    assert_eq!(capture.seed, SEED);
    assert_eq!(capture.pixel_filter, "BOX");
    assert_eq!(capture.filter_width, 1.0);
    assert_eq!(capture.view_transform, "Standard");
    assert_eq!(capture.linear_exr, "hair_shadow_transparency.exr");

    let cases = capture
        .cases
        .iter()
        .map(|case| case.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(cases, vec!["shadow_patch", "lit_patch"]);
    assert_eq!(capture.cases[0].probe, [32, 32, 1]);
    assert_eq!(capture.cases[1].probe, [32, 20, 1]);

    let inputs = &capture.inputs;
    assert_eq!(inputs.attribute_source, "ATTR_STD_SHADOW_TRANSPARENCY");
    assert_eq!(inputs.attribute_name, "shadow_transparency");
    native_oracle::assert_close(
        inputs.expected_baked_shadow_transparency,
        (1.0 - inputs.hair_material.mix_factor)
            * inputs.hair_material.transparent_color[..3]
                .iter()
                .sum::<f64>()
            / 3.0,
        FLOAT_TOLERANCE,
        "Cycles baked shadow-transparency scalar",
    );
    assert_eq!(inputs.curve.name, "Camera Hidden Hair");
    assert_eq!(inputs.curve.spline_type, "CATMULL_ROM");
    assert_eq!(
        inputs.curve.key_positions,
        [
            [-3.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    );
    assert_eq!(inputs.curve.key_radii.as_slice(), &[0.35; 4]);
    assert!(!inputs.curve.visible_camera);
    assert!(inputs.curve.visible_shadow);
    assert!(!inputs.curve.visible_diffuse);
    assert!(!inputs.curve.visible_glossy);
    assert!(!inputs.curve.visible_transmission);
    assert!(!inputs.curve.visible_volume_scatter);

    assert_eq!(
        inputs.hair_material.nodes,
        vec![
            "ShaderNodeBsdfTransparent".to_owned(),
            "ShaderNodeBsdfDiffuse".to_owned(),
            "ShaderNodeMixShader".to_owned(),
        ]
    );
    assert_eq!(inputs.hair_material.transparent_color, [0.2, 0.7, 0.4, 1.0]);
    assert_eq!(inputs.hair_material.mix_factor, 0.35);
    assert_eq!(inputs.hair_material.transparent_weight, 0.65);
    assert_eq!(
        inputs.hair_material.transparent_weight,
        1.0 - inputs.hair_material.mix_factor
    );
    assert_eq!(inputs.hair_material.opaque_closure, "ShaderNodeBsdfDiffuse");
    assert_eq!(inputs.hair_material.dynamic_hair_model, None);
    assert_eq!(inputs.receiver.diffuse_color, [0.8, 0.55, 0.35, 1.0]);
    assert_eq!(inputs.sun.r#type, "SUN");
    assert_eq!(inputs.sun.energy, 2.0);
    assert_eq!(inputs.sun.angle, 0.0);
    assert_eq!(inputs.sun.rotation_euler, [0.0; 3]);

    let expected_probes = [("shadow_patch", [32, 32], 1), ("lit_patch", [32, 20], 1)];
    assert_eq!(inputs.probes.len(), expected_probes.len());
    for (name, pixel, radius) in expected_probes {
        let probe = inputs
            .probes
            .get(name)
            .ok_or_else(|| format!("missing {name} source probe"))?;
        assert_eq!(probe.pixel, pixel);
        assert_eq!(probe.radius, radius);
    }
    Ok(())
}

fn check_imported_curve(scene: &serde_json::Value, inputs: &CaptureInputs) -> TestResult {
    let curve_node_id = node_id(scene, &inputs.curve.name)?;
    let node = &scene["nodes"][&curve_node_id];
    assert_eq!(node["kind"], "hair_curves");
    assert_eq!(node["ray_visibility"]["camera"], false);
    assert_eq!(node["ray_visibility"]["shadow"], true);
    assert_eq!(node["ray_visibility"]["diffuse"], false);
    assert_eq!(node["ray_visibility"]["glossy"], false);
    assert_eq!(node["ray_visibility"]["transmission"], false);
    assert_eq!(node["ray_visibility"]["volume_scatter"], false);
    let data_id = node["data"]
        .as_str()
        .ok_or("hair-curve data id is missing")?;
    let data = &scene["data_blocks"][data_id];
    let curves = data["hair_curves"]["curves"]
        .as_array()
        .ok_or("imported hair curves are missing")?;
    assert_eq!(curves.len(), 1);
    let points: Vec<[f64; 3]> = serde_json::from_value(curves[0]["points"].clone())?;
    assert_eq!(points, inputs.curve.key_positions);
    native_oracle::assert_close(
        native_oracle::number(&curves[0]["radius"])?,
        inputs.curve.key_radii[0],
        FLOAT_TOLERANCE,
        "imported curve radius",
    );

    let attributes = data["hair_curves"]["attributes"]
        .as_object()
        .ok_or("imported hair-curve attributes are missing")?;
    assert!(
        !attributes.contains_key(&inputs.attribute_name),
        "ATTR_STD_SHADOW_TRANSPARENCY is baked internally by Cycles, not stored as a Curves user attribute"
    );
    assert!(
        !attributes.contains_key("curve_type"),
        "the default CATMULL_ROM spline type is implicit, not a named Curves attribute"
    );

    let materials = data["materials"]
        .as_array()
        .ok_or("hair-curve material slots are missing")?;
    assert_eq!(materials.len(), 1);
    let material_id = materials[0]
        .as_str()
        .ok_or("hair-curve material id is missing")?;
    let material = &scene["materials"][material_id];
    let graph_id = material["node_tree"]
        .as_str()
        .ok_or("hair material graph id is missing")?;
    let graph_nodes = scene["node_groups"][graph_id]["nodes"]
        .as_object()
        .ok_or("hair material graph nodes are missing")?;
    let node_types = graph_nodes
        .values()
        .map(|graph_node| -> TestResult<String> {
            Ok(graph_node["type"]
                .as_str()
                .ok_or_else(|| std::io::Error::other("hair material graph node type is missing"))?
                .to_owned())
        })
        .collect::<TestResult<BTreeSet<_>>>()?;
    let transparent_node = graph_nodes
        .values()
        .find(|graph_node| graph_node["type"] == "ShaderNodeBsdfTransparent")
        .ok_or("transparent hair material node is missing")?;
    let transparent_color = native_oracle::floats(&transparent_node["inputs"]["Color"])?;
    native_oracle::assert_floats(
        &transparent_color,
        &inputs.hair_material.transparent_color,
        FLOAT_TOLERANCE,
        "imported transparent BSDF tint",
    );
    let mix_node = graph_nodes
        .values()
        .find(|graph_node| graph_node["type"] == "ShaderNodeMixShader")
        .ok_or("hair material mix node is missing")?;
    native_oracle::assert_close(
        native_oracle::number(&mix_node["inputs"]["Fac"])?,
        inputs.hair_material.mix_factor,
        FLOAT_TOLERANCE,
        "imported hair material mix factor",
    );
    assert_eq!(
        node_types,
        BTreeSet::from([
            "OutputMaterial".to_owned(),
            "ShaderNodeBsdfDiffuse".to_owned(),
            "ShaderNodeBsdfTransparent".to_owned(),
            "ShaderNodeMixShader".to_owned(),
        ])
    );
    let has_hair_bsdf = node_types
        .iter()
        .any(|node_type| node_type.contains("Hair"));
    assert!(
        !has_hair_bsdf,
        "the fixture must not depend on a dynamic hair BSDF"
    );
    Ok(())
}

fn configure_render(project: &Path, root: &Path) -> TestResult {
    let scene = read_scene(project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        project,
        root,
        json!([{
            "op": "render.update",
            "target": {"id": active},
            "set": {
                "resolution_x": RESOLUTION[0],
                "resolution_y": RESOLUTION[1],
                "samples": SAMPLES,
                "seed": SEED,
                "max_bounces": 1,
                "transparent_max_bounces": 8
            }
        }]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    Ok(())
}

fn load_image_file(project: &Path, root: &Path, id: &str, path: &Path) -> TestResult<ImageData> {
    let loaded = apply_operations(
        project,
        root,
        json!([{"op": "image.load", "id": id, "path": path.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let image: Image = serde_json::from_value(read_scene(project)?["images"][id].clone())?;
    Ok(load_image_data(
        &image,
        project,
        ImageInterpolation::Closest,
    )?)
}

fn render_frame(project: &Path, root: &Path, name: &str) -> TestResult<ImageData> {
    let frames = root.join(name);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--engine", "path", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    checked(&mut render, name)?;
    let loaded = apply_operations(
        project,
        root,
        json!([{
            "op": "image.load",
            "id": name,
            "path": frames.join("frame_0001.exr").to_string_lossy()
        }]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let image: Image = serde_json::from_value(read_scene(project)?["images"][name].clone())?;
    Ok(load_image_data(
        &image,
        project,
        ImageInterpolation::Closest,
    )?)
}

fn patch_mean(frame: &ImageData, probe: [u32; 3]) -> [f64; 4] {
    let [x, y, radius] = probe;
    assert!(x >= radius && y >= radius && x + radius < frame.width && y + radius < frame.height);
    let mut total = [0.0; 4];
    let mut count = 0.0;
    for py in y - radius..=y + radius {
        for px in x - radius..=x + radius {
            let pixel = sample(
                frame,
                [
                    (f64::from(px) + 0.5) / f64::from(frame.width),
                    (f64::from(py) + 0.5) / f64::from(frame.height),
                ],
                false,
            );
            for (sum, value) in total.iter_mut().zip(pixel) {
                *sum += value;
            }
            count += 1.0;
        }
    }
    total.map(|sum| sum / count)
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual {actual} differs from Blender {expected} by more than {tolerance}"
    );
}
