//! Consumer-render oracles for the `ShaderP1` Blender 5.2.2 node fixes.
//!
//! `fixtures/native_oracle/shader_p1/generate.py` builds the committed `.blend` inputs and captures
//! Blender-authored pixels. Color-node charts check exact emission probes for the supported shader
//! nodes. Dedicated probes cover constant-folded Voronoi chains with out-of-range Detail/Roughness,
//! unlinked `Position` and `TexCoord.Generated` inputs in both engines, plus rotated `TexCoord` Normal,
//! negative-scale front/back facing, EEVEE world-environment volume scattering and
//! environment projection/filtering, colored Glass `GGX`/`MULTI_GGX`, missing Temperature and mesh-volume
//! Color Attributes per engine, and Cycles/EEVEE VDB temperature taps.
//! Other scenes cover displacement, blackbody volume, high-offset Noise/Voronoi, and the equirectangular
//! environment seam. Negative fixtures assert import losses and precise render-time
//! `UnsupportedFeature` identifiers; mapping fixtures export through Potter, reopen in Blender, and compare persisted structs.
//!
//! Blender 5.2.2 Cycles CPU provides the path-tracing references; EEVEE Next supplies realtime references.
//! Native sources include `source/blender/nodes/shader/nodes/node_shader_tex_coord.cc`,
//! `node_shader_geometry.cc`, `node_shader_tex_voronoi.cc`, `node_shader_tex_environment.cc`,
//! `node_shader_bsdf_glass.cc`, `node_shader_volume_principled.cc`, EEVEE's texture-coordinate,
//! geometry, environment, Glass, and volume material shaders, Cycles' `intern/cycles/scene/shader_nodes.cpp`,
//! and `intern/cycles/kernel/svm/voronoi.h` / `intern/cycles/kernel/closure/bsdf_microfacet.h`.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "this integration test uses a subset of the shared native-oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{collections::BTreeMap, fs, path::Path};

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, apply_operations, assert_close, assert_floats, checked, export_raw, floats,
    import_blend, loss_features, losses, number, pot, pot_json, read_scene, run_blender_arguments,
    vec3, write_script,
};
use serde_json::{Value, json};
use tempfile::tempdir;

const FIXTURES: &str = "tests/fixtures/native_oracle/shader_p1";
const EMISSION_TOLERANCE: f64 = 2.0e-4;
const PHYSICS_TOLERANCE: f64 = 1.0e-2;
// At 9000 K Blender's total drift was about 0.06% from 4096 to 16384 spp;
// two 16384-spp Potter seed totals spanned about 0.10%.
const BLACKBODY_CHROMATICITY_TOLERANCE: f64 = 1.0e-3;
const BLACKBODY_TOTAL_RELATIVE_TOLERANCE: f64 = 5.0e-3;
// Original EEVEE Glass region-mean bound; the colored single-pixel oracle uses f16 ULPs.
const REALTIME_GLASS_TOLERANCE: f64 = 1.6e-2;

const MAPPING_DUMP_SCRIPT: &str = r#"
import json
import sys

import bpy

output_path = sys.argv[sys.argv.index("--") + 1]
nodes = [
    node
    for material in bpy.data.materials
    if material.node_tree
    for node in material.node_tree.nodes
    if node.bl_idname == "ShaderNodeTexNoise"
]
if len(nodes) != 1:
    raise RuntimeError(f"expected one ShaderNodeTexNoise, found {len(nodes)}")
node = nodes[0]
result = {
    "texture_mapping": {
        "vector_type": node.texture_mapping.vector_type,
        "translation": list(node.texture_mapping.translation),
    },
    "color_mapping": {
        "brightness": float(node.color_mapping.brightness),
    },
}
with open(output_path, "w", encoding="utf-8") as output:
    json.dump(result, output)
"#;

#[derive(Debug)]
struct FloatImage {
    width: usize,
    height: usize,
    rgb: Vec<[f64; 3]>,
}

fn fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES)
}

fn read_reference(name: &str) -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(fixtures().join(name))?)?)
}

fn read_exr(path: &Path) -> TestResult<FloatImage> {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _| FloatImage {
            width: resolution.width(),
            height: resolution.height(),
            rgb: vec![[0.0; 3]; resolution.width() * resolution.height()],
        },
        |pixels: &mut FloatImage, position, (red, green, blue, _alpha): (f32, f32, f32, f32)| {
            let index = position.y() * pixels.width + position.x();
            pixels.rgb[index] = [f64::from(red), f64::from(green), f64::from(blue)];
        },
    )?;
    Ok(image.layer_data.channel_data.pixels)
}

fn pixel(image: &FloatImage, x: usize, y_from_top: usize) -> [f64; 3] {
    image.rgb[y_from_top * image.width + x]
}

fn compare_rgb(actual: [f64; 3], expected: [f64; 3], tolerance: f64, context: &str) {
    for (channel, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context} channel {channel}: Potter {actual} differs from Blender {expected} by more than {tolerance}"
        );
    }
}
/// EEVEE accumulates into an RGBA16F buffer; accumulation order can differ, so allow one film-value ULP.
fn eevee_half_float_ulp(value: f64) -> f64 {
    const MIN_NORMAL: f64 = 6.103_515_625e-5;
    let magnitude = value.abs();
    if magnitude < MIN_NORMAL {
        2.0_f64.powi(-24)
    } else {
        2.0_f64.powf(magnitude.log2().floor() - 10.0)
    }
}

fn compare_eevee_rgb(actual: [f64; 3], expected: [f64; 3], context: &str) {
    for (channel, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = eevee_half_float_ulp(expected);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context} channel {channel}: Potter {actual} differs from Blender {expected} by more than one RGBA16F ULP ({tolerance})"
        );
    }
}

fn render_imported(
    blender: &Path,
    root: &Path,
    fixture: &Path,
    project_name: &str,
    width: u32,
    height: u32,
    samples: u32,
    max_bounces: u32,
) -> TestResult<(std::path::PathBuf, Value)> {
    let (project, report) = import_blend(blender, root, project_name, fixture)?;
    assert_eq!(report["ok"], true, "Blender import failed: {report}");
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("imported scene has no active scene")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root,
        json!([{
            "op": "render.update",
            "target": {"id": active},
            "set": {
                "resolution_x": width,
                "resolution_y": height,
                "resolution_percentage": 100,
                "samples": samples,
                "seed": 0,
                "max_bounces": max_bounces
            }
        }]),
    )?;
    assert_eq!(
        applied["ok"], true,
        "could not configure imported render: {applied}"
    );
    Ok((project, report))
}

fn render_imported_allow_loss(
    blender: &Path,
    root: &Path,
    fixture: &Path,
    project_name: &str,
    width: u32,
    height: u32,
    samples: u32,
    max_bounces: u32,
) -> TestResult<(std::path::PathBuf, Value)> {
    let project = root.join(project_name);
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    let report = pot_json(|command| {
        command
            .arg("import")
            .arg(&project)
            .arg("--file")
            .arg(fixture)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(blender)
            .arg("--allow-lossy");
    })?;
    assert_eq!(report["ok"], true, "Blender import failed: {report}");
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("imported scene has no active scene")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root,
        json!([{
            "op": "render.update",
            "target": {"id": active},
            "set": {
                "resolution_x": width,
                "resolution_y": height,
                "resolution_percentage": 100,
                "samples": samples,
                "seed": 0,
                "max_bounces": max_bounces
            }
        }]),
    )?;
    assert_eq!(
        applied["ok"], true,
        "could not configure imported render: {applied}"
    );
    Ok((project, report))
}

fn render_frame(project: &Path, root: &Path, name: &str) -> TestResult<FloatImage> {
    render_frame_with_engine(project, root, name, "path")
}

fn render_frame_with_engine(
    project: &Path,
    root: &Path,
    name: &str,
    engine: &str,
) -> TestResult<FloatImage> {
    let directory = root.join(name);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--engine", engine, "--format", "exr", "--out"])
        .arg(&directory)
        .arg("--json");
    checked(&mut render, name)?;
    read_exr(&directory.join("frame_0001.exr"))
}

fn region_mean(image: &FloatImage, region: &Value) -> TestResult<[f64; 3]> {
    let fields = region
        .as_array()
        .ok_or("region must be a four-element array")?;
    if fields.len() != 4 {
        return Err("region must be a four-element array".into());
    }
    let field = |index: usize| -> TestResult<usize> {
        Ok(usize::try_from(fields[index].as_u64().ok_or_else(
            || format!("region coordinate {index} is missing"),
        )?)?)
    };
    let x = field(0)?;
    let y = field(1)?;
    let width = field(2)?;
    let height = field(3)?;
    let mut total = [0.0; 3];
    for row in y..y + height {
        for column in x..x + width {
            for (sum, value) in total.iter_mut().zip(pixel(image, column, row)) {
                *sum += value;
            }
        }
    }
    let count = (width * height) as f64;
    Ok(total.map(|sum| sum / count))
}

#[test]
fn shader_node_emission_probes_match_blender_522() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the ShaderP1 node oracle; no executable Blender was found");
        return Ok(());
    };
    let reference = read_reference("shader_nodes.json")?;
    let expected_cases = reference["cases"]
        .as_array()
        .ok_or("shader node cases are missing")?;
    assert_eq!(
        expected_cases.len(),
        73,
        "ShaderP1 chart case count changed"
    );
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("shader_nodes.blend"),
        "shader_nodes_project",
        144,
        144,
        1,
        1,
    )?;
    let frame = render_frame(&project, root.path(), "shader_nodes_render")?;
    assert_eq!([frame.width, frame.height], [144, 144]);
    let mut probe_count = 0;
    let mut observed = BTreeMap::new();
    for case in expected_cases {
        let name = case["name"].as_str().ok_or("shader case name is missing")?;
        let mut case_pixels = Vec::new();
        for probe in case["probes"]
            .as_array()
            .ok_or("shader probes are missing")?
        {
            let x = usize::try_from(probe["x"].as_u64().ok_or("probe x is missing")?)?;
            let y = usize::try_from(probe["y"].as_u64().ok_or("probe y is missing")?)?;
            let actual = pixel(&frame, x, y);
            compare_rgb(
                actual,
                vec3(&probe["rgb"])?,
                EMISSION_TOLERANCE,
                &format!("{name} at ({x},{y})"),
            );
            case_pixels.push(actual);
            probe_count += 1;
        }
        observed.insert(name, case_pixels);
    }
    assert!(probe_count > expected_cases.len());
    assert_eq!(
        observed["checker_default_unlinked_generated"],
        observed["checker_generated_coordinates"]
    );
    assert_ne!(
        observed["fresnel_front"],
        observed["fresnel_backface_geometry"]
    );
    assert_ne!(
        observed["layer_weight_fresnel"],
        observed["layer_weight_fresnel_backface"]
    );
    assert_ne!(
        observed["layer_weight_facing"],
        observed["layer_weight_facing_backface"]
    );
    assert_ne!(
        observed["image_projection_box"],
        observed["image_box_rotated_scaled_front"]
    );
    assert_eq!(
        observed["image_box_rotated_scaled_backface"].len(),
        1,
        "the transformed BOX back-face probe must be rendered"
    );
    assert_eq!(
        observed["image_interpolation_smart"],
        observed["image_interpolation_cubic"]
    );
    assert_eq!(
        observed["environment_equirectangular_smart"],
        observed["environment_equirectangular_cubic"]
    );
    assert_eq!(
        observed["environment_mirror_ball_smart"],
        observed["environment_mirror_ball_cubic"]
    );
    let repeat = &observed["image_cubic_extension_repeat_edge"];
    for extension in [
        "image_cubic_extension_extend_edge",
        "image_cubic_extension_clip_edge",
        "image_cubic_extension_mirror_edge",
    ] {
        assert_ne!(
            repeat, &observed[extension],
            "Cubic interpolation must exercise {extension}"
        );
    }
    Ok(())
}

#[test]
fn voronoi_values_above_blender_limits_match_blender_522() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the ShaderP1 Voronoi clamp oracle; no executable Blender was found");
        return Ok(());
    };
    let reference = read_reference("voronoi_clamps.json")?;
    let cases = reference["cases"]
        .as_array()
        .ok_or("Voronoi clamp cases are missing")?;
    assert_eq!(cases.len(), 5);
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("voronoi_clamps.blend"),
        "voronoi_clamps_project",
        64,
        32,
        1,
        1,
    )?;
    let frame = render_frame(&project, root.path(), "voronoi_clamps_render")?;
    assert_eq!([frame.width, frame.height], [64, 32]);
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("Voronoi clamp case name is missing")?;
        let should_exceed_clamp = matches!(name, "detail_over_15" | "roughness_over_1");
        assert_eq!(
            number(&case["input_value"])? > number(&case["clamped_value"])?,
            should_exceed_clamp,
            "{name}: unexpected input/clamp relationship"
        );
        let probe = case["probe"]
            .as_array()
            .ok_or("Voronoi clamp probe is missing")?;
        let x = usize::try_from(probe[0].as_u64().ok_or("probe x is missing")?)?;
        let y = usize::try_from(probe[1].as_u64().ok_or("probe y is missing")?)?;
        compare_rgb(
            pixel(&frame, x, y),
            vec3(&case["rgb"])?,
            EMISSION_TOLERANCE,
            &format!("{name} at ({x},{y})"),
        );
    }
    Ok(())
}

#[test]
fn voronoi_constant_field_and_unlinked_position_inputs_match_each_blender_engine() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the ShaderP1 Voronoi input-folding oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("voronoi_fold_fields.json")?;
    assert_eq!(reference["samples"]["eevee"], 1);
    assert_eq!(reference["samples"]["cycles"], 1);
    assert_eq!(reference["seed"], 0);
    let cases = reference["cases"]
        .as_array()
        .ok_or("Voronoi folding cases are missing")?;
    assert_eq!(cases.len(), 3);
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("voronoi_fold_fields.blend"),
        "voronoi_fold_fields_project",
        64,
        32,
        1,
        1,
    )?;
    let mut expected_by_engine = BTreeMap::new();
    for (engine, reference_key) in [("realtime", "eevee_rgb"), ("path", "cycles_rgb")] {
        let frame = render_frame_with_engine(
            &project,
            root.path(),
            &format!("voronoi_fold_fields_{engine}"),
            engine,
        )?;
        assert_eq!([frame.width, frame.height], [64, 32]);
        let mut values = BTreeMap::new();
        for case in cases {
            let name = case["name"]
                .as_str()
                .ok_or("Voronoi folding case name is missing")?;
            assert_eq!(number(&case["detail"])?, 20.0);
            assert_eq!(number(&case["roughness"])?, 1.4);
            let probe = case["probe"]
                .as_array()
                .ok_or("Voronoi folding probe is missing")?;
            let x = usize::try_from(probe[0].as_u64().ok_or("probe x is missing")?)?;
            let y = usize::try_from(probe[1].as_u64().ok_or("probe y is missing")?)?;
            let actual = pixel(&frame, x, y);
            let expected = vec3(&case[reference_key])?;
            let context = format!("{name} {engine} at ({x},{y})");
            if engine == "realtime" {
                compare_eevee_rgb(actual, expected, &context);
            } else {
                compare_rgb(actual, expected, EMISSION_TOLERANCE, &context);
            }
            values.insert(name, expected);
        }
        expected_by_engine.insert(engine, values);
    }
    for engine in ["realtime", "path"] {
        let values = &expected_by_engine[engine];
        let constant = values
            .get("voronoi_constant_folded")
            .ok_or("constant-folded Voronoi probe is missing")?;
        let position = values
            .get("voronoi_unlinked_position")
            .ok_or("unlinked Position-field Voronoi probe is missing")?;
        let field = values
            .get("voronoi_texcoord_generated")
            .ok_or("TexCoord-driven Voronoi probe is missing")?;
        assert_ne!(
            constant, position,
            "{engine}: unlinked Vector must evaluate the implicit Position field"
        );
        assert_ne!(
            position, field,
            "{engine}: unlinked Position and explicit Generated fields must differ"
        );
        assert_ne!(
            constant, field,
            "{engine}: constant-folded and field-driven Voronoi probes must differ"
        );
    }
    Ok(())
}

#[test]
fn realtime_texcoord_normal_tracks_rotated_object() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the ShaderP1 rotated Normal oracle; no executable Blender was found");
        return Ok(());
    };
    let reference = read_reference("texcoord_normal_eevee.json")?;
    assert_eq!(reference["samples"], 1);
    assert_eq!(reference["seed"], 0);
    let cases = reference["cases"]
        .as_array()
        .ok_or("Texture Coordinate Normal cases are missing")?;
    assert_eq!(cases.len(), 2);
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("texcoord_normal_eevee.blend"),
        "texcoord_normal_eevee_project",
        64,
        32,
        1,
        1,
    )?;
    let frame = render_frame_with_engine(
        &project,
        root.path(),
        "texcoord_normal_eevee_render",
        "realtime",
    )?;
    assert_eq!([frame.width, frame.height], [64, 32]);
    let mut observed = BTreeMap::new();
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("Texture Coordinate Normal case name is missing")?;
        let probe = case["probe"]
            .as_array()
            .ok_or("Texture Coordinate Normal probe is missing")?;
        let x = usize::try_from(probe[0].as_u64().ok_or("probe x is missing")?)?;
        let y = usize::try_from(probe[1].as_u64().ok_or("probe y is missing")?)?;
        let expected = vec3(&case["rgb"])?;
        compare_eevee_rgb(
            pixel(&frame, x, y),
            expected,
            &format!("{name} at ({x},{y})"),
        );
        observed.insert(name, expected);
    }
    assert_eq!(
        observed["normal_rotated_90"],
        [0.0, 0.0, 1.0],
        "TexCoord Normal is object-space: the rotated plane still probes its local +Z normal"
    );
    Ok(())
}

#[test]
fn negative_scale_preserves_blender_front_back_facing() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the ShaderP1 negative-scale facing oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("negative_scale_facing.json")?;
    assert_eq!(reference["samples"]["eevee"], 1);
    assert_eq!(reference["samples"]["cycles"], 1);
    assert_eq!(reference["seed"], 0);
    let cases = reference["cases"]
        .as_array()
        .ok_or("negative-scale facing cases are missing")?;
    assert_eq!(cases.len(), 2);
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("negative_scale_facing.blend"),
        "negative_scale_facing_project",
        64,
        32,
        1,
        1,
    )?;
    for (engine, reference_key) in [("realtime", "eevee_rgb"), ("path", "cycles_rgb")] {
        let frame = render_frame_with_engine(
            &project,
            root.path(),
            &format!("negative_scale_facing_{engine}"),
            engine,
        )?;
        assert_eq!([frame.width, frame.height], [64, 32]);
        let mut observed = BTreeMap::new();
        for case in cases {
            let name = case["name"]
                .as_str()
                .ok_or("negative-scale facing case name is missing")?;
            let probe = case["probe"]
                .as_array()
                .ok_or("negative-scale facing probe is missing")?;
            let x = usize::try_from(probe[0].as_u64().ok_or("probe x is missing")?)?;
            let y = usize::try_from(probe[1].as_u64().ok_or("probe y is missing")?)?;
            let expected = vec3(&case[reference_key])?;
            let context = format!("{name} {engine} at ({x},{y})");
            let actual = pixel(&frame, x, y);
            if engine == "realtime" {
                compare_eevee_rgb(actual, expected, &context);
            } else {
                compare_rgb(actual, expected, EMISSION_TOLERANCE, &context);
            }
            observed.insert(name, expected);
        }
        assert_ne!(
            observed["facing_positive_scale"], observed["facing_negative_scale"],
            "{engine}: a negative object scale must change the face-facing probe"
        );
    }
    Ok(())
}

#[test]
fn world_environment_projection_and_interpolation_match_each_engine() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the ShaderP1 world-environment oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("environment_world.json")?;
    assert_eq!(reference["seed"], 0);
    assert_eq!(reference["samples"]["eevee"], 1);
    assert_eq!(reference["samples"]["cycles"], 1);
    let cases = reference["cases"]
        .as_array()
        .ok_or("world environment cases are missing")?;
    assert_eq!(cases.len(), 2);
    let root = tempdir()?;
    let mut eevee_probes = Vec::new();
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("world environment case name is missing")?;
        let blend = case["blend"]
            .as_str()
            .ok_or("world environment blend is missing")?;
        let (project, _) = render_imported(
            &blender,
            root.path(),
            &fixtures().join(blend),
            &format!("{name}_project"),
            64,
            64,
            1,
            1,
        )?;
        let probe = case["probe"]
            .as_array()
            .ok_or("world environment probe is missing")?;
        let x = usize::try_from(probe[0].as_u64().ok_or("probe x is missing")?)?;
        let y = usize::try_from(probe[1].as_u64().ok_or("probe y is missing")?)?;
        for (engine, reference_key) in [("realtime", "eevee_rgb"), ("path", "cycles_rgb")] {
            let frame = render_frame_with_engine(
                &project,
                root.path(),
                &format!("{name}_{engine}_render"),
                engine,
            )?;
            assert_eq!([frame.width, frame.height], [64, 64]);
            let expected = vec3(&case[reference_key])?;
            let context = format!("{name} {engine}");
            if engine == "realtime" {
                compare_eevee_rgb(pixel(&frame, x, y), expected, &context);
                eevee_probes.push(expected);
            } else {
                compare_rgb(pixel(&frame, x, y), expected, EMISSION_TOLERANCE, &context);
            }
        }
    }
    assert_ne!(
        eevee_probes[0], eevee_probes[1],
        "EEVEE world probes must distinguish environment projection/filtering"
    );
    Ok(())
}

#[test]
fn realtime_volume_ambient_reads_eevee_world_environment_at_v_seam() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the realtime volume-world ambient oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("volume_world_ambient_eevee.json")?;
    assert_eq!(reference["samples"], 16);
    assert_eq!(reference["seed"], 0);
    assert_eq!(reference["environment_projection"], "EQUIRECTANGULAR");
    assert_eq!(reference["environment_interpolation"], "Linear");
    assert_eq!(vec3(&reference["ambient_direction"])?, [0.0, 0.0, 1.0]);
    let case = &reference["case"];
    assert_eq!(
        case["name"].as_str(),
        Some("world_ambient_volume_equirectangular_v_seam")
    );
    let blend = case["blend"]
        .as_str()
        .ok_or("volume world-ambient fixture path is missing")?;
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join(blend),
        "volume_world_ambient_eevee_project",
        64,
        64,
        16,
        1,
    )?;
    let frame = render_frame_with_engine(
        &project,
        root.path(),
        "volume_world_ambient_eevee_render",
        "realtime",
    )?;
    assert_eq!([frame.width, frame.height], [64, 64]);
    let probe = case["probe"]
        .as_array()
        .ok_or("volume world-ambient probe is missing")?;
    let x = usize::try_from(probe[0].as_u64().ok_or("probe x is missing")?)?;
    let y = usize::try_from(probe[1].as_u64().ok_or("probe y is missing")?)?;
    let expected = vec3(&case["rgb"])?;
    let background = vec3(&case["background_rgb"])?;
    assert!(
        (0..3).any(|channel| (expected[channel] - background[channel]).abs() > 1.0e-3),
        "the EEVEE world environment must contribute to volume scattering"
    );
    compare_eevee_rgb(
        pixel(&frame, x, y),
        expected,
        &format!("volume world ambient at the EQUIRECTANGULAR v seam ({x},{y})"),
    );
    Ok(())
}

#[test]
fn glass_distribution_matches_blender_522() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the ShaderP1 Glass distribution oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("glass_distribution.json")?;
    let cases = reference["cases"]
        .as_array()
        .ok_or("Glass distribution cases are missing")?;
    let samples = u32::try_from(
        reference["samples"]
            .as_u64()
            .ok_or("Glass sample count is missing")?,
    )?;
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("glass_distribution.blend"),
        "glass_distribution_project",
        64,
        64,
        samples,
        4,
    )?;
    let frame = render_frame(&project, root.path(), "glass_distribution_render")?;
    assert_eq!([frame.width, frame.height], [64, 64]);
    let mut actual_by_name = BTreeMap::new();
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("Glass distribution case name is missing")?;
        let region = case["region"]
            .as_array()
            .ok_or("Glass distribution region is missing")?;
        let actual = region_mean(&frame, &Value::Array(region.clone()))?;
        let expected = vec3(&case["mean"])?;
        compare_rgb(actual, expected, PHYSICS_TOLERANCE, name);
        actual_by_name.insert(name, actual);
    }
    let ggx = actual_by_name
        .get("glass_ggx")
        .ok_or("GGX render is missing")?;
    let multi_ggx = actual_by_name
        .get("glass_multi_ggx")
        .ok_or("explicit MULTI_GGX render is missing")?;
    let default_glass = actual_by_name
        .get("glass_default")
        .ok_or("default Glass render is missing")?;
    assert_floats(
        default_glass,
        multi_ggx,
        PHYSICS_TOLERANCE,
        "default MULTI_GGX versus explicit MULTI_GGX",
    );
    assert!(
        multi_ggx[0] - ggx[0] > 0.01,
        "Glass GGX and MULTI_GGX must remain distinct at the same roughness and IOR: GGX={ggx:?}, MULTI_GGX={multi_ggx:?}"
    );
    Ok(())
}

#[test]
fn blackbody_volume_matches_blender_522() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the ShaderP1 blackbody Volume oracle; no executable Blender was found");
        return Ok(());
    };
    let reference = read_reference("shader_physics.json")?;
    let samples = u32::try_from(
        reference["samples"]
            .as_u64()
            .ok_or("blackbody Volume sample count is missing")?,
    )?;
    let cases = reference["cases"]
        .as_array()
        .ok_or("blackbody Volume cases are missing")?;
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("shader_physics.blend"),
        "shader_physics_project",
        64,
        64,
        samples,
        4,
    )?;
    let frame = render_frame(&project, root.path(), "shader_physics_render")?;
    assert_eq!([frame.width, frame.height], [64, 64]);
    let mut actual_by_name = BTreeMap::new();
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("blackbody Volume case name is missing")?;
        let region = case["region"]
            .as_array()
            .ok_or("blackbody Volume region is missing")?;
        let actual = region_mean(&frame, &Value::Array(region.clone()))?;
        let expected = vec3(&case["mean"])?;
        if name == "volume_blackbody_9000k" {
            let actual_total: f64 = actual.iter().sum();
            let expected_total: f64 = expected.iter().sum();
            assert!(
                (actual_total - expected_total).abs()
                    <= expected_total * BLACKBODY_TOTAL_RELATIVE_TOLERANCE,
                "{name} total magnitude: Potter {actual_total} differs from Blender {expected_total} by more than {:.1}%",
                BLACKBODY_TOTAL_RELATIVE_TOLERANCE * 100.0
            );
            compare_rgb(
                actual.map(|channel| channel / actual_total),
                expected.map(|channel| channel / expected_total),
                BLACKBODY_CHROMATICITY_TOLERANCE,
                "volume_blackbody_9000k chromaticity",
            );
        } else {
            compare_rgb(actual, expected, PHYSICS_TOLERANCE, name);
        }
        actual_by_name.insert(name, actual);
    }
    let cool = actual_by_name
        .get("volume_blackbody_1800k")
        .ok_or("1800 K volume render is missing")?;
    let hot = actual_by_name
        .get("volume_blackbody_9000k")
        .ok_or("9000 K volume render is missing")?;
    assert!(
        hot[2] > hot[0] && cool[0] > cool[2],
        "blackbody temperatures must differ in hue"
    );
    Ok(())
}

#[test]
fn principled_volume_blackbody_samples_temperature_vdb_per_step() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the ShaderP1 temperature-grid oracle; no executable Blender was found");
        return Ok(());
    };
    let reference = read_reference("volume_temperature_grid.json")?;
    assert_eq!(reference["seed"], 0);
    let samples = u32::try_from(
        reference["samples"]
            .as_u64()
            .ok_or("temperature-grid sample count is missing")?,
    )?;
    let cases = reference["cases"]
        .as_array()
        .ok_or("temperature-grid cases are missing")?;
    assert_eq!(cases.len(), 2);
    let root = tempdir()?;
    let (project, report) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("volume_temperature_grid.blend"),
        "volume_temperature_grid_project",
        32,
        32,
        samples,
        4,
    )?;
    assert!(
        !loss_features(&report)
            .iter()
            .any(|feature| feature.as_str() == "shader.volume.spatially_varying_graph"),
        "default temperature-grid sampling must import without a spatial-graph loss: {report}"
    );
    let frame = render_frame(&project, root.path(), "volume_temperature_grid_render")?;
    assert_eq!([frame.width, frame.height], [32, 32]);
    let mut actual_by_name = BTreeMap::new();
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("temperature-grid case name is missing")?;
        let region = case["region"]
            .as_array()
            .ok_or("temperature-grid patch is missing")?;
        let actual = region_mean(&frame, &Value::Array(region.clone()))?;
        let expected = vec3(&case["mean"])?;
        let actual_total: f64 = actual.iter().sum();
        let expected_total: f64 = expected.iter().sum();
        assert!(
            (actual_total - expected_total).abs()
                <= expected_total * BLACKBODY_TOTAL_RELATIVE_TOLERANCE,
            "{name} total magnitude: Potter {actual_total} differs from Blender {expected_total} by more than {:.1}%",
            BLACKBODY_TOTAL_RELATIVE_TOLERANCE * 100.0
        );
        compare_rgb(
            actual.map(|channel| channel / actual_total),
            expected.map(|channel| channel / expected_total),
            BLACKBODY_CHROMATICITY_TOLERANCE,
            name,
        );
        actual_by_name.insert(name, actual);
    }
    let cool = actual_by_name
        .get("temperature_grid_cool")
        .ok_or("cool temperature-grid patch is missing")?;
    let hot = actual_by_name
        .get("temperature_grid_hot")
        .ok_or("hot temperature-grid patch is missing")?;
    assert!(
        hot[0] - cool[0] > 0.1,
        "VDB temperature samples must change blackbody emission: cool={cool:?}, hot={hot:?}"
    );
    Ok(())
}

#[test]
fn realtime_vdb_temperature_grid_matches_half_float_taps() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the realtime ShaderP1 VDB temperature oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("volume_temperature_grid.json")?;
    assert_eq!(reference["eevee_samples"], 1);
    assert_eq!(reference["seed"], 0);
    let cases = reference["cases"]
        .as_array()
        .ok_or("temperature-grid cases are missing")?;
    assert_eq!(cases.len(), 2);
    let root = tempdir()?;
    let (project, report) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("volume_temperature_grid.blend"),
        "volume_temperature_grid_eevee_project",
        32,
        32,
        1,
        4,
    )?;
    assert!(
        !loss_features(&report)
            .iter()
            .any(|feature| feature.as_str() == "shader.volume.spatially_varying_graph"),
        "default temperature-grid sampling must import without a spatial-graph loss: {report}"
    );
    let frame = render_frame_with_engine(
        &project,
        root.path(),
        "volume_temperature_grid_eevee_render",
        "realtime",
    )?;
    assert_eq!([frame.width, frame.height], [32, 32]);
    let mut observed = BTreeMap::new();
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("temperature-grid case name is missing")?;
        let probe = case["eevee_probe"]
            .as_array()
            .ok_or("temperature-grid EEVEE probe is missing")?;
        let x = usize::try_from(probe[0].as_u64().ok_or("probe x is missing")?)?;
        let y = usize::try_from(probe[1].as_u64().ok_or("probe y is missing")?)?;
        let expected = vec3(&case["eevee_rgb"])?;
        compare_eevee_rgb(
            pixel(&frame, x, y),
            expected,
            &format!("{name} EEVEE temperature tap at ({x},{y})"),
        );
        observed.insert(name, expected);
    }
    assert_ne!(
        observed["temperature_grid_cool"], observed["temperature_grid_hot"],
        "the realtime temperature grid must change the half-float tap"
    );
    Ok(())
}

#[test]
fn realtime_glass_ggx_and_multiscatter_match_eevee() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the realtime ShaderP1 Glass oracle; no executable Blender was found");
        return Ok(());
    };
    let reference = read_reference("glass_realtime.json")?;
    assert_eq!(reference["samples"], 64);
    let cases = reference["cases"]
        .as_array()
        .ok_or("realtime Glass cases are missing")?;
    assert_eq!(cases.len(), 2);
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("glass_realtime.blend"),
        "glass_realtime_project",
        96,
        96,
        64,
        1,
    )?;
    let frame =
        render_frame_with_engine(&project, root.path(), "glass_realtime_render", "realtime")?;
    assert_eq!([frame.width, frame.height], [96, 96]);
    let mut actual_by_name = BTreeMap::new();
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("realtime Glass case name is missing")?;
        let region = case["region"]
            .as_array()
            .ok_or("realtime Glass patch is missing")?;
        let actual = region_mean(&frame, &Value::Array(region.clone()))?;
        compare_rgb(actual, vec3(&case["mean"])?, REALTIME_GLASS_TOLERANCE, name);
        actual_by_name.insert(name, actual);
    }
    let ggx = actual_by_name
        .get("glass_ggx")
        .ok_or("realtime GGX render is missing")?;
    let multi_ggx = actual_by_name
        .get("glass_multi_ggx")
        .ok_or("realtime MULTI_GGX render is missing")?;
    assert!(
        multi_ggx[0] - ggx[0] > 1.0e-3,
        "realtime Glass distributions collapsed: GGX={ggx:?}, MULTI_GGX={multi_ggx:?}"
    );
    Ok(())
}

#[test]
fn realtime_colored_glass_distribution_delta_matches_eevee() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the colored realtime ShaderP1 Glass oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("glass_realtime_colored.json")?;
    assert_eq!(reference["samples"], 64);
    assert_eq!(reference["seed"], 0);
    let cases = reference["cases"]
        .as_array()
        .ok_or("colored realtime Glass cases are missing")?;
    assert_eq!(cases.len(), 2);
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("glass_realtime_colored.blend"),
        "glass_realtime_colored_project",
        96,
        96,
        64,
        1,
    )?;
    let frame = render_frame_with_engine(
        &project,
        root.path(),
        "glass_realtime_colored_render",
        "realtime",
    )?;
    assert_eq!([frame.width, frame.height], [96, 96]);
    let mut actual_by_name = BTreeMap::new();
    let mut reference_by_name = BTreeMap::new();
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("colored realtime Glass case name is missing")?;
        let probe = case["probe"]
            .as_array()
            .ok_or("colored realtime Glass pixel probe is missing")?;
        let x = usize::try_from(probe[0].as_u64().ok_or("probe x is missing")?)?;
        let y = usize::try_from(probe[1].as_u64().ok_or("probe y is missing")?)?;
        let actual = pixel(&frame, x, y);
        let expected = vec3(&case["rgb"])?;
        compare_eevee_rgb(actual, expected, &format!("{name} at ({x},{y})"));
        actual_by_name.insert(name, actual);
        reference_by_name.insert(name, expected);
    }
    let ggx = actual_by_name
        .get("glass_ggx")
        .ok_or("colored realtime GGX render is missing")?;
    let multi_ggx = actual_by_name
        .get("glass_multi_ggx")
        .ok_or("colored realtime MULTI_GGX render is missing")?;
    let expected_ggx = reference_by_name
        .get("glass_ggx")
        .ok_or("colored realtime GGX Blender reference is missing")?;
    let expected_multi_ggx = reference_by_name
        .get("glass_multi_ggx")
        .ok_or("colored realtime MULTI_GGX Blender reference is missing")?;
    let mut distinguishes_distributions = false;
    for channel in 0..3 {
        let blender_delta = expected_multi_ggx[channel] - expected_ggx[channel];
        let actual_delta = multi_ggx[channel] - ggx[channel];
        let pixel_rounding_bound = eevee_half_float_ulp(expected_ggx[channel])
            + eevee_half_float_ulp(expected_multi_ggx[channel]);
        assert!(
            (actual_delta - blender_delta).abs() <= pixel_rounding_bound,
            "Glass distribution delta channel {channel}: Potter {actual_delta} differs from Blender {blender_delta} by more than the two pixel-rounding ULPs ({pixel_rounding_bound})"
        );
        distinguishes_distributions |= blender_delta.abs() > 1.0e-3;
    }
    assert!(
        distinguishes_distributions,
        "Blender's colored Glass oracle must distinguish GGX from MULTI_GGX: GGX={expected_ggx:?}, MULTI_GGX={expected_multi_ggx:?}"
    );
    assert!(
        (0..3).any(|channel| (multi_ggx[channel] - ggx[channel]).abs() > 1.0e-3),
        "colored realtime Glass distributions collapsed: GGX={ggx:?}, MULTI_GGX={multi_ggx:?}"
    );
    Ok(())
}

#[test]
fn displacement_space_matches_blender_engine_probes() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the ShaderP1 displacement oracle; no executable Blender was found");
        return Ok(());
    };
    let reference = read_reference("displacement_space.json")?;
    assert_eq!(reference["samples"]["eevee"], 1);
    assert_eq!(reference["samples"]["cycles"], 1);
    assert_eq!(reference["seed"], 0);
    let expected_cases = reference["cases"]
        .as_array()
        .ok_or("displacement cases are missing")?;
    assert_eq!(expected_cases.len(), 2);
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("displacement_space.blend"),
        "displacement_project",
        64,
        64,
        1,
        1,
    )?;
    for (engine, expected_key) in [("realtime", "eevee_rgb"), ("path", "cycles_rgb")] {
        let frame = render_frame_with_engine(
            &project,
            root.path(),
            &format!("displacement_{engine}_render"),
            engine,
        )?;
        assert_eq!([frame.width, frame.height], [64, 64]);
        for case in expected_cases {
            let name = case["name"]
                .as_str()
                .ok_or("displacement case name is missing")?;
            for probe in case["probes"]
                .as_array()
                .ok_or("displacement probes are missing")?
            {
                let x = usize::try_from(probe["x"].as_u64().ok_or("probe x is missing")?)?;
                let y = usize::try_from(probe["y"].as_u64().ok_or("probe y is missing")?)?;
                let actual = pixel(&frame, x, y);
                let expected = vec3(&probe[expected_key])?;
                let context = format!("{name} {engine} at ({x},{y})");
                if engine == "realtime" {
                    compare_eevee_rgb(actual, expected, &context);
                } else {
                    compare_rgb(actual, expected, EMISSION_TOLERANCE, &context);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn volume_blackbody_matches_engine_specific_blender_probes() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the ShaderP1 engine-specific blackbody oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("volume_blackbody_engines.json")?;
    assert_eq!(reference["samples"]["eevee"], 1);
    assert_eq!(reference["samples"]["cycles"], 1);
    assert_eq!(reference["seed"], 0);
    assert_eq!(reference["density"], 0.0);
    let cases = reference["cases"]
        .as_array()
        .ok_or("engine-specific blackbody cases are missing")?;
    assert_eq!(cases.len(), 2);
    assert_eq!(number(&reference["temperature_socket"])?, 1000.0);
    assert_eq!(number(&reference["blackbody_intensity"])?, 0.2);
    assert_eq!(number(&cases[0]["temperature_attribute"])?, 1.8);
    assert_eq!(number(&cases[1]["temperature_attribute"])?, 10.2);
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("volume_blackbody_engines.blend"),
        "volume_blackbody_engines_project",
        64,
        64,
        1,
        4,
    )?;
    for (engine, reference_key) in [("realtime", "eevee_mean"), ("path", "cycles_mean")] {
        let frame = render_frame_with_engine(
            &project,
            root.path(),
            &format!("volume_blackbody_{engine}_render"),
            engine,
        )?;
        assert_eq!([frame.width, frame.height], [64, 64]);
        for case in cases {
            let name = case["name"]
                .as_str()
                .ok_or("engine-specific blackbody case name is missing")?;
            let region = case["region"]
                .as_array()
                .ok_or("engine-specific blackbody region is missing")?;
            let actual = region_mean(&frame, &Value::Array(region.clone()))?;
            let expected = vec3(&case[reference_key])?;
            let context = format!("{name} {engine}");
            if engine == "realtime" {
                compare_eevee_rgb(actual, expected, &context);
            } else {
                compare_rgb(actual, expected, PHYSICS_TOLERANCE, &context);
            }
        }
    }
    Ok(())
}

#[test]
fn volume_missing_temperature_and_mesh_color_attributes_match_each_engine() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the ShaderP1 volume attribute oracle; no executable Blender was found");
        return Ok(());
    };
    let reference = read_reference("volume_attribute_engines.json")?;
    assert_eq!(reference["samples"]["eevee"], 64);
    assert_eq!(reference["samples"]["cycles"], 64);
    assert_eq!(reference["seed"], 0);
    let cases = reference["cases"]
        .as_array()
        .ok_or("volume attribute cases are missing")?;
    assert_eq!(cases.len(), 2);
    assert_eq!(
        cases[0]["name"].as_str(),
        Some("missing_temperature_attribute")
    );
    assert_eq!(cases[0]["object_type"].as_str(), Some("MESH_VOLUME"));
    assert_eq!(vec3(&cases[0]["eevee_rgb"])?, [0.0; 3]);
    let cycles_missing_temperature = vec3(&cases[0]["cycles_rgb"])?;
    assert!(
        cycles_missing_temperature
            .iter()
            .any(|channel| *channel > 0.0),
        "Cycles must retain the Temperature socket when the mesh lacks that attribute"
    );
    assert_eq!(
        cases[1]["name"].as_str(),
        Some("mesh_volume_color_attribute")
    );
    assert_eq!(cases[1]["object_type"].as_str(), Some("MESH_VOLUME"));
    assert_ne!(
        vec3(&cases[1]["eevee_rgb"])?,
        vec3(&cases[1]["cycles_rgb"])?,
        "mesh-volume Color Attribute references must stay engine-specific"
    );
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("volume_attribute_engines.blend"),
        "volume_attribute_engines_project",
        64,
        32,
        64,
        4,
    )?;
    for (engine, reference_key) in [("realtime", "eevee_rgb"), ("path", "cycles_rgb")] {
        let frame = render_frame_with_engine(
            &project,
            root.path(),
            &format!("volume_attributes_{engine}"),
            engine,
        )?;
        assert_eq!([frame.width, frame.height], [64, 32]);
        for case in cases {
            let name = case["name"]
                .as_str()
                .ok_or("volume attribute case name is missing")?;
            let region = case["region"]
                .as_array()
                .ok_or("volume attribute probe region is missing")?;
            let actual = region_mean(&frame, &Value::Array(region.clone()))?;
            let expected = vec3(&case[reference_key])?;
            let context = format!("{name} {engine}");
            if engine == "realtime" {
                compare_eevee_rgb(actual, expected, &context);
            } else {
                compare_rgb(actual, expected, PHYSICS_TOLERANCE, &context);
            }
        }
    }
    Ok(())
}

#[test]
fn equirectangular_environment_axes_match_eevee() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the ShaderP1 EEVEE environment-axis oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("environment_axis_seam.json")?;
    assert_eq!(reference["samples"], 1);
    let cases = reference["cases"]
        .as_array()
        .ok_or("environment-axis cases are missing")?;
    assert_eq!(cases.len(), 4);
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("environment_axis_seam.blend"),
        "environment_axis_seam_project",
        65,
        65,
        1,
        1,
    )?;
    let frame = render_frame_with_engine(
        &project,
        root.path(),
        "environment_axis_seam_render",
        "realtime",
    )?;
    assert_eq!([frame.width, frame.height], [65, 65]);
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("environment-axis case name is missing")?;
        let probe = case["probe"]
            .as_array()
            .ok_or("environment-axis probe is missing")?;
        let x = usize::try_from(probe[0].as_u64().ok_or("probe x is missing")?)?;
        let y = usize::try_from(probe[1].as_u64().ok_or("probe y is missing")?)?;
        compare_eevee_rgb(pixel(&frame, x, y), vec3(&case["rgb"])?, name);
    }
    Ok(())
}

#[test]
fn high_offset_noise_and_voronoi_match_each_blender_engine() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the ShaderP1 high-offset Noise/Voronoi oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("shader_gpu_precision.json")?;
    assert_eq!(reference["samples"]["eevee"], 1);
    assert_eq!(reference["samples"]["cycles"], 1);
    assert_eq!(reference["seed"], 0);
    let cases = reference["cases"]
        .as_array()
        .ok_or("high-offset Noise/Voronoi cases are missing")?;
    assert_eq!(cases.len(), 2);
    let root = tempdir()?;
    let (project, _) = render_imported(
        &blender,
        root.path(),
        &fixtures().join("shader_gpu_precision.blend"),
        "shader_gpu_precision_project",
        64,
        64,
        1,
        1,
    )?;
    for (engine, reference_key) in [("realtime", "eevee_rgb"), ("path", "cycles_rgb")] {
        let frame = render_frame_with_engine(
            &project,
            root.path(),
            &format!("shader_gpu_precision_{engine}_render"),
            engine,
        )?;
        assert_eq!([frame.width, frame.height], [64, 64]);
        for case in cases {
            let name = case["name"]
                .as_str()
                .ok_or("high-offset texture case name is missing")?;
            for probe in case["probes"]
                .as_array()
                .ok_or("high-offset texture probes are missing")?
            {
                let x = usize::try_from(probe["x"].as_u64().ok_or("probe x is missing")?)?;
                let y = usize::try_from(probe["y"].as_u64().ok_or("probe y is missing")?)?;
                let actual = pixel(&frame, x, y);
                let expected = vec3(&probe[reference_key])?;
                let context = format!("{name} {engine} at ({x},{y})");
                if engine == "realtime" {
                    compare_eevee_rgb(actual, expected, &context);
                } else {
                    compare_rgb(actual, expected, EMISSION_TOLERANCE, &context);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn realtime_linear_image_minification_reports_unsupported_feature() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the realtime ShaderP1 Image Texture minification oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("image_linear_minification.json")?;
    assert_eq!(reference["samples"], 1);
    let case = reference["cases"]
        .as_array()
        .and_then(|cases| cases.first())
        .ok_or("Image Texture minification refusal case is missing")?;
    let name = case["name"]
        .as_str()
        .ok_or("Image Texture minification case name is missing")?;
    let blend = case["blend"]
        .as_str()
        .ok_or("Image Texture minification fixture is missing")?;
    let root = tempdir()?;
    let (project, report) = render_imported(
        &blender,
        root.path(),
        &fixtures().join(blend),
        &format!("{name}_project"),
        16,
        16,
        1,
        1,
    )?;
    assert_eq!(report["ok"], true, "Blender import failed: {report}");
    let output_directory = root.path().join(format!("{name}_render"));
    let mut command = pot();
    command
        .arg("render")
        .arg(&project)
        .args([
            "--engine",
            case["engine"].as_str().ok_or("engine is missing")?,
            "--format",
            "exr",
            "--out",
        ])
        .arg(output_directory)
        .arg("--json");
    let output = command.output()?;
    assert!(
        !output.status.success(),
        "{name}: minified LINEAR image unexpectedly rendered"
    );
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(response["error"]["code"], case["error_code"], "{response}");
    assert_eq!(
        response["error"]["details"]["feature_id"], case["feature_id"],
        "{response}"
    );
    Ok(())
}

#[test]
fn texture_node_mappings_refuse_evaluation_and_survive_blend_roundtrip() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the ShaderP1 mapping oracle; no executable Blender was found");
        return Ok(());
    };
    let reference = read_reference("mapping_refusals.json")?;
    let cases = reference["cases"]
        .as_array()
        .ok_or("mapping cases are missing")?;
    assert_eq!(cases.len(), 2);
    let root = tempdir()?;
    let dump_script = write_script(root.path(), "dump_mapping.py", MAPPING_DUMP_SCRIPT)?;
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("mapping case name is missing")?;
        let blend = case["blend"]
            .as_str()
            .ok_or("mapping fixture path is missing")?;
        let loss_feature_id = case["loss_feature_id"]
            .as_str()
            .ok_or("mapping import-loss ID is missing")?;
        let feature_id = case["feature_id"]
            .as_str()
            .ok_or("mapping refusal ID is missing")?;
        let (project, report) = render_imported_allow_loss(
            &blender,
            root.path(),
            &fixtures().join(blend),
            &format!("{name}_project"),
            16,
            16,
            1,
            1,
        )?;
        assert!(
            loss_features(&report)
                .iter()
                .any(|id| id.as_str() == loss_feature_id),
            "{name}: import did not report {loss_feature_id}: {report}"
        );
        let mut render = pot();
        let output_directory = root.path().join(format!("{name}_render"));
        render
            .arg("render")
            .arg(&project)
            .args(["--engine", "path", "--format", "exr", "--out"])
            .arg(output_directory)
            .arg("--json");
        let output = render.output()?;
        assert!(
            !output.status.success(),
            "{name}: unsupported mapping unexpectedly rendered"
        );
        let response: Value = serde_json::from_slice(&output.stdout)?;
        assert_eq!(response["error"]["code"], case["error_code"], "{response}");
        assert_eq!(
            response["error"]["details"]["feature_id"], feature_id,
            "{response}"
        );

        let exported = root.path().join(format!("{name}_roundtrip.blend"));
        let export_output = export_raw(&blender, &project, "blend", &exported, true)?;
        assert!(
            export_output.status.success(),
            "{name}: lossy .blend export failed: {}",
            String::from_utf8_lossy(&export_output.stdout)
        );
        let dump_path = root.path().join(format!("{name}_mapping.json"));
        run_blender_arguments(&blender, &dump_script, Some(&exported), &[&dump_path])?;
        let dump: Value = serde_json::from_slice(&fs::read(dump_path)?)?;
        assert_eq!(
            dump["texture_mapping"]["vector_type"], case["texture_mapping"]["vector_type"],
            "{name}"
        );
        assert_floats(
            &floats(&dump["texture_mapping"]["translation"])?,
            &floats(&case["texture_mapping"]["translation"])?,
            1.0e-6,
            &format!("{name} Texture Mapping translation"),
        );
        assert_close(
            number(&dump["color_mapping"]["brightness"])?,
            number(&case["color_mapping"]["brightness"])?,
            1.0e-6,
            &format!("{name} Color Mapping brightness"),
        );
    }
    Ok(())
}
fn assert_refusal(blender: &Path, root: &Path, fixture_case: &Value) -> TestResult {
    let fixture_name = fixture_case["name"]
        .as_str()
        .ok_or("refusal case name is missing")?;
    let blend_name = fixture_case["blend"]
        .as_str()
        .ok_or("refusal blend name is missing")?;
    let feature_id = fixture_case["feature_id"]
        .as_str()
        .ok_or("refusal feature id is missing")?;
    let loss_feature_id = fixture_case["loss_feature_id"]
        .as_str()
        .ok_or("refusal import-loss ID is missing")?;
    let (project, report) = render_imported_allow_loss(
        blender,
        root,
        &fixtures().join(blend_name),
        &format!("{fixture_name}_project"),
        16,
        16,
        1,
        1,
    )?;
    assert!(
        loss_features(&report)
            .iter()
            .any(|id| id.as_str() == loss_feature_id),
        "{fixture_name}: Blender import did not report loss {loss_feature_id}: {report}"
    );
    let output_directory = root.join(format!("{fixture_name}_render"));
    let mut command = pot();
    command
        .arg("render")
        .arg(&project)
        .args(["--engine", "path", "--format", "exr", "--out"])
        .arg(output_directory)
        .arg("--json");
    let output = command.output()?;
    assert!(
        !output.status.success(),
        "{fixture_name}: unsupported graph unexpectedly rendered"
    );
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        response["error"]["code"], fixture_case["error_code"],
        "{response}"
    );
    assert_eq!(
        response["error"]["details"]["feature_id"], feature_id,
        "{response}"
    );
    Ok(())
}

#[test]
fn geometry_attribute_import_loss_tracks_reachable_output_type() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping the ShaderP1 geometry-attribute import-loss oracle; no executable Blender was found"
        );
        return Ok(());
    };
    let reference = read_reference("attribute_geometry_import_losses.json")?;
    let cases = reference["cases"]
        .as_array()
        .ok_or("geometry-attribute import-loss cases are missing")?;
    assert_eq!(cases.len(), 2);
    let root = tempdir()?;
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or("geometry-attribute case name is missing")?;
        let blend = case["blend"]
            .as_str()
            .ok_or("geometry-attribute fixture path is missing")?;
        let feature_id = case["loss_feature_id"]
            .as_str()
            .ok_or("geometry-attribute loss ID is missing")?;
        let output_socket = case["output_socket"]
            .as_str()
            .ok_or("geometry-attribute output socket is missing")?;
        let domain = case["domain"]
            .as_str()
            .ok_or("geometry-attribute domain is missing")?;
        let data_type = case["data_type"]
            .as_str()
            .ok_or("geometry-attribute data type is missing")?;
        assert_eq!(case["attribute_name"].as_str(), Some("Probe"));
        assert_eq!(case["attribute_type"].as_str(), Some("GEOMETRY"));
        assert_eq!(domain, "POINT");
        assert_eq!(data_type, "FLOAT");
        let expect_loss = case["expect_loss"]
            .as_bool()
            .ok_or("geometry-attribute loss expectation is missing")?;
        assert_eq!(
            feature_id,
            "shader.node.ShaderNodeAttribute.geometry_attribute.unsupported"
        );
        assert_eq!(output_socket, if expect_loss { "Color" } else { "Fac" });
        let imported = if expect_loss {
            render_imported_allow_loss(
                &blender,
                root.path(),
                &fixtures().join(blend),
                &format!("{name}_project"),
                16,
                16,
                1,
                1,
            )?
        } else {
            render_imported(
                &blender,
                root.path(),
                &fixtures().join(blend),
                &format!("{name}_project"),
                16,
                16,
                1,
                1,
            )?
        };
        let (_project, report) = imported;
        assert_eq!(
            report["ok"], true,
            "{name}: Blender import failed: {report}"
        );
        let report_losses = losses(&report);
        let loss = report_losses
            .iter()
            .find(|loss| loss["feature_id"].as_str() == Some(feature_id));
        if expect_loss {
            let loss = loss
                .ok_or_else(|| format!("{name}: import did not report {feature_id}: {report}"))?;
            let reason = loss["reason"]
                .as_str()
                .ok_or("geometry-attribute loss reason is missing")?;
            for detail in [
                format!("output socket `{output_socket}`"),
                format!("domain `{domain}`"),
                format!("type `{data_type}`"),
            ] {
                assert!(
                    reason.contains(&detail),
                    "{name}: loss reason omitted `{detail}`: {reason}"
                );
            }
        } else {
            assert!(
                loss.is_none(),
                "{name}: the matching Fac output must not report {feature_id}: {report}"
            );
            assert!(
                report_losses.is_empty(),
                "{name}: the matching Fac case must remain loss-free: {report}"
            );
        }
    }
    Ok(())
}

#[test]
fn unsupported_attribute_volume_and_sky_graphs_report_precise_refusals() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping the ShaderP1 refusal oracle; no executable Blender was found");
        return Ok(());
    };
    let reference = read_reference("refusals.json")?;
    let cases = reference["cases"]
        .as_array()
        .ok_or("refusal cases are missing")?;
    assert_eq!(cases.len(), 4);
    let root = tempdir()?;
    for case in cases {
        assert_refusal(&blender, root.path(), case)?;
    }
    Ok(())
}
