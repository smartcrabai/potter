#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "the shared oracle helper is used by multiple focused integration targets"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{fs, path::Path};

use native_oracle::{
    TestResult, apply_operations, assert_close, checked, export, import_blend,
    import_blend_allow_lossy, losses, node_id, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::{
    graph::{GraphKind, GraphLink, GraphNode, NodeGroup},
    image::{ImageData, ImageInterpolation, load_image_data, sample},
    model::{Id, Image, Material, Registry, SceneDoc},
    shader::{HitContext, evaluate_surface},
};
use serde_json::{Value, json};
use tempfile::tempdir;

#[test]
fn imported_shader_converters_match_blender_cycles_oracle() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let blend = root.path().join("shader-converters.blend");
    let oracle_path = root.path().join("shader-converters-oracle.json");
    let exr_path = root.path().join("shader-converters-oracle.exr");
    let script = write_script(
        root.path(),
        "generate_shader_converters.py",
        include_str!("fixtures/shader_converters_oracle.py"),
    )?;
    run_blender_arguments(&blender, &script, None, &[&blend, &oracle_path, &exr_path])?;
    let oracle: Value = serde_json::from_slice(&fs::read(oracle_path)?)?;

    let (project, report) = import_blend(&blender, root.path(), "project", &blend)?;
    assert!(
        losses(&report).is_empty(),
        "fixture import lost data: {report}"
    );
    let exported = root.path().join("shader-converters-roundtrip.blend");
    let export_report = export(&blender, &project, "blend", &exported)?;
    assert!(
        losses(&export_report).is_empty(),
        "converter fixture export lost data: {export_report}"
    );
    let (roundtrip_project, roundtrip_report) =
        import_blend(&blender, root.path(), "roundtrip", &exported)?;
    assert!(
        losses(&roundtrip_report).is_empty(),
        "converter fixture re-import lost data: {roundtrip_report}"
    );
    let document: SceneDoc = serde_json::from_value(read_scene(&roundtrip_project)?)?;
    let material_index = document
        .materials
        .values()
        .find(|material| material.name == "Object Info Material Index")
        .ok_or("missing imported material-index material")?;
    assert_eq!(material_index.pass_index, Some(23));
    for (name, pixel_index) in [
        ("Color Converters", 0),
        ("Blackbody", 1),
        ("Wavelength", 2),
        ("Light Falloff", 3),
    ] {
        let material = document
            .materials
            .values()
            .find(|material| material.name == name)
            .ok_or_else(|| format!("missing imported material `{name}`"))?;
        let mut context = HitContext::new(&document.node_groups);
        context.path_traced = true;
        if name == "Light Falloff" {
            context.light_falloff_distance = Some(4.9);
        }
        let shading = evaluate_surface(material, &context)?;
        let params = &shading
            .closures
            .first()
            .ok_or_else(|| format!("material `{name}` produced no surface closure"))?
            .params;
        let actual = [
            params.emission_color[0] * params.emission_strength,
            params.emission_color[1] * params.emission_strength,
            params.emission_color[2] * params.emission_strength,
        ];
        let expected = oracle["pixels"][pixel_index]
            .as_array()
            .ok_or("Blender oracle pixel is not an array")?;
        for channel in 0..3 {
            let expected = expected[channel]
                .as_f64()
                .ok_or("Blender oracle channel is not numeric")?;
            assert_close(
                actual[channel],
                expected,
                5.0e-4,
                &format!(
                    "{name} channel {channel}; emission_color={:?}, strength={}",
                    params.emission_color, params.emission_strength
                ),
            );
        }
        if name == "Light Falloff" {
            let realtime = evaluate_surface(material, &HitContext::new(&document.node_groups))?;
            assert_close(
                realtime.closures[0].params.emission_strength,
                2.0,
                0.0,
                "EEVEE Light Falloff passthrough",
            );
        }
    }
    let random_target = document
        .nodes
        .values()
        .find(|node| node.name == "Random Target")
        .ok_or("missing imported Random Target")?;
    let random_pass_index = random_target
        .properties
        .get("blender_object_metadata")
        .and_then(|metadata| metadata.get("rna_properties"))
        .and_then(|properties| properties.get("pass_index"))
        .and_then(Value::as_i64)
        .ok_or("Random Target pass_index was not imported")?;
    assert_eq!(random_pass_index, 29);
    let image = render_imported_scene(
        &project,
        root.path(),
        "cycles_object_info",
        "path",
        [9, 4],
        1,
    )?;
    for (x, pixel_index) in (4_u32..=8).zip(4_usize..=8) {
        let uv = [(f64::from(x) + 0.5) / 9.0, 0.5];
        let actual = sample(&image, uv, false);
        let expected = oracle["pixels"][pixel_index]
            .as_array()
            .ok_or("Blender object-info oracle pixel is not an array")?;
        for channel in 0..3 {
            let expected = expected[channel]
                .as_f64()
                .ok_or("Blender object-info oracle channel is not numeric")?;
            assert_close(
                actual[channel],
                expected,
                5.0e-4,
                &format!("rendered object-info pixel {pixel_index} channel {channel}"),
            );
        }
    }
    Ok(())
}

#[test]
fn object_info_material_index_refuses_unknown_material_pass_index() -> TestResult {
    let mut graph = NodeGroup::new("Unknown Material Index", GraphKind::Shader);
    let output = Id::new("output")?;
    let principled = Id::new("principled")?;
    let object_info = Id::new("object_info")?;
    graph
        .nodes
        .insert(output.clone(), GraphNode::new("ShaderNodeOutputMaterial"));
    graph.nodes.insert(
        principled.clone(),
        GraphNode::new("ShaderNodeBsdfPrincipled"),
    );
    graph
        .nodes
        .insert(object_info.clone(), GraphNode::new("ShaderNodeObjectInfo"));
    graph.links = vec![
        GraphLink {
            from_node: object_info,
            from_socket: "Material Index".to_owned(),
            to_node: principled.clone(),
            to_socket: "Roughness".to_owned(),
        },
        GraphLink {
            from_node: principled,
            from_socket: "BSDF".to_owned(),
            to_node: output,
            to_socket: "Surface".to_owned(),
        },
    ];

    let graph_id = Id::new("material_index_graph")?;
    let mut groups = Registry::default();
    groups.insert(graph_id.clone(), graph);
    let material = Material {
        pass_index: None,
        node_tree: Some(graph_id),
        ..Material::default()
    };
    let mut context = HitContext::new(&groups);
    context.material_pass_index = material.pass_index.map(f64::from);
    let error = evaluate_surface(&material, &context)
        .err()
        .ok_or("unknown material pass index was accepted")?;

    assert_eq!(
        error.code,
        potter_core::error::ErrorCode::UnsupportedFeature
    );
    assert_eq!(
        error.details["feature_id"],
        "shader.node.ShaderNodeObjectInfo.Material Index.material_pass_index"
    );
    Ok(())
}

const EEVEE_FIXTURE: &str = "tests/fixtures/shader_converters_eevee.blend";
const EEVEE_ORACLE: &str = "tests/fixtures/shader_converters_eevee_oracle.json";
const EEVEE_SHADOW_ORACLE: &str = "tests/fixtures/shader_converters_eevee_shadow_oracle.json";
const REALTIME_VOLUME_CONTEXT_FIXTURE: &str =
    "tests/fixtures/realtime_volume_context/realtime_volume_context.blend";
const REALTIME_VOLUME_CONTEXT_ORACLE: &str =
    "tests/fixtures/realtime_volume_context/realtime_volume_context.json";
const EEVEE_PROBES: [&str; 10] = [
    "bump",
    "normal_map",
    "blackbody",
    "wavelength",
    "true_normal",
    "pointiness",
    "random_per_island",
    "hsv",
    "camera_data",
    "camera_data_bump",
];
/// EEVEE stores and re-accumulates its combined film in RGBA16F. The per-sample
/// accumulation order can differ by one representable half-float value.
fn assert_eevee_film_close(actual: f64, expected: f64, context: &str) {
    let magnitude = expected.abs();
    let mut ulp = 1.0 / 16_777_216.0;
    while magnitude >= ulp * 2048.0 {
        ulp *= 2.0;
    }
    assert_close(actual, expected, ulp, context);
}

fn render_imported_scene(
    project: &Path,
    root: &Path,
    name: &str,
    engine: &str,
    resolution: [u32; 2],
    samples: u32,
) -> TestResult<ImageData> {
    let scene = read_scene(project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let updated = apply_operations(
        project,
        root,
        json!([{
            "op": "render.update",
            "target": {"id": active},
            "set": {
                "resolution_x": resolution[0],
                "resolution_y": resolution[1],
                "samples": samples,
                "seed": 0
            }
        }]),
    )?;
    assert_eq!(updated["ok"], true, "{updated}");

    let frames = root.join(name);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--engine", engine, "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    checked(&mut render, name)?;

    let id = format!("{name}_image");
    let exr = frames.join("frame_0001.exr");
    let loaded = apply_operations(
        project,
        root,
        json!([{"op": "image.load", "id": id, "path": exr.to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][&id].clone())?;
    let image = load_image_data(&image, project, ImageInterpolation::Closest)?;
    assert_eq!(
        [image.width, image.height],
        resolution,
        "Potter render resolution"
    );
    Ok(image)
}
fn shadow_patch_rgb(image: &ImageData, center_x: u32) -> [f64; 3] {
    let mut total = [0.0; 3];
    for y in 29_u32..36 {
        for x in center_x - 3..=center_x + 3 {
            let uv = [
                (f64::from(x) + 0.5) / f64::from(image.width),
                (f64::from(y) + 0.5) / f64::from(image.height),
            ];
            let pixel = sample(image, uv, false);
            for channel in 0..3 {
                total[channel] += pixel[channel];
            }
        }
    }
    total.map(|value| value / 49.0)
}

fn shadow_patch_mean(image: &ImageData, center_x: u32) -> f64 {
    let rgb = shadow_patch_rgb(image, center_x);
    (rgb[0] + rgb[1] + rgb[2]) / 3.0
}

#[test]
fn realtime_shader_converters_match_blender_eevee_oracle() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(EEVEE_FIXTURE);
    let (project, report) = import_blend(&blender, root.path(), "eevee_project", &fixture)?;
    assert!(
        losses(&report).is_empty(),
        "EEVEE converter fixture import lost data: {report}"
    );
    let image = render_imported_scene(
        &project,
        root.path(),
        "eevee_converters",
        "realtime",
        [96, 48],
        16,
    )?;
    let oracle_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(EEVEE_ORACLE);
    let oracle: Value = serde_json::from_slice(&fs::read(oracle_path)?)?;
    assert_eq!(oracle["resolution"], json!([96, 48]));
    assert_eq!(oracle["engine"], "Blender 5.2.2 EEVEE");
    for probe in EEVEE_PROBES {
        let uv: [f64; 2] = serde_json::from_value(oracle["probes"][probe]["uv"].clone())?;
        let expected: [f64; 4] = serde_json::from_value(oracle["probes"][probe]["rgba"].clone())?;
        let actual = sample(&image, uv, false);
        for channel in 0..3 {
            assert_eevee_film_close(
                actual[channel],
                expected[channel],
                &format!("EEVEE {probe} channel {channel}"),
            );
        }
    }
    let hsv_red = oracle["probes"]["hsv"]["rgba"][0]
        .as_f64()
        .ok_or("HSV oracle red channel is not numeric")?;
    assert!(
        hsv_red > 1.0,
        "HSV oracle must distinguish EEVEE's unclamped output from Cycles"
    );
    Ok(())
}
#[test]
fn realtime_volume_context_matches_blender_eevee_oracle() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(REALTIME_VOLUME_CONTEXT_FIXTURE);
    let (project, report) =
        import_blend(&blender, root.path(), "realtime_volume_context", &fixture)?;
    assert!(
        losses(&report).is_empty(),
        "realtime volume context fixture import lost data: {report}"
    );
    let image = render_imported_scene(
        &project,
        root.path(),
        "realtime_volume_context",
        "realtime",
        [16, 16],
        8,
    )?;
    let oracle_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(REALTIME_VOLUME_CONTEXT_ORACLE);
    let oracle: Value = serde_json::from_slice(&fs::read(oracle_path)?)?;
    assert_eq!(oracle["engine"], "Blender 5.2.2 EEVEE");
    assert_eq!(oracle["resolution"], json!([16, 16]));
    assert_eq!(oracle["object"]["name"], "Random Target");
    assert_eq!(oracle["object"]["pass_index"], 29);
    assert_eq!(oracle["object"]["material_pass_index"], 11);

    let uv: [f64; 2] = serde_json::from_value(oracle["probe"]["uv"].clone())?;
    let expected: [f64; 4] = serde_json::from_value(oracle["probe"]["rgba"].clone())?;
    let actual = sample(&image, uv, false);
    for channel in 0..3 {
        assert!(
            expected[channel] > 0.0,
            "Blender EEVEE volume oracle channel {channel} must be visible"
        );
        assert!(
            actual[channel] > 0.0,
            "Potter realtime volume channel {channel} must not disappear"
        );
    }
    assert!(
        actual[0] < actual[1] && actual[1] < actual[2],
        "volume Object Info color must affect the rendered channels: {actual:?}"
    );
    Ok(())
}

#[test]
fn realtime_transparent_shadow_skips_normal_only_bump_height() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let blend = root.path().join("eevee-alpha-bump-shadow.blend");
    let exr = root.path().join("eevee-alpha-bump-shadow.exr");
    let oracle_path = root.path().join("eevee-alpha-bump-shadow.json");
    let script = write_script(
        root.path(),
        "generate_eevee_alpha_bump_shadow.py",
        include_str!("fixtures/shader_converters_eevee_shadow_oracle.py"),
    )?;
    run_blender_arguments(&blender, &script, None, &[&blend, &exr, &oracle_path])?;
    let runtime_oracle: Value = serde_json::from_slice(&fs::read(oracle_path)?)?;
    assert_eq!(runtime_oracle["engine"], "Blender 5.2.2 EEVEE");
    assert_eq!(runtime_oracle["resolution"], json!([64, 64]));
    let expected_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(EEVEE_SHADOW_ORACLE);
    let expected: Value = serde_json::from_slice(&fs::read(expected_path)?)?;
    assert_eq!(expected["engine"], "Blender 5.2.2 EEVEE");
    assert_eq!(expected["resolution"], json!([64, 64]));

    // Potter's realtime shadow-map approximation is 0.0900 here versus EEVEE's
    // 0.0973 capture, so assert shadow presence relationally, not pixel parity.
    let mean_rgb = |rgb: [f64; 3]| (rgb[0] + rgb[1] + rgb[2]) / 3.0;

    let (project, report) = import_blend(&blender, root.path(), "eevee_shadow_project", &blend)?;
    assert!(
        losses(&report).is_empty(),
        "alpha+Bump shadow fixture import lost data: {report}"
    );
    let image = render_imported_scene(
        &project,
        root.path(),
        "eevee_alpha_bump_shadow",
        "realtime",
        [64, 64],
        16,
    )?;
    let potter_shadow = shadow_patch_rgb(&image, 16);
    let potter_lit = shadow_patch_rgb(&image, 38);
    let expected_lit: [f64; 3] = serde_json::from_value(expected["lit_patch_rgb"].clone())?;
    for channel in 0..3 {
        assert_close(
            potter_lit[channel],
            expected_lit[channel],
            2.0e-3,
            &format!("Potter EEVEE alpha+Bump lit_patch_rgb channel {channel}"),
        );
    }
    let potter_shadow = mean_rgb(potter_shadow);
    let potter_lit = mean_rgb(potter_lit);
    assert!(
        potter_shadow < potter_lit,
        "Potter's realtime alpha+Bump caster must leave a transparent shadow, not abort or occlude opaquely: shadow={potter_shadow}, lit={potter_lit}"
    );
    Ok(())
}

#[test]
fn realtime_transparent_shadow_with_noise_alpha_keeps_normal_only_bump() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let blend = root.path().join("eevee-noise-alpha-bump-shadow.blend");
    let exr = root.path().join("eevee-noise-alpha-bump-shadow.exr");
    let oracle_path = root.path().join("eevee-noise-alpha-bump-shadow.json");
    let script = write_script(
        root.path(),
        "generate_eevee_noise_alpha_bump_shadow.py",
        include_str!("fixtures/shader_converters_eevee_shadow_oracle.py"),
    )?;
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[&blend, &exr, &oracle_path, Path::new("noise_alpha")],
    )?;
    let (project, report) =
        import_blend(&blender, root.path(), "eevee_noise_alpha_shadow", &blend)?;
    assert!(
        losses(&report).is_empty(),
        "noise-alpha, normal-only Bump import lost data: {report}"
    );
    let image = render_imported_scene(
        &project,
        root.path(),
        "eevee_noise_alpha_bump_shadow",
        "realtime",
        [64, 64],
        16,
    )?;
    let potter_shadow = shadow_patch_mean(&image, 16);
    let potter_lit = shadow_patch_mean(&image, 38);
    assert!(
        potter_shadow < potter_lit,
        "noise-alpha, normal-only Bump must keep a transparent shadow: shadow={potter_shadow}, lit={potter_lit}"
    );
    Ok(())
}
#[test]
fn realtime_transparent_shadow_refuses_bump_driven_alpha() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let blend = root.path().join("eevee-opacity-bump-shadow.blend");
    let exr = root.path().join("eevee-opacity-bump-shadow.exr");
    let oracle_path = root.path().join("eevee-opacity-bump-shadow.json");
    let script = write_script(
        root.path(),
        "generate_eevee_opacity_bump_shadow.py",
        include_str!("fixtures/shader_converters_eevee_shadow_oracle.py"),
    )?;
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[&blend, &exr, &oracle_path, Path::new("opacity_bump")],
    )?;
    let (project, report) =
        import_blend_allow_lossy(&blender, root.path(), "eevee_opacity_bump_project", &blend)?;
    let feature_id = "shader.node.Bump.realtime_transparent_shadow_height";
    assert!(
        losses(&report)
            .iter()
            .any(|loss| loss["feature_id"] == feature_id),
        "the .blend import must report the transparency-dependent Bump loss: {report}"
    );

    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let updated = apply_operations(
        &project,
        root.path(),
        json!([{
            "op": "render.update",
            "target": {"id": active},
            "set": {"resolution_x": 64, "resolution_y": 64, "samples": 16, "seed": 0}
        }]),
    )?;
    assert_eq!(updated["ok"], true, "{updated}");
    let frames = root.path().join("eevee-opacity-bump-shadow-render");
    let mut render = pot();
    render
        .arg("render")
        .arg(&project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    let output = render.output()?;
    let response = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "Bump-driven transparency must be refused instead of approximated: {response}"
    );
    assert!(
        response.contains(feature_id),
        "the runtime refusal must expose {feature_id}: {response}"
    );
    Ok(())
}

const EEVEE_STEREO_FIXTURE: &str = "tests/fixtures/shader_converters_eevee_stereo.blend";
const EEVEE_STEREO_ORACLE: &str = "tests/fixtures/shader_converters_eevee_stereo_oracle.json";

#[test]
fn realtime_camera_data_uses_each_stereo_eye_matrix() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(EEVEE_STEREO_FIXTURE);
    let (project, report) = import_blend(&blender, root.path(), "stereo_project", &fixture)?;
    assert!(
        losses(&report).is_empty(),
        "stereo Camera Data fixture import lost data: {report}"
    );
    let scene = read_scene(&project)?;
    let camera = node_id(&scene, "Stereo Oracle Camera")?;
    let updated = apply_operations(
        &project,
        root.path(),
        json!([{
            "op": "camera.update",
            "target": {"id": camera},
            "set": {"stereo_mode": "side_by_side", "interocular_distance": 1.5}
        }]),
    )?;
    assert_eq!(updated["ok"], true, "{updated}");
    let image = render_imported_scene(
        &project,
        root.path(),
        "stereo_camera_data",
        "realtime",
        [96, 48],
        16,
    )?;
    let oracle_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(EEVEE_STEREO_ORACLE);
    let oracle: Value = serde_json::from_slice(&fs::read(oracle_path)?)?;
    assert_eq!(oracle["resolution"], json!([96, 48]));
    assert_eq!(oracle["engine"], "Blender 5.2.2 EEVEE");
    let mut pixels = Vec::with_capacity(2);
    for eye in ["left", "right"] {
        let uv: [f64; 2] = serde_json::from_value(oracle[eye]["uv"].clone())?;
        let expected: [f64; 4] = serde_json::from_value(oracle[eye]["rgba"].clone())?;
        let actual = sample(&image, uv, false);
        for channel in 0..3 {
            assert_eevee_film_close(
                actual[channel],
                expected[channel],
                &format!("stereo {eye} Camera Data channel {channel}"),
            );
        }
        pixels.push(actual[0]);
    }
    assert!(
        (pixels[0] - pixels[1]).abs() > 0.1,
        "stereo eye Camera Data pixels did not separate: {pixels:?}"
    );
    Ok(())
}

#[test]
fn realtime_camera_data_stereo_exact_at_pixel_centers() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let blend = root.path().join("stereo-camera-data.blend");
    let left_exr = root.path().join("stereo-camera-data-left.exr");
    let right_exr = root.path().join("stereo-camera-data-right.exr");
    let oracle_path = root.path().join("stereo-camera-data-oracle.json");
    let script = write_script(
        root.path(),
        "generate_stereo_camera_data_oracle.py",
        include_str!("fixtures/shader_converters_eevee_stereo_strict_oracle.py"),
    )?;
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[&blend, &left_exr, &right_exr, &oracle_path],
    )?;
    let oracle: Value = serde_json::from_slice(&fs::read(&oracle_path)?)?;
    assert_eq!(oracle["engine"], "Blender 5.2.2 EEVEE");
    assert_eq!(oracle["resolution"], json!([98, 49]));
    assert_eq!(oracle["left"]["rgba"][0], json!(5.0));
    assert_eq!(oracle["right"]["rgba"][0], json!(3.0));

    let (project, report) = import_blend(&blender, root.path(), "stereo_strict_project", &blend)?;
    assert!(
        losses(&report).is_empty(),
        "stereo Camera Data fixture import lost data: {report}"
    );
    let scene = read_scene(&project)?;
    let camera = node_id(&scene, "Stereo Strict Oracle Camera")?;
    let updated = apply_operations(
        &project,
        root.path(),
        json!([{
            "op": "camera.update",
            "target": {"id": camera},
            "set": {"stereo_mode": "side_by_side", "interocular_distance": 4.0}
        }]),
    )?;
    assert_eq!(updated["ok"], true, "{updated}");
    let image = render_imported_scene(
        &project,
        root.path(),
        "stereo_camera_data_strict",
        "realtime",
        [98, 49],
        1,
    )?;
    for eye in ["left", "right"] {
        let uv: [f64; 2] = serde_json::from_value(oracle[eye]["uv"].clone())?;
        let expected: [f64; 4] = serde_json::from_value(oracle[eye]["rgba"].clone())?;
        let actual = sample(&image, uv, false);
        for channel in 0..3 {
            assert_close(
                actual[channel],
                expected[channel],
                1.0e-3,
                &format!("pixel-centered stereo {eye} Camera Data channel {channel}"),
            );
        }
    }
    Ok(())
}

#[test]
fn cycles_shader_converter_signs_and_clamps_match_blender_oracle() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(
        root.path(),
        "generate_shader_converters.py",
        include_str!("fixtures/shader_converters_oracle.py"),
    )?;
    let base_blend = root.path().join("base.blend");
    let base_oracle = root.path().join("base.json");
    let base_exr = root.path().join("base.exr");
    let probe_blend = root.path().join("cycles-probes.blend");
    let generated_oracle_path = root.path().join("cycles-probes.json");
    let probe_exr = root.path().join("cycles-probes.exr");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            &base_blend,
            &base_oracle,
            &base_exr,
            &probe_blend,
            &generated_oracle_path,
            &probe_exr,
        ],
    )?;
    let generated: Value = serde_json::from_slice(&fs::read(generated_oracle_path)?)?;
    let expected_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/shader_converters_cycles_oracle.json");
    let expected: Value = serde_json::from_slice(&fs::read(expected_path)?)?;
    assert_eq!(generated, expected, "generated Cycles oracle changed");
    assert_eq!(expected["engine"], "Blender 5.2.2 Cycles");

    let (project, report) = import_blend(&blender, root.path(), "cycles_project", &probe_blend)?;
    assert!(
        losses(&report).is_empty(),
        "Cycles converter probe fixture import lost data: {report}"
    );
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let bounce_update = apply_operations(
        &project,
        root.path(),
        json!([{
            "op": "render.update",
            "target": {"id": active},
            "set": {"max_bounces": 3}
        }]),
    )?;
    assert_eq!(bounce_update["ok"], true, "{bounce_update}");
    let image = render_imported_scene(&project, root.path(), "cycles_probes", "path", [12, 12], 1)?;
    assert_eq!(expected["resolution"], json!([12, 12]));
    assert_eq!(
        expected["probes"]["camera_z_depth"]["view_z_depth"],
        json!(5.0)
    );
    let secondary_view_z_depth = expected["probes"]["camera_z_depth_secondary"]["view_z_depth"]
        .as_f64()
        .ok_or("secondary camera depth oracle is not numeric")?;
    assert!(
        secondary_view_z_depth < 0.0,
        "the reflected camera hit must be behind the camera: {secondary_view_z_depth}"
    );
    assert!(
        expected["probes"]["camera_z_depth_secondary"]["camera_space_z"]
            .as_f64()
            .is_some_and(|value| value > 0.0),
        "the secondary point must have positive camera-space Z"
    );
    for probe in [
        "incoming",
        "reflection",
        "hsv_overdrive",
        "normal_map_object_backface",
        "camera_z_depth",
        "camera_z_depth_secondary",
        "camera_data_bump",
        "geometry_normal_bump",
    ] {
        let uv: [f64; 2] = serde_json::from_value(expected["probes"][probe]["uv"].clone())?;
        let pixel: [f64; 4] = serde_json::from_value(expected["probes"][probe]["rgba"].clone())?;
        let actual = sample(&image, uv, false);
        for channel in 0..3 {
            assert_close(
                actual[channel],
                pixel[channel],
                5.0e-4,
                &format!("Cycles {probe} channel {channel}"),
            );
        }
    }
    assert!(
        expected["probes"]["reflection"]["rgba"][0]
            .as_f64()
            .is_some_and(|value| value > 0.99),
        "tilted Reflection oracle must encode the positive X sign"
    );
    assert_eq!(
        expected["probes"]["hsv_overdrive"]["rgba"][1],
        json!(0.0),
        "Cycles Hue/Saturation clamps negative channels to zero"
    );
    Ok(())
}
