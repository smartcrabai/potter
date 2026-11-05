//! Physical Cycles Hair collection-instance provenance and path-hit/shadow behavior.
//!
//! Blender's render depsgraph capture supplies two instanced Curves Objects sharing one Hair Curves
//! datablock. The CLI import is checked against each physical member's instance key/context, then
//! Potter's path render is checked at fixed hair-hit and receiver-shadow patches from the pinned
//! Blender 5.2.2 capture.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{collections::BTreeMap, fs, path::Path, process::Command};

use blender_checked::blender_executable;
use native_oracle::{
    FLOAT_TOLERANCE, TestResult, apply_operations, checked, import_blend, losses, node_id, pot,
    read_scene,
};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde::Deserialize;
use serde_json::{Value, json};
use tempfile::tempdir;

const FIXTURE_DIR: &str = "tests/fixtures/native_oracle/hair_instance_context18";
const RESOLUTION: [u32; 2] = [96, 96];
const SAMPLES: u32 = 64;
const SEED: u32 = 29;
const CAPTURE: &str =
    include_str!("fixtures/native_oracle/hair_instance_context18/hair_instance_context18.json");

#[derive(Deserialize)]
struct Capture {
    blender_version: String,
    blender_build_hash: String,
    engine: String,
    device: String,
    resolution: [u32; 2],
    resolution_percentage: u32,
    samples: u32,
    seed: u32,
    denoising: bool,
    max_bounces: u32,
    pixel_filter: String,
    filter_width: f64,
    cycles_curves: CyclesCurves,
    view_transform: String,
    view_look: String,
    exposure: f64,
    gamma: f64,
    linear_exr: String,
    protocol: String,
    scene: CaptureScene,
    probes: Vec<Probe>,
}

#[derive(Deserialize)]
struct CyclesCurves {
    shape: String,
    subdivisions: u32,
}

#[derive(Deserialize)]
struct CaptureScene {
    shared_hair_data: SharedHairData,
    members: Vec<CaptureMember>,
}

#[derive(Deserialize)]
struct SharedHairData {
    name: String,
    spline_type: String,
    curve_count: usize,
    key_positions: Vec<[f64; 3]>,
    radius: f64,
    material: String,
    closure: String,
    closure_component: String,
    object_info_output_to_hair_color: String,
    pass_index_range: [i32; 2],
    object_info_color_used: bool,
    object_info_random_used: bool,
    object_info_random_to_hair_roughness_u: bool,
}

#[derive(Deserialize)]
struct CaptureMember {
    name: String,
    parent: String,
    persistent_id: [i32; 8],
    is_instance: bool,
    use_particle_hair: bool,
    object_to_world: [[f64; 4]; 4],
    object_color: [f64; 4],
    source_object_color: [f64; 4],
    object_pass_index: i32,
    object_random_id: i32,
    slot_0_material: String,
    ray_visibility: RayVisibility,
}

#[derive(Deserialize)]
struct RayVisibility {
    camera: bool,
    diffuse: bool,
    glossy: bool,
    transmission: bool,
    volume_scatter: bool,
    shadow: bool,
}

#[derive(Deserialize)]
struct Probe {
    name: String,
    pixel: [u32; 2],
    radius: u32,
    linear_rgba: [f64; 4],
}

#[test]
fn physical_hair_collection_instances_keep_hit_context_and_shadow_visibility() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping physical Hair instance parity; no executable Blender was found");
        return Ok(());
    };
    let capture: Capture = serde_json::from_str(CAPTURE)?;
    check_capture(&capture)?;

    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR);
    let source_blend = fixture_dir.join("hair_instance_context18.blend");
    let (project, imported) = import_blend(
        &blender,
        root.path(),
        "hair_instance_context18",
        &source_blend,
    )?;
    assert!(
        losses(&imported).is_empty(),
        "physical Hair instance import reported losses: {imported}"
    );
    let scene = read_scene(&project)?;
    check_imported_members(&scene, &capture)?;
    configure_render(&project, root.path())?;

    let source = load_image_file(
        &project,
        root.path(),
        "blender_hair_instance_context18",
        &fixture_dir.join(&capture.linear_exr),
    )?;
    let rendered = render_frame(&project, root.path())?;
    assert_eq!([source.width, source.height], RESOLUTION);
    assert_eq!([rendered.width, rendered.height], RESOLUTION);

    let references = capture
        .probes
        .iter()
        .map(|probe| (probe.name.as_str(), probe))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(references.len(), 5);
    for (name, probe) in &references {
        let source_mean = patch_mean(&source, probe);
        for (channel, (actual, expected)) in source_mean.iter().zip(probe.linear_rgba).enumerate() {
            assert_close(
                *actual,
                expected,
                FLOAT_TOLERANCE,
                &format!("Blender capture {name} channel {channel}"),
            );
        }
    }
    let source_red_hit = patch_mean(&source, references["red_hair_hit"]);
    let source_blue_hit = patch_mean(&source, references["blue_hair_hit"]);
    let source_red_shadow = patch_mean(&source, references["red_shadow_visible"]);
    let source_blue_shadow = patch_mean(&source, references["blue_shadow_masked"]);
    let source_lit = patch_mean(&source, references["lit_receiver"]);
    assert!(source_red_hit[0] > source_red_hit[2] * 1.5);
    assert!(source_blue_hit[2] > source_blue_hit[0] * 1.5);
    assert!(source_red_shadow[0] < source_lit[0] * 0.5);
    assert!(source_blue_shadow[0] > source_lit[0] * 0.75);

    let red_hit = patch_mean(&rendered, references["red_hair_hit"]);
    let blue_hit = patch_mean(&rendered, references["blue_hair_hit"]);
    let red_shadow = patch_mean(&rendered, references["red_shadow_visible"]);
    let blue_shadow = patch_mean(&rendered, references["blue_shadow_masked"]);
    let lit = patch_mean(&rendered, references["lit_receiver"]);
    assert!(
        red_hit[0] > red_hit[2] * 1.5,
        "fixed red-member Hair hit must use its Object Info pass-index context: {red_hit:?}"
    );
    assert!(
        blue_hit[2] > blue_hit[0] * 1.5,
        "fixed blue-member Hair hit must use its Object Info pass-index context: {blue_hit:?}"
    );
    assert!(
        red_shadow[0] < lit[0] * 0.5,
        "the fixed red receiver patch must be shadowed by the shadow-visible Hair member: shadow={red_shadow:?}, lit={lit:?}"
    );
    assert!(
        blue_shadow[0] > lit[0] * 0.75,
        "the fixed blue receiver patch must remain lit when its Hair member disables shadow rays: shadow={blue_shadow:?}, lit={lit:?}"
    );

    Ok(())
}
#[test]
fn cycles_hair_provenance_import_does_not_write_compositor_output() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Cycles Hair import side-effect regression; Blender is unavailable");
        return Ok(());
    };
    let capture: Capture = serde_json::from_str(CAPTURE)?;
    check_capture(&capture)?;
    let root = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR);
    let source_blend = fixture_dir.join("hair_instance_context18.blend");
    let variant_blend = root.path().join("hair_instance_file_output.blend");
    let file_output_dir = root.path().join("file_output");
    fs::create_dir(&file_output_dir)?;
    let python = format!(
        r#"import bpy
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.use_compositing = True
scene.render.use_sequencer = True
tree = bpy.data.node_groups.new("Import File Output Side Effect", "CompositorNodeTree")
tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
scene.compositing_node_group = tree
layers = tree.nodes.new("CompositorNodeRLayers")
file_output = tree.nodes.new("CompositorNodeOutputFile")
file_output.directory = {directory}
file_output.file_name = "bridge_import_side_effect_"
file_output.file_output_items.new("RGBA", "Image")
file_output.format.file_format = "OPEN_EXR_MULTILAYER"
group_output = tree.nodes.new("NodeGroupOutput")
tree.links.new(layers.outputs["Image"], file_output.inputs["Image"])
tree.links.new(layers.outputs["Image"], group_output.inputs["Image"])
bpy.ops.wm.save_as_mainfile(filepath={variant})
"#,
        directory = serde_json::to_string(&file_output_dir.to_string_lossy().to_string())?,
        variant = serde_json::to_string(&variant_blend.to_string_lossy().to_string())?,
    );
    let script = root.path().join("add_file_output.py");
    fs::write(&script, python)?;
    let mut prepare = Command::new(&blender);
    prepare
        .args(["--background", "--factory-startup", "--disable-autoexec"])
        .arg(&source_blend)
        .args(["--python-exit-code", "3", "--python"])
        .arg(&script);
    checked(&mut prepare, "prepare Blender File Output import fixture")?;
    let source_bytes = fs::read(&variant_blend)?;

    let (project, imported) = import_blend(
        &blender,
        root.path(),
        "hair_instance_file_output",
        &variant_blend,
    )?;
    assert!(
        fs::read(&variant_blend)? == source_bytes,
        "Cycles provenance import modified its source .blend"
    );
    assert!(
        losses(&imported).is_empty(),
        "File Output Hair import reported losses: {imported}"
    );
    assert!(
        fs::read_dir(&file_output_dir)?.next().is_none(),
        "Cycles provenance import wrote through the source compositor File Output node"
    );
    let document = read_scene(&project)?;
    check_imported_members(&document, &capture)?;
    let scene_id = document["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?;
    let scene = &document["scenes"][scene_id];
    assert_eq!(scene["render"]["engine"], "realtime");
    assert_eq!(scene["render"]["use_sequencer"], true);
    assert_eq!(scene["use_compositing"], true);
    assert!(
        scene["compositor"].is_string(),
        "the imported compositor graph was not preserved"
    );
    Ok(())
}

fn check_capture(capture: &Capture) -> TestResult {
    assert_eq!(capture.blender_version, "5.2.2 LTS");
    assert_eq!(capture.blender_build_hash, "d13f752e3b9c");
    assert_eq!(capture.engine, "CYCLES");
    assert_eq!(capture.device, "CPU");
    assert_eq!(capture.resolution, RESOLUTION);
    assert_eq!(capture.resolution_percentage, 100);
    assert_eq!(capture.samples, SAMPLES);
    assert_eq!(capture.seed, SEED);
    assert!(!capture.denoising);
    assert_eq!(capture.max_bounces, 2);
    assert_eq!(capture.pixel_filter, "BOX");
    assert_eq!(capture.filter_width, 1.0);
    assert_eq!(capture.view_transform, "Standard");
    assert_eq!(capture.view_look, "None");
    assert_eq!(capture.exposure, 0.0);
    assert_eq!(capture.gamma, 1.0);
    assert_eq!(capture.linear_exr, "hair_instance_context18.exr");
    assert!(capture.protocol.contains("fixed seed/sample count"));

    let data = &capture.scene.shared_hair_data;
    assert_eq!(data.name, "Shared Physical Hair Curves");
    assert_eq!(data.spline_type, "CATMULL_ROM");
    assert_eq!(data.curve_count, 3);
    assert_eq!(data.key_positions.len(), 12);
    assert_eq!(data.radius, 0.075);
    assert_eq!(data.material, "Object Info Colored Cycles Hair");
    assert_eq!(data.closure, "ShaderNodeBsdfHair");
    assert_eq!(data.closure_component, "Reflection");
    assert_eq!(
        data.object_info_output_to_hair_color,
        "Object Index -> Map Range -> constant Color Ramp"
    );
    assert_eq!(data.pass_index_range, [11, 22]);
    assert!(!data.object_info_color_used);
    assert!(data.object_info_random_used);
    assert!(data.object_info_random_to_hair_roughness_u);
    assert_eq!(capture.scene.members.len(), 2);
    assert_eq!(capture.scene.members[0].name, "HairMemberRed");
    assert_eq!(capture.scene.members[1].name, "HairMemberBlue");
    assert_floats_close(
        &json!(capture.scene.members[0].source_object_color),
        &[0.95, 0.08, 0.04, 1.0],
        "red source Object color",
    )?;
    assert_floats_close(
        &json!(capture.scene.members[1].source_object_color),
        &[0.04, 0.62, 0.98, 1.0],
        "blue source Object color",
    )?;
    assert!(
        capture
            .scene
            .members
            .iter()
            .all(|member| member.is_instance)
    );
    assert!(
        capture
            .scene
            .members
            .iter()
            .all(|member| !member.use_particle_hair)
    );
    assert_ne!(
        capture.scene.members[0].object_random_id,
        capture.scene.members[1].object_random_id
    );
    assert_ne!(
        capture.scene.members[0].object_pass_index,
        capture.scene.members[1].object_pass_index
    );
    assert_eq!(
        [
            capture.scene.members[0].object_pass_index,
            capture.scene.members[1].object_pass_index,
        ],
        data.pass_index_range
    );
    assert_ne!(
        capture.scene.members[0].object_to_world,
        capture.scene.members[1].object_to_world
    );
    assert!(capture.scene.members[0].ray_visibility.shadow);
    assert!(!capture.scene.members[1].ray_visibility.shadow);
    assert!(
        capture
            .scene
            .members
            .iter()
            .all(|member| member.ray_visibility.camera)
    );

    let probe_names = capture
        .probes
        .iter()
        .map(|probe| probe.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        probe_names,
        [
            "red_hair_hit",
            "blue_hair_hit",
            "red_shadow_visible",
            "blue_shadow_masked",
            "lit_receiver",
        ]
    );
    assert_eq!(capture.probes[0].pixel, [27, 48]);
    assert_eq!(capture.probes[1].pixel, [69, 48]);
    assert_eq!(capture.probes[2].pixel, [27, 57]);
    assert_eq!(capture.probes[3].pixel, [69, 57]);
    assert_eq!(capture.probes[4].pixel, [48, 57]);
    assert!(capture.probes.iter().all(|probe| probe.radius == 2));
    Ok(())
}

fn check_imported_members(scene: &Value, capture: &Capture) -> TestResult {
    let active_scene = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?;
    let scene_data = &scene["scenes"][active_scene];
    let groups = scene_data["cycles_hair_shadow_groups"]
        .as_array()
        .ok_or("schema18 Cycles Hair shadow groups are missing")?;
    assert_eq!(
        groups.len(),
        1,
        "expected one shared Hair Curves geometry group"
    );
    let group = &groups[0];
    assert_eq!(group["members"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        scene_data["render"]["cycles_curves_shape"].as_str(),
        Some(capture.cycles_curves.shape.as_str())
    );
    assert_eq!(
        scene_data["render"]["cycles_curves_subdivisions"].as_u64(),
        Some(u64::from(capture.cycles_curves.subdivisions))
    );

    let first_id = node_id(scene, &capture.scene.members[0].name)?;
    let second_id = node_id(scene, &capture.scene.members[1].name)?;
    let first_node = &scene["nodes"][&first_id];
    let second_node = &scene["nodes"][&second_id];
    assert_eq!(first_node["kind"], "hair_curves");
    assert_eq!(second_node["kind"], "hair_curves");
    let shared_data = first_node["data"]
        .as_str()
        .ok_or("shared Hair data id is missing")?;
    assert_eq!(second_node["data"].as_str(), Some(shared_data));
    assert_eq!(group["geometry_key"]["kind"].as_str(), Some("data"));
    assert_eq!(group["geometry_key"]["data_id"].as_str(), Some(shared_data));
    assert_eq!(
        group["representative_node_id"].as_str(),
        Some(first_id.as_str())
    );
    assert_eq!(
        group["member_node_ids"],
        json!([first_id.as_str(), second_id.as_str()])
    );

    let expected = [&capture.scene.members[0], &capture.scene.members[1]];
    let ids = [&first_id, &second_id];
    for index in 0..expected.len() {
        let source = expected[index];
        let imported = &group["members"][index];
        let parent_id = node_id(scene, &source.parent)?;
        assert_eq!(
            imported["instance"]["real_object_node_id"].as_str(),
            Some(ids[index].as_str())
        );
        assert_eq!(
            imported["instance"]["parent_node_id"].as_str(),
            Some(parent_id.as_str())
        );
        assert_eq!(
            imported["instance"]["persistent_id"],
            json!(source.persistent_id)
        );
        assert_eq!(imported["instance"]["is_instance"], source.is_instance);
        assert_eq!(
            imported["instance"]["use_particle_hair"],
            source.use_particle_hair
        );
        assert_matrix_close(
            &imported["object_to_world"],
            &source.object_to_world,
            &format!("{} world transform", source.name),
        )?;
        assert_floats_close(
            &imported["object_color"],
            &source.object_color,
            &format!("{} Object Info color", source.name),
        )?;
        assert_eq!(imported["object_pass_index"], source.object_pass_index);
        assert_eq!(
            imported["object_random_id"].as_i64(),
            Some(i64::from(source.object_random_id))
        );
        assert_eq!(
            imported["ray_visibility"]["camera"],
            source.ray_visibility.camera
        );
        assert_eq!(
            imported["ray_visibility"]["diffuse"],
            source.ray_visibility.diffuse
        );
        assert_eq!(
            imported["ray_visibility"]["glossy"],
            source.ray_visibility.glossy
        );
        assert_eq!(
            imported["ray_visibility"]["transmission"],
            source.ray_visibility.transmission
        );
        assert_eq!(
            imported["ray_visibility"]["volume_scatter"],
            source.ray_visibility.volume_scatter
        );
        assert_eq!(
            imported["ray_visibility"]["shadow"],
            source.ray_visibility.shadow
        );

        let material_id = scene["materials"]
            .as_object()
            .and_then(|materials| {
                materials.iter().find_map(|(id, material)| {
                    (material["name"].as_str() == Some(source.slot_0_material.as_str()))
                        .then_some(id)
                })
            })
            .ok_or_else(|| format!("Hair material {} was not imported", source.slot_0_material))?;
        assert_eq!(
            imported["slot_0_material_id"].as_str(),
            Some(material_id.as_str())
        );
    }

    let data = &scene["data_blocks"][shared_data]["hair_curves"];
    let curves = data["curves"]
        .as_array()
        .ok_or("shared imported Hair Curves are missing")?;
    assert_eq!(curves.len(), capture.scene.shared_hair_data.curve_count);
    for (curve_index, curve) in curves.iter().enumerate() {
        let points: Vec<[f64; 3]> = serde_json::from_value(curve["points"].clone())?;
        let expected =
            &capture.scene.shared_hair_data.key_positions[curve_index * 4..curve_index * 4 + 4];
        assert_eq!(points.len(), expected.len());
        for (point_index, (actual, expected)) in points.iter().zip(expected).enumerate() {
            for axis in 0..3 {
                assert_close(
                    actual[axis],
                    expected[axis],
                    FLOAT_TOLERANCE,
                    &format!("Hair curve {curve_index} point {point_index} axis {axis}"),
                );
            }
        }
        assert_close(
            native_oracle::number(&curve["radius"])?,
            capture.scene.shared_hair_data.radius,
            FLOAT_TOLERANCE,
            &format!("Hair curve {curve_index} radius"),
        );
    }
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
                "resolution_percentage": 100,
                "samples": SAMPLES,
                "seed": SEED,
                "max_bounces": 2
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

fn render_frame(project: &Path, root: &Path) -> TestResult<ImageData> {
    let out = root.join("pot_hair_instance_context18");
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--engine", "path", "--format", "exr", "--out"])
        .arg(&out)
        .arg("--json");
    checked(&mut render, "physical Hair instance path render")?;
    load_image_file(
        project,
        root,
        "pot_hair_instance_context18",
        &out.join("frame_0001.exr"),
    )
}

fn patch_mean(frame: &ImageData, probe: &Probe) -> [f64; 4] {
    let [x, y] = probe.pixel;
    let radius = probe.radius;
    // Blender's projected probe rows are bottom-up; EXR-backed ImageData uses top-down storage.
    assert!(x >= radius && y >= radius && x + radius < frame.width && y + radius < frame.height);
    let mut total = [0.0; 4];
    let mut count = 0.0;
    for py in y - radius..=y + radius {
        for px in x - radius..=x + radius {
            let pixel = sample(
                frame,
                [
                    (f64::from(px) + 0.5) / f64::from(frame.width),
                    (f64::from(frame.height - 1 - py) + 0.5) / f64::from(frame.height),
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

fn assert_matrix_close(actual: &Value, expected: &[[f64; 4]; 4], label: &str) -> TestResult {
    let rows = actual
        .as_array()
        .ok_or_else(|| format!("{label}: matrix is missing"))?;
    assert_eq!(rows.len(), 4, "{label}: expected four matrix rows");
    for (row_index, expected_row) in expected.iter().enumerate() {
        let values = rows[row_index]
            .as_array()
            .ok_or_else(|| format!("{label}: matrix row {row_index} is missing"))?;
        assert_eq!(values.len(), 4, "{label}: expected four columns");
        for (column_index, expected_value) in expected_row.iter().enumerate() {
            assert_close(
                values[column_index]
                    .as_f64()
                    .ok_or_else(|| format!("{label}: matrix entry is missing"))?,
                *expected_value,
                FLOAT_TOLERANCE,
                &format!("{label} [{row_index}][{column_index}]"),
            );
        }
    }
    Ok(())
}

fn assert_floats_close(actual: &Value, expected: &[f64], label: &str) -> TestResult {
    let values = actual
        .as_array()
        .ok_or_else(|| format!("{label}: value is missing"))?;
    assert_eq!(values.len(), expected.len(), "{label}: channel count");
    for (index, expected_value) in expected.iter().enumerate() {
        assert_close(
            values[index]
                .as_f64()
                .ok_or_else(|| format!("{label}: channel {index} is missing"))?,
            *expected_value,
            FLOAT_TOLERANCE,
            &format!("{label} channel {index}"),
        );
    }
    Ok(())
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: expected {expected}, got {actual}, tolerance {tolerance}"
    );
}
