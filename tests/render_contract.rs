use std::{
    error::Error,
    fs,
    io::{BufReader, Cursor},
    path::Path,
    process::Output,
};

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn json(output: &Output) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn init_scene(scene: &Path) -> Result<(), Box<dyn Error>> {
    let output = pot().arg("init").arg(scene).arg("--json").output()?;
    if !output.status.success() {
        return Err(
            std::io::Error::other(String::from_utf8_lossy(&output.stderr).into_owned()).into(),
        );
    }
    Ok(())
}

fn apply_operations(
    scene: &Path,
    operations: &str,
    extra: &[&str],
) -> Result<Output, Box<dyn Error>> {
    let directory = scene
        .parent()
        .ok_or_else(|| std::io::Error::other("scene directory has no parent"))?;
    let file = directory.join("operations.json");
    fs::write(&file, operations)?;
    let mut command = pot();
    command
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .args(extra);
    Ok(command.output()?)
}

fn first_batch() -> &'static str {
    include_str!("fixtures/first.json")
}

fn assert_success(output: &Output) -> Result<Value, Box<dyn Error>> {
    if !output.status.success() {
        return Err(
            std::io::Error::other(String::from_utf8_lossy(&output.stdout).into_owned()).into(),
        );
    }
    json(output)
}

#[test]
fn top_preview_is_pickable_and_stale_after_an_edit() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let applied = apply_operations(&scene, first_batch(), &["--json"])?;
    assert_success(&applied)?;

    let output_dir = directory.path().join("previews");
    let preview = pot()
        .arg("preview")
        .arg(&scene)
        .args(["--views", "top", "--size", "128", "--out"])
        .arg(&output_dir)
        .arg("--json")
        .output()?;
    let preview_json = assert_success(&preview)?;
    let manifest_path = output_dir.join("top.manifest.json");
    let image_path = output_dir.join("top.png");
    assert!(image_path.is_file());
    let preview_record = preview_json["result"]["previews"][0].clone();
    assert_eq!(preview_record["view"], "top");
    assert_eq!(preview_record["width"], 128);
    let png_file = fs::File::open(&image_path)?;
    let png_reader = png::Decoder::new(BufReader::new(png_file)).read_info()?;
    assert_eq!(png_reader.info().width, 128);
    assert_eq!(png_reader.info().height, 128);

    let reused = pot()
        .arg("preview")
        .arg(&scene)
        .args(["--views", "top", "--size", "128", "--out"])
        .arg(&output_dir)
        .arg("--json")
        .output()?;
    let reused_json = assert_success(&reused)?;
    assert!(
        reused_json["result"]["previews"][0]["reused"]
            .as_bool()
            .is_some_and(|value| value)
    );

    let conflict = pot()
        .arg("preview")
        .arg(&scene)
        .args([
            "--views", "top", "--size", "128", "--mode", "normal", "--out",
        ])
        .arg(&output_dir)
        .arg("--json")
        .output()?;
    assert_eq!(conflict.status.code(), Some(5));
    assert_eq!(json(&conflict)?["error"]["code"], "OUTPUT_EXISTS");

    let face_pick = pot()
        .arg("pick")
        .arg(&scene)
        .arg("--render")
        .arg(&manifest_path)
        .args(["--pixel", "64,64", "--domain", "face", "--json"])
        .output()?;
    let face_json = assert_success(&face_pick)?;
    let hit = &face_json["result"];
    assert!(hit["hit"].as_bool().is_some_and(|value| value));
    assert_eq!(hit["target"]["id"], "body");
    assert_eq!(hit["target"]["elements"]["domain"], "face");
    assert!(
        hit["target"]["elements"]["ids"][0]
            .as_str()
            .is_some_and(|id| id.starts_with('f'))
    );
    let z = hit["world_position"][2]
        .as_f64()
        .ok_or_else(|| std::io::Error::other("world position is missing"))?;
    assert!((z - 0.8).abs() < 1.0e-6);

    let object_pick = pot()
        .arg("pick")
        .arg(&scene)
        .arg("--render")
        .arg(&manifest_path)
        .args(["--pixel", "64,64", "--domain", "object", "--json"])
        .output()?;
    let object_hit = assert_success(&object_pick)?["result"].clone();
    assert!(object_hit["hit"].as_bool().is_some_and(|value| value));
    assert_eq!(object_hit["domain"], "object");
    assert_eq!(object_hit["target"]["id"], "body");
    assert!(object_hit["target"]["elements"].is_null());

    let bone_pick = pot()
        .arg("pick")
        .arg(&scene)
        .arg("--render")
        .arg(&manifest_path)
        .args(["--pixel", "64,64", "--domain", "bone", "--json"])
        .output()?;
    assert_eq!(bone_pick.status.code(), Some(2));
    assert_eq!(json(&bone_pick)?["error"]["code"], "INVALID_ARGUMENT");

    for (domain, pixel, prefix) in [("vertex", "7,29", 'v'), ("edge", "7,64", 'e')] {
        let selected = pot()
            .arg("pick")
            .arg(&scene)
            .arg("--render")
            .arg(&manifest_path)
            .args(["--pixel", pixel, "--domain", domain, "--json"])
            .output()?;
        let selected_json = assert_success(&selected)?;
        let selector = selected_json["result"]["target"]["elements"]["ids"][0]
            .as_str()
            .ok_or_else(|| std::io::Error::other("element selector is missing"))?;
        assert!(selector.starts_with(prefix));
    }

    let background = pot()
        .arg("pick")
        .arg(&scene)
        .arg("--render")
        .arg(&manifest_path)
        .args(["--pixel", "0,0", "--json"])
        .output()?;
    let background_json = assert_success(&background)?;
    assert!(
        background_json["result"]["hit"]
            .as_bool()
            .is_some_and(|hit| !hit)
    );
    assert!(background_json["result"]["target"].is_null());

    let out_of_range = pot()
        .arg("pick")
        .arg(&scene)
        .arg("--render")
        .arg(&manifest_path)
        .args(["--pixel", "128,0", "--json"])
        .output()?;
    assert_eq!(out_of_range.status.code(), Some(2));
    assert_eq!(json(&out_of_range)?["error"]["code"], "INVALID_ARGUMENT");

    let ids_path = output_dir.join("top.ids.bin");
    let ids_bytes = fs::read(&ids_path)?;
    fs::remove_file(&ids_path)?;
    let missing_buffer = pot()
        .arg("pick")
        .arg(&scene)
        .arg("--render")
        .arg(&manifest_path)
        .args(["--pixel", "64,64", "--json"])
        .output()?;
    assert_eq!(missing_buffer.status.code(), Some(3));
    assert_eq!(json(&missing_buffer)?["error"]["code"], "FILE_NOT_FOUND");
    fs::write(&ids_path, ids_bytes)?;

    let depth_path = output_dir.join("top.depth.bin");
    let depth_bytes = fs::read(&depth_path)?;
    let mut corrupted_depth = depth_bytes.clone();
    let first_byte = corrupted_depth
        .first_mut()
        .ok_or_else(|| std::io::Error::other("depth buffer is empty"))?;
    *first_byte ^= 1;
    fs::write(&depth_path, corrupted_depth)?;
    let corrupt_buffer = pot()
        .arg("pick")
        .arg(&scene)
        .arg("--render")
        .arg(&manifest_path)
        .args(["--pixel", "64,64", "--json"])
        .output()?;
    assert_eq!(corrupt_buffer.status.code(), Some(4));
    assert_eq!(json(&corrupt_buffer)?["error"]["code"], "RENDER_INVALID");
    fs::write(&depth_path, depth_bytes)?;

    let next = r#"{"schema_version":1,"base_revision":1,"operations":[{"op":"node.create","id":"extra","kind":"group"}]}"#;
    let changed = apply_operations(&scene, next, &["--json"])?;
    assert_success(&changed)?;
    let stale = pot()
        .arg("pick")
        .arg(&scene)
        .arg("--render")
        .arg(&manifest_path)
        .args(["--pixel", "64,64", "--json"])
        .output()?;
    assert_eq!(stale.status.code(), Some(5));
    assert_eq!(json(&stale)?["error"]["code"], "STALE_RENDER");
    Ok(())
}

#[test]
fn apply_preview_publishes_before_commit_and_failure_preserves_revision()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let applied = apply_operations(
        &scene,
        first_batch(),
        &["--preview", "iso", "--size", "64", "--json"],
    )?;
    let applied_json = assert_success(&applied)?;
    assert_eq!(applied_json["scene"]["revision"], 1);
    assert_eq!(
        applied_json["result"]["previews"].as_array().map(Vec::len),
        Some(1)
    );
    let preview_manifest = applied_json["result"]["previews"][0]["manifest"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("apply preview manifest path is missing"))?;
    assert!(Path::new(preview_manifest).is_file());
    let picked = pot()
        .arg("pick")
        .arg(&scene)
        .args(["--render", preview_manifest, "--pixel", "32,32", "--json"])
        .output()?;
    assert_eq!(assert_success(&picked)?["result"]["target"]["id"], "body");

    let next = r#"{"schema_version":1,"base_revision":1,"operations":[{"op":"node.create","id":"extra","kind":"group"}]}"#;
    let failed = apply_operations(&scene, next, &["--preview", "iso,unknown", "--json"])?;
    assert_eq!(failed.status.code(), Some(2));
    assert_eq!(json(&failed)?["error"]["code"], "INVALID_ARGUMENT");
    let saved: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(saved["revision"], 1);
    assert!(
        !saved["nodes"]
            .as_object()
            .is_some_and(|nodes| nodes.contains_key("extra"))
    );
    Ok(())
}

#[test]
fn render_settings_have_spec_defaults_and_update_atomically() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let initial: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(
        initial["scenes"]["scene_main"]["render"]["resolution_x"],
        1920
    );
    assert_eq!(
        initial["scenes"]["scene_main"]["render"]["resolution_y"],
        1080
    );
    assert_eq!(initial["scenes"]["scene_main"]["render"]["samples"], 64);
    assert_eq!(initial["scenes"]["scene_main"]["render"]["max_bounces"], 4);

    let update = r#"{"schema_version":1,"base_revision":0,"operations":[{"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":320,"resolution_y":240,"resolution_percentage":50,"samples":8,"seed":42,"max_bounces":3,"film_transparent":true,"engine":"realtime"}}]}"#;
    let updated = apply_operations(&scene, update, &["--json"])?;
    let updated_json = assert_success(&updated)?;
    assert_eq!(updated_json["scene"]["revision"], 1);
    let saved: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(saved["scenes"]["scene_main"]["render"]["resolution_x"], 320);
    assert_eq!(saved["scenes"]["scene_main"]["render"]["resolution_y"], 240);
    assert_eq!(
        saved["scenes"]["scene_main"]["render"]["resolution_percentage"],
        50
    );
    assert_eq!(saved["scenes"]["scene_main"]["render"]["samples"], 8);
    assert_eq!(saved["scenes"]["scene_main"]["render"]["seed"], 42);
    assert_eq!(saved["scenes"]["scene_main"]["render"]["max_bounces"], 3);
    assert!(
        saved["scenes"]["scene_main"]["render"]["film_transparent"]
            .as_bool()
            .is_some_and(|value| value)
    );
    assert_eq!(
        saved["scenes"]["scene_main"]["render"]["engine"],
        "realtime"
    );

    let invalid = r#"{"schema_version":1,"base_revision":1,"operations":[{"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_percentage":0}}]}"#;
    let rejected = apply_operations(&scene, invalid, &["--json"])?;
    assert_eq!(rejected.status.code(), Some(2));
    assert_eq!(json(&rejected)?["error"]["code"], "INVALID_OPERATION");
    let unchanged: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(unchanged["revision"], 1);
    assert_eq!(
        unchanged["scenes"]["scene_main"]["render"]["resolution_percentage"],
        50
    );
    Ok(())
}

#[test]
fn render_uses_scene_settings_for_png_and_linear_exr_sequences() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let missing_camera = pot()
        .arg("render")
        .arg(&scene)
        .args(["--format", "png", "--out"])
        .arg(directory.path().join("no-camera"))
        .arg("--json")
        .output()?;
    assert_eq!(missing_camera.status.code(), Some(3));
    assert_eq!(json(&missing_camera)?["error"]["code"], "TARGET_NOT_FOUND");

    for format in ["mp4", "webm"] {
        let unsupported = pot()
            .arg("render")
            .arg(&scene)
            .args(["--format", format, "--out"])
            .arg(directory.path().join(format!("{format}-output")))
            .arg("--json")
            .output()?;
        assert_eq!(unsupported.status.code(), Some(4));
        assert_eq!(json(&unsupported)?["error"]["code"], "UNSUPPORTED_FEATURE");
    }

    let operations = r#"{"schema_version":1,"base_revision":0,"operations":[{"op":"node.create","id":"body","kind":"box","params":{"size":1},"transform":{"translation":[0,0,0.5]}},{"op":"camera.create","id":"camera_main","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":3},{"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},{"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":64,"resolution_y":48,"resolution_percentage":100,"samples":2,"seed":7,"film_transparent":true}}]}"#;
    assert_success(&apply_operations(&scene, operations, &["--json"])?)?;

    let exr_dir = directory.path().join("exr");
    let exr = pot()
        .arg("render")
        .arg(&scene)
        .args(["--frames", "1:2", "--format", "exr", "--out"])
        .arg(&exr_dir)
        .arg("--json")
        .output()?;
    let exr_json = assert_success(&exr)?;
    assert!(exr_dir.join("frame_0001.exr").is_file());
    assert!(exr_dir.join("frame_0002.exr").is_file());
    let exr_bytes = fs::read(exr_dir.join("frame_0001.exr"))?;
    assert_eq!(exr_bytes.get(0..4), Some(&[0x76, 0x2f, 0x31, 0x01][..]));
    let run_manifest: Value =
        serde_json::from_slice(&fs::read(exr_dir.join("render.manifest.json"))?)?;
    assert_eq!(run_manifest["frames"].as_array().map(Vec::len), Some(2));
    assert_eq!(run_manifest["samples"], 2);
    assert_eq!(run_manifest["seed"], 7);
    assert_eq!(run_manifest["resolution"]["width"], 64);
    assert_eq!(exr_json["result"]["frame_count"], 2);

    let png_dir = directory.path().join("png");
    let png_output = pot()
        .arg("render")
        .arg(&scene)
        .args(["--engine", "realtime", "--format", "png", "--out"])
        .arg(&png_dir)
        .arg("--json")
        .output()?;
    assert_success(&png_output)?;
    let mut png_reader = png::Decoder::new(BufReader::new(fs::File::open(
        png_dir.join("frame_0001.png"),
    )?))
    .read_info()?;
    assert_eq!(png_reader.info().width, 64);
    assert_eq!(png_reader.info().height, 48);
    let mut pixels = vec![
        0;
        png_reader
            .output_buffer_size()
            .ok_or("PNG output buffer size overflow")?
    ];
    let frame_info = png_reader.next_frame(&mut pixels)?;
    assert_eq!(frame_info.color_type, png::ColorType::Rgba);
    assert_eq!(pixels.get(3), Some(&0));
    let realtime_manifest: Value =
        serde_json::from_slice(&fs::read(png_dir.join("render.manifest.json"))?)?;
    assert_eq!(realtime_manifest["samples"], 1);
    assert_eq!(realtime_manifest["seed"], 0);
    Ok(())
}

#[test]
fn preview_and_pick_render_evaluated_strokes_and_points() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let mut saved: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let objects = saved["collections"]["collection_root"]["objects"]
        .as_array_mut()
        .ok_or_else(|| std::io::Error::other("root collection objects are missing"))?;
    objects.push(serde_json::json!("ink"));
    saved["nodes"]["ink"] = serde_json::json!({
        "name": "Ink",
        "kind": "grease_pencil",
        "data": "ink_data"
    });
    saved["data_blocks"]["ink_data"] = serde_json::json!({
        "type": "grease_pencil",
        "grease_pencil": {
            "layers": [{
                "id": "layer_main",
                "name": "Ink",
                "opacity": 1.0,
                "visible": true,
                "frames": [{
                    "frame": 1.0,
                    "strokes": [{
                        "id": "stroke_main",
                        "points": [
                            {"position": [-0.3, 0.0, 0.5], "pressure": 1.0, "radius": 0.03, "opacity": 1.0, "time": 0.0},
                            {"position": [0.0, 0.0, 0.5], "pressure": 1.0, "radius": 0.03, "opacity": 1.0, "time": 0.1},
                            {"position": [0.3, 0.0, 0.5], "pressure": 1.0, "radius": 0.03, "opacity": 1.0, "time": 0.2}
                        ],
                        "cyclic": false
                    }]
                }]
            }]
        }
    });
    fs::write(scene.join("scene.json"), serde_json::to_vec_pretty(&saved)?)?;

    let output_dir = directory.path().join("stroke-preview");
    let preview = pot()
        .arg("preview")
        .arg(&scene)
        .args(["--views", "top", "--size", "128", "--out"])
        .arg(&output_dir)
        .arg("--json")
        .output()?;
    let preview_json = assert_success(&preview)?;
    assert!(output_dir.join("top.png").is_file());
    let manifest = preview_json["result"]["previews"][0]["manifest"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("stroke manifest path is missing"))?;
    let evaluation_hash = preview_json["result"]["previews"][0]["evaluation_hash"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("evaluation hash is missing"))?;
    for (domain, pixel, prefix) in [("stroke", "40,64", 's'), ("point", "11,64", 'p')] {
        let picked = pot()
            .arg("pick")
            .arg(&scene)
            .args([
                "--render", manifest, "--pixel", pixel, "--domain", domain, "--json",
            ])
            .output()?;
        let result = assert_success(&picked)?["result"].clone();
        assert!(result["hit"].as_bool().is_some_and(|value| value));
        assert_eq!(result["target"]["id"], "ink");
        assert_eq!(result["target"]["elements"]["domain"], domain);
        assert!(
            result["target"]["elements"]["ids"][0]
                .as_str()
                .is_some_and(|id| id.starts_with(prefix))
        );
        assert!(result["world_normal"].is_null());
        assert_eq!(result["snapshot_hash"], evaluation_hash);
    }
    Ok(())
}

#[test]
fn material_emission_and_transmission_fields_create_update_and_validate()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let create = r#"{"schema_version":1,"base_revision":0,"operations":[{"op":"material.create","id":"glow","emission_color":[1.0,0.25,0.0],"emission_strength":2.0,"transmission":0.5,"ior":1.33},{"op":"material.create","id":"plain"}]}"#;
    assert_success(&apply_operations(&scene, create, &["--json"])?)?;
    let created: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(
        created["materials"]["glow"]["emission_color"],
        serde_json::json!([1.0, 0.25, 0.0])
    );
    assert_eq!(created["materials"]["glow"]["emission_strength"], 2.0);
    assert_eq!(created["materials"]["glow"]["transmission"], 0.5);
    assert_eq!(created["materials"]["glow"]["ior"], 1.33);
    assert_eq!(
        created["materials"]["plain"]["emission_color"],
        serde_json::json!([0.0, 0.0, 0.0])
    );
    assert_eq!(created["materials"]["plain"]["emission_strength"], 0.0);
    assert_eq!(created["materials"]["plain"]["transmission"], 0.0);
    assert_eq!(created["materials"]["plain"]["ior"], 1.45);

    let update = r#"{"schema_version":1,"base_revision":1,"operations":[{"op":"material.update","target":{"id":"glow"},"set":{"emission_strength":4.0,"transmission":0.2,"ior":1.5}}]}"#;
    assert_success(&apply_operations(&scene, update, &["--json"])?)?;
    let updated: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(updated["materials"]["glow"]["emission_strength"], 4.0);
    assert_eq!(updated["materials"]["glow"]["transmission"], 0.2);
    assert_eq!(updated["materials"]["glow"]["ior"], 1.5);
    assert_eq!(
        updated["materials"]["glow"]["emission_color"],
        serde_json::json!([1.0, 0.25, 0.0])
    );

    let invalid = r#"{"schema_version":1,"base_revision":2,"operations":[{"op":"material.update","target":{"id":"glow"},"set":{"transmission":1.1}}]}"#;
    let rejected = apply_operations(&scene, invalid, &["--json"])?;
    assert_eq!(rejected.status.code(), Some(2));
    assert_eq!(json(&rejected)?["error"]["code"], "INVALID_OPERATION");
    let unchanged: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(unchanged["revision"], 2);
    assert_eq!(unchanged["materials"]["glow"]["transmission"], 0.2);
    Ok(())
}
fn render_path_image(
    scene: &Path,
    output_directory: &Path,
) -> Result<(Vec<u8>, String), Box<dyn Error>> {
    let output = pot()
        .arg("render")
        .arg(scene)
        .args(["--format", "png", "--out"])
        .arg(output_directory)
        .arg("--json")
        .output()?;
    assert_success(&output)?;
    let image = fs::read(output_directory.join("frame_0001.png"))?;
    let manifest: Value =
        serde_json::from_slice(&fs::read(output_directory.join("render.manifest.json"))?)?;
    let hash = manifest["frames"][0]["hash"]
        .as_str()
        .ok_or("render frame hash missing")?
        .to_owned();
    Ok((image, hash))
}

fn decode_rgba(image: &[u8]) -> Result<(u32, u32, Vec<u8>), Box<dyn Error>> {
    let mut reader = png::Decoder::new(Cursor::new(image)).read_info()?;
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size overflow")?
    ];
    let info = reader.next_frame(&mut pixels)?;
    if info.color_type != png::ColorType::Rgba {
        return Err("renderer PNG is not RGBA".into());
    }
    pixels.truncate(info.buffer_size());
    Ok((info.width, info.height, pixels))
}

#[test]
fn path_render_is_seeded_per_pixel_and_repeatable() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let operations = r#"{"schema_version":1,"base_revision":0,"operations":[{"op":"material.create","id":"clay","base_color":[0.7,0.4,0.2,1.0],"roughness":0.65},{"op":"node.create","id":"body","kind":"box","params":{"size":1.0},"transform":{"translation":[0,0,0.5]},"material":"clay"},{"op":"camera.create","id":"camera_main","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":2.5},{"op":"light.create","id":"key","light_type":"area","energy":12.0,"radius":0.8,"transform":{"translation":[-1,-1,3]}},{"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},{"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":24,"resolution_y":24,"samples":6,"seed":17,"max_bounces":2}}]}"#;
    assert_success(&apply_operations(&scene, operations, &["--json"])?)?;

    let (first, first_hash) = render_path_image(&scene, &directory.path().join("first"))?;
    let (repeat, repeat_hash) = render_path_image(&scene, &directory.path().join("repeat"))?;
    assert_eq!(first_hash, repeat_hash);
    assert_eq!(first, repeat);

    let update = r#"{"schema_version":1,"base_revision":1,"operations":[{"op":"render.update","target":{"id":"scene_main"},"set":{"seed":18}}]}"#;
    assert_success(&apply_operations(&scene, update, &["--json"])?)?;
    let (different, different_hash) =
        render_path_image(&scene, &directory.path().join("different"))?;
    assert_ne!(first_hash, different_hash);
    assert_ne!(first, different);
    Ok(())
}

#[test]
fn path_emission_strength_scales_rendered_radiance() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init_scene(&scene)?;
    let operations = r#"{"schema_version":1,"base_revision":0,"operations":[{"op":"material.create","id":"glow","base_color":[0.0,0.0,0.0,1.0],"emission_color":[1.0,0.25,0.0],"emission_strength":0.1},{"op":"node.create","id":"body","kind":"box","params":{"size":1.0},"transform":{"translation":[0,0,0.5]},"material":"glow"},{"op":"camera.create","id":"camera_main","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":2.5},{"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},{"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":16,"resolution_y":16,"samples":1,"seed":0,"max_bounces":0}}]}"#;
    assert_success(&apply_operations(&scene, operations, &["--json"])?)?;
    let (dim_image, _) = render_path_image(&scene, &directory.path().join("dim"))?;
    let (width, height, dim_pixels) = decode_rgba(&dim_image)?;
    let center =
        (usize::try_from(height / 2)? * usize::try_from(width)? + usize::try_from(width / 2)?) * 4;
    let dim_red = *dim_pixels
        .get(center)
        .ok_or("dim image center is missing")?;

    let update = r#"{"schema_version":1,"base_revision":1,"operations":[{"op":"material.update","target":{"id":"glow"},"set":{"emission_strength":0.2}}]}"#;
    assert_success(&apply_operations(&scene, update, &["--json"])?)?;
    let (bright_image, _) = render_path_image(&scene, &directory.path().join("bright"))?;
    let (_, _, bright_pixels) = decode_rgba(&bright_image)?;
    let bright_red = *bright_pixels
        .get(center)
        .ok_or("bright image center is missing")?;
    assert!(
        bright_red > dim_red,
        "emission should brighten the rendered pixel"
    );
    Ok(())
}

#[test]
fn path_shadow_rays_darkening_blocked_surface() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let mut center_values = Vec::new();
    for blocked in [false, true] {
        let scene = directory
            .path()
            .join(if blocked { "blocked" } else { "lit" });
        init_scene(&scene)?;
        let mut operations = vec![
            serde_json::json!({"op":"material.create","id":"mat","base_color":[0.9,0.9,0.9,1.0],"roughness":0.9}),
            serde_json::json!({"op":"node.create","id":"floor","kind":"box","params":{"size":4.0},"transform":{"translation":[0,0,-0.05],"scale":[1.0,1.0,0.025]},"material":"mat"}),
        ];
        if blocked {
            operations.push(serde_json::json!({"op":"node.create","id":"blocker","kind":"box","params":{"size":0.3},"transform":{"translation":[0.25,0,1.0]},"material":"mat"}));
        }
        operations.extend([
            serde_json::json!({"op":"camera.create","id":"camera_main","transform":{"translation":[0,-4,4],"rotation":[0.382_683_432_365_089_8,0.0,0.0,0.923_879_532_511_286_7]},"projection":"orthographic","ortho_scale":5.0}),
            serde_json::json!({"op":"light.create","id":"key","light_type":"point","energy":20.0,"radius":0.05,"transform":{"translation":[0.5,0,2.0]}}),
            serde_json::json!({"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}}),
            serde_json::json!({"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":32,"resolution_y":32,"samples":1,"seed":0,"max_bounces":0}}),
        ]);
        let batch = serde_json::json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        });
        assert_success(&apply_operations(&scene, &batch.to_string(), &["--json"])?)?;
        let (image, _) = render_path_image(
            &scene,
            &directory
                .path()
                .join(if blocked { "shadow" } else { "light" }),
        )?;
        let (width, height, pixels) = decode_rgba(&image)?;
        let center = (usize::try_from(height / 2)? * usize::try_from(width)?
            + usize::try_from(width / 2)?)
            * 4;
        let center_luminance = u16::from(*pixels.get(center).ok_or("center pixel is missing")?)
            + u16::from(
                *pixels
                    .get(center + 1)
                    .ok_or("center green channel is missing")?,
            )
            + u16::from(
                *pixels
                    .get(center + 2)
                    .ok_or("center blue channel is missing")?,
            );
        center_values.push(center_luminance);
    }
    let lit = center_values.first().ok_or("lit frame was not rendered")?;
    let shadowed = center_values
        .get(1)
        .ok_or("shadowed frame was not rendered")?;
    assert!(
        shadowed < lit,
        "a blocking object should darken the directly lit surface: {lit} vs {shadowed}"
    );
    Ok(())
}
