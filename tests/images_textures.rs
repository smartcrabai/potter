use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Output,
};

use assert_cmd::Command;
use serde_json::{Value, json};
use sha2::Digest;
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn init(scene: &Path) -> Result<(), Box<dyn Error>> {
    let output = pot().arg("init").arg(scene).arg("--json").output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn scene_doc(scene: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(
        scene.join("scene.json"),
    )?)?)
}

fn apply(scene: &Path, operations: &Value, flags: &[&str]) -> Result<Output, Box<dyn Error>> {
    let revision = scene_doc(scene)?["revision"]
        .as_u64()
        .ok_or_else(|| std::io::Error::other("scene revision is missing"))?;
    let batch = scene.with_extension("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": revision,
            "operations": operations,
        }))?,
    )?;
    let mut command = pot();
    command
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .args(flags)
        .arg("--json");
    Ok(command.output()?)
}

fn apply_success(scene: &Path, operations: &Value) -> Result<Value, Box<dyn Error>> {
    let output = apply(scene, operations, &[])?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn image_blob_path(scene: &Path, image: &Value) -> Result<PathBuf, Box<dyn Error>> {
    let blob = image["blob"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("image blob reference is missing"))?;
    let digest = blob
        .strip_prefix("sha256:")
        .ok_or_else(|| std::io::Error::other("image blob is not a SHA-256 reference"))?;
    Ok(scene
        .join("assets")
        .join("sha256")
        .join(digest)
        .join("blob"))
}

fn image_blob_json(scene: &Path, image: &Value) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(image_blob_path(
        scene, image,
    )?)?)?)
}

fn assert_blob_hash(image: &Value, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let blob = image["blob"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("image blob reference is missing"))?;
    assert_eq!(
        blob,
        format!("sha256:{}", hex::encode(sha2::Sha256::digest(bytes)))
    );
    Ok(())
}

#[test]
fn generated_image_pixels_material_reference_and_delete_guard() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;

    apply_success(
        &scene,
        &json!([
            {"op":"image.create","id":"paint","name":"Paint","width":16,"height":16,"colorspace":"srgb","fill_color":[0.1,0.2,0.3,1.0]},
            {"op":"image.create","id":"orphan","width":1,"height":1,"fill_color":[0.0,0.0,0.0,1.0]},
            {"op":"material.create","id":"surface","base_color_texture":{"image":"paint","uv_map":"uv_map","interpolation":"closest"}}
        ]),
    )?;

    let created = scene_doc(&scene)?;
    assert_eq!(created["images"]["paint"]["source"], "generated");
    assert_eq!(created["images"]["paint"]["width"], 16);
    assert_eq!(created["images"]["paint"]["height"], 16);
    assert_eq!(created["images"]["paint"]["name"], "Paint");
    let initial_blob = image_blob_path(&scene, &created["images"]["paint"])?;
    let initial_bytes = fs::read(initial_blob)?;
    assert_blob_hash(&created["images"]["paint"], &initial_bytes)?;
    let initial_payload: Value = serde_json::from_slice(&initial_bytes)?;
    assert_eq!(initial_payload["format"], "potter.rgba_f64.v1");
    assert_eq!(initial_payload["width"], 16);
    assert_eq!(initial_payload["height"], 16);
    assert_eq!(initial_payload["pixels"][0], json!([0.1, 0.2, 0.3, 1.0]));
    assert_eq!(
        created["materials"]["surface"]["base_color_texture"]["image"],
        "paint"
    );
    assert_eq!(
        created["materials"]["surface"]["base_color_texture"]["uv_map"],
        "uv_map"
    );
    assert_eq!(
        created["materials"]["surface"]["base_color_texture"]["interpolation"],
        "closest"
    );
    apply_success(
        &scene,
        &json!([{
            "op":"material.update",
            "target":{"id":"surface"},
            "set":{
                "roughness_texture":{"image":"paint","uv_map":null,"interpolation":"linear"},
                "metallic_texture":{"image":"paint","uv_map":"uv_map","interpolation":"closest"},
                "normal_texture":{"image":"paint"}
            }
        }]),
    )?;
    let textured_material = scene_doc(&scene)?["materials"]["surface"].clone();
    assert_eq!(textured_material["roughness_texture"]["image"], "paint");
    assert!(textured_material["roughness_texture"]["uv_map"].is_null());
    assert_eq!(
        textured_material["roughness_texture"]["interpolation"],
        "linear"
    );
    assert_eq!(textured_material["metallic_texture"]["image"], "paint");
    assert_eq!(textured_material["normal_texture"]["image"], "paint");

    let full_pixels: Vec<Value> = (0_u32..16)
        .flat_map(|y| {
            (0_u32..16).map(move |x| json!([f64::from(x) / 16.0, f64::from(y) / 16.0, 0.25, 1.0]))
        })
        .collect();
    apply_success(
        &scene,
        &json!([{"op":"image.set_pixels","id":"paint","width":16,"height":16,"pixels":full_pixels}]),
    )?;
    let after_full = scene_doc(&scene)?;
    let full_payload = image_blob_json(&scene, &after_full["images"]["paint"])?;
    assert_eq!(full_payload["pixels"], json!(full_pixels));

    let region = vec![
        json!([1.0, 0.0, 0.0, 1.0]),
        json!([0.0, 1.0, 0.0, 1.0]),
        json!([0.0, 0.0, 1.0, 1.0]),
        json!([1.0, 1.0, 0.0, 1.0]),
    ];
    apply_success(
        &scene,
        &json!([{"op":"image.set_pixels","id":"paint","x":5,"y":6,"width":2,"height":2,"pixels":region}]),
    )?;
    apply_success(
        &scene,
        &json!([{"op":"image.update","target":{"id":"paint"},"set":{"name":"Updated Paint","colorspace":"linear"}}]),
    )?;
    let updated = scene_doc(&scene)?;
    assert_eq!(updated["images"]["paint"]["name"], "Updated Paint");
    assert_eq!(updated["images"]["paint"]["colorspace"], "linear");
    let final_payload = image_blob_json(&scene, &updated["images"]["paint"])?;
    let final_pixels = final_payload["pixels"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("generated image blob pixels are not an array"))?;
    assert_eq!(final_pixels.len(), 256);
    assert_eq!(final_pixels[6 * 16 + 5], json!([1.0, 0.0, 0.0, 1.0]));
    assert_eq!(final_pixels[6 * 16 + 6], json!([0.0, 1.0, 0.0, 1.0]));
    assert_eq!(final_pixels[7 * 16 + 5], json!([0.0, 0.0, 1.0, 1.0]));
    assert_eq!(final_pixels[7 * 16 + 6], json!([1.0, 1.0, 0.0, 1.0]));
    assert_eq!(final_pixels[6 * 16 + 4], json!(full_pixels[6 * 16 + 4]));
    assert_eq!(final_pixels[5 * 16 + 5], json!(full_pixels[5 * 16 + 5]));

    apply_success(
        &scene,
        &json!([{"op":"image.delete","target":{"id":"orphan"}}]),
    )?;
    let guarded = apply(
        &scene,
        &json!([{"op":"image.delete","target":{"id":"paint"}}]),
        &[],
    )?;
    assert_eq!(guarded.status.code(), Some(2));
    let guarded_error: Value = serde_json::from_slice(&guarded.stdout)?;
    assert_eq!(guarded_error["error"]["code"], "INVALID_OPERATION");
    let after_guard = scene_doc(&scene)?;
    assert!(after_guard["images"].get("orphan").is_none());
    assert_eq!(after_guard["images"]["paint"]["name"], "Updated Paint");
    Ok(())
}

#[test]
fn texture_paint_maps_world_stroke_through_face_corner_uvs() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply_success(
        &scene,
        &json!([
            {"op":"image.create","id":"canvas","width":16,"height":16,"fill_color":[0.0,0.0,0.0,1.0]},
            {"op":"material.create","id":"surface","base_color_texture":{"image":"canvas","uv_map":"uv_map","interpolation":"closest"}},
            {"op":"node.create","id":"body","kind":"box","params":{"size":2},"material":"surface"},
            {"op":"uv.unwrap","target":{"id":"body"},"elements":{"domain":"face","ids":["f1"]},"method":"smart"}
        ]),
    )?;

    let before = scene_doc(&scene)?;
    let uv_entries = before["data_blocks"]["body_mesh"]["mesh"]["attributes"]["uv_map"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("box has no face-corner UV map"))?;
    let top_uv = uv_entries
        .iter()
        .find(|entry| entry["face_id"] == 1)
        .ok_or_else(|| std::io::Error::other("top face UV coordinates are missing"))?;
    assert_eq!(top_uv["uv"].as_array().map(std::vec::Vec::len), Some(4));
    for corner in top_uv["uv"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("UV corners are invalid"))?
    {
        let pair = corner
            .as_array()
            .ok_or_else(|| std::io::Error::other("UV corner is invalid"))?;
        let u = pair[0]
            .as_f64()
            .ok_or_else(|| std::io::Error::other("U coordinate is missing"))?;
        let v = pair[1]
            .as_f64()
            .ok_or_else(|| std::io::Error::other("V coordinate is missing"))?;
        assert!((0.0..=1.0).contains(&u));
        assert!((0.0..=1.0).contains(&v));
    }

    apply_success(
        &scene,
        &json!([{
            "op":"paint.texture",
            "target":{"id":"body"},
            "image":"canvas",
            "samples":[{"position":[0.0,0.0,1.0],"radius":0.2,"pressure":1.0,"strength":1.0,"time":0.0}],
            "color":[1.0,0.0,0.0,1.0],
            "blend":"mix",
            "falloff":"linear",
            "seed":7
        }]),
    )?;

    let after = scene_doc(&scene)?;
    let painted = image_blob_json(&scene, &after["images"]["canvas"])?;
    let pixels = painted["pixels"]
        .as_array()
        .ok_or_else(|| std::io::Error::other("painted image blob pixels are not an array"))?;
    assert_eq!(pixels.len(), 256);
    let mut changed = 0;
    let pixel_world_size = 2.0 / 16.0;
    let radius_squared = 0.2_f64 * 0.2;
    for (index, pixel) in pixels.iter().enumerate() {
        if pixel != &json!([0.0, 0.0, 0.0, 1.0]) {
            changed += 1;
            let x = f64::from(u32::try_from(index % 16)?);
            let y = f64::from(u32::try_from(index / 16)?);
            let min_x = -1.0 + x * pixel_world_size;
            let max_x = min_x + pixel_world_size;
            let min_y = -1.0 + y * pixel_world_size;
            let max_y = min_y + pixel_world_size;
            let nearest_x = 0.0_f64.clamp(min_x, max_x);
            let nearest_y = 0.0_f64.clamp(min_y, max_y);
            assert!(nearest_x * nearest_x + nearest_y * nearest_y <= radius_squared + 1.0e-12);
        }
    }
    assert!(
        changed > 0,
        "world-space texture stroke changed no image texels"
    );
    assert!(
        changed < pixels.len(),
        "texture stroke painted outside its local brush region"
    );
    assert_eq!(pixels[0], json!([0.0, 0.0, 0.0, 1.0]));
    assert_eq!(pixels[15 * 16 + 15], json!([0.0, 0.0, 0.0, 1.0]));
    Ok(())
}

fn write_png(path: &Path, color: [u8; 4]) -> Result<(), Box<dyn Error>> {
    let file = fs::File::create(path)?;
    let mut encoder = png::Encoder::new(file, 2, 2);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let pixels = color.repeat(4);
    writer.write_image_data(&pixels)?;
    Ok(())
}

#[test]
fn image_load_copy_and_link_assets_check_changed_file() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let source = directory.path().join("source.png");
    init(&scene)?;
    write_png(&source, [10, 20, 30, 255])?;
    let original_bytes = fs::read(&source)?;
    let source_path = source.display().to_string();

    apply_success(
        &scene,
        &json!([
            {"op":"image.load","id":"copied","path":source_path,"name":"Copied","asset_policy":"copy"},
            {"op":"image.load","id":"linked","path":source_path,"name":"Linked","asset_policy":"link"}
        ]),
    )?;
    let loaded = scene_doc(&scene)?;
    let copied = &loaded["images"]["copied"];
    let linked = &loaded["images"]["linked"];
    assert_eq!(copied["name"], "Copied");
    assert_eq!(copied["width"], 2);
    assert_eq!(copied["height"], 2);
    let copied_bytes = fs::read(image_blob_path(&scene, copied)?)?;
    assert_eq!(copied_bytes, original_bytes);
    assert_blob_hash(copied, &copied_bytes)?;
    assert_eq!(linked["source"], "file");
    assert!(linked["source_path"].as_str().is_some());
    assert!(linked["source_hash"].as_str().is_some());
    assert!(linked["blob"].is_null());

    write_png(&source, [200, 40, 60, 255])?;
    let checked = pot()
        .arg("assets")
        .arg(&scene)
        .arg("--check")
        .arg("--json")
        .output()?;
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stdout)
    );
    let report: Value = serde_json::from_slice(&checked.stdout)?;
    assert_eq!(report["result"]["summary"]["changed"], 1);
    let changed_link = report["result"]["assets"]
        .as_array()
        .and_then(|assets| assets.iter().find(|asset| asset["id"] == "linked"))
        .ok_or_else(|| std::io::Error::other("linked image is missing from the asset check"))?;
    assert_eq!(changed_link["changed"], true);
    assert_eq!(changed_link["missing"], false);
    Ok(())
}

#[test]
fn dry_run_image_changes_do_not_write_asset_blobs() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let before_scene = fs::read(scene.join("scene.json"))?;

    let output = apply(
        &scene,
        &json!([
            {"op":"image.create","id":"preview","width":16,"height":16,"fill_color":[0.25,0.5,0.75,1.0]},
            {"op":"image.set_pixels","id":"preview","x":3,"y":4,"width":1,"height":1,"pixels":[[1.0,0.0,0.0,1.0]]}
        ]),
        &["--dry-run"],
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(envelope["result"]["committed"], false);
    assert_eq!(envelope["result"]["changed"], true);
    assert_eq!(fs::read(scene.join("scene.json"))?, before_scene);
    let asset_root = scene.join("assets").join("sha256");
    assert!(!asset_root.exists() || fs::read_dir(asset_root)?.next().is_none());
    Ok(())
}
#[test]
fn image_set_pixels_persists_udim_tile_blobs() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply_success(
        &scene,
        &json!([
            {"op":"image.create","id":"atlas","width":1,"height":1,"fill_color":[0.0,0.0,0.0,1.0]},
            {"op":"image.set_pixels","id":"atlas","tile":1012,"width":2,"height":1,"pixels":[
                [1.0,0.0,0.0,1.0],
                [0.0,1.0,0.0,1.0]
            ]}
        ]),
    )?;
    let created = scene_doc(&scene)?;
    let image = &created["images"]["atlas"];
    assert_eq!(image["source"], "packed");
    assert_eq!(image["width"], 1);
    assert_eq!(image["height"], 1);
    assert_eq!(image["tiles"][0]["number"], 1012);
    assert_eq!(image["tiles"][0]["width"], 2);
    assert_eq!(image["tiles"][0]["height"], 1);
    let tile_ref = json!({"blob":image["tiles"][0]["blob"]});
    let tile_bytes = fs::read(image_blob_path(&scene, &tile_ref)?)?;
    assert_blob_hash(&tile_ref, &tile_bytes)?;
    let tile_pixels: Value = serde_json::from_slice(&tile_bytes)?;
    assert_eq!(
        tile_pixels["pixels"],
        json!([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    );
    assert_eq!(
        image_blob_json(&scene, image)?["pixels"][0],
        json!([0.0, 0.0, 0.0, 1.0])
    );
    Ok(())
}
#[test]
fn failed_image_batch_leaves_scene_and_asset_store_unchanged() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let original_scene = fs::read(scene.join("scene.json"))?;
    let output = apply(
        &scene,
        &json!([
            {"op":"image.create","id":"temporary","width":4,"height":4},
            {"op":"image.set_pixels","id":"temporary","x":3,"y":3,"width":2,"height":2,"pixels":[
                [1.0,0.0,0.0,1.0],
                [0.0,1.0,0.0,1.0],
                [0.0,0.0,1.0,1.0],
                [1.0,1.0,1.0,1.0]
            ]}
        ]),
        &[],
    )?;
    assert_eq!(output.status.code(), Some(2));
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(envelope["error"]["code"], "INVALID_OPERATION");
    assert_eq!(fs::read(scene.join("scene.json"))?, original_scene);
    let assets = scene.join("assets").join("sha256");
    assert!(!assets.exists() || fs::read_dir(assets)?.next().is_none());
    Ok(())
}
