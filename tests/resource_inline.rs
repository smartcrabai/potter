#![expect(
    clippy::unwrap_used,
    reason = "integration tests use fixed valid scene fixtures"
)]

use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn init(path: &Path) -> Result<(), Box<dyn Error>> {
    let output = pot().arg("init").arg(path).arg("--json").output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn apply(
    scene: &Path,
    revision: u64,
    operations: &Value,
) -> Result<std::process::Output, Box<dyn Error>> {
    let directory = tempdir()?;
    let batch = directory.path().join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(
            &json!({"schema_version":1,"base_revision":revision,"operations":operations}),
        )?,
    )?;
    Ok(pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .arg("--json")
        .output()?)
}

fn read_scene(scene: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(
        scene.join("scene.json"),
    )?)?)
}

#[expect(
    clippy::too_many_lines,
    reason = "the fixture covers inline and binary resources through library copying"
)]
#[test]
fn inline_texture_coexists_with_binary_resources_and_library_copy() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let source = directory.path().join("source");
    let target = directory.path().join("target");
    let binary_path = directory.path().join("binary.bin");
    let unpack_path = directory.path().join("inline-unpacked.bin");
    fs::write(&binary_path, [0_u8, 1, 127, 255])?;
    init(&source)?;

    let packed = apply(
        &source,
        0,
        &json!([{"op":"resource.pack","id":"packed_binary","uri":binary_path}]),
    )?;
    assert!(
        packed.status.success(),
        "{}",
        String::from_utf8_lossy(&packed.stdout)
    );

    let inline = json!({
        "storage":"inline",
        "kind":"legacy_texture",
        "texture_type":"BLEND",
        "settings":{"progression":"LINEAR","use_flip_axis":"HORIZONTAL"}
    });
    let mut source_document = read_scene(&source)?;
    source_document["resources"]["inline_texture"] = inline.clone();
    let sequence_bytes = [9_u8, 8, 7];
    source_document["resources"]["linked_cache"] = json!({
        "uri":"cache_sequence.abc",
        "kind":"alembic",
        "packed":true,
        "sequence_start":4,
        "sequence_end":4,
        "sequence_frames":{
            "4":{
                "uri":"cache_sequence_0004.abc",
                "kind":"alembic_cache",
                "hash":potter_core::hash::sha256(&sequence_bytes),
                "packed":true,
                "bytes":sequence_bytes
            }
        }
    });
    fs::write(
        source.join("scene.json"),
        serde_json::to_vec_pretty(&source_document)?,
    )?;

    let listing = pot()
        .arg("assets")
        .arg(&source)
        .arg("--check")
        .arg("--json")
        .output()?;
    assert!(
        listing.status.success(),
        "{}",
        String::from_utf8_lossy(&listing.stdout)
    );
    let listing: Value = serde_json::from_slice(&listing.stdout)?;
    let assets = listing["result"]["assets"].as_array().unwrap();
    assert!(assets.iter().any(|asset| asset["owner"] == "packed_binary"));
    assert!(
        !assets
            .iter()
            .any(|asset| asset["owner"] == "inline_texture")
    );
    assert!(assets.iter().any(|asset| {
        asset["owner"] == "linked_cache"
            && asset["sequence_frame"] == 4
            && asset["kind"] == "alembic_cache"
    }));
    assert_eq!(listing["result"]["summary"]["missing"], 0);
    assert_eq!(listing["result"]["summary"]["changed"], 0);

    let pack_refusal = apply(
        &source,
        1,
        &json!([{"op":"resource.pack","id":"inline_texture","uri":binary_path}]),
    )?;
    assert!(!pack_refusal.status.success());
    let pack_error: Value = serde_json::from_slice(&pack_refusal.stdout)?;
    assert_eq!(pack_error["error"]["code"], "INVALID_OPERATION");
    assert_eq!(read_scene(&source)?["resources"]["inline_texture"], inline);

    let unpack_refusal = apply(
        &source,
        1,
        &json!([{"op":"resource.unpack","id":"inline_texture","path":unpack_path}]),
    )?;
    assert!(!unpack_refusal.status.success());
    let unpack_error: Value = serde_json::from_slice(&unpack_refusal.stdout)?;
    assert_eq!(unpack_error["error"]["code"], "INVALID_OPERATION");
    assert_eq!(read_scene(&source)?["resources"]["inline_texture"], inline);
    assert!(!unpack_path.exists());

    init(&target)?;
    let linked = apply(
        &target,
        0,
        &json!([{
            "op":"library.link",
            "id":"source_library",
            "uri":source,
            "items":[
                {"registry":"resources","id":"inline_texture"},
                {"registry":"resources","id":"linked_cache"}
            ]
        }]),
    )?;
    assert!(
        linked.status.success(),
        "{}",
        String::from_utf8_lossy(&linked.stdout)
    );
    let target_document = read_scene(&target)?;
    assert_eq!(
        target_document["resources"]["source_library__inline_texture"],
        inline
    );
    assert_eq!(
        target_document["resources"]["source_library__linked_cache"],
        source_document["resources"]["linked_cache"]
    );
    Ok(())
}
#[expect(
    clippy::too_many_lines,
    reason = "the sequence fixture checks reports, hashes, and incomplete packed ranges"
)]
#[test]
fn asset_report_checks_each_sequence_leaf_and_rejects_missing_packed_frames()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("sequence");
    init(&scene)?;
    let master = scene.join("master.abc");
    let first_path = scene.join("frame-1.abc");
    let second_path = scene.join("frame-2.abc");
    let master_uri = master.to_string_lossy().into_owned();
    let first_uri = first_path.to_string_lossy().into_owned();
    let second_uri = second_path.to_string_lossy().into_owned();
    let missing_path = scene.join("missing-frame.abc");
    let missing_uri = missing_path.to_string_lossy().into_owned();
    let first_bytes = [1_u8, 3, 5];
    let second_bytes = [2_u8, 4, 6];
    fs::write(&master, b"container master must not be read as a leaf")?;
    fs::write(&first_path, &first_bytes)?;
    fs::write(&second_path, &second_bytes)?;

    let mut document = read_scene(&scene)?;
    document["resources"]["cache_sequence"] = json!({
        "uri":master_uri.clone(),
        "kind":"alembic",
        "packed":false,
        "sequence_start":1,
        "sequence_end":3,
        "sequence_frames":{
            "1":{
                "uri":first_uri,
                "kind":"alembic",
                "hash":potter_core::hash::sha256(&first_bytes),
                "packed":false
            },
            "2":{
                "uri":second_uri.clone(),
                "kind":"alembic_cache",
                "hash":potter_core::hash::sha256(&first_bytes),
                "packed":true,
                "bytes":second_bytes
            },
            "3":{
                "uri":missing_uri,
                "kind":"alembic_cache",
                "hash":potter_core::hash::sha256(&first_bytes),
                "packed":false
            }
        }
    });
    document["resources"]["packed_sequence"] = json!({
        "uri":master_uri.clone(),
        "kind":"alembic",
        "packed":true,
        "sequence_start":1,
        "sequence_end":2,
        "sequence_frames":{
            "1":{
                "uri":"frame-1.abc",
                "kind":"alembic",
                "hash":potter_core::hash::sha256(&first_bytes),
                "packed":true,
                "bytes":first_bytes
            },
            "2":{
                "uri":"frame-2.abc",
                "kind":"alembic",
                "hash":potter_core::hash::sha256(&second_bytes),
                "packed":true,
                "bytes":second_bytes
            }
        }
    });
    fs::write(
        scene.join("scene.json"),
        serde_json::to_vec_pretty(&document)?,
    )?;

    let listing = pot()
        .arg("assets")
        .arg(&scene)
        .arg("--check")
        .arg("--json")
        .output()?;
    assert!(
        listing.status.success(),
        "{}",
        String::from_utf8_lossy(&listing.stdout)
    );
    let listing: Value = serde_json::from_slice(&listing.stdout)?;
    let assets = listing["result"]["assets"]
        .as_array()
        .ok_or("asset report has no assets array")?;
    let sequence_assets = assets
        .iter()
        .filter(|asset| asset["owner"] == "cache_sequence")
        .collect::<Vec<_>>();
    assert_eq!(sequence_assets.len(), 3);
    assert!(
        sequence_assets
            .iter()
            .all(|asset| asset["uri"].as_str() != Some(master_uri.as_str()))
    );
    assert!(sequence_assets.iter().any(|asset| {
        asset["sequence_frame"] == 1 && asset["kind"] == "alembic" && asset["changed"] == false
    }));
    assert!(sequence_assets.iter().any(|asset| {
        asset["sequence_frame"] == 2 && asset["kind"] == "alembic_cache" && asset["changed"] == true
    }));
    assert!(sequence_assets.iter().any(|asset| {
        asset["sequence_frame"] == 3 && asset["kind"] == "alembic_cache" && asset["missing"] == true
    }));
    assert_eq!(
        assets
            .iter()
            .filter(|asset| asset["owner"] == "packed_sequence")
            .count(),
        2
    );
    assert_eq!(listing["result"]["summary"]["missing"], 1);
    assert_eq!(listing["result"]["summary"]["changed"], 1);

    document["resources"]["packed_sequence"]["sequence_frames"]
        .as_object_mut()
        .ok_or("sequence frames are not an object")?
        .remove("2");
    fs::write(
        scene.join("scene.json"),
        serde_json::to_vec_pretty(&document)?,
    )?;
    let incomplete = pot()
        .arg("assets")
        .arg(&scene)
        .arg("--check")
        .arg("--json")
        .output()?;
    assert!(!incomplete.status.success());
    let error: Value = serde_json::from_slice(&incomplete.stdout)?;
    assert_eq!(error["error"]["code"], "SCENE_INVALID");
    Ok(())
}

#[test]
fn image_delete_uses_only_explicit_image_texture_resource_references() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let scene = directory.path().join("image-refs");
    init(&scene)?;
    let created = apply(
        &scene,
        0,
        &json!([{"op":"image.create","id":"shared_image","width":1,"height":1}]),
    )?;
    assert!(created.status.success());
    let mut document = read_scene(&scene)?;
    document["resources"]["opaque_payload"] = json!({
        "kind":"binary",
        "image":"shared_image"
    });
    fs::write(
        scene.join("scene.json"),
        serde_json::to_vec_pretty(&document)?,
    )?;
    let deleted = apply(
        &scene,
        1,
        &json!([{"op":"image.delete","id":"shared_image"}]),
    )?;
    assert!(
        deleted.status.success(),
        "{}",
        String::from_utf8_lossy(&deleted.stdout)
    );
    assert!(read_scene(&scene)?["images"].get("shared_image").is_none());

    let recreated = apply(
        &scene,
        2,
        &json!([{"op":"image.create","id":"shared_image","width":1,"height":1}]),
    )?;
    assert!(recreated.status.success());
    let mut document = read_scene(&scene)?;
    document["resources"]["image_texture"] = json!({
        "storage":"inline",
        "kind":"legacy_texture",
        "texture_type":"IMAGE",
        "image":"shared_image",
        "settings":{}
    });
    fs::write(
        scene.join("scene.json"),
        serde_json::to_vec_pretty(&document)?,
    )?;
    let refused = apply(
        &scene,
        3,
        &json!([{"op":"image.delete","id":"shared_image"}]),
    )?;
    assert!(!refused.status.success());
    let error: Value = serde_json::from_slice(&refused.stdout)?;
    assert_eq!(error["error"]["code"], "INVALID_OPERATION");
    let references = error["error"]["details"]["resources"]
        .as_array()
        .ok_or("image deletion error has no resource reference list")?;
    assert_eq!(references.as_slice(), &[json!("image_texture")]);
    assert!(read_scene(&scene)?["images"].get("shared_image").is_some());
    Ok(())
}
