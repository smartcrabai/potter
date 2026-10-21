use std::{collections::BTreeSet, error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_json(command: &mut Command, description: &str) -> Result<Value, Box<dyn Error>> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "{description} failed\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(response["ok"], true, "{description}: {response}");
    Ok(response)
}

fn apply(scene: &Path, operations: &Value) -> Result<(), Box<dyn Error>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let batch_path = scene.with_extension("ocean-cache-operations.json");
    fs::write(
        &batch_path,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": document["revision"],
            "operations": operations,
        }))?,
    )?;
    run_json(
        pot()
            .args(["apply"])
            .arg(scene)
            .args(["--file"])
            .arg(batch_path)
            .args(["--json"]),
        "apply Ocean project operations",
    )?;
    Ok(())
}

fn inspect_surface(scene: &Path, frame: &str) -> Result<Value, Box<dyn Error>> {
    let response = run_json(
        pot().args(["inspect"]).arg(scene).args([
            "--id",
            "ocean_surface",
            "--frame",
            frame,
            "--json",
        ]),
        "inspect evaluated Ocean mesh",
    )?;
    Ok(response["result"]["items"][0]["evaluated_geometry"]["positions"].clone())
}

fn preview_image(scene: &Path, output: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let response = run_json(
        pot()
            .args(["preview"])
            .arg(scene)
            .args(["--views", "iso", "--size", "64", "--frame", "1", "--out"])
            .arg(output)
            .args(["--json"]),
        "render Ocean preview",
    )?;
    let image = response["result"]["previews"][0]["image"]
        .as_str()
        .ok_or("preview response did not contain an image path")?;
    let bytes = fs::read(image)?;
    assert!(!bytes.is_empty(), "preview image is empty");
    Ok(bytes)
}

fn render_image(scene: &Path, output: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    run_json(
        pot()
            .args(["render"])
            .arg(scene)
            .args([
                "--camera",
                "ocean_camera",
                "--frames",
                "1:1",
                "--format",
                "png",
                "--out",
            ])
            .arg(output)
            .args(["--json"]),
        "render Ocean frame",
    )?;
    let bytes = fs::read(output.join("frame_0001.png"))?;
    assert!(!bytes.is_empty(), "rendered image is empty");
    Ok(bytes)
}

fn export_obj(scene: &Path, output: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    run_json(
        pot()
            .args(["export"])
            .arg(scene)
            .args(["--format", "obj", "--allow-lossy", "--out"])
            .arg(output)
            .args(["--frame", "1", "--json"]),
        "export evaluated Ocean mesh",
    )?;
    let bytes = fs::read(output)?;
    assert!(
        String::from_utf8_lossy(&bytes)
            .lines()
            .any(|line| line.starts_with("v ")),
        "OBJ export contains no mesh vertices"
    );
    Ok(bytes)
}

fn ocean_modifier(document: &Value) -> Result<&Value, Box<dyn Error>> {
    document["nodes"]["ocean_surface"]["modifiers"]
        .as_array()
        .and_then(|modifiers| {
            modifiers
                .iter()
                .find(|modifier| modifier["id"] == "ocean_wave")
        })
        .ok_or_else(|| "Ocean modifier is missing from persisted project".into())
}

fn cached_ocean_params(choppiness: f64) -> Value {
    json!({
        "geometry_mode":"GENERATE",
        "resolution":4,
        "viewport_resolution":3,
        "spatial_size":12,
        "time":9.25,
        "choppiness":choppiness,
        "use_foam":true,
        "use_spray":true,
        "use_normals":true,
        "foam_layer_name":"foam",
        "spray_layer_name":"spray",
        "frame_start":1,
        "frame_end":2,
        "filepath":"//ocean-cache"
    })
}
#[test]
fn ocean_bake_cache_survives_reopen_and_is_used_by_cli_consumers() -> Result<(), Box<dyn Error>> {
    let temporary = tempdir()?;
    let project = temporary.path().join("ocean-project");
    let mut initialize = pot();
    run_json(
        initialize.args(["init"]).arg(&project).args(["--json"]),
        "initialize Ocean project",
    )?;
    apply(
        &project,
        &json!([
            {"op":"node.create","id":"ocean_surface","kind":"plane","params":{"size":2.0}},
            {
                "op":"modifier.create",
                "target":{"id":"ocean_surface"},
                "id":"ocean_wave",
                "type":"ocean",
                "params":{
                    "geometry_mode":"GENERATE",
                    "resolution":4,
                    "viewport_resolution":3,
                    "spatial_size":12,
                    "time":0.75,
                    "choppiness":1.0,
                    "use_foam":true,
                    "use_spray":true,
                    "use_normals":true,
                    "foam_layer_name":"foam",
                    "spray_layer_name":"spray",
                    "frame_start":1,
                    "frame_end":2,
                    "filepath":"//ocean-cache"
                }
            },
            {"op":"camera.create","id":"ocean_camera","projection":"orthographic","ortho_scale":14.0,"transform":{"translation":[0.0,0.0,20.0]}},
            {"op":"light.create","id":"ocean_light","light_type":"sun","energy":3.0},
            {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"ocean_camera"}},
            {"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":32,"resolution_y":32,"samples":1}}
        ]),
    )?;

    let live_positions = inspect_surface(&project, "1")?;
    assert!(
        live_positions
            .as_array()
            .is_some_and(|positions| !positions.is_empty()),
        "live Ocean evaluation returned no vertex positions"
    );
    let _live_preview = preview_image(&project, &temporary.path().join("preview-live"))?;
    let _live_render = render_image(&project, &temporary.path().join("render-live"))?;
    let _live_export = export_obj(&project, &temporary.path().join("live.obj"))?;

    let before_bake: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let previous_revision = before_bake["revision"]
        .as_u64()
        .ok_or("project revision is missing before Ocean bake")?;
    let report = temporary.path().join("ocean-bake-report");
    run_json(
        pot()
            .args(["bake"])
            .arg(&project)
            .args(["--kind", "ocean", "--target", "ocean_wave", "--out"])
            .arg(&report)
            .args(["--json"]),
        "bake Ocean cache through CLI",
    )?;

    let baked_document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert!(
        baked_document["revision"]
            .as_u64()
            .is_some_and(|revision| revision > previous_revision),
        "Ocean bake did not commit the project"
    );
    let cache = &ocean_modifier(&baked_document)?["baked_ocean_cache"];
    assert!(cache.is_object(), "Ocean cache metadata was not persisted");
    assert_eq!(cache["frame_start"], 1);
    assert_eq!(cache["frame_end"], 2);
    assert_eq!(cache["resolution"], 16);
    let manifest = cache["manifest"]
        .as_object()
        .ok_or("persisted Ocean cache file manifest is missing")?;
    let expected_files = (1..=2)
        .flat_map(|frame| {
            ["disp", "foam", "spray", "spray_inverse", "normal"]
                .map(move |channel| format!("{channel}_{frame:04}.exr"))
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        manifest.keys().cloned().collect::<BTreeSet<_>>(),
        expected_files,
        "persisted cache manifest must include every enabled Ocean channel and frame"
    );
    let cache_directory = project.join("ocean-cache");
    for (filename, expected_hash) in manifest {
        let bytes = fs::read(cache_directory.join(filename.as_str()))?;
        assert_eq!(
            bytes.get(..4),
            Some(&[0x76, 0x2f, 0x31, 0x01][..]),
            "{filename} is not an OpenEXR file"
        );
        let exr =
            exr::prelude::read_first_flat_layer_from_file(cache_directory.join(filename.as_str()))?;
        assert_eq!(exr.layer_data.size.width(), 16);
        assert_eq!(exr.layer_data.size.height(), 16);
        let actual_hash = potter_core::hash::sha256(&bytes);
        assert_eq!(expected_hash.as_str(), Some(actual_hash.as_str()));
    }
    let report_manifest: Value = serde_json::from_slice(&fs::read(report.join("manifest.json"))?)?;
    assert_eq!(report_manifest["kind"], "ocean");
    assert_eq!(report_manifest["modifier_id"], "ocean_wave");
    assert_eq!(
        report_manifest["files"]
            .as_object()
            .map(|files| files.keys().cloned().collect::<BTreeSet<_>>()),
        Some(expected_files.clone())
    );

    let reopened_positions = inspect_surface(&project, "1")?;
    // Blender samples a baked cache at texel centres (`BKE_ocean_cache_eval_uv`, `texel = uv * res - 0.5`),
    // not at the live grid corners (`BKE_ocean_eval_uv`), so cached positions are not live positions. The
    // cached sampling rule is pinned by `cached_sampling_is_blenders_cache_eval_uv` in `geom/modifiers/ocean.rs`.
    // Reopened cached positions are checked against later cached reads below.
    apply(
        &project,
        &json!([{
            "op":"modifier.update",
            "target":{"id":"ocean_surface"},
            "id":"ocean_wave",
            "set":{"params":{
                "geometry_mode":"GENERATE",
                "resolution":4,
                "viewport_resolution":3,
                "spatial_size":12,
                "time":9.25,
                "choppiness":1.0,
                "use_foam":true,
                "use_spray":true,
                "use_normals":true,
                "foam_layer_name":"foam",
                "spray_layer_name":"spray",
                "frame_start":1,
                "frame_end":2,
                "filepath":"//ocean-cache"
            }}
        }]),
    )?;
    let updated_document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let updated_modifier = ocean_modifier(&updated_document)?;
    assert_eq!(updated_modifier["params"]["time"], 9.25);
    assert_eq!(updated_modifier["baked_ocean_cache"], *cache);

    let cached_positions = inspect_surface(&project, "1")?;
    assert_eq!(
        cached_positions, reopened_positions,
        "changed-time cached evaluation differs from reopened baked frame"
    );
    preview_image(&project, &temporary.path().join("preview-cached"))?;
    render_image(&project, &temporary.path().join("render-cached"))?;
    export_obj(&project, &temporary.path().join("cached.obj"))?;
    apply(
        &project,
        &json!([{
            "op":"modifier.update",
            "target":{"id":"ocean_surface"},
            "id":"ocean_wave",
            "set":{"params":cached_ocean_params(2.0)}
        }]),
    )?;
    let nonzero_choppiness_document: Value =
        serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert_eq!(
        ocean_modifier(&nonzero_choppiness_document)?["baked_ocean_cache"],
        *cache,
        "changing Ocean choppiness without crossing zero preserves Blender's baked cache"
    );
    assert_eq!(
        inspect_surface(&project, "1")?,
        cached_positions,
        "nonzero choppiness playback continues using cached frame output"
    );

    apply(
        &project,
        &json!([{
            "op":"modifier.update",
            "target":{"id":"ocean_surface"},
            "id":"ocean_wave",
            "set":{"params":cached_ocean_params(0.0)}
        }]),
    )?;
    let crossed_zero_document: Value =
        serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert!(
        ocean_modifier(&crossed_zero_document)?["baked_ocean_cache"].is_null(),
        "crossing Ocean choppiness to zero invalidates its baked cache"
    );
    assert_ne!(
        inspect_surface(&project, "1")?,
        cached_positions,
        "Ocean evaluates live after the choppiness cache invalidation crossing"
    );
    Ok(())
}
