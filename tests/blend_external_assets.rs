#![expect(clippy::unwrap_used, reason = "Blender integration fixture setup")]

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;

use potter_core::hash;
use serde_json::{Value, json};
use tempfile::tempdir;

const FIXTURE_SCRIPT: &str = r#"
import bpy
import json
import os
import shutil
import sys
import numpy as np
import openvdb

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
assets = os.path.join(root, "assets")
os.makedirs(assets, exist_ok=True)

vdb_path = os.path.join(assets, "cloud.vdb")
dense = np.zeros((16, 16, 16), dtype=np.float32)
coordinates = np.indices(dense.shape) - 7.5
dense[(coordinates**2).sum(axis=0) < 36] = 1.0
grid = openvdb.FloatGrid()
grid.copyFromArray(dense)
grid.name = "density"
openvdb.write(vdb_path, grids=[grid])

font_candidates = [
    os.environ.get("POTTER_TEST_FONT"),
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
font_source = next((path for path in font_candidates if path and os.path.isfile(path)), None)
if font_source is None:
    raise RuntimeError("no external test font is available")
font_path = os.path.join(assets, os.path.basename(font_source))
shutil.copyfile(font_source, font_path)

movie_path = os.path.join(assets, "clip.mp4")
render_scene = bpy.data.scenes.new("MovieSource")
render_scene.render.engine = "BLENDER_WORKBENCH"
render_scene.render.resolution_x = 64
render_scene.render.resolution_y = 64
render_scene.frame_start, render_scene.frame_end = 1, 8
settings = render_scene.render.image_settings
if hasattr(settings, "media_type"):
    settings.media_type = "VIDEO"
settings.file_format = "FFMPEG"
render_scene.render.ffmpeg.format = "MPEG4"
render_scene.render.ffmpeg.codec = "H264"
render_scene.render.filepath = movie_path
camera_data = bpy.data.cameras.new("MovieCam")
camera = bpy.data.objects.new("MovieCam", camera_data)
render_scene.collection.objects.link(camera)
render_scene.camera = camera
bpy.ops.render.render(animation=True, scene=render_scene.name)
produced = [name for name in os.listdir(assets) if name.startswith("clip")]
if movie_path not in [os.path.join(assets, name) for name in produced]:
    os.rename(os.path.join(assets, produced[0]), movie_path)
bpy.data.scenes.remove(render_scene)
bpy.data.objects.remove(camera)
bpy.data.cameras.remove(camera_data)

scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)
volume_data = bpy.data.volumes.new("CloudVolume")
volume_data.filepath = "//assets/cloud.vdb"
volume = bpy.data.objects.new("Cloud", volume_data)
volume.location = (2.0, 0.0, 1.0)
scene.collection.objects.link(volume)
font = bpy.data.fonts.load(font_path)
font.filepath = "//assets/" + os.path.basename(font_path)
text_data = bpy.data.curves.new("TitleText", type="FONT")
text_data.body = "potter"
text_data.font = font
text_data.size = 0.5
text = bpy.data.objects.new("Title", text_data)
scene.collection.objects.link(text)
editor = scene.sequence_editor_create()
strips = editor.strips if hasattr(editor, "strips") else editor.sequences
strip = strips.new_movie("Clip", movie_path, 1, 1)
strip.filepath = "//assets/clip.mp4"
blend_path = os.path.join(root, "scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path, relative_remap=True)
volume_data.grids.load()
expected = {
    "objects": sorted((obj.name, obj.type) for obj in bpy.data.objects),
    "volume": {"filepath": volume_data.filepath,
               "grids": [item.name for item in volume_data.grids],
               "location": list(volume.location)},
    "text": {"body": text_data.body, "font": text_data.font.name,
             "font_filepath": text_data.font.filepath},
    "strip": {"name": strip.name, "type": strip.type,
              "filepath": strip.filepath,
              "frame_final_duration": strip.frame_final_duration},
    "asset_names": sorted(os.listdir(assets)),
}
with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as handle:
    json.dump(expected, handle, indent=1)
print("FIXTURE_OK", json.dumps(expected))
"#;

const VARIANT_SCRIPT: &str = r#"
import bpy
import sys
keep = sys.argv[sys.argv.index("--") + 1]
out = sys.argv[sys.argv.index("--") + 2]
scene = bpy.context.scene
if keep != "volume":
    bpy.data.objects.remove(bpy.data.objects["Cloud"])
if keep != "text":
    bpy.data.objects.remove(bpy.data.objects["Title"])
if keep != "movie":
    scene.sequence_editor_clear()
bpy.data.orphans_purge(do_recursive=True)
bpy.ops.wm.save_as_mainfile(filepath=out, relative_remap=True)
print("VARIANT_OK", keep)
"#;

const REOPEN_SCRIPT: &str = r#"
import bpy
import json
import os
import sys
output = sys.argv[sys.argv.index("--") + 1]
objects = sorted((obj.name, obj.type) for obj in bpy.data.objects)
volumes = []
for volume in bpy.data.volumes:
    try:
        volume.grids.load()
    except Exception:
        pass
    volumes.append({"name": volume.name, "filepath": volume.filepath,
                    "resolved": bpy.path.abspath(volume.filepath),
                    "grids": [grid.name for grid in volume.grids]})
texts = []
for curve in bpy.data.curves:
    if curve.bl_rna.identifier == "TextCurve":
        font = curve.font
        texts.append({"name": curve.name, "body": curve.body,
                      "font": font.name if font else None,
                      "font_filepath": font.filepath if font else None,
                      "font_resolved": bpy.path.abspath(font.filepath) if font else None})
strips = []
for scene in bpy.data.scenes:
    editor = scene.sequence_editor
    sequences = (getattr(editor, "strips", None) or getattr(editor, "sequences", ())) if editor else ()
    for strip in sequences:
        strips.append({"name": strip.name, "type": strip.type,
                       "filepath": getattr(strip, "filepath", ""),
                       "resolved": bpy.path.abspath(getattr(strip, "filepath", "") or ""),
                       "duration": float(strip.frame_final_duration),
                       "channel": int(strip.channel),
                       "mute": bool(strip.mute),
                       "blend_type": str(strip.blend_type).lower()})
with open(output, "w", encoding="utf-8") as handle:
    json.dump({"objects": objects, "volumes": volumes, "texts": texts, "strips": strips}, handle)
print("REOPEN_OK")
"#;

fn run_ok(command: &mut Command, label: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(output)
}

fn pot_ok(args: &[&str], label: &str) -> Result<Value, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(args).arg("--json");
    let output = run_ok(&mut command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn blender_ok(blender: &Path, args: &[&str], label: &str) -> Result<Output, Box<dyn Error>> {
    let mut command = Command::new(blender);
    command.args(args);
    run_ok(&mut command, label)
}

fn copy_directory(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_directory(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn make_fixture(blender: &Path, root: &Path) -> Result<Value, Box<dyn Error>> {
    let script = root.join("make_fixture.py");
    fs::write(&script, FIXTURE_SCRIPT)?;
    let canonical_root = fs::canonicalize(root)?;
    blender_ok(
        blender,
        &[
            "--background",
            "--factory-startup",
            "--python",
            script.to_str().unwrap(),
            "--",
            canonical_root.to_str().unwrap(),
        ],
        "fixture generation",
    )?;
    Ok(serde_json::from_slice(&fs::read(
        root.join("expected.json"),
    )?)?)
}

fn make_case_blend(
    blender: &Path,
    source_root: &Path,
    case_root: &Path,
    kind: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(case_root)?;
    copy_directory(&source_root.join("assets"), &case_root.join("assets"))?;
    let source_scene = source_root.join("scene.blend");
    if kind == "combined" {
        let destination = case_root.join("scene.blend");
        fs::copy(source_scene, &destination)?;
        return Ok(fs::canonicalize(destination)?);
    }
    let source = case_root.join("source.blend");
    fs::copy(source_scene, &source)?;
    let script = case_root.join("variant.py");
    fs::write(&script, VARIANT_SCRIPT)?;
    let output = case_root.join(format!("only_{kind}.blend"));
    blender_ok(
        blender,
        &[
            "--background",
            "--factory-startup",
            source.to_str().unwrap(),
            "--python",
            script.to_str().unwrap(),
            "--",
            kind,
            output.to_str().unwrap(),
        ],
        "variant generation",
    )?;
    Ok(fs::canonicalize(output)?)
}

fn expected_resource_kinds(case: &str) -> &'static [&'static str] {
    match case {
        "volume" => &["volume"],
        "movie" => &["movie"],
        "text" => &["font"],
        _ => &["volume", "movie", "font"],
    }
}

fn verify_export(
    blender: &Path,
    output_blend: &Path,
    expected: &Value,
    case: &str,
    case_root: &Path,
) -> Result<(), Box<dyn Error>> {
    let script = case_root.join("reopen.py");
    let result_path = case_root.join("reopened.json");
    fs::write(&script, REOPEN_SCRIPT)?;
    blender_ok(
        blender,
        &[
            "--background",
            "--factory-startup",
            output_blend.to_str().unwrap(),
            "--python",
            script.to_str().unwrap(),
            "--",
            result_path.to_str().unwrap(),
        ],
        "Blender exported-file reopen",
    )?;
    let actual: Value = serde_json::from_slice(&fs::read(result_path)?)?;
    let expected_objects = match case {
        "volume" => vec![json!(["Cloud", "VOLUME"])],
        "movie" => Vec::new(),
        "text" => vec![json!(["Title", "FONT"])],
        _ => vec![json!(["Cloud", "VOLUME"]), json!(["Title", "FONT"])],
    };
    assert_eq!(actual["objects"], json!(expected_objects));
    if case == "volume" || case == "combined" {
        let volume = &actual["volumes"][0];
        assert_eq!(volume["grids"], expected["volume"]["grids"]);
        assert!(Path::new(volume["resolved"].as_str().unwrap()).is_file());
        assert_eq!(
            Path::new(volume["resolved"].as_str().unwrap())
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            "cloud.vdb"
        );
    }
    if case == "text" || case == "combined" {
        let text = &actual["texts"][0];
        assert_eq!(text["body"], expected["text"]["body"]);
        assert_eq!(text["font"], expected["text"]["font"]);
        assert!(Path::new(text["font_resolved"].as_str().unwrap()).is_file());
        assert_eq!(
            Path::new(text["font_resolved"].as_str().unwrap())
                .file_name()
                .unwrap(),
            Path::new(expected["text"]["font_filepath"].as_str().unwrap())
                .file_name()
                .unwrap()
        );
    }
    if case == "movie" || case == "combined" {
        let strip = actual["strips"]
            .as_array()
            .unwrap()
            .iter()
            .find(|strip| strip["type"] == "MOVIE")
            .unwrap();
        assert_eq!(strip["name"], expected["strip"]["name"]);
        assert_eq!(strip["type"], "MOVIE");
        assert!((strip["duration"].as_f64().unwrap() - 8.0).abs() < 0.01);
        assert!(Path::new(strip["resolved"].as_str().unwrap()).is_file());
        assert_eq!(
            Path::new(strip["resolved"].as_str().unwrap())
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            "clip.mp4"
        );
    }
    Ok(())
}

#[test]
fn blender_external_assets_copy_link_and_roundtrip() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender external-asset integration test: Blender is unavailable");
        return Ok(());
    };
    let root = tempdir()?;
    let source_root = root.path().join("source");
    fs::create_dir_all(&source_root)?;
    let expected = make_fixture(&blender, &source_root)?;
    let canonical_source = fs::canonicalize(&source_root)?;
    let cases = ["volume", "movie", "text", "combined"];

    for case in cases {
        let case_root = canonical_source.join(format!("case_{case}"));
        let input_blend = make_case_blend(&blender, &canonical_source, &case_root, case)?;
        let project = case_root.join("project");
        let init = pot_ok(
            &["init", project.to_str().unwrap()],
            "project initialization",
        )?;
        assert_eq!(init["ok"], true);
        let imported = pot_ok(
            &[
                "import",
                project.to_str().unwrap(),
                "--file",
                input_blend.to_str().unwrap(),
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
                blender.to_str().unwrap(),
            ],
            "strict Blender import",
        )?;
        assert_eq!(imported["ok"], true);
        assert_eq!(imported["result"]["losses"], json!([]));
        let imported_resources = imported["result"]["resources"].as_array().unwrap();
        let asset_check = pot_ok(
            &["assets", project.to_str().unwrap(), "--check"],
            "copied asset check",
        )?;
        assert_eq!(asset_check["ok"], true);
        let checked_assets = asset_check["result"]["assets"].as_array().unwrap();
        for kind in expected_resource_kinds(case) {
            let resource = imported_resources
                .iter()
                .find(|resource| resource["kind"] == *kind)
                .unwrap_or_else(|| panic!("{case} import omitted {kind} resource"));
            let uri = resource["uri"].as_str().unwrap();
            assert!(uri.starts_with("assets/sha256/"), "{uri}");
            let original_path = resource["original_path"].as_str().unwrap();
            let original_hash = hash::sha256(&fs::read(original_path)?);
            assert_eq!(resource["hash"], original_hash);
            assert!(project.join(uri).is_file(), "copied asset missing: {uri}");
            let copied_hash = hash::sha256(&fs::read(project.join(uri))?);
            assert_eq!(copied_hash, original_hash);
            assert!(checked_assets.iter().any(|asset| {
                asset["uri"] == uri
                    && asset["hash"] == original_hash
                    && asset["missing"] == false
                    && asset["changed"] == false
            }));
            assert_eq!(
                checked_assets
                    .iter()
                    .filter(|asset| asset["uri"] == uri)
                    .count(),
                1
            );
            assert!(!checked_assets.iter().any(|asset| {
                asset["uri"]
                    .as_str()
                    .is_some_and(|asset_uri| asset_uri.ends_with("/blob"))
                    && asset["hash"] == original_hash
            }));
        }

        if case != "volume" {
            fs::remove_dir_all(case_root.join("assets"))?;
        }
        if case == "text" {
            let preview = pot_ok(
                &[
                    "preview",
                    project.to_str().unwrap(),
                    "--views",
                    "iso",
                    "--size",
                    "64",
                    "--out",
                    case_root.join("preview").to_str().unwrap(),
                ],
                "fallback-font preview",
            )?;
            assert_eq!(preview["ok"], true);
        }
        if case == "volume" {
            let preview = pot_ok(
                &[
                    "preview",
                    project.to_str().unwrap(),
                    "--views",
                    "iso",
                    "--size",
                    "64",
                    "--out",
                    case_root.join("volume_preview").to_str().unwrap(),
                ],
                "volume solid preview",
            )?;
            assert_eq!(preview["ok"], true);
            assert_eq!(preview["warnings"], json!([]));
        }
        let validation = pot_ok(&["validate", project.to_str().unwrap()], "scene validation")?;
        assert_eq!(validation["ok"], true);
        let output_blend = case_root.join("roundtrip.blend");
        let exported = pot_ok(
            &[
                "export",
                project.to_str().unwrap(),
                "--format",
                "blend",
                "--out",
                output_blend.to_str().unwrap(),
                "--blender",
                blender.to_str().unwrap(),
            ],
            "strict Blender export",
        )?;
        assert_eq!(exported["ok"], true);
        assert!(
            !exported["result"]["files"].as_array().unwrap().is_empty(),
            "Blender export produced no files"
        );
        verify_export(&blender, &output_blend, &expected, case, &case_root)?;
    }

    let link_root = canonical_source.join("case_link");
    let input_blend = make_case_blend(&blender, &canonical_source, &link_root, "combined")?;
    let link_project = link_root.join("project");
    pot_ok(
        &["init", link_project.to_str().unwrap()],
        "link project initialization",
    )?;
    let linked = pot_ok(
        &[
            "import",
            link_project.to_str().unwrap(),
            "--file",
            input_blend.to_str().unwrap(),
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--asset-policy",
            "link",
            "--blender",
            blender.to_str().unwrap(),
        ],
        "linked Blender import",
    )?;
    assert_eq!(linked["ok"], true);
    for kind in expected_resource_kinds("combined") {
        let resource = linked["result"]["resources"]
            .as_array()
            .unwrap()
            .iter()
            .find(|resource| resource["kind"] == *kind)
            .unwrap();
        let uri = resource["uri"].as_str().unwrap();
        assert!(Path::new(uri).is_absolute());
        assert_eq!(resource["status"], "available");
    }
    let linked_assets = pot_ok(
        &["assets", link_project.to_str().unwrap(), "--check"],
        "linked asset check",
    )?;
    assert_eq!(linked_assets["result"]["summary"]["missing"], 0);
    Ok(())
}
