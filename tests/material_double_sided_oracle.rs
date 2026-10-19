//! Blender 5.2.2 oracle for the `double_sided` value of imported materials.
//!
//! Blender builds a one-triangle scene with a default material and with a single-sided material
//! (`use_backface_culling` on), exports it with each format's exporter, re-imports the file with
//! that format's importer, and reports the culling of the imported material. Potter imports the
//! same file and must hold `double_sided == !use_backface_culling`. Blender's new-material default
//! is reported too, and Potter's `Material::default()` must match it.
//!
//! Skips without Blender, as the other Blender-gated tests do. `POTTER_REQUIRE_BLENDER` turns the
//! skip into a failure.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use std::{
    error::Error,
    fs, io,
    path::Path,
    process::{Command, Output},
};

use blender_checked::blender_executable;
use serde_json::Value;
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Exchange formats with the file name Blender writes for each. FBX is not checked: Blender's FBX
/// exporter writes binary FBX, which Potter's importer rejects (`format.fbx_binary`), so Potter
/// cannot read Blender's FBX output. FBX round trips are covered by `material_double_sided.rs`.
const FORMATS: [(&str, &str); 3] = [("glb", "box.glb"), ("obj", "box.obj"), ("usda", "box.usda")];
/// `double`: Blender's default material. `single`: `use_backface_culling` on.
const VARIANTS: [&str; 2] = ["double", "single"];

/// Blender side: export each variant with each exporter, re-import it with the matching importer,
/// and record the culling of every imported material except the factory `Dots Stroke`.
const BLENDER_SCRIPT: &str = r#"
import json
import os
import sys

import bpy

out_dir, out_path = sys.argv[sys.argv.index("--") + 1 :][:2]
FILES = {"glb": "box.glb", "obj": "box.obj", "usda": "box.usda"}


def exporter(fmt, path):
    if fmt == "glb":
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB")
    elif fmt == "obj":
        bpy.ops.wm.obj_export(filepath=path)
    else:
        bpy.ops.wm.usd_export(filepath=path)


def importer(fmt, path):
    if fmt == "glb":
        bpy.ops.import_scene.gltf(filepath=path)
    elif fmt == "obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.wm.usd_import(filepath=path)


def build(single_sided):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    material = bpy.data.materials.new("Probe")
    if single_sided:
        material.use_backface_culling = True
    mesh = bpy.data.meshes.new("Tri")
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    mesh.update()
    mesh.materials.append(material)
    obj = bpy.data.objects.new("Tri", mesh)
    bpy.context.collection.objects.link(obj)


bpy.ops.wm.read_factory_settings(use_empty=True)
result = {"new_material": bpy.data.materials.new("Fresh").use_backface_culling}
for variant, single_sided in (("double", False), ("single", True)):
    for fmt, name in FILES.items():
        build(single_sided)
        path = os.path.join(out_dir, variant + "-" + name)
        exporter(fmt, path)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        importer(fmt, path)
        culling = [m.use_backface_culling for m in bpy.data.materials if m.name != "Dots Stroke"]
        result.setdefault(fmt, {})[variant] = culling
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"#;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn text(path: &Path) -> TestResult<&str> {
    path.to_str()
        .ok_or_else(|| format!("{} is not UTF-8", path.display()).into())
}

fn ok_json(output: &Output) -> TestResult<Value> {
    if !output.status.success() {
        return Err(format!("pot failed: {}", String::from_utf8_lossy(&output.stdout)).into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn init(scene: &Path) -> TestResult {
    ok_json(&pot().args(["init", text(scene)?, "--json"]).output()?)?;
    Ok(())
}

fn import_replace(scene: &Path, file: &Path, format: &str) -> TestResult {
    ok_json(
        &pot()
            .args(["import", text(scene)?, "--file"])
            .arg(file)
            .args([
                "--format",
                format,
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--json",
            ])
            .output()?,
    )?;
    Ok(())
}

/// Runs the Blender script in `root` and returns the culling it reports per format and variant.
fn blender_reference(blender: &Path, root: &Path) -> TestResult<Value> {
    let out_dir = root.join("blender");
    fs::create_dir_all(&out_dir)?;
    let script = root.join("double_sided.py");
    let out_path = root.join("blender.json");
    fs::write(&script, BLENDER_SCRIPT)?;
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "3",
            "--python",
        ])
        .arg(&script)
        .arg("--")
        .arg(&out_dir)
        .arg(&out_path)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "blender double-sided probe failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(serde_json::from_slice(&fs::read(&out_path)?)?)
}

/// The `double_sided` Potter gives the single material of `file` imported as `format`.
fn potter_double_sided(file: &Path, format: &str) -> TestResult<bool> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    import_replace(&scene, file, format)?;
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let values = document["materials"]
        .as_object()
        .ok_or("scene has no materials")?
        .values()
        .map(|material| {
            material["double_sided"]
                .as_bool()
                .ok_or("material has no explicit double_sided")
        })
        .collect::<Result<Vec<_>, _>>()?;
    match values.as_slice() {
        [value] => Ok(*value),
        _ => Err(format!("{format} imported {} materials, expected 1", values.len()).into()),
    }
}

#[test]
fn imported_double_sided_matches_blender_culling() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let reference = blender_reference(&blender, directory.path())?;
    let fresh = reference["new_material"]
        .as_bool()
        .ok_or("Blender reported no culling for a new material")?;
    assert_eq!(
        potter_core::model::Material::default().double_sided,
        !fresh,
        "a new material's double_sided must match Blender's use_backface_culling"
    );
    for (format, name) in FORMATS {
        for variant in VARIANTS {
            let culling = reference[format][variant]
                .as_array()
                .ok_or_else(|| format!("Blender gave no {format} {variant} materials"))?;
            let [culling] = culling.as_slice() else {
                return Err(format!(
                    "Blender imported {} {format} {variant} materials, expected 1",
                    culling.len()
                )
                .into());
            };
            let culling = culling
                .as_bool()
                .ok_or_else(|| format!("{format} {variant} culling is not a boolean"))?;
            let file = directory
                .path()
                .join("blender")
                .join(format!("{variant}-{name}"));
            assert_eq!(
                potter_double_sided(&file, format)?,
                !culling,
                "{format} {variant}: Potter's import disagrees with Blender's use_backface_culling"
            );
        }
    }
    Ok(())
}
