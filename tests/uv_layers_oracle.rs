//! Blender oracle for multiple UV layers.
//!
//! Potter imports one `.blend` with three UV layers, a pinned layer, active and render indices that
//! differ from the defaults, and two materials (one with no UV Map node, one whose texture reads
//! `Detail`). It exports the scene to each UV-carrying format. Blender re-imports every Potter file
//! and the file Blender's own exporter wrote from the same scene, and both must agree: the same
//! layers in the same order, with the same values in Blender's V-up convention. Layers are matched
//! by loop, keyed by corner position and face normal, because glTF re-imports as triangles and each
//! importer orders loops its own way. The `.blend` leg also checks pins and the active and render
//! indices. Other formats carry those only as far as the expectations below say.
//!
//! Skips without Blender, as the other Blender-gated tests do. `POTTER_REQUIRE_BLENDER` turns the
//! skip into a failure.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use std::{
    collections::BTreeMap,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use blender_checked::blender_executable;
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Blender-convention tolerance for values and glTF-free comparisons.
const TOLERANCE: f64 = 1.0e-5;
/// Tolerance for store values after a `.blend` round trip (float32 storage in Blender).
const STORE_TOLERANCE: f64 = 1.0e-9;

/// Layers of the fixture in creation order, with the active (`Detail`, index 2) and render
/// (`Light`, index 1) indices that differ from the defaults.
const SOURCE_ACTIVE: i64 = 2;
const SOURCE_RENDER: i64 = 1;

/// Source order: the layers of the fixture as `(name after re-import, fixture layer)`.
const IN_SOURCE_ORDER: &[(&str, &str)] =
    &[("Base", "Base"), ("Light", "Light"), ("Detail", "Detail")];

/// One exported format. `blender` and `potter` list the layers each re-import yields, in order, as
/// `(name after re-import, fixture layer it carries)`. Blender's own exports define `blender`.
struct Case {
    format: &'static str,
    extension: &'static str,
    /// Potter's FBX writer rejects material node trees, so only Potter's FBX export uses
    /// `plain.blend`; Blender's own FBX export uses `textured.blend`.
    source: &'static str,
    allow_lossy: bool,
    /// Layers Potter reports as `mesh.uv_layer` losses for the format.
    lost_layers: usize,
    blender: &'static [(&'static str, &'static str)],
    potter: &'static [(&'static str, &'static str)],
}

const CASES: [Case; 6] = [
    Case {
        format: "glb",
        extension: "glb",
        source: "textured.blend",
        allow_lossy: false,
        lost_layers: 0,
        blender: &[
            ("UVMap", "Base"),
            ("UVMap.001", "Light"),
            ("UVMap.002", "Detail"),
        ],
        potter: &[
            ("UVMap", "Base"),
            ("UVMap.001", "Light"),
            ("UVMap.002", "Detail"),
        ],
    },
    Case {
        format: "usdc",
        extension: "usdc",
        source: "textured.blend",
        allow_lossy: false,
        lost_layers: 0,
        // The render layer is written as `primvars:st`, and Blender re-imports UV maps in
        // case-insensitive name order (probe: a file written `Zeta, Alpha, st` re-imports as
        // `Alpha, st, Zeta`). The render layer comes back as `st`, and the source order is lost.
        blender: &[("Base", "Base"), ("Detail", "Detail"), ("st", "Light")],
        potter: &[("Base", "Base"), ("Detail", "Detail"), ("st", "Light")],
    },
    Case {
        format: "obj",
        extension: "obj",
        source: "textured.blend",
        allow_lossy: true,
        lost_layers: 2,
        // OBJ carries the active layer only, under the importer's default name.
        blender: &[("UVMap", "Detail")],
        potter: &[("UVMap", "Detail")],
    },
    Case {
        format: "ply",
        extension: "ply",
        source: "textured.blend",
        allow_lossy: true,
        lost_layers: 2,
        blender: &[("UVMap", "Detail")],
        potter: &[("UVMap", "Detail")],
    },
    Case {
        format: "alembic",
        extension: "abc",
        source: "textured.blend",
        allow_lossy: true,
        lost_layers: 0,
        // Both exporters put the active layer first. Blender's Alembic importer preserves the
        // exported layer order, so both re-import as `Detail, Base, Light`.
        blender: &[("Detail", "Detail"), ("Base", "Base"), ("Light", "Light")],
        potter: &[("Detail", "Detail"), ("Base", "Base"), ("Light", "Light")],
    },
    Case {
        format: "fbx-binary",
        extension: "fbx",
        source: "plain.blend",
        allow_lossy: true,
        lost_layers: 0,
        blender: IN_SOURCE_ORDER,
        potter: IN_SOURCE_ORDER,
    },
];

/// Building script: writes `textured.blend` (three layers, pins, two materials) and `plain.blend`
/// (the same mesh and layers with no materials) to the two paths after `--`.
const BUILD_SCRIPT: &str = r#"
import sys

import bpy

textured_path, plain_path = sys.argv[sys.argv.index("--") + 1 :][:2]


def build(path, textured):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2)
    mesh = bpy.context.object.data
    mesh.uv_layers[0].name = "Base"
    mesh.uv_layers.new(name="Light")
    mesh.uv_layers.new(name="Detail")
    for layer_index, layer in enumerate(mesh.uv_layers):
        for loop_index, item in enumerate(layer.data):
            item.uv = (
                (0.043 * loop_index + 0.17 * layer_index) % 1.0,
                (0.071 * loop_index + 0.29 * layer_index) % 1.0,
            )
    for loop_index in (0, 5, 11, 17):
        mesh.uv_layers["Light"].data[loop_index].pin_uv = True
    mesh.uv_layers.active_index = 2
    mesh.uv_layers.active_render_index = 1
    if textured:
        for material_name, uv_name in (("Mat_Render", None), ("Mat_Detail", "Detail")):
            material = bpy.data.materials.new(material_name)
            material.use_nodes = True
            tree = material.node_tree
            texture = tree.nodes.new("ShaderNodeTexImage")
            image = bpy.data.images.new(material_name + "_image", 8, 8)
            base = (0.12, 0.19, 0.41) if uv_name is None else (0.58, 0.16, 0.24)
            pixels = []
            for y in range(8):
                for x in range(8):
                    pixels.extend((base[0] + 0.02 * x, base[1] + 0.015 * y,
                                   base[2] + 0.01 * (x + y), 1.0))
            image.pixels = pixels
            texture.image = image
            image.pack()
            tree.links.new(texture.outputs["Color"], tree.nodes["Principled BSDF"].inputs["Base Color"])
            if uv_name is not None:
                uv_map = tree.nodes.new("ShaderNodeUVMap")
                uv_map.uv_map = uv_name
                tree.links.new(uv_map.outputs["UV"], texture.inputs["Vector"])
            mesh.materials.append(material)
        for polygon in mesh.polygons:
            polygon.material_index = 0 if polygon.index < 3 else 1
    bpy.ops.wm.save_as_mainfile(filepath=path)


build(textured_path, True)
build(plain_path, False)
"#;

/// Blender's own exports of every format from `textured.blend`. Potter's FBX export uses the
/// material-free `plain.blend` because Potter's FBX writer rejects non-default material node trees.
/// Arguments after `--`: textured path, output directory.
const EXPORT_SCRIPT: &str = r#"
import os
import sys

import bpy

textured, out_dir = sys.argv[sys.argv.index("--") + 1 :][:2]


def out(name):
    return os.path.join(out_dir, name)


bpy.ops.wm.open_mainfile(filepath=textured)
bpy.ops.export_scene.gltf(filepath=out("blender.glb"), export_format="GLB")
bpy.ops.wm.usd_export(filepath=out("blender.usdc"))
bpy.ops.wm.obj_export(filepath=out("blender.obj"), export_uv=True)
bpy.ops.wm.ply_export(filepath=out("blender.ply"), export_uv=True)
bpy.ops.wm.alembic_export(filepath=out("blender.abc"), uvs=True)
bpy.ops.export_scene.fbx(filepath=out("blender.fbx"))
"#;

/// Dump script: the first argument is the JSON output path, the rest are files. Each file is opened
/// in a fresh session (`.blend` directly, other formats by their importer). Loops are keyed by corner
/// position and face normal in world space.
const DUMP_SCRIPT: &str = r#"
import json
import os
import sys

import bpy

out_path, sources = sys.argv[sys.argv.index("--") + 1], sys.argv[sys.argv.index("--") + 2 :]
IMPORTERS = {
    ".glb": bpy.ops.import_scene.gltf,
    ".usdc": bpy.ops.wm.usd_import,
    ".fbx": bpy.ops.import_scene.fbx,
    ".obj": bpy.ops.wm.obj_import,
    ".ply": bpy.ops.wm.ply_import,
    ".abc": bpy.ops.wm.alembic_import,
}


def fmt(value):
    return "%.4f" % (round(value, 4) + 0.0)


def dump_scene():
    meshes = [ob for ob in bpy.context.scene.objects if ob.type == "MESH" and len(ob.data.uv_layers)]
    if len(meshes) != 1:
        raise RuntimeError("expected one mesh with UV layers, found %d" % len(meshes))
    ob = meshes[0]
    me = ob.data
    world = ob.matrix_world
    normal_matrix = world.to_3x3().inverted_safe().transposed()
    loops = []
    for poly in me.polygons:
        normal = (normal_matrix @ poly.normal).normalized()
        for loop_index in poly.loop_indices:
            co = world @ me.vertices[me.loops[loop_index].vertex_index].co
            key = " ".join(fmt(c) for c in co) + " | " + " ".join(fmt(c) for c in normal)
            loops.append((loop_index, key))
    return {
        "loop_count": len(loops),
        "layers": [
            {
                "name": layer.name,
                "loops": [
                    {
                        "key": key,
                        "uv": [float(layer.data[i].uv[0]), float(layer.data[i].uv[1])],
                        "pin": bool(layer.data[i].pin_uv),
                    }
                    for i, key in loops
                ],
            }
            for layer in me.uv_layers
        ],
        "active": me.uv_layers.active_index,
        "render": me.uv_layers.active_render_index,
    }


result = {}
for source in sources:
    extension = os.path.splitext(source)[1].lower()
    try:
        if extension == ".blend":
            bpy.ops.wm.open_mainfile(filepath=source)
        else:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            IMPORTERS[extension](filepath=source)
        result[os.path.basename(source)] = dump_scene()
    except Exception as error:
        raise RuntimeError("failed to import/dump %s: %s" % (source, error)) from error
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"#;

fn checked_output(command: &mut Command, label: &str) -> TestResult<Output> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(output)
}

/// Runs `pot` with the arguments added by `configure` and parses its `--json` output.
fn run_pot(configure: impl FnOnce(&mut Command)) -> TestResult<Value> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    configure(&mut command);
    command.arg("--json");
    let output = checked_output(&mut command, "pot")?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Writes `source` as `name` under `root` and runs it in Blender with `arguments` after `--`.
fn run_blender_script(
    blender: &Path,
    root: &Path,
    name: &str,
    source: &str,
    arguments: &[&Path],
) -> TestResult {
    let script = root.join(name);
    fs::write(&script, source)?;
    let mut command = Command::new(blender);
    command
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "3",
            "--python",
        ])
        .arg(&script)
        .arg("--");
    for argument in arguments {
        command.arg(argument);
    }
    checked_output(&mut command, name)?;
    Ok(())
}

/// Builds `textured.blend` and `plain.blend` under `root`.
fn build_fixtures(blender: &Path, root: &Path) -> TestResult<(PathBuf, PathBuf)> {
    let textured = root.join("textured.blend");
    let plain = root.join("plain.blend");
    run_blender_script(
        blender,
        root,
        "build.py",
        BUILD_SCRIPT,
        &[textured.as_path(), plain.as_path()],
    )?;
    Ok((textured, plain))
}

/// Initializes a Potter project under `root` and imports the `.blend` at `source` into it.
fn import_project(blender: &Path, root: &Path, name: &str, source: &Path) -> TestResult<PathBuf> {
    let project = root.join(name);
    run_pot(|command| {
        command.arg("init").arg(&project);
    })?;
    let imported = run_pot(|command| {
        command
            .arg("import")
            .arg(&project)
            .arg("--file")
            .arg(source)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(blender);
    })?;
    assert!(
        uv_losses(&imported).is_empty(),
        "{name}: importing the .blend lost UV layers"
    );
    Ok(project)
}

fn potter_export(
    blender: &Path,
    project: &Path,
    format: &str,
    out: &Path,
    allow_lossy: bool,
) -> TestResult<Value> {
    run_pot(|command| {
        command
            .arg("export")
            .arg(project)
            .args(["--format", format, "--out"])
            .arg(out)
            .arg("--blender")
            .arg(blender);
        if allow_lossy {
            command.arg("--allow-lossy");
        }
    })
}

/// The `mesh.uv_layer` and `uv.layer_missing` entries of an import or export report.
fn uv_losses(report: &Value) -> Vec<&Value> {
    report["result"]["losses"]
        .as_array()
        .map_or_else(Vec::new, |losses| {
            losses
                .iter()
                .filter(|loss| {
                    matches!(
                        loss["feature_id"].as_str(),
                        Some("mesh.uv_layer" | "uv.layer_missing")
                    )
                })
                .collect()
        })
}

/// The `uv_layers` store of the mesh that has one, read from the project's `scene.json`.
fn uv_store(project: &Path) -> TestResult<Value> {
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let store = scene["data_blocks"]
        .as_object()
        .ok_or("scene data blocks are not an object")?
        .values()
        .filter_map(|block| block.get("mesh"))
        .find_map(|mesh| mesh["attributes"].get("uv_layers").cloned())
        .ok_or("project has no uv_layers store")?;
    Ok(store)
}

fn store_layer_names(store: &Value) -> Vec<String> {
    store["layers"].as_array().map_or_else(Vec::new, |layers| {
        layers
            .iter()
            .filter_map(|layer| layer["name"].as_str().map(str::to_owned))
            .collect()
    })
}

/// Compares two JSON values: numbers within `STORE_TOLERANCE`, everything else exactly.
fn assert_json_close(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap_or(f64::NAN);
            let b = b.as_f64().unwrap_or(f64::NAN);
            assert!(
                (a - b).abs() <= STORE_TOLERANCE,
                "{path}: {a} differs from {b}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}: length");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                assert_json_close(a, b, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(
                a.keys().collect::<Vec<_>>(),
                b.keys().collect::<Vec<_>>(),
                "{path}: keys"
            );
            for (key, a) in a {
                let Some(b) = b.get(key) else {
                    panic!("{path}: key {key} is missing");
                };
                assert_json_close(a, b, &format!("{path}.{key}"));
            }
        }
        (a, b) => assert_eq!(a, b, "{path}"),
    }
}

/// One dumped layer: loop values and pins keyed by corner position and face normal.
struct Layer {
    name: String,
    values: BTreeMap<String, [f64; 2]>,
    pins: BTreeMap<String, bool>,
}

/// One dumped mesh: its layers in order plus the active and render indices.
struct Mesh {
    layers: Vec<Layer>,
    active: i64,
    render: i64,
}

fn parse_mesh(value: &Value) -> TestResult<Mesh> {
    let layers = value["layers"]
        .as_array()
        .ok_or("dumped layers are missing")?
        .iter()
        .map(parse_layer)
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Mesh {
        layers,
        active: value["active"].as_i64().ok_or("active index is missing")?,
        render: value["render"].as_i64().ok_or("render index is missing")?,
    })
}

fn parse_layer(value: &Value) -> TestResult<Layer> {
    let name = value["name"]
        .as_str()
        .ok_or("layer name is missing")?
        .to_owned();
    let mut values = BTreeMap::new();
    let mut pins = BTreeMap::new();
    for entry in value["loops"].as_array().ok_or("layer loops are missing")? {
        let key = entry["key"]
            .as_str()
            .ok_or("loop key is missing")?
            .to_owned();
        let uv = pair(&entry["uv"])?;
        if let Some(previous) = values.insert(key.clone(), uv) {
            assert!(same_uv(previous, uv), "{name}: loop {key} is ambiguous");
        }
        pins.insert(key, entry["pin"].as_bool().ok_or("loop pin is missing")?);
    }
    Ok(Layer { name, values, pins })
}

fn same_uv(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() <= TOLERANCE && (a[1] - b[1]).abs() <= TOLERANCE
}

fn pair(value: &Value) -> TestResult<[f64; 2]> {
    let values = value.as_array().ok_or("UV value is not an array")?;
    let (Some(u), Some(v)) = (
        values.first().and_then(Value::as_f64),
        values.get(1).and_then(Value::as_f64),
    ) else {
        return Err("UV value is not a pair of numbers".into());
    };
    Ok([u, v])
}

fn dumped_mesh(dumps: &Value, file: &str) -> TestResult<Mesh> {
    let entry = dumps
        .get(file)
        .ok_or_else(|| format!("no Blender dump for {file}"))?;
    parse_mesh(entry)
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "{label}: {actual} differs from {expected}"
    );
}

/// Checks that `mesh` holds exactly the layers `expected` names, in order, and that each carries
/// the loop values (and pins, when `compare_pins`) of the fixture layer it pairs with.
fn assert_layers(
    label: &str,
    mesh: &Mesh,
    source: &Mesh,
    expected: &[(&str, &str)],
    compare_pins: bool,
) {
    let names: Vec<&str> = mesh
        .layers
        .iter()
        .map(|layer| layer.name.as_str())
        .collect();
    let wanted: Vec<&str> = expected.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, wanted, "{label}: layer names and order");
    for (layer, (_, source_name)) in mesh.layers.iter().zip(expected) {
        let Some(source_layer) = source
            .layers
            .iter()
            .find(|candidate| candidate.name == *source_name)
        else {
            panic!("{label}: fixture has no layer {source_name}");
        };
        assert_eq!(
            layer.values.len(),
            source_layer.values.len(),
            "{label} {}: loop count",
            layer.name
        );
        for (key, expected_uv) in &source_layer.values {
            let Some(actual) = layer.values.get(key) else {
                panic!("{label} {}: loop {key} is missing", layer.name);
            };
            assert_close(
                actual[0],
                expected_uv[0],
                &format!("{label} {} {key} u", layer.name),
            );
            assert_close(
                actual[1],
                expected_uv[1],
                &format!("{label} {} {key} v", layer.name),
            );
            if compare_pins {
                assert_eq!(
                    layer.pins.get(key),
                    source_layer.pins.get(key),
                    "{label} {}: pin of {key}",
                    layer.name
                );
            }
        }
    }
}

/// Material name to glTF `texCoord` of its base color texture. Blender omits `texCoord` 0, the
/// glTF default.
fn glb_texcoords(path: &Path) -> TestResult<BTreeMap<String, u64>> {
    let bytes = fs::read(path)?;
    let header: [u8; 4] = bytes
        .get(12..16)
        .ok_or("GLB header is truncated")?
        .try_into()?;
    let length = u32::from_le_bytes(header);
    let json = bytes
        .get(20..20 + length as usize)
        .ok_or("GLB JSON chunk is truncated")?;
    let document: Value = serde_json::from_slice(json)?;
    document["materials"]
        .as_array()
        .ok_or("GLB has no materials")?
        .iter()
        .map(|material| -> TestResult<(String, u64)> {
            let name = material["name"]
                .as_str()
                .ok_or("GLB material name is missing")?;
            let texcoord = material["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"]
                .as_u64()
                .unwrap_or(0);
            Ok((name.to_owned(), texcoord))
        })
        .collect()
}

#[test]
fn every_format_carries_layers_as_blender_does() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let dir = tempdir()?;
    let root = dir.path();
    let (textured, plain) = build_fixtures(&blender, root)?;
    run_blender_script(
        &blender,
        root,
        "export.py",
        EXPORT_SCRIPT,
        &[textured.as_path(), root],
    )?;

    let textured_project = import_project(&blender, root, "textured", &textured)?;
    let plain_project = import_project(&blender, root, "plain", &plain)?;
    let store = uv_store(&textured_project)?;
    assert_eq!(
        store_layer_names(&store),
        vec!["Base".to_owned(), "Light".to_owned(), "Detail".to_owned()]
    );
    assert_eq!(store["active"], json!("Detail"));
    assert_eq!(store["render"], json!("Light"));

    let mut sources = vec![textured.clone(), plain.clone()];
    for case in &CASES {
        let project = if case.source == "plain.blend" {
            &plain_project
        } else {
            &textured_project
        };
        let out = root.join(format!("potter.{}", case.extension));
        let report = potter_export(&blender, project, case.format, &out, case.allow_lossy)?;
        assert_eq!(
            uv_losses(&report).len(),
            case.lost_layers,
            "{}: lost UV layers",
            case.format
        );
        sources.push(root.join(format!("blender.{}", case.extension)));
        sources.push(out);
    }
    let blend = root.join("potter.blend");
    let report = potter_export(&blender, &textured_project, "blend", &blend, false)?;
    assert!(uv_losses(&report).is_empty(), "blend export lost UV layers");
    sources.push(blend);

    let dumps = dump_meshes(&blender, root, &sources)?;
    let textured_source = dumped_mesh(&dumps, "textured.blend")?;
    let plain_source = dumped_mesh(&dumps, "plain.blend")?;
    assert_eq!(textured_source.active, SOURCE_ACTIVE);
    assert_eq!(textured_source.render, SOURCE_RENDER);

    for case in &CASES {
        let source = if case.source == "plain.blend" {
            &plain_source
        } else {
            &textured_source
        };
        let blender_mesh = dumped_mesh(&dumps, &format!("blender.{}", case.extension))?;
        let potter_mesh = dumped_mesh(&dumps, &format!("potter.{}", case.extension))?;
        assert_layers(
            &format!("blender {}", case.format),
            &blender_mesh,
            source,
            case.blender,
            false,
        );
        assert_layers(
            &format!("potter {}", case.format),
            &potter_mesh,
            source,
            case.potter,
            false,
        );
        let blender_names: Vec<&str> = case.blender.iter().map(|(name, _)| *name).collect();
        let potter_names: Vec<&str> = case.potter.iter().map(|(name, _)| *name).collect();
        if case.blender == case.potter {
            assert_eq!(
                potter_names, blender_names,
                "{}: Potter and Blender layer order",
                case.format
            );
        } else {
            let mut sorted_potter = potter_names.clone();
            let mut sorted_blender = blender_names.clone();
            sorted_potter.sort_unstable();
            sorted_blender.sort_unstable();
            assert_eq!(
                sorted_potter, sorted_blender,
                "{}: Potter and Blender layer set",
                case.format
            );
        }
    }

    let blend_mesh = dumped_mesh(&dumps, "potter.blend")?;
    assert_layers(
        "potter blend",
        &blend_mesh,
        &textured_source,
        IN_SOURCE_ORDER,
        true,
    );
    assert_eq!(
        blend_mesh.active, SOURCE_ACTIVE,
        "potter blend: active index"
    );
    assert_eq!(
        blend_mesh.render, SOURCE_RENDER,
        "potter blend: render index"
    );

    let potter_glb = glb_texcoords(&root.join("potter.glb"))?;
    let blender_glb = glb_texcoords(&root.join("blender.glb"))?;
    let expected_glb = BTreeMap::from([("Mat_Render".to_owned(), 1), ("Mat_Detail".to_owned(), 2)]);
    assert_eq!(blender_glb, expected_glb, "Blender glTF texCoord");
    assert_eq!(potter_glb, blender_glb, "Potter glTF texCoord");
    let plain_names: Vec<&str> = plain_source
        .layers
        .iter()
        .map(|layer| layer.name.as_str())
        .collect();
    assert_eq!(
        plain_names,
        vec!["Base", "Light", "Detail"],
        "plain fixture layers"
    );
    let light_pins = textured_source
        .layers
        .iter()
        .find(|layer| layer.name == "Light")
        .map_or(0, |layer| {
            layer.pins.values().filter(|pinned| **pinned).count()
        });
    assert_eq!(light_pins, 4, "textured fixture pins on Light");
    Ok(())
}

/// Runs Blender's dump over `sources` and returns the JSON map keyed by file name.
fn dump_meshes(blender: &Path, root: &Path, sources: &[PathBuf]) -> TestResult<Value> {
    let out = root.join("dumps.json");
    let mut arguments: Vec<&Path> = vec![out.as_path()];
    arguments.extend(sources.iter().map(PathBuf::as_path));
    run_blender_script(blender, root, "dump.py", DUMP_SCRIPT, &arguments)?;
    Ok(serde_json::from_slice(&fs::read(&out)?)?)
}

/// A `.blend` round trip inside Potter: export and re-import must leave the store unchanged.
#[test]
fn blend_store_survives_potter_export_and_reimport() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let dir = tempdir()?;
    let root = dir.path();
    let (textured, _plain) = build_fixtures(&blender, root)?;
    let first = import_project(&blender, root, "first", &textured)?;
    let exported = root.join("round-trip.blend");
    let report = potter_export(&blender, &first, "blend", &exported, false)?;
    assert!(uv_losses(&report).is_empty(), "blend export lost UV layers");
    let second = import_project(&blender, root, "second", &exported)?;
    let first_store = uv_store(&first)?;
    let second_store = uv_store(&second)?;
    assert_eq!(
        store_layer_names(&second_store),
        store_layer_names(&first_store)
    );
    assert_json_close(&second_store, &first_store, "uv_layers");
    Ok(())
}
