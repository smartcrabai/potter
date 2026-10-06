#![expect(
    clippy::unwrap_used,
    reason = "Blender integration fixtures use fixed valid paths and values"
)]

use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::{Value, json};
use tempfile::tempdir;

const CACHE_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
abc_path = os.path.join(root, "animated.abc")
scene = bpy.context.scene
scene.render.fps = 24
scene.render.fps_base = 1.0
scene.frame_start, scene.frame_end = 1, 4
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

vertices = [(-1, -1, 0), (1, -1, 0), (0, 1, 0)]
mesh = bpy.data.meshes.new("AnimatedMesh")
mesh.from_pydata(vertices, [], [(0, 1, 2)])
source = bpy.data.objects.new("Animated", mesh)
scene.collection.objects.link(source)
basis = source.shape_key_add(name="Basis")
lift = source.shape_key_add(name="Lift")
lift.data[0].co.z = 1.5
lift.data[1].co.z = 0.75
lift.value = 0.0
lift.keyframe_insert(data_path="value", frame=1)
lift.value = 1.0
lift.keyframe_insert(data_path="value", frame=4)
source.location.x = 1.0
source.select_set(True)
bpy.context.view_layer.objects.active = source
bpy.ops.wm.alembic_export(filepath=abc_path, start=1, end=4, selected=True,
                          flatten=False, uvs=False, normals=False)

before = {obj.as_pointer() for obj in bpy.data.objects}
bpy.ops.wm.alembic_import(filepath=abc_path, set_frame_range=False,
                          always_add_cache_reader=True, as_background_job=False)
cache_file = next(iter(bpy.data.cache_files))
object_path = cache_file.object_paths[0].path
for obj in list(bpy.data.objects):
    if obj.as_pointer() not in before:
        bpy.data.objects.remove(obj, do_unlink=True)

cache_mesh = bpy.data.meshes.new("CacheMeshData")
cache_mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (0, 1, 0)], [], [(0, 1, 2)])
cache_mesh.update()
cache_object = bpy.data.objects.new("CacheMesh", cache_mesh)
scene.collection.objects.link(cache_object)
cache_file.filepath = abc_path
cache_file.override_frame = False
modifier = cache_object.modifiers.new("Animated Cache", "MESH_SEQUENCE_CACHE")
modifier.cache_file = cache_file
modifier.object_path = object_path
modifier.read_data = {"VERT", "POLY"}
modifier.use_vertex_interpolation = True

transform_object = bpy.data.objects.new("CacheTransform", None)
scene.collection.objects.link(transform_object)
constraint = transform_object.constraints.new("TRANSFORM_CACHE")
constraint.name = "Animated Transform Cache"
constraint.cache_file = cache_file
constraint.object_path = "/" + source.name

expected = {}
for frame in (1, 2, 4):
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = cache_object.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    positions = [[float(v.co.x), float(v.co.y), float(v.co.z)]
                 for v in evaluated_mesh.vertices]
    world_positions = [evaluated.matrix_world @ v.co for v in evaluated_mesh.vertices]
    bounds = [[min(float(p[i]) for p in world_positions) for i in range(3)],
              [max(float(p[i]) for p in world_positions) for i in range(3)]]
    evaluated.to_mesh_clear()
    transform_matrix = transform_object.evaluated_get(depsgraph).matrix_world
    matrix = [float(transform_matrix[row][column])
              for column in range(4) for row in range(4)]
    expected[str(frame)] = {"positions": positions, "bounds": bounds, "matrix": matrix}

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "cache_scene.blend"),
                            relative_remap=True)
with open(os.path.join(root, "cache_expected.json"), "w", encoding="utf-8") as handle:
    json.dump({"object_path": object_path, "constraint_path": constraint.object_path,
               "frames": expected}, handle)
"#;

const CACHE_REOPEN: &str = r#"
import bpy
import json
import os
import sys

output = sys.argv[sys.argv.index("--") + 1]
cache_object = bpy.data.objects["CacheMesh"]
transform_object = bpy.data.objects["CacheTransform"]
modifier = next(item for item in cache_object.modifiers
                if item.type == "MESH_SEQUENCE_CACHE")
constraint = next(item for item in transform_object.constraints
                  if item.type == "TRANSFORM_CACHE")
cache_file = modifier.cache_file
assert cache_file is not None and constraint.cache_file == cache_file
resolved = bpy.path.abspath(cache_file.filepath)
assert os.path.isfile(resolved), resolved
actual = {}
for frame in (1, 2, 4):
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = cache_object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    positions = [[float(v.co.x), float(v.co.y), float(v.co.z)] for v in mesh.vertices]
    matrix = [float(transform_object.evaluated_get(depsgraph).matrix_world[row][column])
              for column in range(4) for row in range(4)]
    actual[str(frame)] = {"positions": positions, "matrix": matrix}
    evaluated.to_mesh_clear()
with open(output, "w", encoding="utf-8") as handle:
    json.dump({"modifier": modifier.type, "constraint": constraint.type,
               "resolved": resolved, "object_path": modifier.object_path,
               "constraint_path": constraint.object_path,
               "frames": actual}, handle)
print("CACHE_REOPEN_OK")
"#;

const LIBRARY_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
# Libraries live in subdirectories and are linked with relative paths (Blender's
# default) so resolution must use the linking file's directory, not the library's.
lib2_path = os.path.join(root, "lib", "nested", "lib2.blend")
lib1_path = os.path.join(root, "lib", "lib1.blend")
os.makedirs(os.path.dirname(lib2_path), exist_ok=True)
main_path = os.path.join(root, "main.blend")

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)

reset()
scene = bpy.context.scene
mat = bpy.data.materials.new("NestedMaterial")
mesh = bpy.data.meshes.new("NestedMesh")
mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (0, 1, 0)], [], [(0, 1, 2)])
obj = bpy.data.objects.new("NestedObject", mesh)
obj.data.materials.append(mat)
collection = bpy.data.collections.new("NestedCollection")
collection.objects.link(obj)
scene.collection.children.link(collection)
bpy.ops.wm.save_as_mainfile(filepath=lib2_path)

reset()
scene = bpy.context.scene
with bpy.data.libraries.load(lib2_path, link=True, relative=True) as (data_from, data_to):
    data_to.collections = ["NestedCollection"]
nested = data_to.collections[0]
scene.collection.children.link(nested)
mat = bpy.data.materials.new("PropMaterial")
mesh = bpy.data.meshes.new("PropMesh")
mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (0, 1, 0)], [], [(0, 1, 2)])
obj = bpy.data.objects.new("PropObject", mesh)
obj.data.materials.append(mat)
collection = bpy.data.collections.new("PropsCollection")
collection.objects.link(obj)
scene.collection.children.link(collection)
bpy.ops.wm.save_as_mainfile(filepath=lib1_path, relative_remap=True)

reset()
scene = bpy.context.scene
with bpy.data.libraries.load(lib1_path, link=True, relative=True) as (data_from, data_to):
    data_to.collections = ["PropsCollection"]
    data_to.objects = ["PropObject"]
linked_collection = data_to.collections[0]
linked_object = data_to.objects[0]
scene.collection.children.link(linked_collection)
scene.collection.objects.link(linked_object)
instance = bpy.data.objects.new("PropsInstance", None)
instance.instance_type = "COLLECTION"
instance.instance_collection = linked_collection
instance.location.x = -3.0
scene.collection.objects.link(instance)
view_layer = bpy.context.view_layer
for item in view_layer.objects:
    item.select_set(item.as_pointer() == linked_object.as_pointer())
view_layer.objects.active = linked_object
area = next(item for item in bpy.context.screen.areas if item.type == "VIEW_3D")
region = next(item for item in area.regions if item.type == "WINDOW")
with bpy.context.temp_override(window=bpy.context.window, screen=bpy.context.screen,
                               scene=scene, view_layer=view_layer, area=area, region=region,
                               object=linked_object, active_object=linked_object,
                               selected_objects=[linked_object]):
    for item in view_layer.objects:
        item.select_set(item.as_pointer() == linked_object.as_pointer())
    view_layer.objects.active = linked_object
    linked_object.select_set(True, view_layer=view_layer)
    bpy.ops.object.make_override_library(collection=0)
override = next(item for item in bpy.data.objects
                if item.override_library and item.override_library.reference == linked_object)
override.location = (2.0, 0.0, 0.0)
bpy.ops.wm.save_as_mainfile(filepath=main_path, relative_remap=True)
with open(os.path.join(root, "library_expected.json"), "w", encoding="utf-8") as handle:
    json.dump({"linked_object": linked_object.name_full,
               "linked_collection": linked_collection.name_full,
               "override": override.name_full,
               "nested_library_count": len(bpy.data.libraries)}, handle)
print("LIBRARY_FIXTURE_OK", linked_object.name_full, override.name_full)
"#;

const LIBRARY_REOPEN: &str = r#"
import bpy
import json
import os
import sys

output = sys.argv[sys.argv.index("--") + 1]
linked = next(obj for obj in bpy.data.objects
              if obj.name.startswith("PropObject") and obj.library is not None)
assert linked.data is not None and linked.data.library is not None
assert linked.data.materials and linked.data.materials[0].library is not None
linked_collection = next(collection for collection in bpy.data.collections
                         if collection.name.startswith("PropsCollection")
                         and collection.library is not None)
instance = bpy.data.objects["PropsInstance"]
assert instance.instance_collection == linked_collection
libraries = []
for library in bpy.data.libraries:
    path = bpy.path.abspath(library.filepath)
    assert os.path.isfile(path), path
    libraries.append({"name": library.name, "path": path})
lib1_path = next(entry["path"] for entry in libraries if entry["name"] == "lib1.blend")
overrides = [obj for obj in bpy.data.objects if obj.override_library is not None]
assert overrides, "no library override was reconstructed"
override = next(obj for obj in overrides
                if all(abs(actual - expected) < 1.0e-6
                       for actual, expected in zip(obj.location, (7.0, 8.0, 9.0))))
linked_state = {"linked": bool(linked.library), "linked_data": bool(linked.data.library),
                "linked_material": bool(linked.data.materials[0].library),
                "collection_linked": bool(linked_collection.library),
                "override_location": list(override.location)}
bpy.ops.wm.open_mainfile(filepath=lib1_path, load_ui=False, use_scripts=False)
nested_libraries = []
for library in bpy.data.libraries:
    path = bpy.path.abspath(library.filepath, library=library.parent)
    assert os.path.isfile(path), path
    nested_libraries.append({"name": library.name, "path": path})
with open(output, "w", encoding="utf-8") as handle:
    json.dump(dict(linked_state, libraries=libraries, nested_libraries=nested_libraries), handle)
print("LIBRARY_REOPEN_OK")
"#;

fn blender_executable() -> Option<PathBuf> {
    fn usable(path: PathBuf) -> Option<PathBuf> {
        path.is_file().then_some(path)
    }
    if let Some(path) = env::var_os("POTTER_BLENDER") {
        return usable(PathBuf::from(path));
    }
    if let Some(path) = env::split_paths(&env::var_os("PATH")?)
        .map(|directory| directory.join("blender"))
        .find_map(usable)
    {
        return Some(path);
    }
    usable(PathBuf::from(
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ))
}

fn run(command: &mut Command, label: &str) -> Result<Output, Box<dyn Error>> {
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

fn blender_script(
    blender: &Path,
    root: &Path,
    name: &str,
    source: &str,
) -> Result<(), Box<dyn Error>> {
    let script = root.join(name);
    fs::write(&script, source)?;
    let canonical_root = fs::canonicalize(root)?;
    let mut command = Command::new(blender);
    command.args([
        "--background",
        "--factory-startup",
        "--python",
        script.to_str().unwrap(),
        "--",
        canonical_root.to_str().unwrap(),
    ]);
    let output = run(&mut command, name)?;
    let fixture_file = if name == "make_cache.py" {
        "cache_expected.json"
    } else {
        "library_expected.json"
    };
    if !root.join(fixture_file).is_file() {
        return Err(std::io::Error::other(format!(
            "{name} did not create {fixture_file}: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(())
}

fn blender_reopen(
    blender: &Path,
    blend: &Path,
    script: &str,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let script_path = output.with_extension("py");
    fs::write(&script_path, script)?;
    let mut command = Command::new(blender);
    command.args([
        "--background",
        blend.to_str().unwrap(),
        "--python",
        script_path.to_str().unwrap(),
        "--",
        output.to_str().unwrap(),
    ]);
    let child = run(&mut command, "Blender reopen")?;
    if !output.is_file() {
        return Err(std::io::Error::other(format!(
            "Blender reopen did not write {}: stdout={} stderr={}",
            output.display(),
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr),
        ))
        .into());
    }
    Ok(())
}

fn pot_json(arguments: &[&str], label: &str) -> Result<Value, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(arguments).arg("--json");
    let output = run(&mut command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn pot_output(arguments: &[&str]) -> Result<(Output, Value), Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(arguments).arg("--json");
    let output = command.output()?;
    let value = serde_json::from_slice(&output.stdout)?;
    Ok((output, value))
}

fn apply_raw(
    scene: &Path,
    revision: u64,
    operations: &Value,
) -> Result<(Output, Value), Box<dyn Error>> {
    let directory = tempdir()?;
    let path = directory.path().join("operations.json");
    fs::write(
        &path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":revision,
            "operations":operations,
        }))?,
    )?;
    pot_output(&[
        "apply",
        scene.to_str().unwrap(),
        "--file",
        path.to_str().unwrap(),
    ])
}

fn apply(scene: &Path, revision: u64, operations: &Value) -> Result<Value, Box<dyn Error>> {
    let (output, value) = apply_raw(scene, revision, operations)?;
    if !output.status.success() {
        return Err(
            std::io::Error::other(String::from_utf8_lossy(&output.stdout).to_string()).into(),
        );
    }
    Ok(value)
}

fn item_by_id<'a>(document: &'a Value, id: &str) -> &'a Value {
    document["result"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == id)
        .unwrap()
}

fn import_blend(project: &Path, source: &Path, blender: &Path) -> Result<Value, Box<dyn Error>> {
    pot_json(
        &[
            "import",
            project.to_str().unwrap(),
            "--file",
            source.to_str().unwrap(),
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
            blender.to_str().unwrap(),
        ],
        "Blender import",
    )
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one scenario verifies cache import, evaluation, deletion, and editable round-trip"
)]
fn alembic_cache_modifier_and_transform_constraint_round_trip() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script(&blender, root, "make_cache.py", CACHE_FIXTURE)?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("cache_expected.json"))?)?;
    let project = root.join("project");
    pot_json(&["init", project.to_str().unwrap()], "scene initialization")?;
    let imported = import_blend(&project, &root.join("cache_scene.blend"), &blender)?;
    assert!(
        imported["result"]["losses"].as_array().unwrap().is_empty(),
        "{imported}"
    );
    let mappings = imported["result"]["id_mappings"].as_object().unwrap();
    let cache_id = mappings["Object:CacheMesh"].as_str().unwrap();
    let transform_id = mappings["Object:CacheTransform"].as_str().unwrap();
    assert!(fs::remove_file(root.join("animated.abc")).is_ok());

    for frame in [1, 2, 4] {
        let frame_text = frame.to_string();
        let inspected = pot_json(
            &[
                "inspect",
                project.to_str().unwrap(),
                "--id",
                cache_id,
                "--frame",
                &frame_text,
            ],
            "cache frame inspection",
        )?;
        let cache_item = item_by_id(&inspected, cache_id);
        let expected_frame = &expected["frames"][frame_text.as_str()];
        let actual_positions = cache_item["evaluated_geometry"]["positions"]
            .as_array()
            .unwrap();
        let expected_positions = expected_frame["positions"].as_array().unwrap();
        assert_eq!(actual_positions.len(), expected_positions.len());
        for actual in actual_positions {
            let found = expected_positions.iter().any(|expected| {
                (0..3).all(|axis| {
                    let expected_value = expected[axis].as_f64().unwrap();
                    (actual[axis].as_f64().unwrap() - expected_value).abs()
                        < 1.0e-6 + expected_value.abs() * 1.0e-6
                })
            });
            assert!(
                found,
                "frame {frame}: actual {actual_positions:?}, expected {expected_positions:?}"
            );
        }
        let expected_bounds = &expected_frame["bounds"];
        let actual_bounds = cache_item["bounds"].as_object().unwrap();
        for (actual_name, expected_index) in [("min", 0), ("max", 1)] {
            for axis in 0..3 {
                let expected_value = expected_bounds[expected_index][axis].as_f64().unwrap();
                let delta = actual_bounds[actual_name][axis].as_f64().unwrap() - expected_value;
                assert!(
                    delta.abs() < 1.0e-6 + expected_value.abs() * 1.0e-6,
                    "frame {frame}, bounds {actual_name}: {delta}"
                );
            }
        }
        let transform = pot_json(
            &[
                "inspect",
                project.to_str().unwrap(),
                "--id",
                transform_id,
                "--frame",
                &frame_text,
            ],
            "transform cache inspection",
        )?;
        let matrix = item_by_id(&transform, transform_id)["transform"]["world"]["matrix"]
            .as_array()
            .unwrap();
        for (index, (actual_component, expected_component)) in matrix
            .iter()
            .zip(expected_frame["matrix"].as_array().unwrap())
            .enumerate()
        {
            let expected_value = expected_component.as_f64().unwrap();
            let delta = actual_component.as_f64().unwrap() - expected_value;
            assert!(
                delta.abs() < 1.0e-6 + expected_value.abs() * 1.0e-6,
                "frame {frame}, matrix index {index}: actual={actual_component}, expected={expected_component}, delta={delta}"
            );
        }
    }

    let output_blend = root.join("cache_roundtrip.blend");
    pot_json(
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
        "cache Blend export",
    )?;
    let reopen = root.join("cache_reopen.json");
    blender_reopen(&blender, &output_blend, CACHE_REOPEN, &reopen)?;
    let actual: Value = serde_json::from_slice(&fs::read(reopen)?)?;
    assert_eq!(actual["modifier"], "MESH_SEQUENCE_CACHE");
    assert_eq!(actual["constraint"], "TRANSFORM_CACHE");
    assert!(Path::new(actual["resolved"].as_str().unwrap()).is_file());
    assert_eq!(actual["object_path"], expected["object_path"]);
    assert_eq!(actual["constraint_path"], expected["constraint_path"]);
    for frame in ["1", "2", "4"] {
        for (actual, expected) in actual["frames"][frame]["matrix"]
            .as_array()
            .unwrap()
            .iter()
            .zip(expected["frames"][frame]["matrix"].as_array().unwrap())
        {
            assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1.0e-4);
        }
    }
    Ok(())
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one scenario verifies nested links, linked-ID protection, overrides, and sidecar export"
)]
fn blender_libraries_link_overrides_and_nested_dependencies_round_trip()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script(&blender, root, "make_library.py", LIBRARY_FIXTURE)?;
    let nested_project = root.join("nested_project");
    pot_json(
        &["init", nested_project.to_str().unwrap()],
        "nested project initialization",
    )?;
    let nested_import = import_blend(&nested_project, &root.join("lib/lib1.blend"), &blender)?;
    assert!(
        nested_import["result"]["losses"]
            .as_array()
            .unwrap()
            .is_empty(),
        "{nested_import}"
    );
    let nested_inspection = pot_json(
        &["inspect", nested_project.to_str().unwrap()],
        "nested library inspection",
    )?;
    assert!(
        nested_inspection["result"]["libraries"]
            .as_object()
            .unwrap()
            .values()
            .any(|library| library["name"] == "lib2.blend")
    );
    let project = root.join("project");
    pot_json(&["init", project.to_str().unwrap()], "scene initialization")?;
    let imported = import_blend(&project, &root.join("main.blend"), &blender)?;
    assert!(
        imported["result"]["losses"].as_array().unwrap().is_empty(),
        "{imported}"
    );
    let inspected = pot_json(
        &["inspect", project.to_str().unwrap()],
        "library inspection",
    )?;
    let linked = inspected["result"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| {
            item["name"]
                .as_str()
                .is_some_and(|name| name.starts_with("PropObject"))
                && item["editable"] == false
        })
        .unwrap();
    let linked_id = linked["id"].as_str().unwrap();
    let imported_override = inspected["result"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["library_override"]["reference_id"] == linked_id)
        .unwrap();
    assert_eq!(
        imported_override["library_override"]["operations"][0]["value"],
        json!([2.0, 0.0, 0.0])
    );
    let library_id = linked["library"]["id"].as_str().unwrap();
    assert!(inspected["result"]["libraries"][library_id].is_object());
    assert!(
        inspected["result"]["items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["kind"] == "collection_instance")
    );
    assert!(
        !inspected["result"]["libraries"]
            .as_object()
            .unwrap()
            .is_empty()
    );

    let revision = imported["result"]["candidate_revision"].as_u64().unwrap();
    let (_, rejected) = apply_raw(
        &project,
        revision,
        &json!([{"op":"node.update","target":{"id":linked_id},"set":{"transform":{"translation":[1,2,3]}}}]),
    )?;
    assert_eq!(rejected["ok"], false);
    let overridden = apply(
        &project,
        revision,
        &json!([{
            "op":"library.override",
            "target":{"id":linked_id},
            "id":"edited_override",
            "operations":[{"op":"replace","path":"transform.translation","value":[7.0,8.0,9.0]}]
        }]),
    )?;
    assert_eq!(overridden["ok"], true, "{overridden}");
    fs::remove_file(root.join("lib/lib1.blend"))?;
    fs::remove_file(root.join("lib/nested/lib2.blend"))?;
    let output_blend = root.join("library_roundtrip.blend");
    pot_json(
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
        "library Blend export",
    )?;
    let reopened = root.join("library_reopen.json");
    blender_reopen(&blender, &output_blend, LIBRARY_REOPEN, &reopened)?;
    let actual: Value = serde_json::from_slice(&fs::read(reopened)?)?;
    assert_eq!(actual["linked"], true);
    assert_eq!(actual["linked_data"], true);
    assert_eq!(actual["linked_material"], true);
    assert_eq!(actual["collection_linked"], true);
    assert_eq!(actual["override_location"], json!([7.0, 8.0, 9.0]));
    assert!(!actual["libraries"].as_array().unwrap().is_empty());
    assert!(!actual["nested_libraries"].as_array().unwrap().is_empty());
    assert!(
        root.join("library_roundtrip_assets/nested/lib2.blend")
            .is_file()
    );
    let missing_project = root.join("missing_project");
    pot_json(
        &["init", missing_project.to_str().unwrap()],
        "missing-library project initialization",
    )?;
    let missing_import = import_blend(&missing_project, &root.join("main.blend"), &blender)?;
    assert!(
        missing_import["result"]["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|warning| warning["code"] == "ASSET_MISSING")
    );
    let missing_doc: Value =
        serde_json::from_slice(&fs::read(missing_project.join("scene.json"))?)?;
    assert!(
        missing_doc["libraries"]
            .as_object()
            .unwrap()
            .values()
            .any(|library| library["status"] == "missing")
    );
    let (output, error) = pot_output(&["inspect", missing_project.to_str().unwrap()])?;
    assert!(!output.status.success());
    assert_eq!(error["error"]["code"], "ASSET_CHANGED");
    Ok(())
}
