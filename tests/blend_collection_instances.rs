//! Blender parity for collection instances. A `.blend` with two members (different material lists,
//! UVs, seams, vertex colors, smooth shading and vertex groups) and a nested instance must export to
//! glTF with the same primitives, materials, index counts, and TEXCOORD / NORMAL values per primitive
//! as Blender's own exporter, after Potter imports the `.blend` and evaluates the instances.

use std::{collections::BTreeSet, error::Error, fs, path::Path, process::Command};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use serde_json::Value;
use tempfile::tempdir;

type TestResult<T> = Result<T, Box<dyn Error>>;

const FIXTURE: &str = r#"
import bpy
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
    return mat


def cube(name, size, materials, location=(0.0, 0.0, 0.0)):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(
        [(-size, -size, -size), (size, -size, -size), (size, size, -size), (-size, size, -size),
         (-size, -size, size), (size, -size, size), (size, size, size), (-size, size, size)],
        [],
        [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    )
    mesh.update()
    for mat in materials:
        mesh.materials.append(mat)
    uv = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (co.x * 0.5 + 0.5, co.y * 0.5 + 0.5)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    return obj


jacket_mat = material("jacket", (0.8, 0.1, 0.1, 1.0))
skin_mat = material("skin", (0.9, 0.7, 0.5, 1.0))
members = bpy.data.collections.new("Members")
outer = bpy.data.collections.new("Outer")
scene.collection.children.link(outer)

jacket = cube("Jacket", 1.0, [jacket_mat, skin_mat])
for polygon in jacket.data.polygons:
    polygon.material_index = 1 if polygon.index == 1 else 0
members.objects.link(jacket)

skin = cube("Skin", 1.0, [skin_mat], location=(2.5, 0.0, 0.0))
for polygon in skin.data.polygons:
    polygon.use_smooth = True
skin.data.normals_split_custom_set([(0.0, 0.6, 0.8)] * len(skin.data.loops))
for edge in skin.data.edges:
    edge.use_seam = edge.index < 4
color = skin.data.color_attributes.new(name="Col", type="BYTE_COLOR", domain="CORNER")
for index, datum in enumerate(color.data):
    datum.color = (0.1 * index % 1.0, 0.5, 0.25, 1.0)
group = skin.vertex_groups.new(name="Weight")
group.add(list(range(8)), 0.5, "REPLACE")
members.objects.link(skin)

inner = bpy.data.objects.new("InnerInstance", None)
inner.instance_type = "COLLECTION"
inner.instance_collection = members
outer.objects.link(inner)
outer.objects.link(cube("Hat", 0.5, [skin_mat], location=(0.0, 3.0, 0.0)))

instance = bpy.data.objects.new("Instance", None)
instance.instance_type = "COLLECTION"
instance.instance_collection = outer
instance.location = (0.0, 0.0, 5.0)
instance.rotation_euler = (0.0, 0.0, 0.5235987755982988)
instance.scale = (-1.0, 1.0, 1.0)
scene.collection.objects.link(instance)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "oracle.blend"), check_existing=False)
bpy.ops.export_scene.gltf(filepath=os.path.join(root, "blender.glb"), export_format="GLB")
"#;

/// One exported primitive, reduced to what both exporters must agree on: material, index count,
/// TEXCOORD and NORMAL presence, and the UV and normal values of the vertices its triangles use.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Primitive {
    material: Option<String>,
    indices: usize,
    texcoord_sets: usize,
    has_normal: bool,
    uvs: BTreeSet<Vec<i64>>,
    normals: Vec<Vec<i64>>,
}

fn run_pot(args: &[&str]) -> TestResult<()> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(args)
        .output()?;
    assert!(
        output.status.success(),
        "pot {} failed: stdout={} stderr={}",
        args.first().copied().unwrap_or("command"),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

fn run_pot_json(args: &[&str]) -> TestResult<Value> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(args)
        .output()?;
    assert!(
        output.status.success(),
        "pot {} failed: stdout={} stderr={}",
        args.first().copied().unwrap_or("command"),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// The JSON chunk and binary chunk of a GLB file.
fn read_glb(path: &Path) -> TestResult<(Value, Vec<u8>)> {
    let data = fs::read(path)?;
    let json_length =
        u32::from_le_bytes(data.get(12..16).ok_or("GLB header is short")?.try_into()?) as usize;
    let json_end = 20 + json_length;
    let gltf = serde_json::from_slice(data.get(20..json_end).ok_or("GLB JSON chunk is short")?)?;
    let bin_length = u32::from_le_bytes(
        data.get(json_end..json_end + 4)
            .ok_or("GLB BIN chunk is missing")?
            .try_into()?,
    ) as usize;
    let bin = data
        .get(json_end + 8..json_end + 8 + bin_length)
        .ok_or("GLB BIN chunk is short")?
        .to_vec();
    Ok((gltf, bin))
}

/// Byte offset and element stride of an accessor's buffer view.
fn accessor_layout(
    gltf: &Value,
    accessor: &Value,
    width: usize,
) -> TestResult<(usize, usize, usize)> {
    let view_index = accessor["bufferView"]
        .as_u64()
        .ok_or("accessor has no buffer view")?;
    let view = &gltf["bufferViews"][view_index as usize];
    let start =
        view["byteOffset"].as_u64().unwrap_or(0) + accessor["byteOffset"].as_u64().unwrap_or(0);
    let stride = view["byteStride"]
        .as_u64()
        .map_or(width, |stride| stride as usize);
    let count = accessor["count"].as_u64().ok_or("accessor has no count")? as usize;
    Ok((start as usize, stride, count))
}

/// Float elements of `width` components, rounded to 1e-3 so both exporters compare exactly.
fn float_elements(
    gltf: &Value,
    bin: &[u8],
    accessor: &Value,
    width: usize,
) -> TestResult<Vec<Vec<i64>>> {
    let (start, stride, count) = accessor_layout(gltf, accessor, width * 4)?;
    let mut elements = Vec::with_capacity(count);
    for item in 0..count {
        let base = start + item * stride;
        let bytes = bin
            .get(base..base + width * 4)
            .ok_or("accessor exceeds the binary chunk")?;
        elements.push(
            (0..width)
                .map(|component| {
                    let at = component * 4;
                    let value = f32::from_le_bytes([
                        bytes[at],
                        bytes[at + 1],
                        bytes[at + 2],
                        bytes[at + 3],
                    ]);
                    (f64::from(value) * 1000.0).round() as i64
                })
                .collect(),
        );
    }
    Ok(elements)
}

/// Index values of an unsigned byte, short, or int accessor.
fn index_elements(gltf: &Value, bin: &[u8], accessor: &Value) -> TestResult<Vec<usize>> {
    let width = match accessor["componentType"].as_u64() {
        Some(5121) => 1,
        Some(5123) => 2,
        Some(5125) => 4,
        _ => return Err("index accessor component type is unsupported".into()),
    };
    let (start, _, count) = accessor_layout(gltf, accessor, width)?;
    let bytes = bin
        .get(start..start + count * width)
        .ok_or("index accessor exceeds the binary chunk")?;
    Ok(bytes
        .chunks_exact(width)
        .map(|chunk| {
            chunk
                .iter()
                .rev()
                .fold(0_usize, |value, byte| (value << 8) | usize::from(*byte))
        })
        .collect())
}

/// Every primitive a scene renders, flattened through the node tree. A mesh referenced by two nodes
/// counts twice, as it renders twice.
fn primitives(path: &Path) -> TestResult<Vec<Primitive>> {
    let (gltf, bin) = read_glb(path)?;
    let materials: Vec<String> = gltf["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|item| item["name"].as_str().unwrap_or_default().to_owned())
        .collect();
    let mut stack: Vec<u64> = gltf["scenes"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|scene| scene["nodes"].as_array().into_iter().flatten())
        .filter_map(Value::as_u64)
        .collect();
    let mut result = Vec::new();
    while let Some(index) = stack.pop() {
        let node = &gltf["nodes"][index as usize];
        stack.extend(
            node["children"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_u64),
        );
        let Some(mesh) = node["mesh"].as_u64() else {
            continue;
        };
        for primitive in gltf["meshes"][mesh as usize]["primitives"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let attributes = &primitive["attributes"];
            let index_accessor = &gltf["accessors"][primitive["indices"]
                .as_u64()
                .ok_or("primitive has no indices")?
                as usize];
            let indices = index_elements(&gltf, &bin, index_accessor)?;
            let used: BTreeSet<usize> = indices.iter().copied().collect();
            let uvs = match attributes["TEXCOORD_0"].as_u64() {
                Some(accessor) => {
                    let values =
                        float_elements(&gltf, &bin, &gltf["accessors"][accessor as usize], 2)?;
                    used.iter()
                        .filter_map(|vertex| values.get(*vertex).cloned())
                        .collect()
                }
                None => BTreeSet::new(),
            };
            let mut normals = match attributes["NORMAL"].as_u64() {
                Some(accessor) => {
                    let values =
                        float_elements(&gltf, &bin, &gltf["accessors"][accessor as usize], 3)?;
                    indices
                        .iter()
                        .filter_map(|vertex| values.get(*vertex).cloned())
                        .collect()
                }
                None => Vec::new(),
            };
            normals.sort_unstable();
            result.push(Primitive {
                material: primitive["material"]
                    .as_u64()
                    .and_then(|slot| materials.get(slot as usize).cloned()),
                indices: indices.len(),
                texcoord_sets: attributes
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter(|(key, _)| key.starts_with("TEXCOORD_"))
                    .count(),
                has_normal: attributes.get("NORMAL").is_some(),
                uvs,
                normals,
            });
        }
    }
    result.sort();
    Ok(result)
}

#[test]
fn collection_instance_glb_matches_blenders_exporter() -> TestResult<()> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping collection instance parity: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let script = root.join("make_oracle.py");
    fs::write(&script, FIXTURE)?;
    let output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script)
        .arg("--")
        .arg(root)
        .output()?;
    assert!(
        output.status.success(),
        "Blender fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let project = root.join("project");
    let project_text = project.to_str().ok_or("project path is not UTF-8")?;
    let blend = root.join("oracle.blend");
    let blend_text = blend.to_str().ok_or("blend path is not UTF-8")?;
    let blender_text = blender.to_str().ok_or("Blender path is not UTF-8")?;
    let potter_glb = root.join("potter.glb");
    let potter_glb_text = potter_glb.to_str().ok_or("output path is not UTF-8")?;
    run_pot(&["init", project_text])?;
    run_pot(&[
        "import",
        project_text,
        "--file",
        blend_text,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender_text,
    ])?;
    // glTF export does not write vertex colors; Blender's exporter omits them here too, so the
    // export is lossy in that one respect.
    run_pot(&[
        "export",
        project_text,
        "--format",
        "glb",
        "--out",
        potter_glb_text,
        "--allow-lossy",
        "--overwrite",
    ])?;

    let expected = primitives(&root.join("blender.glb"))?;
    let actual = primitives(&potter_glb)?;
    assert_eq!(
        expected.len(),
        8,
        "Blender exports eight primitives for this scene"
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn public_cli_nested_instances_preview_render_and_export_member_materials() -> TestResult<()> {
    let directory = tempdir()?;
    let root = directory.path();
    let project = root.join("project");
    let project_text = project.to_str().ok_or("project path is not UTF-8")?;
    run_pot(&["init", project_text])?;

    let operations = root.join("operations.json");
    fs::write(
        &operations,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"material.create","id":"jacket","base_color":[0.9,0.05,0.02,1.0]},
                {"op":"material.create","id":"skin","base_color":[0.02,0.1,0.9,1.0]},
                {"op":"collection.create","id":"members"},
                {"op":"node.create","id":"jacket_mesh","kind":"box","collection":"members","params":{"size":1.0},"material":"jacket"},
                {"op":"node.create","id":"skin_mesh","kind":"box","collection":"members","params":{"size":1.0},"transform":{"translation":[1.5,0.0,0.0]},"material":"skin"},
                {"op":"collection.create","id":"wrapper"},
                {"op":"collection.instance_create","id":"inner","collection":"members","transform":{"translation":[0.0,1.5,0.0]}},
                {"op":"collection.link","collection":"wrapper","object":"inner"},
                {"op":"node.create","id":"hat","kind":"box","collection":"wrapper","params":{"size":0.5},"transform":{"translation":[0.0,0.0,1.5]},"material":"jacket"},
                {"op":"collection.instance_create","id":"assembly","collection":"wrapper","transform":{"translation":[0.0,0.0,0.5]}},
                {"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,10.0]},"projection":"orthographic","ortho_scale":5.0},
                {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},
                {"op":"light.create","id":"key_light","light_type":"point","transform":{"translation":[0.0,0.0,6.0]},"energy":100.0,"radius":0.5},
                {"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":64,"resolution_y":64,"samples":1,"seed":0,"max_bounces":1}}
            ]
        }))?,
    )?;
    let operations_text = operations.to_str().ok_or("operations path is not UTF-8")?;
    let apply = run_pot_json(&["apply", project_text, "--file", operations_text, "--json"])?;
    assert_eq!(apply["ok"], true);

    let preview_dir = root.join("preview");
    let preview_text = preview_dir.to_str().ok_or("preview path is not UTF-8")?;
    let preview = run_pot_json(&[
        "preview",
        project_text,
        "--views",
        "top",
        "--size",
        "64",
        "--out",
        preview_text,
        "--json",
    ])?;
    let preview_path = preview_dir.join("top.png");
    assert_eq!(preview["result"]["previews"][0]["view"], "top");
    assert!(preview_path.is_file());

    let render_dir = root.join("render");
    let render_text = render_dir.to_str().ok_or("render path is not UTF-8")?;
    let rendered = run_pot_json(&[
        "render",
        project_text,
        "--engine",
        "path",
        "--format",
        "png",
        "--out",
        render_text,
        "--json",
    ])?;
    let render_path = rendered["result"]["frames"][0]["path"]
        .as_str()
        .ok_or("rendered image path is missing")?;
    assert!(Path::new(render_path).is_file());
    let image = image::open(render_path)?.to_rgb8();
    let mut red_material_pixels = 0;
    let mut blue_material_pixels = 0;
    for pixel in image.pixels() {
        if pixel[0] > pixel[2].saturating_add(40) && pixel[0] > pixel[1] {
            red_material_pixels += 1;
        }
        if pixel[2] > pixel[0].saturating_add(40) && pixel[2] > pixel[1] {
            blue_material_pixels += 1;
        }
    }
    assert!(
        red_material_pixels > 0 && blue_material_pixels > 0,
        "render must show both member materials"
    );

    let glb = root.join("assembly.glb");
    let glb_text = glb.to_str().ok_or("GLB path is not UTF-8")?;
    run_pot(&["export", project_text, "--format", "glb", "--out", glb_text])?;
    let exported = primitives(&glb)?;
    let exported_materials: BTreeSet<Option<String>> = exported
        .iter()
        .map(|primitive| primitive.material.clone())
        .collect();
    assert_eq!(
        exported_materials,
        BTreeSet::from([Some("jacket".to_owned()), Some("skin".to_owned())])
    );
    eprintln!("nested collection preview: {}", preview_path.display());
    eprintln!("nested collection render: {render_path}");
    eprintln!("nested collection glTF materials: {exported_materials:?}");
    Ok(())
}
