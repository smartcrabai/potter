//! Native color, UV, vertex-group, and material exchange through real Blender 5.2.2 and Potter.
//!
//! The starting mesh is the explicit Blender oracle written by `build_oracle.py`. Each Potter
//! export is decoded from its file bytes and independently imported by Blender 5.2.2. Values are
//! compared to the oracle's explicit source arrays, never to Potter's own import of the exported
//! file. Tests skip when Blender is absent, and fail when `POTTER_REQUIRE_BLENDER` is set.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "this integration binary uses a subset of the shared native oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use blender_checked::blender_executable;
use native_oracle::{
    TestResult, apply_operations, assert_close, assert_floats, build_oracle, dump_blend,
    dumped_object, export_raw, import_blend, oracle_section, pot, pot_json, read_scene,
    run_blender_arguments, write_script,
};
use serde_json::{Value, json};
use tempfile::tempdir;

const TOLERANCE: f64 = 2.0e-5;
const NORMAL_TOLERANCE: f64 = 3.0e-4;
/// Color tolerance for values that pass through f32 storage and an sRGB transfer.
const COLOR_TOLERANCE: f64 = 3.0e-5;

const PRUNE_SCRIPT: &str = r#"
import json
import sys

import bpy

source, expected_path, destination = sys.argv[sys.argv.index("--") + 1 :][:3]
bpy.ops.wm.open_mainfile(filepath=source)
obj = bpy.data.objects.get("Oracle")
if obj is None or obj.type != "MESH":
    raise RuntimeError("native oracle has no Oracle mesh")
for other in list(bpy.data.objects):
    if other != obj:
        bpy.data.objects.remove(other, do_unlink=True)
obj.location = (0.0, 0.0, 0.0)
obj.rotation_euler = (0.0, 0.0, 0.0)
obj.scale = (1.0, 1.0, 1.0)
obj.data.materials.clear()
expected = json.load(open(expected_path, encoding="utf-8"))["oracle"]
for name in expected["generic"]:
    attribute = obj.data.attributes.get(name)
    if attribute is not None:
        obj.data.attributes.remove(attribute)
for material in list(bpy.data.materials):
    if material.users == 0:
        bpy.data.materials.remove(material)
bpy.ops.wm.save_as_mainfile(filepath=destination)
"#;

const ARMATURE_GROUP_SCRIPT: &str = r#"
import sys

import bpy

destination = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("WeightedMesh")
mesh.from_pydata([(-1, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
mesh.update()
obj = bpy.data.objects.new("WeightedMesh", mesh)
bpy.context.scene.collection.objects.link(obj)

armature_data = bpy.data.armatures.new("Rig")
armature = bpy.data.objects.new("Rig", armature_data)
bpy.context.scene.collection.objects.link(armature)
bpy.context.view_layer.objects.active = armature
armature.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bone = armature_data.edit_bones.new("DeformBone")
bone.head = (0, 0, 0)
bone.tail = (0, 0, 1)
bpy.ops.object.mode_set(mode="OBJECT")

group = obj.vertex_groups.new(name="DeformBone")
for index, weight in ((0, 0.25), (1, 0.5), (2, 0.75)):
    group.add([index], weight, "REPLACE")
modifier = obj.modifiers.new("Armature", "ARMATURE")
modifier.object = armature
modifier.use_vertex_groups = True
bpy.ops.wm.save_as_mainfile(filepath=destination)
"#;

const IMPORT_DUMP_SCRIPT: &str = r#"
import json
import os
import sys

import bpy

out_path, sources = sys.argv[sys.argv.index("--") + 1], sys.argv[sys.argv.index("--") + 2 :]
IMPORTERS = {
    ".obj": bpy.ops.wm.obj_import,
    ".ply": bpy.ops.wm.ply_import,
    ".usda": bpy.ops.wm.usd_import,
    ".usd": bpy.ops.wm.usd_import,
    ".fbx": bpy.ops.import_scene.fbx,
    ".abc": bpy.ops.wm.alembic_import,
}


def dump_mesh(obj):
    mesh = obj.data
    mesh.update()
    world = obj.matrix_world
    normal_matrix = world.to_3x3().inverted_safe().transposed()
    vertices = [list(world @ vertex.co) for vertex in mesh.vertices]
    faces = [list(poly.vertices) for poly in mesh.polygons]
    uvs = {layer.name: [[float(c) for c in item.uv] for item in layer.data]
           for layer in mesh.uv_layers}
    colors = {}
    for layer in mesh.color_attributes:
        colors[layer.name] = {
            "domain": layer.domain,
            "data_type": layer.data_type,
            "values": [[float(c) for c in item.color] for item in layer.data],
        }
    corners = []
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            position = vertices[vertex_index]
            normal = normal_matrix @ mesh.corner_normals[loop_index].vector
            corner_colors = {}
            for name, layer in ((layer.name, layer) for layer in mesh.color_attributes):
                index = vertex_index if layer.domain == "POINT" else loop_index
                corner_colors[name] = [float(c) for c in layer.data[index].color]
            corners.append({
                "vertex": vertex_index,
                "position": position,
                "normal": [float(c) for c in normal],
                "uvs": {name: values[loop_index] for name, values in uvs.items()},
                "colors": corner_colors,
            })
    groups = {}
    for group in obj.vertex_groups:
        members = {}
        for vertex in mesh.vertices:
            for assignment in vertex.groups:
                if assignment.group == group.index:
                    members[str(vertex.index)] = float(assignment.weight)
        groups[group.name] = members
    material_slots = []
    for slot in obj.material_slots:
        material = slot.material
        images = []
        if material is not None and material.use_nodes and material.node_tree is not None:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    images.append({
                        "name": node.image.name,
                        "filepath": bpy.path.abspath(node.image.filepath),
                    })
        material_slots.append({
            "name": material.name if material is not None else None,
            "diffuse_color": [float(c) for c in material.diffuse_color] if material is not None else None,
            "images": images,
        })
    return {
        "object": obj.name,
        "vertices": vertices,
        "faces": faces,
        "face_material_indices": [int(poly.material_index) for poly in mesh.polygons],
        "material_slots": material_slots,
        "uv_layers": [{"name": name, "values": values} for name, values in uvs.items()],
        "color_layers": [{"name": name, **layer} for name, layer in colors.items()],
        "corners": corners,
        "vertex_groups": groups,
    }


result = {}
for source in sources:
    extension = os.path.splitext(source)[1].lower()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    importer = IMPORTERS.get(extension)
    if importer is None:
        raise RuntimeError("no independent Blender importer for " + extension)
    importer(filepath=source)
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(objects) != 1:
        raise RuntimeError("expected one mesh from %s, found %d" % (source, len(objects)))
    result[os.path.basename(source)] = dump_mesh(objects[0])
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"#;

const FBX_EMPTY_SLOT_ORACLE_SCRIPT: &str = r#"
import json
import struct
import sys
import zlib

import bpy


def read_property(data, cursor, node_name):
    kind = chr(data[cursor])
    cursor += 1
    if kind in ("S", "R"):
        size = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        raw = data[cursor : cursor + size]
        cursor += size
        return (raw.decode("utf-8") if kind == "S" else None), cursor
    scalar_sizes = {"Y": 2, "C": 1, "I": 4, "F": 4, "D": 8, "L": 8}
    if kind in scalar_sizes:
        size = scalar_sizes[kind]
        return None, cursor + size
    if kind in ("f", "d", "i", "l", "b", "c"):
        count, encoding, size = struct.unpack_from("<III", data, cursor)
        cursor += 12
        raw = data[cursor : cursor + size]
        cursor += size
        if node_name not in ("Materials", "a"):
            return None, cursor
        if encoding == 1:
            raw = zlib.decompress(raw)
        elif encoding != 0:
            raise RuntimeError("unknown FBX array encoding %d" % encoding)
        if kind == "i":
            return list(struct.unpack("<%di" % count, raw)), cursor
        return None, cursor
    raise RuntimeError("unsupported FBX property type %r" % kind)


def read_node(data, cursor, version):
    wide = version >= 7500
    header_size = 25 if wide else 13
    null_record = b"\0" * header_size
    if data[cursor : cursor + header_size] == null_record:
        return None, cursor + header_size
    if wide:
        end, property_count, property_size = struct.unpack_from("<QQQ", data, cursor)
        name_size = data[cursor + 24]
    else:
        end, property_count, property_size = struct.unpack_from("<III", data, cursor)
        name_size = data[cursor + 12]
    name_start = cursor + header_size
    name = data[name_start : name_start + name_size].decode("utf-8")
    cursor = name_start + name_size
    property_end = cursor + property_size
    arrays = []
    for _ in range(property_count):
        value, cursor = read_property(data, cursor, name)
        if value is not None:
            arrays.append(value)
    if cursor != property_end:
        raise RuntimeError("FBX property length mismatch for " + name)
    children = []
    while cursor < end:
        child, cursor = read_node(data, cursor, version)
        if child is None:
            break
        children.append(child)
    if cursor != end:
        raise RuntimeError("FBX node end offset mismatch for " + name)
    return {"name": name, "arrays": arrays, "children": children}, cursor


def find_node(nodes, name):
    for node in nodes:
        if node["name"] == name:
            return node
        child = find_node(node["children"], name)
        if child is not None:
            return child
    return None


def decode_material_indices(path):
    data = open(path, "rb").read()
    if not data.startswith(b"Kaydara FBX Binary  \0\x1a\0"):
        raise RuntimeError("Blender did not write binary FBX")
    version = struct.unpack_from("<I", data, 23)[0]
    cursor = 27
    roots = []
    while cursor < len(data):
        node, cursor = read_node(data, cursor, version)
        if node is None:
            break
        roots.append(node)
    layer = find_node(roots, "LayerElementMaterial")
    if layer is None:
        raise RuntimeError("FBX has no LayerElementMaterial")
    materials = next((child for child in layer["children"] if child["name"] == "Materials"), None)
    if materials is None:
        raise RuntimeError("FBX material layer has no Materials array")
    if materials["arrays"]:
        return materials["arrays"][0]
    array_node = next((child for child in materials["children"] if child["name"] == "a"), None)
    if array_node is None or not array_node["arrays"]:
        raise RuntimeError("FBX Materials array was not decoded")
    return array_node["arrays"][0]


source_blend, fbx_path, out_path = sys.argv[sys.argv.index("--") + 1 :][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("FbxMaterialOracle")
mesh.from_pydata(
    [(-1, -1, 0), (0, -1, 0), (1, -1, 0), (1, 1, 0), (0, 1, 0), (-1, 1, 0)],
    [],
    [(0, 5, 4, 1), (1, 4, 3, 2)],
)
mesh.update()
obj = bpy.data.objects.new("FbxMaterialOracle", mesh)
bpy.context.scene.collection.objects.link(obj)
bound = bpy.data.materials.new("Bound")
bound.diffuse_color = (0.8, 0.4, 0.2, 1.0)
mesh.materials.append(bound)
mesh.materials.append(None)
mesh.polygons[0].material_index = 0
mesh.polygons[1].material_index = 1
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
source_slots = [slot.material.name if slot.material is not None else None for slot in obj.material_slots]
source_face_material_indices = [int(poly.material_index) for poly in mesh.polygons]
bpy.ops.wm.save_as_mainfile(filepath=source_blend)
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=True,
    object_types={"MESH"},
    add_leaf_bones=False,
)
raw_material_indices = decode_material_indices(fbx_path)

# Blender's FBX importer drops the unconnected slot and resets the -1 polygon index to the first
# connected material.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_path)
objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if len(objects) != 1:
    raise RuntimeError("expected one imported FBX mesh, found %d" % len(objects))
imported = objects[0]
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump({
        "source_slots": source_slots,
        "source_face_material_indices": source_face_material_indices,
        "raw_material_indices": raw_material_indices,
        "blender_slots": [
            slot.material.name if slot.material is not None else None
            for slot in imported.material_slots
        ],
        "blender_face_material_indices": [int(poly.material_index) for poly in imported.data.polygons],
    }, handle)
"#;

#[derive(Clone, Copy)]
struct FormatCase {
    format: &'static str,
    extension: &'static str,
    uv_layers: &'static [(&'static str, &'static str)],
    color_layers: &'static [(&'static str, &'static str)],
    color_encoding: ColorEncoding,
    groups_from_blender: bool,
}

#[derive(Clone, Copy)]
enum ColorEncoding {
    /// OBJ `v x y z r g b`: the sRGB-encoded RGB is read back as written; alpha is opaque.
    ObjSrgb,
    /// PLY `uchar` colors: sRGB RGB bytes (floored) and a linear alpha byte.
    PlySrgbBytes,
    /// USD `color4f` in `lin_rec709_scene`: exact linear floats.
    Linear,
    /// FBX and Alembic `BYTE_COLOR`: sRGB RGB and linear alpha, each rounded to the nearest byte.
    ByteColor,
}

const ONE_ACTIVE_UV: &[(&str, &str)] = &[("UVMap", "Detail")];
const USD_UVS: &[(&str, &str)] = &[("Base", "Base"), ("Detail", "Detail"), ("st", "Light")];
const FULL_UVS: &[(&str, &str)] = &[("Base", "Base"), ("Light", "Light"), ("Detail", "Detail")];
const ALEMBIC_UVS: &[(&str, &str)] = &[("Detail", "Detail"), ("Base", "Base"), ("Light", "Light")];
const ONLY_ACTIVE_COLOR: &[(&str, &str)] = &[("Color", "Glow")];
const ONLY_ACTIVE_PLY_COLOR: &[(&str, &str)] = &[("Col", "Glow")];
const FULL_COLORS: &[(&str, &str)] = &[
    ("Paint", "Paint"),
    ("Glow", "Glow"),
    ("Dirt", "Dirt"),
    ("Trim", "Trim"),
];

const CASES: [FormatCase; 5] = [
    FormatCase {
        format: "obj",
        extension: "obj",
        uv_layers: ONE_ACTIVE_UV,
        color_layers: ONLY_ACTIVE_COLOR,
        color_encoding: ColorEncoding::ObjSrgb,
        groups_from_blender: true,
    },
    FormatCase {
        format: "ply",
        extension: "ply",
        uv_layers: ONE_ACTIVE_UV,
        color_layers: ONLY_ACTIVE_PLY_COLOR,
        color_encoding: ColorEncoding::PlySrgbBytes,
        groups_from_blender: false,
    },
    FormatCase {
        format: "usda",
        extension: "usda",
        uv_layers: USD_UVS,
        color_layers: FULL_COLORS,
        color_encoding: ColorEncoding::Linear,
        groups_from_blender: false,
    },
    FormatCase {
        format: "fbx-binary",
        extension: "fbx",
        uv_layers: FULL_UVS,
        color_layers: FULL_COLORS,
        color_encoding: ColorEncoding::ByteColor,
        groups_from_blender: false,
    },
    FormatCase {
        format: "alembic",
        extension: "abc",
        uv_layers: ALEMBIC_UVS,
        color_layers: FULL_COLORS,
        color_encoding: ColorEncoding::ByteColor,
        groups_from_blender: false,
    },
];

#[derive(Clone, Debug)]
struct Corner {
    vertex: usize,
    position: [f64; 3],
    normal: [f64; 3],
    uvs: BTreeMap<String, [f64; 2]>,
    colors: BTreeMap<String, [f64; 4]>,
}

#[derive(Clone, Debug)]
struct ColorLayer {
    domain: String,
    values: Vec<[f64; 4]>,
}

#[derive(Clone, Debug)]
struct DumpedMesh {
    vertices: Vec<[f64; 3]>,
    faces: Vec<Vec<usize>>,
    corners: Vec<Corner>,
    uv_layers: BTreeMap<String, Vec<[f64; 2]>>,
    color_layers: BTreeMap<String, ColorLayer>,
    groups: BTreeMap<String, BTreeMap<usize, f64>>,
}

#[derive(Clone, Debug)]
struct OracleMesh {
    corners: Vec<Corner>,
    faces: Vec<Vec<usize>>,
    vertex_count: usize,
    vertex_positions: Vec<[f64; 3]>,
    groups: BTreeMap<String, BTreeMap<usize, f64>>,
    colors: BTreeMap<String, ColorLayer>,
    /// The color each vertex carries in a point-domain writer: a POINT layer's own value, or the
    /// average of a CORNER layer's incident corners. Both are linear RGBA.
    point_values: BTreeMap<String, Vec<[f64; 4]>>,
    uvs: BTreeMap<String, Vec<[f64; 2]>>,
}

fn number(value: &Value) -> TestResult<f64> {
    value
        .as_f64()
        .ok_or_else(|| "expected a JSON number".into())
}

fn vec2(value: &Value) -> TestResult<[f64; 2]> {
    let values = value.as_array().ok_or("expected a vec2")?;
    if values.len() != 2 {
        return Err("expected a vec2".into());
    }
    Ok([number(&values[0])?, number(&values[1])?])
}

fn vec3(value: &Value) -> TestResult<[f64; 3]> {
    let values = value.as_array().ok_or("expected a vec3")?;
    if values.len() != 3 {
        return Err("expected a vec3".into());
    }
    Ok([
        number(&values[0])?,
        number(&values[1])?,
        number(&values[2])?,
    ])
}

fn vec4(value: &Value) -> TestResult<[f64; 4]> {
    let values = value.as_array().ok_or("expected a vec4")?;
    if values.len() != 4 {
        return Err("expected a vec4".into());
    }
    Ok([
        number(&values[0])?,
        number(&values[1])?,
        number(&values[2])?,
        number(&values[3])?,
    ])
}

fn parse_index_array(value: &Value) -> TestResult<Vec<usize>> {
    value
        .as_array()
        .ok_or("expected an index array")?
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| "invalid mesh index".into())
        })
        .collect()
}

fn parse_dump_mesh(value: &Value) -> TestResult<DumpedMesh> {
    let vertices = value["vertices"]
        .as_array()
        .ok_or("Blender dump has no vertices")?
        .iter()
        .map(vec3)
        .collect::<TestResult<Vec<_>>>()?;
    let faces = value["faces"]
        .as_array()
        .ok_or("Blender dump has no faces")?
        .iter()
        .map(parse_index_array)
        .collect::<TestResult<Vec<_>>>()?;
    let mut uv_layers = BTreeMap::new();
    for layer in value["uv_layers"].as_array().ok_or("UV layers missing")? {
        let name = layer["name"]
            .as_str()
            .ok_or("UV layer name missing")?
            .to_owned();
        let values = layer["values"]
            .as_array()
            .ok_or("UV values missing")?
            .iter()
            .map(vec2)
            .collect::<TestResult<Vec<_>>>()?;
        uv_layers.insert(name, values);
    }
    let mut color_layers = BTreeMap::new();
    for layer in value["color_layers"]
        .as_array()
        .ok_or("color layers missing")?
    {
        let name = layer["name"]
            .as_str()
            .ok_or("color layer name missing")?
            .to_owned();
        let values = layer["values"]
            .as_array()
            .ok_or("color values missing")?
            .iter()
            .map(vec4)
            .collect::<TestResult<Vec<_>>>()?;
        color_layers.insert(
            name,
            ColorLayer {
                domain: layer["domain"]
                    .as_str()
                    .ok_or("color domain missing")?
                    .to_owned(),
                values,
            },
        );
    }
    let mut corners = Vec::new();
    for corner in value["corners"].as_array().ok_or("corner data missing")? {
        let vertex = corner["vertex"]
            .as_u64()
            .and_then(|index| usize::try_from(index).ok())
            .ok_or("corner vertex is invalid")?;
        let uvs = corner["uvs"]
            .as_object()
            .ok_or("corner UVs missing")?
            .iter()
            .map(|(name, uv)| Ok((name.clone(), vec2(uv)?)))
            .collect::<TestResult<BTreeMap<_, _>>>()?;
        let colors = corner["colors"]
            .as_object()
            .ok_or("corner colors missing")?
            .iter()
            .map(|(name, color)| Ok((name.clone(), vec4(color)?)))
            .collect::<TestResult<BTreeMap<_, _>>>()?;
        corners.push(Corner {
            vertex,
            position: vec3(&corner["position"])?,
            normal: vec3(&corner["normal"])?,
            uvs,
            colors,
        });
    }
    let mut groups = BTreeMap::new();
    for (name, members) in value["vertex_groups"]
        .as_object()
        .ok_or("vertex groups missing")?
    {
        let members = members.as_object().ok_or("group members missing")?;
        let weights = members
            .iter()
            .map(|(index, weight)| Ok((index.parse::<usize>()?, number(weight)?)))
            .collect::<TestResult<BTreeMap<_, _>>>()?;
        groups.insert(name.clone(), weights);
    }
    Ok(DumpedMesh {
        vertices,
        faces,
        corners,
        uv_layers,
        color_layers,
        groups,
    })
}

/// The average of each vertex's incident corner values of the CORNER layer `name`, in linear RGBA.
/// This is Blender's POINT lookup of CORNER data, and what OBJ and PLY write per vertex.
fn average_corner_colors(corners: &[Corner], name: &str, vertex_count: usize) -> Vec<[f64; 4]> {
    let mut sums = vec![[0.0_f32; 4]; vertex_count];
    let mut counts = vec![0_usize; vertex_count];
    for corner in corners {
        for (sum, value) in sums[corner.vertex].iter_mut().zip(corner.colors[name]) {
            *sum += value as f32;
        }
        counts[corner.vertex] += 1;
    }
    for (sum, count) in sums.iter_mut().zip(counts) {
        if count > 0 {
            let inverse = 1.0_f32 / count as f32;
            for value in sum.iter_mut() {
                *value *= inverse;
            }
        }
    }
    sums.into_iter().map(|sum| sum.map(f64::from)).collect()
}

fn parse_source_oracle(expected: &Value, source_dump: &Value) -> TestResult<OracleMesh> {
    let oracle = oracle_section(expected);
    let object = dumped_object(source_dump, "Oracle")?;
    let vertices = oracle["vertices"]
        .as_array()
        .ok_or("oracle vertices missing")?;
    let faces = oracle["faces"]
        .as_array()
        .ok_or("oracle faces missing")?
        .iter()
        .map(|face| parse_index_array(&face["v"]))
        .collect::<TestResult<Vec<_>>>()?;
    let source_positions = vertices.iter().map(vec3).collect::<TestResult<Vec<_>>>()?;
    let source_normals = object["corner_normals"]
        .as_array()
        .ok_or("source corner normals missing")?
        .iter()
        .map(vec3)
        .collect::<TestResult<Vec<_>>>()?;
    let uvs = oracle["uv_layers"]["values"]
        .as_object()
        .ok_or("oracle UV layers missing")?
        .iter()
        .map(|(name, values)| {
            Ok((
                name.clone(),
                values
                    .as_array()
                    .ok_or("oracle UV values missing")?
                    .iter()
                    .map(vec2)
                    .collect::<TestResult<Vec<_>>>()?,
            ))
        })
        .collect::<TestResult<BTreeMap<_, _>>>()?;
    let colors = oracle["color_layers"]["layers"]
        .as_object()
        .ok_or("oracle color layers missing")?
        .iter()
        .map(|(name, layer)| {
            Ok((
                name.clone(),
                ColorLayer {
                    domain: layer["domain"]
                        .as_str()
                        .ok_or("oracle color domain missing")?
                        .to_owned(),
                    values: layer["values"]
                        .as_array()
                        .ok_or("oracle color values missing")?
                        .iter()
                        .map(vec4)
                        .collect::<TestResult<Vec<_>>>()?,
                },
            ))
        })
        .collect::<TestResult<BTreeMap<_, _>>>()?;
    let groups = oracle["vertex_groups"]
        .as_object()
        .ok_or("oracle groups missing")?
        .iter()
        .map(|(name, members)| {
            let members = members.as_object().ok_or("oracle group members missing")?;
            Ok((
                name.clone(),
                members
                    .iter()
                    .map(|(index, weight)| Ok((index.parse::<usize>()?, number(weight)?)))
                    .collect::<TestResult<BTreeMap<_, _>>>()?,
            ))
        })
        .collect::<TestResult<BTreeMap<_, _>>>()?;
    let loop_count = faces.iter().map(Vec::len).sum::<usize>();
    assert_eq!(
        source_positions.len(),
        object["vertices"]
            .as_array()
            .ok_or("source vertices missing")?
            .len()
    );
    assert_eq!(source_normals.len(), loop_count, "source normal count");
    for (index, (actual, expected)) in object["vertices"]
        .as_array()
        .ok_or("source vertices missing")?
        .iter()
        .zip(&source_positions)
        .enumerate()
    {
        assert_floats(
            &native_oracle::floats(actual)?,
            expected,
            1.0e-6,
            &format!("source vertex {index}"),
        );
    }
    let mut corners = Vec::with_capacity(loop_count);
    let mut loop_index = 0;
    for face in &faces {
        for &vertex in face {
            let point = *source_positions
                .get(vertex)
                .ok_or("source face has invalid vertex")?;
            let normal = source_normals[loop_index];
            let corner_uvs = uvs
                .iter()
                .map(|(name, values)| {
                    Ok((
                        name.clone(),
                        *values
                            .get(loop_index)
                            .ok_or("source UV is missing a loop")?,
                    ))
                })
                .collect::<TestResult<BTreeMap<_, _>>>()?;
            let corner_colors = colors
                .iter()
                .map(|(name, layer)| {
                    let index = if layer.domain.eq_ignore_ascii_case("point") {
                        vertex
                    } else {
                        loop_index
                    };
                    Ok((
                        name.clone(),
                        *layer
                            .values
                            .get(index)
                            .ok_or("source color is missing a value")?,
                    ))
                })
                .collect::<TestResult<BTreeMap<_, _>>>()?;
            corners.push(Corner {
                vertex,
                position: point,
                normal,
                uvs: corner_uvs,
                colors: corner_colors,
            });
            loop_index += 1;
        }
    }
    let vertex_count = source_positions.len();
    let point_values: BTreeMap<String, Vec<[f64; 4]>> = colors
        .iter()
        .map(|(name, layer)| {
            let values = if layer.domain.eq_ignore_ascii_case("point") {
                layer.values.clone()
            } else {
                average_corner_colors(&corners, name, vertex_count)
            };
            (name.clone(), values)
        })
        .collect();
    Ok(OracleMesh {
        corners,
        faces,
        vertex_count,
        vertex_positions: source_positions,
        groups,
        colors,
        point_values,
        uvs,
    })
}

fn srgb_encode(value: f64) -> f64 {
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_decode(value: f64) -> f64 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// PLY's `uchar` color: sRGB-encoded RGB or linear alpha, clamped and floored to a byte.
fn ply_byte(value: f64, alpha: bool) -> u8 {
    let encoded = if alpha { value } else { srgb_encode(value) };
    (encoded.clamp(0.0, 1.0) * 255.0).floor() as u8
}

/// Blender's nearest-byte rounding of a `[0, 1]` value, as an integer-valued float.
fn nearest_byte(value: f64) -> f64 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5).floor()
}

/// A linear RGBA value after one `BYTE_COLOR` round trip: sRGB RGB and linear alpha, each rounded
/// to the nearest byte, then read back by Blender as linear light.
fn byte_color(value: [f64; 4]) -> [f64; 4] {
    [
        srgb_decode(nearest_byte(srgb_encode(value[0])) / 255.0),
        srgb_decode(nearest_byte(srgb_encode(value[1])) / 255.0),
        srgb_decode(nearest_byte(srgb_encode(value[2])) / 255.0),
        nearest_byte(value[3]) / 255.0,
    ]
}

/// OBJ writes nonnegative sRGB-encoded RGB; alpha is not written.
fn obj_rgb(value: [f64; 4]) -> [f64; 3] {
    [
        srgb_encode(value[0].max(0.0)),
        srgb_encode(value[1].max(0.0)),
        srgb_encode(value[2].max(0.0)),
    ]
}

fn expected_import_color(value: [f64; 4], encoding: ColorEncoding) -> [f64; 4] {
    match encoding {
        ColorEncoding::ObjSrgb => {
            let [red, green, blue] = obj_rgb(value);
            [red, green, blue, 1.0]
        }
        ColorEncoding::PlySrgbBytes => [
            srgb_decode(f64::from(ply_byte(value[0], false)) / 255.0),
            srgb_decode(f64::from(ply_byte(value[1], false)) / 255.0),
            srgb_decode(f64::from(ply_byte(value[2], false)) / 255.0),
            f64::from(ply_byte(value[3], true)) / 255.0,
        ],
        ColorEncoding::Linear => value,
        ColorEncoding::ByteColor => byte_color(value),
    }
}

/// The source color a Blender corner must match. OBJ and PLY write one color per vertex, the
/// average of its incident corners; the other formats keep each element's own value.
fn source_color(case: FormatCase, source: &OracleMesh, corner: &Corner, name: &str) -> [f64; 4] {
    if matches!(case.format, "obj" | "ply") {
        source.point_values[name][corner.vertex]
    } else {
        corner.colors[name]
    }
}

fn vec2_close(actual: [f64; 2], expected: [f64; 2], tolerance: f64) -> bool {
    actual
        .iter()
        .zip(expected)
        .all(|(a, b)| (a - b).abs() <= tolerance)
}

fn vec3_close(actual: [f64; 3], expected: [f64; 3], tolerance: f64) -> bool {
    actual
        .iter()
        .zip(expected)
        .all(|(a, b)| (a - b).abs() <= tolerance)
}

fn vec4_close(actual: [f64; 4], expected: [f64; 4], tolerance: f64) -> bool {
    actual
        .iter()
        .zip(expected)
        .all(|(a, b)| (a - b).abs() <= tolerance)
}

fn has_uv_seam(corners: &[Corner], layer: &str) -> bool {
    corners.iter().enumerate().any(|(left_index, left)| {
        let Some(left_uv) = left.uvs.get(layer) else {
            return false;
        };
        corners.iter().skip(left_index + 1).any(|right| {
            vec3_close(left.position, right.position, 1.0e-6)
                && right
                    .uvs
                    .get(layer)
                    .is_some_and(|right_uv| !vec2_close(*left_uv, *right_uv, 1.0e-6))
        })
    })
}

fn signed_volume(vertices: &[[f64; 3]], faces: &[Vec<usize>]) -> f64 {
    let mut volume = 0.0;
    for face in faces {
        let Some(&first) = face.first() else { continue };
        for pair in face[1..].windows(2) {
            let a = vertices[first];
            let b = vertices[pair[0]];
            let c = vertices[pair[1]];
            volume += dot(a, cross(b, c)) / 6.0;
        }
    }
    volume
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn loss_values(report: &Value) -> TestResult<Vec<Value>> {
    let values = report["result"]["losses"]
        .as_array()
        .or_else(|| report["error"]["details"]["result"]["losses"].as_array())
        .ok_or_else(|| format!("loss list missing from report: {report}"))?;
    Ok(values.clone())
}

fn loss_count(losses: &[Value], feature: &str) -> usize {
    losses
        .iter()
        .filter(|loss| loss["feature_id"].as_str() == Some(feature))
        .count()
}

fn group_loss_names(losses: &[Value]) -> BTreeSet<String> {
    losses
        .iter()
        .filter(|loss| loss["feature_id"] == "mesh.vertex_group")
        .filter_map(|loss| loss["reason"].as_str())
        .filter_map(|reason| reason.split("vertex group `").nth(1)?.split('`').next())
        .map(str::to_owned)
        .collect()
}

fn assert_loss_parity(
    case: FormatCase,
    project: &Path,
    blender: &Path,
    root: &Path,
) -> TestResult<(Vec<Value>, Value, PathBuf)> {
    let validation_output = pot()
        .arg("validate")
        .arg(project)
        .args(["--format", case.format, "--strict", "--json"])
        .output()?;
    let validation: Value = serde_json::from_slice(&validation_output.stdout)?;
    let validation_losses = loss_values(&validation)?;
    assert_eq!(
        validation_output.status.success(),
        validation_losses.is_empty(),
        "{} strict validation status: {validation}",
        case.format
    );

    let strict_path = root.join(format!("strict.{}", case.extension));
    let strict_output = export_raw(blender, project, case.format, &strict_path, false)?;
    let strict_report: Value = serde_json::from_slice(&strict_output.stdout)?;
    let strict_losses = loss_values(&strict_report)?;
    assert_eq!(
        strict_losses, validation_losses,
        "{} strict validate/export loss set",
        case.format
    );
    assert_eq!(
        strict_output.status.success(),
        validation_losses.is_empty(),
        "{} strict export status: {strict_report}",
        case.format
    );
    if !validation_losses.is_empty() {
        assert_eq!(
            strict_report["error"]["code"], "UNREPRESENTABLE_FEATURE",
            "{strict_report}"
        );
    }

    let allow_path = root.join(format!("allow.{}", case.extension));
    let allow_output = export_raw(blender, project, case.format, &allow_path, true)?;
    assert!(
        allow_output.status.success(),
        "{} --allow-lossy export failed: stdout={} stderr={}",
        case.format,
        String::from_utf8_lossy(&allow_output.stdout),
        String::from_utf8_lossy(&allow_output.stderr),
    );
    let allow_report: Value = serde_json::from_slice(&allow_output.stdout)?;
    assert_eq!(
        loss_values(&allow_report)?,
        validation_losses,
        "{} allow-lossy report",
        case.format
    );
    assert!(
        fs::metadata(&allow_path)?.len() > 0,
        "{} output is empty",
        case.format
    );
    let reported_bytes = allow_report["result"]["bytes"]
        .as_u64()
        .ok_or("export byte count missing")?;
    assert!(
        reported_bytes >= fs::metadata(&allow_path)?.len(),
        "{} byte count is too small",
        case.format
    );

    let inactive_uv_losses = validation_losses
        .iter()
        .filter(|loss| loss["feature_id"] == "mesh.uv_layer")
        .filter_map(|loss| loss["reason"].as_str())
        .filter(|reason| {
            reason.contains("UV layer `Base` is lost")
                || reason.contains("UV layer `Light` is lost")
        })
        .count();
    assert_eq!(
        inactive_uv_losses,
        if matches!(case.format, "obj" | "ply") {
            2
        } else {
            0
        },
        "{} inactive UV layer losses",
        case.format
    );
    // Per color layer, the rules are these. OBJ keeps only the active layer: the three others are
    // lost, and Glow adds a rename to `Color`, a CORNER-to-POINT average, a render-designation loss,
    // and non-opaque alpha. PLY adds the same rename, average, and designation losses, plus RGB and
    // alpha quantization of the averaged values. FBX and Alembic quantize the FLOAT layers (Paint,
    // Glow), clip Glow's HDR value, and import POINT layers as CORNER; Alembic also keeps only the
    // first layer's selections. USD writes linear float colors and loses none of them.
    assert_eq!(
        loss_count(&validation_losses, "mesh.color_attribute"),
        match case.format {
            "obj" => 7,
            "ply" => 8,
            "fbx-binary" => 5,
            "alembic" => 6,
            _ => 0,
        },
        "{} color losses",
        case.format
    );
    assert_eq!(
        loss_count(&validation_losses, "mesh.vertex_group"),
        match case.format {
            // OBJ drops the sparse Side group and the Top group, which splits side faces. PLY and
            // Alembic have no vertex-weight channel. FBX writes groups only as clusters on an armature
            // skin, so this oracle's two groups are lost. USD keeps them in its deformation payload.
            "obj" | "ply" | "alembic" | "fbx-binary" => 2,
            _ => 0,
        },
        "{} group losses",
        case.format
    );
    Ok((validation_losses, allow_report, allow_path))
}

#[derive(Clone, Debug)]
struct ObjFile {
    vertices: Vec<Vec<f64>>,
    uvs: Vec<[f64; 2]>,
    normals: Vec<[f64; 3]>,
    faces: Vec<Vec<[usize; 3]>>,
    groups: BTreeSet<String>,
}

fn decode_obj(path: &Path, source: &OracleMesh) -> TestResult<ObjFile> {
    let text = fs::read_to_string(path)?;
    let mut decoded = ObjFile {
        vertices: Vec::new(),
        uvs: Vec::new(),
        normals: Vec::new(),
        faces: Vec::new(),
        groups: BTreeSet::new(),
    };
    let mut corner_count = 0;
    for line in text.lines() {
        let mut words = line.split_whitespace();
        match words.next() {
            Some("v") => decoded
                .vertices
                .push(words.map(str::parse).collect::<Result<Vec<f64>, _>>()?),
            Some("vt") => {
                let values = words.map(str::parse).collect::<Result<Vec<f64>, _>>()?;
                assert_eq!(values.len(), 2, "OBJ vt must contain two values");
                decoded.uvs.push([values[0], values[1]]);
            }
            Some("vn") => {
                let values = words.map(str::parse).collect::<Result<Vec<f64>, _>>()?;
                assert_eq!(values.len(), 3, "OBJ vn must contain three values");
                decoded.normals.push([values[0], values[1], values[2]]);
            }
            Some("f") => {
                let face = words
                    .map(|token| {
                        let fields = token.split('/').collect::<Vec<_>>();
                        if fields.len() != 3 {
                            return Err("OBJ face corner must index position/UV/normal".into());
                        }
                        let index = |field: &str, count: usize| -> TestResult<usize> {
                            let one_based = field.parse::<usize>()?;
                            if one_based == 0 || one_based > count {
                                return Err("OBJ face index out of range".into());
                            }
                            Ok(one_based - 1)
                        };
                        Ok([
                            index(fields[0], decoded.vertices.len())?,
                            index(fields[1], decoded.uvs.len())?,
                            index(fields[2], decoded.normals.len())?,
                        ])
                    })
                    .collect::<TestResult<Vec<_>>>()?;
                corner_count += face.len();
                decoded.faces.push(face);
            }
            // `g` takes the whole remainder of the line as one group name, which may hold spaces.
            Some("g") => {
                let name = line.trim().strip_prefix('g').unwrap_or_default().trim();
                if !name.is_empty() {
                    decoded.groups.insert(name.to_owned());
                }
            }
            _ => {}
        }
    }
    assert_eq!(
        decoded.faces.len(),
        source.faces.len(),
        "OBJ decoded face count"
    );
    assert_eq!(
        corner_count,
        source.corners.len(),
        "OBJ decoded corner count"
    );
    assert_eq!(
        decoded.vertices.len(),
        source.vertex_count,
        "OBJ shared position vertex count"
    );
    assert_eq!(
        decoded.uvs.len(),
        source.corners.len(),
        "OBJ active UV table count"
    );
    assert!(!decoded.normals.is_empty(), "OBJ has no vn records");
    let glow = &source.point_values["Glow"];
    for vertex in &decoded.vertices {
        assert_eq!(vertex.len(), 6, "OBJ vertex must carry xyzrgb");
        assert!(
            (0..source.vertex_count).any(|index| {
                vec3_close(
                    [vertex[0], vertex[1], vertex[2]],
                    obj_axes(source.vertex_positions[index]),
                    TOLERANCE,
                ) && vec3_close(
                    [vertex[3], vertex[4], vertex[5]],
                    obj_rgb(glow[index]),
                    COLOR_TOLERANCE,
                )
            }),
            "OBJ xyzrgb record does not match an oracle vertex and its averaged color: {vertex:?}"
        );
    }
    for uv in &decoded.uvs {
        assert!(
            source.uvs["Detail"]
                .iter()
                .any(|expected| vec2_close(*uv, *expected, TOLERANCE)),
            "OBJ vt does not match active Detail UV: {uv:?}"
        );
    }
    for normal in &decoded.normals {
        assert!(
            source.corners.iter().any(|corner| vec3_close(
                *normal,
                obj_axes(corner.normal),
                NORMAL_TOLERANCE
            )),
            "OBJ vn does not match an oracle normal: {normal:?}"
        );
    }
    let native_vertices = decoded
        .vertices
        .iter()
        .map(|vertex| [vertex[0], -vertex[2], vertex[1]])
        .collect::<Vec<_>>();
    let native_faces = decoded
        .faces
        .iter()
        .map(|face| face.iter().map(|indices| indices[0]).collect())
        .collect::<Vec<Vec<usize>>>();
    assert!(
        signed_volume(&native_vertices, &native_faces) > 7.99,
        "OBJ face indices reverse/mirror the source shell"
    );
    assert!(obj_has_uv_seam(&decoded), "OBJ lost the active UV seam");
    Ok(decoded)
}

fn obj_axes(value: [f64; 3]) -> [f64; 3] {
    [value[0], value[2], -value[1]]
}

fn obj_has_uv_seam(mesh: &ObjFile) -> bool {
    let mut corners = Vec::new();
    for face in &mesh.faces {
        for indices in face {
            corners.push((
                mesh.vertices[indices[0]][..3].to_vec(),
                mesh.uvs[indices[1]],
            ));
        }
    }
    corners.iter().enumerate().any(|(index, (position, uv))| {
        corners
            .iter()
            .skip(index + 1)
            .any(|(other_position, other_uv)| {
                (0..3).all(|axis| (position[axis] - other_position[axis]).abs() < 1.0e-6)
                    && !vec2_close(*uv, *other_uv, 1.0e-6)
            })
    })
}

#[derive(Clone, Debug)]
struct PlyVertex {
    values: BTreeMap<String, f64>,
}

struct PlyFile {
    vertices: Vec<PlyVertex>,
}

/// PLY writes one vertex per distinct (position, active UV) pair. The UV is stored as float32, so
/// the key uses those float32 bits.
fn ply_split_vertex_count(source: &OracleMesh) -> usize {
    source
        .corners
        .iter()
        .map(|corner| {
            let uv = corner.uvs["Detail"];
            (
                corner.vertex,
                (uv[0] as f32).to_bits(),
                (uv[1] as f32).to_bits(),
            )
        })
        .collect::<BTreeSet<_>>()
        .len()
}

fn decode_ply(path: &Path, source: &OracleMesh) -> TestResult<PlyFile> {
    let bytes = fs::read(path)?;
    let end = bytes
        .windows(b"end_header\n".len())
        .position(|window| window == b"end_header\n")
        .ok_or("PLY header terminator missing")?
        + b"end_header\n".len();
    let header = std::str::from_utf8(&bytes[..end])?;
    assert!(
        header
            .lines()
            .any(|line| line == "format binary_little_endian 1.0"),
        "PLY is not binary little-endian"
    );
    let mut element = "";
    let mut vertex_count = None;
    let mut face_count = None;
    let mut vertex_properties = Vec::new();
    let mut face_index_types = None;
    for line in header.lines() {
        let words = line.split_whitespace().collect::<Vec<_>>();
        match words.as_slice() {
            ["element", "vertex", count] => {
                vertex_count = Some(count.parse::<usize>()?);
                element = "vertex";
            }
            ["element", "face", count] => {
                face_count = Some(count.parse::<usize>()?);
                element = "face";
            }
            ["property", scalar, name] if element == "vertex" => {
                vertex_properties.push(((*name).to_owned(), (*scalar).to_owned()));
            }
            ["property", "list", count_type, item_type, "vertex_indices"] if element == "face" => {
                face_index_types = Some(((*count_type).to_owned(), (*item_type).to_owned()));
            }
            _ => {}
        }
    }
    let vertex_count = vertex_count.ok_or("PLY vertex count missing")?;
    let face_count = face_count.ok_or("PLY face count missing")?;
    let mut cursor = end;
    let mut vertices = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let mut values = BTreeMap::new();
        for (name, kind) in &vertex_properties {
            values.insert(name.clone(), read_ply_scalar(&bytes, &mut cursor, kind)?);
        }
        vertices.push(PlyVertex { values });
    }
    let (count_type, item_type) = face_index_types.ok_or("PLY face index property missing")?;
    let mut faces = Vec::with_capacity(face_count);
    for _ in 0..face_count {
        let count = read_ply_scalar(&bytes, &mut cursor, &count_type)? as usize;
        let mut face = Vec::with_capacity(count);
        for _ in 0..count {
            let index = read_ply_scalar(&bytes, &mut cursor, &item_type)? as usize;
            if index >= vertex_count {
                return Err("PLY face index out of range".into());
            }
            face.push(index);
        }
        faces.push(face);
    }
    assert_eq!(
        cursor,
        bytes.len(),
        "PLY decoder left trailing payload bytes"
    );
    assert_eq!(faces.len(), source.faces.len(), "PLY face count");
    assert_eq!(
        faces.iter().map(Vec::len).sum::<usize>(),
        source.corners.len(),
        "PLY face corner count"
    );
    assert!(
        faces
            .iter()
            .zip(&source.faces)
            .all(|(actual, expected)| actual.len() == expected.len()),
        "PLY polygon sizes changed"
    );
    for property in ["x", "y", "z", "red", "green", "blue", "alpha", "s", "t"] {
        assert!(
            vertex_properties.iter().any(|(name, _)| name == property),
            "PLY missing {property} property"
        );
    }
    assert!(
        vertex_properties.iter().any(|(name, _)| name == "nx"),
        "PLY missing vertex normals"
    );
    assert_eq!(
        vertices.len(),
        ply_split_vertex_count(source),
        "PLY (position, UV) split vertex count"
    );
    for vertex in &vertices {
        let position = [vertex.values["x"], vertex.values["y"], vertex.values["z"]];
        let uv = [vertex.values["s"], vertex.values["t"]];
        let color = [
            vertex.values["red"] as u8,
            vertex.values["green"] as u8,
            vertex.values["blue"] as u8,
            vertex.values["alpha"] as u8,
        ];
        assert!(
            source.corners.iter().any(|corner| {
                let source_color = source.point_values["Glow"][corner.vertex];
                vec3_close(position, corner.position, TOLERANCE)
                    && vec2_close(uv, corner.uvs["Detail"], TOLERANCE)
                    && color
                        == [
                            ply_byte(source_color[0], false),
                            ply_byte(source_color[1], false),
                            ply_byte(source_color[2], false),
                            ply_byte(source_color[3], true),
                        ]
            }),
            "PLY vertex does not match an oracle position/UV/averaged-color tuple: {vertex:?}"
        );
        assert!(
            vertex.values["nx"].is_finite()
                && vertex.values["ny"].is_finite()
                && vertex.values["nz"].is_finite()
        );
    }
    assert!(
        signed_volume(
            &vertices
                .iter()
                .map(|vertex| [vertex.values["x"], vertex.values["y"], vertex.values["z"]])
                .collect::<Vec<_>>(),
            &faces,
        ) > 7.99,
        "PLY face indices reverse/mirror the source shell"
    );
    assert!(
        ply_has_uv_seam(&vertices, &faces),
        "PLY lost the active UV seam"
    );
    Ok(PlyFile { vertices })
}

fn take_ply_bytes<'a>(bytes: &'a [u8], cursor: &mut usize, size: usize) -> TestResult<&'a [u8]> {
    let end = cursor.checked_add(size).ok_or("PLY cursor overflow")?;
    let value = bytes.get(*cursor..end).ok_or("PLY payload truncated")?;
    *cursor = end;
    Ok(value)
}

fn read_ply_scalar(bytes: &[u8], cursor: &mut usize, kind: &str) -> TestResult<f64> {
    Ok(match kind {
        "float" | "float32" => f64::from(f32::from_le_bytes(
            take_ply_bytes(bytes, cursor, 4)?.try_into()?,
        )),
        "double" | "float64" => f64::from_le_bytes(take_ply_bytes(bytes, cursor, 8)?.try_into()?),
        "uchar" | "uint8" => f64::from(take_ply_bytes(bytes, cursor, 1)?[0]),
        "char" | "int8" => f64::from(take_ply_bytes(bytes, cursor, 1)?[0] as i8),
        "ushort" | "uint16" => f64::from(u16::from_le_bytes(
            take_ply_bytes(bytes, cursor, 2)?.try_into()?,
        )),
        "short" | "int16" => f64::from(i16::from_le_bytes(
            take_ply_bytes(bytes, cursor, 2)?.try_into()?,
        )),
        "uint" | "uint32" => f64::from(u32::from_le_bytes(
            take_ply_bytes(bytes, cursor, 4)?.try_into()?,
        )),
        "int" | "int32" => f64::from(i32::from_le_bytes(
            take_ply_bytes(bytes, cursor, 4)?.try_into()?,
        )),
        _ => return Err(format!("unsupported PLY scalar {kind}").into()),
    })
}

fn ply_has_uv_seam(vertices: &[PlyVertex], faces: &[Vec<usize>]) -> bool {
    let corners = faces
        .iter()
        .flatten()
        .filter_map(|index| {
            let vertex = vertices.get(*index)?;
            Some((
                [vertex.values["x"], vertex.values["y"], vertex.values["z"]],
                [vertex.values["s"], vertex.values["t"]],
            ))
        })
        .collect::<Vec<_>>();
    corners.iter().enumerate().any(|(index, (position, uv))| {
        corners
            .iter()
            .skip(index + 1)
            .any(|(other_position, other_uv)| {
                vec3_close(*position, *other_position, 1.0e-6)
                    && !vec2_close(*uv, *other_uv, 1.0e-6)
            })
    })
}

fn assert_ply_import_normals(actual: &DumpedMesh, ply: &PlyFile) {
    for corner in &actual.corners {
        let Some(uv) = corner.uvs.get("UVMap") else {
            panic!("PLY import lacks UVMap")
        };
        let candidates = ply.vertices.iter().filter(|vertex| {
            let position = [vertex.values["x"], vertex.values["y"], vertex.values["z"]];
            let ply_uv = [vertex.values["s"], vertex.values["t"]];
            vec3_close(position, corner.position, TOLERANCE) && vec2_close(ply_uv, *uv, TOLERANCE)
        });
        assert!(
            candidates.into_iter().any(|vertex| {
                let normal = [
                    vertex.values["nx"],
                    vertex.values["ny"],
                    vertex.values["nz"],
                ];
                let length = dot(normal, normal).sqrt();
                length > 0.0
                    && vec3_close(
                        corner.normal,
                        [normal[0] / length, normal[1] / length, normal[2] / length],
                        NORMAL_TOLERANCE,
                    )
            }),
            "PLY imported normal has no corresponding binary vertex normal: {corner:?}"
        );
    }
}

fn usd_array<'a>(text: &'a str, marker: &str) -> TestResult<(Vec<&'a str>, &'a str)> {
    let property = text
        .find(marker)
        .ok_or_else(|| format!("USDA property {marker} missing"))?;
    let after = &text[property + marker.len()..];
    let equals = after
        .find('=')
        .ok_or_else(|| format!("USDA property {marker} has no value"))?;
    let after_equals = &after[equals + 1..];
    let opening = after_equals
        .find('[')
        .ok_or_else(|| format!("USDA property {marker} has no array"))?;
    let bytes = after_equals.as_bytes();
    let mut depth = 0_usize;
    let mut closing = None;
    for (offset, byte) in bytes.iter().enumerate().skip(opening) {
        if *byte == b'[' {
            depth += 1;
        }
        if *byte == b']' {
            depth -= 1;
            if depth == 0 {
                closing = Some(offset);
                break;
            }
        }
    }
    let closing = closing.ok_or_else(|| format!("USDA property {marker} array is unterminated"))?;
    let body = &after_equals[opening + 1..closing];
    let values = split_top_level(body)?;
    let suffix = &after_equals[closing + 1..];
    Ok((values, suffix))
}

fn split_top_level(input: &str) -> TestResult<Vec<&str>> {
    let mut values = Vec::new();
    let mut start = 0;
    let mut depth = 0_i32;
    let mut quoted = false;
    let mut escaped = false;
    for (index, byte) in input.bytes().enumerate() {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                let value = input[start..index].trim();
                if !value.is_empty() {
                    values.push(value);
                }
                start = index + 1;
            }
            _ => {}
        }
    }
    let last = input[start..].trim();
    if !last.is_empty() {
        values.push(last);
    }
    if depth != 0 || quoted {
        return Err("malformed USDA array".into());
    }
    Ok(values)
}

fn numeric_tokens(value: &str) -> TestResult<Vec<f64>> {
    let normalized = value
        .chars()
        .map(|character| {
            if matches!(character, '(' | ')' | '[' | ']' | ',') {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    normalized
        .split_whitespace()
        .map(|word| word.parse::<f64>().map_err(Into::into))
        .collect()
}

fn assert_usda(path: &Path, source: &OracleMesh) -> TestResult<()> {
    let text = fs::read_to_string(path)?;
    assert!(
        text.starts_with("#usda 1.0"),
        "USD output is not readable USDA ASCII"
    );
    let (points, _) = usd_array(&text, "point3f[] points")?;
    let (face_counts, _) = usd_array(&text, "int[] faceVertexCounts")?;
    let (face_indices, _) = usd_array(&text, "int[] faceVertexIndices")?;
    let (normals, _) = usd_array(&text, "normal3f[] normals")?;
    assert_eq!(points.len(), source.vertex_count, "USDA point count");
    assert_eq!(face_counts.len(), source.faces.len(), "USDA face count");
    assert_eq!(
        face_indices.len(),
        source.corners.len(),
        "USDA corner count"
    );
    assert_eq!(normals.len(), source.corners.len(), "USDA normal count");
    assert_eq!(
        face_counts
            .iter()
            .map(|value| value.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?,
        source.faces.iter().map(Vec::len).collect::<Vec<_>>(),
        "USDA face sizes"
    );
    for (index, point) in points.iter().enumerate() {
        let actual = numeric_tokens(point)?;
        assert_eq!(actual.len(), 3);
        let expected = source
            .corners
            .iter()
            .find(|corner| corner.vertex == index)
            .ok_or("USDA point has no source vertex")?
            .position;
        assert!(
            vec3_close([actual[0], actual[1], actual[2]], expected, TOLERANCE),
            "USDA point {index}"
        );
    }
    for (index, normal) in normals.iter().enumerate() {
        let actual = numeric_tokens(normal)?;
        assert_eq!(actual.len(), 3);
        assert!(
            vec3_close(
                [actual[0], actual[1], actual[2]],
                source.corners[index].normal,
                NORMAL_TOLERANCE
            ),
            "USDA normal {index}"
        );
    }
    for (actual_name, source_name) in USD_UVS {
        let (values, suffix) = usd_array(&text, &format!("texCoord2f[] primvars:{actual_name}"))?;
        assert!(
            suffix.contains("interpolation = \"faceVarying\""),
            "USDA UV interpolation {actual_name}"
        );
        let index_marker = format!("int[] primvars:{actual_name}:indices");
        let (mapped, expected_values) = if text.contains(&index_marker) {
            let (indices, _) = usd_array(&text, &index_marker)?;
            assert_eq!(
                indices.len(),
                source.corners.len(),
                "USDA UV index count {actual_name}"
            );
            let mapped = indices
                .iter()
                .map(|index| index.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()?;
            (mapped, values)
        } else {
            ((0..values.len()).collect(), values)
        };
        assert_eq!(
            mapped.len(),
            source.corners.len(),
            "USDA UV mapping {actual_name}"
        );
        for (corner_index, value_index) in mapped.iter().enumerate() {
            let parsed = numeric_tokens(
                expected_values
                    .get(*value_index)
                    .ok_or("USDA UV index out of range")?,
            )?;
            let wanted = source.corners[corner_index].uvs[*source_name];
            assert!(
                vec2_close([parsed[0], parsed[1]], wanted, TOLERANCE),
                "USDA UV {actual_name} corner {corner_index}"
            );
        }
    }
    for (actual_name, source_name) in FULL_COLORS {
        let (values, suffix) = usd_array(&text, &format!("color4f[] primvars:{actual_name}"))?;
        let expected_domain = if source.colors[*source_name].domain == "point" {
            "vertex"
        } else {
            "faceVarying"
        };
        assert!(
            suffix.contains(&format!("interpolation = \"{expected_domain}\"")),
            "USDA color interpolation {actual_name}"
        );
        let index_marker = format!("int[] primvars:{actual_name}:indices");
        let indices = if text.contains(&index_marker) {
            let (indices, _) = usd_array(&text, &index_marker)?;
            indices
                .iter()
                .map(|index| index.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let expected_count = if expected_domain == "vertex" {
            source.vertex_count
        } else {
            source.corners.len()
        };
        if indices.is_empty() {
            assert_eq!(
                values.len(),
                expected_count,
                "USDA color value count {actual_name}"
            );
        } else {
            assert_eq!(
                indices.len(),
                expected_count,
                "USDA color index count {actual_name}"
            );
        }
        for (element_index, source_corner) in source.corners.iter().enumerate() {
            let value_index = if expected_domain == "vertex" {
                if indices.is_empty() {
                    source_corner.vertex
                } else {
                    indices[source_corner.vertex]
                }
            } else if indices.is_empty() {
                element_index
            } else {
                indices[element_index]
            };
            let parsed = numeric_tokens(
                values
                    .get(value_index)
                    .ok_or("USDA color index out of range")?,
            )?;
            let wanted = source_corner.colors[*source_name];
            assert_eq!(parsed.len(), 4);
            assert!(
                vec4_close([parsed[0], parsed[1], parsed[2], parsed[3]], wanted, 2.0e-5),
                "USDA {actual_name} color corner {element_index}"
            );
        }
    }
    assert_usda_group_payload(&text, source)?;
    Ok(())
}

fn assert_usda_group_payload(text: &str, source: &OracleMesh) -> TestResult<()> {
    let line = text
        .lines()
        .find(|line| line.contains("custom string potter:deformation ="))
        .ok_or("USDA deformation payload missing")?;
    let encoded = line
        .split_once('=')
        .ok_or("USDA deformation assignment malformed")?
        .1
        .trim();
    let payload_string: String = serde_json::from_str(encoded)?;
    let payload: Value = serde_json::from_str(&payload_string)?;
    let groups = payload["vertex_groups"]
        .as_array()
        .ok_or("USDA vertex groups missing")?;
    let group_ids = groups
        .iter()
        .map(|group| {
            Ok((
                group["id"]
                    .as_str()
                    .ok_or("USDA group ID missing")?
                    .to_owned(),
                group["name"]
                    .as_str()
                    .ok_or("USDA group name missing")?
                    .to_owned(),
            ))
        })
        .collect::<TestResult<BTreeMap<_, _>>>()?;
    let vertex_ids = payload["vertex_ids"]
        .as_array()
        .ok_or("USDA vertex IDs missing")?;
    assert_eq!(
        vertex_ids.len(),
        source.vertex_count,
        "USDA group point count"
    );
    let weights = payload["vertex_weights"]
        .as_object()
        .ok_or("USDA sparse group weights missing")?;
    let mut decoded = BTreeMap::<String, BTreeMap<usize, f64>>::new();
    for (vertex_index, id) in vertex_ids.iter().enumerate() {
        let id = id.as_u64().ok_or("USDA vertex ID invalid")?.to_string();
        if let Some(by_group) = weights.get(&id) {
            for (group_id, weight) in by_group.as_object().ok_or("USDA group weights invalid")? {
                let name = group_ids
                    .get(group_id)
                    .ok_or("USDA weight references unknown group")?
                    .clone();
                decoded
                    .entry(name)
                    .or_default()
                    .insert(vertex_index, number(weight)?);
            }
        }
    }
    assert_eq!(decoded, source.groups, "USDA sparse group assignments");
    Ok(())
}

fn assert_fbx_binary(path: &Path) -> TestResult<()> {
    const MAGIC: &[u8] = b"Kaydara FBX Binary  \0\x1a\0";
    let bytes = fs::read(path)?;
    assert!(bytes.starts_with(MAGIC), "FBX binary signature missing");
    let version = u32::from_le_bytes(
        bytes
            .get(23..27)
            .ok_or("FBX version truncated")?
            .try_into()?,
    );
    assert_eq!(version, 7400, "FBX binary version");
    for token in [
        "Geometry",
        "Vertices",
        "PolygonVertexIndex",
        "LayerElementNormal",
        "LayerElementUV",
        "LayerElementColor",
        "Paint",
        "Glow",
        "Dirt",
        "Trim",
    ] {
        assert!(
            bytes
                .windows(token.len())
                .any(|window| window == token.as_bytes()),
            "FBX binary lacks {token}"
        );
    }
    Ok(())
}

fn assert_alembic_ogawa(path: &Path) -> TestResult<()> {
    let bytes = fs::read(path)?;
    assert!(bytes.len() > 16, "Alembic Ogawa file is truncated");
    assert_eq!(&bytes[..5], b"Ogawa", "Alembic magic");
    assert_eq!(bytes[5], 0xff, "Alembic closed archive marker");
    assert_eq!(&bytes[6..8], &[0, 1], "Alembic Ogawa version");
    let root_offset = u64::from_le_bytes(bytes[8..16].try_into()?);
    assert!(
        root_offset > 0 && root_offset < bytes.len() as u64,
        "Alembic root offset is invalid"
    );
    Ok(())
}

fn assert_groups_for_format(
    case: FormatCase,
    source: &OracleMesh,
    actual: &DumpedMesh,
    losses: &[Value],
) {
    if case.format == "usda" {
        return;
    }
    if !case.groups_from_blender {
        assert!(
            actual.groups.is_empty(),
            "{} unexpectedly imported unsupported vertex groups",
            case.format
        );
        return;
    }
    let lost = group_loss_names(losses);
    let expected = source
        .groups
        .iter()
        .filter(|(name, _)| !lost.contains(*name))
        .map(|(name, weights)| (name.clone(), weights.clone()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        actual.groups, expected,
        "{} sparse vertex-group assignments",
        case.format
    );
}

fn blender_dumps(blender: &Path, root: &Path, files: &[PathBuf]) -> TestResult<Value> {
    let script = write_script(root, "import_exchange_files.py", IMPORT_DUMP_SCRIPT)?;
    let output = root.join("blender_imports.json");
    let mut arguments = vec![output.as_path()];
    arguments.extend(files.iter().map(PathBuf::as_path));
    run_blender_arguments(blender, &script, None, &arguments)?;
    Ok(serde_json::from_slice(&fs::read(output)?)?)
}

#[test]
fn native_color_uv_and_sparse_groups_exchange_through_blender_and_cli() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let (oracle_blend, expected) = build_oracle(&blender, root)?;
    let source_blend = root.join("exchange_oracle.blend");
    let expected_path = root.join("oracle/expected.json");
    let prune = write_script(root, "prune_oracle.py", PRUNE_SCRIPT)?;
    let prune_arguments = [
        oracle_blend.as_path(),
        expected_path.as_path(),
        source_blend.as_path(),
    ];
    run_blender_arguments(&blender, &prune, None, &prune_arguments)?;
    let (project, _) = import_blend(&blender, root, "exchange_project", &source_blend)?;
    let source_dump = dump_blend(&blender, root, &source_blend, "exchange_source")?;
    let source = parse_source_oracle(&expected, &source_dump)?;
    assert_eq!(source.faces.len(), 24, "oracle quad face count");
    assert_eq!(source.corners.len(), 96, "oracle face-corner count");
    assert_close(
        signed_volume(&source.vertex_positions, &source.faces),
        8.0,
        1.0e-5,
        "source oriented volume",
    );

    let mut outputs = Vec::new();
    let mut losses_by_format = BTreeMap::new();
    for case in CASES {
        let (losses, report, output) = assert_loss_parity(case, &project, &blender, root)?;
        let counts = &report["result"]["counts"];
        assert_eq!(
            loss_values(&report)?,
            losses,
            "{} exported loss set",
            case.format
        );
        assert_eq!(counts["objects"], 1, "{} output object count", case.format);
        assert_eq!(
            counts["vertices"], source.vertex_count,
            "{} source vertex count",
            case.format
        );
        assert_eq!(
            counts["faces"],
            source.faces.len(),
            "{} source face count",
            case.format
        );
        assert_eq!(
            counts["triangles"], 48,
            "{} source triangle count",
            case.format
        );
        losses_by_format.insert(case.format, losses);
        outputs.push((case, output));
    }

    let obj = outputs
        .iter()
        .find(|(case, _)| case.format == "obj")
        .ok_or("OBJ output missing")?;
    let obj_data = decode_obj(&obj.1, &source)?;
    let lost_obj_groups = group_loss_names(&losses_by_format["obj"]);
    let expected_obj_groups = source
        .groups
        .keys()
        .filter(|name| !lost_obj_groups.contains(*name))
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(obj_data.groups, expected_obj_groups, "OBJ raw face groups");
    let ply_case = outputs
        .iter()
        .find(|(case, _)| case.format == "ply")
        .ok_or("PLY output missing")?;
    let ply = decode_ply(&ply_case.1, &source)?;
    let usd = outputs
        .iter()
        .find(|(case, _)| case.format == "usda")
        .ok_or("USDA output missing")?;
    assert_usda(&usd.1, &source)?;
    let fbx = outputs
        .iter()
        .find(|(case, _)| case.format == "fbx-binary")
        .ok_or("FBX output missing")?;
    assert_fbx_binary(&fbx.1)?;
    let alembic = outputs
        .iter()
        .find(|(case, _)| case.format == "alembic")
        .ok_or("Alembic output missing")?;
    assert_alembic_ogawa(&alembic.1)?;

    let files = outputs
        .iter()
        .map(|(_, path)| path.clone())
        .collect::<Vec<_>>();
    let blender_dumps = blender_dumps(&blender, root, &files)?;
    for (case, path) in &outputs {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("output filename is not UTF-8")?;
        let dump = parse_dump_mesh(&blender_dumps[file_name])?;
        assert_imported_mesh(*case, &source, &dump);
        assert_groups_for_format(*case, &source, &dump, &losses_by_format[case.format]);
        if case.format == "ply" {
            assert_ply_import_normals(&dump, &ply);
        }
        if case.format == "alembic" {
            assert_eq!(
                blender_dumps[file_name]["object"], "Oracle",
                "Blender's Alembic import object name"
            );
        }
    }

    // USD's documented Potter deformation payload carries sparse groups independently of Blender's
    // UI group importer; the raw USDA decoder above compares it directly to source oracle weights.
    let usd_project = root.join("usd_import_project");
    pot_json(|command| {
        command.arg("init").arg(&usd_project);
    })?;
    let imported = pot_json(|command| {
        command
            .arg("import")
            .arg(&usd_project)
            .args(["--file"])
            .arg(&usd.1)
            .args([
                "--format",
                "usda",
                "--mode",
                "replace",
                "--base-revision",
                "0",
            ]);
    })?;
    assert_eq!(
        imported["result"]["losses"],
        json!([]),
        "USD CLI import lost canonical data: {imported}"
    );
    let scene = read_scene(&usd_project)?;
    let imported_groups = scene["data_blocks"]
        .as_object()
        .ok_or("USD imported data blocks missing")?
        .values()
        .find_map(|block| block["vertex_groups"].as_array())
        .ok_or("USD CLI import lost vertex groups")?;
    let imported_names = imported_groups
        .iter()
        .filter_map(|group| group["name"].as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        imported_names,
        source.groups.keys().map(String::as_str).collect()
    );
    let preview_dir = root.join("usd_preview");
    let preview = pot()
        .arg("preview")
        .arg(&usd_project)
        .args(["--views", "iso", "--size", "64", "--out"])
        .arg(&preview_dir)
        .arg("--json")
        .output()?;
    assert!(
        preview.status.success(),
        "USD imported-scene preview failed: stdout={} stderr={}",
        String::from_utf8_lossy(&preview.stdout),
        String::from_utf8_lossy(&preview.stderr)
    );
    let previews = fs::read_dir(&preview_dir)?.collect::<Result<Vec<_>, _>>()?;
    assert!(
        !previews.is_empty(),
        "USD CLI import preview produced no files"
    );
    assert!(
        previews
            .iter()
            .any(|entry| entry.metadata().is_ok_and(|metadata| metadata.len() > 0)),
        "USD CLI import preview output is empty"
    );
    Ok(())
}

/// Blender's view of each imported mesh: one entry per face, naming its slot's material.
fn assert_imported_mesh(case: FormatCase, source: &OracleMesh, actual: &DumpedMesh) {
    assert_eq!(
        actual.faces.len(),
        source.faces.len(),
        "{} face count",
        case.format
    );
    assert_eq!(
        actual.corners.len(),
        source.corners.len(),
        "{} corner count",
        case.format
    );
    assert!(
        actual.vertices.len() >= source.vertex_count,
        "{} lost source vertices",
        case.format
    );
    assert!(
        signed_volume(&actual.vertices, &actual.faces) > 7.99,
        "{} imported winding is mirrored or reversed",
        case.format
    );

    let expected_uv_names: BTreeSet<_> = case.uv_layers.iter().map(|(name, _)| *name).collect();
    let actual_uv_names: BTreeSet<_> = actual.uv_layers.keys().map(String::as_str).collect();
    assert_eq!(
        actual_uv_names, expected_uv_names,
        "{} UV layers",
        case.format
    );
    let expected_color_names: BTreeSet<_> =
        case.color_layers.iter().map(|(name, _)| *name).collect();
    let actual_color_names: BTreeSet<_> = actual.color_layers.keys().map(String::as_str).collect();
    assert_eq!(
        actual_color_names, expected_color_names,
        "{} color layers",
        case.format
    );
    for (actual_name, source_name) in case.color_layers {
        let layer = &actual.color_layers[*actual_name];
        // Blender's FBX and Alembic importers always create CORNER color layers, and OBJ and PLY
        // always create POINT ones. USD keeps the domain it was written with.
        let expected_domain = match case.format {
            "obj" | "ply" => "POINT",
            "fbx-binary" | "alembic" => "CORNER",
            _ if source.colors[*source_name].domain == "point" => "POINT",
            _ => "CORNER",
        };
        assert_eq!(
            layer.domain, expected_domain,
            "{} color domain {actual_name}",
            case.format
        );
    }

    let mut matched = vec![false; source.corners.len()];
    for actual_corner in &actual.corners {
        let found = source
            .corners
            .iter()
            .enumerate()
            .position(|(index, source_corner)| {
                if matched[index]
                    || !vec3_close(actual_corner.position, source_corner.position, TOLERANCE)
                {
                    return false;
                }
                if case.format != "ply"
                    && !vec3_close(actual_corner.normal, source_corner.normal, NORMAL_TOLERANCE)
                {
                    return false;
                }
                for (actual_name, source_name) in case.uv_layers {
                    let (Some(actual_uv), Some(source_uv)) = (
                        actual_corner.uvs.get(*actual_name),
                        source_corner.uvs.get(*source_name),
                    ) else {
                        return false;
                    };
                    if !vec2_close(*actual_uv, *source_uv, TOLERANCE) {
                        return false;
                    }
                }
                for (actual_name, source_name) in case.color_layers {
                    let Some(actual_color) = actual_corner.colors.get(*actual_name) else {
                        return false;
                    };
                    let wanted = expected_import_color(
                        source_color(case, source, source_corner, source_name),
                        case.color_encoding,
                    );
                    if !vec4_close(*actual_color, wanted, COLOR_TOLERANCE) {
                        return false;
                    }
                }
                true
            });
        let Some(index) = found else {
            panic!(
                "{} has a corner without its source position/normal/UV/color tuple: {actual_corner:?}",
                case.format
            );
        };
        matched[index] = true;
    }
    assert!(
        matched.iter().all(|matched| *matched),
        "{} omitted an oracle corner",
        case.format
    );
    for (actual_name, source_name) in case.uv_layers {
        assert!(
            has_uv_seam(&source.corners, source_name),
            "source oracle has no seam in {source_name}"
        );
        assert!(
            has_uv_seam(&actual.corners, actual_name),
            "{} lost {actual_name} seams",
            case.format
        );
    }
}

fn write_solid_png(path: &Path, color: [u8; 3]) -> TestResult {
    let mut encoder = png::Encoder::new(fs::File::create(path)?, 2, 2);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let mut pixels = Vec::with_capacity(12);
    for _ in 0..4 {
        pixels.extend_from_slice(&color);
    }
    writer.write_image_data(&pixels)?;
    Ok(())
}

/// A two-material plane with a third face that has no material: `usemtl` with no name.
fn write_obj_mtl_fixture(directory: &Path) -> TestResult<PathBuf> {
    fs::create_dir_all(directory)?;
    write_solid_png(&directory.join("warm.png"), [240, 12, 8])?;
    write_solid_png(&directory.join("cool.png"), [8, 232, 18])?;
    fs::write(
        directory.join("source.mtl"),
        concat!(
            "newmtl Warm\n",
            "Kd 0.8 0.25 0.1\n",
            "d 0.8\n",
            "map_Kd warm.png\n\n",
            "newmtl Cool\n",
            "Kd 0.1 0.7 0.25\n",
            "d 0.8\n",
            "map_Kd cool.png\n",
        ),
    )?;
    let object = directory.join("source.obj");
    fs::write(
        &object,
        concat!(
            "mtllib source.mtl\n",
            "o TwoMaterialPlane\n",
            "v -1 0 -1\n",
            "v 0 0 -1\n",
            "v 1 0 -1\n",
            "v 2 0 -1\n",
            "v 2 0 1\n",
            "v 1 0 1\n",
            "v 0 0 1\n",
            "v -1 0 1\n",
            "vt 0 0\n",
            "vt 1 0\n",
            "vt 1 1\n",
            "vt 0 1\n",
            "vn 0 1 0\n",
            "usemtl Warm\n",
            "f 1/1/1 8/4/1 7/3/1 2/2/1\n",
            "usemtl Cool\n",
            "f 2/1/1 7/4/1 6/3/1 3/2/1\n",
            "usemtl\n",
            "f 3/1/1 6/4/1 5/3/1 4/2/1\n",
        ),
    )?;
    Ok(object)
}

/// The material name of each face in OBJ file order. `None` is a face under an empty `usemtl`.
fn obj_face_materials(path: &Path) -> TestResult<Vec<Option<String>>> {
    let text = fs::read_to_string(path)?;
    let mut active = None::<Option<String>>;
    let mut result = Vec::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("usemtl") => active = Some(fields.next().map(str::to_owned)),
            Some("f") => result.push(active.clone().ok_or("OBJ face has no usemtl")?),
            _ => {}
        }
    }
    Ok(result)
}

fn obj_mtl_path(path: &Path) -> TestResult<PathBuf> {
    let text = fs::read_to_string(path)?;
    let name = text
        .lines()
        .find_map(|line| line.strip_prefix("mtllib "))
        .ok_or("OBJ has no mtllib declaration")?;
    let name = Path::new(name.trim());
    Ok(if name.is_absolute() {
        name.to_owned()
    } else {
        path.parent()
            .ok_or("OBJ has no parent directory")?
            .join(name)
    })
}

#[derive(Default)]
struct MtlMaterial {
    kd: Option<[f64; 3]>,
    alpha: Option<f64>,
    texture: Option<PathBuf>,
}

fn obj_mtl_materials(path: &Path) -> TestResult<BTreeMap<String, MtlMaterial>> {
    let mtl = obj_mtl_path(path)?;
    let text = fs::read_to_string(&mtl)?;
    let mut current = None::<String>;
    let mut materials = BTreeMap::<String, MtlMaterial>::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("newmtl") => {
                let name = fields.next().ok_or("MTL newmtl has no name")?.to_owned();
                current = Some(name.clone());
                materials.entry(name).or_default();
            }
            Some("Kd") => {
                let values = fields
                    .map(str::parse::<f64>)
                    .collect::<Result<Vec<_>, _>>()?;
                if values.len() < 3 {
                    return Err("MTL Kd needs three values".into());
                }
                let name = current.as_ref().ok_or("MTL Kd precedes newmtl")?;
                materials
                    .get_mut(name)
                    .ok_or("MTL material disappeared")?
                    .kd = Some([values[0], values[1], values[2]]);
            }
            Some("d" | "Tr") => {
                let value = fields
                    .next()
                    .ok_or("MTL dissolve value missing")?
                    .parse::<f64>()?;
                let alpha = if line.trim_start().starts_with("Tr ") {
                    1.0 - value
                } else {
                    value
                };
                let name = current.as_ref().ok_or("MTL dissolve precedes newmtl")?;
                materials
                    .get_mut(name)
                    .ok_or("MTL material disappeared")?
                    .alpha = Some(alpha);
            }
            Some("map_Kd") => {
                let texture = fields.last().ok_or("MTL map_Kd has no texture path")?;
                let name = current.as_ref().ok_or("MTL map_Kd precedes newmtl")?;
                let texture = Path::new(texture);
                let resolved = if texture.is_absolute() {
                    texture.to_owned()
                } else {
                    mtl.parent()
                        .ok_or("MTL has no parent directory")?
                        .join(texture)
                };
                materials
                    .get_mut(name)
                    .ok_or("MTL material disappeared")?
                    .texture = Some(resolved);
            }
            _ => {}
        }
    }
    Ok(materials)
}

/// Checks the OBJ's face-to-material binding and its MTL, comparing parsed numbers so the exact
/// float text Potter writes does not matter.
fn assert_obj_mtl_files(path: &Path, expected_faces: &[Option<&str>], has_kd: bool) -> TestResult {
    let expected_faces = expected_faces
        .iter()
        .map(|name| name.map(str::to_owned))
        .collect::<Vec<_>>();
    assert_eq!(obj_face_materials(path)?, expected_faces);
    let materials = obj_mtl_materials(path)?;
    assert_eq!(
        materials.keys().cloned().collect::<BTreeSet<_>>(),
        BTreeSet::from(["Cool".to_owned(), "Warm".to_owned()]),
        "MTL defines only the bound materials"
    );
    if has_kd {
        assert_eq!(materials["Warm"].kd, Some([0.8, 0.25, 0.1]), "Warm Kd");
        assert_eq!(materials["Cool"].kd, Some([0.1, 0.7, 0.25]), "Cool Kd");
    } else {
        // Blender's writer omits `Kd` under a colour map (`obj_export_file_writer.cc`), and Potter's does too.
        assert_eq!(materials["Warm"].kd, None, "Warm Kd under map_Kd");
        assert_eq!(materials["Cool"].kd, None, "Cool Kd under map_Kd");
    }
    if has_kd {
        assert_eq!(materials["Warm"].alpha, Some(0.8), "Warm dissolve");
        assert_eq!(materials["Cool"].alpha, Some(0.8), "Cool dissolve");
    } else {
        // The factor alpha has no alpha map, so the writer keeps `d` (`obj.rs`, `alpha_from_map`).
        assert_eq!(materials["Warm"].alpha, Some(0.8), "Warm dissolve");
        assert_eq!(materials["Cool"].alpha, Some(0.8), "Cool dissolve");
    }
    assert_ne!(
        materials["Warm"].texture, materials["Cool"].texture,
        "MTL materials must reference distinct images"
    );
    for name in ["Warm", "Cool"] {
        let texture = materials[name]
            .texture
            .as_ref()
            .ok_or_else(|| format!("{name} MTL map_Kd is missing"))?;
        let metadata = fs::metadata(texture).map_err(|error| {
            format!(
                "{name} MTL texture {} is missing: {error}",
                texture.display()
            )
        })?;
        assert!(metadata.len() > 0, "{name} MTL texture is empty");
    }
    Ok(())
}

fn assert_mtl_rgba(actual: &Value, expected: [f64; 4], label: &str) -> TestResult {
    let actual = native_oracle::vec4(actual)?;
    for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert_close(
            actual,
            expected,
            1.0e-6,
            &format!("{label} channel {channel}"),
        );
    }
    Ok(())
}

fn assert_project_obj_materials(scene: &Value) -> TestResult {
    let node_id = native_oracle::node_id(scene, "TwoMaterialPlane")?;
    let data_id = scene["nodes"][&node_id]["data"]
        .as_str()
        .ok_or("OBJ node data missing")?;
    let slots = scene["data_blocks"][data_id]["materials"]
        .as_array()
        .ok_or("OBJ mesh material slots missing")?;
    let slot_names = slots
        .iter()
        .map(|slot| {
            if slot.is_null() {
                Ok(None)
            } else {
                let id = slot.as_str().ok_or("OBJ material slot ID missing")?;
                scene["materials"][id]["name"]
                    .as_str()
                    .map(|name| Some(name.to_owned()))
                    .ok_or_else(|| "OBJ slot material name missing".into())
            }
        })
        .collect::<TestResult<Vec<Option<String>>>>()?;
    assert_eq!(
        slot_names,
        vec![Some("Warm".to_owned()), Some("Cool".to_owned()), None],
        "OBJ material slot order and unassigned marker",
    );

    let mesh = native_oracle::mesh_of_node(scene, "TwoMaterialPlane")?;
    let face_material_indices = mesh["faces"]
        .as_array()
        .ok_or("OBJ mesh faces missing")?
        .iter()
        .map(|face| {
            face["material_index"]
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| "OBJ face material index missing".into())
        })
        .collect::<TestResult<Vec<_>>>()?;
    assert_eq!(
        face_material_indices,
        vec![0, 1, 2],
        "OBJ face-to-slot binding"
    );

    let mut texture_ids = BTreeSet::new();
    for slot in slots.iter().filter_map(Value::as_str) {
        let material = &scene["materials"][slot];
        let name = material["name"]
            .as_str()
            .ok_or("OBJ material name missing")?;
        // The map carries the colour; a linked Base Color overrides the socket default `Kd`, so the
        // imported factor is white (`obj.rs`, textured materials).
        let expected = match name {
            "Warm" | "Cool" => [1.0, 1.0, 1.0, 0.8],
            _ => return Err(format!("unexpected bound OBJ material {name}").into()),
        };
        assert_mtl_rgba(&material["base_color"], expected, name)?;
        assert_eq!(material["alpha_mode"], "blend", "{name} MTL dissolve mode");
        let image = material["base_color_texture"]["image"]
            .as_str()
            .ok_or_else(|| format!("{name} has no imported map_Kd"))?;
        assert!(
            scene["images"].get(image).is_some(),
            "material texture image {image} is missing"
        );
        texture_ids.insert(image.to_owned());
    }
    assert_eq!(
        texture_ids.len(),
        2,
        "OBJ MTL imported distinct base-color images"
    );
    Ok(())
}

fn file_name(path: &Path) -> TestResult<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "file name is not UTF-8".into())
}

/// The material name of each face as Blender assigned it on import of the OBJ at `path`.
fn blender_obj_faces(dumps: &Value, path: &Path) -> TestResult<Vec<Option<String>>> {
    let mesh = &dumps[file_name(path)?];
    let slots = mesh["material_slots"]
        .as_array()
        .ok_or("Blender OBJ material slots missing")?;
    mesh["face_material_indices"]
        .as_array()
        .ok_or("Blender OBJ face material indices missing")?
        .iter()
        .map(|index| -> TestResult<Option<String>> {
            let index = index
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .ok_or("Blender OBJ face material index is invalid")?;
            let slot = slots
                .get(index)
                .ok_or("Blender OBJ face slot is out of range")?;
            Ok(slot["name"].as_str().map(str::to_owned))
        })
        .collect()
}

/// Blender's import of the source and of the exported OBJ must bind each face to the same slot
/// name. `Warm` and `Cool` carry the MTL colors and textures. The unassigned face (empty `usemtl`)
/// must bind neither MTL material and any slot it leaves must carry no MTL texture. Whether Blender
/// leaves that face with no material or adds its own default slot is the importer's behavior, so
/// both are accepted.
fn assert_blender_obj_materials(dumps: &Value, source: &Path, exported: &Path) -> TestResult {
    let source_faces = blender_obj_faces(dumps, source)?;
    let exported_faces = blender_obj_faces(dumps, exported)?;
    assert_eq!(
        exported_faces, source_faces,
        "Blender face-to-slot binding of the export differs from the source"
    );
    assert_eq!(
        exported_faces[0].as_deref(),
        Some("Warm"),
        "first face binds Warm"
    );
    assert_eq!(
        exported_faces[1].as_deref(),
        Some("Cool"),
        "second face binds Cool"
    );
    // Blender's OBJ importer gives the empty-`usemtl` face the previous material (Cool here) for the
    // source too, so only the source-to-export equality above is checked for that face.
    for path in [source, exported] {
        let mesh = &dumps[file_name(path)?];
        let mut names = BTreeSet::new();
        for slot in mesh["material_slots"]
            .as_array()
            .ok_or("Blender OBJ material slots missing")?
        {
            let Some(name) = slot["name"].as_str() else {
                continue;
            };
            names.insert(name.to_owned());
            let images = slot["images"]
                .as_array()
                .ok_or("Blender OBJ material images missing")?;
            match name {
                "Warm" | "Cool" => {
                    // Blender sets the viewport colour from `Kd` only (`obj_import_mtl.cc`: `mat->r/g/b`),
                    // and never from `d`. The exported file omits `Kd` under a map, so Blender keeps its
                    // default 0.8.
                    let expected = match (path == source, name) {
                        (true, "Warm") => [0.8, 0.25, 0.1, 1.0],
                        (true, _) => [0.1, 0.7, 0.25, 1.0],
                        (false, _) => [0.8, 0.8, 0.8, 1.0],
                    };
                    assert_mtl_rgba(&slot["diffuse_color"], expected, name)?;
                    assert!(
                        !images.is_empty(),
                        "Blender material {name} has no imported MTL texture"
                    );
                }
                _ => assert!(
                    images.is_empty(),
                    "Blender slot {name} for the unassigned face carries an MTL texture"
                ),
            }
        }
        assert!(
            names.contains("Warm") && names.contains("Cool"),
            "Blender OBJ material slots for {} lack Warm or Cool: {names:?}",
            file_name(path)?
        );
    }
    Ok(())
}

fn preview_rgb(path: &Path, x: u32, y: u32) -> TestResult<[u8; 3]> {
    let file = std::io::BufReader::new(fs::File::open(path)?);
    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let width = reader.info().width;
    let height = reader.info().height;
    if x >= width || y >= height {
        return Err("preview pixel is outside the image".into());
    }
    let mut bytes = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output size is unknown")?
    ];
    let frame = reader.next_frame(&mut bytes)?;
    let channels = match frame.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err("PNG palette was not expanded".into()),
    };
    let offset = (y as usize * width as usize + x as usize) * channels;
    let pixel = &bytes[offset..offset + channels];
    Ok(match frame.color_type {
        png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => [pixel[0]; 3],
        png::ColorType::Rgb | png::ColorType::Rgba => [pixel[0], pixel[1], pixel[2]],
        png::ColorType::Indexed => unreachable!(),
    })
}

#[test]
fn obj_mtl_material_slots_and_textures_survive_import_export_and_preview() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let source_project = root.join("source_project");
    pot_json(|command| {
        command.arg("init").arg(&source_project);
    })?;
    let source_obj = write_obj_mtl_fixture(&source_project.join("source_fixture"))?;
    assert_obj_mtl_files(&source_obj, &[Some("Warm"), Some("Cool"), None], true)?;

    let source_import = pot_json(|command| {
        command
            .arg("import")
            .arg(&source_project)
            .args(["--file"])
            .arg(&source_obj)
            .args([
                "--format",
                "obj",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--allow-lossy",
            ]);
    })?;
    // Blender's MTL importer sets specular to zero when Ks is absent (`set_bsdf_socket_values`,
    // `obj_import_mtl.cc`), which Potter cannot hold, so each material reports the loss.
    let source_losses = loss_values(&source_import)?;
    assert_eq!(
        source_losses.len(),
        2,
        "OBJ MTL import losses: {source_import}"
    );
    assert_eq!(loss_count(&source_losses, "material.specular"), 2);
    assert_project_obj_materials(&read_scene(&source_project)?)?;

    let export_dir = root.join("exported_obj");
    fs::create_dir_all(&export_dir)?;
    let exported_obj = export_dir.join("roundtrip.obj");
    let exported = pot_json(|command| {
        command
            .arg("export")
            .arg(&source_project)
            .args(["--format", "obj", "--out"])
            .arg(&exported_obj)
            .arg("--blender")
            .arg(&blender);
    })?;
    assert_eq!(
        exported["result"]["losses"],
        json!([]),
        "OBJ MTL export losses"
    );
    assert!(
        fs::metadata(&exported_obj)?.len() > 0,
        "OBJ export is empty"
    );
    assert_obj_mtl_files(&exported_obj, &[Some("Warm"), Some("Cool"), None], false)?;

    let blender_sources = [source_obj.clone(), exported_obj.clone()];
    let blender_imports = blender_dumps(&blender, root, &blender_sources)?;
    assert_blender_obj_materials(&blender_imports, &source_obj, &exported_obj)?;

    let roundtrip_project = root.join("roundtrip_project");
    pot_json(|command| {
        command.arg("init").arg(&roundtrip_project);
    })?;
    let roundtrip_import = pot_json(|command| {
        command
            .arg("import")
            .arg(&roundtrip_project)
            .args(["--file"])
            .arg(&exported_obj)
            .args([
                "--format",
                "obj",
                "--mode",
                "replace",
                "--base-revision",
                "0",
            ]);
    })?;
    assert_eq!(
        roundtrip_import["result"]["losses"],
        json!([]),
        "round-trip OBJ MTL import losses"
    );
    assert_project_obj_materials(&read_scene(&roundtrip_project)?)?;

    apply_operations(
        &roundtrip_project,
        root,
        json!([
            {"op": "light.create", "id": "preview_light", "light_type": "point",
             "transform": {"translation": [0.0, 0.0, 6.0]}, "energy": 100.0, "radius": 0.5}
        ]),
    )?;
    let preview_dir = root.join("obj_material_preview");
    let preview = pot()
        .arg("preview")
        .arg(&roundtrip_project)
        .args([
            "--views", "top", "--size", "128", "--mode", "beauty", "--out",
        ])
        .arg(&preview_dir)
        .arg("--json")
        .output()?;
    assert!(
        preview.status.success(),
        "OBJ MTL beauty preview failed: stdout={} stderr={}",
        String::from_utf8_lossy(&preview.stdout),
        String::from_utf8_lossy(&preview.stderr),
    );
    let preview_path = preview_dir.join("top.png");
    assert!(
        fs::metadata(&preview_path)?.len() > 0,
        "OBJ MTL preview image is empty"
    );
    let left = preview_rgb(&preview_path, 21, 64)?;
    let middle = preview_rgb(&preview_path, 64, 64)?;
    let is_warm =
        |rgb: [u8; 3]| rgb[0] > rgb[1].saturating_add(20) && rgb[0] > rgb[2].saturating_add(20);
    let is_cool =
        |rgb: [u8; 3]| rgb[1] > rgb[0].saturating_add(20) && rgb[1] > rgb[2].saturating_add(20);
    assert!(
        (is_warm(left) && is_cool(middle)) || (is_cool(left) && is_warm(middle)),
        "OBJ MTL textures did not render as distinct warm/cool faces: {left:?}, {middle:?}"
    );
    Ok(())
}

fn fbx_material_oracle(blender: &Path, root: &Path) -> TestResult<(PathBuf, Value)> {
    let script = write_script(
        root,
        "fbx_empty_slot_oracle.py",
        FBX_EMPTY_SLOT_ORACLE_SCRIPT,
    )?;
    let source_blend = root.join("fbx_empty_slot_source.blend");
    let fbx = root.join("fbx_empty_slot.fbx");
    let output = root.join("fbx_empty_slot_oracle.json");
    let arguments = [source_blend.as_path(), fbx.as_path(), output.as_path()];
    run_blender_arguments(blender, &script, None, &arguments)?;
    Ok((fbx, serde_json::from_slice(&fs::read(output)?)?))
}

/// Potter's import of a Blender FBX whose second slot is unconnected must match Blender's own
/// resolution: the unconnected slot is dropped and its face index resolves to the first slot.
fn assert_fbx_import_matches_blender(scene: &Value, oracle: &Value) -> TestResult {
    let id = native_oracle::node_id(scene, "FbxMaterialOracle")?;
    let slots = scene["nodes"][&id]["materials"]
        .as_array()
        .ok_or("FBX node material slots missing")?;
    let slot_names = slots
        .iter()
        .map(|slot| {
            let material = slot.as_str().ok_or("FBX imported slot is empty")?;
            scene["materials"][material]["name"]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "FBX imported material has no name".into())
        })
        .collect::<TestResult<Vec<String>>>()?;
    assert_eq!(json!(slot_names), oracle["blender_slots"], "FBX slot names");

    let mesh = native_oracle::mesh_of_node(scene, "FbxMaterialOracle")?;
    let face_material_indices = mesh["faces"]
        .as_array()
        .ok_or("FBX mesh faces missing")?
        .iter()
        .map(|face| {
            face["material_index"]
                .as_u64()
                .ok_or_else(|| "FBX face material index missing".into())
        })
        .collect::<TestResult<Vec<u64>>>()?;
    assert_eq!(
        json!(face_material_indices),
        oracle["blender_face_material_indices"],
        "FBX face-to-slot binding"
    );
    Ok(())
}

/// Potter keeps a null material slot on a node. FBX cannot write one, so each FBX export refuses
/// it strictly and `--allow-lossy` writes the file with the loss reported.
fn assert_fbx_material_slot_loss_parity(blender: &Path, project: &Path, root: &Path) -> TestResult {
    for format in ["fbx", "fbx-binary"] {
        let validation_output = pot()
            .arg("validate")
            .arg(project)
            .args(["--format", format, "--strict", "--json"])
            .output()?;
        let validation: Value = serde_json::from_slice(&validation_output.stdout)?;
        let validation_losses = loss_values(&validation)?;
        assert!(
            !validation_output.status.success(),
            "{format} strict validation accepted an empty slot"
        );
        assert_eq!(
            validation_losses.len(),
            1,
            "{format} empty-slot loss set: {validation}"
        );
        assert_eq!(
            validation_losses[0]["feature_id"], "format.fbx.material_slot_empty",
            "{format} empty-slot loss feature"
        );

        let strict_path = root.join(format!("strict_empty_slot_{format}.fbx"));
        let strict_output = export_raw(blender, project, format, &strict_path, false)?;
        let strict_report: Value = serde_json::from_slice(&strict_output.stdout)?;
        assert!(
            !strict_output.status.success(),
            "strict {format} export accepted an empty slot"
        );
        assert_eq!(
            loss_values(&strict_report)?,
            validation_losses,
            "{format} strict validate/export losses"
        );
        assert_eq!(strict_report["error"]["code"], "UNREPRESENTABLE_FEATURE");

        let lossy_path = root.join(format!("allow_lossy_empty_slot_{format}.fbx"));
        let lossy_output = export_raw(blender, project, format, &lossy_path, true)?;
        assert!(
            lossy_output.status.success(),
            "allow-lossy {format} export failed: stdout={} stderr={}",
            String::from_utf8_lossy(&lossy_output.stdout),
            String::from_utf8_lossy(&lossy_output.stderr),
        );
        let lossy_report: Value = serde_json::from_slice(&lossy_output.stdout)?;
        assert_eq!(
            loss_values(&lossy_report)?,
            validation_losses,
            "allow-lossy {format} loss report"
        );
        assert!(
            fs::metadata(lossy_path)?.len() > 0,
            "allow-lossy {format} file is empty"
        );
    }
    Ok(())
}

#[test]
fn blender_fbx_minus_one_material_import_resolves_like_blender() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let (fbx, oracle) = fbx_material_oracle(&blender, root)?;
    assert_eq!(oracle["source_slots"], json!(["Bound", null]));
    assert_eq!(oracle["source_face_material_indices"], json!([0, 1]));
    // Blender's FBX exporter writes one AllSame index for a single exported material, never -1
    // (`export_fbx_bin.py`, AllSame branch).
    assert_eq!(oracle["raw_material_indices"], json!([0]));
    assert_eq!(oracle["blender_slots"], json!(["Bound"]));
    assert_eq!(oracle["blender_face_material_indices"], json!([0, 0]));

    let project = root.join("fbx_import_project");
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    let imported = pot_json(|command| {
        command
            .arg("import")
            .arg(&project)
            .args(["--file"])
            .arg(&fbx)
            .args([
                "--format",
                "fbx-binary",
                "--mode",
                "replace",
                "--base-revision",
                "0",
            ]);
    })?;
    assert_eq!(
        imported["result"]["losses"],
        json!([]),
        "FBX unconnected-slot import losses"
    );
    assert_fbx_import_matches_blender(&read_scene(&project)?, &oracle)
}

/// A Potter-native node whose second material slot is empty: FBX has no way to write it.
#[test]
fn potter_empty_material_slot_is_reported_by_fbx_exports() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let project = root.join("fbx_empty_slot_project");
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    apply_operations(
        &project,
        root,
        json!([
            {"op": "material.create", "id": "bound", "name": "Bound"},
            {"op": "node.create", "id": "empty_slot", "kind": "box",
             "params": {"size": 1.0}, "material": "bound"},
            {"op": "node.update", "target": {"id": "empty_slot"},
             "set": {"materials": ["bound", null]}}
        ]),
    )?;
    let scene = read_scene(&project)?;
    let id = native_oracle::node_id(&scene, "empty_slot")?;
    let slots = scene["nodes"][&id]["materials"]
        .as_array()
        .ok_or("empty-slot node material slots missing")?;
    assert_eq!(slots.len(), 2, "empty-slot node slot count");
    assert_eq!(
        scene["materials"][slots[0].as_str().ok_or("bound slot missing")?]["name"],
        "Bound"
    );
    assert!(
        slots[1].is_null(),
        "empty material slot was not retained as null"
    );
    assert_fbx_material_slot_loss_parity(&blender, &project, root)
}

#[test]
fn armature_bound_vertex_group_roundtrips_through_blender_fbx_without_loss() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let source_blend = root.join("armature_group_source.blend");
    let script = write_script(root, "armature_group_source.py", ARMATURE_GROUP_SCRIPT)?;
    run_blender_arguments(&blender, &script, None, &[source_blend.as_path()])?;

    let (project, import_report) =
        import_blend(&blender, root, "armature_group_project", &source_blend)?;
    assert!(
        !native_oracle::loss_features(&import_report).contains(&"mesh.vertex_group".to_owned()),
        "BLEND import lost the deform group: {import_report}"
    );
    let scene = read_scene(&project)?;
    let node_id = native_oracle::node_id(&scene, "WeightedMesh")?;
    let node = &scene["nodes"][&node_id];
    let data_id = node["data"]
        .as_str()
        .ok_or("weighted mesh data is missing")?;
    let data = &scene["data_blocks"][data_id];
    let group = data["vertex_groups"]
        .as_array()
        .ok_or("native vertex groups are missing")?
        .iter()
        .find(|group| group["name"] == "DeformBone")
        .ok_or("native DeformBone group is missing")?;
    let group_id = group["id"].as_str().ok_or("native group ID is missing")?;
    let weights = data["vertex_weights"]
        .as_object()
        .ok_or("native vertex weights are missing")?;
    let vertices = data["mesh"]["vertices"]
        .as_array()
        .ok_or("native mesh vertices are missing")?;
    let native_weights = vertices
        .iter()
        .map(|vertex| {
            let id = vertex["id"].as_u64().ok_or("native vertex ID is missing")?;
            weights
                .get(&id.to_string())
                .and_then(|groups| groups[group_id].as_f64())
                .ok_or_else(|| "native deform weight is missing".into())
        })
        .collect::<TestResult<Vec<_>>>()?;
    let mut native_weights = native_weights;
    native_weights.sort_by(f64::total_cmp);
    assert_eq!(native_weights, vec![0.25, 0.5, 0.75]);

    let validation = pot_json(|command| {
        command
            .arg("validate")
            .arg(&project)
            .args(["--format", "fbx-binary"]);
    })?;
    assert!(
        !native_oracle::loss_features(&validation).contains(&"mesh.vertex_group".to_owned()),
        "armature-bound FBX group was reported as lost: {validation}"
    );
    let fbx_path = root.join("armature_group.fbx");
    let exported = pot_json(|command| {
        command
            .arg("export")
            .arg(&project)
            .args(["--format", "fbx-binary", "--out"])
            .arg(&fbx_path)
            .arg("--blender")
            .arg(&blender);
    })?;
    assert!(
        !native_oracle::loss_features(&exported).contains(&"mesh.vertex_group".to_owned()),
        "armature-bound FBX export reports a group loss: {exported}"
    );

    let source_dump = dump_blend(&blender, root, &source_blend, "armature_source")?;
    let source_mesh = dumped_object(&source_dump, "WeightedMesh")?;
    let blender_dumps = blender_dumps(&blender, root, std::slice::from_ref(&fbx_path))?;
    let imported_mesh = &blender_dumps[file_name(&fbx_path)?];
    let group_values = |mesh: &Value| -> TestResult<Vec<([f64; 3], f64)>> {
        let vertices = mesh["vertices"]
            .as_array()
            .ok_or("Blender vertices are missing")?;
        let assignments = match &mesh["vertex_groups"] {
            Value::Array(groups) => groups
                .iter()
                .find(|group| group["name"] == "DeformBone")
                .map(|group| &group["weights"]),
            groups => Some(&groups["DeformBone"]),
        }
        .and_then(Value::as_object)
        .ok_or("Blender DeformBone group is missing")?;
        let mut values = assignments
            .iter()
            .map(|(index, weight)| {
                let index = index.parse::<usize>()?;
                Ok((vec3(&vertices[index])?, number(weight)?))
            })
            .collect::<TestResult<Vec<_>>>()?;
        values.sort_by(|left, right| {
            left.0[0]
                .total_cmp(&right.0[0])
                .then_with(|| left.0[1].total_cmp(&right.0[1]))
                .then_with(|| left.0[2].total_cmp(&right.0[2]))
        });
        Ok(values)
    };
    let source_weights = group_values(source_mesh)?;
    let imported_weights = group_values(imported_mesh)?;
    assert_eq!(source_weights.len(), 3);
    assert_eq!(imported_weights.len(), 3);
    for ((source_position, source_weight), (imported_position, imported_weight)) in
        source_weights.iter().zip(&imported_weights)
    {
        assert!(vec3_close(*source_position, *imported_position, TOLERANCE));
        assert_close(*source_weight, *imported_weight, 1.0e-6, "deform weight");
    }
    Ok(())
}
