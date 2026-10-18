//! PR9 normal-code persistence against Blender 5.2.2.
//!
//! * `subdivision_codes_and_corner_data_follow_one_loop_basis_through_blend`: a Subdivision with Use
//!   Custom Normals stores its codes with a nonidentity `corner_order`. The same loop basis must
//!   carry the UV, color, and generic face/corner data. The test exports `.blend`, reopens it in
//!   Blender, reimports it into a second project, exports again, and then continues with Displace.
//!   Every value is compared by corner identity (position and face center), never by raw index.
//!   The expected values come from Blender's own Subdivision evaluation of the same file.
//! * `subdivision_without_custom_normals_keeps_loop_basis_and_later_authoring_allocates_the_layer`:
//!   a Subdivision without Use Custom Normals allocates no `values` layer but keeps its
//!   `corner_order` basis. After save and reload, `mesh.set_custom_normals` allocates the layer
//!   through that basis, and Blender's own reference confirms every normal and Auto status.
//! * `mirror_raw_codes_round_trip_before_bisect_and_wave`: Blender applies X Mirror, Potter exports
//!   the raw codes and loop basis, and a second project reimports them before Y-bisect Mirror and
//!   Wave. The final geometry and corner normals match Blender's own X, Y-bisect, Wave reference.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use native_oracle::{
    BYTE_TOLERANCE, FLOAT_TOLERANCE, NORMAL_TOLERANCE, StoredCorner, TestResult, apply_operations,
    assert_close, assert_floats, attribute, build_oracle, code_pair, code_within_one_unit,
    dump_blend, dumped_object, export, floats, import_blend, losses, mesh_of_node, node_id,
    normal_code_rows, number, read_scene, run_blender, run_blender_arguments, vec3, vectors_close,
    write_script,
};
use serde_json::{Map, Value, json};
use tempfile::tempdir;

/// Blender's no-layer authoring reference (see `fixtures/native_oracle/authoring_reference.py`).
const AUTHORING_REFERENCE_SCRIPT: &str =
    include_str!("fixtures/native_oracle/authoring_reference.py");

/// Attaches UV layers, two color layers, and generic corner/face attributes to `Regress`, keeping its
/// custom normals. Values are written to the file and read back by Blender, never by Potter.
const ATTACH_LAYERS_SCRIPT: &str = r#""""Attaches native corner and face data to the oracle's Regress object (Blender 5.2.2).

Usage: Blender -b oracle.blend --python attach_layers.py -- OUT.blend
"""

import math
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]
mesh = bpy.data.objects["Regress"].data


def round4(value):
    return round(value, 4)


def srgb_to_linear(value):
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


for layer_index, name in enumerate(["Base", "Light", "Detail"]):
    layer = mesh.uv_layers.new(name=name)
    for loop_index, item in enumerate(layer.data):
        item.uv = [
            round4(0.1 * (layer_index + 1) + 0.013 * loop_index),
            round4(0.9 - 0.007 * loop_index - 0.2 * layer_index),
        ]
mesh.uv_layers.active = mesh.uv_layers["Detail"]
mesh.uv_layers["Light"].active_render = True

glow = mesh.color_attributes.new("Glow", "FLOAT_COLOR", "CORNER")
for loop_index, item in enumerate(glow.data):
    item.color = [
        round4(0.031 + 0.0457 * ((loop_index * 11) % 23)),
        round4(0.5 + 0.4 * math.sin(loop_index)),
        round4(0.05 + 0.0231 * ((loop_index * 13) % 31)),
        round4(0.4 + 0.5 * ((loop_index % 5) / 4.0)),
    ]
trim = mesh.color_attributes.new("Trim", "BYTE_COLOR", "CORNER")
for loop_index, item in enumerate(trim.data):
    rgb = [(loop_index * 47 + 9) % 256, (loop_index * 23 + 77) % 256, (loop_index * 61 + 31) % 256]
    alpha = 255 if loop_index % 4 == 0 else (loop_index * 19 + 13) % 256
    item.color = [
        srgb_to_linear(rgb[0] / 255.0),
        srgb_to_linear(rgb[1] / 255.0),
        srgb_to_linear(rgb[2] / 255.0),
        alpha / 255.0,
    ]
mesh.color_attributes.active_color_index = 0
mesh.color_attributes.render_color_index = 1

rank = mesh.attributes.new("corner_rank", "FLOAT", "CORNER")
for loop_index, item in enumerate(rank.data):
    item.value = round4(0.01 * loop_index + 0.3)
island = mesh.attributes.new("island", "INT", "FACE")
for poly in mesh.polygons:
    island.data[poly.index].value = poly.index % 4

bpy.ops.wm.save_as_mainfile(filepath=OUT)
"#;

/// Blender's own Subdivision (levels 1, Use Custom Normals on) evaluation of `Regress`, read back
/// per corner and per face. Corner identity is the position plus the face center.
const SUBDIVISION_DATA_REFERENCE_SCRIPT: &str = r#""""Blender's evaluated corner and face data of the Subdivision result of `Regress` (5.2.2).

Usage: Blender -b attached.blend --python subdivision_data_reference.py -- OUT.json

Each loop entry carries its position, its face center, every UV layer, every color layer, and
every generic corner attribute. Each face entry carries its center and generic face attributes.
"""

import json
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]
UV_LAYERS = ["Base", "Light", "Detail"]
COLOR_LAYERS = ["Glow", "Trim"]
CORNER_ATTRIBUTES = ["corner_rank"]
FACE_ATTRIBUTES = ["island"]

regress = bpy.data.objects["Regress"]
regress.modifiers.clear()
subdivision = regress.modifiers.new("Subdivision", "SUBSURF")
subdivision.levels = 1
subdivision.render_levels = 1
subdivision.use_custom_normals = True
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = regress.evaluated_get(depsgraph).to_mesh()
mesh = evaluated.copy()
regress.evaluated_get(depsgraph).to_mesh_clear()
regress.modifiers.clear()
mesh.update()

loops = [None] * len(mesh.loops)
faces = []
for poly in mesh.polygons:
    center = [float(c) for c in poly.center]
    face = {}
    for name in FACE_ATTRIBUTES:
        face[name] = int(mesh.attributes[name].data[poly.index].value)
    faces.append({"center": center, "face": face})
    for k in range(poly.loop_total):
        index = poly.loop_start + k
        co = mesh.vertices[mesh.loops[index].vertex_index].co
        loops[index] = {
            "co": [float(c) for c in co],
            "center": center,
            "uv": {name: [float(c) for c in mesh.uv_layers[name].data[index].uv] for name in UV_LAYERS},
            "color": {
                name: [float(c) for c in mesh.color_attributes[name].data[index].color]
                for name in COLOR_LAYERS
            },
            "corner": {
                name: float(mesh.attributes[name].data[index].value) for name in CORNER_ATTRIBUTES
            },
        }
if any(entry is None for entry in loops):
    raise SystemExit("some loops belong to no polygon")
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump({"blender": bpy.app.version_string, "loops": loops, "faces": faces}, handle)
"#;

/// One corner of a dumped mesh in Blender's loop order: position, face center, normal, raw code.
struct BlenderLoops {
    co: Vec<[f64; 3]>,
    center: Vec<[f64; 3]>,
    normal: Vec<[f64; 3]>,
    /// Raw `custom_normal` pair per loop; `None` when the file has no custom-normal layer.
    code: Option<Vec<[i64; 2]>>,
}

fn corner_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    (0..3)
        .map(|axis| (left[axis] - right[axis]).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn blender_loops(object: &Value) -> TestResult<BlenderLoops> {
    let vertices = object["vertices"]
        .as_array()
        .ok_or("dumped vertices are missing")?;
    let loops = object["loops"]
        .as_array()
        .ok_or("dumped loops are missing")?;
    let normals = object["corner_normals"]
        .as_array()
        .ok_or("dumped corner normals are missing")?;
    let count = loops.len();
    let code = match &object["codes"] {
        Value::Null => None,
        Value::Array(codes) => {
            if codes.len() != count {
                return Err(format!("{} codes for {count} loops", codes.len()).into());
            }
            Some(
                codes
                    .iter()
                    .map(code_pair)
                    .collect::<TestResult<Vec<_>>>()?,
            )
        }
        other => return Err(format!("dumped codes are malformed: {other}").into()),
    };
    let mut result = BlenderLoops {
        co: vec![[0.0; 3]; count],
        center: vec![[0.0; 3]; count],
        normal: vec![[0.0; 3]; count],
        code,
    };
    let mut covered = vec![false; count];
    for face in object["faces"]
        .as_array()
        .ok_or("dumped faces are missing")?
    {
        let center = vec3(&face["center"])?;
        let start = face["loop_start"].as_u64().ok_or("loop start is missing")? as usize;
        let total = face["loop_total"].as_u64().ok_or("loop total is missing")? as usize;
        for index in start..start + total {
            let vertex = loops
                .get(index)
                .and_then(|entry| entry["vertex"].as_u64())
                .ok_or("face loop is outside the dumped loops")? as usize;
            result.co[index] = vec3(&vertices[vertex])?;
            result.center[index] = center;
            result.normal[index] = vec3(&normals[index])?;
            covered[index] = true;
        }
    }
    if covered.contains(&false) {
        return Err("some dumped loops belong to no face".into());
    }
    Ok(result)
}

fn loop_keys(blender: &BlenderLoops) -> Vec<([f64; 3], [f64; 3])> {
    blender
        .co
        .iter()
        .copied()
        .zip(blender.center.iter().copied())
        .collect()
}

/// The `(co, center)` key of each corner of a Blender reference list.
fn reference_keys(loops: &[Value]) -> TestResult<Vec<([f64; 3], [f64; 3])>> {
    loops
        .iter()
        .map(|corner| -> TestResult<([f64; 3], [f64; 3])> {
            Ok((vec3(&corner["co"])?, vec3(&corner["center"])?))
        })
        .collect()
}

/// One Potter face in `mesh.faces` order: its corner positions and the mean of them, which is
/// Blender's `poly.center` for the same polygon.
struct PotterFace {
    corners: Vec<[f64; 3]>,
    center: [f64; 3],
}

fn potter_faces(mesh: &Value) -> TestResult<Vec<PotterFace>> {
    let mut positions = BTreeMap::new();
    for vertex in mesh["vertices"]
        .as_array()
        .ok_or("mesh vertices are missing")?
    {
        let id = vertex["id"].as_u64().ok_or("vertex id is missing")?;
        positions.insert(id, vec3(&vertex["co"])?);
    }
    let mut faces = Vec::new();
    for face in mesh["faces"].as_array().ok_or("mesh faces are missing")? {
        let mut corners = Vec::new();
        for id in face["v"].as_array().ok_or("face vertices are missing")? {
            let id = id.as_u64().ok_or("face vertex id is missing")?;
            let position = positions
                .get(&id)
                .ok_or_else(|| format!("face uses missing vertex {id}"))?;
            corners.push(*position);
        }
        let mut center = [0.0; 3];
        for corner in &corners {
            for axis in 0..3 {
                center[axis] += corner[axis] / corners.len() as f64;
            }
        }
        faces.push(PotterFace { corners, center });
    }
    Ok(faces)
}

/// Blender loop index of every Potter corner, `[face][corner]` in `mesh.faces` order: the stored
/// `corner_order` ordinal, or the flattened corner index when no order is stored.
fn loop_ordinals(mesh: &Value) -> TestResult<Vec<Vec<usize>>> {
    let order = attribute(mesh, "custom_normal")?.get("corner_order");
    let mut next = 0;
    let mut rows = Vec::new();
    for face in mesh["faces"].as_array().ok_or("mesh faces are missing")? {
        let id = face["id"].as_u64().ok_or("face id is missing")?;
        let corners = face["v"]
            .as_array()
            .ok_or("face vertices are missing")?
            .len();
        let row: Vec<usize> = match order {
            Some(order) => order
                .get(format!("f{id}"))
                .and_then(Value::as_array)
                .ok_or_else(|| format!("corner_order has no row for face {id}"))?
                .iter()
                .map(|ordinal| {
                    ordinal
                        .as_u64()
                        .map(|ordinal| ordinal as usize)
                        .ok_or("corner_order ordinal is not an unsigned integer")
                })
                .collect::<Result<_, _>>()?,
            None => (next..next + corners).collect(),
        };
        next += corners;
        rows.push(row);
    }
    Ok(rows)
}

/// Whether the loop basis is the flattened mesh order: corner `k` of the flattened corners is loop `k`.
fn is_identity(basis: &[Vec<usize>]) -> bool {
    basis
        .iter()
        .flatten()
        .copied()
        .eq(0..basis.iter().map(Vec::len).sum::<usize>())
}

/// (authored, automatic) corner counts of a mesh's stored `normal_code` layer.
fn stored_counts(mesh: &Value) -> TestResult<(usize, usize)> {
    let (mut authored, mut automatic) = (0, 0);
    for row in normal_code_rows(mesh)? {
        for stored in row {
            match stored {
                StoredCorner::Auto => automatic += 1,
                StoredCorner::Code(_) => authored += 1,
                StoredCorner::ExplicitZero => {
                    return Err("a Blender-backed mesh cannot hold a literal zero vector".into());
                }
            }
        }
    }
    Ok((authored, automatic))
}

/// Checks Potter's stored `normal_code` state against Blender's own dump of the same `.blend`.
/// Potter corner `[face][corner]` maps to Blender loop `loop_ordinals[face][corner]`: that loop
/// must sit at the corner's position inside a polygon with the face's center, and its raw code
/// pair must equal the stored code (`null` is `[0, 0]`).
fn check_codes_against_blender(mesh: &Value, dumped: &Value, label: &str) -> TestResult {
    let rows = normal_code_rows(mesh)?;
    let ordinals = loop_ordinals(mesh)?;
    let faces = potter_faces(mesh)?;
    let blender = blender_loops(dumped)?;
    let codes = blender
        .code
        .as_ref()
        .ok_or_else(|| format!("{label}: Blender found no custom_normal layer"))?;
    for (face_index, face) in faces.iter().enumerate() {
        for (corner, stored) in rows[face_index].iter().enumerate() {
            let loop_index = ordinals[face_index][corner];
            let at = format!("{label}: face {face_index} corner {corner} at loop {loop_index}");
            if loop_index >= blender.co.len() {
                return Err(format!("{at} is outside the Blender loops").into());
            }
            assert_floats(
                &blender.co[loop_index],
                &face.corners[corner],
                FLOAT_TOLERANCE,
                &format!("{at}: position"),
            );
            assert_floats(
                &blender.center[loop_index],
                &face.center,
                FLOAT_TOLERANCE,
                &format!("{at}: face center"),
            );
            let raw = codes[loop_index];
            match stored {
                StoredCorner::Auto => assert_eq!(raw, [0, 0], "{at}: stored Auto"),
                StoredCorner::Code(code) => assert_eq!(raw, *code, "{at}: stored code"),
                StoredCorner::ExplicitZero => {
                    return Err(format!("{at}: a literal zero has no Blender code").into());
                }
            }
        }
    }
    Ok(())
}

/// The loop whose `(co, center)` key is nearest to `(co, center)`, which must lie within `1e-4`
/// (the bound the existing Subdivision test uses).
fn nearest_corner(
    keys: &[([f64; 3], [f64; 3])],
    co: [f64; 3],
    center: [f64; 3],
) -> TestResult<usize> {
    let (index, distance) = keys
        .iter()
        .enumerate()
        .map(|(index, (key_co, key_center))| {
            (
                index,
                corner_distance(*key_co, co) + corner_distance(*key_center, center),
            )
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or("no corners to match")?;
    if distance < 1.0e-4 {
        Ok(index)
    } else {
        Err(
            format!("no corner at {co:?} with center {center:?}; nearest distance {distance}")
                .into(),
        )
    }
}

/// Matches each Blender reference corner (`co`, `center`) to exactly one loop of `actual`. Returns
/// the loop of each reference corner, in reference order.
fn match_reference_loops(
    actual: &BlenderLoops,
    reference: &[Value],
    label: &str,
) -> TestResult<Vec<usize>> {
    if reference.len() != actual.co.len() {
        return Err(format!(
            "{label}: Blender has {} corners, the reference has {}",
            actual.co.len(),
            reference.len()
        )
        .into());
    }
    let keys = loop_keys(actual);
    let mut claimed = BTreeSet::new();
    let mut loops = Vec::with_capacity(reference.len());
    for (index, wanted) in reference.iter().enumerate() {
        let co = vec3(&wanted["co"])?;
        let center = vec3(&wanted["center"])?;
        let loop_index = nearest_corner(&keys, co, center)
            .map_err(|error| format!("{label} corner {index}: {error}"))?;
        assert!(
            claimed.insert(loop_index),
            "{label} corner {index} matches loop {loop_index} twice"
        );
        loops.push(loop_index);
    }
    Ok(loops)
}

/// Matches each Blender reference corner (`co`, `center`, `normal`) to one loop of `actual` and
/// checks that loop's normal within `NORMAL_TOLERANCE`. Returns the loop of each reference corner.
fn match_reference_normals(
    actual: &BlenderLoops,
    reference: &[Value],
    label: &str,
) -> TestResult<Vec<usize>> {
    let loops = match_reference_loops(actual, reference, label)?;
    for (index, (wanted, &loop_index)) in reference.iter().zip(&loops).enumerate() {
        let co = vec3(&wanted["co"])?;
        let normal = vec3(&wanted["normal"])?;
        assert!(
            vectors_close(actual.normal[loop_index], normal, NORMAL_TOLERANCE),
            "{label} corner {index} at {co:?}: Potter-exported normal {:?} differs from Blender {normal:?}",
            actual.normal[loop_index]
        );
    }
    Ok(loops)
}

/// The layer or attribute named `name` in a dump list.
fn named<'a>(layers: &'a [Value], name: &str) -> TestResult<&'a Value> {
    layers
        .iter()
        .find(|layer| layer["name"] == name)
        .ok_or_else(|| format!("Blender dump has no {name}").into())
}

/// Checks Blender's readback of an exported Subdivision against Blender's own evaluation: every UV
/// layer, color layer, and generic corner attribute at its corner identity, and every generic face
/// attribute at its face center. Each dumped value must match within the same tolerance as the
/// stored form (byte color uses one byte step).
fn check_subdivision_data(object: &Value, reference: &Value, label: &str) -> TestResult {
    let reference_loops = reference["loops"]
        .as_array()
        .ok_or("reference loops are missing")?;
    let reference_faces = reference["faces"]
        .as_array()
        .ok_or("reference faces are missing")?;
    let actual = blender_loops(object)?;
    let loops = match_reference_loops(&actual, reference_loops, label)?;
    let uv_layers = object["uv_layers"]
        .as_array()
        .ok_or("dumped UV layers are missing")?;
    let color_layers = object["color_layers"]
        .as_array()
        .ok_or("dumped color layers are missing")?;
    let attributes = object["attributes"]
        .as_array()
        .ok_or("dumped attributes are missing")?;
    let dumped_faces = object["faces"]
        .as_array()
        .ok_or("dumped faces are missing")?;

    for (corner, wanted) in reference_loops.iter().enumerate() {
        let loop_index = loops[corner];
        for (name, expected) in wanted["uv"].as_object().ok_or("reference UV is missing")? {
            let layer = named(uv_layers, name)?;
            assert_floats(
                &floats(&layer["values"][loop_index])?,
                &floats(expected)?,
                FLOAT_TOLERANCE,
                &format!("{label} UV {name} corner {corner}"),
            );
        }
        for (name, expected) in wanted["color"]
            .as_object()
            .ok_or("reference color is missing")?
        {
            let layer = named(color_layers, name)?;
            let tolerance = if layer["data_type"] == "BYTE_COLOR" {
                BYTE_TOLERANCE
            } else {
                FLOAT_TOLERANCE
            };
            assert_floats(
                &floats(&layer["values"][loop_index])?,
                &floats(expected)?,
                tolerance,
                &format!("{label} color {name} corner {corner}"),
            );
        }
        for (name, expected) in wanted["corner"]
            .as_object()
            .ok_or("reference corner data is missing")?
        {
            let layer_value = named(attributes, name)?;
            assert_close(
                number(&layer_value["values"][loop_index])?,
                number(expected)?,
                FLOAT_TOLERANCE,
                &format!("{label} {name} corner {corner}"),
            );
        }
    }

    let centers: Vec<[f64; 3]> = dumped_faces
        .iter()
        .map(|face| vec3(&face["center"]))
        .collect::<TestResult<_>>()?;
    for (index, wanted) in reference_faces.iter().enumerate() {
        let center = vec3(&wanted["center"])?;
        let face = centers
            .iter()
            .position(|candidate| corner_distance(*candidate, center) < 1.0e-4)
            .ok_or_else(|| format!("{label} face {index} has no dumped face at {center:?}"))?;
        for (name, expected) in wanted["face"]
            .as_object()
            .ok_or("reference face data is missing")?
        {
            let layer_value = named(attributes, name)?;
            assert_close(
                number(&layer_value["values"][face])?,
                number(expected)?,
                FLOAT_TOLERANCE,
                &format!("{label} {name} face {index}"),
            );
        }
    }
    Ok(())
}

/// Attaches the native corner and face data to `blend` (a copy of the oracle) and returns the new file.
fn attach_native_layers(blender: &Path, root: &Path, blend: &Path) -> TestResult<PathBuf> {
    let script = write_script(root, "attach_layers.py", ATTACH_LAYERS_SCRIPT)?;
    let out = root.join("attached.blend");
    run_blender(blender, &script, Some(blend), &out)?;
    Ok(out)
}

/// Blender's evaluated Subdivision corner and face data of `blend` (see the script above).
fn subdivision_data_reference(blender: &Path, root: &Path, blend: &Path) -> TestResult<Value> {
    let script = write_script(
        root,
        "subdivision_data_reference.py",
        SUBDIVISION_DATA_REFERENCE_SCRIPT,
    )?;
    let out = root.join("subdivision_data_reference.json");
    run_blender(blender, &script, Some(blend), &out)?;
    Ok(serde_json::from_slice(&fs::read(out)?)?)
}

/// Blender's no-layer authoring reference for the Subdivision topology of `Regress`.
fn authoring_reference(blender: &Path, root: &Path, blend: &Path) -> TestResult<Value> {
    let script = write_script(root, "authoring_reference.py", AUTHORING_REFERENCE_SCRIPT)?;
    let out = root.join("authoring_reference.json");
    run_blender(blender, &script, Some(blend), &out)?;
    Ok(serde_json::from_slice(&fs::read(out)?)?)
}

#[test]
fn subdivision_codes_and_corner_data_follow_one_loop_basis_through_blend() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (oracle, expected) = build_oracle(&blender, root.path())?;
    let attached = attach_native_layers(&blender, root.path(), &oracle)?;
    let data_reference = subdivision_data_reference(&blender, root.path(), &attached)?;
    let (project, _) = import_blend(&blender, root.path(), "project", &attached)?;
    let node = node_id(&read_scene(&project)?, "Regress")?;
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "modifier.create", "target": {"id": node}, "id": "subdivide", "type": "subdivision", "params": {"levels": 1, "use_custom_normals": true}},
            {"op": "modifier.apply", "target": {"id": node}, "modifier_id": "subdivide"}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    // The Subdivision result stores its codes with a nonidentity loop basis.
    let subdivided = read_scene(&project)?;
    let subdivided_mesh = mesh_of_node(&subdivided, "Regress")?;
    assert!(
        !is_identity(&loop_ordinals(subdivided_mesh)?),
        "the Subdivision result must store a nonidentity corner_order"
    );
    let (authored, automatic) = stored_counts(subdivided_mesh)?;
    assert!(
        authored > 0 && automatic > 0,
        "the Subdivision result must mix authored and automatic corners"
    );

    // Export and reopen in Blender: raw codes map through the stored basis, and every UV, color,
    // and generic value sits at the same corner identity as Blender's own evaluation.
    let exported = root.path().join("subdivided.blend");
    export(&blender, &project, "blend", &exported)?;
    let dump = dump_blend(&blender, root.path(), &exported, "subdivided")?;
    let exported_object = dumped_object(&dump, "Regress")?;
    check_codes_against_blender(subdivided_mesh, exported_object, "exported Subdivision")?;
    check_subdivision_data(exported_object, &data_reference, "exported Subdivision")?;

    // Reimport the saved file into a second project, then export again before any deformation.
    let (reimported, _) = import_blend(&blender, root.path(), "reimported", &exported)?;
    let reimported_scene = read_scene(&reimported)?;
    let reimported_node = node_id(&reimported_scene, "Regress")?;
    check_codes_against_blender(
        mesh_of_node(&reimported_scene, "Regress")?,
        exported_object,
        "reimported Subdivision",
    )?;
    let reimported_blend = root.path().join("reimported.blend");
    export(&blender, &reimported, "blend", &reimported_blend)?;
    let reimported_dump = dump_blend(&blender, root.path(), &reimported_blend, "reimported")?;
    let reimported_object = dumped_object(&reimported_dump, "Regress")?;
    check_codes_against_blender(
        mesh_of_node(&reimported_scene, "Regress")?,
        reimported_object,
        "re-exported Subdivision",
    )?;
    check_subdivision_data(
        reimported_object,
        &data_reference,
        "re-exported Subdivision",
    )?;

    // A geometry-only modifier continues the chain: codes survive and decode on displaced fans.
    let applied = apply_operations(
        &reimported,
        root.path(),
        json!([
            {"op": "modifier.create", "target": {"id": reimported_node}, "id": "displace", "type": "displace", "params": {"strength": 0.1, "mid_level": 0.0}},
            {"op": "modifier.apply", "target": {"id": reimported_node}, "modifier_id": "displace"}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let displaced = read_scene(&reimported)?;
    let (authored, automatic) = stored_counts(mesh_of_node(&displaced, "Regress")?)?;
    assert!(
        authored > 0 && automatic > 0,
        "Displace must keep both authored and automatic corners"
    );

    let displaced_blend = root.path().join("displaced.blend");
    export(&blender, &reimported, "blend", &displaced_blend)?;
    let displaced_dump = dump_blend(&blender, root.path(), &displaced_blend, "displaced")?;
    let actual = blender_loops(dumped_object(&displaced_dump, "Regress")?)?;
    let reference = expected["regress"]["reference_corners"]
        .as_array()
        .ok_or("reference corners are missing")?;
    match_reference_normals(&actual, reference, "reimported Subdivision then Displace")?;
    Ok(())
}

#[test]
fn subdivision_without_custom_normals_keeps_loop_basis_and_later_authoring_allocates_the_layer()
-> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, _) = build_oracle(&blender, root.path())?;
    let reference = authoring_reference(&blender, root.path(), &blend)?;
    let reference_loops = reference["loops"]
        .as_array()
        .ok_or("authoring reference loops are missing")?;
    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let node = node_id(&read_scene(&project)?, "Regress")?;
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "modifier.create", "target": {"id": node}, "id": "subdivide", "type": "subdivision", "params": {"levels": 1, "use_custom_normals": false}},
            {"op": "modifier.apply", "target": {"id": node}, "modifier_id": "subdivide"}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    // The saved project, read back from disk: no value layer, but the nonidentity basis remains.
    let saved = read_scene(&project)?;
    let saved_mesh = mesh_of_node(&saved, "Regress")?;
    let layer = attribute(saved_mesh, "custom_normal")?;
    assert_eq!(layer["domain"], "corner");
    assert_eq!(layer["type"], "normal_code");
    assert!(
        layer.get("values").is_none(),
        "Subdivision without use_custom_normals must not allocate a value layer"
    );
    assert!(
        layer.as_object().is_some_and(|fields| fields
            .keys()
            .all(|key| matches!(key.as_str(), "domain" | "type" | "corner_order"))),
        "basis-only state holds only domain, type and corner_order"
    );
    let basis = loop_ordinals(saved_mesh)?;
    assert!(
        layer.get("corner_order").is_some() && !is_identity(&basis),
        "the Subdivision result must store a nonidentity corner_order"
    );

    // Author normals through the saved basis. Each Potter corner takes Blender's input vector for
    // the loop at the same position and face center; automatic loops send null.
    let keys = reference_keys(reference_loops)?;
    let mut normals = Map::new();
    let mut face_ids = Vec::new();
    let faces = potter_faces(saved_mesh)?;
    let face_json = saved_mesh["faces"]
        .as_array()
        .ok_or("mesh faces are missing")?;
    for (face, face_json) in faces.iter().zip(face_json) {
        let id = face_json["id"].as_u64().ok_or("face id is missing")?;
        let row = face
            .corners
            .iter()
            .map(|co| -> TestResult<Value> {
                let loop_index = nearest_corner(&keys, *co, face.center)?;
                Ok(reference_loops[loop_index]["vector"].clone())
            })
            .collect::<TestResult<Vec<_>>>()?;
        normals.insert(format!("f{id}"), Value::Array(row));
        face_ids.push(format!("f{id}"));
    }
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "mesh.set_custom_normals", "target": {"id": node}, "elements": {"domain": "face", "ids": face_ids}, "normals": normals}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    let authored_state = read_scene(&project)?;
    let authored_mesh = mesh_of_node(&authored_state, "Regress")?;
    assert_eq!(
        loop_ordinals(authored_mesh)?,
        basis,
        "authoring must keep the stored loop basis"
    );
    let (authored, automatic) = stored_counts(authored_mesh)?;
    assert_eq!(
        (authored as u64, automatic as u64),
        (
            reference["authored"]
                .as_u64()
                .ok_or("reference authored count is missing")?,
            reference["automatic"]
                .as_u64()
                .ok_or("reference automatic count is missing")?
        ),
        "stored authored and automatic corners must match Blender's reference"
    );

    let exported = root.path().join("authored.blend");
    export(&blender, &project, "blend", &exported)?;
    let dump = dump_blend(&blender, root.path(), &exported, "authored")?;
    let object = dumped_object(&dump, "Regress")?;
    check_codes_against_blender(authored_mesh, object, "authored Subdivision")?;

    // Blender allocated the layer, and its normals and Auto statuses match its own reference.
    let actual = blender_loops(object)?;
    let codes = actual
        .code
        .as_ref()
        .ok_or("Blender allocated no custom_normal layer after authoring")?;
    let matched = match_reference_normals(&actual, reference_loops, "authored Subdivision")?;
    for (wanted, loop_index) in reference_loops.iter().zip(matched) {
        let automatic_in_blender = code_pair(&wanted["code"])? == [0, 0];
        assert_eq!(
            codes[loop_index] == [0, 0],
            automatic_in_blender,
            "Auto status of loop {loop_index} differs from Blender's reference"
        );
    }
    Ok(())
}
const NONMANIFOLD_SUBDIVISION_ORACLE: &str = r#"
import json
import math
import sys

import bpy
if bpy.app.version != (5, 2, 2):
    raise SystemExit("this oracle requires Blender 5.2.2, got %r" % (bpy.app.version,))


OUT_BLEND, OUT_JSON = sys.argv[sys.argv.index("--") + 1:]
vertices = [
    (0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0),
    (-1, 0, 0), (-1, -1, 0), (0, 1, 0), (0, 2, 1),
]
faces = [(0, 1, 2, 3), (1, 0, 4, 5), (0, 1, 6, 7)]
mesh = bpy.data.meshes.new("NonmanifoldData")
mesh.from_pydata(vertices, [], faces)
mesh.update()
obj = bpy.data.objects.new("Nonmanifold", mesh)
bpy.context.scene.collection.objects.link(obj)
for polygon in mesh.polygons:
    polygon.use_smooth = True
edge_map = {tuple(sorted(edge.vertices)): edge for edge in mesh.edges}
for pair in ((0, 1), (2, 3), (4, 5), (6, 7)):
    edge_map[tuple(sorted(pair))].use_seam = True
crease = mesh.attributes.new(name="crease_edge", type="FLOAT", domain="EDGE")
for item in crease.data:
    item.value = 0.0

vectors = []
for polygon in mesh.polygons:
    for corner in range(polygon.loop_total):
        loop_index = polygon.loop_start + corner
        vertex_index = mesh.loops[loop_index].vertex_index
        x, y, z = mesh.vertices[vertex_index].co
        vector = (
            0.13 + 0.025 * polygon.index + 0.02 * corner,
            -0.11 + 0.015 * vertex_index,
            1.0 + 0.03 * x - 0.02 * y + 0.01 * z,
        )
        length = math.sqrt(sum(component * component for component in vector))
        vectors.append(tuple(component / length for component in vector))
mesh.normals_split_custom_set(vectors)
mesh.update()
bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)

modifier = obj.modifiers.new("Nonmanifold Subdivision", "SUBSURF")
modifier.subdivision_type = "CATMULL_CLARK"
modifier.levels = 2
modifier.render_levels = 2
modifier.quality = 3
modifier.boundary_smooth = "ALL"
modifier.uv_smooth = "PRESERVE_BOUNDARIES"
modifier.use_creases = True
modifier.use_custom_normals = True
modifier.use_limit_surface = True
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = obj.evaluated_get(depsgraph)
result = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
codes = result.attributes.get("custom_normal")
if codes is None:
    raise SystemExit("Blender Subdivision returned no custom_normal layer")
raw_codes = [tuple(int(component) for component in item.value) for item in codes.data]
loops = []
for polygon in result.polygons:
    center = [float(component) for component in polygon.center]
    for corner in range(polygon.loop_total):
        loop_index = polygon.loop_start + corner
        vertex_index = result.loops[loop_index].vertex_index
        code = raw_codes[loop_index]
        loops.append({
            "co": [float(component) for component in result.vertices[vertex_index].co],
            "center": center,
            "normal": [float(component) for component in result.corner_normals[loop_index].vector],
            "code": None if code == (0, 0) else list(code),
        })
witness = next(
    (polygon for polygon in result.polygons if list(polygon.vertices) == [0, 11, 56, 29]),
    None,
)
if witness is None:
    raise SystemExit("Blender nonmanifold Subdivision witness face is missing")
witness_codes = [
    None if raw_codes[index] == (0, 0) else list(raw_codes[index])
    for index in range(witness.loop_start, witness.loop_start + witness.loop_total)
]
with open(OUT_JSON, "w", encoding="utf-8") as handle:
    json.dump({
        "blender": bpy.app.version_string,
        "loops": loops,
        "witness": {"vertices": list(witness.vertices), "codes": witness_codes},
    }, handle)
evaluated.to_mesh_clear()
"#;

#[test]
fn nonmanifold_subdivision_auto_codes_match_blenders_evaluated_fan() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(
        root.path(),
        "nonmanifold_subdivision_oracle.py",
        NONMANIFOLD_SUBDIVISION_ORACLE,
    )?;
    let source_blend = root.path().join("nonmanifold.blend");
    let expected_path = root.path().join("nonmanifold.expected.json");
    run_blender_arguments(&blender, &script, None, &[&source_blend, &expected_path])?;
    let expected: Value = serde_json::from_slice(&fs::read(&expected_path)?)?;
    let source_loops = expected["loops"]
        .as_array()
        .ok_or("Blender source Subdivision loops are missing")?;
    let witness_codes = expected["witness"]["codes"]
        .as_array()
        .ok_or("Blender nonmanifold witness codes are missing")?;
    assert_eq!(expected["witness"]["vertices"], json!([0, 11, 56, 29]));
    assert_eq!(
        witness_codes[0],
        Value::Null,
        "source witness loop zero is Auto"
    );
    assert_eq!(
        witness_codes[1],
        Value::Null,
        "source witness loop one is Auto"
    );

    let (project, _) = import_blend(&blender, root.path(), "project", &source_blend)?;
    let node = node_id(&read_scene(&project)?, "Nonmanifold")?;
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {
                "op": "modifier.create",
                "target": {"id": node},
                "id": "subsurf",
                "type": "subdivision",
                "params": {
                    "levels": 2,
                    "subdivision_type": "CATMULL_CLARK",
                    "quality": 3,
                    "boundary_smooth": "ALL",
                    "uv_smooth": "PRESERVE_BOUNDARIES",
                    "use_creases": true,
                    "use_custom_normals": true,
                    "use_limit_surface": true
                }
            },
            {"op": "modifier.apply", "target": {"id": node}, "modifier_id": "subsurf"}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let scene = read_scene(&project)?;
    let mesh = mesh_of_node(&scene, "Nonmanifold")?;
    let actual_rows = normal_code_rows(mesh)?;
    let faces = potter_faces(mesh)?;
    let keys = reference_keys(source_loops)?;
    let mut matched = BTreeSet::new();
    for (face_index, face) in faces.iter().enumerate() {
        for (corner_index, position) in face.corners.iter().enumerate() {
            let source_loop = nearest_corner(&keys, *position, face.center)?;
            assert!(
                matched.insert(source_loop),
                "source loop {source_loop} matched twice"
            );
            let raw = &source_loops[source_loop]["code"];
            let expected_code = if raw.is_null() {
                StoredCorner::Auto
            } else {
                StoredCorner::Code(code_pair(raw)?)
            };
            assert!(
                code_within_one_unit(&actual_rows[face_index][corner_index], &expected_code),
                "Subsurf loop {source_loop} code within one unit of Blender: actual {:?}, expected {expected_code:?}",
                actual_rows[face_index][corner_index]
            );
        }
    }
    assert_eq!(matched.len(), source_loops.len());
    Ok(())
}

const MIRROR_BLEND_CHAIN: &str = r#"
import bpy
import json
import sys

fixture_path, base_path, mirror_path, bisect_path, expected_path = sys.argv[sys.argv.index("--") + 1:]
with open(fixture_path, encoding="utf-8") as handle:
    source = json.load(handle)["input"]
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("SubjectMesh")
mesh.from_pydata(source["vertices"], source.get("edges", []), source["faces"])
mesh.update()
for polygon, smooth in zip(mesh.polygons, source["face_smooth"]):
    polygon.use_smooth = smooth
edge_by_vertices = {tuple(sorted(edge.vertices)): edge for edge in mesh.edges}
for a, b in source.get("sharp_edges", []):
    edge_by_vertices[tuple(sorted((a, b)))].use_edge_sharp = True
for a, b in source.get("seam_edges", []):
    edge_by_vertices[tuple(sorted((a, b)))].use_seam = True
mesh.update()
corner_normals = []
for row, polygon in zip(source["custom_normals"], mesh.polygons):
    for corner in range(polygon.loop_total):
        vector = row[corner]
        corner_normals.append((0.0, 0.0, 0.0) if vector is None else tuple(vector))
mesh.normals_split_custom_set(corner_normals)
subject = bpy.data.objects.new("Subject", mesh)
bpy.context.scene.collection.objects.link(subject)
bpy.context.view_layer.objects.active = subject
subject.select_set(True)
bpy.ops.wm.save_as_mainfile(filepath=base_path)
mirror_x = subject.modifiers.new("MirrorX", "MIRROR")
mirror_x.use_axis = (True, False, False)
bpy.ops.object.modifier_apply(modifier=mirror_x.name)
bpy.ops.wm.save_as_mainfile(filepath=mirror_path)
mirror_y = subject.modifiers.new("MirrorY", "MIRROR")
mirror_y.use_axis = (False, True, False)
mirror_y.use_bisect_axis = (False, True, False)
bpy.ops.object.modifier_apply(modifier=mirror_y.name)
bpy.ops.wm.save_as_mainfile(filepath=bisect_path)
wave = subject.modifiers.new("Wave", "WAVE")
wave.height = 0.2
wave.width = 1.5
wave.speed = 0.0
wave.time_offset = 0.25
bpy.ops.object.modifier_apply(modifier=wave.name)
bpy.ops.wm.save_as_mainfile(filepath=expected_path)
"#;

fn codes_by_loop(mesh: &Value) -> TestResult<BTreeMap<usize, StoredCorner>> {
    let codes = normal_code_rows(mesh)?;
    let ordinals = loop_ordinals(mesh)?;
    let mut result = BTreeMap::new();
    for (code_row, ordinal_row) in codes.iter().zip(ordinals) {
        for (code, ordinal) in code_row.iter().zip(ordinal_row) {
            if result.insert(ordinal, *code).is_some() {
                return Err(format!("duplicate custom-normal loop ordinal {ordinal}").into());
            }
        }
    }
    Ok(result)
}

fn dumped_reference_loop_rows(object: &Value) -> TestResult<Vec<Value>> {
    let vertices = object["vertices"]
        .as_array()
        .ok_or("dumped vertices are missing")?;
    let loops = object["loops"]
        .as_array()
        .ok_or("dumped loops are missing")?;
    let normals = object["corner_normals"]
        .as_array()
        .ok_or("dumped corner normals are missing")?;
    let mut result = Vec::with_capacity(loops.len());
    for face in object["faces"]
        .as_array()
        .ok_or("dumped faces are missing")?
    {
        let start = usize::try_from(face["loop_start"].as_u64().ok_or("loop start is missing")?)?;
        let count = usize::try_from(face["loop_total"].as_u64().ok_or("loop count is missing")?)?;
        let center = face.get("center").ok_or("face center is missing")?;
        for loop_index in start..start + count {
            let vertex = usize::try_from(
                loops[loop_index]["vertex"]
                    .as_u64()
                    .ok_or("loop vertex is missing")?,
            )?;
            result.push(json!({
                "co": vertices[vertex],
                "center": center,
                "normal": normals[loop_index],
            }));
        }
    }
    Ok(result)
}

#[test]
fn mirror_raw_codes_round_trip_before_bisect_and_wave() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pr18b_mirror/custom_chain_x_then_y_vertex.json");
    let script = write_script(root.path(), "mirror_code_chain.py", MIRROR_BLEND_CHAIN)?;
    let base = root.path().join("mirror_base.blend");
    let expected_stage = root.path().join("mirror_x_reference.blend");
    let expected_bisect = root.path().join("mirror_x_y_bisect_reference.blend");
    let expected_final = root.path().join("mirror_x_y_wave_reference.blend");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            &fixture,
            &base,
            &expected_stage,
            &expected_bisect,
            &expected_final,
        ],
    )?;

    let (project, imported) = import_blend(&blender, root.path(), "mirror_x", &base)?;
    assert!(
        losses(&imported).is_empty(),
        "Mirror base import losses: {:?}",
        losses(&imported)
    );
    let imported_scene = read_scene(&project)?;
    let subject = node_id(&imported_scene, "Subject")?;
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op":"modifier.create","target":{"id":subject},"id":"mirror_x","type":"mirror","params":{"use_axis":[true,false,false]}},
            {"op":"modifier.apply","target":{"id":subject},"modifier_id":"mirror_x"}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let stage_scene = read_scene(&project)?;
    let stage_mesh = mesh_of_node(&stage_scene, "Subject")?;
    let stage_codes = codes_by_loop(stage_mesh)?;
    let stage_blend = root.path().join("potter_mirror_x.blend");
    export(&blender, &project, "blend", &stage_blend)?;
    let stage_dump = dump_blend(&blender, root.path(), &stage_blend, "potter_mirror_x")?;
    check_codes_against_blender(
        stage_mesh,
        dumped_object(&stage_dump, "Subject")?,
        "Mirror X output",
    )?;

    let (reloaded_project, _) =
        import_blend(&blender, root.path(), "mirror_x_reloaded", &stage_blend)?;
    let reloaded_scene = read_scene(&reloaded_project)?;
    let reloaded_mesh = mesh_of_node(&reloaded_scene, "Subject")?;
    assert_eq!(
        codes_by_loop(reloaded_mesh)?,
        stage_codes,
        "raw Mirror codes across .blend reload"
    );
    let reloaded_subject = node_id(&reloaded_scene, "Subject")?;
    let continued = apply_operations(
        &reloaded_project,
        root.path(),
        json!([
            {"op":"modifier.create","target":{"id":reloaded_subject},"id":"mirror_y","type":"mirror","params":{"use_axis":[false,true,false],"use_bisect_axis":[false,true,false]}},
            {"op":"modifier.apply","target":{"id":reloaded_subject},"modifier_id":"mirror_y"},
            {"op":"modifier.create","target":{"id":reloaded_subject},"id":"wave","type":"wave","params":{"height":0.2,"width":1.5,"speed":0.0,"time_offset":0.25}},
            {"op":"modifier.apply","target":{"id":reloaded_subject},"modifier_id":"wave"}
        ]),
    )?;
    assert_eq!(continued["ok"], true, "{continued}");
    let continued_blend = root.path().join("potter_mirror_x_y_wave.blend");
    export(&blender, &reloaded_project, "blend", &continued_blend)?;
    let actual_dump = dump_blend(
        &blender,
        root.path(),
        &continued_blend,
        "potter_mirror_x_y_wave",
    )?;
    let expected_dump = dump_blend(
        &blender,
        root.path(),
        &expected_final,
        "blender_mirror_x_y_wave",
    )?;
    let bisect_dump = dump_blend(
        &blender,
        root.path(),
        &expected_bisect,
        "blender_mirror_x_y_bisect",
    )?;
    let bisected_object = dumped_object(&bisect_dump, "Subject")?;
    let expected_object = dumped_object(&expected_dump, "Subject")?;
    let before_wave = bisected_object["vertices"]
        .as_array()
        .ok_or("bisect vertices missing")?;
    let after_wave = expected_object["vertices"]
        .as_array()
        .ok_or("wave vertices missing")?;
    assert_eq!(
        before_wave.len(),
        after_wave.len(),
        "Wave preserves Mirror topology"
    );
    let wave_changed_positions = before_wave
        .iter()
        .zip(after_wave)
        .map(|(before, after)| -> TestResult<bool> {
            Ok(corner_distance(vec3(before)?, vec3(after)?) > 1.0e-4)
        })
        .collect::<TestResult<Vec<_>>>()?
        .into_iter()
        .any(|changed| changed);
    assert!(
        wave_changed_positions,
        "Wave must deform the bisected Mirror output"
    );
    let expected_loops = dumped_reference_loop_rows(expected_object)?;
    let actual = blender_loops(dumped_object(&actual_dump, "Subject")?)?;
    match_reference_normals(
        &actual,
        &expected_loops,
        "reloaded Mirror X then Y bisect and Wave",
    )?;
    Ok(())
}
