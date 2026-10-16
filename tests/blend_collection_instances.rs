//! Blender parity for collection instances. Each case in `fixtures/blend_collection_instances.py` is
//! one `.blend` scene, written in full and as a geometry-only variant whose members have no color
//! attributes, vertex groups, or seams. Potter imports the geometry-only variant and exports GLB,
//! and Blender exports the same variant itself. glTF export refuses `blender_attributes`, so only
//! geometry is compared there: every primitive is reduced to its triangles' world-space corners, each
//! corner a (position, normal, TEXCOORD values) tuple. Triangles sharing an edge form a patch, and
//! patches must match with tolerance by triangle count and unique corner set, and each primitive's
//! total corner count (with multiplicity) must agree. Per-corner multiplicity inside a patch is not
//! compared, since diagonal choice inside a quad changes it. Cases that
//! carry attributes also round-trip the full scene through Potter's `.blend` export, and Blender's
//! reading of both files must agree on each member's color attributes, vertex groups, and seams.
//! Blender-gated: `POTTER_REQUIRE_BLENDER` turns a missing Blender into a failure.

use std::{collections::BTreeSet, error::Error, fs, path::Path, process::Command};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const BUILDER: &str = include_str!("fixtures/blend_collection_instances.py");
const DUMPER: &str = include_str!("fixtures/blend_collection_instances_dump.py");

/// Model-space float32 round trips through two exporters: positions and UVs agree to 1e-4, and
/// normals to 1e-3 after normalization.
const POSITION_TOLERANCE: f64 = 1.0e-4;
const NORMAL_TOLERANCE: f64 = 1.0e-3;
const UV_TOLERANCE: f64 = 1.0e-4;
/// `BYTE_COLOR` quantizes to 1/255, so colors compare to two steps of that.
const COLOR_TOLERANCE: f64 = 0.01;
const WEIGHT_TOLERANCE: f64 = 1.0e-5;
const SEAM_TOLERANCE: f64 = 1.0e-5;

const IDENTITY: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Check {
    /// Every primitive's world-space corners match Blender's export.
    Geometry,
    /// Geometry, and a face whose material index exceeds the table clamps to the last slot: the
    /// exported primitives must include that slot's material with the clamped faces' corner count.
    ClampToLastSlot,
    /// Geometry, then each member's color, vertex group, and seam records match Blender's
    /// `.blend` round trip of the same scene.
    GeometryAndAttributes,
    /// Blender-authored members joined with a Potter `uv.unwrap` member. Blender's members match
    /// Blender's export; the Potter member matches its own standalone export.
    MixedWithPotterUnwrap,
    /// A textured member without UVs is refused, and Blender exports it without `TEXCOORD_0`.
    TexturedWithoutUv,
    /// A user metadata conflict fails only the instance that owns it.
    ConflictIsolated,
}

const CASES: [(&str, Check); 13] = [
    ("nested_instances", Check::GeometryAndAttributes),
    ("rotated_member", Check::Geometry),
    ("mirrored_member", Check::Geometry),
    ("uv_layer_names", Check::Geometry),
    ("potter_unwrap_mixed", Check::MixedWithPotterUnwrap),
    ("missing_uv_member", Check::Geometry),
    ("missing_uv_textured", Check::TexturedWithoutUv),
    ("mixed_smooth", Check::Geometry),
    ("custom_normals_rotated", Check::Geometry),
    ("out_of_range_material", Check::ClampToLastSlot),
    ("color_type_conflict", Check::GeometryAndAttributes),
    ("potter_id_reimport", Check::Geometry),
    ("metadata_conflict", Check::ConflictIsolated),
];

/// One triangle corner in world space: its position, its normal, and each TEXCOORD set it reads.
#[derive(Debug)]
struct Corner {
    position: [f64; 3],
    normal: [f64; 3],
    uvs: Vec<[f64; 2]>,
}

/// One exported primitive: its material, the number of TEXCOORD sets its mesh carries, and its
/// corners in index order.
#[derive(Debug)]
struct Primitive {
    material: Option<String>,
    texcoord_sets: usize,
    corners: Vec<Corner>,
}

fn ensure(condition: bool, message: impl Into<String>) -> TestResult {
    if condition {
        Ok(())
    } else {
        let message: String = message.into();
        Err(message.into())
    }
}

fn path_str(path: &Path) -> TestResult<&str> {
    Ok(path.to_str().ok_or("path is not UTF-8")?)
}

/// Runs `pot` and asserts success. The caller appends `--json`.
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

/// Runs `pot` and asserts that it refuses, returning the failure envelope it printed.
fn run_pot_refusal(args: &[&str]) -> TestResult<Value> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(args)
        .output()?;
    assert!(
        !output.status.success(),
        "pot {} unexpectedly succeeded",
        args.first().copied().unwrap_or("command")
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn run_blender(blender: &Path, script: &Path, arguments: &[&str]) -> TestResult {
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(script)
        .arg("--")
        .args(arguments)
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    ensure(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        format!(
            "Blender script {} failed: stdout={stdout} stderr={stderr}",
            script.display()
        ),
    )
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

/// Byte offset, element stride, and element count of an accessor.
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

/// Float elements of `width` components, unrounded.
fn float_values(
    gltf: &Value,
    bin: &[u8],
    accessor: &Value,
    width: usize,
) -> TestResult<Vec<Vec<f64>>> {
    let (start, stride, count) = accessor_layout(gltf, accessor, width * 4)?;
    let mut elements = Vec::with_capacity(count);
    for item in 0..count {
        let base = start + item * stride;
        let bytes = bin
            .get(base..base + width * 4)
            .ok_or("accessor exceeds the binary chunk")?;
        elements.push(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| {
                    f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
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

fn attribute_accessor<'a>(
    gltf: &'a Value,
    attributes: &Value,
    name: &str,
) -> TestResult<&'a Value> {
    let index = attributes[name]
        .as_u64()
        .ok_or_else(|| format!("primitive has no {name} attribute"))?;
    Ok(&gltf["accessors"][index as usize])
}

fn array_from<const N: usize>(values: &[f64]) -> TestResult<[f64; N]> {
    Ok(values.get(..N).ok_or("glTF element is short")?.try_into()?)
}

fn json_numbers<const N: usize>(value: &Value) -> TestResult<[f64; N]> {
    let numbers = value
        .as_array()
        .ok_or("glTF transform value must be an array")?
        .iter()
        .map(|number| {
            number
                .as_f64()
                .ok_or("glTF transform value must be a number")
        })
        .collect::<Result<Vec<f64>, &str>>()?;
    array_from::<N>(&numbers)
}

/// A node's local matrix, column-major: its `matrix`, or translation, rotation quaternion, and
/// scale composed as T * R * S.
fn local_matrix(node: &Value) -> TestResult<[f64; 16]> {
    if !node["matrix"].is_null() {
        return json_numbers::<16>(&node["matrix"]);
    }
    let translation = if node["translation"].is_null() {
        [0.0; 3]
    } else {
        json_numbers::<3>(&node["translation"])?
    };
    let rotation = if node["rotation"].is_null() {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        json_numbers::<4>(&node["rotation"])?
    };
    let scale = if node["scale"].is_null() {
        [1.0; 3]
    } else {
        json_numbers::<3>(&node["scale"])?
    };
    let [x, y, z, w] = rotation;
    let rows = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut matrix = [0.0; 16];
    for column in 0..3 {
        for row in 0..3 {
            matrix[column * 4 + row] = rows[row][column] * scale[column];
        }
    }
    matrix[12..15].copy_from_slice(&translation);
    matrix[15] = 1.0;
    Ok(matrix)
}

fn multiply(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = (0..4)
                .map(|k| a[k * 4 + row] * b[column * 4 + k])
                .sum::<f64>();
        }
    }
    out
}

fn transform_point(m: &[f64; 16], p: [f64; 3]) -> [f64; 3] {
    [0_usize, 1, 2].map(|row| m[row] * p[0] + m[4 + row] * p[1] + m[8 + row] * p[2] + m[12 + row])
}

/// Normals transform by the inverse transpose of the upper 3x3. The cofactor matrix equals
/// `det(M) * M^-T`, so a mirrored matrix (negative determinant) needs its sign flipped back; without
/// that, mirrored members get inward normals and fail against the outward ones Potter writes.
fn transform_normal(m: &[f64; 16], n: [f64; 3]) -> [f64; 3] {
    let a = |row: usize, column: usize| m[column * 4 + row];
    let determinant = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1))
        - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
        + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
    let sign = if determinant < 0.0 { -1.0 } else { 1.0 };
    let mut out = [0.0; 3];
    for (i, value) in out.iter_mut().enumerate() {
        *value = sign
            * (0..3)
                .map(|j| {
                    let (r1, r2) = ((i + 1) % 3, (i + 2) % 3);
                    let (c1, c2) = ((j + 1) % 3, (j + 2) % 3);
                    (a(r1, c1) * a(r2, c2) - a(r1, c2) * a(r2, c1)) * n[j]
                })
                .sum::<f64>();
    }
    let length = out.iter().map(|value| value * value).sum::<f64>().sqrt();
    if length > 0.0 {
        out.map(|value| value / length)
    } else {
        out
    }
}

fn read_primitive(
    gltf: &Value,
    bin: &[u8],
    primitive: &Value,
    world: &[f64; 16],
    materials: &[String],
) -> TestResult<Primitive> {
    let attributes = &primitive["attributes"];
    let index_accessor = &gltf["accessors"][primitive["indices"]
        .as_u64()
        .ok_or("primitive has no indices")? as usize];
    let indices = index_elements(gltf, bin, index_accessor)?;
    let positions = float_values(
        gltf,
        bin,
        attribute_accessor(gltf, attributes, "POSITION")?,
        3,
    )?;
    let normals = float_values(
        gltf,
        bin,
        attribute_accessor(gltf, attributes, "NORMAL")?,
        3,
    )?;
    let texcoord_sets = attributes
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(key, _)| key.starts_with("TEXCOORD_"))
        .count();
    let mut uv_sets = Vec::with_capacity(texcoord_sets);
    for set in 0..texcoord_sets {
        let accessor = attribute_accessor(gltf, attributes, &format!("TEXCOORD_{set}"))?;
        uv_sets.push(float_values(gltf, bin, accessor, 2)?);
    }
    let mut corners = Vec::with_capacity(indices.len());
    for vertex in indices {
        let position = array_from::<3>(positions.get(vertex).ok_or("index exceeds POSITION")?)?;
        let normal = array_from::<3>(normals.get(vertex).ok_or("index exceeds NORMAL")?)?;
        let mut uvs = Vec::with_capacity(uv_sets.len());
        for set in &uv_sets {
            uvs.push(array_from::<2>(
                set.get(vertex).ok_or("index exceeds TEXCOORD")?,
            )?);
        }
        corners.push(Corner {
            position: transform_point(world, position),
            normal: transform_normal(world, normal),
            uvs,
        });
    }
    Ok(Primitive {
        material: primitive["material"]
            .as_u64()
            .and_then(|slot| materials.get(slot as usize).cloned()),
        texcoord_sets,
        corners,
    })
}

/// Every primitive a scene renders, in world space, flattened through the node tree. A mesh
/// referenced by two nodes counts twice, as it renders twice.
fn primitives(path: &Path) -> TestResult<Vec<Primitive>> {
    let (gltf, bin) = read_glb(path)?;
    let materials: Vec<String> = gltf["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|item| item["name"].as_str().unwrap_or_default().to_owned())
        .collect();
    let mut stack: Vec<(u64, [f64; 16])> = gltf["scenes"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|scene| scene["nodes"].as_array().into_iter().flatten())
        .filter_map(Value::as_u64)
        .map(|index| (index, IDENTITY))
        .collect();
    let mut result = Vec::new();
    while let Some((index, parent)) = stack.pop() {
        let node = &gltf["nodes"][index as usize];
        let world = multiply(&parent, &local_matrix(node)?);
        stack.extend(
            node["children"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_u64)
                .map(|child| (child, world)),
        );
        let Some(mesh) = node["mesh"].as_u64() else {
            continue;
        };
        for primitive in gltf["meshes"][mesh as usize]["primitives"]
            .as_array()
            .into_iter()
            .flatten()
        {
            result.push(read_primitive(&gltf, &bin, primitive, &world, &materials)?);
        }
    }
    Ok(result)
}

fn close(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
}

fn corner_matches(expected: &Corner, actual: &Corner) -> bool {
    close(&expected.position, &actual.position, POSITION_TOLERANCE)
        && close(&expected.normal, &actual.normal, NORMAL_TOLERANCE)
        && expected.uvs.len() <= actual.uvs.len()
        && expected
            .uvs
            .iter()
            .zip(&actual.uvs)
            .all(|(e, a)| close(e, a, UV_TOLERANCE))
}

/// A polygon-level piece of a primitive: the triangles of one edge-connected run, their count, and
/// the run's unique corners. What is checked, with tolerance: each primitive's total corner count
/// (with multiplicity), and for every patch its triangle count and its unique (position, normal,
/// TEXCOORD) corner set, matched one-to-one. Per-corner multiplicity inside a patch is not checked:
/// Blender and Potter may split one quad along different diagonals, which changes how many times
/// each corner appears without changing the polygon, so no multiplicity is invariant across them.
struct Patch<'a> {
    triangles: usize,
    corners: Vec<&'a Corner>,
}

/// Union-find root with path halving.
fn root(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn patches(primitive: &Primitive) -> Vec<Patch<'_>> {
    let triangles: Vec<&[Corner]> = primitive.corners.chunks(3).collect();
    let mut parent: Vec<usize> = (0..triangles.len()).collect();
    for first in 0..triangles.len() {
        for second in first + 1..triangles.len() {
            let shared = triangles[first]
                .iter()
                .filter(|corner| {
                    triangles[second]
                        .iter()
                        .any(|other| corner_matches(corner, other))
                })
                .count();
            if shared >= 2 {
                let (a, b) = (root(&mut parent, first), root(&mut parent, second));
                parent[a] = b;
            }
        }
    }
    let mut grouped: Vec<Option<Patch<'_>>> = (0..triangles.len()).map(|_| None).collect();
    for (index, &triangle) in triangles.iter().enumerate() {
        let patch = grouped[root(&mut parent, index)].get_or_insert_with(|| Patch {
            triangles: 0,
            corners: Vec::new(),
        });
        patch.triangles += 1;
        for corner in triangle {
            if !patch
                .corners
                .iter()
                .any(|known| corner_matches(known, corner))
            {
                patch.corners.push(corner);
            }
        }
    }
    grouped.into_iter().flatten().collect()
}

fn patch_matches(expected: &Patch<'_>, actual: &Patch<'_>) -> bool {
    expected.triangles == actual.triangles
        && expected.corners.len() == actual.corners.len()
        && expected.corners.iter().all(|corner| {
            actual
                .corners
                .iter()
                .any(|other| corner_matches(corner, other))
        })
        && actual.corners.iter().all(|corner| {
            expected
                .corners
                .iter()
                .any(|other| corner_matches(other, corner))
        })
}

/// One-to-one matching of patches: every expected patch takes an unused actual patch with the same
/// triangle count and corner set.
fn patches_match(expected: &Primitive, actual: &Primitive) -> bool {
    let expected = patches(expected);
    let actual = patches(actual);
    if expected.len() != actual.len() {
        return false;
    }
    let mut used = vec![false; actual.len()];
    expected.iter().all(|patch| {
        let found = actual
            .iter()
            .enumerate()
            .position(|(index, candidate)| !used[index] && patch_matches(patch, candidate));
        match found {
            Some(index) => {
                used[index] = true;
                true
            }
            None => false,
        }
    })
}

fn primitive_matches(expected: &Primitive, actual: &Primitive) -> bool {
    // Potter shares one attribute layout per instance mesh, so a primitive may carry more TEXCOORD
    // sets than its Blender member; the extra sets are not compared.
    expected.material == actual.material
        && expected.corners.len() == actual.corners.len()
        && actual.texcoord_sets >= expected.texcoord_sets
        && patches_match(expected, actual)
}

fn summarize(primitives: &[Primitive]) -> Vec<(Option<String>, usize)> {
    primitives
        .iter()
        .map(|primitive| (primitive.material.clone(), primitive.corners.len()))
        .collect()
}

fn fmt_values(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.5}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn describe_corner(corner: &Corner) -> String {
    let uvs: Vec<String> = corner
        .uvs
        .iter()
        .map(|uv| format!("[{}]", fmt_values(uv)))
        .collect();
    format!(
        "pos [{}] normal [{}] uv {}",
        fmt_values(&corner.position),
        fmt_values(&corner.normal),
        uvs.join(" ")
    )
}

fn max_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Largest per-component difference of a corner's position, normal, and UV sets.
fn corner_deltas(expected: &Corner, actual: &Corner) -> (f64, f64, f64) {
    let uv = expected
        .uvs
        .iter()
        .zip(&actual.uvs)
        .map(|(e, a)| max_delta(e, a))
        .fold(0.0, f64::max);
    (
        max_delta(&expected.position, &actual.position),
        max_delta(&expected.normal, &actual.normal),
        uv,
    )
}

/// Describes the first unmatched expected corners of `expected` against the first exported
/// primitive with the same material and corner count: the first corners of both, and for each
/// unmatched corner its nearest candidate corner with per-field deltas.
fn mismatch_report(expected: &Primitive, actual: &[Primitive]) -> String {
    let Some(candidate) = actual.iter().find(|candidate| {
        candidate.material == expected.material && candidate.corners.len() == expected.corners.len()
    }) else {
        return "no exported primitive has the same material and corner count".to_owned();
    };
    let mut lines = vec!["first expected corners:".to_owned()];
    lines.extend(
        expected
            .corners
            .iter()
            .take(3)
            .map(|corner| format!("  {}", describe_corner(corner))),
    );
    lines.push("first candidate corners:".to_owned());
    lines.extend(
        candidate
            .corners
            .iter()
            .take(3)
            .map(|corner| format!("  {}", describe_corner(corner))),
    );
    lines.push(
        "unmatched expected corners, nearest candidate (max delta pos/normal/uv):".to_owned(),
    );
    for corner in expected
        .corners
        .iter()
        .filter(|corner| {
            !candidate
                .corners
                .iter()
                .any(|other| corner_matches(corner, other))
        })
        .take(3)
    {
        let nearest = candidate.corners.iter().min_by(|a, b| {
            let (ap, an, au) = corner_deltas(corner, a);
            let (bp, bn, bu) = corner_deltas(corner, b);
            ap.max(an).max(au).total_cmp(&bp.max(bn).max(bu))
        });
        if let Some(nearest) = nearest {
            let (position, normal, uv) = corner_deltas(corner, nearest);
            lines.push(format!(
                "  expected {}\n    nearest {}\n    delta pos {position:.6} normal {normal:.6} uv {uv:.6}",
                describe_corner(corner),
                describe_corner(nearest)
            ));
        }
    }
    lines.join("\n")
}

fn assert_primitives_match(expected: &[Primitive], actual: &[Primitive], case: &str) -> TestResult {
    ensure(
        expected.len() == actual.len(),
        format!(
            "{case}: primitive count differs: expected {:?}, got {:?}",
            summarize(expected),
            summarize(actual)
        ),
    )?;
    let mut used = vec![false; actual.len()];
    for primitive in expected {
        let found = actual
            .iter()
            .enumerate()
            .position(|(index, candidate)| !used[index] && primitive_matches(primitive, candidate));
        match found {
            Some(index) => used[index] = true,
            None => {
                return Err(format!(
                    "{case}: no exported primitive matches material {:?} with {} corners\n{}\nexpected primitives {:?}\ngot primitives {:?}",
                    primitive.material,
                    primitive.corners.len(),
                    mismatch_report(primitive, actual),
                    summarize(expected),
                    summarize(actual)
                )
                .into());
            }
        }
    }
    Ok(())
}

fn names(value: &Value) -> BTreeSet<String> {
    value
        .as_object()
        .into_iter()
        .flatten()
        .map(|(name, _)| name.clone())
        .collect()
}

fn records(value: &Value) -> TestResult<Vec<Vec<f64>>> {
    let mut result = Vec::new();
    for record in value.as_array().ok_or("dump records must be an array")? {
        let mut numbers = Vec::new();
        for number in record.as_array().ok_or("dump record must be an array")? {
            numbers.push(number.as_f64().ok_or("dump value must be a number")?);
        }
        result.push(numbers);
    }
    Ok(result)
}

/// Order-insensitive, one-to-one comparison of position-keyed records.
fn same_records(expected: &[Vec<f64>], actual: &[Vec<f64>], tolerance: f64) -> bool {
    if expected.len() != actual.len() {
        return false;
    }
    let mut used = vec![false; actual.len()];
    expected.iter().all(|record| {
        let found = actual
            .iter()
            .enumerate()
            .position(|(index, candidate)| !used[index] && close(record, candidate, tolerance));
        match found {
            Some(index) => {
                used[index] = true;
                true
            }
            None => false,
        }
    })
}

/// Compares Potter's `.blend` round trip against Blender's own reading of the source `.blend`,
/// object by object: color attributes (domain and corner values), vertex group weights, and seams.
fn assert_attribute_records_match(expected: &Value, actual: &Value, case: &str) -> TestResult {
    let objects = expected["objects"]
        .as_object()
        .ok_or("oracle dump has no objects")?;
    for (object, expected_object) in objects {
        let actual_object = &actual["objects"][object];
        ensure(
            !actual_object.is_null(),
            format!("{case}: Potter's .blend export has no object {object}"),
        )?;
        let colors = names(&expected_object["colors"]);
        ensure(
            colors == names(&actual_object["colors"]),
            format!("{case}: {object} color attribute names differ"),
        )?;
        for name in &colors {
            let expected_color = &expected_object["colors"][name];
            let actual_color = &actual_object["colors"][name];
            ensure(
                expected_color["domain"] == actual_color["domain"],
                format!("{case}: {object} color {name} domain differs"),
            )?;
            ensure(
                same_records(
                    &records(&expected_color["records"])?,
                    &records(&actual_color["records"])?,
                    COLOR_TOLERANCE,
                ),
                format!("{case}: {object} color {name} values differ"),
            )?;
        }
        let groups = names(&expected_object["vertex_groups"]);
        ensure(
            groups == names(&actual_object["vertex_groups"]),
            format!("{case}: {object} vertex group names differ"),
        )?;
        for group in &groups {
            ensure(
                same_records(
                    &records(&expected_object["vertex_groups"][group])?,
                    &records(&actual_object["vertex_groups"][group])?,
                    WEIGHT_TOLERANCE,
                ),
                format!("{case}: {object} vertex group {group} weights differ"),
            )?;
        }
        ensure(
            same_records(
                &records(&expected_object["seams"])?,
                &records(&actual_object["seams"])?,
                SEAM_TOLERANCE,
            ),
            format!("{case}: {object} seams differ"),
        )?;
    }
    Ok(())
}

fn read_json(path: &Path) -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

/// The `uv.unwrap`ped member that the Potter side adds: a box with a Potter-authored material.
/// `collection` places it in the imported member collection; without one it sits in the scene root.
fn unwrapped_member_operations(base_revision: u64, collection: Option<&str>) -> Value {
    let mut create = json!({
        "op": "node.create",
        "id": "unwrapped",
        "kind": "box",
        "params": { "size": 1.0 },
        "transform": { "translation": [3.0, 0.0, 0.0] },
        "material": "potter_mat"
    });
    if let Some(collection) = collection {
        create["collection"] = json!(collection);
    }
    json!({
        "schema_version": 1,
        "base_revision": base_revision,
        "operations": [
            { "op": "material.create", "id": "potter_mat", "base_color": [0.1, 0.7, 0.2, 1.0] },
            create,
            { "op": "uv.unwrap", "target": { "id": "unwrapped" }, "method": "smart" }
        ]
    })
}

/// The Potter member exported on its own, at the same world placement as in the instance.
fn standalone_unwrap_primitives(case_root: &Path) -> TestResult<Vec<Primitive>> {
    let project = case_root.join("standalone");
    let project_text = path_str(&project)?;
    let operations = case_root.join("standalone-operations.json");
    fs::write(
        &operations,
        serde_json::to_vec(&unwrapped_member_operations(0, None))?,
    )?;
    let glb = case_root.join("standalone.glb");
    run_pot_json(&["init", project_text, "--json"])?;
    run_pot_json(&[
        "apply",
        project_text,
        "--file",
        path_str(&operations)?,
        "--json",
    ])?;
    run_pot_json(&[
        "export",
        project_text,
        "--format",
        "glb",
        "--out",
        path_str(&glb)?,
        "--overwrite",
        "--json",
    ])?;
    primitives(&glb)
}

fn inspected_item<'a>(items: &'a [Value], id: &str) -> TestResult<&'a Value> {
    items
        .iter()
        .find(|item| item["id"] == id)
        .ok_or_else(|| format!("inspect has no item {id}").into())
}

fn run_case(
    blender: &Path,
    builder: &Path,
    dumper: &Path,
    case_root: &Path,
    case: &str,
    check: Check,
) -> TestResult {
    run_blender(blender, builder, &[path_str(case_root)?, case])?;
    let oracle = case_root.join(format!("{case}.blend"));
    let oracle_text = path_str(&oracle)?;
    let geometry = case_root.join(format!("{case}-geometry.blend"));
    let geometry_text = path_str(&geometry)?;
    let blender_glb = case_root.join(format!("{case}-geometry.blender.glb"));
    let project = case_root.join("project");
    let project_text = path_str(&project)?;
    let blender_text = path_str(blender)?;
    let potter_glb = case_root.join("potter.glb");
    let potter_glb_text = path_str(&potter_glb)?;

    run_pot_json(&["init", project_text, "--json"])?;
    let imported = run_pot_json(&[
        "import",
        project_text,
        "--file",
        geometry_text,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender_text,
        "--json",
    ])?;
    let revision = imported["scene"]["revision"]
        .as_u64()
        .ok_or("import reports no scene revision")?;

    match check {
        Check::TexturedWithoutUv => {
            let refusal = run_pot_refusal(&[
                "export",
                project_text,
                "--format",
                "glb",
                "--out",
                potter_glb_text,
                "--overwrite",
                "--json",
            ])?;
            ensure(
                refusal["error"]["code"] == "UNSUPPORTED_FEATURE"
                    && refusal["error"]["details"]["feature_id"] == "material.texture_uv_map",
                format!("a textured member without UVs must be refused: {refusal}"),
            )?;
            ensure(
                !potter_glb.exists(),
                "a refused export must not write a GLB",
            )?;
            let expected = primitives(&blender_glb)?;
            ensure(
                expected.iter().any(|primitive| {
                    primitive.material.as_deref() == Some("textured")
                        && primitive.texcoord_sets == 0
                }),
                "Blender must export the textured member without TEXCOORD_0",
            )?;
            return Ok(());
        }
        Check::ConflictIsolated => {
            let refusal = run_pot_refusal(&[
                "export",
                project_text,
                "--format",
                "glb",
                "--out",
                potter_glb_text,
                "--overwrite",
                "--json",
            ])?;
            ensure(
                refusal["error"]["details"]["node_id"] == "clash_instance",
                format!("the export must fail on the conflicting instance only: {refusal}"),
            )?;
            let inspected = run_pot_json(&["inspect", project_text, "--json"])?;
            let items = inspected["result"]["items"]
                .as_array()
                .ok_or("inspect reports no items")?;
            ensure(
                !inspected_item(items, "clash_instance")?["evaluation_error"].is_null(),
                "the conflicting instance must report an evaluation error",
            )?;
            ensure(
                inspected_item(items, "clean_instance")?["evaluation_error"].is_null(),
                "the clean instance must still evaluate",
            )?;
            return Ok(());
        }
        Check::MixedWithPotterUnwrap => {
            let operations = case_root.join("operations.json");
            fs::write(
                &operations,
                serde_json::to_vec(&unwrapped_member_operations(
                    revision,
                    Some("collection_members"),
                ))?,
            )?;
            run_pot_json(&[
                "apply",
                project_text,
                "--file",
                path_str(&operations)?,
                "--json",
            ])?;
        }
        Check::Geometry | Check::GeometryAndAttributes | Check::ClampToLastSlot => {}
    }

    run_pot_json(&[
        "export",
        project_text,
        "--format",
        "glb",
        "--out",
        potter_glb_text,
        "--allow-lossy",
        "--overwrite",
        "--json",
    ])?;
    let mut expected = primitives(&blender_glb)?;
    if check == Check::MixedWithPotterUnwrap {
        expected.extend(standalone_unwrap_primitives(case_root)?);
    }
    let exported = primitives(&potter_glb)?;
    assert_primitives_match(&expected, &exported, case)?;
    if check == Check::ClampToLastSlot {
        // Faces 3..5 of the two-slot member carry index 5, which clamps to slot 1 (blue): 3 faces,
        // 18 corners. Slot 0 (red) keeps faces 0..2, 18 corners.
        let blue_corners: usize = exported
            .iter()
            .filter(|primitive| primitive.material.as_deref() == Some("blue"))
            .map(|primitive| primitive.corners.len())
            .sum();
        ensure(
            blue_corners == 18,
            format!(
                "{case}: clamped faces must export on blue with 18 corners, got {blue_corners}"
            ),
        )?;
    }

    if check == Check::GeometryAndAttributes {
        // The attribute round trip keeps the full scene. The glTF comparison above used the
        // geometry-only variant, because glTF export refuses `blender_attributes`.
        let attribute_project = case_root.join("attribute-project");
        let attribute_project_text = path_str(&attribute_project)?;
        let potter_blend = case_root.join("potter.blend");
        let potter_blend_text = path_str(&potter_blend)?;
        run_pot_json(&["init", attribute_project_text, "--json"])?;
        run_pot_json(&[
            "import",
            attribute_project_text,
            "--file",
            oracle_text,
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
            blender_text,
            "--json",
        ])?;
        run_pot_json(&[
            "export",
            attribute_project_text,
            "--format",
            "blend",
            "--out",
            potter_blend_text,
            "--allow-lossy",
            "--overwrite",
            "--blender",
            blender_text,
            "--json",
        ])?;
        let expected_dump = case_root.join("oracle.json");
        let actual_dump = case_root.join("potter.json");
        run_blender(blender, dumper, &[oracle_text, path_str(&expected_dump)?])?;
        run_blender(
            blender,
            dumper,
            &[potter_blend_text, path_str(&actual_dump)?],
        )?;
        assert_attribute_records_match(
            &read_json(&expected_dump)?,
            &read_json(&actual_dump)?,
            case,
        )?;
    }
    Ok(())
}

#[test]
fn collection_instance_cases_match_blenders_exporter() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping collection instance parity: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let builder = root.join("blend_collection_instances.py");
    fs::write(&builder, BUILDER)?;
    let dumper = root.join("blend_collection_instances_dump.py");
    fs::write(&dumper, DUMPER)?;

    let mut exercised = Vec::new();
    let mut failures = Vec::new();
    for (case, check) in CASES {
        let case_root = root.join(case);
        fs::create_dir_all(&case_root)?;
        match run_case(&blender, &builder, &dumper, &case_root, case, check) {
            Ok(()) => exercised.push(case),
            Err(error) => failures.push(format!("{case}: {error}")),
        }
    }
    ensure(
        failures.is_empty(),
        format!("collection instance cases failed:\n{}", failures.join("\n")),
    )?;
    ensure(
        exercised.len() == CASES.len(),
        "every collection instance case must run when Blender is present",
    )?;
    Ok(())
}

#[test]
fn public_cli_nested_instances_preview_render_and_export_member_materials() -> TestResult {
    let directory = tempdir()?;
    let root = directory.path();
    let project = root.join("project");
    let project_text = path_str(&project)?;
    run_pot_json(&["init", project_text, "--json"])?;

    let operations = root.join("operations.json");
    fs::write(
        &operations,
        serde_json::to_vec(&json!({
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
    let operations_text = path_str(&operations)?;
    let apply = run_pot_json(&["apply", project_text, "--file", operations_text, "--json"])?;
    assert_eq!(apply["ok"], true);

    let preview_dir = root.join("preview");
    let preview_text = path_str(&preview_dir)?;
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
    let render_text = path_str(&render_dir)?;
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
    let glb_text = path_str(&glb)?;
    run_pot_json(&[
        "export",
        project_text,
        "--format",
        "glb",
        "--out",
        glb_text,
        "--json",
    ])?;
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
