//! Shared fixture for exchange shading-normal tests.
//!
//! The scene has three boxes with identical topology and shading: `Mixed` at the origin,
//! `Mirror` with a negative X scale, and `NonUni` with a non-uniform scale. Shading is
//! `f0` flat, `f1`..`f5` smooth, edge `e0` sharp, and face `f2` with custom normals. Blender
//! 5.2.2 computed the reference corner normals for exactly this scene
//! (`tests/fixtures/exchange_normals/reference_corners.json`, one entry per face corner in
//! world space).

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde::Deserialize;
use serde_json::{Value, json};

pub type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Tolerance for Potter-side corner normals against Blender's reference numbers. Blender stores
/// custom normals as 16-bit components (steps of 1/32767). The fixture's non-uniform object
/// (scale z 0.5) amplifies that quantization in world space: measured worst error is 8.6e-5
/// over all 72 corners, so the bound is 1e-4. A wrong axis or winding produces errors near 1.
pub const NORMAL_TOLERANCE: f64 = 1.0e-4;

/// Bound for values that Blender itself re-quantized: PLY per-vertex sums are built from Blender's
/// int16-quantized corner normals, so they can differ from exact sums by more than
/// `NORMAL_TOLERANCE`.
pub const BLENDER_NORMAL_TOLERANCE: f64 = 2.0e-4;

/// Inverse of Potter's OBJ axes: a file line `(x, z, -y)` is Potter's `(x, y, z)`.
pub fn obj_to_potter(value: [f64; 3]) -> [f64; 3] {
    [value[0], -value[2], value[1]]
}

/// Parses every mesh in a usda document into world-space corners. Each `Mesh` takes the
/// `xformOp:translate` and `xformOp:scale` declared before it in its enclosing `Xform` (Potter
/// writes no rotation for these fixtures). Local normals map to world by `n / scale`, normalized.
pub fn usd_world_corners(text: &str) -> TestResult<Vec<ObservedCorner>> {
    let mut corners = Vec::new();
    let mut translate = [0.0_f64; 3];
    let mut scale = [1.0_f64; 3];
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed.starts_with("def Xform") {
            translate = [0.0; 3];
            scale = [1.0; 3];
        } else if let Some(value) = trimmed.strip_prefix("double3 xformOp:translate = ") {
            translate = triple_of(value)?;
        } else if let Some(value) = trimmed.strip_prefix("float3 xformOp:scale = ") {
            scale = triple_of(value)?;
        } else if let Some(value) = trimmed.strip_prefix("double3 xformOp:scale = ") {
            scale = triple_of(value)?;
        } else if trimmed.starts_with("def Mesh") {
            // Gather the mesh body until its matching closing brace.
            let mut body = String::new();
            let mut depth = 0_i32;
            let mut opened = false;
            for inner in lines.by_ref() {
                depth += inner.matches('{').count() as i32;
                depth -= inner.matches('}').count() as i32;
                opened |= inner.contains('{');
                body.push_str(inner);
                body.push('\n');
                if opened && depth == 0 {
                    break;
                }
            }
            corners.extend(mesh_world_corners(&body, translate, scale)?);
        }
    }
    Ok(corners)
}

fn mesh_world_corners(
    body: &str,
    translate: [f64; 3],
    scale: [f64; 3],
) -> TestResult<Vec<ObservedCorner>> {
    let points = numbers_in(bracket_after(body, "point3f[] points")?)
        .chunks(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();
    let counts = numbers_in(bracket_after(body, "int[] faceVertexCounts")?)
        .into_iter()
        .map(|value| value as usize)
        .collect::<Vec<_>>();
    let indices = numbers_in(bracket_after(body, "int[] faceVertexIndices")?)
        .into_iter()
        .map(|value| value as usize)
        .collect::<Vec<_>>();
    let normals = numbers_in(bracket_after(body, "normal3f[] normals")?)
        .chunks(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();
    let world_point =
        |point: [f64; 3]| std::array::from_fn(|axis| translate[axis] + scale[axis] * point[axis]);
    let world_normal = |normal: [f64; 3]| {
        let scaled: [f64; 3] = std::array::from_fn(|axis| normal[axis] / scale[axis]);
        let length = scaled.iter().map(|value| value * value).sum::<f64>().sqrt();
        scaled.map(|value| value / length)
    };
    let mut corners = Vec::new();
    let mut start = 0;
    let mut normal_index = 0;
    for count in counts {
        let face = &indices[start..start + count];
        let mut center = [0.0_f64; 3];
        for index in face {
            let world = world_point(points[*index]);
            for axis in 0..3 {
                center[axis] += world[axis] / count as f64;
            }
        }
        for index in face {
            corners.push(ObservedCorner {
                face_center: center,
                corner: world_point(points[*index]),
                normal: world_normal(normals[normal_index]),
            });
            normal_index += 1;
        }
        start += count;
    }
    Ok(corners)
}

/// The bracketed text after `key = [` in `body`, without the brackets.
fn bracket_after<'a>(body: &'a str, key: &str) -> TestResult<&'a str> {
    let start = body
        .find(key)
        .ok_or_else(|| format!("usda attribute `{key}` is missing"))?;
    let rest = &body[start + key.len()..];
    let open = rest.find('[').ok_or("usda array has no opening bracket")?;
    let close = rest[open..]
        .find(']')
        .ok_or("usda array has no closing bracket")?;
    Ok(&rest[open + 1..open + close])
}

fn numbers_in(text: &str) -> Vec<f64> {
    text.split(|c: char| {
        !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
    })
    .filter(|token| !token.is_empty())
    .filter_map(|token| token.parse::<f64>().ok())
    .collect()
}

fn triple_of(value: &str) -> TestResult<[f64; 3]> {
    let values = numbers_in(value);
    let values = values
        .get(..3)
        .ok_or("usda triple has fewer than three values")?;
    Ok([values[0], values[1], values[2]])
}

/// One face corner as observed in an exported or imported file, in world space.
#[derive(Clone, Debug)]
pub struct ObservedCorner {
    pub face_center: [f64; 3],
    pub corner: [f64; 3],
    pub normal: [f64; 3],
}

#[derive(Deserialize)]
struct ReferenceCorner {
    face_center: [f64; 3],
    corner: [f64; 3],
    normal: [f64; 3],
}

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

/// Creates the fixture scene at `scene` (a project directory).
pub fn write_fixture_scene(scene: &Path) -> TestResult {
    let init = pot().arg("init").arg(scene).arg("--json").output()?;
    assert!(init.status.success(), "pot init failed");
    let operations = fixture_operations();
    let batch = json!({"schema_version": 1, "base_revision": 0, "operations": operations});
    let batch_file = scene.with_extension("json");
    fs::write(&batch_file, batch.to_string())?;
    let apply = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(&batch_file)
        .output()?;
    assert!(apply.status.success(), "pot apply failed");
    Ok(())
}

fn fixture_operations() -> Value {
    let mut operations = vec![
        json!({"op": "node.create", "id": "mixed", "kind": "box", "params": {"size": 2.0}}),
        json!({
            "op": "node.create", "id": "mirror", "kind": "box", "params": {"size": 2.0},
            "transform": {"translation": [5.0, 0.0, 0.0], "scale": [-1.0, 1.0, 1.0]}
        }),
        json!({
            "op": "node.create", "id": "nonuni", "kind": "box", "params": {"size": 2.0},
            "transform": {"translation": [0.0, 5.0, 0.0], "scale": [2.0, 1.0, 0.5]}
        }),
    ];
    for object in ["mixed", "mirror", "nonuni"] {
        operations.push(json!({
            "op": "mesh.shade_smooth", "target": {"id": object},
            "elements": {"domain": "face", "ids": ["f1", "f2", "f3", "f4", "f5"]}
        }));
        operations.push(json!({
            "op": "mesh.mark_sharp_by_angle", "target": {"id": object},
            "elements": {"domain": "edge", "ids": ["e0"]}, "angle": 0.1
        }));
        operations.push(json!({
            "op": "mesh.set_custom_normals", "target": {"id": object},
            "elements": {"domain": "face", "ids": ["f2"]},
            "normals": {"f2": [[0.6, 0.6, 0.0], [0.6, 0.6, 0.0], [0.6, 0.6, 0.0], [0.6, 0.6, 0.0]]}
        }));
    }
    Value::Array(operations)
}

/// Exports the fixture scene to `out` in `format` with `--allow-lossy` and returns `out`.
/// Use `export_strict` to observe the loss gate itself.
pub fn export(scene: &Path, format: &str, out: &Path) -> TestResult<PathBuf> {
    let output = pot()
        .arg("export")
        .arg(scene)
        .args(["--format", format, "--out"])
        .arg(out)
        .args(["--overwrite", "--allow-lossy", "--json"])
        .output()?;
    assert!(
        output.status.success(),
        "pot export {format} failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(out.to_path_buf())
}

/// Runs `pot export` without `--allow-lossy` and returns the raw output, so callers can assert
/// success or the lossy-error envelope.
pub fn export_strict(scene: &Path, format: &str, out: &Path) -> TestResult<std::process::Output> {
    Ok(pot()
        .arg("export")
        .arg(scene)
        .args(["--format", format, "--out"])
        .arg(out)
        .args(["--overwrite", "--json"])
        .output()?)
}

/// Blender's corner normals for the fixture, in world space.
pub fn reference_corners() -> TestResult<Vec<ObservedCorner>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/exchange_normals/reference_corners.json");
    let rows: Vec<ReferenceCorner> = serde_json::from_slice(&fs::read(path)?)?;
    Ok(rows
        .into_iter()
        .map(|row| ObservedCorner {
            face_center: row.face_center,
            corner: row.corner,
            normal: row.normal,
        })
        .collect())
}

/// Builds observed corners from mesh data: `faces` index `positions`, and `normals` holds one
/// normal per face corner in `faces` order. Face centers are corner averages, as Blender's
/// `polygon.center` is.
pub fn observed_corners(
    positions: &[[f64; 3]],
    faces: &[Vec<usize>],
    normals: &[[f64; 3]],
) -> Vec<ObservedCorner> {
    let mut corners = Vec::with_capacity(normals.len());
    let mut normal_index = 0;
    for face in faces {
        let mut center = [0.0_f64; 3];
        for index in face {
            for axis in 0..3 {
                center[axis] += positions[*index][axis] / face.len() as f64;
            }
        }
        for index in face {
            corners.push(ObservedCorner {
                face_center: center,
                corner: positions[*index],
                normal: normals[normal_index],
            });
            normal_index += 1;
        }
    }
    corners
}

/// Asserts every reference corner has exactly one observed corner at the same world position
/// and face center, with a normal within `NORMAL_TOLERANCE`. Returns the largest component error.
pub fn assert_matches_reference(observed: &[ObservedCorner]) -> TestResult<f64> {
    let reference = reference_corners()?;
    assert_eq!(
        observed.len(),
        reference.len(),
        "corner count differs from the Blender reference"
    );
    let mut worst = 0.0_f64;
    for expected in &reference {
        let found = observed
            .iter()
            .filter(|corner| {
                near(corner.corner, expected.corner)
                    && near(corner.face_center, expected.face_center)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            found.len(),
            1,
            "expected one observed corner at {:?} in face centred {:?}",
            expected.corner,
            expected.face_center
        );
        let error = (0..3)
            .map(|axis| (found[0].normal[axis] - expected.normal[axis]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            error <= NORMAL_TOLERANCE,
            "normal {:?} differs from Blender {:?} at corner {:?} (error {error})",
            found[0].normal,
            expected.normal,
            expected.corner
        );
        worst = worst.max(error);
    }
    Ok(worst)
}

fn near(left: [f64; 3], right: [f64; 3]) -> bool {
    (0..3).all(|axis| (left[axis] - right[axis]).abs() <= 1.0e-4)
}
