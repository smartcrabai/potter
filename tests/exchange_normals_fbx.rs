//! FBX shading normals: Blender's default layer layout, import mapping, and round trips
//! against Blender's reference corner normals.

#[path = "common/exchange_normals.rs"]
mod exchange_normals;

use std::{
    fs,
    path::{Path, PathBuf},
    process::Output,
};

use exchange_normals::{
    ObservedCorner, TestResult, assert_matches_reference, export, export_strict, init,
    normal_tolerance, obj_to_potter, observed_corners, pot, usd_world_corners, write_fixture_scene,
};
use serde_json::{Value, json};
use tempfile::{TempDir, tempdir};

/// `LayerElementNormal` as Blender writes it: one value per polygon vertex, via an index.
const POLYGON_VERTEX_NORMALS: &str = r#"    LayerElementNormal: 0 {
      Version: 101
      Name: ""
      MappingInformationType: "ByPolygonVertex"
      ReferenceInformationType: "IndexToDirect"
      Normals: *9 {
        a: 0,0,1,0.6,0,0.8,0,0.6,0.8
      }
      NormalsIndex: *3 {
        a: 0,1,2
      }
    }
"#;

/// `LayerElementNormal` per control point, stored directly.
const VERTEX_NORMALS: &str = r#"    LayerElementNormal: 0 {
      Version: 101
      Name: ""
      MappingInformationType: "ByVertice"
      ReferenceInformationType: "Direct"
      Normals: *9 {
        a: 0,0,1,0.6,0,0.8,0,0.6,0.8
      }
    }
"#;

/// Same normals as `POLYGON_VERTEX_NORMALS`, but the middle normal is zero-length.
const ZERO_LENGTH_NORMAL: &str = r#"    LayerElementNormal: 0 {
      Version: 101
      Name: ""
      MappingInformationType: "ByPolygonVertex"
      ReferenceInformationType: "IndexToDirect"
      Normals: *9 {
        a: 0,0,1,0,0,0,0,0.6,0.8
      }
      NormalsIndex: *3 {
        a: 0,1,2
      }
    }
"#;

/// A mapping this importer does not represent (`ByPolygon`).
const BY_POLYGON_NORMALS: &str = r#"    LayerElementNormal: 0 {
      Version: 101
      Name: ""
      MappingInformationType: "ByPolygon"
      ReferenceInformationType: "Direct"
      Normals: *3 {
        a: 0,0,1
      }
    }
"#;

/// Blender's default export writes no smoothing layer; this one exists only in older files.
const SMOOTHING: &str = r#"    LayerElementSmoothing: 0 {
      Version: 102
      Name: ""
      MappingInformationType: "ByPolygon"
      ReferenceInformationType: "Direct"
      Smoothing: *1 {
        a: 1
      }
    }
"#;

const TRIANGLE_POSITIONS: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
const TRIANGLE_NORMALS: [[f64; 3]; 3] = [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [0.0, 0.6, 0.8]];

fn import_file(scene: &Path, file: &Path, format: &str, allow_lossy: bool) -> TestResult<Output> {
    let mut command = pot();
    command
        .arg("import")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .args([
            "--format",
            format,
            "--base-revision",
            "0",
            "--mode",
            "replace",
        ]);
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    Ok(command.arg("--json").output()?)
}

/// Feature IDs of the losses reported by a successful import.
fn loss_ids(output: &Output) -> TestResult<Vec<String>> {
    let json: Value = serde_json::from_slice(&output.stdout)?;
    Ok(json["result"]["losses"]
        .as_array()
        .ok_or("import result has no losses array")?
        .iter()
        .filter_map(|loss| loss["feature_id"].as_str().map(str::to_owned))
        .collect())
}

fn triple<'a>(parts: impl Iterator<Item = &'a str>) -> TestResult<[f64; 3]> {
    let values = parts
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()?;
    match values.as_slice() {
        [x, y, z] => Ok([*x, *y, *z]),
        _ => Err("OBJ vertex needs three coordinates".into()),
    }
}

/// Corners of an OBJ export, converted from Blender's OBJ axes to Potter space. The export bakes
/// world transforms and writes `f v//vn` per corner.
fn obj_corners(text: &str) -> TestResult<Vec<ObservedCorner>> {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut faces = Vec::new();
    let mut corner_normals = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => positions.push(obj_to_potter(triple(parts)?)),
            Some("vn") => normals.push(obj_to_potter(triple(parts)?)),
            Some("f") => {
                let mut face = Vec::new();
                for token in parts {
                    let fields = token.split('/').collect::<Vec<_>>();
                    let vertex = fields
                        .first()
                        .ok_or("OBJ corner has no vertex")?
                        .parse::<usize>()?;
                    let normal = fields
                        .get(2)
                        .ok_or("OBJ corner has no normal")?
                        .parse::<usize>()?;
                    face.push(vertex.checked_sub(1).ok_or("OBJ vertex index is zero")?);
                    let normal = normal
                        .checked_sub(1)
                        .and_then(|index| normals.get(index))
                        .ok_or("OBJ normal index is out of range")?;
                    corner_normals.push(*normal);
                }
                faces.push(face);
            }
            _ => {}
        }
    }
    Ok(observed_corners(&positions, &faces, &corner_normals))
}

/// Values of the first `a:` array after `key` in an ASCII FBX layer block.
fn list_after<'a>(block: &'a str, key: &str) -> TestResult<Vec<&'a str>> {
    let start = block.find(key).ok_or("FBX layer block is missing a key")?;
    let rest = &block[start..];
    let values = rest.find("a: ").ok_or("FBX array has no values")?;
    let line = rest[values + 3..]
        .lines()
        .next()
        .ok_or("FBX array has no values line")?;
    Ok(line.split(',').map(str::trim).collect())
}

struct Imported {
    directory: TempDir,
    scene: PathBuf,
    output: Output,
}

/// Imports a one-triangle ASCII FBX whose geometry carries the given layer blocks.
fn import_minimal(layers: &str, allow_lossy: bool) -> TestResult<Imported> {
    let directory = tempdir()?;
    let file = directory.path().join("triangle.fbx");
    let text = format!(
        r#"; FBX 7.4.0 project file
Objects:  {{
  Geometry: 1, "Geometry::Triangle", "Mesh" {{
    Vertices: *9 {{
      a: 0,0,0,1,0,0,0,1,0
    }}
    PolygonVertexIndex: *3 {{
      a: 0,1,-3
    }}
{layers}  }}
  Model: 2, "Model::Triangle", "Mesh" {{
    Version: 232
  }}
}}
Connections:  {{
  C: "OO",1,2
}}
"#
    );
    fs::write(&file, text)?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let output = import_file(&scene, &file, "fbx", allow_lossy)?;
    Ok(Imported {
        directory,
        scene,
        output,
    })
}

/// Exports the imported triangle as OBJ and checks each corner's position and normal.
fn assert_triangle_corners(imported: &Imported) -> TestResult {
    assert!(
        imported.output.status.success(),
        "{}",
        String::from_utf8_lossy(&imported.output.stdout)
    );
    let obj = export(
        &imported.scene,
        "obj",
        &imported.directory.path().join("triangle.obj"),
    )?;
    let corners = obj_corners(&fs::read_to_string(obj)?)?;
    assert_eq!(corners.len(), 3, "triangle corner count");
    for ((corner, position), normal) in corners.iter().zip(TRIANGLE_POSITIONS).zip(TRIANGLE_NORMALS)
    {
        for axis in 0..3 {
            assert!(
                (corner.corner[axis] - position[axis]).abs() <= 1.0e-6,
                "corner position {:?} differs from {position:?}",
                corner.corner
            );
            assert!(
                (corner.normal[axis] - normal[axis]).abs() <= normal_tolerance(true),
                "corner normal {:?} differs from {normal:?}",
                corner.normal
            );
        }
    }
    Ok(())
}

#[test]
fn fbx_export_writes_blender_default_normal_layer() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    let path = export(&scene, "fbx", &directory.path().join("fixture.fbx"))?;
    let text = fs::read_to_string(path)?;
    assert!(
        !text.contains("LayerElementSmoothing"),
        "Blender's default FBX export writes no smoothing layer"
    );
    let blocks = text
        .split("LayerElementNormal: 0 {")
        .skip(1)
        .collect::<Vec<_>>();
    assert_eq!(blocks.len(), 3, "one normal layer per fixture object");
    for block in blocks {
        assert!(block.contains("MappingInformationType: \"ByPolygonVertex\""));
        assert!(block.contains("ReferenceInformationType: \"IndexToDirect\""));
        let values = list_after(block, "Normals: *")?;
        let indices = list_after(block, "NormalsIndex: *")?;
        assert_eq!(values.len() % 3, 0, "Normals must hold xyz triples");
        assert_eq!(indices.len(), 24, "one index per polygon vertex");
        let unique = values.len() / 3;
        for index in indices {
            let index = index.parse::<usize>()?;
            assert!(index < unique, "normal index {index} is out of range");
        }
    }
    Ok(())
}

#[test]
fn fbx_fixture_export_reports_no_losses() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    let output = export_strict(&scene, "fbx", &directory.path().join("fixture.fbx"))?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let json: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(json["result"]["losses"], json!([]));
    Ok(())
}

fn round_trip_matches_reference(format: &str, extension: &str) -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    let file = export(
        &scene,
        format,
        &directory.path().join(format!("fixture.{extension}")),
    )?;
    let imported = directory.path().join("imported");
    init(&imported)?;
    let output = import_file(&imported, &file, format, false)?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        loss_ids(&output)?.is_empty(),
        "{format} import reported losses"
    );
    let usda = export(&imported, "usda", &directory.path().join("imported.usda"))?;
    assert_matches_reference(&usd_world_corners(&fs::read_to_string(usda)?)?)?;
    Ok(())
}

#[test]
fn fbx_round_trip_matches_blender_reference() -> TestResult {
    round_trip_matches_reference("fbx", "fbx")
}

#[test]
fn fbx_binary_round_trip_matches_blender_reference() -> TestResult {
    round_trip_matches_reference("fbx-binary", "fbx")
}

#[test]
fn polygon_vertex_normals_import_as_corner_normals() -> TestResult {
    let imported = import_minimal(POLYGON_VERTEX_NORMALS, false)?;
    assert_triangle_corners(&imported)
}

#[test]
fn vertex_normals_import_per_control_point() -> TestResult {
    let imported = import_minimal(VERTEX_NORMALS, false)?;
    assert_triangle_corners(&imported)
}

#[test]
fn normal_layer_overrides_smoothing_without_loss() -> TestResult {
    let layers = format!("{POLYGON_VERTEX_NORMALS}{SMOOTHING}");
    let imported = import_minimal(&layers, false)?;
    assert_eq!(loss_ids(&imported.output)?, Vec::<String>::new());
    assert_triangle_corners(&imported)
}

#[test]
fn smoothing_without_normals_is_a_loss() -> TestResult {
    let blocked = import_minimal(SMOOTHING, false)?;
    assert!(
        !blocked.output.status.success(),
        "import must need --allow-lossy"
    );
    assert!(
        String::from_utf8_lossy(&blocked.output.stdout).contains("format.fbx.smoothing"),
        "lossy refusal must name format.fbx.smoothing"
    );
    let allowed = import_minimal(SMOOTHING, true)?;
    assert!(allowed.output.status.success());
    assert_eq!(loss_ids(&allowed.output)?, ["format.fbx.smoothing"]);
    Ok(())
}

#[test]
fn unsupported_normal_mapping_is_a_mesh_normals_loss() -> TestResult {
    let blocked = import_minimal(BY_POLYGON_NORMALS, false)?;
    assert!(
        !blocked.output.status.success(),
        "import must need --allow-lossy"
    );
    let allowed = import_minimal(BY_POLYGON_NORMALS, true)?;
    assert!(allowed.output.status.success());
    assert_eq!(loss_ids(&allowed.output)?, ["mesh.normals"]);
    Ok(())
}

#[test]
fn zero_length_normal_imports_as_zero_without_loss() -> TestResult {
    let imported = import_minimal(ZERO_LENGTH_NORMAL, false)?;
    assert!(imported.output.status.success());
    assert_eq!(loss_ids(&imported.output)?, Vec::<String>::new());
    Ok(())
}
