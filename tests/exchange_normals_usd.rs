//! USD shading normals: Blender-shaped `faceVarying` export, usda/usdc/usdz round trips, and the
//! `interpolation` modes Potter imports from hand-written layers.

#[path = "common/exchange_normals.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared fixture helpers"
)]
mod exchange_normals;

use std::{fs, path::Path, process::Output};

use exchange_normals::{
    ObservedCorner, TestResult, assert_matches_reference, export, init, normal_tolerance, pot,
    usd_world_corners, write_fixture_scene,
};
use serde_json::{Value, json};
use tempfile::tempdir;

fn run_import(project: &Path, file: &Path, format: &str, allow_lossy: bool) -> TestResult<Output> {
    let mut command = pot();
    command
        .arg("import")
        .arg(project)
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

/// Imports `file` into `project`, asserting success, and returns the JSON report.
fn import(project: &Path, file: &Path, format: &str, allow_lossy: bool) -> TestResult<Value> {
    let output = run_import(project, file, format, allow_lossy)?;
    assert!(
        output.status.success(),
        "pot import {format} failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn unit(value: [f64; 3]) -> [f64; 3] {
    let length = value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    value.map(|component| component / length)
}

fn assert_normal(observed: [f64; 3], expected: [f64; 3]) {
    for axis in 0..3 {
        assert!(
            (observed[axis] - expected[axis]).abs() <= normal_tolerance(true),
            "normal {observed:?} differs from expected {expected:?}"
        );
    }
}

/// Cube with Potter's box topology at (±1, ±1, ±1), with `normals` as its normals attribute.
fn cube_usda(normals: &str) -> String {
    format!(
        r#"#usda 1.0
(
    defaultPrim = "Cube"
    metersPerUnit = 1
    upAxis = "Z"
)

def Mesh "Cube"
{{
    point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
    int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    uniform token subdivisionScheme = "none"
    {normals}
}}
"#
    )
}

/// Flat normals of the cube's six faces, in `cube_usda` face order.
const CUBE_FLAT_NORMALS: [[f64; 3]; 6] = [
    [0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
];

/// Imports a hand-written cube with no losses and returns its re-exported corner normals.
fn reimport_cube(normals: &str) -> TestResult<Vec<ObservedCorner>> {
    let directory = tempdir()?;
    let source = directory.path().join("cube.usda");
    fs::write(&source, cube_usda(normals))?;
    let project = directory.path().join("project");
    init(&project)?;
    let imported = import(&project, &source, "usda", false)?;
    assert_eq!(imported["result"]["losses"], json!([]));
    let exported = export(&project, "usda", &directory.path().join("out.usda"))?;
    usd_world_corners(&fs::read_to_string(exported)?)
}

#[test]
fn usda_export_writes_face_varying_blender_normals() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    let file = export(&scene, "usda", &directory.path().join("fixture.usda"))?;
    let text = fs::read_to_string(file)?;
    assert!(text.starts_with("#usda 1.0"));
    assert_eq!(
        text.matches("def Mesh").count(),
        3,
        "Mixed, Mirror, and NonUni"
    );
    assert_eq!(
        text.matches("uniform token subdivisionScheme = \"none\"")
            .count(),
        3
    );
    assert_eq!(text.matches("normal3f[] normals = [").count(), 3);
    assert_eq!(text.matches("(interpolation = \"faceVarying\")").count(), 3);
    assert_matches_reference(&usd_world_corners(&text)?)?;
    Ok(())
}

#[test]
fn usda_round_trip_preserves_corner_normals() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    let first = export(&scene, "usda", &directory.path().join("first.usda"))?;
    let project = directory.path().join("imported");
    init(&project)?;
    import(&project, &first, "usda", false)?;
    let second = export(&project, "usda", &directory.path().join("second.usda"))?;
    assert_matches_reference(&usd_world_corners(&fs::read_to_string(&first)?)?)?;
    assert_matches_reference(&usd_world_corners(&fs::read_to_string(&second)?)?)?;
    Ok(())
}

#[test]
fn usdc_and_usdz_round_trips_preserve_corner_normals() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    for format in ["usdc", "usdz"] {
        let file = export(
            &scene,
            format,
            &directory.path().join(format!("fixture.{format}")),
        )?;
        let bytes = fs::read(&file)?;
        if format == "usdc" {
            assert!(bytes.starts_with(b"PXR-USDC"), "usdc magic is missing");
        } else {
            assert!(
                bytes.starts_with(b"PK\x03\x04"),
                "usdz zip header is missing"
            );
        }
        let project = directory.path().join(format!("imported-{format}"));
        init(&project)?;
        import(&project, &file, format, false)?;
        let text = export(
            &project,
            "usda",
            &directory.path().join(format!("{format}.usda")),
        )?;
        assert_matches_reference(&usd_world_corners(&fs::read_to_string(text)?)?)?;
    }
    Ok(())
}

#[test]
fn imported_vertex_normals_map_per_point() -> TestResult {
    // Unnormalized per-point values; each corner takes its vertex's direction.
    let corners = reimport_cube(
        r#"normal3f[] normals = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)] (interpolation = "vertex")"#,
    )?;
    assert_eq!(corners.len(), 24);
    for corner in &corners {
        assert_normal(corner.normal, unit(corner.corner));
    }
    Ok(())
}

#[test]
fn imported_uniform_normals_map_per_face() -> TestResult {
    let values = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ];
    let block = format!(
        "normal3f[] normals = [{}] (interpolation = \"uniform\")",
        values
            .iter()
            .map(|value| format!("({}, {}, {})", value[0], value[1], value[2]))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let corners = reimport_cube(&block)?;
    assert_eq!(corners.len(), 24);
    for (index, corner) in corners.iter().enumerate() {
        assert_normal(corner.normal, unit(values[index / 4]));
    }
    Ok(())
}

#[test]
fn imported_face_varying_normals_map_per_corner() -> TestResult {
    let values = (0..24)
        .map(|index| [f64::from(index + 1), 2.0, 1.0])
        .collect::<Vec<_>>();
    let block = format!(
        "normal3f[] normals = [{}] (interpolation = \"faceVarying\")",
        values
            .iter()
            .map(|value| format!("({}, {}, {})", value[0], value[1], value[2]))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let corners = reimport_cube(&block)?;
    assert_eq!(corners.len(), 24);
    for (index, corner) in corners.iter().enumerate() {
        assert_normal(corner.normal, unit(values[index]));
    }
    Ok(())
}

#[test]
fn imported_constant_normal_applies_to_every_corner() -> TestResult {
    let corners = reimport_cube("normal3f[] normals = [(0, 0, 1)] (interpolation = \"constant\")")?;
    assert_eq!(corners.len(), 24);
    for corner in &corners {
        assert_normal(corner.normal, [0.0, 0.0, 1.0]);
    }
    Ok(())
}

/// Imports a cube whose normals cannot be mapped: the strict import fails, `--allow-lossy`
/// reports a `mesh.normals` loss, and the normals are dropped to flat shading.
fn assert_normals_reported_as_loss(normals: &str) -> TestResult {
    let directory = tempdir()?;
    let source = directory.path().join("cube.usda");
    fs::write(&source, cube_usda(normals))?;
    let project = directory.path().join("project");
    init(&project)?;
    let strict = run_import(&project, &source, "usda", false)?;
    assert!(
        !strict.status.success(),
        "lossy USD normals must need --allow-lossy"
    );
    let lossy = import(&project, &source, "usda", true)?;
    let losses = lossy["result"]["losses"]
        .as_array()
        .ok_or("import report has no losses array")?;
    assert!(
        losses
            .iter()
            .any(|loss| loss["feature_id"] == "mesh.normals"),
        "missing mesh.normals loss: {losses:?}"
    );
    let exported = export(&project, "usda", &directory.path().join("out.usda"))?;
    let corners = usd_world_corners(&fs::read_to_string(exported)?)?;
    assert_eq!(corners.len(), 24);
    for (index, corner) in corners.iter().enumerate() {
        assert_normal(corner.normal, CUBE_FLAT_NORMALS[index / 4]);
    }
    Ok(())
}

#[test]
fn unknown_interpolation_is_reported_as_a_loss() -> TestResult {
    assert_normals_reported_as_loss(
        "normal3f[] normals = [(0, 0, 1)] (interpolation = \"varying2\")",
    )
}

#[test]
fn zero_normals_import_as_zero_without_loss() -> TestResult {
    let directory = tempdir()?;
    let source = directory.path().join("cube.usda");
    fs::write(
        &source,
        cube_usda("normal3f[] normals = [(0, 0, 0)] (interpolation = \"constant\")"),
    )?;
    let project = directory.path().join("project");
    init(&project)?;
    let imported = import(&project, &source, "usda", false)?;
    assert_eq!(imported["result"]["losses"], json!([]));
    let exported = export(&project, "usda", &directory.path().join("out.usda"))?;
    let text = fs::read_to_string(exported)?;
    assert!(
        text.contains("normal3f[] normals = [(0, 0, 0)"),
        "zero normals must round-trip"
    );
    Ok(())
}

#[test]
fn plain_normals_without_interpolation_are_per_point() -> TestResult {
    // An unauthored interpolation on the plain `normals` attribute means `vertex`.
    let corners = reimport_cube(
        "normal3f[] normals = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]",
    )?;
    assert_eq!(corners.len(), 24);
    for corner in &corners {
        assert_normal(corner.normal, unit(corner.corner));
    }
    Ok(())
}

#[test]
fn indexed_normals_map_through_primvars_indices() -> TestResult {
    // Six unique face normals, stored once and addressed in reverse face order.
    let values = CUBE_FLAT_NORMALS
        .iter()
        .map(|normal| format!("({}, {}, {})", normal[0], normal[1], normal[2]))
        .collect::<Vec<_>>()
        .join(", ");
    let corners = reimport_cube(&format!(
        "normal3f[] primvars:normals = [{values}] (interpolation = \"uniform\")\n    int[] primvars:normals:indices = [5, 4, 3, 2, 1, 0]"
    ))?;
    assert_eq!(corners.len(), 24);
    for (index, corner) in corners.iter().enumerate() {
        assert_normal(corner.normal, CUBE_FLAT_NORMALS[5 - index / 4]);
    }
    Ok(())
}
