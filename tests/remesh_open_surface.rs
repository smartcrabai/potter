//! Dual-contouring Remesh on open, non-manifold, mixed, and disconnected meshes.
//!
//! The unconditional tests pin counts recorded from Blender 5.2.2 (`--factory-startup`). The
//! Blender-gated matrix regenerates the reference from a local Blender executable and skips with a
//! message when none is installed.

use std::{error::Error, fs, path::Path, process::Command};

use glam::DVec3;
use potter_core::{
    error::ErrorCode,
    geom::{Mesh, modifiers::evaluate_modifiers},
    model::{Id, Modifier},
};
use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;

type Geometry = (Vec<DVec3>, Vec<Vec<usize>>);

fn cube_faces(offset: [f64; 3], size: f64, skip: &[&str]) -> Geometry {
    let half = size / 2.0;
    let [ox, oy, oz] = offset;
    let positions = vec![
        DVec3::new(ox - half, oy - half, oz - half),
        DVec3::new(ox + half, oy - half, oz - half),
        DVec3::new(ox + half, oy + half, oz - half),
        DVec3::new(ox - half, oy + half, oz - half),
        DVec3::new(ox - half, oy - half, oz + half),
        DVec3::new(ox + half, oy - half, oz + half),
        DVec3::new(ox + half, oy + half, oz + half),
        DVec3::new(ox - half, oy + half, oz + half),
    ];
    let faces: [(&str, [usize; 4]); 6] = [
        ("bottom", [0, 3, 2, 1]),
        ("top", [4, 5, 6, 7]),
        ("front", [0, 1, 5, 4]),
        ("back", [2, 3, 7, 6]),
        ("left", [0, 4, 7, 3]),
        ("right", [1, 2, 6, 5]),
    ];
    let polygons = faces
        .iter()
        .filter(|(name, _)| !skip.contains(name))
        .map(|(_, face)| face.to_vec())
        .collect();
    (positions, polygons)
}

fn cylinder(segments: usize, radius: f64, height: f64, caps: bool) -> Geometry {
    let mut positions = Vec::new();
    for z in [-height / 2.0, height / 2.0] {
        for index in 0..segments {
            let angle = 2.0 * std::f64::consts::PI * index as f64 / segments as f64;
            positions.push(DVec3::new(radius * angle.cos(), radius * angle.sin(), z));
        }
    }
    let mut polygons: Vec<Vec<usize>> = (0..segments)
        .map(|index| {
            let next = (index + 1) % segments;
            vec![index, next, segments + next, segments + index]
        })
        .collect();
    if caps {
        polygons.push((0..segments).collect());
        polygons.push((segments..2 * segments).rev().collect());
    }
    (positions, polygons)
}

fn merge(parts: Vec<Geometry>) -> Geometry {
    let mut positions = Vec::new();
    let mut polygons = Vec::new();
    for (part_positions, part_polygons) in parts {
        let base = positions.len();
        positions.extend(part_positions);
        polygons.extend(part_polygons.into_iter().map(|polygon| {
            polygon
                .into_iter()
                .map(|index| base + index)
                .collect::<Vec<_>>()
        }));
    }
    (positions, polygons)
}

fn geometry(kind: &str) -> Result<Geometry, Box<dyn Error>> {
    Ok(match kind {
        "plane" => (
            vec![
                DVec3::new(-1.0, -1.0, 0.0),
                DVec3::new(1.0, -1.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(-1.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3]],
        ),
        "grid_open" => {
            let mut positions = Vec::new();
            for y in [-1.0, 0.0, 1.0] {
                for x in [-1.0, 0.0, 1.0] {
                    positions.push(DVec3::new(x, y, 0.0));
                }
            }
            (
                positions,
                vec![
                    vec![0, 1, 4, 3],
                    vec![1, 2, 5, 4],
                    vec![3, 4, 7, 6],
                    vec![4, 5, 8, 7],
                ],
            )
        }
        "cube" => cube_faces([0.0; 3], 1.0, &[]),
        "open_box" => cube_faces([0.0; 3], 1.0, &["top"]),
        "cyl_open" => cylinder(12, 1.0, 2.0, false),
        "cyl_closed" => cylinder(12, 1.0, 2.0, true),
        "book_nonmanifold" => (
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(0.0, 0.0, 1.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 1.0),
                DVec3::new(0.0, 1.0, 0.0),
                DVec3::new(0.0, 1.0, 1.0),
                DVec3::new(-1.0, 0.0, 0.0),
                DVec3::new(-1.0, 0.0, 1.0),
            ],
            vec![vec![0, 1, 3, 2], vec![0, 1, 5, 4], vec![0, 1, 7, 6]],
        ),
        "cube_plus_plane" => merge(vec![
            cube_faces([0.0; 3], 1.0, &[]),
            (
                vec![
                    DVec3::new(2.0, -0.5, 0.0),
                    DVec3::new(3.0, -0.5, 0.0),
                    DVec3::new(3.0, 0.5, 0.0),
                    DVec3::new(2.0, 0.5, 0.0),
                ],
                vec![vec![0, 1, 2, 3]],
            ),
        ]),
        "cube_plus_small" => merge(vec![
            cube_faces([0.0; 3], 1.0, &[]),
            cube_faces([2.5, 0.0, 0.0], 0.5, &[]),
        ]),
        "cube_open_attached" => merge(vec![
            cube_faces([0.0; 3], 1.0, &["top"]),
            cube_faces([0.0, 0.0, 1.0], 0.5, &[]),
        ]),
        other => return Err(format!("unknown fixture {other}").into()),
    })
}

fn remesh_params(
    mode: &str,
    depth: u32,
    scale: f64,
    sharpness: f64,
    flood: bool,
    threshold: f64,
) -> Value {
    json!({
        "mode": mode,
        "octree_depth": depth,
        "scale": scale,
        "sharpness": sharpness,
        "use_remove_disconnected": flood,
        "threshold": threshold,
        "use_smooth_shade": false,
    })
}

fn remesh_modifier(params: Value) -> Result<Modifier, Box<dyn Error>> {
    Ok(Modifier {
        id: Id::new("remesh_open_surface")?,
        modifier_type: "remesh".to_owned(),
        name: "remesh".to_owned(),
        enabled: true,
        params: serde_json::from_value(params)?,
        binding_data: None,
        runtime: potter_core::model::ModifierRuntime::default(),
    })
}

fn remesh_potter(kind: &str, params: Value) -> potter_core::error::Result<Mesh> {
    let (positions, polygons) = geometry(kind)
        .map_err(|error| potter_core::error::PotError::invalid_operation(error.to_string()))?;
    let input = Mesh::from_positions_and_faces(positions, polygons)
        .map_err(|error| potter_core::error::PotError::invalid_argument(error.to_string()))?;
    let modifier = remesh_modifier(params)
        .map_err(|error| potter_core::error::PotError::invalid_argument(error.to_string()))?;
    evaluate_modifiers(&input, &[modifier])
}

fn remesh(kind: &str, params: Value) -> Result<Mesh, Box<dyn Error>> {
    Ok(remesh_potter(kind, params)?)
}

fn counts(mesh: &Mesh) -> (usize, usize) {
    (mesh.vertices.len(), mesh.faces.len())
}

#[test]
fn open_planes_and_nonmanifold_books_remesh_to_empty_geometry() -> Result<(), Box<dyn Error>> {
    // Blender 5.2.2 yields a zero-vertex, zero-face mesh for these inputs in every mode.
    for kind in ["plane", "grid_open", "book_nonmanifold"] {
        for mode in ["BLOCKS", "SMOOTH", "SHARP"] {
            let output = remesh(kind, remesh_params(mode, 4, 0.9, 1.0, false, 0.15))?;
            assert_eq!(counts(&output), (0, 0), "{kind}/{mode}");
            assert!(output.validate().is_ok(), "{kind}/{mode} output is invalid");
        }
    }
    Ok(())
}

#[test]
fn remesh_open_box_and_open_cylinder_match_their_closed_counts_at_depth_four()
-> Result<(), Box<dyn Error>> {
    // Blender 5.2.2 also gives open_box 56/54 at depth 2/scale 0.9 and 98/96 at
    // depth 3/scale 0.5; cyl_open gives 56/54 and 75/78 for those respective settings.
    for mode in ["BLOCKS", "SMOOTH", "SHARP"] {
        assert_eq!(
            counts(&remesh(
                "open_box",
                remesh_params(mode, 2, 0.9, 1.0, false, 0.15)
            )?),
            (56, 54),
            "open_box/{mode}/depth2"
        );
        assert_eq!(
            counts(&remesh(
                "open_box",
                remesh_params(mode, 3, 0.5, 1.0, false, 0.15)
            )?),
            (98, 96),
            "open_box/{mode}/depth3_scale0.5"
        );
        assert_eq!(
            counts(&remesh(
                "cyl_open",
                remesh_params(mode, 2, 0.9, 1.0, false, 0.15)
            )?),
            (56, 54),
            "cyl_open/{mode}/depth2"
        );
        assert_eq!(
            counts(&remesh(
                "cyl_open",
                remesh_params(mode, 3, 0.5, 1.0, false, 0.15)
            )?),
            (75, 78),
            "cyl_open/{mode}/depth3_scale0.5"
        );
    }
    for mode in ["BLOCKS", "SMOOTH", "SHARP"] {
        let params = remesh_params(mode, 4, 0.9, 1.0, false, 0.15);
        assert_eq!(
            counts(&remesh("open_box", params.clone())?),
            (1352, 1350),
            "open_box/{mode}"
        );
        assert_eq!(
            counts(&remesh("cube", params.clone())?),
            (1352, 1350),
            "cube/{mode}"
        );
        assert_eq!(
            counts(&remesh("cyl_open", params.clone())?),
            (1200, 1198),
            "cyl_open/{mode}"
        );
        assert_eq!(
            counts(&remesh("cyl_closed", params)?),
            (1200, 1198),
            "cyl_closed/{mode}"
        );
    }
    Ok(())
}

#[test]
fn remesh_mixed_and_disconnected_inputs_remove_components_inside_the_octree()
-> Result<(), Box<dyn Error>> {
    // Blender 5.2.2, SMOOTH, depth 4, scale 0.9. Flood fill keeps the largest component and drops
    // components smaller than threshold * largest; with flood disabled every component stays.
    let cases = [
        ("cube_plus_small", false, 0.15, (208, 204)),
        ("cube_plus_small", true, 0.15, (208, 204)),
        ("cube_plus_small", true, 0.5, (152, 150)),
        ("cube_plus_plane", true, 0.15, (132, 130)),
        ("cube_plus_plane", true, 0.5, (132, 130)),
        ("cube_open_attached", false, 0.15, (620, 616)),
        ("cube_open_attached", true, 0.15, (620, 616)),
        ("cube_open_attached", true, 0.5, (488, 486)),
        ("book_nonmanifold", true, 0.5, (0, 0)),
    ];
    for (kind, flood, threshold, expected) in cases {
        let output = remesh(kind, remesh_params("SMOOTH", 4, 0.9, 1.0, flood, threshold))?;
        assert_eq!(
            counts(&output),
            expected,
            "{kind} flood={flood} threshold={threshold}"
        );
        assert!(output.validate().is_ok(), "{kind} output is invalid");
    }
    Ok(())
}

#[test]
fn remesh_octree_depth_above_twenty_returns_limit_exceeded() -> Result<(), Box<dyn Error>> {
    // Blender's GRID_DIMENSION is 20, so greater depths make mindimen zero. Blender 5.2.2
    // does not complete at depth 21 (observed under a 40-second timeout).
    for depth in [21_u32, 24] {
        let error = remesh_potter(
            "cube",
            remesh_params("SMOOTH", depth, 0.9, 1.0, false, 0.15),
        )
        .err()
        .ok_or("depth above 20 unexpectedly succeeded")?;
        assert_eq!(error.code, ErrorCode::LimitExceeded);
    }
    Ok(())
}

#[test]
fn remesh_high_depth_cube_hits_budget_and_empty_mesh_stays_safe() -> Result<(), Box<dyn Error>> {
    let error = remesh_potter("cube", remesh_params("SMOOTH", 20, 0.9, 1.0, false, 0.15))
        .err()
        .ok_or("depth-20 cube unexpectedly succeeded")?;
    assert_eq!(error.code, ErrorCode::LimitExceeded);
    assert!(
        error.message.contains("combined allocation budget"),
        "high-depth failure was not the allocation budget: {}",
        error.message
    );

    let empty = Mesh::new();
    let modifier = remesh_modifier(remesh_params("SMOOTH", 20, 0.9, 1.0, false, 0.15))?;
    let output = evaluate_modifiers(&empty, &[modifier])?;
    assert_eq!(counts(&output), (0, 0));
    Ok(())
}

#[test]
fn remesh_zero_scale_is_rejected_as_an_invalid_parameter() -> Result<(), Box<dyn Error>> {
    let error = remesh_potter("cube", remesh_params("SMOOTH", 3, 0.0, 1.0, false, 0.15))
        .err()
        .ok_or("zero scale unexpectedly succeeded")?;
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    Ok(())
}

#[test]
fn remesh_open_surface_matrix_matches_blender_topology_and_positions() -> Result<(), Box<dyn Error>>
{
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender DualCon open-surface matrix: no Blender executable was found");
        return Ok(());
    };

    let mut matrix: Vec<(&str, &str, u32, f64, f64, bool, f64)> = Vec::new();
    let all_kinds = [
        "plane",
        "grid_open",
        "cube",
        "open_box",
        "cyl_open",
        "cyl_closed",
        "book_nonmanifold",
        "cube_plus_plane",
        "cube_plus_small",
        "cube_open_attached",
    ];
    for kind in all_kinds {
        for mode in ["BLOCKS", "SMOOTH", "SHARP"] {
            matrix.push((kind, mode, 4, 0.9, 1.0, false, 0.15));
        }
    }
    for kind in [
        "open_box",
        "cyl_open",
        "cube_open_attached",
        "cube_plus_small",
        "cube_plus_plane",
        "plane",
    ] {
        for depth in [2, 3, 5] {
            matrix.push((kind, "SMOOTH", depth, 0.9, 1.0, false, 0.15));
        }
    }
    for kind in ["open_box", "cyl_open"] {
        for mode in ["BLOCKS", "SMOOTH", "SHARP"] {
            matrix.push((kind, mode, 2, 0.9, 1.0, false, 0.15));
            matrix.push((kind, mode, 3, 0.5, 1.0, false, 0.15));
            matrix.push((kind, mode, 5, 0.9, 1.0, false, 0.15));
        }
    }
    for kind in [
        "open_box",
        "cube_plus_plane",
        "cube_open_attached",
        "cyl_open",
        "cube",
    ] {
        for mode in ["SMOOTH", "SHARP"] {
            matrix.push((kind, mode, 4, 0.5, 1.0, false, 0.15));
        }
    }
    matrix.push(("open_box", "SMOOTH", 4, 0.9, 1.0, true, 0.5));
    matrix.push(("open_box", "SMOOTH", 4, 0.9, 1.0, true, 0.0));
    matrix.push(("cube_plus_plane", "SMOOTH", 3, 0.5, 1.0, false, 0.15));
    matrix.push(("cube_plus_plane", "SMOOTH", 3, 0.5, 1.0, true, 0.15));
    matrix.push(("cube_plus_small", "BLOCKS", 3, 0.9, 1.0, false, 0.15));
    matrix.push(("cube_plus_small", "BLOCKS", 3, 0.9, 1.0, true, 0.15));
    matrix.push(("cube_plus_small", "BLOCKS", 3, 0.9, 1.0, true, 0.5));
    for mode in ["BLOCKS", "SMOOTH", "SHARP"] {
        matrix.push(("cube", mode, 2, 0.9, 2.0, true, 0.15));
    }
    for kind in [
        "cube_plus_small",
        "cube_plus_plane",
        "cube_open_attached",
        "book_nonmanifold",
        "cube",
    ] {
        for threshold in [0.15, 0.5] {
            matrix.push((kind, "SMOOTH", 4, 0.9, 1.0, true, threshold));
        }
    }
    matrix.push(("cube_plus_small", "BLOCKS", 4, 0.9, 1.0, true, 0.15));
    matrix.push(("cube_plus_small", "BLOCKS", 4, 0.9, 1.0, true, 0.5));
    matrix.push(("cube_plus_small", "SHARP", 4, 0.9, 1.0, true, 0.15));
    for kind in ["open_box", "cyl_open", "cube"] {
        for sharpness in [0.0, 1.0, 2.0] {
            matrix.push((kind, "SHARP", 4, 0.9, sharpness, false, 0.15));
        }
    }
    matrix.push(("cube", "SMOOTH", 6, 0.9, 1.0, false, 0.15));
    matrix.push(("open_box", "SMOOTH", 6, 0.9, 1.0, false, 0.15));

    let inputs = matrix
        .iter()
        .map(|(kind, mode, depth, scale, sharpness, flood, threshold)| {
            let (positions, polygons) = geometry(kind)?;
            Ok(json!({
                "kind": kind,
                "mode": mode,
                "params": remesh_params(mode, *depth, *scale, *sharpness, *flood, *threshold),
                "positions": positions.iter().map(DVec3::to_array).collect::<Vec<_>>(),
                "polygons": polygons,
            }))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    let directory = tempdir()?;
    let references = blender_reference(&blender, directory.path(), &inputs)?;
    let references = references
        .as_array()
        .ok_or("Blender reference list is missing")?;
    assert_eq!(references.len(), matrix.len(), "Blender reference count");

    for (case, reference) in inputs.iter().zip(references) {
        let kind = case["kind"].as_str().ok_or("case kind")?;
        let mode = case["mode"].as_str().ok_or("case mode")?;
        let context = format!("{kind}/{mode}/{}", case["params"]);
        let params = case["params"].clone();
        let native = remesh(kind, params)?;
        let expected_positions = points(&reference["positions"])?;
        let expected_polygons = polygons(&reference["polygons"])?;
        assert_eq!(
            native.vertices.len(),
            expected_positions.len(),
            "{context}: vertex count"
        );
        assert_eq!(
            native.faces.len(),
            expected_polygons.len(),
            "{context}: face count"
        );

        let tolerance = if mode == "SHARP" { 2.5e-5 } else { 1.0e-6 };
        let native_positions = native
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let (_, maximum_error) =
            match_vertex_positions(&native_positions, &expected_positions, tolerance, &context);
        let ordered_error = native_positions
            .iter()
            .zip(&expected_positions)
            .map(|(actual, expected)| actual.distance(*expected))
            .fold(0.0, f64::max);
        assert!(
            ordered_error <= tolerance,
            "{context}: ordered vertex error {ordered_error} exceeds {tolerance}"
        );
        let ordinals = native
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<std::collections::HashMap<_, _>>();
        let native_polygons = native
            .faces
            .iter()
            .map(|face| {
                face.vertices
                    .iter()
                    .map(|vertex| {
                        ordinals
                            .get(vertex)
                            .copied()
                            .ok_or("native remesh face references an unknown vertex")
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            native_polygons, expected_polygons,
            "{context}: exact polygon cycles/order"
        );
        eprintln!(
            "{context}: vertices={} faces={} max matched vertex error={maximum_error:.9e}",
            native.vertices.len(),
            native.faces.len()
        );
    }
    Ok(())
}

fn blender_reference(
    blender: &Path,
    directory: &Path,
    inputs: &[Value],
) -> Result<Value, Box<dyn Error>> {
    const SCRIPT: &str = r#"
import bpy, json, sys
cases = json.loads(open(sys.argv[-2], encoding="utf-8").read())
results = []
for case in cases:
    mesh = bpy.data.meshes.new("Input")
    mesh.from_pydata(case["positions"], [], case["polygons"])
    mesh.update()
    obj = bpy.data.objects.new("Owner", mesh)
    bpy.context.scene.collection.objects.link(obj)
    modifier = obj.modifiers.new("Remesh", "REMESH")
    for name, value in case["params"].items():
        setattr(modifier, name, value)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    result = evaluated.to_mesh()
    results.append({
        "positions": [list(vertex.co) for vertex in result.vertices],
        "polygons": [list(polygon.vertices) for polygon in result.polygons],
        "smooth": [polygon.use_smooth for polygon in result.polygons],
    })
    evaluated.to_mesh_clear()
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)
with open(sys.argv[-1], "w", encoding="utf-8") as output:
    json.dump(results, output)
"#;
    let script_path = directory.join("dualcon_open_surface.py");
    fs::write(&script_path, SCRIPT)?;
    let input_path = directory.join("dualcon_cases.json");
    fs::write(&input_path, serde_json::to_vec(inputs)?)?;
    let output_path = directory.join("dualcon_reference.json");
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .arg("--")
        .arg(&input_path)
        .arg(&output_path)
        .output()?;
    if !output.status.success() || !output_path.is_file() {
        return Err(std::io::Error::other(format!(
            "Blender DualCon reference failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(serde_json::from_slice(&fs::read(output_path)?)?)
}

fn points(value: &Value) -> Result<Vec<DVec3>, Box<dyn Error>> {
    let rows = value
        .as_array()
        .ok_or_else(|| std::io::Error::other("Blender vertex positions are missing"))?;
    rows.iter()
        .map(|row| {
            let coordinates = row
                .as_array()
                .filter(|coordinates| coordinates.len() == 3)
                .ok_or_else(|| std::io::Error::other("Blender vertex position is invalid"))?;
            let component = |index: usize| {
                coordinates[index]
                    .as_f64()
                    .ok_or_else(|| std::io::Error::other("Blender coordinate is invalid"))
            };
            Ok(DVec3::new(component(0)?, component(1)?, component(2)?))
        })
        .collect()
}

fn polygons(value: &Value) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let rows = value.as_array().ok_or("Blender polygon list is missing")?;
    rows.iter()
        .map(|polygon| {
            polygon
                .as_array()
                .and_then(|indices| {
                    indices
                        .iter()
                        .map(|index| index.as_u64().and_then(|index| usize::try_from(index).ok()))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| "Blender polygon indices are malformed".into())
        })
        .collect()
}

/// Bijective nearest-match between native and reference vertices; returns the native-to-reference
/// index map and the largest matched distance. Blender's SHARP minimizer differs between builds by
/// float32 rounding, hence the caller-provided tolerance.
fn match_vertex_positions(
    actual: &[DVec3],
    expected: &[DVec3],
    tolerance: f64,
    context: &str,
) -> (Vec<usize>, f64) {
    fn assign(
        actual_index: usize,
        candidates: &[Vec<(usize, f64)>],
        expected_to_actual: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for &(expected_index, _) in &candidates[actual_index] {
            if visited[expected_index] {
                continue;
            }
            visited[expected_index] = true;
            if let Some(previous_actual) = expected_to_actual[expected_index] {
                if assign(previous_actual, candidates, expected_to_actual, visited) {
                    expected_to_actual[expected_index] = Some(actual_index);
                    return true;
                }
            } else {
                expected_to_actual[expected_index] = Some(actual_index);
                return true;
            }
        }
        false
    }

    assert_eq!(actual.len(), expected.len(), "{context}: vertex count");
    let candidates = actual
        .iter()
        .map(|actual_position| {
            let mut matches = expected
                .iter()
                .enumerate()
                .filter_map(|(index, expected_position)| {
                    let distance = actual_position.distance(*expected_position);
                    (distance <= tolerance).then_some((index, distance))
                })
                .collect::<Vec<_>>();
            matches.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            matches
        })
        .collect::<Vec<_>>();
    let mut actual_order = (0..actual.len()).collect::<Vec<_>>();
    actual_order.sort_by_key(|index| (candidates[*index].len(), *index));

    let mut expected_to_actual = vec![None; expected.len()];
    for actual_index in actual_order {
        let mut visited = vec![false; expected.len()];
        assert!(
            assign(
                actual_index,
                &candidates,
                &mut expected_to_actual,
                &mut visited
            ),
            "{context}: no bijective vertex correspondence within {tolerance}; vertex {actual_index} has {} candidates",
            candidates[actual_index].len(),
        );
    }

    let mut actual_to_expected = vec![usize::MAX; actual.len()];
    for (expected_index, actual_index) in expected_to_actual.into_iter().enumerate() {
        let Some(actual_index) = actual_index else {
            panic!("{context}: perfect matching did not cover expected vertex {expected_index}");
        };
        actual_to_expected[actual_index] = expected_index;
    }
    let maximum_error = actual
        .iter()
        .zip(&actual_to_expected)
        .map(|(actual_position, expected_index)| {
            actual_position.distance(expected[*expected_index])
        })
        .fold(0.0, f64::max);
    assert!(
        maximum_error <= tolerance,
        "{context}: maximum matched vertex error {maximum_error} exceeds {tolerance}"
    );
    (actual_to_expected, maximum_error)
}

#[test]
fn smooth_shading_matches_blender_on_open_and_closed_dualcontours() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender Remesh shading parity: no Blender executable was found");
        return Ok(());
    };
    let mut cases = Vec::new();
    for kind in ["open_box", "cyl_open", "cube"] {
        for mode in ["BLOCKS", "SMOOTH", "SHARP"] {
            for smooth in [false, true] {
                let (positions, polygons) = geometry(kind)?;
                let mut params = remesh_params(mode, 4, 0.9, 1.0, false, 0.15);
                params["use_smooth_shade"] = json!(smooth);
                cases.push(json!({
                    "kind":kind,
                    "mode":mode,
                    "positions":positions.iter().map(DVec3::to_array).collect::<Vec<_>>(),
                    "polygons":polygons,
                    "params":params
                }));
            }
        }
    }
    let directory = tempdir()?;
    let references = blender_reference(&blender, directory.path(), &cases)?;
    let references = references
        .as_array()
        .ok_or("Blender smooth flags are missing")?;
    assert_eq!(references.len(), cases.len(), "Blender smooth case count");
    for (case, reference) in cases.iter().zip(references) {
        let kind = case["kind"].as_str().ok_or("case kind")?;
        let mode = case["mode"].as_str().ok_or("case mode")?;
        let context = format!(
            "{kind}/{mode}/smooth={}",
            case["params"]["use_smooth_shade"]
        );
        let native = remesh(kind, case["params"].clone())?;
        let expected = reference["smooth"]
            .as_array()
            .ok_or("Blender face smooth flags are missing")?;
        assert_eq!(native.faces.len(), expected.len(), "{context}: face count");
        let values = native
            .attributes
            .get("shade_smooth")
            .and_then(|attribute| attribute["values"].as_object());
        for (face, expected_smooth) in native.faces.iter().zip(expected) {
            let actual_smooth = values
                .and_then(|values| values.get(&format!("f{}", face.id)))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            assert_eq!(
                Some(actual_smooth),
                expected_smooth.as_bool(),
                "{context}: face {} smooth flag",
                face.id
            );
        }
    }
    Ok(())
}
