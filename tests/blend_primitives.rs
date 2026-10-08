use std::{collections::BTreeMap, error::Error, fs, io, path::Path, process::Command};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use serde::Serialize;
use serde_json::{Value, json};
use tempfile::tempdir;

const POSITION_TOLERANCE: f64 = 1.0e-5;

type TestResult<T> = Result<T, Box<dyn Error>>;

#[derive(Serialize)]
struct PrimitiveCase {
    name: &'static str,
    kind: &'static str,
    params: Value,
}

fn primitive_cases() -> Vec<PrimitiveCase> {
    vec![
        PrimitiveCase {
            name: "box",
            kind: "box",
            params: json!({"size":2.25}),
        },
        PrimitiveCase {
            name: "plane",
            kind: "plane",
            params: json!({"size":1.75}),
        },
        PrimitiveCase {
            name: "uv_sphere",
            kind: "uv_sphere",
            params: json!({"segments":7,"ring_count":5,"radius":1.25}),
        },
        PrimitiveCase {
            name: "sphere_alias",
            kind: "sphere",
            params: json!({"segments":7,"ring_count":5,"radius":1.25}),
        },
        PrimitiveCase {
            name: "cylinder_nothing",
            kind: "cylinder",
            params: json!({"vertices":5,"radius":0.75,"depth":2.25,"end_fill_type":"NOTHING"}),
        },
        PrimitiveCase {
            name: "cylinder_ngon",
            kind: "cylinder",
            params: json!({"vertices":5,"radius":0.75,"depth":2.25,"end_fill_type":"NGON"}),
        },
        PrimitiveCase {
            name: "cylinder_trifan",
            kind: "cylinder",
            params: json!({"vertices":5,"radius":0.75,"depth":2.25,"end_fill_type":"TRIFAN"}),
        },
        PrimitiveCase {
            name: "cone_nothing",
            kind: "cone",
            params: json!({"vertices":5,"radius1":0.75,"radius2":0.3,"depth":2.25,"end_fill_type":"NOTHING"}),
        },
        PrimitiveCase {
            name: "cone_ngon",
            kind: "cone",
            params: json!({"vertices":5,"radius1":0.75,"radius2":0.3,"depth":2.25,"end_fill_type":"NGON"}),
        },
        PrimitiveCase {
            name: "cone_trifan",
            kind: "cone",
            params: json!({"vertices":5,"radius1":0.75,"radius2":0.3,"depth":2.25,"end_fill_type":"TRIFAN"}),
        },
        PrimitiveCase {
            name: "cone_apex",
            kind: "cone",
            params: json!({"vertices":5,"radius1":0.75,"radius2":0.0,"depth":2.25,"end_fill_type":"NGON"}),
        },
        PrimitiveCase {
            name: "torus_major_minor",
            kind: "torus",
            params: json!({"mode":"MAJOR_MINOR","major_segments":7,"minor_segments":5,"major_radius":1.4,"minor_radius":0.3}),
        },
        PrimitiveCase {
            name: "torus_ext_int",
            kind: "torus",
            params: json!({"mode":"EXT_INT","major_segments":7,"minor_segments":5,"abso_major_rad":1.7,"abso_minor_rad":0.5}),
        },
        PrimitiveCase {
            name: "icosphere_subdivision_1",
            kind: "icosphere",
            params: json!({"subdivisions":1,"radius":1.2}),
        },
        PrimitiveCase {
            name: "icosphere_subdivision_2",
            kind: "icosphere",
            params: json!({"subdivisions":2,"radius":1.2}),
        },
        PrimitiveCase {
            name: "icosphere_subdivision_3",
            kind: "icosphere",
            params: json!({"subdivisions":3,"radius":1.2}),
        },
        PrimitiveCase {
            name: "icosphere_subdivision_4",
            kind: "icosphere",
            params: json!({"subdivisions":4,"radius":1.2}),
        },
        PrimitiveCase {
            name: "icosphere_subdivision_5",
            kind: "icosphere",
            params: json!({"subdivisions":5,"radius":1.2}),
        },
        PrimitiveCase {
            name: "circle_nothing",
            kind: "circle",
            params: json!({"vertices":5,"radius":0.75,"fill_type":"NOTHING"}),
        },
        PrimitiveCase {
            name: "circle_ngon",
            kind: "circle",
            params: json!({"vertices":5,"radius":0.75,"fill_type":"NGON"}),
        },
        PrimitiveCase {
            name: "circle_trifan",
            kind: "circle",
            params: json!({"vertices":5,"radius":0.75,"fill_type":"TRIFAN"}),
        },
        PrimitiveCase {
            name: "grid",
            kind: "grid",
            params: json!({"x_subdivisions":3,"y_subdivisions":2,"size":2.25}),
        },
    ]
}

fn run_blender_fixture(blender: &Path, root: &Path, cases: &[PrimitiveCase]) -> TestResult<Value> {
    let cases_path = root.join("cases.json");
    fs::write(&cases_path, serde_json::to_vec(cases)?)?;
    let output_path = root.join("blender_meshes.json");
    let script_path = root.join("primitive_parity.py");
    fs::write(
        &script_path,
        r#"
import bpy
import json
import os
import sys

root = sys.argv[sys.argv.index("--") + 1]
with open(os.path.join(root, "cases.json"), encoding="utf-8") as source:
    cases = json.load(source)
operators = {
    "box": bpy.ops.mesh.primitive_cube_add,
    "plane": bpy.ops.mesh.primitive_plane_add,
    "uv_sphere": bpy.ops.mesh.primitive_uv_sphere_add,
    "sphere": bpy.ops.mesh.primitive_uv_sphere_add,
    "cylinder": bpy.ops.mesh.primitive_cylinder_add,
    "cone": bpy.ops.mesh.primitive_cone_add,
    "torus": bpy.ops.mesh.primitive_torus_add,
    "icosphere": bpy.ops.mesh.primitive_ico_sphere_add,
    "circle": bpy.ops.mesh.primitive_circle_add,
    "grid": bpy.ops.mesh.primitive_grid_add,
}
result = {}
for case in cases:
    operators[case["kind"]](**case["params"])
    mesh = bpy.context.object.data
    result[case["name"]] = {
        "vertices": [list(vertex.co) for vertex in mesh.vertices],
        "edges": [list(edge.vertices) for edge in mesh.edges],
        "faces": [list(face.vertices) for face in mesh.polygons],
    }
    bpy.data.objects.remove(bpy.context.object, do_unlink=True)
with open(os.path.join(root, "blender_meshes.json"), "w", encoding="utf-8") as destination:
    json.dump(result, destination, separators=(",", ":"))
"#,
    )?;
    let root = fs::canonicalize(root)?;
    let root = root
        .to_str()
        .ok_or_else(|| io::Error::other("temporary Blender fixture path is not UTF-8"))?;
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(script_path)
        .args(["--", root])
        .output()?;
    assert!(
        output.status.success(),
        "Blender primitive fixture failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&fs::read(output_path)?)?)
}

fn positions(value: &Value, label: &str) -> TestResult<Vec<[f64; 3]>> {
    value
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{label} vertices are missing")))?
        .iter()
        .map(|value| {
            let point = value
                .as_array()
                .filter(|point| point.len() == 3)
                .ok_or_else(|| {
                    io::Error::other(format!("{label} vertex must have three coordinates"))
                })?;
            Ok([
                point[0]
                    .as_f64()
                    .ok_or_else(|| io::Error::other(format!("{label} vertex x is invalid")))?,
                point[1]
                    .as_f64()
                    .ok_or_else(|| io::Error::other(format!("{label} vertex y is invalid")))?,
                point[2]
                    .as_f64()
                    .ok_or_else(|| io::Error::other(format!("{label} vertex z is invalid")))?,
            ])
        })
        .collect()
}

fn index(value: &Value, label: &str) -> TestResult<usize> {
    let raw = value
        .as_u64()
        .ok_or_else(|| io::Error::other(format!("{label} index is invalid")))?;
    Ok(usize::try_from(raw)?)
}

fn indices(value: &Value, label: &str) -> TestResult<Vec<usize>> {
    value
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{label} indices are missing")))?
        .iter()
        .map(|item| index(item, label))
        .collect()
}

fn canonical_cycle(vertices: &[usize]) -> Vec<usize> {
    let mut canonical = vertices.to_vec();
    for start in 1..vertices.len() {
        let candidate = (0..vertices.len())
            .map(|offset| vertices[(start + offset) % vertices.len()])
            .collect::<Vec<_>>();
        if candidate < canonical {
            canonical = candidate;
        }
    }
    canonical
}

fn coordinate_error(first: [f64; 3], second: [f64; 3]) -> f64 {
    first
        .into_iter()
        .zip(second)
        .map(|(first, second)| (first - second).abs())
        .fold(0.0, f64::max)
}

fn assert_mesh_matches_blender(case: &PrimitiveCase, expected: &Value) -> TestResult<()> {
    let blender = expected
        .get(case.name)
        .ok_or_else(|| io::Error::other(format!("{} Blender geometry is missing", case.name)))?;
    let blender_positions = positions(&blender["vertices"], case.name)?;
    let actual = potter::geom::primitive(case.kind, &case.params)?;
    assert_eq!(
        actual.vertices.len(),
        blender_positions.len(),
        "{} vertex count",
        case.name
    );

    let mut actual_to_blender = BTreeMap::new();
    if case.kind == "icosphere" {
        for (index, (vertex, expected)) in
            actual.vertices.iter().zip(&blender_positions).enumerate()
        {
            let coordinate = vertex.co.to_array();
            let error = coordinate_error(coordinate, *expected);
            assert!(
                error <= POSITION_TOLERANCE,
                "{} vertex order {} differs by {error}: Potter={coordinate:?}, Blender={expected:?}",
                case.name,
                index
            );
            actual_to_blender.insert(vertex.id, index);
        }
    } else {
        let mut matched = vec![false; blender_positions.len()];
        for vertex in &actual.vertices {
            let coordinate = vertex.co.to_array();
            let mut closest = None;
            let mut minimum_error = f64::INFINITY;
            for (index, candidate) in blender_positions.iter().copied().enumerate() {
                if matched[index] {
                    continue;
                }
                let error = coordinate_error(coordinate, candidate);
                if error < minimum_error {
                    closest = Some(index);
                    minimum_error = error;
                }
            }
            let index = closest.ok_or_else(|| {
                io::Error::other(format!("{} Blender vertex match is missing", case.name))
            })?;
            assert!(
                minimum_error <= POSITION_TOLERANCE,
                "{} vertex {} differs by {minimum_error}: Potter={coordinate:?}, Blender={:?}",
                case.name,
                vertex.id,
                blender_positions[index]
            );
            matched[index] = true;
            actual_to_blender.insert(vertex.id, index);
        }
    }

    let mut expected_edges = blender["edges"]
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{} Blender edges are missing", case.name)))?
        .iter()
        .map(|edge| {
            let endpoints = indices(edge, case.name)?;
            if endpoints.len() != 2 {
                return Err(io::Error::other(format!(
                    "{} Blender edge must have two endpoints",
                    case.name
                ))
                .into());
            }
            Ok([
                endpoints[0].min(endpoints[1]),
                endpoints[0].max(endpoints[1]),
            ])
        })
        .collect::<TestResult<Vec<_>>>()?;
    let mut actual_edges = actual
        .edges
        .iter()
        .map(|edge| {
            let first = *actual_to_blender.get(&edge.vertices[0]).ok_or_else(|| {
                io::Error::other(format!("{} edge vertex mapping is missing", case.name))
            })?;
            let second = *actual_to_blender.get(&edge.vertices[1]).ok_or_else(|| {
                io::Error::other(format!("{} edge vertex mapping is missing", case.name))
            })?;
            Ok([first.min(second), first.max(second)])
        })
        .collect::<TestResult<Vec<_>>>()?;
    expected_edges.sort_unstable();
    actual_edges.sort_unstable();
    assert_eq!(actual_edges, expected_edges, "{} edge topology", case.name);

    let expected_faces = blender["faces"]
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{} Blender faces are missing", case.name)))?
        .iter()
        .map(|face| indices(face, case.name))
        .collect::<TestResult<Vec<_>>>()?;
    let actual_faces = actual
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| {
                    actual_to_blender.get(id).copied().ok_or_else(|| {
                        io::Error::other(format!("{} face vertex mapping is missing", case.name))
                            .into()
                    })
                })
                .collect::<TestResult<Vec<_>>>()
        })
        .collect::<TestResult<Vec<_>>>()?;
    if case.kind == "icosphere" {
        assert_eq!(
            actual_faces, expected_faces,
            "{} ordered faces and winding",
            case.name
        );
    } else {
        let mut expected_faces = expected_faces
            .iter()
            .map(|face| canonical_cycle(face))
            .collect::<Vec<_>>();
        let mut actual_faces = actual_faces
            .iter()
            .map(|face| canonical_cycle(face))
            .collect::<Vec<_>>();
        expected_faces.sort_unstable();
        actual_faces.sort_unstable();
        assert_eq!(
            actual_faces, expected_faces,
            "{} face topology and winding",
            case.name
        );
    }
    Ok(())
}

#[test]
fn node_create_primitive_meshes_match_blender_operator_positions_and_topology() -> TestResult<()> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping primitive Blender parity: Blender is unavailable");
        return Ok(());
    };
    let cases = primitive_cases();
    let directory = tempdir()?;
    let expected = run_blender_fixture(&blender, directory.path(), &cases)?;
    for case in &cases {
        assert_mesh_matches_blender(case, &expected)?;
    }
    Ok(())
}
