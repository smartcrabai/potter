#![expect(
    clippy::unwrap_used,
    reason = "fixed geometry fixtures make failures easier to diagnose"
)]

use std::{
    cmp::Ordering,
    collections::HashMap,
    fs,
    path::Path,
    process::{Command, Output},
};

use glam::DVec3;
use potter_core::{
    error::ErrorCode,
    geom::{BoxParams, Mesh, PlaneParams, edit},
};
use serde_json::{Map, Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;

fn selected_ids(mesh: &Mesh, domain: &str, selector: &Value) -> Vec<String> {
    let mut selected = mesh.clone();
    edit::apply(
        &mut selected,
        "set_attribute",
        &json!({
            "elements":{"domain":domain,"selector":selector},
            "name":"selected",
            "value":true
        }),
    )
    .unwrap();
    selected.attributes["selected"]["values"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

fn assert_invalid_selection(mesh: &Mesh, elements: &Value, message: &str) {
    let mut edited = mesh.clone();
    let error = edit::apply(
        &mut edited,
        "set_attribute",
        &json!({"elements":elements,"name":"selected","value":true}),
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidArgument);
    assert!(error.message.contains(message), "{}", error.message);
}

#[test]
fn selectors_resolve_geometry_attributes_connectivity_and_materials() {
    let cube = Mesh::box_mesh(BoxParams::default()).unwrap();

    assert_eq!(
        selected_ids(
            &cube,
            "vertex",
            &json!({"type":"position_sphere","center":[-1,-1,-1],"radius":0})
        ),
        ["v0"]
    );
    assert_eq!(
        selected_ids(
            &cube,
            "edge",
            &json!({"type":"position_sphere","center":[-1,0,-1],"radius":0})
        ),
        ["e0"]
    );
    assert_eq!(
        selected_ids(
            &cube,
            "face",
            &json!({"type":"position_box","min":[-1,-1,-1],"max":[1,1,-0.999_999]})
        ),
        ["f0"]
    );
    assert_eq!(
        selected_ids(
            &cube,
            "face",
            &json!({"type":"normal_cone","axis":[0,0,1],"angle":0})
        ),
        ["f1"]
    );

    let mut attributed = cube.clone();
    let scores = attributed
        .vertices
        .iter()
        .map(|vertex| (format!("v{}", vertex.id), json!(vertex.id)))
        .collect::<Map<_, _>>();
    attributed.attributes.insert(
        "score".to_owned(),
        json!({"domain":"point","values":scores}),
    );
    for (operator, value, expected) in [
        ("equals", 3, ["v3"].as_slice()),
        ("less_than", 3, ["v0", "v1", "v2"].as_slice()),
        ("less_than_or_equal", 3, ["v0", "v1", "v2", "v3"].as_slice()),
        ("greater_than", 5, ["v6", "v7"].as_slice()),
        ("greater_than_or_equal", 5, ["v5", "v6", "v7"].as_slice()),
    ] {
        let actual = selected_ids(
            &attributed,
            "point",
            &json!({"type":"attribute","name":"score","operator":operator,"value":value}),
        );
        assert_eq!(actual, expected, "attribute operator {operator}");
    }

    assert_eq!(
        selected_ids(
            &attributed,
            "vertex",
            &json!({"type":"attribute","name":"score","value":3}),
        ),
        ["v3"]
    );

    let disconnected = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(3.0, 0.0, 0.0),
            DVec3::new(4.0, 0.0, 0.0),
            DVec3::new(4.0, 1.0, 0.0),
            DVec3::new(3.0, 1.0, 0.0),
        ],
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
    )
    .unwrap();
    assert_eq!(
        selected_ids(
            &disconnected,
            "vertex",
            &json!({"type":"connected","ids":["v0"]})
        ),
        ["v0", "v1", "v2", "v3"]
    );
    assert_eq!(
        selected_ids(
            &disconnected,
            "edge",
            &json!({"type":"linked","ids":["e0"]})
        ),
        ["e0", "e1", "e2", "e3"]
    );
    assert_eq!(
        selected_ids(
            &disconnected,
            "face",
            &json!({"type":"connected","ids":["f0"]})
        ),
        ["f0"]
    );

    let mut materials = cube.clone();
    materials.faces[0].material_index = 7;
    materials.faces[2].material_index = 7;
    assert_eq!(
        selected_ids(
            &materials,
            "face",
            &json!({"type":"material_index","index":7})
        ),
        ["f0", "f2"]
    );

    let mut no_match = cube.clone();
    edit::apply(
        &mut no_match,
        "set_attribute",
        &json!({
            "elements":{"domain":"vertex","selector":{"type":"position_box","min":[10,10,10],"max":[11,11,11]}},
            "name":"picked",
            "value":true
        }),
    )
    .unwrap();
    assert!(
        no_match.attributes["picked"]["values"]
            .as_object()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn invalid_and_ambiguous_selector_inputs_are_invalid_arguments() {
    let mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
    let invalid_selectors = [
        ("vertex", json!({}), "type is required"),
        (
            "vertex",
            json!({"type":"nearest"}),
            "unknown element selector",
        ),
        (
            "vertex",
            json!({"type":"position_box","min":[0,0,0],"max":[1,1,1],"extra":true}),
            "unknown position_box selector field",
        ),
        (
            "vertex",
            json!({"type":"position_box","min":[1,0,0],"max":[0,1,1]}),
            "min must not exceed max",
        ),
        (
            "vertex",
            json!({"type":"position_sphere","center":[0,0,0],"radius":-1}),
            "radius must be non-negative",
        ),
        (
            "face",
            json!({"type":"normal_cone","axis":[0,0,0],"angle":0}),
            "axis must be non-zero",
        ),
        (
            "face",
            json!({"type":"normal_cone","axis":[0,0,1],"angle":4}),
            "angle must be between zero and pi",
        ),
        (
            "vertex",
            json!({"type":"attribute","name":"","value":1}),
            "non-empty name",
        ),
        (
            "vertex",
            json!({"type":"attribute","name":"score","operator":"between","value":1}),
            "unknown attribute selector operator",
        ),
        (
            "vertex",
            json!({"type":"attribute","name":"missing","value":1}),
            "has no value map",
        ),
        (
            "vertex",
            json!({"type":"connected","ids":[]}),
            "at least one seed ID",
        ),
        (
            "vertex",
            json!({"type":"material_index","index":0}),
            "requires the face domain",
        ),
        (
            "vertex",
            json!({"type":"position_box","max":[1,1,1]}),
            "requires min",
        ),
        (
            "vertex",
            json!({"type":"position_box","min":"bad","max":[1,1,1]}),
            "must be an array of three numbers",
        ),
        (
            "vertex",
            json!({"type":"position_sphere","radius":1}),
            "requires center",
        ),
        (
            "vertex",
            json!({"type":"position_sphere","center":[0,0,0]}),
            "finite number",
        ),
        (
            "face",
            json!({"type":"normal_cone","angle":0}),
            "requires axis",
        ),
        (
            "face",
            json!({"type":"normal_cone","axis":[0,0,1]}),
            "finite number",
        ),
        (
            "vertex",
            json!({"type":"attribute","name":"score"}),
            "requires value",
        ),
        (
            "vertex",
            json!({"type":"attribute","name":"score","value":1,"extra":true}),
            "unknown attribute selector field",
        ),
        ("vertex", json!({"type":"connected"}), "requires seed ids"),
        (
            "face",
            json!({"type":"material_index"}),
            "requires a u32 index",
        ),
        (
            "face",
            json!({"type":"material_index","index":u64::from(u32::MAX)+1}),
            "requires a u32 index",
        ),
    ];
    for (domain, selector, message) in invalid_selectors {
        assert_invalid_selection(
            &mesh,
            &json!({"domain":domain,"selector":selector}),
            message,
        );
    }

    assert_invalid_selection(
        &mesh,
        &json!({"domain":"vertex","selector":null}),
        "must be a JSON object",
    );
    for (elements, message) in [
        (
            json!({"domain":"vertex","ids":["v0","v0"]}),
            "duplicate element ID",
        ),
        (json!({"domain":"vertex","ids":["v999"]}), "does not exist"),
        (
            json!({"domain":"vertex","ids":["v4294967296"]}),
            "outside the u32 range",
        ),
        (
            json!({"domain":"vertex","ids":["vertex-0"]}),
            "does not match",
        ),
        (json!({"domain":"vertex"}), "requires ids or selector"),
        (
            json!({"ids":["v0"]}),
            "domain must be vertex, edge, or face",
        ),
        (
            json!({"domain":"corners","ids":["v0"]}),
            "unknown element domain",
        ),
    ] {
        assert_invalid_selection(&mesh, &elements, message);
    }

    assert_invalid_selection(
        &mesh,
        &json!({"domain":"vertex","ids":[]}),
        "elements.ids must not be empty",
    );
    assert_invalid_selection(
        &mesh,
        &json!({
            "domain":"vertex",
            "ids":["v0"],
            "selector":{"type":"position_sphere","center":[0,0,0],"radius":1}
        }),
        "either ids or selector",
    );
    assert_invalid_selection(
        &mesh,
        &json!({"domain":"vertex","ids":["e0"]}),
        "does not match the vertices domain",
    );
}

#[derive(Clone, Copy)]
enum LegacyKind {
    Scalar,
    Vector,
    Text,
    Flag,
}

const LEGACY_KINDS: [(&str, LegacyKind); 4] = [
    ("scalar", LegacyKind::Scalar),
    ("vector", LegacyKind::Vector),
    ("text", LegacyKind::Text),
    ("flag", LegacyKind::Flag),
];
const LEGACY_DOMAINS: [&str; 4] = ["point", "edge", "face", "corner"];

fn legacy_value(kind: LegacyKind, domain: &str, corner_count: usize) -> Value {
    let value = match kind {
        LegacyKind::Scalar => json!(2.5),
        LegacyKind::Vector => json!([0.25, 0.5, 0.75]),
        LegacyKind::Text => json!("painted"),
        LegacyKind::Flag => json!(true),
    };
    if domain == "corner" {
        Value::Array(vec![value; corner_count])
    } else {
        value
    }
}

fn attach_legacy_attributes(mesh: &mut Mesh) {
    for domain in LEGACY_DOMAINS {
        let values = match domain {
            "point" => mesh
                .vertices
                .iter()
                .map(|vertex| (format!("v{}", vertex.id), None))
                .collect::<Vec<_>>(),
            "edge" => mesh
                .edges
                .iter()
                .map(|edge| (format!("e{}", edge.id), None))
                .collect::<Vec<_>>(),
            "face" | "corner" => mesh
                .faces
                .iter()
                .map(|face| (format!("f{}", face.id), Some(face.vertices.len())))
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };
        for (suffix, kind) in LEGACY_KINDS {
            let values = values
                .iter()
                .map(|(key, corners)| {
                    (
                        key.clone(),
                        legacy_value(kind, domain, corners.unwrap_or_default()),
                    )
                })
                .collect::<Map<_, _>>();
            mesh.attributes.insert(
                format!("legacy_{domain}_{suffix}"),
                json!({"domain":domain,"values":values}),
            );
        }
    }
}

fn assert_legacy_attributes_cover_live_elements(mesh: &Mesh, operation: &str) {
    for domain in LEGACY_DOMAINS {
        let live_values = match domain {
            "point" => mesh
                .vertices
                .iter()
                .map(|vertex| (format!("v{}", vertex.id), None))
                .collect::<Vec<_>>(),
            "edge" => mesh
                .edges
                .iter()
                .map(|edge| (format!("e{}", edge.id), None))
                .collect::<Vec<_>>(),
            "face" | "corner" => mesh
                .faces
                .iter()
                .map(|face| (format!("f{}", face.id), Some(face.vertices.len())))
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };
        for (suffix, kind) in LEGACY_KINDS {
            let name = format!("legacy_{domain}_{suffix}");
            let attribute = &mesh.attributes[&name];
            assert_eq!(attribute["domain"], domain, "{operation} {name} domain");
            let values = attribute["values"].as_object().unwrap();
            assert_eq!(
                values.len(),
                live_values.len(),
                "{operation} {name} retained stale keys"
            );
            for (key, corners) in &live_values {
                assert_eq!(
                    values[key],
                    legacy_value(kind, domain, corners.unwrap_or_default()),
                    "{operation} {name} value for {key}"
                );
            }
        }
    }
}

#[test]
fn legacy_attributes_survive_split_subdivide_dissolve_and_bevel() {
    let mut cases = Vec::new();

    let mut split = Mesh::box_mesh(BoxParams::default()).unwrap();
    attach_legacy_attributes(&mut split);
    cases.push((
        split,
        "split",
        json!({"elements":{"domain":"face","ids":["f0"]}}),
    ));

    let mut subdivide = Mesh::plane(PlaneParams::default()).unwrap();
    attach_legacy_attributes(&mut subdivide);
    cases.push((
        subdivide,
        "subdivide",
        json!({"elements":{"domain":"face","ids":["f0"]},"cuts":1}),
    ));

    let mut dissolve = Mesh::box_mesh(BoxParams::default()).unwrap();
    attach_legacy_attributes(&mut dissolve);
    cases.push((
        dissolve,
        "dissolve",
        json!({"elements":{"domain":"edge","ids":["e0"]}}),
    ));

    let mut bevel = Mesh::box_mesh(BoxParams::default()).unwrap();
    attach_legacy_attributes(&mut bevel);
    cases.push((
        bevel,
        "bevel",
        json!({"elements":{"domain":"edge","ids":["e0"]},"width":0.1}),
    ));

    for (mut mesh, operation, arguments) in cases {
        let result = edit::apply(&mut mesh, operation, &arguments).unwrap();
        mesh.validate().unwrap();
        assert_legacy_attributes_cover_live_elements(&mesh, operation);
        assert!(
            !result["created"]["vertices"].as_array().unwrap().is_empty()
                || !result["created"]["edges"].as_array().unwrap().is_empty()
                || !result["created"]["faces"].as_array().unwrap().is_empty(),
            "{operation} must report the topology it created"
        );
    }
}

#[test]
fn fill_and_merge_change_loose_topology_without_losing_geometry() {
    let blender = blender_mesh_outputs();
    let mut mesh = Mesh::default();
    let corners = [
        DVec3::new(-1.0, -1.0, 0.0),
        DVec3::new(1.0, -1.0, 0.0),
        DVec3::new(1.0, 1.0, 0.0),
        DVec3::new(-1.0, 1.0, 0.0),
    ];
    let ids = corners
        .into_iter()
        .map(|position| mesh.insert_vertex(position).unwrap())
        .collect::<Vec<_>>();
    let edge_ids = (0..4)
        .map(|index| {
            mesh.insert_edge([ids[index], ids[(index + 1) % 4]])
                .unwrap()
        })
        .collect::<Vec<_>>();
    let fill = edit::apply(
        &mut mesh,
        "fill",
        &json!({
            "elements":{"domain":"edge","ids":edge_ids.iter().map(|id|format!("e{id}")).collect::<Vec<_>>()}
        }),
    )
    .unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.faces.len(), 1);
    assert_eq!(mesh.faces[0].vertices.len(), 4);
    assert_eq!(fill["created"]["faces"].as_array().unwrap().len(), 1);
    if let Some(blender) = &blender {
        assert_signatures_match(
            &mesh_signature(&mesh),
            &blender_signature(&blender["fill"]),
            "fill",
        );
    }

    let mut loose = Mesh::default();
    let first = loose.insert_vertex(DVec3::new(-1.0, 0.0, 0.0)).unwrap();
    let second = loose.insert_vertex(DVec3::new(1.0, 0.0, 0.0)).unwrap();
    let keep = loose.insert_vertex(DVec3::new(0.0, 2.0, 0.0)).unwrap();
    let merged = edit::apply(
        &mut loose,
        "merge",
        &json!({"elements":{"domain":"vertex","ids":[format!("v{first}"),format!("v{second}")]}}),
    )
    .unwrap();
    loose.validate().unwrap();
    assert_eq!(loose.vertices.len(), 2);
    assert_eq!(loose.vertex(first).unwrap().co, DVec3::ZERO);
    assert_eq!(loose.vertex(keep).unwrap().co, DVec3::new(0.0, 2.0, 0.0));
    assert_eq!(merged["deleted"]["vertices"], json!([format!("v{second}")]));
    if let Some(blender) = &blender {
        assert_signatures_match(
            &mesh_signature(&loose),
            &blender_signature(&blender["merge"]),
            "merge",
        );
    }
}

fn init_cli_scene(path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(path)
        .arg("--json")
        .output()
        .unwrap()
}

fn apply_cli(scene: &Path, operations: &Value) -> Output {
    let directory = tempdir().unwrap();
    let batch = directory.path().join("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":operations
        }))
        .unwrap(),
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .arg("--json")
        .output()
        .unwrap()
}

#[test]
fn mesh_wrapper_conflicts_report_actionable_codes_and_json_pointers() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("scene");
    let initialized = init_cli_scene(&scene);
    assert!(initialized.status.success());

    let conflict = apply_cli(
        &scene,
        &json!([
            {"op":"node.create","id":"body","kind":"box","params":{"size":2}},
            {
                "op":"mesh.subdivide",
                "target":{"id":"body","elements":{"domain":"face","ids":["f0"]}},
                "elements":{"domain":"face","ids":["f1"]},
                "cuts":1
            }
        ]),
    );
    assert_eq!(conflict.status.code(), Some(2));
    let envelope: Value = serde_json::from_slice(&conflict.stdout).unwrap();
    assert_eq!(envelope["error"]["code"], "INVALID_OPERATION");
    assert_eq!(envelope["error"]["details"]["pointer"], "/elements");

    let invalid_bevel = apply_cli(
        &scene,
        &json!([
            {"op":"node.create","id":"body","kind":"box","params":{"size":2}},
            {
                "op":"mesh.bevel",
                "target":{"id":"body"},
                "elements":{"domain":"edge","ids":["e0"]},
                "width":0.1,
                "segments":2
            }
        ]),
    );
    assert_eq!(invalid_bevel.status.code(), Some(2));
    let envelope: Value = serde_json::from_slice(&invalid_bevel.stdout).unwrap();
    assert_eq!(envelope["error"]["code"], "INVALID_OPERATION");
    assert_eq!(envelope["error"]["details"]["pointer"], "/segments");
}

#[derive(Debug)]
struct MeshSignature {
    vertices: Vec<[f64; 3]>,
    edges: Vec<[[f64; 3]; 2]>,
    faces: Vec<Vec<[f64; 3]>>,
}

fn coordinate_cmp(left: &[f64; 3], right: &[f64; 3]) -> Ordering {
    for axis in 0..3 {
        let ordering = left[axis].total_cmp(&right[axis]);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    Ordering::Equal
}

fn coordinate_list_cmp(left: &[[f64; 3]], right: &[[f64; 3]]) -> Ordering {
    for (left, right) in left.iter().zip(right) {
        let ordering = coordinate_cmp(left, right);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

fn rotate_cycle_to_minimum(mut points: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
    if points.is_empty() {
        return points;
    }
    let first = (0..points.len())
        .min_by(|left, right| coordinate_cmp(&points[*left], &points[*right]))
        .unwrap();
    points.rotate_left(first);
    points
}

fn canonical_cycle(points: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
    let forward = rotate_cycle_to_minimum(points);
    let mut reversed = forward.clone();
    reversed.reverse();
    let reversed = rotate_cycle_to_minimum(reversed);
    if coordinate_list_cmp(&forward, &reversed) == Ordering::Greater {
        reversed
    } else {
        forward
    }
}

fn normalize_signature(mut signature: MeshSignature) -> MeshSignature {
    let normalize = |point: &mut [f64; 3]| {
        for value in point {
            if *value == 0.0 {
                *value = 0.0;
            }
        }
    };
    for point in &mut signature.vertices {
        normalize(point);
    }
    for edge in &mut signature.edges {
        normalize(&mut edge[0]);
        normalize(&mut edge[1]);
        edge.sort_by(coordinate_cmp);
    }
    for face in &mut signature.faces {
        for point in face.iter_mut() {
            normalize(point);
        }
        *face = canonical_cycle(std::mem::take(face));
    }
    signature.vertices.sort_by(coordinate_cmp);
    signature.edges.sort_by(|left, right| {
        coordinate_cmp(&left[0], &right[0]).then_with(|| coordinate_cmp(&left[1], &right[1]))
    });
    signature
        .faces
        .sort_by(|left, right| coordinate_list_cmp(left, right));
    signature
}

fn mesh_signature(mesh: &Mesh) -> MeshSignature {
    normalize_signature(MeshSignature {
        vertices: mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co.to_array())
            .collect(),
        edges: mesh
            .edges
            .iter()
            .map(|edge| {
                [
                    mesh.vertex(edge.vertices[0]).unwrap().co.to_array(),
                    mesh.vertex(edge.vertices[1]).unwrap().co.to_array(),
                ]
            })
            .collect(),
        faces: mesh
            .faces
            .iter()
            .map(|face| {
                face.vertices
                    .iter()
                    .map(|id| mesh.vertex(*id).unwrap().co.to_array())
                    .collect()
            })
            .collect(),
    })
}

fn blender_signature(value: &Value) -> MeshSignature {
    let vertices = value["vertices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|vertex| {
            let co = vertex.as_array().unwrap();
            [
                co[0].as_f64().unwrap(),
                co[1].as_f64().unwrap(),
                co[2].as_f64().unwrap(),
            ]
        })
        .collect::<Vec<_>>();
    let indexed_points = |ids: &Value| {
        ids.as_array()
            .unwrap()
            .iter()
            .map(|id| vertices[usize::try_from(id.as_u64().unwrap()).unwrap()])
            .collect::<Vec<_>>()
    };
    let edges = value["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            let points = indexed_points(edge);
            [points[0], points[1]]
        })
        .collect();
    let faces = value["faces"]
        .as_array()
        .unwrap()
        .iter()
        .map(indexed_points)
        .collect();
    normalize_signature(MeshSignature {
        vertices,
        edges,
        faces,
    })
}

fn assert_signatures_match(actual: &MeshSignature, expected: &MeshSignature, operation: &str) {
    assert_eq!(
        actual.vertices.len(),
        expected.vertices.len(),
        "{operation} vertex count"
    );
    assert_eq!(
        actual.edges.len(),
        expected.edges.len(),
        "{operation} edge count"
    );
    assert_eq!(
        actual.faces.len(),
        expected.faces.len(),
        "{operation} face count"
    );
    let close = |left: &[f64; 3], right: &[f64; 3]| {
        (0..3).all(|axis| (left[axis] - right[axis]).abs() <= 1.0e-10)
    };
    for (left, right) in actual.vertices.iter().zip(&expected.vertices) {
        assert!(
            close(left, right),
            "{operation} vertex positions: {left:?} != {right:?}"
        );
    }
    for (left, right) in actual.edges.iter().zip(&expected.edges) {
        assert!(
            close(&left[0], &right[0]) && close(&left[1], &right[1]),
            "{operation} edge endpoints: {left:?} != {right:?}"
        );
    }
    for (left, right) in actual.faces.iter().zip(&expected.faces) {
        assert_eq!(left.len(), right.len(), "{operation} polygon degree");
        for (left, right) in left.iter().zip(right) {
            assert!(
                close(left, right),
                "{operation} face positions: {left:?} != {right:?}"
            );
        }
    }
}

fn blender_mesh_outputs() -> Option<HashMap<String, Value>> {
    let executable = blender_file::blender_executable()?;
    let version = Command::new(&executable).arg("--version").output().ok()?;
    let version_text = String::from_utf8_lossy(&version.stdout);
    if !version_text.contains("Blender 5.2.2") {
        return None;
    }

    let directory = tempdir().unwrap();
    let output_path = directory.path().join("mesh-signatures.json");
    let output_literal = serde_json::to_string(output_path.to_str().unwrap()).unwrap();
    let script = format!(
        r"
import bpy, bmesh, json

def reset():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def capture(obj):
    mesh = obj.data
    return {{
        'vertices': [list(vertex.co) for vertex in mesh.vertices],
        'edges': [list(edge.vertices) for edge in mesh.edges],
        'faces': [list(face.vertices) for face in mesh.polygons],
    }}

results = {{}}
reset()
bpy.ops.mesh.primitive_plane_add(size=2)
obj = bpy.context.object
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.subdivide(number_cuts=1)
bpy.ops.object.mode_set(mode='OBJECT')
results['subdivide'] = capture(obj)

reset()
bpy.ops.mesh.primitive_cube_add(size=2)
obj = bpy.context.object
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type='FACE')
bm = bmesh.from_edit_mesh(obj.data)
face = next(face for face in bm.faces if sum(vertex.co.z for vertex in face.verts) / len(face.verts) < -0.99)
face.select_set(True)
bpy.ops.mesh.split()
bpy.ops.object.mode_set(mode='OBJECT')
results['split'] = capture(obj)

reset()
mesh = bpy.data.meshes.new('two-triangles')
mesh.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)], [], [(0,1,2),(0,2,3)])
obj = bpy.data.objects.new('two-triangles', mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type='EDGE')
bm = bmesh.from_edit_mesh(obj.data)
diagonal = next(edge for edge in bm.edges if {{vertex.index for vertex in edge.verts}} == {{0,2}})
diagonal.select_set(True)
bpy.ops.mesh.dissolve_edges(use_verts=False)
bpy.ops.object.mode_set(mode='OBJECT')
results['dissolve'] = capture(obj)

reset()
mesh = bpy.data.meshes.new('loose-square')
mesh.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)], [(0,1),(1,2),(2,3),(3,0)], [])
obj = bpy.data.objects.new('loose-square', mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type='EDGE')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.edge_face_add()
bpy.ops.object.mode_set(mode='OBJECT')
results['fill'] = capture(obj)

reset()
mesh = bpy.data.meshes.new('loose-points')
mesh.from_pydata([(-1,0,0),(1,0,0),(0,2,0)], [], [])
obj = bpy.data.objects.new('loose-points', mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type='VERT')
bm = bmesh.from_edit_mesh(obj.data)
bm.verts.ensure_lookup_table()
bm.verts[0].select_set(True)
bm.verts[1].select_set(True)
bpy.ops.mesh.merge(type='CENTER')
bpy.ops.object.mode_set(mode='OBJECT')
results['merge'] = capture(obj)

with open({output_literal}, 'w', encoding='utf-8') as handle:
    json.dump(results, handle)
"
    );
    let script_path = directory.path().join("mesh_reference.py");
    fs::write(&script_path, script).unwrap();
    let output = Command::new(executable)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Blender mesh reference failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let results: Value = serde_json::from_slice(&fs::read(output_path).unwrap()).unwrap();
    Some(
        results
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, signature)| (name.clone(), signature.clone()))
            .collect(),
    )
}

#[test]
fn topology_edits_match_blender_522_when_available() {
    let Some(blender) = blender_mesh_outputs() else {
        return;
    };

    let mut subdivide = Mesh::plane(PlaneParams::default()).unwrap();
    edit::apply(
        &mut subdivide,
        "subdivide",
        &json!({"elements":{"domain":"face","ids":["f0"]},"cuts":1}),
    )
    .unwrap();
    assert_signatures_match(
        &mesh_signature(&subdivide),
        &blender_signature(&blender["subdivide"]),
        "subdivide",
    );

    let mut split = Mesh::box_mesh(BoxParams::default()).unwrap();
    edit::apply(
        &mut split,
        "split",
        &json!({"elements":{"domain":"face","ids":["f0"]}}),
    )
    .unwrap();
    assert_signatures_match(
        &mesh_signature(&split),
        &blender_signature(&blender["split"]),
        "split",
    );

    let mut dissolve = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-1.0, -1.0, 0.0),
            DVec3::new(1.0, -1.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(-1.0, 1.0, 0.0),
        ],
        vec![vec![0, 1, 2], vec![0, 2, 3]],
    )
    .unwrap();
    let diagonal = dissolve
        .edges
        .iter()
        .find(|edge| edge.vertices.contains(&0) && edge.vertices.contains(&2))
        .unwrap()
        .id;
    edit::apply(
        &mut dissolve,
        "dissolve",
        &json!({"elements":{"domain":"edge","ids":[format!("e{diagonal}")]}}),
    )
    .unwrap();
    assert_signatures_match(
        &mesh_signature(&dissolve),
        &blender_signature(&blender["dissolve"]),
        "dissolve",
    );
}
