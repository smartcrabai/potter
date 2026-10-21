#![expect(
    clippy::unwrap_used,
    reason = "integration tests use concise assertions"
)]

use glam::{DMat4, DQuat, DVec3, EulerRot};
use potter_core::error::ErrorCode;
use potter_core::geom::{
    BoxParams, CylinderParams, Mesh, boolean::BooleanOperand, modifiers::evaluate_modifiers,
};
use potter_core::model::{Id, Modifier};
use proptest::prelude::*;
use serde_json::{Value, json};

fn modifier(modifier_type: &str, params: Value) -> Modifier {
    Modifier {
        id: Id::new("modifier_test").unwrap(),
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: serde_json::from_value(params).unwrap(),
        binding_data: None,
        baked_ocean_cache: None,
        runtime: potter_core::model::ModifierRuntime::default(),
    }
}

/// Evaluates a FLOAT Boolean modifier with its scene-resolved mesh operands.
fn float_boolean(subject: &Mesh, operands: &[Mesh], operation: &str) -> Mesh {
    let mut boolean = modifier(
        "boolean",
        json!({"solver": "FLOAT", "operation": operation}),
    );
    boolean.runtime.boolean_operands = operands
        .iter()
        .cloned()
        .map(|mesh| BooleanOperand {
            mesh,
            to_subject: DMat4::IDENTITY,
            float_flip: None,
            material_remap: Vec::new(),
        })
        .collect();
    evaluate_modifiers(subject, &[boolean], "TestMesh").unwrap()
}

fn signed_volume(mesh: &Mesh) -> f64 {
    let vertices: std::collections::HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    mesh.faces
        .iter()
        .map(|face| {
            let points: Vec<_> = face.vertices.iter().map(|id| vertices[id]).collect();
            (1..points.len() - 1)
                .map(|index| points[0].dot(points[index].cross(points[index + 1])) / 6.0)
                .sum::<f64>()
        })
        .sum::<f64>()
}

fn edge_face_counts(mesh: &Mesh) -> std::collections::HashMap<[u32; 2], usize> {
    let mut counts = std::collections::HashMap::new();
    for face in &mesh.faces {
        for index in 0..face.vertices.len() {
            let first = face.vertices[index];
            let second = face.vertices[(index + 1) % face.vertices.len()];
            *counts
                .entry([first.min(second), first.max(second)])
                .or_default() += 1;
        }
    }
    counts
}
type PositionKey = [u64; 3];
type CanonicalGeometry = (
    Vec<PositionKey>,
    Vec<[PositionKey; 2]>,
    Vec<Vec<PositionKey>>,
);

fn point_key(point: DVec3) -> PositionKey {
    let key = |value: f64| if value == 0.0 { 0 } else { value.to_bits() };
    [key(point.x), key(point.y), key(point.z)]
}

fn canonical_face(points: &[PositionKey]) -> Vec<PositionKey> {
    let mut canonical = points.to_vec();
    for offset in 1..points.len() {
        let candidate = points
            .iter()
            .cycle()
            .skip(offset)
            .take(points.len())
            .copied()
            .collect::<Vec<_>>();
        if candidate < canonical {
            canonical = candidate;
        }
    }
    canonical
}

fn canonical_geometry(mesh: &Mesh) -> CanonicalGeometry {
    let point_by_id: std::collections::HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, point_key(vertex.co)))
        .collect();
    let mut positions: Vec<_> = point_by_id.values().copied().collect();
    positions.sort_unstable();
    let mut edges: Vec<_> = mesh
        .edges
        .iter()
        .map(|edge| {
            let mut endpoints = [
                point_by_id[&edge.vertices[0]],
                point_by_id[&edge.vertices[1]],
            ];
            endpoints.sort_unstable();
            endpoints
        })
        .collect();
    edges.sort_unstable();
    let mut faces: Vec<_> = mesh
        .faces
        .iter()
        .map(|face| {
            let points: Vec<_> = face.vertices.iter().map(|id| point_by_id[id]).collect();
            canonical_face(&points)
        })
        .collect();
    faces.sort_unstable();
    (positions, edges, faces)
}

#[test]
fn boolean_float_without_scene_operand_is_identity() {
    let source = Mesh::box_mesh(BoxParams::default()).unwrap();
    let result = evaluate_modifiers(
        &source,
        &[modifier(
            "boolean",
            json!({"solver": "FLOAT", "operation": "DIFFERENCE"}),
        )],
        "TestMesh",
    )
    .unwrap();
    assert_eq!(canonical_geometry(&result), canonical_geometry(&source));
}

#[test]
fn boolean_float_difference_matches_blender_offset_box_parity() {
    let source = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(-1.0, -1.0, -1.0),
            DVec3::new(1.0, -1.0, -1.0),
            DVec3::new(1.0, 1.0, -1.0),
            DVec3::new(-1.0, 1.0, -1.0),
            DVec3::new(-1.0, -1.0, 1.0),
            DVec3::new(1.0, -1.0, 1.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(-1.0, 1.0, 1.0),
        ],
        vec![
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![2, 3, 7, 6],
            vec![0, 4, 7, 3],
            vec![1, 2, 6, 5],
        ],
    )
    .unwrap();
    let mut operand = source.clone();
    for vertex in &mut operand.vertices {
        vertex.co += DVec3::new(1.5, 0.5, 1.0);
    }
    let result = float_boolean(&source, &[operand], "DIFFERENCE");
    assert_eq!(result.vertices.len(), 14);
    assert_eq!(result.edges.len(), 21);
    assert_eq!(result.faces.len(), 9);
    assert!((signed_volume(&result) - 7.25).abs() <= 1.0e-5);
    assert!(result.validate().is_ok());
}

#[test]
fn boolean_float_empty_material_remap_preserves_source_indices() {
    let subject = Mesh::box_mesh(BoxParams::default()).unwrap();
    let mut operand = subject.clone();
    for vertex in &mut operand.vertices {
        vertex.co += DVec3::new(0.0, 10.0, 0.0);
    }
    for face in &mut operand.faces {
        face.material_index = 1;
    }
    let operand_faces = operand.faces.len();

    let result = float_boolean(&subject, &[operand], "UNION");

    assert_eq!(
        result
            .faces
            .iter()
            .filter(|face| face.material_index == 1)
            .count(),
        operand_faces
    );
}

#[test]
fn boolean_float_difference_of_box_and_tilted_cylinder_has_closed_bore() {
    let source = Mesh::box_mesh(BoxParams {
        size: DVec3::splat(1.8),
    })
    .unwrap();
    let mut operand = Mesh::cylinder(CylinderParams {
        vertices: 40,
        radius: 0.39,
        depth: 2.8,
        ..CylinderParams::default()
    })
    .unwrap();
    let rotation = DQuat::from_euler(
        EulerRot::XYZ,
        16.0_f64.to_radians(),
        8.0_f64.to_radians(),
        17.0_f64.to_radians(),
    );
    for vertex in &mut operand.vertices {
        vertex.co = rotation * vertex.co;
    }

    let result = float_boolean(&source, &[operand], "DIFFERENCE");
    let volume = signed_volume(&result);
    assert!(result.validate().is_ok());
    assert!(volume > 0.0 && volume < signed_volume(&source) - 0.1);
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    #[test]
    fn boolean_float_union_of_overlapping_boxes_is_watertight(
        x in -0.75_f64..0.75,
        y in -0.75_f64..0.75,
        z in -0.75_f64..0.75,
        size in 0.25_f64..1.5,
    ) {
        let source = Mesh::box_mesh(BoxParams::default()).unwrap();
        let mut operand = Mesh::box_mesh(BoxParams { size: DVec3::splat(size) }).unwrap();
        for vertex in &mut operand.vertices {
            vertex.co += DVec3::new(x, y, z);
        }
        let result = float_boolean(&source, &[operand], "UNION");
        prop_assert!(result.validate().is_ok());
        prop_assert!(edge_face_counts(&result).values().all(|count| *count == 2));
        prop_assert!((signed_volume(&result) - (8.0 + size.powi(3) - intersection_volume(size, x, y, z))).abs() < 2.0e-5);
    }
}

fn intersection_volume(size: f64, x: f64, y: f64, z: f64) -> f64 {
    [x, y, z]
        .into_iter()
        .map(|offset| {
            let lower = (offset - size * 0.5).max(-1.0);
            let upper = (offset + size * 0.5).min(1.0);
            (upper - lower).max(0.0)
        })
        .product()
}

#[test]
fn screw_revolves_a_profile_into_a_closed_cylinder() {
    let mut line = Mesh::new();
    let first = line.insert_vertex(DVec3::new(1.0, 0.0, -1.0)).unwrap();
    let second = line.insert_vertex(DVec3::new(1.0, 0.0, 1.0)).unwrap();
    line.insert_edge([first, second]).unwrap();
    let result = evaluate_modifiers(
        &line,
        &[modifier(
            "screw",
            json!({
                "steps": 16,
                "angle": std::f64::consts::TAU,
                "axis": "Z",
            }),
        )],
        "ScrewLine",
    )
    .unwrap();
    assert_eq!(result.vertices.len(), 32);
    assert_eq!(result.faces.len(), 16);
    assert!(result.validate().is_ok());
}

#[test]
fn simple_deform_zero_angle_is_identity() {
    let source = Mesh::box_mesh(BoxParams::default()).unwrap();
    let result = evaluate_modifiers(
        &source,
        &[modifier(
            "simple_deform",
            json!({
                "deform_method": "TWIST",
                "angle": 0.0,
            }),
        )],
        "DeformBox",
    )
    .unwrap();
    assert_eq!(result, source);
}

/// A cube whose vertex 0 is moved to (-2,-1,-1), so the vertices are not equidistant from the origin.
fn cast_uneven_cube() -> Mesh {
    let positions = vec![
        DVec3::new(-2.0, -1.0, -1.0),
        DVec3::new(1.0, -1.0, -1.0),
        DVec3::new(1.0, 1.0, -1.0),
        DVec3::new(-1.0, 1.0, -1.0),
        DVec3::new(-1.0, -1.0, 1.0),
        DVec3::new(1.0, -1.0, 1.0),
        DVec3::new(1.0, 1.0, 1.0),
        DVec3::new(-1.0, 1.0, 1.0),
    ];
    Mesh::from_positions_and_faces(positions, cast_cube_faces()).unwrap()
}

fn cast_cube_faces() -> Vec<Vec<usize>> {
    vec![
        vec![0, 3, 2, 1],
        vec![4, 5, 6, 7],
        vec![0, 1, 5, 4],
        vec![2, 3, 7, 6],
        vec![1, 2, 6, 5],
        vec![0, 4, 7, 3],
    ]
}

fn assert_cast_positions(mesh: &Mesh, expected: [[f64; 3]; 8], label: &str) {
    for (index, (vertex, wanted)) in mesh.vertices.iter().zip(expected).enumerate() {
        for (axis, (actual, wanted)) in vertex.co.to_array().into_iter().zip(wanted).enumerate() {
            assert!(
                (actual - wanted).abs() < 1.0e-5,
                "{label}: vertex {index} axis {axis} is {actual}, Blender has {wanted}"
            );
        }
    }
}

/// Blender 5.2.2 oracle (Cast factor 0.5) for the uneven cube above and the all-zero cube.
#[test]
fn cast_defaults_and_limits_match_blender_on_uneven_and_zero_geometry() {
    const UNEVEN_SPHERE_DEFAULT: [[f64; 3]; 8] = [
        [-1.743_718, -0.871_859, -0.871_859],
        [1.025_888, -1.025_888, -1.025_888],
        [1.025_888, 1.025_888, -1.025_888],
        [-1.025_888, 1.025_888, -1.025_888],
        [-1.025_888, -1.025_888, 1.025_888],
        [1.025_888, -1.025_888, 1.025_888],
        [1.025_888, 1.025_888, 1.025_888],
        [-1.025_888, 1.025_888, 1.025_888],
    ];
    const UNEVEN_SOURCE: [[f64; 3]; 8] = [
        [-2.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    const UNEVEN_CUBOID_SIZE: [[f64; 3]; 8] = [
        [-1.75, -0.875, -0.875],
        [1.25, -1.25, -1.25],
        [1.25, 1.25, -1.25],
        [-1.25, 1.25, -1.25],
        [-1.25, -1.25, 1.25],
        [1.25, -1.25, 1.25],
        [1.25, 1.25, 1.25],
        [-1.25, 1.25, 1.25],
    ];
    const UNEVEN_CYLINDER_DEFAULT: [[f64; 3]; 8] = [
        [-1.814_703, -0.907_351, -1.0],
        [1.144_079, -1.144_079, -1.0],
        [1.144_079, 1.144_079, -1.0],
        [-1.144_079, 1.144_079, -1.0],
        [-1.144_079, -1.144_079, 1.0],
        [1.144_079, -1.144_079, 1.0],
        [1.144_079, 1.144_079, 1.0],
        [-1.144_079, 1.144_079, 1.0],
    ];
    let uneven = cast_uneven_cube();
    // `{}` is the schema default: SPHERE, factor 0.5, radius 0, size 0, use_radius_as_size true.
    let sphere =
        evaluate_modifiers(&uneven, &[modifier("cast", json!({}))], "CastFixture").unwrap();
    assert_cast_positions(&sphere, UNEVEN_SPHERE_DEFAULT, "uneven sphere default");
    let cuboid = evaluate_modifiers(
        &uneven,
        &[modifier(
            "cast",
            json!({"cast_type": "CUBOID", "use_radius_as_size": false, "size": 1.5}),
        )],
        "CastFixture",
    )
    .unwrap();
    assert_cast_positions(&cuboid, UNEVEN_CUBOID_SIZE, "uneven cuboid size 1.5");
    let cylinder = evaluate_modifiers(
        &uneven,
        &[modifier("cast", json!({"cast_type": "CYLINDER"}))],
        "CastFixture",
    )
    .unwrap();
    assert_cast_positions(
        &cylinder,
        UNEVEN_CYLINDER_DEFAULT,
        "uneven cylinder default",
    );
    // Every vertex is farther than 1.2 from the origin, so the radius limit leaves all of them.
    let limited = evaluate_modifiers(
        &uneven,
        &[modifier("cast", json!({"factor": 0.5, "radius": 1.2}))],
        "CastFixture",
    )
    .unwrap();
    assert_cast_positions(&limited, UNEVEN_SOURCE, "uneven radius limit");
    // All-zero geometry: the automatic sphere/cylinder size is 10 and the cuboid bounds are zero.
    let zero = Mesh::from_positions_and_faces(vec![DVec3::ZERO; 8], cast_cube_faces()).unwrap();
    for cast_type in ["SPHERE", "CUBOID", "CYLINDER"] {
        let result = evaluate_modifiers(
            &zero,
            &[modifier("cast", json!({"cast_type": cast_type}))],
            "ZeroCastFixture",
        )
        .unwrap();
        assert_cast_positions(&result, [[0.0; 3]; 8], cast_type);
    }
}

#[test]
fn modifier_parameter_errors_include_a_json_pointer() {
    let source = Mesh::box_mesh(BoxParams::default()).unwrap();
    let error = evaluate_modifiers(
        &source,
        &[modifier("simple_deform", json!({"angle": "invalid"}))],
        "DeformBox",
    )
    .unwrap_err();
    assert_eq!(error.details["pointer"], "/params/angle");
}

#[test]
fn mask_drops_faces_that_use_vertices_outside_the_group() {
    let mut source = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::ONE, DVec3::Y],
        vec![vec![0, 1, 2, 3]],
    )
    .unwrap();
    source.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"handle":{"v0":1.0,"v1":1.0,"v2":1.0,"v3":0.0}}),
    );
    let result = evaluate_modifiers(
        &source,
        &[modifier("mask", json!({"vertex_group":"handle"}))],
        "MaskQuad",
    )
    .unwrap();
    assert!(result.faces.is_empty(), "{:?}", result.faces);
}

#[test]
fn boolean_modifier_apply_resolves_scene_object_operand() -> Result<(), Box<dyn std::error::Error>>
{
    use std::{fs, process::Command};

    use tempfile::tempdir;

    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success());
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"body","kind":"box","params":{"size":2}},
                {"op":"node.create","id":"cutter","kind":"box","params":{"size":1}},
                {"op":"modifier.create","target":{"id":"body"},"id":"boolean_one","type":"boolean",
                 "params":{"object":"cutter","operation":"DIFFERENCE"}},
                {"op":"modifier.apply","target":{"id":"body"},"modifier_id":"boolean_one"}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let inspected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("body")
        .arg("--json")
        .output()?;
    assert!(inspected.status.success());
    let envelope: Value = serde_json::from_slice(&inspected.stdout)?;
    let item = &envelope["result"]["items"][0];
    assert_eq!(item["modifiers"], json!([]));
    assert!(
        item["evaluated_geometry"]["face_count"]
            .as_u64()
            .unwrap_or_default()
            > 6
    );
    Ok(())
}

#[test]
fn multires_levels_are_bounded_by_the_stored_top_level_and_budget() {
    let source = Mesh::box_mesh(BoxParams::default()).unwrap();
    let clamped = evaluate_modifiers(
        &source,
        &[modifier(
            "multires",
            json!({"levels": 6, "total_levels": 1}),
        )],
        "TestObject",
    )
    .unwrap();
    assert_eq!(clamped.faces.len(), 24);
    assert_eq!(clamped.vertices.len(), 26);
    let multires_level_six = evaluate_modifiers(
        &source,
        &[modifier(
            "multires",
            json!({"levels": 6, "total_levels": 6}),
        )],
        "TestObject",
    )
    .unwrap();
    assert_eq!(multires_level_six.faces.len(), 24_576);
    assert_eq!(multires_level_six.vertices.len(), 24_578);
    let resource_limited = evaluate_modifiers(
        &source,
        &[modifier(
            "multires",
            json!({"levels": 11, "total_levels": 11}),
        )],
        "TestObject",
    )
    .unwrap_err();
    assert_eq!(resource_limited.code, ErrorCode::LimitExceeded);
}

#[test]
fn custom_normals_follow_blender_per_modifier_type() {
    let source = Mesh::box_mesh(BoxParams::default()).unwrap();
    let mut sharp_only = source.clone();
    sharp_only.attributes.insert(
        "blender_edge_flags".to_owned(),
        Value::Array(
            (0..sharp_only.edges.len())
                .map(|_| json!({"sharp": true, "seam": false}))
                .collect(),
        ),
    );

    let mut mixed_sharp = source.clone();
    let mut edge_flags = (0..mixed_sharp.edges.len())
        .map(|_| json!({"sharp": false, "seam": false}))
        .collect::<Vec<_>>();
    edge_flags[0] = json!({"sharp": true, "seam": false});
    mixed_sharp
        .attributes
        .insert("blender_edge_flags".to_owned(), Value::Array(edge_flags));
    let mixed_result = evaluate_modifiers(
        &mixed_sharp,
        &[modifier(
            "multires",
            json!({"levels": 1, "use_custom_normals": true}),
        )],
        "TestObject",
    )
    .unwrap();
    // Blender 5.2.2 (tests/fixtures/custom_normal_presence): Multires with Use Custom Normals on writes a layer
    // only when the input has one, so sharp edges alone leave it out.
    assert!(
        !mixed_result.attributes.contains_key("custom_normal"),
        "multires with mixed sharp inputs and no custom-normal layer writes no custom normals, as Blender does"
    );
    // Blender 5.2.2: a Subdivision Surface with Use Custom Normals on writes a layer even with no input layer.
    let subdivision_mixed = evaluate_modifiers(
        &mixed_sharp,
        &[modifier(
            "subdivision",
            json!({"levels": 1, "use_custom_normals": true}),
        )],
        "TestObject",
    )
    .unwrap();
    assert!(
        subdivision_mixed.attributes.contains_key("custom_normal"),
        "subdivision with Use Custom Normals on writes custom normals, as Blender does"
    );
    let subdivision_off = evaluate_modifiers(
        &mixed_sharp,
        &[modifier(
            "subdivision",
            json!({"levels": 1, "use_custom_normals": false}),
        )],
        "TestObject",
    )
    .unwrap();
    assert!(!subdivision_off.attributes.contains_key("custom_normal"));
    let sharp_result = evaluate_modifiers(
        &sharp_only,
        &[modifier(
            "multires",
            json!({"levels": 1, "use_custom_normals": true}),
        )],
        "TestObject",
    )
    .unwrap();
    assert!(!sharp_result.attributes.contains_key("custom_normal"));

    let mut with_layer = source;
    let values = with_layer
        .faces
        .iter()
        .map(|face| {
            (
                format!("f{}", face.id),
                json!(vec![[8192_i32, 0]; face.vertices.len()]),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    with_layer.attributes.insert(
        "custom_normal".to_owned(),
        json!({"domain": "corner", "type": "normal_code", "values": values}),
    );
    let enabled = evaluate_modifiers(
        &with_layer,
        &[modifier(
            "multires",
            json!({"levels": 1, "use_custom_normals": true}),
        )],
        "TestObject",
    )
    .unwrap();
    assert!(enabled.attributes.contains_key("custom_normal"));
    let disabled = evaluate_modifiers(
        &with_layer,
        &[modifier(
            "multires",
            json!({"levels": 1, "use_custom_normals": false}),
        )],
        "TestObject",
    )
    .unwrap();
    assert!(!disabled.attributes.contains_key("custom_normal"));
}
#[test]
fn skin_modifier_builds_a_closed_profile_around_skeleton_edges() {
    let mut line = Mesh::new();
    let first = line.insert_vertex(DVec3::ZERO).unwrap();
    let second = line.insert_vertex(DVec3::Z).unwrap();
    line.insert_edge([first, second]).unwrap();
    let result = evaluate_modifiers(&line, &[modifier("skin", json!({}))], "SkinSkeleton").unwrap();
    assert_eq!(result.vertices.len(), 16);
    assert_eq!(result.faces.len(), 14);
    assert!(edge_face_counts(&result).values().all(|count| *count == 2));
}
#[test]
fn numeric_identity_settings_preserve_mesh_topology_and_positions() {
    let source = Mesh::box_mesh(BoxParams::default()).unwrap();
    let source_geometry = canonical_geometry(&source);
    let cases = [
        ("displace", json!({"strength":0.0})),
        ("smooth", json!({"factor":0.0})),
        ("simple_deform", json!({"angle":0.0})),
        ("cast", json!({"factor":0.0})),
        ("wave", json!({"height":0.0})),
        ("warp", json!({"strength":0.0})),
        ("laplacian_smooth", json!({"iterations":0})),
        ("corrective_smooth", json!({"iterations":0})),
        (
            "hook",
            json!({"target_position":[1.0,0.0,0.0],"strength":0.0}),
        ),
    ];

    for (modifier_type, params) in cases {
        let result =
            evaluate_modifiers(&source, &[modifier(modifier_type, params)], "IdentityCube")
                .unwrap();
        assert_eq!(
            canonical_geometry(&result),
            source_geometry,
            "{modifier_type}"
        );
    }
}

#[test]
fn mask_threshold_uses_a_strict_weight_cutoff() {
    let mut source =
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])
            .unwrap();
    source.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"region":{"v0":0.75,"v1":0.75,"v2":0.75}}),
    );
    let retained = evaluate_modifiers(
        &source,
        &[modifier(
            "mask",
            json!({"vertex_group":"region","threshold":0.5}),
        )],
        "MaskTriangle",
    )
    .unwrap();
    let excluded = evaluate_modifiers(
        &source,
        &[modifier(
            "mask",
            json!({"vertex_group":"region","threshold":0.75}),
        )],
        "MaskTriangle",
    )
    .unwrap();
    assert_eq!(retained.faces.len(), 1);
    assert!(excluded.faces.is_empty(), "{:?}", excluded.faces);
}

#[test]
fn mirror_merge_threshold_merges_vertices_within_tolerance() {
    let source = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(0.0004, 0.0, 0.0),
            DVec3::X,
            DVec3::new(0.5, 1.0, 0.0),
        ],
        vec![vec![0, 1, 2]],
    )
    .unwrap();
    let merged = evaluate_modifiers(
        &source,
        &[modifier(
            "mirror",
            json!({"use_axis":[true,false,false],"use_mirror_merge":true,"merge_threshold":0.001}),
        )],
        "MirrorTriangle",
    )
    .unwrap();

    let outside = evaluate_modifiers(
        &source,
        &[modifier(
            "mirror",
            json!({"use_axis":[true,false,false],"use_mirror_merge":true,"merge_threshold":0.0005}),
        )],
        "MirrorTriangle",
    )
    .unwrap();
    assert_eq!(outside.vertices.len(), 6);
    let separate = evaluate_modifiers(
        &source,
        &[modifier(
            "mirror",
            json!({"use_axis":[true,false,false],"use_mirror_merge":false,"merge_threshold":0.001}),
        )],
        "MirrorTriangle",
    )
    .unwrap();
    assert_eq!(merged.vertices.len(), 5);
    assert_eq!(merged.faces.len(), 2);
    assert_eq!(separate.vertices.len(), 6);
}

#[test]
fn skin_vertex_radii_set_cross_section_bounds() {
    let mut line = Mesh::new();
    let first = line.insert_vertex(DVec3::ZERO).unwrap();
    let second = line.insert_vertex(DVec3::Z).unwrap();
    line.insert_edge([first, second]).unwrap();
    line.attributes.insert(
        "skin_radii".to_owned(),
        json!({
            "0":[0.5,0.25],
            "1":[0.5,0.25]
        }),
    );
    // Blender builds a skin only from a skeleton with a root vertex; without one the input is kept
    // (tests/pr18f_skin.rs, skeleton_without_root_reports_blender_modifier_warning_and_keeps_input).
    line.attributes
        .insert("skin_roots".to_owned(), json!({"0": true, "1": false}));
    let result = evaluate_modifiers(&line, &[modifier("skin", json!({}))], "SkinSkeleton").unwrap();
    let bounds = result.bounds().unwrap();
    assert!((bounds.min.x + 0.5).abs() < 1.0e-12);
    assert!((bounds.max.x - 0.5).abs() < 1.0e-12);
    assert!((bounds.min.y + 0.25).abs() < 1.0e-12);
    assert!((bounds.max.y - 0.25).abs() < 1.0e-12);
}

#[test]
fn simple_deform_methods_match_blender_vertex_positions() {
    let mut source = Mesh::new();
    let origin = source.insert_vertex(DVec3::ZERO).unwrap();
    let end = source.insert_vertex(DVec3::X).unwrap();
    let corner = source.insert_vertex(DVec3::new(0.5, 1.0, 0.0)).unwrap();
    let depth = source.insert_vertex(DVec3::new(0.5, 0.0, 1.0)).unwrap();
    for vertex in [end, corner, depth] {
        source.insert_edge([origin, vertex]).unwrap();
    }
    let cases = [
        (
            "TWIST",
            json!({"deform_method":"TWIST","angle":std::f64::consts::FRAC_PI_2}),
            [
                DVec3::ZERO,
                DVec3::X,
                DVec3::new(
                    0.5,
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                ),
                DVec3::new(
                    0.5,
                    -std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                ),
            ],
        ),
        (
            "BEND",
            json!({"deform_method":"BEND","angle":std::f64::consts::FRAC_PI_2}),
            [
                DVec3::ZERO,
                DVec3::X,
                DVec3::new(0.5, 1.0, 0.0),
                DVec3::new(0.5, 2.0 / std::f64::consts::PI, 2.0 / std::f64::consts::PI),
            ],
        ),
        (
            "TAPER",
            json!({"deform_method":"TAPER","factor":1.0}),
            [
                DVec3::ZERO,
                DVec3::X,
                DVec3::new(0.5, 1.5, 0.0),
                DVec3::new(0.5, 0.0, 1.5),
            ],
        ),
        (
            "STRETCH",
            json!({"deform_method":"STRETCH","factor":1.0}),
            [
                DVec3::ZERO,
                DVec3::X * 2.0,
                DVec3::new(1.0, 0.25, 0.0),
                DVec3::new(1.0, 0.0, 0.25),
            ],
        ),
    ];

    for (method, params, expected) in cases {
        let result = evaluate_modifiers(
            &source,
            &[modifier("simple_deform", params)],
            "SimpleDeformFixture",
        )
        .unwrap();
        for (vertex_id, position) in [
            (origin, expected[0]),
            (end, expected[1]),
            (corner, expected[2]),
            (depth, expected[3]),
        ] {
            assert!(
                result.vertex(vertex_id).unwrap().co.distance(position) < 1.0e-5,
                "{method} at vertex {vertex_id}"
            );
        }
    }
}

#[test]
fn screw_offset_extends_the_profile_along_its_axis() {
    let mut line = Mesh::new();
    let first = line.insert_vertex(DVec3::new(1.0, 0.0, -1.0)).unwrap();
    let second = line.insert_vertex(DVec3::new(1.0, 0.0, 1.0)).unwrap();
    line.insert_edge([first, second]).unwrap();
    let result = evaluate_modifiers(
        &line,
        &[modifier(
            "screw",
            json!({
                "steps":4,
                "angle":std::f64::consts::PI,
                "screw_offset":2.0,
                "axis":"Z"
            }),
        )],
        "ScrewLine",
    )
    .unwrap();
    assert_eq!(result.vertices.len(), 10);
    assert_eq!(result.faces.len(), 4);
    assert!((result.bounds().unwrap().max.z - 3.0).abs() < 1.0e-12);
}
