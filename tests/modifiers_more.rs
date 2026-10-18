#![expect(
    clippy::unwrap_used,
    reason = "integration tests use concise assertions"
)]

use glam::DVec3;
use potter_core::geom::{BoxParams, Mesh, modifiers::evaluate_modifiers};
use potter_core::model::{Id, Modifier};
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
        runtime: potter_core::model::ModifierRuntime::default(),
    }
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
