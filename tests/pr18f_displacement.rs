#![expect(
    clippy::unwrap_used,
    reason = "integration tests use concise assertions"
)]

//! Displace, Wave, and Smooth regressions against Blender 5.2.2.
//!
//! The expected coordinates come from running Blender's `MOD_displace`, `MOD_wave`, and `MOD_smooth`
//! headless (`Blender -b --factory-startup`) on the seven-vertex input below: two quads' worth of
//! asymmetric geometry, an isolated vertex (index 6), and vertex group `G` with weights on vertices
//! 0 through 5 (vertex 6 is outside `G`). Blender stores float32, so the tolerance allows its
//! rounding. Each chained case applies its first modifier to the input the second one reads.

use std::collections::BTreeMap;

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, EvaluationPurpose, Snapshot},
    geom::{
        Mesh,
        modifiers::{evaluate_modifiers, evaluate_modifiers_with_groups_at_frame},
    },
    model::{Id, Modifier, ModifierRuntime, SceneDoc, VertexGroup},
    ops::apply_batch,
};
use serde_json::{Map, Value, json};

/// Blender's float32 rounding; the implementation computes in f64 with the same algorithm.
const TOLERANCE: f64 = 2.0e-5;

const POSITIONS: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.1],
    [0.0, 1.0, 0.2],
    [2.0, 1.5, 0.3],
    [1.0, -1.0, 0.5],
    [0.5, 0.5, -0.7],
    [-1.0, 0.5, 0.0],
];
const FACES: [&[usize]; 3] = [&[0, 1, 3, 2], &[0, 4, 1], &[1, 5, 3]];

const DISP_X: [[f64; 3]; 7] = [
    [0.4, 0.0, 0.0],
    [2.4, 0.0, 0.1],
    [0.4, 1.0, 0.2],
    [2.4, 1.5, 0.3],
    [1.4, -1.0, 0.5],
    [0.9, 0.5, -0.7],
    [-0.6, 0.5, 0.0],
];
const DISP_NORMAL: [[f64; 3]; 7] = [
    [-0.016414464, 0.017582612, 0.449356705],
    [2.198567152, 0.085076153, 0.494757116],
    [-0.013324607, 0.928935409, 0.644153476],
    [2.316853285, 1.434091926, 0.612666249],
    [0.979503036, -0.815527558, 0.909938812],
    [0.723639488, 0.551609159, -1.087068319],
    [-1.402492285, 0.701246142, 0.0],
];
const DISP_NORMAL_GROUP: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.0],
    [2.220630169, 0.094529055, 0.538619041],
    [-0.007402559, 0.960519671, 0.446751922],
    [2.088014841, 1.481692195, 0.386851728],
    [0.982919216, -0.846272945, 0.841615677],
    [0.748488307, 0.557343483, -1.130075932],
    [-1.0, 0.5, 0.0],
];
const DISP_NORMAL_GROUP_INVERT: [[f64; 3]; 7] = [
    [-0.018238293, 0.019536234, 0.499285221],
    [2.0, 0.0, 0.1],
    [-0.007402559, 0.960519671, 0.446751922],
    [2.264044285, 1.445076585, 0.560555220],
    [0.994306386, -0.948757648, 0.613871872],
    [0.5, 0.5, -0.699999988],
    [-1.447213650, 0.723606825, 0.0],
];
const DISP_NORMAL_NEGATIVE_STRENGTH: [[f64; 3]; 7] = [
    [0.016414464, -0.017582610, -0.449356675],
    [1.801432967, -0.085076146, -0.294757098],
    [0.013324606, 1.071064591, -0.244153455],
    [1.683146715, 1.565908074, -0.012666196],
    [1.020496964, -1.184472442, 0.090061247],
    [0.276360542, 0.448390871, -0.312931687],
    [-0.597507775, 0.298753887, 0.0],
];
const DISP_Z_LOCAL_GROUP: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.7],
    [0.0, 1.0, 0.5],
    [2.0, 1.5, 0.450000018],
    [1.0, -1.0, 0.950000048],
    [0.5, 0.5, -0.099999964],
    [-1.0, 0.5, 0.0],
];
const CHAIN_WAVE_NORMAL_THEN_DISPLACE: [[f64; 3]; 7] = [
    [0.022964422, 0.003656663, 0.413499951],
    [2.182352304, 0.014779855, 0.289295197],
    [0.022200976, 0.846529782, 0.947293162],
    [2.207600832, 1.437441707, 0.424452156],
    [0.995894670, -0.779922068, 0.912266791],
    [0.983255625, 0.589327097, -1.373891711],
    [-1.615886688, 0.807943344, 0.0],
];
const WAVE_DEFAULT: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.431242645],
    [2.0, 0.0, 0.1],
    [0.0, 1.0, 0.337866604],
    [2.0, 1.5, 0.3],
    [1.0, -1.0, 0.520523250],
    [0.5, 0.5, -0.390704334],
    [-1.0, 0.5, 0.088603832],
];
const WAVE_FRAME7_LIFE_DAMP: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.003426778],
    [2.0, 0.0, 0.140380204],
    [0.0, 1.0, 0.326308072],
    [2.0, 1.5, 0.303426802],
    [1.0, -1.0, 0.636898398],
    [0.5, 0.5, -0.625473797],
    [-1.0, 0.5, 0.139892280],
];
const WAVE_FALLOFF_START: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.084879264],
    [2.0, 0.0, 0.279906869],
    [0.0, 1.0, 0.379906863],
    [2.0, 1.5, 0.3],
    [1.0, -1.0, 0.584879279],
    [0.5, 0.5, -0.530964077],
    [-1.0, 0.5, 0.112572186],
];
const WAVE_NORMAL_XYZ: [[f64; 3]; 7] = [
    [-0.005028903, 0.005386789, 0.137669533],
    [2.005162716, 0.002211933, 0.110263467],
    [-0.012769197, 0.931897640, 0.625639796],
    [2.0, 1.5, 0.3],
    [0.991704166, -0.925337434, 0.665916800],
    [0.745888948, 0.556743622, -1.125576973],
    [-1.326899886, 0.663449943, 0.0],
];
const WAVE_NORMAL_X_ONLY: [[f64; 3]; 7] = [
    [-0.005028903, 0.0, 0.0],
    [2.005162716, 0.0, 0.1],
    [-0.012769197, 1.0, 0.2],
    [2.0, 1.5, 0.3],
    [0.991704166, -1.0, 0.5],
    [0.745888948, 0.5, -0.7],
    [-1.326899886, 0.5, 0.0],
];
const WAVE_X_ONLY_NO_CYCLIC: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.145287365],
    [2.0, 0.0, 0.1],
    [0.0, 1.0, 0.345287383],
    [2.0, 1.5, 0.3],
    [1.0, -1.0, 0.645287395],
    [0.5, 0.5, -0.238652349],
    [-1.0, 0.5, 0.0],
];
const WAVE_GROUP_INVERT: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.563453078],
    [2.0, 0.0, 0.1],
    [0.0, 1.0, 0.481726527],
    [2.0, 1.5, 0.3],
    [1.0, -1.0, 0.536545277],
    [0.5, 0.5, -0.7],
    [-1.0, 0.5, 0.417076349],
];
const CHAIN_SMOOTH_THEN_WAVE: [[f64; 3]; 7] = [
    [0.434895843, 0.020833334, 0.665404141],
    [1.528645873, 0.113281250, 0.152507961],
    [0.432291687, 0.859375, 0.615921438],
    [1.518229246, 1.067708373, 0.140001357],
    [0.996093750, -0.554687500, 0.616546154],
    [1.084635496, 0.585937500, -0.082219556],
    [-0.25, 0.125, 0.472394258],
];
const SMOOTH_ALL: [[f64; 3]; 7] = [
    [0.434895843, 0.020833334, 0.104687497],
    [1.528645873, 0.113281250, 0.071614586],
    [0.432291687, 0.859375, 0.172916666],
    [1.518229246, 1.067708373, 0.126562506],
    [0.996093750, -0.554687500, 0.309114575],
    [1.084635496, 0.585937500, -0.322135389],
    [-0.25, 0.125, 0.0],
];
const SMOOTH_XZ: [[f64; 3]; 7] = [
    [0.773333371, 0.0, 0.143377781],
    [1.222666740, 0.0, 0.064666674],
    [0.768000066, 1.0, 0.133599997],
    [1.227555633, 1.5, 0.045511119],
    [0.987333298, -1.0, 0.177066669],
    [1.298000097, 0.5, -0.073333323],
    [-0.007999999, 0.5, 0.0],
];
const SMOOTH_GROUP_INVERT: [[f64; 3]; 7] = [
    [0.5, 0.0, 0.133333340],
    [2.0, 0.0, 0.100000001],
    [0.25, 0.9375, 0.1875],
    [1.5625, 1.125, 0.137500003],
    [1.0, -0.875, 0.443749994],
    [0.5, 0.5, -0.699999988],
    [0.0, 0.0, 0.0],
];
const SMOOTH_GROUP: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.0],
    [1.453437448, 0.144375011, 0.071968749],
    [0.270937502, 0.924999952, 0.183687493],
    [1.838124990, 1.358437538, 0.243437529],
    [0.962031245, -0.592187464, 0.317749977],
    [1.201249957, 0.627499998, -0.249249965],
    [-1.0, 0.5, 0.0],
];

/// The seven-vertex input, with vertex group `G` attached when `grouped` is set.
fn input(grouped: bool) -> Mesh {
    let positions = POSITIONS.iter().map(|p| DVec3::from_array(*p)).collect();
    let faces = FACES.iter().map(|face| face.to_vec()).collect();
    let mut mesh = Mesh::from_positions_and_faces(positions, faces).unwrap();
    if grouped {
        let weights: Map<String, Value> =
            [(0, 0.0), (1, 1.0), (2, 0.5), (3, 0.25), (4, 0.75), (5, 1.0)]
                .into_iter()
                .map(|(vertex, weight)| (format!("v{vertex}"), json!(weight)))
                .collect();
        mesh.attributes
            .insert("vertex_groups".to_owned(), json!({ "G": weights }));
    }
    mesh
}

fn modifier(modifier_type: &str, params: Value) -> Modifier {
    Modifier {
        id: Id::new("modifier_test").unwrap(),
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        params: serde_json::from_value(params).unwrap(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    }
}

fn assert_points(label: &str, actual: &Mesh, expected: &[[f64; 3]; 7]) {
    assert_eq!(
        actual.vertices.len(),
        expected.len(),
        "{label}: vertex count"
    );
    for (index, (vertex, want)) in actual.vertices.iter().zip(expected).enumerate() {
        let error = (vertex.co - DVec3::from_array(*want)).abs().max_element();
        assert!(
            error < TOLERANCE,
            "{label}: vertex {index} is {:?}, Blender gives {want:?}",
            vertex.co
        );
    }
}

fn evaluate(mesh: &Mesh, modifiers: &[Modifier]) -> Mesh {
    evaluate_modifiers(mesh, modifiers).unwrap()
}

fn evaluate_at(mesh: &Mesh, modifiers: &[Modifier], frame: f64) -> Mesh {
    evaluate_modifiers_with_groups_at_frame(mesh, modifiers, &BTreeMap::new(), frame)
        .unwrap()
        .mesh
}

#[test]
fn displace_axis_matches_blender() {
    let mesh = evaluate(
        &input(false),
        &[modifier(
            "displace",
            json!({"direction":"X","strength":0.8,"mid_level":0.5}),
        )],
    );
    assert_points("displace X", &mesh, &DISP_X);
}

#[test]
fn displace_normal_matches_blender_without_and_with_vertex_group() {
    let plain = evaluate(
        &input(false),
        &[modifier(
            "displace",
            json!({"direction":"NORMAL","strength":0.6,"mid_level":0.25}),
        )],
    );
    assert_points("displace NORMAL", &plain, &DISP_NORMAL);

    let grouped = evaluate(
        &input(true),
        &[modifier(
            "displace",
            json!({"direction":"NORMAL","strength":1.0,"mid_level":0.5,"vertex_group":"G"}),
        )],
    );
    assert_points("displace NORMAL group", &grouped, &DISP_NORMAL_GROUP);

    let inverted = evaluate(
        &input(true),
        &[modifier(
            "displace",
            json!({"direction":"NORMAL","strength":1.0,"mid_level":0.5,"vertex_group":"G","invert_vertex_group":true}),
        )],
    );
    assert_points(
        "displace NORMAL group inverted",
        &inverted,
        &DISP_NORMAL_GROUP_INVERT,
    );
}

#[test]
fn displace_negative_strength_reverses_normal_offset_like_blender() {
    let mesh = evaluate(
        &input(false),
        &[modifier(
            "displace",
            json!({"direction":"NORMAL","strength":-0.9,"mid_level":0.5}),
        )],
    );
    assert_points(
        "displace NORMAL negative strength",
        &mesh,
        &DISP_NORMAL_NEGATIVE_STRENGTH,
    );
}

#[test]
fn displace_local_axis_with_vertex_group_matches_blender() {
    let mesh = evaluate(
        &input(true),
        &[modifier(
            "displace",
            json!({"direction":"Z","strength":1.2,"mid_level":0.5,"vertex_group":"G"}),
        )],
    );
    assert_points("displace Z group", &mesh, &DISP_Z_LOCAL_GROUP);
}

#[test]
fn displace_rgb_to_xyz_without_texture_is_unchanged() {
    let mesh = evaluate(
        &input(false),
        &[modifier("displace", json!({"direction":"RGB_TO_XYZ"}))],
    );
    assert_points("displace RGB without texture", &mesh, &POSITIONS);
}

#[test]
fn wave_default_matches_blender_at_frame_one() {
    let mesh = evaluate(&input(false), &[modifier("wave", json!({}))]);
    assert_points("wave default", &mesh, &WAVE_DEFAULT);
}

#[test]
fn wave_lifetime_and_damping_match_blender_at_frame_seven() {
    let mesh = evaluate_at(
        &input(false),
        &[modifier(
            "wave",
            json!({"time_offset":2.0,"lifetime":4.0,"damping_time":2.0}),
        )],
        7.0,
    );
    assert_points("wave life and damping", &mesh, &WAVE_FRAME7_LIFE_DAMP);
}

#[test]
fn wave_falloff_and_start_position_match_blender() {
    let mesh = evaluate_at(
        &input(false),
        &[modifier(
            "wave",
            json!({"falloff_radius":2.5,"start_position_x":0.5,"start_position_y":-0.5,"speed":0.5}),
        )],
        3.0,
    );
    assert_points("wave falloff and start", &mesh, &WAVE_FALLOFF_START);
}

#[test]
fn wave_along_normals_matches_blender_for_each_axis_flag() {
    let all = evaluate_at(
        &input(false),
        &[modifier("wave", json!({"use_normal":true}))],
        3.0,
    );
    assert_points("wave normal xyz", &all, &WAVE_NORMAL_XYZ);

    let x_only = evaluate_at(
        &input(false),
        &[modifier(
            "wave",
            json!({"use_normal":true,"use_normal_y":false,"use_normal_z":false}),
        )],
        3.0,
    );
    assert_points("wave normal x only", &x_only, &WAVE_NORMAL_X_ONLY);
}

#[test]
fn wave_single_axis_without_cycle_and_vertex_group_match_blender() {
    let single = evaluate_at(
        &input(false),
        &[modifier(
            "wave",
            json!({"use_y":false,"use_cyclic":false,"width":0.8,"narrowness":2.0}),
        )],
        2.0,
    );
    assert_points("wave x only no cycle", &single, &WAVE_X_ONLY_NO_CYCLIC);

    let grouped = evaluate_at(
        &input(true),
        &[modifier(
            "wave",
            json!({"vertex_group":"G","invert_vertex_group":true,"height":1.0}),
        )],
        2.0,
    );
    assert_points("wave group inverted", &grouped, &WAVE_GROUP_INVERT);
}

#[test]
fn smooth_matches_blender_for_all_axes_including_isolated_vertex() {
    let mesh = evaluate(
        &input(false),
        &[modifier("smooth", json!({"factor":0.5,"iterations":2}))],
    );
    assert_points("smooth all", &mesh, &SMOOTH_ALL);
}

#[test]
fn smooth_axis_subset_and_vertex_group_match_blender() {
    let axes = evaluate(
        &input(false),
        &[modifier(
            "smooth",
            json!({"factor":0.8,"iterations":3,"use_y":false}),
        )],
    );
    assert_points("smooth xz", &axes, &SMOOTH_XZ);

    let inverted = evaluate(
        &input(true),
        &[modifier(
            "smooth",
            json!({"factor":1.0,"iterations":1,"vertex_group":"G","invert_vertex_group":true}),
        )],
    );
    assert_points("smooth group inverted", &inverted, &SMOOTH_GROUP_INVERT);

    let grouped = evaluate(
        &input(true),
        &[modifier(
            "smooth",
            json!({"factor":0.6,"iterations":2,"vertex_group":"G"}),
        )],
    );
    assert_points("smooth group", &grouped, &SMOOTH_GROUP);
}

#[test]
fn smooth_vertex_group_absent_from_mesh_is_full_weight_like_blender() {
    let mesh = evaluate(
        &input(false),
        &[modifier(
            "smooth",
            json!({"factor":0.5,"iterations":2,"vertex_group":"Nope"}),
        )],
    );
    assert_points("smooth missing group", &mesh, &SMOOTH_ALL);
}

#[test]
fn wave_then_displace_normal_uses_the_deformed_normals_like_blender() {
    let mesh = evaluate_at(
        &input(false),
        &[
            modifier("wave", json!({"use_normal":true,"height":0.6})),
            modifier(
                "displace",
                json!({"direction":"NORMAL","strength":0.5,"mid_level":0.5}),
            ),
        ],
        3.0,
    );
    assert_points(
        "wave normal then displace normal",
        &mesh,
        &CHAIN_WAVE_NORMAL_THEN_DISPLACE,
    );
}

#[test]
fn smooth_then_wave_matches_blender() {
    let mesh = evaluate_at(
        &input(false),
        &[
            modifier("smooth", json!({"factor":0.5,"iterations":2})),
            modifier("wave", json!({"height":0.6,"speed":0.3})),
        ],
        2.0,
    );
    assert_points("smooth then wave", &mesh, &CHAIN_SMOOTH_THEN_WAVE);
}

/// Blender's 4x3 test texture: `pixels[y * 4 + x]` with row 0 at the bottom, values k/255.
fn texture_pixels() -> Value {
    let mut pixels = Vec::new();
    for y in 0..3_u32 {
        for x in 0..4_u32 {
            pixels.push(json!([
                f64::from((x * 37 + y * 11) % 256) / 255.0,
                f64::from((y * 70 + 40) % 256) / 255.0,
                f64::from((x * 5 + y * 3) % 256) / 255.0,
                1.0
            ]));
        }
    }
    Value::Array(pixels)
}

fn texture_ops() -> Vec<Value> {
    vec![
        json!({"op":"image.create","id":"tex","name":"Tex","width":4,"height":3,"colorspace":"linear","fill_color":[0.0,0.0,0.0,1.0]}),
        json!({"op":"image.set_pixels","id":"tex","x":0,"y":0,"width":4,"height":3,"pixels":texture_pixels()}),
    ]
}

/// Evaluates the scene built by `operations` and returns the evaluated mesh of node `subject`.
fn scene_subject(
    operations: &[Value],
    frame: Option<f64>,
) -> Result<Mesh, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let scene = directory.path().join("displacement");
    let batch = directory.path().join("operations.json");
    let init = std::process::Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    if !init.status.success() {
        return Err(String::from_utf8_lossy(&init.stderr).into_owned().into());
    }
    std::fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":0,
            "operations":operations,
        }))?,
    )?;
    let applied = std::process::Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    if !applied.status.success() {
        return Err(String::from_utf8_lossy(&applied.stderr).into_owned().into());
    }
    let document: potter_core::model::SceneDoc =
        serde_json::from_slice(&std::fs::read(scene.join("scene.json"))?)?;
    let node = document
        .nodes
        .get(&Id::new("subject".to_owned())?)
        .ok_or("subject node is missing")?;
    let data_id = node.data.as_ref().ok_or("subject mesh data is missing")?;
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        &document,
        &potter_core::eval::EvaluationContext {
            frame,
            ..Default::default()
        },
        Some(&scene),
        None,
    )?;
    snapshot
        .meshes
        .get(data_id)
        .cloned()
        .ok_or_else(|| "subject mesh was not evaluated".into())
}

fn plane_ops(transform: Value, deform: Value) -> Vec<Value> {
    let mut ops = texture_ops();
    ops.push(json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0},"transform":transform}));
    ops.push(deform);
    ops
}

fn assert_plane(label: &str, mesh: &Mesh, expected: &[[f64; 3]; 4]) {
    assert_eq!(mesh.vertices.len(), expected.len(), "{label}: vertex count");
    for (index, (vertex, want)) in mesh.vertices.iter().zip(expected).enumerate() {
        let error = (vertex.co - DVec3::from_array(*want)).abs().max_element();
        assert!(
            error < TOLERANCE,
            "{label}: vertex {index} is {:?}, Blender gives {want:?}",
            vertex.co
        );
    }
}

const SCENE_TEX_LOCAL_X: [[f64; 3]; 4] = [
    [-1.255555511, -1.0, 0.0],
    [0.744444489, -1.0, 0.0],
    [0.744444489, 1.0, 0.0],
    [-1.255555511, 1.0, 0.0],
];
const SCENE_TEX_LOCAL_RGB: [[f64; 3]; 4] = [
    [-1.119607806, -1.034313679, -0.229411781],
    [0.880392194, -1.034313679, -0.229411781],
    [0.880392194, 0.965686321, -0.229411781],
    [-1.119607806, 0.965686321, -0.229411781],
];
const SCENE_TEX_GLOBAL_X_TRANSFORM: [[f64; 3]; 4] = [
    [-1.372565031, -0.946224868, 0.0],
    [0.729355395, -0.960935831, 0.0],
    [0.487294734, 1.074002624, 0.0],
    [-1.614625692, 1.088713646, 0.0],
];
const SCENE_TEX_OBJECT_X_TRANSFORM: [[f64; 3]; 4] = [
    [-1.286264300, -1.0, 0.0],
    [0.726361394, -1.0, 0.0],
    [0.875387669, 1.0, 0.0],
    [-1.307304144, 1.0, 0.0],
];
const SCENE_WARP_TEX_LOCAL: [[f64; 3]; 4] = [
    [-1.170758367, -0.647514105, 0.0],
    [0.975437999, -0.769736409, 0.0],
    [1.342104673, 0.953069687, 0.0],
    [-0.804091692, 1.075292110, 0.0],
];
const SCENE_WAVE_TEX_LOCAL: [[f64; 3]; 4] = [
    [-1.0, -1.0, 0.062329289],
    [1.0, -1.0, 0.062329289],
    [1.0, 1.0, 0.062329289],
    [-1.0, 1.0, 0.062329289],
];

/// The subject transform of the transformed-texture cases (translation, Euler Z 30 degrees, scale).
fn subject_transform() -> Value {
    json!({"translation":[1.0,2.0,3.0],"rotation_deg":[0.0,0.0,30.0],"scale":[2.0,0.5,1.5]})
}

#[test]
fn displace_texture_local_axis_matches_blender() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = scene_subject(
        &plane_ops(
            json!({}),
            json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"X","strength":1.0,"mid_level":0.5,"texture":"tex","texture_coords":"LOCAL"}}),
        ),
        None,
    )?;
    assert_plane("displace texture local X", &mesh, &SCENE_TEX_LOCAL_X);
    Ok(())
}

#[test]
fn displace_rgb_to_xyz_texture_matches_blender() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = scene_subject(
        &plane_ops(
            json!({}),
            json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"RGB_TO_XYZ","strength":0.5,"mid_level":0.5,"texture":"tex","texture_coords":"LOCAL"}}),
        ),
        None,
    )?;
    assert_plane("displace RGB to XYZ", &mesh, &SCENE_TEX_LOCAL_RGB);
    Ok(())
}

#[test]
fn displace_texture_global_coordinates_use_subject_transform()
-> Result<(), Box<dyn std::error::Error>> {
    let mesh = scene_subject(
        &plane_ops(
            subject_transform(),
            json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"X","space":"GLOBAL","strength":1.0,"mid_level":0.5,"texture":"tex","texture_coords":"GLOBAL"}}),
        ),
        None,
    )?;
    assert_plane(
        "displace texture global",
        &mesh,
        &SCENE_TEX_GLOBAL_X_TRANSFORM,
    );
    Ok(())
}

#[test]
fn displace_texture_object_coordinates_use_map_object_transform()
-> Result<(), Box<dyn std::error::Error>> {
    let mut ops = texture_ops();
    ops.push(json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0},"transform":subject_transform()}));
    ops.push(json!({"op":"node.create","id":"map","kind":"empty","transform":{"translation":[0.3,-0.2,0.5],"rotation_deg":[0.0,0.0,20.0],"scale":[1.5,1.0,0.8]}}));
    ops.push(json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"X","strength":1.0,"mid_level":0.5,"texture":"tex","texture_coords":"OBJECT","texture_coords_object":"map"}}));
    let mesh = scene_subject(&ops, None)?;
    assert_plane(
        "displace texture object",
        &mesh,
        &SCENE_TEX_OBJECT_X_TRANSFORM,
    );
    Ok(())
}

#[test]
fn displace_texture_uses_named_native_uv_layer() -> Result<(), Box<dyn std::error::Error>> {
    let operations = [
        json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}}),
        json!({"op":"texture.create","id":"blend_tex","texture_type":"BLEND","settings":{"progression":"DIAGONAL"}}),
        json!({"op":"uv.layer_create","target":{"id":"subject"},"name":"UVMap"}),
        json!({"op":"uv.unwrap","target":{"id":"subject"},"method":"box","uv_layer":"UVMap"}),
        json!({"op":"uv.transform","target":{"id":"subject"},"uv_layer":"UVMap","scale":[-0.5,0.5],"translation":[0.5,0.5]}),
        json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"X","strength":1.0,"mid_level":0.5,"texture":"blend_tex","texture_coords":"UV","uv_layer":"UVMap"}}),
    ];
    let mesh = scene_subject(&operations, None)?;
    // Blender 5.2.2's local Blend oracle permuted by the named UV layer's horizontal flip.
    let expected = [
        [1.0, -1.0, 0.0],
        [-1.5, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.5, 1.0, 0.0],
    ];
    assert_plane("displace texture named UV layer", &mesh, &expected);
    Ok(())
}

#[test]
fn warp_texture_factor_matches_blender() -> Result<(), Box<dyn std::error::Error>> {
    let mut ops = texture_ops();
    ops.push(json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}}));
    ops.push(json!({"op":"node.create","id":"from","kind":"empty","transform":{"translation":[0.5,0.0,0.0],"rotation_deg":[0.0,0.0,30.0]}}));
    ops.push(json!({"op":"node.create","id":"to","kind":"empty","transform":{"translation":[1.0,0.5,0.0],"scale":[1.5,0.5,1.0]}}));
    ops.push(json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"warp","params":{"object_from":"from","object_to":"to","falloff_type":"NONE","strength":1.0,"texture":"tex","texture_coords":"LOCAL"}}));
    let mesh = scene_subject(&ops, None)?;
    assert_plane("warp texture local", &mesh, &SCENE_WARP_TEX_LOCAL);
    Ok(())
}

#[test]
fn wave_texture_factor_matches_blender_at_frame_three() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = scene_subject(
        &plane_ops(
            json!({}),
            json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"wave","params":{"height":0.7,"texture":"tex","texture_coords":"LOCAL"}}),
        ),
        Some(3.0),
    )?;
    assert_plane("wave texture local", &mesh, &SCENE_WAVE_TEX_LOCAL);
    Ok(())
}

#[test]
fn inline_blend_texture_diagonal_matches_blender() -> Result<(), Box<dyn std::error::Error>> {
    let operations = [
        json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}}),
        json!({"op":"texture.create","id":"blend_tex","texture_type":"BLEND"}),
        json!({"op":"texture.update","id":"blend_tex","settings":{"progression":"DIAGONAL"}}),
        json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"X","strength":1.0,"mid_level":0.5,"texture":"blend_tex","texture_coords":"LOCAL"}}),
    ];
    let mesh = scene_subject(&operations, None)?;
    let expected = [
        [-1.5, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.5, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    assert_plane("inline Blend diagonal", &mesh, &expected);
    Ok(())
}

#[test]
fn inline_clouds_texture_uses_blender_noise_turbulence() -> Result<(), Box<dyn std::error::Error>> {
    let operations = [
        json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}}),
        json!({"op":"texture.create","id":"cloud_tex","texture_type":"CLOUDS","settings":{"noise_scale":0.7,"noise_depth":2,"noise_basis":"BLENDER_ORIGINAL","noise_type":"SOFT_NOISE","cloud_type":"GRAYSCALE"}}),
        json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"X","strength":1.0,"mid_level":0.5,"texture":"cloud_tex","texture_coords":"LOCAL"}}),
    ];
    let mesh = scene_subject(&operations, None)?;
    let expected = [
        [-0.980161905, -1.0, 0.0],
        [1.006510735, -1.0, 0.0],
        [0.890910566, 1.0, 0.0],
        [-1.306399941, 1.0, 0.0],
    ];
    assert_plane("inline Clouds texture", &mesh, &expected);
    Ok(())
}

#[test]
fn legacy_procedural_texture_types_match_blender() -> Result<(), Box<dyn std::error::Error>> {
    let cases: [(&str, &str, Value, [[f64; 3]; 4]); 7] = [
        (
            "wood",
            "WOOD",
            json!({"noise_scale":0.7,"turbulence":1.3,"wood_type":"BANDNOISE","noise_basis":"IMPROVED_PERLIN","noise_type":"SOFT_NOISE","noise_basis_2":"TRI"}),
            [
                [-0.786335468, -1.0, 0.0],
                [1.388136148, -1.0, 0.0],
                [0.876659393, 1.0, 0.0],
                [-0.723966121, 1.0, 0.0],
            ],
        ),
        (
            "marble",
            "MARBLE",
            json!({"noise_scale":0.7,"turbulence":1.7,"noise_depth":3,"noise_type":"HARD_NOISE","marble_type":"SHARP","noise_basis":"ORIGINAL_PERLIN","noise_basis_2":"SAW"}),
            [
                [-0.826449752, -1.0, 0.0],
                [0.694319665, -1.0, 0.0],
                [1.307487249, 1.0, 0.0],
                [-1.291079879, 1.0, 0.0],
            ],
        ),
        (
            "magic",
            "MAGIC",
            json!({"turbulence":4.0,"noise_depth":3}),
            [
                [-0.891056001, -1.0, 0.0],
                [1.002521873, -1.0, 0.0],
                [1.099560261, 1.0, 0.0],
                [-0.997478127, 1.0, 0.0],
            ],
        ),
        (
            "stucci",
            "STUCCI",
            json!({"noise_scale":0.7,"turbulence":6.0,"noise_basis":"IMPROVED_PERLIN","noise_type":"HARD_NOISE","stucci_type":"WALL_IN"}),
            [
                [-0.890909791, -1.0, 0.0],
                [0.956595302, -1.0, 0.0],
                [0.742240787, 1.0, 0.0],
                [-1.417513132, 1.0, 0.0],
            ],
        ),
        (
            "voronoi",
            "VORONOI",
            json!({"noise_scale":0.7,"weight_1":0.7,"weight_2":0.3,"weight_3":0.1,"weight_4":0.0,"distance_metric":"MANHATTAN","color_mode":"INTENSITY"}),
            [
                [-0.914880514, -1.0, 0.0],
                [1.328364611, -1.0, 0.0],
                [1.523912549, 1.0, 0.0],
                [-0.446137071, 1.0, 0.0],
            ],
        ),
        (
            "distorted noise",
            "DISTORTED_NOISE",
            json!({"noise_scale":0.7,"distortion":1.2,"noise_basis":"IMPROVED_PERLIN","noise_distortion":"ORIGINAL_PERLIN"}),
            [
                [-2.025891304, -1.0, 0.0],
                [0.319869339, -1.0, 0.0],
                [0.645129561, 1.0, 0.0],
                [-1.473574638, 1.0, 0.0],
            ],
        ),
        (
            "musgrave",
            "MUSGRAVE",
            json!({"noise_scale":0.7,"musgrave_type":"HYBRID_MULTIFRACTAL","dimension_max":1.2,"lacunarity":2.0,"octaves":3.0,"offset":1.0,"gain":2.0,"noise_intensity":0.75,"noise_basis":"ORIGINAL_PERLIN"}),
            [
                [-0.465015054, -1.0, 0.0],
                [1.687701941, -1.0, 0.0],
                [1.442296505, 1.0, 0.0],
                [-0.498639464, 1.0, 0.0],
            ],
        ),
    ];
    for (id, texture_type, settings, expected) in cases {
        let operations = [
            json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}}),
            json!({"op":"texture.create","id":"procedural_tex","texture_type":texture_type,"settings":settings}),
            json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"X","strength":1.0,"mid_level":0.5,"texture":"procedural_tex","texture_coords":"LOCAL"}}),
        ];
        let mesh = scene_subject(&operations, None)?;
        assert_plane(id, &mesh, &expected);
    }
    Ok(())
}

#[test]
fn procedural_rgb_textures_feed_blender_rgb_to_xyz() -> Result<(), Box<dyn std::error::Error>> {
    let cases: [(&str, &str, Value, [[f64; 3]; 4]); 3] = [
        (
            "clouds color",
            "CLOUDS",
            json!({"noise_scale":0.7,"noise_depth":2,"noise_basis":"IMPROVED_PERLIN","noise_type":"HARD_NOISE","cloud_type":"COLOR"}),
            [
                [-1.020454764, -1.020454764, -0.234896079],
                [0.887225568, -1.247156024, -0.070604682],
                [0.826765239, 0.826765239, -0.218865976],
                [-1.247156024, 0.887225568, -0.214647397],
            ],
        ),
        (
            "magic",
            "MAGIC",
            json!({"turbulence":4.0,"noise_depth":3}),
            [
                [-1.259343863, -0.713979006, 0.169422061],
                [1.006774307, -0.702482820, -0.299752236],
                [0.727355063, 1.253543615, 0.198309809],
                [-0.993225694, 1.297517180, -0.299752236],
            ],
        ),
        (
            "voronoi color",
            "VORONOI",
            json!({"noise_scale":0.7,"weight_1":0.7,"weight_2":0.3,"weight_3":0.1,"weight_4":0.0,"distance_metric":"MANHATTAN","color_mode":"POSITION_OUTLINE_INTENSITY"}),
            [
                [-1.157602906, -1.126606226, 0.032202959],
                [0.890928745, -1.047678709, 0.170679048],
                [0.871754527, 1.027703524, -0.112273268],
                [-0.654694438, 0.936932743, 0.195293725],
            ],
        ),
    ];
    for (id, texture_type, settings, expected) in cases {
        let operations = [
            json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}}),
            json!({"op":"texture.create","id":"procedural_tex","texture_type":texture_type,"settings":settings}),
            json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"RGB_TO_XYZ","strength":0.6,"mid_level":0.5,"texture":"procedural_tex","texture_coords":"LOCAL"}}),
        ];
        let mesh = scene_subject(&operations, None)?;
        assert_plane(id, &mesh, &expected);
    }
    Ok(())
}

#[test]
fn native_value_noise_record_preserves_old_volume_noise_color()
-> Result<(), Box<dyn std::error::Error>> {
    let operations = [
        json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}}),
        json!({"op":"texture.create","id":"native_noise","texture_type":"NATIVE_VALUE_NOISE"}),
        json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"RGB_TO_XYZ","strength":1.0,"mid_level":0.5,"texture":"native_noise","texture_coords":"LOCAL"}}),
    ];
    let mesh = scene_subject(&operations, None)?;
    let expected = [
        [-0.867057829, -1.453320316, 0.359089834],
        [0.906857550, -0.835678429, 0.219494253],
        [1.132942171, 0.546679684, 0.359089834],
        [-1.093142450, 1.164321571, 0.219494253],
    ];
    assert_plane("native value noise migration", &mesh, &expected);
    Ok(())
}

#[test]
fn legacy_noise_texture_consumes_one_seeded_rng_value_per_vertex()
-> Result<(), Box<dyn std::error::Error>> {
    let operations = [
        json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}}),
        json!({"op":"texture.create","id":"noise_tex","texture_type":"NOISE"}),
        json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"displace","params":{"direction":"X","strength":1.0,"mid_level":0.5,"texture":"noise_tex","texture_coords":"LOCAL"}}),
    ];
    let mesh = scene_subject(&operations, None)?;
    let expected = [
        [-1.5, -1.0, 0.0],
        [0.5, -1.0, 0.0],
        [0.537037037, 1.0, 0.0],
        [-1.055555556, 1.0, 0.0],
    ];
    assert_plane("legacy Noise seed 0 sequence", &mesh, &expected);
    Ok(())
}

fn snapshot_noise_doc(render_seed: u32) -> SceneDoc {
    let operations = vec![
        json!({"op":"texture.create","id":"noise_tex","texture_type":"NOISE"}),
        json!({"op":"node.create","id":"a_noise","kind":"plane","params":{"size":2.0}}),
        json!({"op":"node.create","id":"b_subject","kind":"plane","params":{"size":2.0}}),
        json!({"op":"modifier.create","target":{"id":"a_noise"},"id":"a_displace","type":"displace","params":{"direction":"X","strength":0.8,"mid_level":0.5,"texture":"noise_tex","texture_coords":"LOCAL"}}),
        json!({"op":"modifier.create","target":{"id":"b_subject"},"id":"b_first","type":"displace","params":{"direction":"X","strength":0.7,"mid_level":0.5,"texture":"noise_tex","texture_coords":"LOCAL","vertex_group":"rng_first_vertex"}}),
        json!({"op":"modifier.create","target":{"id":"b_subject"},"id":"b_second","type":"displace","params":{"direction":"X","strength":0.9,"mid_level":0.5,"texture":"noise_tex","texture_coords":"LOCAL"}}),
    ];
    let mut doc = apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":operations}),
    )
    .unwrap()
    .doc;
    doc.scenes.get_mut(&doc.active_scene).unwrap().render.seed = render_seed;
    let subject_id = Id::new("b_subject").unwrap();
    let data_id = doc.nodes[&subject_id].data.as_ref().unwrap().clone();
    let data = doc.data_blocks.get_mut(&data_id).unwrap();
    let group_id = Id::new("rng_first_vertex").unwrap();
    data.vertex_groups.push(VertexGroup {
        id: group_id.clone(),
        name: "rng_first_vertex".to_owned(),
    });
    let vertex_ids: Vec<u32> = data
        .mesh
        .as_ref()
        .unwrap()
        .vertices
        .iter()
        .map(|vertex| vertex.id)
        .collect();
    let first_vertex = vertex_ids[0];
    for vertex_id in vertex_ids {
        data.vertex_weights.insert(
            vertex_id,
            BTreeMap::from([(
                group_id.clone(),
                if vertex_id == first_vertex { 1.0 } else { 0.0 },
            )]),
        );
    }
    doc
}

fn snapshot_positions(doc: &SceneDoc, purpose: EvaluationPurpose, node: &str) -> Vec<DVec3> {
    let snapshot = Snapshot::evaluate(
        doc,
        &EvaluationContext {
            purpose,
            ..EvaluationContext::default()
        },
    )
    .unwrap();
    snapshot.meshes[&Id::new(node.to_owned()).unwrap()]
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect()
}

#[test]
fn snapshot_noise_rng_uses_purpose_seed_and_one_shared_stream() {
    let render_seed_zero = snapshot_noise_doc(0);
    let render_seed_nonzero = snapshot_noise_doc(73);

    assert_eq!(
        snapshot_positions(&render_seed_zero, EvaluationPurpose::Preview, "b_subject"),
        snapshot_positions(
            &render_seed_nonzero,
            EvaluationPurpose::Preview,
            "b_subject"
        ),
        "Preview must use seed 0 instead of the scene render seed",
    );
    let render_positions =
        snapshot_positions(&render_seed_nonzero, EvaluationPurpose::Render, "b_subject");
    assert_ne!(
        snapshot_positions(&render_seed_zero, EvaluationPurpose::Render, "b_subject"),
        render_positions,
        "Render must use the scene render seed",
    );
    assert_eq!(
        render_positions,
        snapshot_positions(&render_seed_nonzero, EvaluationPurpose::Render, "b_subject"),
        "same-context Snapshot evaluation must be deterministic",
    );

    let mut without_preceding_object = render_seed_nonzero.clone();
    without_preceding_object
        .nodes
        .get_mut(&Id::new("a_noise").unwrap())
        .unwrap()
        .modifiers[0]
        .enabled = false;
    assert_ne!(
        render_positions,
        snapshot_positions(
            &without_preceding_object,
            EvaluationPurpose::Render,
            "b_subject"
        ),
        "a preceding object's Noise samples must advance the Snapshot stream",
    );

    let mut without_first_modifier = render_seed_nonzero;
    without_first_modifier
        .nodes
        .get_mut(&Id::new("b_subject").unwrap())
        .unwrap()
        .modifiers
        .remove(0);
    let without_first_positions = snapshot_positions(
        &without_first_modifier,
        EvaluationPurpose::Render,
        "b_subject",
    );
    assert!(
        (render_positions[1] - without_first_positions[1]).length() > TOLERANCE,
        "the first modifier's Noise sample must advance the stream for the next modifier",
    );
}
