#![expect(
    clippy::unwrap_used,
    reason = "integration tests use concise assertions"
)]
#![expect(clippy::expect_used, reason = "integration test fixtures")]

//! Displace, Wave, and Smooth regressions against Blender 5.2.2.
//!
//! The expected coordinates come from running Blender's `MOD_displace`, `MOD_wave`, and `MOD_smooth`
//! headless (`Blender -b --factory-startup`) on the seven-vertex input below: two quads' worth of
//! asymmetric geometry, an isolated vertex (index 6), and vertex group `G` with weights on vertices
//! 0 through 5 (vertex 6 is outside `G`). Blender stores float32, so the tolerance allows its
//! rounding. Each chained case applies its first modifier to the input the second one reads.

#[path = "common/blender_file.rs"]
mod blender_file;

use std::{collections::BTreeMap, env, path::PathBuf, process::Command};

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
const SUBJECT_NAME: &str = "DisplacementSubject";

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
    [-0.016_414_464, 0.017_582_612, 0.449_356_705],
    [2.198_567_152, 0.085_076_153, 0.494_757_116],
    [-0.013_324_607, 0.928_935_409, 0.644_153_476],
    [2.316_853_285, 1.434_091_926, 0.612_666_249],
    [0.979_503_036, -0.815_527_558, 0.909_938_812],
    [0.723_639_488, 0.551_609_159, -1.087_068_319],
    [-1.402_492_285, 0.701_246_142, 0.0],
];
const DISP_NORMAL_GROUP: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.0],
    [2.220_630_169, 0.094_529_055, 0.538_619_041],
    [-0.007_402_559, 0.960_519_671, 0.446_751_922],
    [2.088_014_841, 1.481_692_195, 0.386_851_728],
    [0.982_919_216, -0.846_272_945, 0.841_615_677],
    [0.748_488_307, 0.557_343_483, -1.130_075_932],
    [-1.0, 0.5, 0.0],
];
const DISP_NORMAL_GROUP_INVERT: [[f64; 3]; 7] = [
    [-0.018_238_293, 0.019_536_234, 0.499_285_221],
    [2.0, 0.0, 0.1],
    [-0.007_402_559, 0.960_519_671, 0.446_751_922],
    [2.264_044_285, 1.445_076_585, 0.560_555_220],
    [0.994_306_386, -0.948_757_648, 0.613_871_872],
    [0.5, 0.5, -0.699_999_988],
    [-1.447_213_650, 0.723_606_825, 0.0],
];
const DISP_NORMAL_NEGATIVE_STRENGTH: [[f64; 3]; 7] = [
    [0.016_414_464, -0.017_582_610, -0.449_356_675],
    [1.801_432_967, -0.085_076_146, -0.294_757_098],
    [0.013_324_606, 1.071_064_591, -0.244_153_455],
    [1.683_146_715, 1.565_908_074, -0.012_666_196],
    [1.020_496_964, -1.184_472_442, 0.090_061_247],
    [0.276_360_542, 0.448_390_871, -0.312_931_687],
    [-0.597_507_775, 0.298_753_887, 0.0],
];
const DISP_Z_LOCAL_GROUP: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.7],
    [0.0, 1.0, 0.5],
    [2.0, 1.5, 0.450_000_018],
    [1.0, -1.0, 0.950_000_048],
    [0.5, 0.5, -0.099_999_964],
    [-1.0, 0.5, 0.0],
];
const CHAIN_WAVE_NORMAL_THEN_DISPLACE: [[f64; 3]; 7] = [
    [0.022_964_422, 0.003_656_663, 0.413_499_951],
    [2.182_352_304, 0.014_779_855, 0.289_295_197],
    [0.022_200_976, 0.846_529_782, 0.947_293_162],
    [2.207_600_832, 1.437_441_707, 0.424_452_156],
    [0.995_894_670, -0.779_922_068, 0.912_266_791],
    [0.983_255_625, 0.589_327_097, -1.373_891_711],
    [-1.615_886_688, 0.807_943_344, 0.0],
];
const WAVE_DEFAULT: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.431_242_645],
    [2.0, 0.0, 0.1],
    [0.0, 1.0, 0.337_866_604],
    [2.0, 1.5, 0.3],
    [1.0, -1.0, 0.520_523_250],
    [0.5, 0.5, -0.390_704_334],
    [-1.0, 0.5, 0.088_603_832],
];
const WAVE_FRAME7_LIFE_DAMP: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.003_426_778],
    [2.0, 0.0, 0.140_380_204],
    [0.0, 1.0, 0.326_308_072],
    [2.0, 1.5, 0.303_426_802],
    [1.0, -1.0, 0.636_898_398],
    [0.5, 0.5, -0.625_473_797],
    [-1.0, 0.5, 0.139_892_280],
];
const WAVE_FALLOFF_START: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.084_879_264],
    [2.0, 0.0, 0.279_906_869],
    [0.0, 1.0, 0.379_906_863],
    [2.0, 1.5, 0.3],
    [1.0, -1.0, 0.584_879_279],
    [0.5, 0.5, -0.530_964_077],
    [-1.0, 0.5, 0.112_572_186],
];
const WAVE_NORMAL_XYZ: [[f64; 3]; 7] = [
    [-0.005_028_903, 0.005_386_789, 0.137_669_533],
    [2.005_162_716, 0.002_211_933, 0.110_263_467],
    [-0.012_769_197, 0.931_897_640, 0.625_639_796],
    [2.0, 1.5, 0.3],
    [0.991_704_166, -0.925_337_434, 0.665_916_800],
    [0.745_888_948, 0.556_743_622, -1.125_576_973],
    [-1.326_899_886, 0.663_449_943, 0.0],
];
const WAVE_NORMAL_X_ONLY: [[f64; 3]; 7] = [
    [-0.005_028_903, 0.0, 0.0],
    [2.005_162_716, 0.0, 0.1],
    [-0.012_769_197, 1.0, 0.2],
    [2.0, 1.5, 0.3],
    [0.991_704_166, -1.0, 0.5],
    [0.745_888_948, 0.5, -0.7],
    [-1.326_899_886, 0.5, 0.0],
];
const WAVE_X_ONLY_NO_CYCLIC: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.145_287_365],
    [2.0, 0.0, 0.1],
    [0.0, 1.0, 0.345_287_383],
    [2.0, 1.5, 0.3],
    [1.0, -1.0, 0.645_287_395],
    [0.5, 0.5, -0.238_652_349],
    [-1.0, 0.5, 0.0],
];
const WAVE_GROUP_INVERT: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.563_453_078],
    [2.0, 0.0, 0.1],
    [0.0, 1.0, 0.481_726_527],
    [2.0, 1.5, 0.3],
    [1.0, -1.0, 0.536_545_277],
    [0.5, 0.5, -0.7],
    [-1.0, 0.5, 0.417_076_349],
];
const CHAIN_SMOOTH_THEN_WAVE: [[f64; 3]; 7] = [
    [0.434_895_843, 0.020_833_334, 0.665_404_141],
    [1.528_645_873, 0.113_281_250, 0.152_507_961],
    [0.432_291_687, 0.859_375, 0.615_921_438],
    [1.518_229_246, 1.067_708_373, 0.140_001_357],
    [0.996_093_750, -0.554_687_500, 0.616_546_154],
    [1.084_635_496, 0.585_937_500, -0.082_219_556],
    [-0.25, 0.125, 0.472_394_258],
];
const SMOOTH_ALL: [[f64; 3]; 7] = [
    [0.434_895_843, 0.020_833_334, 0.104_687_497],
    [1.528_645_873, 0.113_281_250, 0.071_614_586],
    [0.432_291_687, 0.859_375, 0.172_916_666],
    [1.518_229_246, 1.067_708_373, 0.126_562_506],
    [0.996_093_750, -0.554_687_500, 0.309_114_575],
    [1.084_635_496, 0.585_937_500, -0.322_135_389],
    [-0.25, 0.125, 0.0],
];
const SMOOTH_XZ: [[f64; 3]; 7] = [
    [0.773_333_371, 0.0, 0.143_377_781],
    [1.222_666_740, 0.0, 0.064_666_674],
    [0.768_000_066, 1.0, 0.133_599_997],
    [1.227_555_633, 1.5, 0.045_511_119],
    [0.987_333_298, -1.0, 0.177_066_669],
    [1.298_000_097, 0.5, -0.073_333_323],
    [-0.007_999_999, 0.5, 0.0],
];
const SMOOTH_GROUP_INVERT: [[f64; 3]; 7] = [
    [0.5, 0.0, 0.133_333_340],
    [2.0, 0.0, 0.100_000_001],
    [0.25, 0.9375, 0.1875],
    [1.5625, 1.125, 0.137_500_003],
    [1.0, -0.875, 0.443_749_994],
    [0.5, 0.5, -0.699_999_988],
    [0.0, 0.0, 0.0],
];
const SMOOTH_GROUP: [[f64; 3]; 7] = [
    [0.0, 0.0, 0.0],
    [1.453_437_448, 0.144_375_011, 0.071_968_749],
    [0.270_937_502, 0.924_999_952, 0.183_687_493],
    [1.838_124_990, 1.358_437_538, 0.243_437_529],
    [0.962_031_245, -0.592_187_464, 0.317_749_977],
    [1.201_249_957, 0.627_499_998, -0.249_249_965],
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
        show_viewport: true,
        show_render: true,
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
    evaluate_modifiers(mesh, modifiers, SUBJECT_NAME).unwrap()
}

fn evaluate_at(mesh: &Mesh, modifiers: &[Modifier], frame: f64) -> Mesh {
    evaluate_modifiers_with_groups_at_frame(mesh, modifiers, &BTreeMap::new(), frame, SUBJECT_NAME)
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
        return Err(pot_failure(&init));
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
        return Err(pot_failure(&applied));
    }
    let document: potter_core::model::SceneDoc =
        serde_json::from_slice(&std::fs::read(scene.join("scene.json"))?)?;
    let subject = Id::new("subject".to_owned())?;
    let node = document
        .nodes
        .get(&subject)
        .ok_or("subject node is missing")?;
    if node.data.is_none() {
        return Err("subject mesh data is missing".into());
    }
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
        .get(&subject)
        .cloned()
        .ok_or_else(|| "subject mesh was not evaluated".into())
}

/// The message of a failed `pot --json` run. Failure envelopes are printed on stdout, not stderr.
fn pot_failure(output: &std::process::Output) -> Box<dyn std::error::Error> {
    let detail = serde_json::from_slice::<Value>(&output.stdout)
        .ok()
        .and_then(|envelope| envelope.get("error").cloned())
        .map_or_else(
            || String::from_utf8_lossy(&output.stderr).into_owned(),
            |error| error.to_string(),
        );
    format!("pot exited with {}: {detail}", output.status).into()
}

fn plane_ops(transform: &Value, deform: Value) -> Vec<Value> {
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
    [-1.255_555_511, -1.0, 0.0],
    [0.744_444_489, -1.0, 0.0],
    [0.744_444_489, 1.0, 0.0],
    [-1.255_555_511, 1.0, 0.0],
];
const SCENE_TEX_LOCAL_RGB: [[f64; 3]; 4] = [
    [-1.119_607_806, -1.034_313_679, -0.229_411_781],
    [0.880_392_194, -1.034_313_679, -0.229_411_781],
    [0.880_392_194, 0.965_686_321, -0.229_411_781],
    [-1.119_607_806, 0.965_686_321, -0.229_411_781],
];
const SCENE_TEX_GLOBAL_X_TRANSFORM: [[f64; 3]; 4] = [
    [-1.372_565_031, -0.946_224_868, 0.0],
    [0.729_355_395, -0.960_935_831, 0.0],
    [0.487_294_734, 1.074_002_624, 0.0],
    [-1.614_625_692, 1.088_713_646, 0.0],
];
const SCENE_TEX_OBJECT_X_TRANSFORM: [[f64; 3]; 4] = [
    [-1.286_264_300, -1.0, 0.0],
    [0.726_361_394, -1.0, 0.0],
    [0.875_387_669, 1.0, 0.0],
    [-1.307_304_144, 1.0, 0.0],
];
const SCENE_WARP_TEX_LOCAL: [[f64; 3]; 4] = [
    [-1.170_758_367, -0.647_514_105, 0.0],
    [0.975_437_999, -0.769_736_409, 0.0],
    [1.342_104_673, 0.953_069_687, 0.0],
    [-0.804_091_692, 1.075_292_110, 0.0],
];
const SCENE_WAVE_TEX_LOCAL: [[f64; 3]; 4] = [
    [-1.0, -1.0, 0.062_329_289],
    [1.0, -1.0, 0.062_329_289],
    [1.0, 1.0, 0.062_329_289],
    [-1.0, 1.0, 0.062_329_289],
];

/// The subject transform of the transformed-texture cases (translation, Euler Z 30 degrees, scale).
fn subject_transform() -> Value {
    json!({"translation":[1.0,2.0,3.0],"rotation_deg":[0.0,0.0,30.0],"scale":[2.0,0.5,1.5]})
}

#[test]
fn displace_texture_local_axis_matches_blender() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = scene_subject(
        &plane_ops(
            &json!({}),
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
            &json!({}),
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
            &subject_transform(),
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
    // Blender 5.2.2 displace on the product's box corner UVs (see
    // tests/fixtures/pr18f_displacement/generate_named_uv_exact_oracle.py).
    let expected = [
        [-1.5, -1.0, 0.0],
        [1.0, -1.0, 0.0],
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

const WARP_CURVE_VERTICES: [[f64; 3]; 4] = [
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
];
const WARP_CURVE_FROM: [f64; 3] = [-0.8, -0.4, 0.0];
const WARP_CURVE_TO: [f64; 3] = [0.0, 0.0, 2.0];
const WARP_CURVE_RADIUS: f64 = 2.7;
const WARP_CURVE_POINTS: [[f64; 2]; 4] = [[0.0, 0.0], [0.3, 0.9], [0.65, 0.15], [1.0, 1.0]];
const WARP_CURVE_RAW_POINTS: [[f64; 2]; 5] = [
    [-0.2, -0.2],
    [0.7, 0.9],
    [0.7, 0.4],
    [0.35, 0.1],
    [1.25, 1.2],
];
const WARP_CURVE_HANDLE_POINTS: [[f64; 2]; 5] = [
    [0.0, 0.0],
    [0.2, 0.1],
    [0.68, 0.95],
    [0.72, 0.05],
    [1.0, 1.0],
];

fn warp_curve_handles(handle: &str) -> Value {
    json!(WARP_CURVE_HANDLE_POINTS.map(|point| json!([point[0], point[1], handle])))
}

fn blender_5_2() -> Option<PathBuf> {
    let blender = blender_file::blender_executable()?;
    let version = match Command::new(&blender).arg("--version").output() {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .unwrap_or_default()
            .to_owned(),
        Ok(output) => format!("Blender --version exited with {}", output.status),
        Err(error) => format!("could not run Blender --version: {error}"),
    };
    if version.starts_with("Blender 5.2") {
        return Some(blender);
    }
    assert!(
        env::var_os("POTTER_REQUIRE_BLENDER").is_none_or(|value| value.is_empty()),
        "POTTER_REQUIRE_BLENDER is set but this test requires Blender 5.2.x, found `{version}`"
    );
    eprintln!("skipping Warp curve parity: requires Blender 5.2.x, found `{version}`");
    None
}

const WARP_CURVE_ORACLE: &str = r#"
import bpy
import json
import sys

with open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8") as source:
    data = json.load(source)

bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("Warp Curve Oracle Mesh")
mesh.from_pydata(data["vertices"], [], data["faces"])
mesh.update()
subject = bpy.data.objects.new("Warp Curve Oracle", mesh)
bpy.context.scene.collection.objects.link(subject)
source = bpy.data.objects.new("Warp From", None)
source.location = data["from"]
bpy.context.scene.collection.objects.link(source)
target = bpy.data.objects.new("Warp To", None)
target.location = data["to"]
bpy.context.scene.collection.objects.link(target)

warp = subject.modifiers.new("Warp Curve Oracle", "WARP")
warp.object_from = source
warp.object_to = target
warp.falloff_type = "CURVE"
warp.falloff_radius = data["radius"]
warp.strength = 1.0
curve = warp.falloff_curve.curves[0]
while len(curve.points) > 2:
    curve.points.remove(curve.points[-1])
for point in data["curve"][1:-1]:
    curve.points.new(float(point[0]), float(point[1]))
for point, coordinates in zip(curve.points, data["curve"]):
    point.location = tuple(coordinates[:2])
    if len(coordinates) == 3:
        point.handle_type = str(coordinates[2])
if data["update_curve"]:
    warp.falloff_curve.update()
if data.get("curve_settings") is not None:
    settings = data["curve_settings"]
    mapping = warp.falloff_curve
    if "use_clip" in settings:
        mapping.use_clip = bool(settings["use_clip"])
    if "clip_bounds" in settings:
        mapping.clip_min_x = settings["clip_bounds"][0]
        mapping.clip_min_y = settings["clip_bounds"][1]
        mapping.clip_max_x = settings["clip_bounds"][2]
        mapping.clip_max_y = settings["clip_bounds"][3]
    if "extend" in settings:
        mapping.extend = settings["extend"]
    mapping.update()

depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = subject.evaluated_get(depsgraph)
evaluated_mesh = evaluated.to_mesh()
coordinates = [[float(component) for component in vertex.co] for vertex in evaluated_mesh.vertices]
evaluated.to_mesh_clear()
print("POTTER_WARP_CURVE=" + json.dumps(coordinates))
"#;

fn assert_warp_curve_matches_blender_522(
    curve_points: &Value,
    update_curve: bool,
    curve_settings: Option<&Value>,
) -> Result<Option<Mesh>, Box<dyn std::error::Error>> {
    let Some(blender) = blender_5_2() else {
        return Ok(None);
    };
    let mut modifier_params = json!({
        "object_from":"from",
        "object_to":"to",
        "falloff_type":"CURVE",
        "falloff_radius":WARP_CURVE_RADIUS,
        "strength":1.0,
        "falloff_curve":curve_points.clone()
    });
    if let Some(settings) = curve_settings {
        modifier_params["falloff_curve_settings"] = settings.clone();
    }
    let operations = [
        json!({"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}}),
        json!({"op":"node.create","id":"from","kind":"empty","transform":{"translation":WARP_CURVE_FROM}}),
        json!({"op":"node.create","id":"to","kind":"empty","transform":{"translation":WARP_CURVE_TO}}),
        json!({"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"warp","params":modifier_params}),
    ];
    let actual = scene_subject(&operations, None)?;
    let directory = tempfile::tempdir()?;
    let input_path = directory.path().join("warp_curve_input.json");
    let script_path = directory.path().join("warp_curve_oracle.py");
    std::fs::write(
        &input_path,
        serde_json::to_vec(&json!({
            "vertices": WARP_CURVE_VERTICES,
            "faces": [[0, 1, 2, 3]],
            "from": WARP_CURVE_FROM,
            "to": WARP_CURVE_TO,
            "radius": WARP_CURVE_RADIUS,
            "curve": curve_points.clone(),
            "update_curve": update_curve,
            "curve_settings": curve_settings.cloned(),
        }))?,
    )?;
    std::fs::write(&script_path, WARP_CURVE_ORACLE)?;
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(&script_path)
        .arg("--")
        .arg(&input_path)
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Blender Warp curve oracle failed:\n{stdout}\n{stderr}"
    );
    let result = stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_WARP_CURVE="))
        .expect("Blender Warp curve oracle emitted a result line");
    let coordinates: Vec<[f64; 3]> = serde_json::from_str(result)?;
    assert_eq!(
        actual.vertices.len(),
        coordinates.len(),
        "Warp curve vertex count"
    );
    for (index, (vertex, expected)) in actual.vertices.iter().zip(&coordinates).enumerate() {
        let error = (vertex.co - DVec3::from_array(*expected))
            .abs()
            .max_element();
        assert!(
            error < TOLERANCE,
            "Warp curve vertex {index} is {:?}, Blender gives {expected:?}",
            vertex.co
        );
    }
    Ok(Some(actual))
}

#[test]
fn warp_curve_falloff_matches_blender_522() -> Result<(), Box<dyn std::error::Error>> {
    let _ = assert_warp_curve_matches_blender_522(&json!(WARP_CURVE_POINTS), true, None)?;
    Ok(())
}

#[test]
fn warp_curve_raw_controls_match_blender_522() -> Result<(), Box<dyn std::error::Error>> {
    let _ = assert_warp_curve_matches_blender_522(&json!(WARP_CURVE_RAW_POINTS), false, None)?;
    Ok(())
}

#[test]
fn warp_curve_handle_modes_match_blender_522() -> Result<(), Box<dyn std::error::Error>> {
    let Some(auto) =
        assert_warp_curve_matches_blender_522(&json!(WARP_CURVE_HANDLE_POINTS), true, None)?
    else {
        return Ok(());
    };
    for handle in ["AUTO_CLAMPED", "VECTOR"] {
        let actual =
            assert_warp_curve_matches_blender_522(&warp_curve_handles(handle), true, None)?
                .expect("Blender availability is unchanged within the test");
        let max_difference = auto
            .vertices
            .iter()
            .zip(&actual.vertices)
            .map(|(auto, handled)| (auto.co - handled.co).abs().max_element())
            .fold(0.0_f64, f64::max);
        assert!(
            max_difference > TOLERANCE,
            "Warp {handle} handles did not change evaluated vertices"
        );
    }
    Ok(())
}

#[test]
fn warp_curve_extension_modes_match_blender_522() -> Result<(), Box<dyn std::error::Error>> {
    let points = json!([[0.2, 0.2], [0.8, 0.8]]);
    let horizontal_settings = json!({"extend":"HORIZONTAL"});
    let extrapolated_settings = json!({"extend":"EXTRAPOLATED"});
    let Some(horizontal) =
        assert_warp_curve_matches_blender_522(&points, true, Some(&horizontal_settings))?
    else {
        return Ok(());
    };
    let extrapolated =
        assert_warp_curve_matches_blender_522(&points, true, Some(&extrapolated_settings))?
            .expect("Blender availability is unchanged within the test");
    let max_difference = horizontal
        .vertices
        .iter()
        .zip(&extrapolated.vertices)
        .map(|(horizontal, extrapolated)| (horizontal.co - extrapolated.co).abs().max_element())
        .fold(0.0_f64, f64::max);
    assert!(
        max_difference > TOLERANCE,
        "Warp HORIZONTAL and EXTRAPOLATED extension did not change evaluated vertices"
    );
    Ok(())
}

const HOOK_CURVE_VERTICES: [[f64; 3]; 7] = [
    [0.1, 0.0, 0.0],
    [0.23, 0.0, 0.0],
    [0.4, 0.0, 0.0],
    [0.6, 0.0, 0.0],
    [0.7, 0.0, 0.0],
    [0.85, 0.0, 0.0],
    [0.95, 0.0, 0.0],
];
const HOOK_TARGET_MATRIX: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8, 1.0,
];

const HOOK_CURVE_ORACLE: &str = r#"
import bpy
import json
import sys
from mathutils import Matrix

with open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8") as source:
    data = json.load(source)

bpy.ops.wm.read_factory_settings(use_empty=True)
results = {}
for case in data["cases"]:
    mesh = bpy.data.meshes.new("Hook Curve Oracle Mesh")
    mesh.from_pydata(data["vertices"], [], [])
    mesh.update()
    subject = bpy.data.objects.new("Hook Curve Oracle", mesh)
    bpy.context.scene.collection.objects.link(subject)
    target = bpy.data.objects.new("Hook Curve Target", None)
    target.location = data["target"]
    bpy.context.scene.collection.objects.link(target)

    hook = subject.modifiers.new("Hook Curve Oracle", "HOOK")
    hook.object = target
    hook.matrix_inverse = Matrix.Identity(4)
    hook.vertex_indices_set(list(range(len(data["vertices"]))))
    hook.falloff_type = "CURVE"
    hook.falloff_radius = data["radius"]
    hook.strength = data["strength"]
    hook.center = (0.0, 0.0, 0.0)
    hook.use_falloff_uniform = False
    curve = hook.falloff_curve.curves[0]
    while len(curve.points) > 2:
        curve.points.remove(curve.points[-1])
    for point in case["curve"][1:-1]:
        curve.points.new(float(point[0]), float(point[1]))
    for point, coordinates in zip(curve.points, case["curve"]):
        point.location = tuple(coordinates[:2])
        if len(coordinates) == 3:
            point.handle_type = str(coordinates[2])
    hook.falloff_curve.update()

    evaluated = subject.evaluated_get(bpy.context.evaluated_depsgraph_get())
    evaluated_mesh = evaluated.to_mesh()
    results[case["name"]] = [
        [float(component) for component in vertex.co] for vertex in evaluated_mesh.vertices
    ]
    evaluated.to_mesh_clear()

print("POTTER_HOOK_CURVE=" + json.dumps(results))
"#;

fn evaluate_hook_curve(curve: &Value) -> Result<Mesh, Box<dyn std::error::Error>> {
    let mesh = Mesh::from_positions_and_faces(
        HOOK_CURVE_VERTICES
            .iter()
            .map(|point| DVec3::from_array(*point))
            .collect(),
        vec![],
    )?;
    let mut hook = modifier(
        "hook",
        json!({
            "falloff_type":"CURVE",
            "falloff_radius":1.0,
            "strength":0.8,
            "center":[0.0,0.0,0.0],
            "use_falloff_uniform":false,
            "falloff_curve":curve,
            "vertex_indices":[0,1,2,3,4,5,6],
            "matrix_inverse":[1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0]
        }),
    );
    hook.runtime.object_matrix = Some(HOOK_TARGET_MATRIX);
    Ok(evaluate_modifiers(&mesh, &[hook], SUBJECT_NAME)?)
}

#[test]
fn hook_curve_handle_modes_match_blender_522() -> Result<(), Box<dyn std::error::Error>> {
    let Some(blender) = blender_5_2() else {
        return Ok(());
    };
    let curves = [
        ("AUTO", json!(WARP_CURVE_HANDLE_POINTS)),
        ("AUTO_CLAMPED", warp_curve_handles("AUTO_CLAMPED")),
        ("VECTOR", warp_curve_handles("VECTOR")),
    ];
    let cases: Vec<Value> = curves
        .iter()
        .map(|(name, curve)| json!({"name":name,"curve":curve}))
        .collect();
    let directory = tempfile::tempdir()?;
    let input_path = directory.path().join("hook_curve_input.json");
    let script_path = directory.path().join("hook_curve_oracle.py");
    std::fs::write(
        &input_path,
        serde_json::to_vec(&json!({
            "vertices":HOOK_CURVE_VERTICES,
            "target":[0.0,0.0,0.8],
            "radius":1.0,
            "strength":0.8,
            "cases":cases,
        }))?,
    )?;
    std::fs::write(&script_path, HOOK_CURVE_ORACLE)?;
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(&script_path)
        .arg("--")
        .arg(&input_path)
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Blender Hook curve oracle failed:\n{stdout}\n{stderr}"
    );
    let result = stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_HOOK_CURVE="))
        .expect("Blender Hook curve oracle emitted a result line");
    let expected: Value = serde_json::from_str(result)?;

    let mut auto_positions: Option<Vec<DVec3>> = None;
    for (name, curve) in &curves {
        let actual = evaluate_hook_curve(curve)?;
        let expected_positions: Vec<[f64; 3]> = serde_json::from_value(expected[*name].clone())?;
        assert_eq!(
            actual.vertices.len(),
            expected_positions.len(),
            "Hook {name} vertex count"
        );
        for (index, (vertex, point)) in actual.vertices.iter().zip(&expected_positions).enumerate()
        {
            let error = (vertex.co - DVec3::from_array(*point)).abs().max_element();
            assert!(
                error < TOLERANCE,
                "Hook {name} vertex {index} is {:?}, Blender gives {point:?}",
                vertex.co
            );
        }
        if *name == "AUTO" {
            auto_positions = Some(actual.vertices.iter().map(|vertex| vertex.co).collect());
        } else {
            let auto = auto_positions.as_ref().expect("AUTO case runs first");
            let max_difference = auto
                .iter()
                .zip(&actual.vertices)
                .map(|(auto, handled)| (*auto - handled.co).abs().max_element())
                .fold(0.0_f64, f64::max);
            assert!(
                max_difference > TOLERANCE,
                "Hook {name} handles did not change evaluated vertices"
            );
        }
    }
    Ok(())
}

#[test]
fn wave_texture_factor_matches_blender_at_frame_three() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = scene_subject(
        &plane_ops(
            &json!({}),
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
        [-0.980_161_905, -1.0, 0.0],
        [1.006_510_735, -1.0, 0.0],
        [0.890_910_566, 1.0, 0.0],
        [-1.306_399_941, 1.0, 0.0],
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
                [-0.786_335_468, -1.0, 0.0],
                [1.388_136_148, -1.0, 0.0],
                [0.876_659_393, 1.0, 0.0],
                [-0.723_966_121, 1.0, 0.0],
            ],
        ),
        (
            "marble",
            "MARBLE",
            json!({"noise_scale":0.7,"turbulence":1.7,"noise_depth":3,"noise_type":"HARD_NOISE","marble_type":"SHARP","noise_basis":"ORIGINAL_PERLIN","noise_basis_2":"SAW"}),
            [
                [-0.826_449_752, -1.0, 0.0],
                [0.694_319_665, -1.0, 0.0],
                [1.307_487_249, 1.0, 0.0],
                [-1.291_079_879, 1.0, 0.0],
            ],
        ),
        (
            "magic",
            "MAGIC",
            json!({"turbulence":4.0,"noise_depth":3}),
            [
                [-0.891_056_001, -1.0, 0.0],
                [1.002_521_873, -1.0, 0.0],
                [1.099_560_261, 1.0, 0.0],
                [-0.997_478_127, 1.0, 0.0],
            ],
        ),
        (
            "stucci",
            "STUCCI",
            json!({"noise_scale":0.7,"turbulence":6.0,"noise_basis":"IMPROVED_PERLIN","noise_type":"HARD_NOISE","stucci_type":"WALL_IN"}),
            [
                [-0.890_909_791, -1.0, 0.0],
                [0.956_595_302, -1.0, 0.0],
                [0.742_240_787, 1.0, 0.0],
                [-1.417_513_132, 1.0, 0.0],
            ],
        ),
        (
            "voronoi",
            "VORONOI",
            json!({"noise_scale":0.7,"weight_1":0.7,"weight_2":0.3,"weight_3":0.1,"weight_4":0.0,"distance_metric":"MANHATTAN","color_mode":"INTENSITY"}),
            [
                [-0.914_880_514, -1.0, 0.0],
                [1.328_364_611, -1.0, 0.0],
                [1.523_912_549, 1.0, 0.0],
                [-0.446_137_071, 1.0, 0.0],
            ],
        ),
        (
            "distorted noise",
            "DISTORTED_NOISE",
            json!({"noise_scale":0.7,"distortion":1.2,"noise_basis":"IMPROVED_PERLIN","noise_distortion":"ORIGINAL_PERLIN"}),
            [
                [-2.025_891_304, -1.0, 0.0],
                [0.319_869_339, -1.0, 0.0],
                [0.645_129_561, 1.0, 0.0],
                [-1.473_574_638, 1.0, 0.0],
            ],
        ),
        (
            "musgrave",
            "MUSGRAVE",
            json!({"noise_scale":0.7,"musgrave_type":"HYBRID_MULTIFRACTAL","dimension_max":1.2,"lacunarity":2.0,"octaves":3.0,"offset":1.0,"gain":2.0,"noise_intensity":0.75,"noise_basis":"ORIGINAL_PERLIN"}),
            [
                [-0.465_015_054, -1.0, 0.0],
                [1.687_701_941, -1.0, 0.0],
                [1.442_296_505, 1.0, 0.0],
                [-0.498_639_464, 1.0, 0.0],
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
                [-1.020_454_764, -1.020_454_764, -0.234_896_079],
                [0.887_225_568, -1.247_156_024, -0.070_604_682],
                [0.826_765_239, 0.826_765_239, -0.218_865_976],
                [-1.247_156_024, 0.887_225_568, -0.214_647_397],
            ],
        ),
        (
            "magic",
            "MAGIC",
            json!({"turbulence":4.0,"noise_depth":3}),
            [
                [-1.259_343_863, -0.713_979_006, 0.169_422_061],
                [1.006_774_307, -0.702_482_820, -0.299_752_236],
                [0.727_355_063, 1.253_543_615, 0.198_309_809],
                [-0.993_225_694, 1.297_517_180, -0.299_752_236],
            ],
        ),
        (
            "voronoi color",
            "VORONOI",
            json!({"noise_scale":0.7,"weight_1":0.7,"weight_2":0.3,"weight_3":0.1,"weight_4":0.0,"distance_metric":"MANHATTAN","color_mode":"POSITION_OUTLINE_INTENSITY"}),
            [
                [-1.157_602_906, -1.126_606_226, 0.032_202_959],
                [0.890_928_745, -1.047_678_709, 0.170_679_048],
                [0.871_754_527, 1.027_703_524, -0.112_273_268],
                [-0.654_694_438, 0.936_932_743, 0.195_293_725],
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
        [-0.867_057_829, -1.453_320_316, 0.359_089_834],
        [0.906_857_550, -0.835_678_429, 0.219_494_253],
        [1.132_942_171, 0.546_679_684, 0.359_089_834],
        [-1.093_142_450, 1.164_321_571, 0.219_494_253],
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
        [0.537_037_037, 1.0, 0.0],
        [-1.055_555_556, 1.0, 0.0],
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
