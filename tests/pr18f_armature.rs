//! PR18-F armature parity with Blender 5.2.2 (`MOD_armature.cc`, `armature_deform.cc`).
//!
//! The fixture is a three-bone armature with nonuniform armature and pose scale, a 4x4 grid with
//! asymmetric vertex-group weights (including an explicit zero-weight entry), a `Mask` group, and
//! bone envelopes. Expected positions are Blender 5.2.2 evaluated-mesh coordinates in mesh-local
//! space, rounded to six decimals.

use std::error::Error;

use glam::{DQuat, DVec3};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
    ops,
};
use serde_json::{Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Blender stores float32 positions; the rounded expectations differ from Potter by under 1e-6.
const TOLERANCE: f64 = 1e-5;

const ROOT_WEIGHTS: &[(usize, f64)] = &[(0, 1.0), (2, 0.6), (3, 0.9), (4, 0.7), (5, 0.5), (6, 0.3)];
const UPPER_WEIGHTS: &[(usize, f64)] = &[
    (4, 0.3),
    (5, 0.5),
    (6, 0.7),
    (7, 0.0),
    (8, 0.4),
    (9, 0.9),
    (10, 0.6),
    (11, 0.2),
    (12, 0.8),
];
const LOWER_WEIGHTS: &[(usize, f64)] = &[
    (8, 0.6),
    (9, 0.1),
    (10, 0.4),
    (11, 0.8),
    (12, 0.2),
    (13, 0.5),
    (14, 1.0),
];
const MASK_WEIGHTS: &[(usize, f64)] = &[
    (0, 1.0),
    (1, 0.25),
    (2, 0.5),
    (3, 0.75),
    (4, 1.0),
    (5, 0.0),
    (6, 0.6),
    (7, 0.3),
    (8, 0.9),
    (9, 0.1),
    (10, 0.4),
    (11, 0.65),
    (12, 0.85),
    (13, 0.2),
    (14, 0.55),
    (15, 0.0),
];

/// Blender XYZ Euler (Rz * Ry * Rx) as the xyzw quaternion Potter's transforms store.
fn transform(translation: [f64; 3], euler: [f64; 3], scale: [f64; 3]) -> Value {
    let rotation = DQuat::from_rotation_z(euler[2])
        * DQuat::from_rotation_y(euler[1])
        * DQuat::from_rotation_x(euler[0]);
    json!({
        "translation": translation,
        "rotation": [rotation.x, rotation.y, rotation.z, rotation.w],
        "scale": scale,
    })
}

fn weight_json(weights: &[(usize, f64)], ids: &[u32]) -> TestResult<Vec<Value>> {
    weights
        .iter()
        .map(|&(index, weight)| -> TestResult<Value> {
            let vertex_id = ids
                .get(index)
                .copied()
                .ok_or("grid vertex index is out of range")?;
            Ok(json!({"vertex_id": vertex_id, "weight": weight}))
        })
        .collect()
}

/// Builds the oracle scene with one armature modifier per entry of `modifiers`, in stack order.
fn armature_doc(modifiers: &[Value]) -> TestResult<SceneDoc> {
    let operations = vec![
        json!({"op":"node.create","id":"arm","kind":"armature","transform":transform([0.4, -0.2, 0.3], [0.0, 0.0, 0.35], [1.5, 0.8, 1.2])}),
        json!({"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,0.0,1.0],"roll":0.0,"head_radius":0.4,"tail_radius":0.3,"envelope_distance":0.5,"envelope_weight":1.0}),
        json!({"op":"bone.create","target":{"id":"arm"},"id":"upper","name":"Upper","parent":"root","head":[0.0,0.0,1.0],"tail":[0.6,0.2,2.0],"roll":0.0,"head_radius":0.3,"tail_radius":0.25,"envelope_distance":0.4,"envelope_weight":1.0}),
        json!({"op":"bone.create","target":{"id":"arm"},"id":"lower","name":"Lower","parent":"upper","head":[0.6,0.2,2.0],"tail":[1.1,0.4,2.4],"roll":0.0,"head_radius":0.25,"tail_radius":0.15,"envelope_distance":0.3,"envelope_weight":0.8}),
        json!({"op":"pose.set","target":{"id":"arm"},"bone_id":"root","set":{"translation":[0.05,0.0,0.0],"rotation":ROOT_ROTATION_XYZW,"scale":[1.0,1.0,1.0]}}),
        json!({"op":"pose.set","target":{"id":"arm"},"bone_id":"upper","set":{"translation":[0.0,0.0,0.0],"rotation":UPPER_ROTATION_XYZW,"scale":[1.1,0.9,1.25]}}),
        json!({"op":"pose.set","target":{"id":"arm"},"bone_id":"lower","set":{"translation":[0.1,-0.05,0.2],"rotation":LOWER_ROTATION_XYZW,"scale":[1.0,1.0,1.0]}}),
        json!({"op":"node.create","id":"body","kind":"grid","params":{"size":2.0,"x_subdivisions":3,"y_subdivisions":3},"transform":transform([-0.3, 0.5, 0.1], [0.2, 0.0, -0.5], [0.9, 1.3, 1.0])}),
        json!({"op":"vertex_group.create","target":{"id":"body"},"id":"group_root","name":"Root"}),
        json!({"op":"vertex_group.create","target":{"id":"body"},"id":"group_upper","name":"Upper"}),
        json!({"op":"vertex_group.create","target":{"id":"body"},"id":"group_lower","name":"Lower"}),
        json!({"op":"vertex_group.create","target":{"id":"body"},"id":"group_mask","name":"Mask"}),
    ];
    let mut doc = ops::apply_batch(
        &SceneDoc::new("00000000-0000-4000-8000-000000000018".to_owned()),
        &json!({"schema_version":1,"base_revision":0,"operations":operations}),
    )?
    .doc;

    // The grid's topology is kept; its vertices take the oracle's positions in storage order.
    let mesh = doc
        .data_blocks
        .get_mut(&Id::new("body_mesh".to_owned())?)
        .and_then(|data| data.mesh.as_mut())
        .ok_or("body mesh is missing")?;
    if mesh.vertices.len() != COORDS.len() {
        return Err("grid vertex count differs from the oracle".into());
    }
    for (vertex, co) in mesh.vertices.iter_mut().zip(COORDS) {
        vertex.co = DVec3::from_array(co);
    }
    let ids: Vec<u32> = mesh.vertices.iter().map(|vertex| vertex.id).collect();

    let mut operations = vec![
        json!({"op":"vertex_group.assign","target":{"id":"body"},"group_id":"group_root","weights":weight_json(ROOT_WEIGHTS, &ids)?}),
        json!({"op":"vertex_group.assign","target":{"id":"body"},"group_id":"group_upper","weights":weight_json(UPPER_WEIGHTS, &ids)?}),
        json!({"op":"vertex_group.assign","target":{"id":"body"},"group_id":"group_lower","weights":weight_json(LOWER_WEIGHTS, &ids)?}),
        json!({"op":"vertex_group.assign","target":{"id":"body"},"group_id":"group_mask","weights":weight_json(MASK_WEIGHTS, &ids)?}),
    ];
    for (index, params) in modifiers.iter().enumerate() {
        operations.push(json!({
            "op":"modifier.create","target":{"id":"body"},
            "id":format!("skin_{index}"),"type":"armature","params":params,
        }));
    }
    Ok(ops::apply_batch(
        &doc,
        &json!({"schema_version":1,"base_revision":doc.revision,"operations":operations}),
    )?
    .doc)
}

fn evaluated_body(doc: &SceneDoc) -> TestResult<Vec<[f64; 3]>> {
    let snapshot = Snapshot::evaluate(doc, &EvaluationContext::default())?;
    let mesh = snapshot
        .meshes
        .get(&Id::new("body".to_owned())?)
        .ok_or("evaluated body mesh is missing")?;
    Ok(mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect())
}

fn assert_matches_blender(actual: &[[f64; 3]], expected: &[[f64; 3]], case: &str) {
    assert_eq!(actual.len(), expected.len(), "{case}: vertex count");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() <= TOLERANCE,
                "{case}: vertex {index} is {got:?}, Blender 5.2.2 gives {want:?}"
            );
        }
    }
}

// ---- Blender-derived tables (generated from the oracle run) ----

const COORDS: [[f64; 3]; 16] = [
    [-0.6, -0.3, 0.1],
    [-0.2, 0.25, 0.45],
    [0.25, -0.15, 0.3],
    [0.7, 0.4, 0.15],
    [-0.45, 0.1, 0.9],
    [0.05, -0.4, 1.1],
    [0.4, 0.3, 0.95],
    [0.9, -0.05, 0.8],
    [-0.35, -0.25, 1.5],
    [0.15, 0.15, 1.65],
    [0.55, -0.3, 1.4],
    [0.95, 0.35, 1.7],
    [-0.25, 0.2, 2.1],
    [0.25, -0.2, 2.25],
    [0.65, 0.1, 2.0],
    [1.0, -0.1, 2.45],
];

const BLENDER_NONE: [[f64; 3]; 16] = [
    [-0.6, -0.3, 0.1],
    [-0.2, 0.25, 0.45],
    [0.25, -0.15, 0.3],
    [0.7, 0.4, 0.15],
    [-0.45, 0.1, 0.9],
    [0.05, -0.4, 1.1],
    [0.4, 0.3, 0.95],
    [0.9, -0.05, 0.8],
    [-0.35, -0.25, 1.5],
    [0.15, 0.15, 1.65],
    [0.55, -0.3, 1.4],
    [0.95, 0.35, 1.7],
    [-0.25, 0.2, 2.1],
    [0.25, -0.2, 2.25],
    [0.65, 0.1, 2.0],
    [1.0, -0.1, 2.45],
];

const BLENDER_LINEAR_GROUPS: [[f64; 3]; 16] = [
    [-0.79158, -0.788_196, 0.310_765],
    [-0.2, 0.25, 0.45],
    [0.145_024, -0.401_419, 0.420_864],
    [0.310_447, 0.124_122, 0.387_655],
    [-0.704_464, -0.337_537, 1.638_941],
    [0.092_603, -0.551_823, 1.460_287],
    [0.239_853, 0.19829, 1.849_564],
    [0.9, -0.05, 0.8],
    [0.071_682, -0.247_359, 2.038_882],
    [0.400_573, -0.076_544, 2.457_071],
    [0.784_294, -0.413_098, 1.394_661],
    [1.48819, 0.136_952, 1.641_773],
    [0.225_799, -0.16443, 3.014_151],
    [0.773_947, -0.491_069, 2.093_434],
    [1.223_106, -0.172_836, 1.828_816],
    [1.0, -0.1, 2.45],
];

const BLENDER_LINEAR_MASK: [[f64; 3]; 16] = [
    [-0.79158, -0.788_196, 0.310_765],
    [-0.2, 0.25, 0.45],
    [0.197_512, -0.27571, 0.360_432],
    [0.407_835, 0.193_092, 0.328_241],
    [-0.704_464, -0.337_537, 1.638_941],
    [0.05, -0.4, 1.1],
    [0.303_912, 0.238_974, 1.489_738],
    [0.9, -0.05, 0.8],
    [0.029_514, -0.247_623, 1.984_993],
    [0.175_057, 0.127_346, 1.730_707],
    [0.643_718, -0.345_239, 1.397_865],
    [1.299_823, 0.211_519, 1.662_152],
    [0.154_429, -0.109_766, 2.877_029],
    [0.354_789, -0.258_214, 2.218_687],
    [0.965_208, -0.05006, 1.905_849],
    [1.0, -0.1, 2.45],
];

const BLENDER_LINEAR_MASK_INVERT: [[f64; 3]; 16] = [
    [-0.6, -0.3, 0.1],
    [-0.2, 0.25, 0.45],
    [0.197_512, -0.27571, 0.360_432],
    [0.602_612, 0.331_031, 0.209_414],
    [-0.45, 0.1, 0.9],
    [0.092_603, -0.551_823, 1.460_287],
    [0.335_941, 0.259_316, 1.309_826],
    [0.9, -0.05, 0.8],
    [-0.307_832, -0.249_736, 1.553_888],
    [0.375_516, -0.05389, 2.376_364],
    [0.690_577, -0.367_859, 1.396_797],
    [1.138_366, 0.275_433, 1.679_621],
    [-0.17863, 0.145_335, 2.237_123],
    [0.669_158, -0.432_855, 2.124_747],
    [0.907_898, -0.022_776, 1.922_967],
    [1.0, -0.1, 2.45],
];

const BLENDER_LINEAR_MISSING_MASK: [[f64; 3]; 16] = [
    [-0.79158, -0.788_196, 0.310_765],
    [-0.2, 0.25, 0.45],
    [0.145_024, -0.401_419, 0.420_864],
    [0.310_447, 0.124_122, 0.387_655],
    [-0.704_464, -0.337_537, 1.638_941],
    [0.092_603, -0.551_823, 1.460_287],
    [0.239_853, 0.19829, 1.849_564],
    [0.9, -0.05, 0.8],
    [0.071_682, -0.247_359, 2.038_882],
    [0.400_573, -0.076_544, 2.457_071],
    [0.784_294, -0.413_098, 1.394_661],
    [1.48819, 0.136_952, 1.641_773],
    [0.225_799, -0.16443, 3.014_151],
    [0.773_947, -0.491_069, 2.093_434],
    [1.223_106, -0.172_836, 1.828_816],
    [1.0, -0.1, 2.45],
];

const BLENDER_DQ_GROUPS: [[f64; 3]; 16] = [
    [-0.79158, -0.788_197, 0.310_765],
    [-0.2, 0.25, 0.45],
    [0.145_024, -0.401_419, 0.420_864],
    [0.310_447, 0.124_122, 0.387_655],
    [-0.823_059, -0.331_568, 1.617_718],
    [0.029_214, -0.555_324, 1.430_644],
    [0.164_354, 0.207_929, 1.850_251],
    [0.9, -0.05, 0.8],
    [0.035_887, -0.220_437, 2.049_341],
    [0.388_052, -0.067_759, 2.46038],
    [0.781_268, -0.390_577, 1.369_524],
    [1.487_242, 0.153_257, 1.622_083],
    [0.194_111, -0.154_517, 3.038_859],
    [0.773_948, -0.491_069, 2.093_433],
    [1.223_107, -0.172_836, 1.828_815],
    [1.0, -0.1, 2.45],
];

const BLENDER_DQ_MASK: [[f64; 3]; 16] = [
    [-0.79158, -0.788_197, 0.310_765],
    [-0.2, 0.25, 0.45],
    [0.197_512, -0.27571, 0.360_432],
    [0.407_835, 0.193_092, 0.328_241],
    [-0.823_059, -0.331_568, 1.617_718],
    [0.05, -0.4, 1.1],
    [0.258_613, 0.244_757, 1.49015],
    [0.9, -0.05, 0.8],
    [-0.002_702, -0.223_393, 1.994_407],
    [0.173_805, 0.128_224, 1.731_038],
    [0.642_507, -0.336_231, 1.38781],
    [1.299_207, 0.222_117, 1.649_354],
    [0.127_494, -0.101_339, 2.89803],
    [0.354_789, -0.258_214, 2.218_687],
    [0.965_209, -0.05006, 1.905_849],
    [1.0, -0.1, 2.45],
];

const BLENDER_ENVELOPE_ONLY: [[f64; 3]; 16] = [
    [-0.6, -0.3, 0.1],
    [-0.2, 0.25, 0.45],
    [0.145_024, -0.401_419, 0.420_864],
    [0.7, 0.4, 0.15],
    [-0.45, 0.1, 0.9],
    [0.169_361, -0.671_548, 1.138_136],
    [0.4, 0.3, 0.95],
    [0.84118, -0.046_782, 0.982_772],
    [-0.35, -0.25, 1.5],
    [0.15, 0.15, 1.65],
    [0.688_676, -0.421_369, 1.469_154],
    [1.015_678, 0.129_651, 1.932_204],
    [-0.25, 0.2, 2.1],
    [0.25, -0.2, 2.25],
    [0.859_257, -0.205_274, 2.126_866],
    [1.542_341, -0.548_129, 1.831_478],
];

const BLENDER_ENVELOPE_DQ: [[f64; 3]; 16] = [
    [-0.6, -0.3, 0.1],
    [-0.2, 0.25, 0.45],
    [0.145_024, -0.401_419, 0.420_864],
    [0.7, 0.4, 0.15],
    [-0.45, 0.1, 0.9],
    [0.16936, -0.671_548, 1.138_136],
    [0.4, 0.3, 0.95],
    [0.799_968, -0.055_302, 0.947_421],
    [-0.35, -0.25, 1.5],
    [0.15, 0.15, 1.65],
    [0.671_391, -0.423_866, 1.456_888],
    [1.004_363, 0.143_017, 1.964_483],
    [-0.25, 0.2, 2.1],
    [0.25, -0.2, 2.25],
    [0.861_052, -0.191_246, 2.165_475],
    [1.542_341, -0.548_129, 1.831_478],
];

const BLENDER_ENVELOPE_GROUPS: [[f64; 3]; 16] = [
    [-0.79158, -0.788_196, 0.310_765],
    [-0.2, 0.25, 0.45],
    [0.145_024, -0.401_419, 0.420_864],
    [0.310_447, 0.124_122, 0.387_655],
    [-0.704_464, -0.337_537, 1.638_941],
    [0.092_603, -0.551_823, 1.460_287],
    [0.239_853, 0.19829, 1.849_564],
    [0.9, -0.05, 0.8],
    [0.071_682, -0.247_359, 2.038_882],
    [0.400_573, -0.076_544, 2.457_071],
    [0.784_294, -0.413_098, 1.394_661],
    [1.48819, 0.136_952, 1.641_773],
    [0.225_799, -0.16443, 3.014_151],
    [0.773_947, -0.491_069, 2.093_434],
    [1.223_106, -0.172_836, 1.828_816],
    [1.542_341, -0.548_129, 1.831_478],
];

const BLENDER_MULTI_MASKED: [[f64; 3]; 16] = [
    [-0.79158, -0.788_197, 0.310_765],
    [-0.2, 0.25, 0.45],
    [0.145_024, -0.401_419, 0.420_864],
    [0.407_835, 0.193_092, 0.328_241],
    [-0.823_059, -0.331_568, 1.617_718],
    [0.169_361, -0.671_548, 1.138_136],
    [0.258_613, 0.244_757, 1.49015],
    [0.858_826, -0.047_747, 0.927_941],
    [-0.002_702, -0.223_393, 1.994_407],
    [0.173_805, 0.128_224, 1.731_038],
    [0.725_713, -0.409_052, 1.429_302],
    [1.322_194, 0.144_995, 1.730_625],
    [0.127_494, -0.101_339, 2.89803],
    [0.354_789, -0.258_214, 2.218_687],
    [1.059_374, -0.187_433, 1.962_938],
    [1.542_341, -0.548_129, 1.831_478],
];

const BLENDER_MULTI_MASKED_INVERT: [[f64; 3]; 16] = [
    [-0.6, -0.3, 0.1],
    [-0.2, 0.25, 0.45],
    [0.145_024, -0.401_419, 0.420_864],
    [0.602_612, 0.33103, 0.209_414],
    [-0.45, 0.1, 0.9],
    [0.029_214, -0.555_324, 1.430_644],
    [0.305_742, 0.263_172, 1.3101],
    [0.882_354, -0.049_035, 0.854_832],
    [-0.311_411, -0.247_044, 1.554_934],
    [0.364_247, -0.045_983, 2.379_342],
    [0.744_231, -0.402_894, 1.409_376],
    [1.180_725, 0.137_913, 1.823_661],
    [-0.183_383, 0.146_822, 2.240_829],
    [0.669_158, -0.432_855, 2.124_747],
    [1.022_989, -0.190_677, 1.992_743],
    [1.0, -0.1, 2.45],
];

const BLENDER_MULTI_UNMASKED: [[f64; 3]; 16] = [
    [-0.79158, -0.788_197, 0.310_765],
    [-0.2, 0.25, 0.45],
    [0.145_024, -0.401_419, 0.420_864],
    [0.310_447, 0.124_122, 0.387_655],
    [-0.823_059, -0.331_568, 1.617_718],
    [0.029_214, -0.555_324, 1.430_644],
    [0.164_354, 0.207_929, 1.850_251],
    [0.9, -0.05, 0.8],
    [0.035_887, -0.220_437, 2.049_341],
    [0.388_052, -0.067_759, 2.46038],
    [0.781_268, -0.390_577, 1.369_524],
    [1.487_242, 0.153_257, 1.622_083],
    [0.194_111, -0.154_517, 3.038_859],
    [0.773_948, -0.491_069, 2.093_433],
    [1.223_107, -0.172_836, 1.828_815],
    [1.0, -0.1, 2.45],
];

const BLENDER_MULTI_MISSING_MASK: [[f64; 3]; 16] = [
    [-0.79158, -0.788_197, 0.310_765],
    [-0.2, 0.25, 0.45],
    [0.145_024, -0.401_419, 0.420_864],
    [0.310_447, 0.124_122, 0.387_655],
    [-0.823_059, -0.331_568, 1.617_718],
    [0.029_214, -0.555_324, 1.430_644],
    [0.164_354, 0.207_929, 1.850_251],
    [0.9, -0.05, 0.8],
    [0.035_887, -0.220_437, 2.049_341],
    [0.388_052, -0.067_759, 2.46038],
    [0.781_268, -0.390_577, 1.369_524],
    [1.487_242, 0.153_257, 1.622_083],
    [0.194_111, -0.154_517, 3.038_859],
    [0.773_948, -0.491_069, 2.093_433],
    [1.223_107, -0.172_836, 1.828_815],
    [1.0, -0.1, 2.45],
];

const BLENDER_CHAIN_LINEAR_DQ: [[f64; 3]; 16] = [
    [-0.6825, -1.1855, 0.382_719],
    [-0.2, 0.25, 0.45],
    [0.195_317, -0.608_789, 0.470_726],
    [0.05452, -0.204_052, 0.585_588],
    [-0.705_072, -0.827_866, 2.093_157],
    [0.260_258, -0.740_888, 1.567_743],
    [0.372_384, -0.143_315, 2.454_087],
    [0.9, -0.05, 0.8],
    [0.523_776, -0.49177, 2.139_665],
    [0.950_397, -0.576_438, 2.407_101],
    [0.993_894, -0.530_524, 1.078_001],
    [1.910_903, -0.224_681, 1.096_022],
    [0.965_922, -0.863_952, 2.771_421],
    [1.118_361, -0.985_359, 1.523_197],
    [1.615_954, -0.656_661, 1.229_869],
    [1.0, -0.1, 2.45],
];

const ROOT_ROTATION_XYZW: [f64; 4] = [
    0.069_034_992_623,
    0.207_104_977_87,
    0.046_023_328_416,
    0.974_794_107_069,
];
const UPPER_ROTATION_XYZW: [f64; 4] = [
    0.355_488_683_981,
    0.071_097_736_796,
    -0.142_195_473_592,
    0.921_060_994_003,
];
const LOWER_ROTATION_XYZW: [f64; 4] = [-0.0, -0.295_520_206_661, -0.0, 0.955_336_489_126];

// ---- Regressions ----

#[test]
fn armature_with_no_deform_source_leaves_the_mesh_unchanged() -> TestResult {
    let doc = armature_doc(&[json!({"object":"arm","use_vertex_groups":false})])?;
    assert_matches_blender(&evaluated_body(&doc)?, &BLENDER_NONE, "no deform source");
    Ok(())
}

#[test]
fn linear_vertex_group_skinning_matches_blender() -> TestResult {
    let doc = armature_doc(&[json!({"object":"arm","use_vertex_groups":true})])?;
    assert_matches_blender(
        &evaluated_body(&doc)?,
        &BLENDER_LINEAR_GROUPS,
        "linear groups",
    );
    Ok(())
}

#[test]
fn vertex_group_mask_scales_displacement_and_inverts_like_blender() -> TestResult {
    let masked = armature_doc(&[json!({
        "object":"arm","use_vertex_groups":true,"vertex_group":"Mask"
    })])?;
    assert_matches_blender(&evaluated_body(&masked)?, &BLENDER_LINEAR_MASK, "mask");

    let inverted = armature_doc(&[json!({
        "object":"arm","use_vertex_groups":true,"vertex_group":"Mask","invert_vertex_group":true
    })])?;
    assert_matches_blender(
        &evaluated_body(&inverted)?,
        &BLENDER_LINEAR_MASK_INVERT,
        "inverted mask",
    );

    // A mask naming no vertex group has no effect, as in Blender.
    let missing = armature_doc(&[json!({
        "object":"arm","use_vertex_groups":true,"vertex_group":"Missing"
    })])?;
    assert_matches_blender(
        &evaluated_body(&missing)?,
        &BLENDER_LINEAR_MISSING_MASK,
        "missing mask group",
    );
    Ok(())
}

#[test]
fn preserve_volume_blends_dual_quaternions_with_pose_scale() -> TestResult {
    let plain = armature_doc(&[json!({
        "object":"arm","use_vertex_groups":true,"use_deform_preserve_volume":true
    })])?;
    assert_matches_blender(
        &evaluated_body(&plain)?,
        &BLENDER_DQ_GROUPS,
        "dual quaternion",
    );

    let masked = armature_doc(&[json!({
        "object":"arm","use_vertex_groups":true,"use_deform_preserve_volume":true,
        "vertex_group":"Mask"
    })])?;
    assert_matches_blender(
        &evaluated_body(&masked)?,
        &BLENDER_DQ_MASK,
        "dual quaternion mask",
    );
    Ok(())
}

#[test]
fn bone_envelopes_apply_only_where_no_vertex_group_matched() -> TestResult {
    let envelopes_only = armature_doc(&[json!({
        "object":"arm","use_vertex_groups":false,"use_bone_envelopes":true
    })])?;
    assert_matches_blender(
        &evaluated_body(&envelopes_only)?,
        &BLENDER_ENVELOPE_ONLY,
        "envelopes only",
    );

    // Vertices matched by a deform group keep group skinning. A zero-weight group entry still
    // counts as matched, so vertex 7 (Upper weight 0) stays put.
    let with_groups = armature_doc(&[json!({
        "object":"arm","use_vertex_groups":true,"use_bone_envelopes":true
    })])?;
    assert_matches_blender(
        &evaluated_body(&with_groups)?,
        &BLENDER_ENVELOPE_GROUPS,
        "envelopes with groups",
    );

    let envelopes_dq = armature_doc(&[json!({
        "object":"arm","use_vertex_groups":false,"use_bone_envelopes":true,
        "use_deform_preserve_volume":true
    })])?;
    assert_matches_blender(
        &evaluated_body(&envelopes_dq)?,
        &BLENDER_ENVELOPE_DQ,
        "envelopes dual quaternion",
    );
    Ok(())
}

#[test]
fn multi_modifier_mixes_its_deformed_input_by_the_mask() -> TestResult {
    // The first armature (envelopes only) feeds the multi armature (dual quaternion groups).
    let first = json!({"object":"arm","use_vertex_groups":false,"use_bone_envelopes":true});

    let masked = armature_doc(&[
        first.clone(),
        json!({
            "object":"arm","use_vertex_groups":true,"use_deform_preserve_volume":true,
            "use_multi_modifier":true,"vertex_group":"Mask"
        }),
    ])?;
    assert_matches_blender(
        &evaluated_body(&masked)?,
        &BLENDER_MULTI_MASKED,
        "multi masked",
    );

    let inverted = armature_doc(&[
        first.clone(),
        json!({
            "object":"arm","use_vertex_groups":true,"use_deform_preserve_volume":true,
            "use_multi_modifier":true,"vertex_group":"Mask","invert_vertex_group":true
        }),
    ])?;
    assert_matches_blender(
        &evaluated_body(&inverted)?,
        &BLENDER_MULTI_MASKED_INVERT,
        "multi inverted mask",
    );

    let unmasked = armature_doc(&[
        first.clone(),
        json!({
            "object":"arm","use_vertex_groups":true,"use_deform_preserve_volume":true,
            "use_multi_modifier":true
        }),
    ])?;
    assert_matches_blender(
        &evaluated_body(&unmasked)?,
        &BLENDER_MULTI_UNMASKED,
        "multi unmasked",
    );

    let missing = armature_doc(&[
        first,
        json!({
            "object":"arm","use_vertex_groups":true,"use_deform_preserve_volume":true,
            "use_multi_modifier":true,"vertex_group":"Missing"
        }),
    ])?;
    assert_matches_blender(
        &evaluated_body(&missing)?,
        &BLENDER_MULTI_MISSING_MASK,
        "multi missing mask group",
    );
    Ok(())
}

#[test]
fn armature_followed_by_another_armature_deform_chains_like_blender() -> TestResult {
    let doc = armature_doc(&[
        json!({"object":"arm","use_vertex_groups":true}),
        json!({"object":"arm","use_vertex_groups":true,"use_deform_preserve_volume":true}),
    ])?;
    assert_matches_blender(&evaluated_body(&doc)?, &BLENDER_CHAIN_LINEAR_DQ, "chain");
    Ok(())
}
