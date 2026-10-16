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
    Ok(mesh.vertices.iter().map(|vertex| vertex.co.to_array()).collect())
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
    [-0.79158, -0.788196, 0.310765],
    [-0.2, 0.25, 0.45],
    [0.145024, -0.401419, 0.420864],
    [0.310447, 0.124122, 0.387655],
    [-0.704464, -0.337537, 1.638941],
    [0.092603, -0.551823, 1.460287],
    [0.239853, 0.19829, 1.849564],
    [0.9, -0.05, 0.8],
    [0.071682, -0.247359, 2.038882],
    [0.400573, -0.076544, 2.457071],
    [0.784294, -0.413098, 1.394661],
    [1.48819, 0.136952, 1.641773],
    [0.225799, -0.16443, 3.014151],
    [0.773947, -0.491069, 2.093434],
    [1.223106, -0.172836, 1.828816],
    [1.0, -0.1, 2.45],
];

const BLENDER_LINEAR_MASK: [[f64; 3]; 16] = [
    [-0.79158, -0.788196, 0.310765],
    [-0.2, 0.25, 0.45],
    [0.197512, -0.27571, 0.360432],
    [0.407835, 0.193092, 0.328241],
    [-0.704464, -0.337537, 1.638941],
    [0.05, -0.4, 1.1],
    [0.303912, 0.238974, 1.489738],
    [0.9, -0.05, 0.8],
    [0.029514, -0.247623, 1.984993],
    [0.175057, 0.127346, 1.730707],
    [0.643718, -0.345239, 1.397865],
    [1.299823, 0.211519, 1.662152],
    [0.154429, -0.109766, 2.877029],
    [0.354789, -0.258214, 2.218687],
    [0.965208, -0.05006, 1.905849],
    [1.0, -0.1, 2.45],
];

const BLENDER_LINEAR_MASK_INVERT: [[f64; 3]; 16] = [
    [-0.6, -0.3, 0.1],
    [-0.2, 0.25, 0.45],
    [0.197512, -0.27571, 0.360432],
    [0.602612, 0.331031, 0.209414],
    [-0.45, 0.1, 0.9],
    [0.092603, -0.551823, 1.460287],
    [0.335941, 0.259316, 1.309826],
    [0.9, -0.05, 0.8],
    [-0.307832, -0.249736, 1.553888],
    [0.375516, -0.05389, 2.376364],
    [0.690577, -0.367859, 1.396797],
    [1.138366, 0.275433, 1.679621],
    [-0.17863, 0.145335, 2.237123],
    [0.669158, -0.432855, 2.124747],
    [0.907898, -0.022776, 1.922967],
    [1.0, -0.1, 2.45],
];

const BLENDER_LINEAR_MISSING_MASK: [[f64; 3]; 16] = [
    [-0.79158, -0.788196, 0.310765],
    [-0.2, 0.25, 0.45],
    [0.145024, -0.401419, 0.420864],
    [0.310447, 0.124122, 0.387655],
    [-0.704464, -0.337537, 1.638941],
    [0.092603, -0.551823, 1.460287],
    [0.239853, 0.19829, 1.849564],
    [0.9, -0.05, 0.8],
    [0.071682, -0.247359, 2.038882],
    [0.400573, -0.076544, 2.457071],
    [0.784294, -0.413098, 1.394661],
    [1.48819, 0.136952, 1.641773],
    [0.225799, -0.16443, 3.014151],
    [0.773947, -0.491069, 2.093434],
    [1.223106, -0.172836, 1.828816],
    [1.0, -0.1, 2.45],
];

const BLENDER_DQ_GROUPS: [[f64; 3]; 16] = [
    [-0.79158, -0.788197, 0.310765],
    [-0.2, 0.25, 0.45],
    [0.145024, -0.401419, 0.420864],
    [0.310447, 0.124122, 0.387655],
    [-0.823059, -0.331568, 1.617718],
    [0.029214, -0.555324, 1.430644],
    [0.164354, 0.207929, 1.850251],
    [0.9, -0.05, 0.8],
    [0.035887, -0.220437, 2.049341],
    [0.388052, -0.067759, 2.46038],
    [0.781268, -0.390577, 1.369524],
    [1.487242, 0.153257, 1.622083],
    [0.194111, -0.154517, 3.038859],
    [0.773948, -0.491069, 2.093433],
    [1.223107, -0.172836, 1.828815],
    [1.0, -0.1, 2.45],
];

const BLENDER_DQ_MASK: [[f64; 3]; 16] = [
    [-0.79158, -0.788197, 0.310765],
    [-0.2, 0.25, 0.45],
    [0.197512, -0.27571, 0.360432],
    [0.407835, 0.193092, 0.328241],
    [-0.823059, -0.331568, 1.617718],
    [0.05, -0.4, 1.1],
    [0.258613, 0.244757, 1.49015],
    [0.9, -0.05, 0.8],
    [-0.002702, -0.223393, 1.994407],
    [0.173805, 0.128224, 1.731038],
    [0.642507, -0.336231, 1.38781],
    [1.299207, 0.222117, 1.649354],
    [0.127494, -0.101339, 2.89803],
    [0.354789, -0.258214, 2.218687],
    [0.965209, -0.05006, 1.905849],
    [1.0, -0.1, 2.45],
];

const BLENDER_ENVELOPE_ONLY: [[f64; 3]; 16] = [
    [-0.6, -0.3, 0.1],
    [-0.2, 0.25, 0.45],
    [0.145024, -0.401419, 0.420864],
    [0.7, 0.4, 0.15],
    [-0.45, 0.1, 0.9],
    [0.169361, -0.671548, 1.138136],
    [0.4, 0.3, 0.95],
    [0.84118, -0.046782, 0.982772],
    [-0.35, -0.25, 1.5],
    [0.15, 0.15, 1.65],
    [0.688676, -0.421369, 1.469154],
    [1.015678, 0.129651, 1.932204],
    [-0.25, 0.2, 2.1],
    [0.25, -0.2, 2.25],
    [0.859257, -0.205274, 2.126866],
    [1.542341, -0.548129, 1.831478],
];

const BLENDER_ENVELOPE_DQ: [[f64; 3]; 16] = [
    [-0.6, -0.3, 0.1],
    [-0.2, 0.25, 0.45],
    [0.145024, -0.401419, 0.420864],
    [0.7, 0.4, 0.15],
    [-0.45, 0.1, 0.9],
    [0.16936, -0.671548, 1.138136],
    [0.4, 0.3, 0.95],
    [0.799968, -0.055302, 0.947421],
    [-0.35, -0.25, 1.5],
    [0.15, 0.15, 1.65],
    [0.671391, -0.423866, 1.456888],
    [1.004363, 0.143017, 1.964483],
    [-0.25, 0.2, 2.1],
    [0.25, -0.2, 2.25],
    [0.861052, -0.191246, 2.165475],
    [1.542341, -0.548129, 1.831478],
];

const BLENDER_ENVELOPE_GROUPS: [[f64; 3]; 16] = [
    [-0.79158, -0.788196, 0.310765],
    [-0.2, 0.25, 0.45],
    [0.145024, -0.401419, 0.420864],
    [0.310447, 0.124122, 0.387655],
    [-0.704464, -0.337537, 1.638941],
    [0.092603, -0.551823, 1.460287],
    [0.239853, 0.19829, 1.849564],
    [0.9, -0.05, 0.8],
    [0.071682, -0.247359, 2.038882],
    [0.400573, -0.076544, 2.457071],
    [0.784294, -0.413098, 1.394661],
    [1.48819, 0.136952, 1.641773],
    [0.225799, -0.16443, 3.014151],
    [0.773947, -0.491069, 2.093434],
    [1.223106, -0.172836, 1.828816],
    [1.542341, -0.548129, 1.831478],
];

const BLENDER_MULTI_MASKED: [[f64; 3]; 16] = [
    [-0.79158, -0.788197, 0.310765],
    [-0.2, 0.25, 0.45],
    [0.145024, -0.401419, 0.420864],
    [0.407835, 0.193092, 0.328241],
    [-0.823059, -0.331568, 1.617718],
    [0.169361, -0.671548, 1.138136],
    [0.258613, 0.244757, 1.49015],
    [0.858826, -0.047747, 0.927941],
    [-0.002702, -0.223393, 1.994407],
    [0.173805, 0.128224, 1.731038],
    [0.725713, -0.409052, 1.429302],
    [1.322194, 0.144995, 1.730625],
    [0.127494, -0.101339, 2.89803],
    [0.354789, -0.258214, 2.218687],
    [1.059374, -0.187433, 1.962938],
    [1.542341, -0.548129, 1.831478],
];

const BLENDER_MULTI_MASKED_INVERT: [[f64; 3]; 16] = [
    [-0.6, -0.3, 0.1],
    [-0.2, 0.25, 0.45],
    [0.145024, -0.401419, 0.420864],
    [0.602612, 0.33103, 0.209414],
    [-0.45, 0.1, 0.9],
    [0.029214, -0.555324, 1.430644],
    [0.305742, 0.263172, 1.3101],
    [0.882354, -0.049035, 0.854832],
    [-0.311411, -0.247044, 1.554934],
    [0.364247, -0.045983, 2.379342],
    [0.744231, -0.402894, 1.409376],
    [1.180725, 0.137913, 1.823661],
    [-0.183383, 0.146822, 2.240829],
    [0.669158, -0.432855, 2.124747],
    [1.022989, -0.190677, 1.992743],
    [1.0, -0.1, 2.45],
];

const BLENDER_MULTI_UNMASKED: [[f64; 3]; 16] = [
    [-0.79158, -0.788197, 0.310765],
    [-0.2, 0.25, 0.45],
    [0.145024, -0.401419, 0.420864],
    [0.310447, 0.124122, 0.387655],
    [-0.823059, -0.331568, 1.617718],
    [0.029214, -0.555324, 1.430644],
    [0.164354, 0.207929, 1.850251],
    [0.9, -0.05, 0.8],
    [0.035887, -0.220437, 2.049341],
    [0.388052, -0.067759, 2.46038],
    [0.781268, -0.390577, 1.369524],
    [1.487242, 0.153257, 1.622083],
    [0.194111, -0.154517, 3.038859],
    [0.773948, -0.491069, 2.093433],
    [1.223107, -0.172836, 1.828815],
    [1.0, -0.1, 2.45],
];

const BLENDER_MULTI_MISSING_MASK: [[f64; 3]; 16] = [
    [-0.79158, -0.788197, 0.310765],
    [-0.2, 0.25, 0.45],
    [0.145024, -0.401419, 0.420864],
    [0.310447, 0.124122, 0.387655],
    [-0.823059, -0.331568, 1.617718],
    [0.029214, -0.555324, 1.430644],
    [0.164354, 0.207929, 1.850251],
    [0.9, -0.05, 0.8],
    [0.035887, -0.220437, 2.049341],
    [0.388052, -0.067759, 2.46038],
    [0.781268, -0.390577, 1.369524],
    [1.487242, 0.153257, 1.622083],
    [0.194111, -0.154517, 3.038859],
    [0.773948, -0.491069, 2.093433],
    [1.223107, -0.172836, 1.828815],
    [1.0, -0.1, 2.45],
];

const BLENDER_CHAIN_LINEAR_DQ: [[f64; 3]; 16] = [
    [-0.6825, -1.1855, 0.382719],
    [-0.2, 0.25, 0.45],
    [0.195317, -0.608789, 0.470726],
    [0.05452, -0.204052, 0.585588],
    [-0.705072, -0.827866, 2.093157],
    [0.260258, -0.740888, 1.567743],
    [0.372384, -0.143315, 2.454087],
    [0.9, -0.05, 0.8],
    [0.523776, -0.49177, 2.139665],
    [0.950397, -0.576438, 2.407101],
    [0.993894, -0.530524, 1.078001],
    [1.910903, -0.224681, 1.096022],
    [0.965922, -0.863952, 2.771421],
    [1.118361, -0.985359, 1.523197],
    [1.615954, -0.656661, 1.229869],
    [1.0, -0.1, 2.45],
];

const ROOT_ROTATION_XYZW: [f64; 4] = [0.069034992623, 0.20710497787, 0.046023328416, 0.974794107069];
const UPPER_ROTATION_XYZW: [f64; 4] = [0.355488683981, 0.071097736796, -0.142195473592, 0.921060994003];
const LOWER_ROTATION_XYZW: [f64; 4] = [-0.0, -0.295520206661, -0.0, 0.955336489126];

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
    assert_matches_blender(&evaluated_body(&doc)?, &BLENDER_LINEAR_GROUPS, "linear groups");
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
    assert_matches_blender(&evaluated_body(&plain)?, &BLENDER_DQ_GROUPS, "dual quaternion");

    let masked = armature_doc(&[json!({
        "object":"arm","use_vertex_groups":true,"use_deform_preserve_volume":true,
        "vertex_group":"Mask"
    })])?;
    assert_matches_blender(&evaluated_body(&masked)?, &BLENDER_DQ_MASK, "dual quaternion mask");
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
    assert_matches_blender(&evaluated_body(&masked)?, &BLENDER_MULTI_MASKED, "multi masked");

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
    assert_matches_blender(&evaluated_body(&unmasked)?, &BLENDER_MULTI_UNMASKED, "multi unmasked");

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
