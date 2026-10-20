#![expect(clippy::unwrap_used, clippy::expect_used, reason = "integration tests")]

//! Scene-referenced deformers checked against Blender 5.2.2 headless results.
//! Blender oracles: `fixtures/pr18f_scene_refs/oracle.py`, `oracle_lattice_*`, and
//! `oracle_texture_modes.py` (each with its captured JSON and log).
//! Subjects are a size-1 box (corners at +-0.5) with non-uniform, rotated transforms and
//! reference objects (hook, cast, origin, warp, wave start, lattice) that are also transformed.

use glam::{DMat4, DVec3};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::{Id, SceneDoc},
    ops::apply_batch,
};
use serde_json::{Value, json};

/// Tolerance for Blender's float32 evaluation of the oracle values.
const TOLERANCE: f64 = 2.0e-6;

/// Box corners in the oracle's order (x outer, then y, then z).
const CORNERS: [[f64; 3]; 8] = [
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
];

/// Hook `matrix_inverse` in Blender's column-major layout.
const HOOK_OBJECT_MATRIX_INVERSE: [f64; 16] = [
    0.939_692_616_462_707_5,
    0.342_020_064_592_361_45,
    0.0,
    0.0,
    -0.342_020_064_592_361_45,
    0.939_692_616_462_707_5,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.200_000_002_980_232_24,
    -0.300_000_011_920_928_96,
    0.100_000_001_490_116_12,
    1.0,
];

/// Hook with a vertex group (weights 1, 0.5, 0, 0.25, 1, 0.75 by position) and a moved hook object.
const HOOK_OBJECT: [[f64; 3]; 8] = [
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.859_420_18, -0.699_808_836, -1.626_155_615],
    [-0.902_172_685, -0.847_907_901, -0.186_463_058],
    [-0.484_628_558, -3.201_371_431, -3.038_493_633],
    [-0.570_133_567, -3.497_569_561, -1.159_108_52],
    [-0.552_248_24, -2.409_552_574, -2.588_542_461],
    [-0.637_753_248, -2.705_750_704, -0.709_157_228],
];

/// Cast (SPHERE, size 2, factor 0.5) centred on a transformed control object.
const CAST_OBJECT: [[f64; 3]; 8] = [
    [-0.525_572_24, -0.653_433_323, -1.757_563_114],
    [-0.529_200_196, -0.675_201_237, -1.255_081_892],
    [-0.526_529_312, 0.050_907_016, -1.804_627_895],
    [-0.529_740_751, -0.003_457_427, -1.287_571_192],
    [0.188_328_981, -0.656_773_508, -1.784_939_528],
    [0.147_979_736, -0.677_069_545, -1.273_797_512],
    [0.177_791_774, 0.042_730_331, -1.828_381_538],
    [0.141_878_128, -0.008_237_481, -1.304_543_257],
];

/// Simple Deform TAPER along Z with a transformed origin object.
const SIMPLE_ORIGIN: [[f64; 3]; 8] = [
    [-0.411_550_164, 0.567_631_841, -0.066_393_852],
    [-0.432_710_707, 0.990_750_909, 1.168_603_897],
    [-0.409_765_601, 1.882_573_485, 0.088_823_318],
    [-0.419_214_666, 2.475_524_902, 1.395_445_824],
    [1.382_965_684, 1.045_860_887, -0.720_227_718],
    [1.526_098_371, 1.515_826_344, 0.353_585_72],
    [1.302_733_302, 2.551_594_734, -0.377_626_896],
    [1.457_577_348, 3.191_392_422, 0.767_812_252],
];

/// Warp from one transformed empty to another (falloff NONE, strength 1).
const WARP_OBJECTS: [[f64; 3]; 8] = [
    [-0.882_513_583, -0.714_806_676, 2.095_781_326],
    [-0.978_931_665, -1.048_808_813, 3.015_033_722],
    [-0.816_932_619, -0.209_790_468, 2.939_809_799],
    [-0.913_350_821, -0.543_792_605, 3.859_064_102],
    [-0.155_877_918, -1.349_052_191, 2.393_430_71],
    [-0.252_296_09, -1.683_054_328, 3.312_685_013],
    [-0.090_296_984, -0.844_035_983, 3.237_461_09],
    [-0.186_715_156, -1.178_038_12, 4.156_713_486],
];

/// Wave at frame 10 whose start position is a transformed object (the start object is (1, -1, 0)).
const WAVE_START_OBJECT: [[f64; 3]; 8] = [
    [-0.5, -0.5, -0.146_254_122],
    [-0.5, -0.5, 0.853_745_878],
    [-0.5, 0.5, -0.285_607_576],
    [-0.5, 0.5, 0.714_392_424],
    [0.5, -0.5, -0.006_906_986],
    [0.5, -0.5, 0.993_093_014],
    [0.5, 0.5, -0.401_244_581],
    [0.5, 0.5, 0.598_755_419],
];

/// Lattice cage points (deformed positions in lattice space, u fastest), bspline interpolation.
const LATTICE_POINTS: [[f64; 3]; 8] = [
    [-0.373_779_356, -0.400_000_006, -0.5],
    [0.636_394_62, -0.541_614_711, -0.620_000_005],
    [-0.478_832_006, 0.434_635_639, -0.5],
    [0.386_479_616, 0.596_017_003, -0.620_000_005],
    [-0.643_838_644, -0.514_550_03, 0.5],
    [0.458_087_683, -0.583_907_127, 0.379_999_995],
    [-0.401_452_005, 0.584_385_395, 0.5],
    [0.648_403_764, 0.513_673_723, 0.379_999_995],
];

/// Lattice strength 0.6 with a vertex group (weights 1, 0.5, 0, 0.25, 1, 0.75 by position).
const LATTICE_STRENGTH_GROUP: [[f64; 3]; 8] = [
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.449_668_139, 0.493_081_719, -0.544_151_008],
    [-0.452_683_061, 0.495_519_519, 0.461_997_896],
    [0.584_085_703, -0.481_330_276, -0.637_177_765],
    [0.578_109_98, -0.481_549_263, 0.376_648_754],
    [0.562_391_579, 0.515_667_26, -0.612_684_429],
    [0.578_146_577, 0.508_973_42, 0.395_640_135],
];

/// Same as above with `invert_vertex_group`.
const LATTICE_STRENGTH_GROUP_INVERT: [[f64; 3]; 8] = [
    [-0.377_685_308, -0.502_437_472, -0.607_570_171],
    [-0.407_126_427, -0.491_691_947, 0.410_931_975],
    [-0.449_668_139, 0.493_081_719, -0.544_151_008],
    [-0.452_683_061, 0.495_519_519, 0.461_997_896],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
];

/// Negative strength (-0.5) without a vertex group.
const LATTICE_NEGATIVE_NO_GROUP: [[f64; 3]; 8] = [
    [-0.601_929_188, -0.497_968_763, -0.410_358_28],
    [-0.577_394_783, -0.506_923_437, 0.574_223_042],
    [-0.583_886_445, 0.511_530_34, -0.426_414_996],
    [-0.578_861_952, 0.507_467_508, 0.563_336_968],
    [0.429_928_631, -0.515_558_064, -0.385_685_027],
    [0.434_908_181, -0.515_375_435, 0.602_792_68],
    [0.448_007_047, 0.486_943_96, -0.406_096_339],
    [0.434_877_813, 0.492_522_329, 0.586_966_753],
];

fn apply(doc: &SceneDoc, operations: &Value) -> SceneDoc {
    apply_batch(
        doc,
        &json!({
            "schema_version": 1,
            "base_revision": doc.revision,
            "operations": operations,
        }),
    )
    .unwrap()
    .doc
}

fn base_scene() -> SceneDoc {
    SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned())
}

fn evaluated_mesh(doc: &SceneDoc, node: &str, frame: Option<f64>) -> Mesh {
    let snapshot = Snapshot::evaluate(
        doc,
        &EvaluationContext {
            frame,
            ..EvaluationContext::default()
        },
    )
    .unwrap();
    snapshot.meshes[&Id::new(node).unwrap()].clone()
}

fn source_mesh<'a>(doc: &'a SceneDoc, node: &str) -> &'a Mesh {
    let data = doc.nodes[&Id::new(node).unwrap()].data.as_ref().unwrap();
    doc.data_blocks[data].mesh.as_ref().unwrap()
}

/// Vertex-group weights keyed by position: 1 for x > 0, 0.5 for x <= 0 and y > 0, else unassigned.
fn group_assignments(mesh: &Mesh) -> Value {
    let weights = mesh
        .vertices
        .iter()
        .filter_map(|vertex| {
            let weight = if vertex.co.x > 0.0 {
                1.0
            } else if vertex.co.y > 0.0 {
                0.5
            } else {
                0.0
            };
            (weight > 0.0).then(|| json!({"vertex_id": vertex.id, "weight": weight}))
        })
        .collect::<Vec<_>>();
    json!(weights)
}

/// Compares each evaluated vertex with the Blender value of the box corner it started from.
fn assert_corner_positions(result: &Mesh, source: &Mesh, expected: &[[f64; 3]; 8]) {
    assert_eq!(result.vertices.len(), expected.len());
    for (vertex, original) in result.vertices.iter().zip(&source.vertices) {
        let corner = CORNERS
            .iter()
            .position(|corner| DVec3::from_array(*corner).distance(original.co) < 1.0e-9)
            .expect("source vertex is a box corner");
        let want = DVec3::from_array(expected[corner]);
        assert!(
            vertex.co.distance(want) < TOLERANCE,
            "corner {corner}: {:?} differs from Blender {:?}",
            vertex.co,
            want
        );
    }
}

#[test]
fn hook_object_uses_inverse_subject_times_hook_matrix_inverse_and_weights() {
    let staged = apply(
        &base_scene(),
        &json!([
            {"op":"node.create","id":"subj","kind":"box","params":{"size":1.0},
             "transform":{"translation":[1.0,2.0,3.0],"rotation_deg":[0.0,0.0,30.0],"scale":[2.0,1.0,0.5]}},
            {"op":"node.create","id":"hk","kind":"empty",
             "transform":{"translation":[0.5,-1.0,2.0],"rotation_deg":[20.0,0.0,0.0],"scale":[1.5,0.7,1.0]}},
            {"op":"vertex_group.create","target":{"id":"subj"},"id":"h","name":"H"}
        ]),
    );
    let doc = apply(
        &staged,
        &json!([
            {"op":"vertex_group.assign","target":{"id":"subj"},"group_id":"h",
             "weights":group_assignments(source_mesh(&staged, "subj"))},
            {"op":"modifier.create","target":{"id":"subj"},"id":"hook","type":"hook",
             "params":{"object":"hk","vertex_group":"H","falloff_type":"NONE","strength":1.0,
                       "matrix_inverse":HOOK_OBJECT_MATRIX_INVERSE}}
        ]),
    );
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &HOOK_OBJECT,
    );
}

#[test]
fn cast_object_centres_on_the_transformed_control_object() {
    let doc = apply(
        &base_scene(),
        &json!([
            {"op":"node.create","id":"subj","kind":"box","params":{"size":1.0},
             "transform":{"translation":[1.0,2.0,3.0],"rotation_deg":[0.0,0.0,30.0],"scale":[2.0,1.0,0.5]}},
            {"op":"node.create","id":"ct","kind":"empty",
             "transform":{"translation":[0.5,0.5,0.5],"rotation_deg":[0.0,0.0,40.0],"scale":[1.3,0.8,1.0]}},
            {"op":"modifier.create","target":{"id":"subj"},"id":"cast","type":"cast",
             "params":{"cast_type":"SPHERE","factor":0.5,"radius":0.0,"size":2.0,
                       "use_radius_as_size":false,"object":"ct","use_transform":false}}
        ]),
    );
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &CAST_OBJECT,
    );
}

#[test]
fn simple_deform_origin_uses_the_origin_object_transform() {
    let doc = apply(
        &base_scene(),
        &json!([
            {"op":"node.create","id":"subj","kind":"box","params":{"size":1.0},
             "transform":{"translation":[1.0,2.0,3.0],"rotation_deg":[0.0,0.0,30.0],"scale":[2.0,1.0,0.5]}},
            {"op":"node.create","id":"og","kind":"empty",
             "transform":{"translation":[0.5,-0.5,1.0],"rotation_deg":[11.5,17.2,22.9],"scale":[1.2,0.9,1.1]}},
            {"op":"modifier.create","target":{"id":"subj"},"id":"simple","type":"simple_deform",
             "params":{"deform_method":"TAPER","deform_axis":"Z","factor":0.5,"origin":"og"}}
        ]),
    );
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &SIMPLE_ORIGIN,
    );
}

#[test]
fn warp_objects_map_from_and_to_through_the_subject_frame() {
    let doc = apply(
        &base_scene(),
        &json!([
            {"op":"node.create","id":"subj","kind":"box","params":{"size":1.0},
             "transform":{"translation":[1.0,2.0,3.0],"rotation_deg":[0.0,0.0,30.0],"scale":[2.0,1.0,0.5]}},
            {"op":"node.create","id":"wf","kind":"empty",
             "transform":{"translation":[1.0,0.0,0.0],"rotation_deg":[0.0,0.0,20.0],"scale":[1.0,1.5,1.0]}},
            {"op":"node.create","id":"wt","kind":"empty",
             "transform":{"translation":[0.0,2.0,1.0],"rotation_deg":[40.0,0.0,0.0],"scale":[0.8,1.0,1.2]}},
            {"op":"modifier.create","target":{"id":"subj"},"id":"warp","type":"warp",
             "params":{"object_from":"wf","object_to":"wt","falloff_type":"NONE","strength":1.0}}
        ]),
    );
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &WARP_OBJECTS,
    );
}

fn warp_bone_scene(object_from: &str, bone_from: Option<&str>) -> SceneDoc {
    let mut params = json!({
        "object_from": object_from,
        "object_to": "wt",
        "falloff_type": "NONE",
        "strength": 1.0
    });
    if let Some(bone) = bone_from {
        params["bone_from"] = json!(bone);
    }
    apply(
        &base_scene(),
        &json!([
            {"op":"node.create","id":"subj","kind":"box","params":{"size":1.0},
             "transform":{"translation":[1.0,2.0,3.0],"rotation_deg":[0.0,0.0,30.0],"scale":[2.0,1.0,0.5]}},
            {"op":"node.create","id":"rig","kind":"armature","transform":{"translation":[1.0,0.0,0.0]}},
            {"op":"bone.create","target":{"id":"rig"},"id":"bone_a","name":"Arm.L",
             "head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
            {"op":"pose.set","target":{"id":"rig"},"bone_id":"bone_a","set":{"translation":[0.0,0.5,0.25]}},
            {"op":"node.create","id":"wf","kind":"empty","transform":{"translation":[1.0,0.0,0.0]}},
            {"op":"node.create","id":"wt","kind":"empty",
             "transform":{"translation":[0.0,2.0,1.0],"rotation_deg":[40.0,0.0,0.0]}},
            {"op":"modifier.create","target":{"id":"subj"},"id":"warp","type":"warp","params":params}
        ]),
    )
}

fn warp_positions(doc: &SceneDoc) -> Vec<DVec3> {
    evaluated_mesh(doc, "subj", None)
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect()
}

fn max_distance(actual: &[DVec3], expected: &[DVec3]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(vertex, want)| vertex.distance(*want))
        .fold(0.0, f64::max)
}

fn assert_same_positions(actual: &[DVec3], expected: &[DVec3]) {
    let distance = max_distance(actual, expected);
    assert!(distance < TOLERANCE, "positions differ by {distance}");
}

fn assert_differs_from(actual: &[DVec3], other: &[DVec3]) {
    let distance = max_distance(actual, other);
    assert!(
        distance > 1.0e-3,
        "posed bone matches the object fallback ({distance})"
    );
}

#[test]
fn warp_posed_bone_resolves_by_name_and_id_to_the_same_output() {
    let by_name = warp_positions(&warp_bone_scene("rig", Some("Arm.L")));
    let by_id = warp_positions(&warp_bone_scene("rig", Some("bone_a")));
    assert_same_positions(&by_name, &by_id);
}

#[test]
fn warp_posed_bone_differs_from_the_object_matrix_fallback() {
    let posed = warp_positions(&warp_bone_scene("rig", Some("Arm.L")));
    let object = warp_positions(&warp_bone_scene("rig", None));
    assert_differs_from(&posed, &object);
}

#[test]
fn warp_missing_bone_name_uses_the_armature_object_matrix() {
    assert_same_positions(
        &warp_positions(&warp_bone_scene("rig", Some("Missing"))),
        &warp_positions(&warp_bone_scene("rig", None)),
    );
}

#[test]
fn warp_bone_name_on_a_non_armature_uses_the_object_matrix() {
    assert_same_positions(
        &warp_positions(&warp_bone_scene("wf", Some("Arm.L"))),
        &warp_positions(&warp_bone_scene("wf", None)),
    );
}

/// A hook that follows a Blender bone name, which Potter ids cannot hold when it has uppercase.
fn hook_bone_scene(subtarget: Option<&str>) -> SceneDoc {
    let mut params = json!({
        "object": "rig",
        "vertex_group": "H",
        "falloff_type": "NONE",
        "strength": 1.0,
        "matrix_inverse": [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0]
    });
    if let Some(bone) = subtarget {
        params["subtarget"] = json!(bone);
    }
    let staged = apply(
        &base_scene(),
        &json!([
            {"op":"node.create","id":"subj","kind":"box","params":{"size":1.0},
             "transform":{"translation":[1.0,2.0,3.0],"rotation_deg":[0.0,0.0,30.0],"scale":[2.0,1.0,0.5]}},
            {"op":"node.create","id":"rig","kind":"armature","transform":{"translation":[1.0,0.0,0.0]}},
            {"op":"bone.create","target":{"id":"rig"},"id":"bone_1","name":"Bone",
             "head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
            {"op":"pose.set","target":{"id":"rig"},"bone_id":"bone_1","set":{"translation":[0.0,0.5,0.25]}},
            {"op":"vertex_group.create","target":{"id":"subj"},"id":"h","name":"H"}
        ]),
    );
    apply(
        &staged,
        &json!([
            {"op":"vertex_group.assign","target":{"id":"subj"},"group_id":"h",
             "weights":group_assignments(source_mesh(&staged, "subj"))},
            {"op":"modifier.create","target":{"id":"subj"},"id":"hook","type":"hook","params":params}
        ]),
    )
}

#[test]
fn hook_follows_imported_bone_name_with_object_fallback() {
    let object = warp_positions(&hook_bone_scene(None));
    let by_id = warp_positions(&hook_bone_scene(Some("bone_1")));
    let by_name = warp_positions(&hook_bone_scene(Some("Bone")));
    let missing = warp_positions(&hook_bone_scene(Some("Missing")));
    assert_same_positions(&by_name, &by_id);
    assert_differs_from(&by_name, &object);
    assert_same_positions(&missing, &object);
}

#[test]
fn wave_start_position_object_uses_the_subject_relative_object_location() {
    let doc = apply(
        &base_scene(),
        &json!([
            {"op":"node.create","id":"subj","kind":"box","params":{"size":1.0},
             "transform":{"translation":[1.0,2.0,3.0],"rotation_deg":[0.0,0.0,30.0],"scale":[2.0,1.0,0.5]}},
            {"op":"node.create","id":"ws","kind":"empty","transform":{"translation":[1.0,-1.0,0.0]}},
            {"op":"modifier.create","target":{"id":"subj"},"id":"wave","type":"wave",
             "params":{"start_position_object":"ws"}}
        ]),
    );
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", Some(10.0)),
        source_mesh(&doc, "subj"),
        &WAVE_START_OBJECT,
    );
}

fn lattice_scene(strength: f64, grouped: bool, invert: bool, missing_group: bool) -> SceneDoc {
    let mut operations = vec![
        json!({"op":"node.create","id":"subj","kind":"box","params":{"size":1.0},
               "transform":{"translation":[-0.2,0.1,0.1],"rotation_deg":[0.0,0.0,25.0],"scale":[0.26,0.24,0.26]}}),
        json!({"op":"lattice.create","id":"lat","points":LATTICE_POINTS,
               "domain":{"min":[-0.5,-0.5,-0.5],"max":[0.5,0.5,0.5]},"interpolation":"bspline",
               "transform":{"translation":[0.2,-0.1,0.3],
                            "rotation":[0.091_999_677_163,0.126_136_585_176,-0.054_446_932_243,0.986_235_850_52],
                            "scale":[1.2,0.9,1.1]}}),
    ];
    let mut params = json!({"object":"lat","strength":strength});
    if grouped {
        operations
            .push(json!({"op":"vertex_group.create","target":{"id":"subj"},"id":"g","name":"G"}));
        params["vertex_group"] = json!("G");
        params["invert_vertex_group"] = json!(invert);
    } else if missing_group {
        params["vertex_group"] = json!("missing");
    }
    let staged = apply(&base_scene(), &json!(operations));
    let mut second = Vec::new();
    if grouped {
        second.push(
            json!({"op":"vertex_group.assign","target":{"id":"subj"},"group_id":"g",
                           "weights":group_assignments(source_mesh(&staged, "subj"))}),
        );
    }
    second.push(
        json!({"op":"modifier.create","target":{"id":"subj"},"id":"lattice","type":"lattice",
                       "params":params}),
    );
    apply(&staged, &json!(second))
}

#[test]
fn lattice_object_blends_grid_displacement_by_strength_and_vertex_group() {
    let doc = lattice_scene(0.6, true, false, false);
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &LATTICE_STRENGTH_GROUP,
    );
}

#[test]
fn lattice_invert_vertex_group_moves_the_complementary_weights() {
    let doc = lattice_scene(0.6, true, true, false);
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &LATTICE_STRENGTH_GROUP_INVERT,
    );
}

#[test]
fn lattice_negative_strength_without_group_reverses_the_displacement() {
    let doc = lattice_scene(-0.5, false, false, false);
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &LATTICE_NEGATIVE_NO_GROUP,
    );
}

#[test]
fn lattice_missing_vertex_group_keeps_full_blender_influence() {
    let doc = lattice_scene(-0.5, false, false, true);
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &LATTICE_NEGATIVE_NO_GROUP,
    );
}
/// Blender 5.2.2 fractional out-of-cage oracle; left corners map to lattice cell coordinate -0.8.
const LINEAR_OUTSIDE: [[f64; 3]; 8] = [
    [-0.373_779_327, -0.400_000_006, -0.5],
    [-0.643_838_644, -0.514_550_03, 0.5],
    [-0.478_832_006, 0.434_635_639, -0.5],
    [-0.401_451_975, 0.584_385_395, 0.5],
    [0.628_255_427, -0.428_322_971, -0.523_999_989],
    [0.376_546_621, -0.528_421_462, 0.476_000_011],
    [0.494_230_33, 0.466_911_912, -0.523_999_989],
    [0.608_519_137, 0.570_243_061, 0.476_000_011],
];

const CARDINAL_OUTSIDE: [[f64; 3]; 8] = [
    [-0.374_010_473, -0.396_782_517, -0.497_273_594],
    [-0.646_154_463, -0.512_974_203, 0.502_726_376],
    [-0.475_771_904, 0.430_969_089, -0.497_273_594],
    [-0.402_584_702, 0.585_991_979, 0.502_726_376],
    [0.627_972_186, -0.424_380_392, -0.520_659_208],
    [0.373_709_023, -0.526_490_51, 0.479_340_822],
    [0.497_980_028, 0.462_419_063, -0.520_659_208],
    [0.607_131_183, 0.572_211_683, 0.479_340_822],
];

const CATMULL_ROM_OUTSIDE: [[f64; 3]; 8] = [
    [-0.373_942_107, -0.397_734_135, -0.498_079_985],
    [-0.645_469_487, -0.513_440_251, 0.501_919_985],
    [-0.476_677_001, 0.432_053_506, -0.498_079_985],
    [-0.402_249_724, 0.585_516_751, 0.501_919_985],
    [0.627_767_086, -0.421_525_419, -0.518_240_035],
    [0.371_654_212, -0.525_092_244, 0.481_760_025],
    [0.500_695_348, 0.459_165_573, -0.518_240_035],
    [0.606_126_07, 0.573_637_187, 0.481_760_025],
];

const BSPLINE_OUTSIDE: [[f64; 3]; 8] = [
    [-0.426_641_583, -0.439_427_376, -0.500_160_038],
    [-0.567_981_541, -0.486_408_293, 0.499_839_962],
    [-0.458_187_342, 0.479_920_536, -0.500_160_038],
    [-0.445_054_829, 0.550_225_735, 0.499_839_962],
    [0.574_460_626, -0.464_168_251, -0.534_079_969],
    [0.453_325_391, -0.507_091_522, 0.465_919_912],
    [0.518_530_726, 0.502_718_627, -0.534_079_969],
    [0.563_525_319, 0.538_846_195, 0.465_919_912],
];

/// Blender's scale-8 B-spline result clamps cell coordinates beyond both ends to `[-2, 2]`.
const BSPLINE_EXTREME: [[f64; 3]; 8] = [
    [-0.484_222_412, -0.487_500_012, -0.5],
    [-0.517_980_099, -0.501_818_478, 0.5],
    [-0.497_354_001, 0.491_829_515, -0.5],
    [-0.487_681_538, 0.510_547_936, 0.5],
    [0.517_049_134, -0.505_201_817, -0.514_999_568],
    [0.494_761_02, -0.510_488_987, 0.484_999_865],
    [0.485_810_101, 0.512_002_17, -0.514_999_568],
    [0.518_550_336, 0.501_709_282, 0.484_999_865],
];

fn outside_cage_lattice_scene(interpolation: &str, scale: f64, translation: [f64; 3]) -> SceneDoc {
    apply(
        &base_scene(),
        &json!([
            {"op":"node.create","id":"subj","kind":"box","params":{"size":1.0},
             "transform":{"translation":translation,"scale":[scale,scale,scale]}},
            {"op":"lattice.create","id":"lat","points":LATTICE_POINTS,
             "domain":{"min":[-0.5,-0.5,-0.5],"max":[0.5,0.5,0.5]},
             "interpolation":interpolation},
            {"op":"modifier.create","target":{"id":"subj"},"id":"lattice","type":"lattice",
             "params":{"object":"lat","strength":1.0}}
        ]),
    )
}

#[test]
fn lattice_linear_extrapolates_outside_the_cage() {
    let doc = outside_cage_lattice_scene("linear", 1.0, [-0.8, 0.0, 0.0]);
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &LINEAR_OUTSIDE,
    );
}

#[test]
fn lattice_cardinal_extrapolates_outside_the_cage() {
    let doc = outside_cage_lattice_scene("cardinal", 1.0, [-0.8, 0.0, 0.0]);
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &CARDINAL_OUTSIDE,
    );
}

#[test]
fn lattice_catmull_rom_extrapolates_outside_the_cage() {
    let doc = outside_cage_lattice_scene("catmull_rom", 1.0, [-0.8, 0.0, 0.0]);
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &CATMULL_ROM_OUTSIDE,
    );
}

#[test]
fn lattice_bspline_extrapolates_outside_the_cage() {
    let doc = outside_cage_lattice_scene("bspline", 1.0, [-0.8, 0.0, 0.0]);
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &BSPLINE_OUTSIDE,
    );
}

#[test]
fn lattice_bspline_holds_basis_clamp_at_extreme_outside_cage_coordinates() {
    let doc = outside_cage_lattice_scene("bspline", 8.0, [0.0, 0.0, 0.0]);
    assert_corner_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &BSPLINE_EXTREME,
    );
}
/// Blender's diagonal Blend texture sampled in LOCAL, GLOBAL, and OBJECT coordinates.
const BLEND_TEXTURE_LOCAL: [[f64; 3]; 4] = [
    [-1.5, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.5, 1.0, 0.0],
];
const BLEND_TEXTURE_GLOBAL: [[f64; 3]; 4] = [
    [-0.978_765_845, -1.0, 0.0],
    [2.387_259_483, -1.0, 0.0],
    [-0.887_259_483, 1.0, 0.0],
    [2.478_765_965, 1.0, 0.0],
];
const BLEND_TEXTURE_OBJECT: [[f64; 3]; 4] = [
    [-0.831_708_074, -1.0, 0.0],
    [1.998_478_651, -1.0, 0.0],
    [-0.614_447_474, 1.0, 0.0],
    [2.215_739_25, 1.0, 0.0],
];
const PLANE_CORNERS: [[f64; 3]; 4] = [
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
];

fn blend_texture_scene(texture_coords: &str) -> SceneDoc {
    let mut operations = vec![
        json!({"op":"node.create","id":"subj","kind":"plane","params":{"size":2.0},
               "transform":{"translation":[1.0,2.0,3.0],"rotation_deg":[0.0,0.0,30.0],"scale":[2.0,0.5,1.5]}}),
        json!({"op":"texture.create","id":"blend_tex","texture_type":"BLEND",
               "settings":{"progression":"DIAGONAL"}}),
    ];
    if texture_coords == "OBJECT" {
        operations.push(json!({"op":"node.create","id":"map","kind":"empty",
                               "transform":{"translation":[0.3,-0.2,0.5],"rotation_deg":[0.0,0.0,20.0],"scale":[1.5,1.0,0.8]}}));
    }
    let mut params = json!({
        "direction":"X",
        "strength":1.0,
        "mid_level":0.5,
        "texture":"blend_tex",
        "texture_coords":texture_coords
    });
    if texture_coords == "OBJECT" {
        params["texture_coords_object"] = json!("map");
    }
    operations.push(
        json!({"op":"modifier.create","target":{"id":"subj"},"id":"displace","type":"displace",
                           "params":params}),
    );
    apply(&base_scene(), &json!(operations))
}

fn assert_plane_positions(result: &Mesh, source: &Mesh, expected: &[[f64; 3]; 4]) {
    assert_eq!(result.vertices.len(), expected.len());
    for (vertex, original) in result.vertices.iter().zip(&source.vertices) {
        let corner = PLANE_CORNERS
            .iter()
            .position(|corner| DVec3::from_array(*corner).distance(original.co) < 1.0e-9)
            .expect("source vertex is a plane corner");
        let want = DVec3::from_array(expected[corner]);
        assert!(
            vertex.co.distance(want) < TOLERANCE,
            "plane corner {corner}: {:?} differs from Blender {:?}",
            vertex.co,
            want
        );
    }
}

#[test]
fn displace_blend_texture_uses_local_coordinates() {
    let doc = blend_texture_scene("LOCAL");
    assert_plane_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &BLEND_TEXTURE_LOCAL,
    );
}

#[test]
fn displace_blend_texture_uses_subject_world_coordinates() {
    let doc = blend_texture_scene("GLOBAL");
    assert_plane_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &BLEND_TEXTURE_GLOBAL,
    );
}

#[test]
fn displace_blend_texture_uses_mapping_object_coordinates() {
    let doc = blend_texture_scene("OBJECT");
    assert_plane_positions(
        &evaluated_mesh(&doc, "subj", None),
        source_mesh(&doc, "subj"),
        &BLEND_TEXTURE_OBJECT,
    );
}

fn vertex_positions(mesh: &Mesh) -> Vec<DVec3> {
    mesh.vertices.iter().map(|vertex| vertex.co).collect()
}

#[test]
fn array_fit_curve_evaluates_the_curve_path_in_preview() {
    let staged = apply(
        &base_scene(),
        &json!([
            {"op":"curve.create","id":"path","dimensions":"three_d","transform":{"scale":[2.0,2.0,2.0]},
             "splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[2.0,0.0,0.0]}],
                         "order":2,"cyclic":false,"resolution":1,"use_endpoint":true}]},
            {"op":"node.create","id":"subj","kind":"box","params":{"size":1.0}},
            {"op":"modifier.create","target":{"id":"subj"},"id":"array","type":"array",
             "params":{"count":2,"fit_type":"FIT_CURVE","curve":"path","use_relative_offset":true,
                       "relative_offset_displace":[1.0,0.0,0.0],"use_constant_offset":false}}
        ]),
    );
    let preview = vertex_positions(&evaluated_mesh(&staged, "subj", None));
    // The preview must be non-trivial: the array repeats the source box.
    assert!(
        preview.len() > source_mesh(&staged, "subj").vertices.len(),
        "preview did not run the array modifier"
    );
    // Applying the same array must produce exactly the geometry the preview showed.
    let applied_doc = apply(
        &staged,
        &json!([{"op":"modifier.apply","target":{"id":"subj"},"modifier_id":"array"}]),
    );
    let applied = vertex_positions(source_mesh(&applied_doc, "subj"));
    assert_same_positions(&preview, &applied);
}

/// A nested scene Hook must retain the carrier across the first Hook, and the internal key must not
/// overwrite a legacy canonical user attribute named `blender_origindex`.
#[test]
fn nested_listed_hooks_after_array_preserve_origindex_and_user_attribute() {
    let mut doc = apply(
        &base_scene(),
        &json!([
            {"op":"node.create","id":"subj","kind":"box","params":{"size":1.0}},
            {"op":"node.create","id":"hook_x","kind":"empty","transform":{"translation":[1.0,0.0,0.0]}},
            {"op":"node.create","id":"hook_y","kind":"empty","transform":{"translation":[0.0,1.0,0.0]}},
            {"op":"mesh.attribute_create","target":{"id":"subj"},"name":"blender_origindex","domain":"point","type":"int"},
            {"op":"modifier.create","target":{"id":"subj"},"id":"array","type":"array","params":{"count":2,"use_relative_offset":true,"relative_offset_displace":[1.0,0.0,0.0],"use_constant_offset":false}},
            {"op":"modifier.create","target":{"id":"subj"},"id":"hook_x_mod","type":"hook","params":{"object":"hook_x","vertex_indices":[0],"falloff_type":"NONE","falloff_radius":0.0,"strength":1.0,"center":[0.0,0.0,0.0],"use_falloff_uniform":false,"matrix_inverse":DMat4::IDENTITY.to_cols_array()}},
            {"op":"modifier.create","target":{"id":"subj"},"id":"hook_y_mod","type":"hook","params":{"object":"hook_y","vertex_indices":[0],"falloff_type":"NONE","falloff_radius":0.0,"strength":1.0,"center":[0.0,0.0,0.0],"use_falloff_uniform":false,"matrix_inverse":DMat4::IDENTITY.to_cols_array()}}
        ]),
    );
    let data_id = doc.nodes[&Id::new("subj").unwrap()].data.clone().unwrap();
    let source_mesh = doc
        .data_blocks
        .get_mut(&data_id)
        .unwrap()
        .mesh
        .as_mut()
        .unwrap();
    let user_values = source_mesh.attributes.get_mut("blender_origindex").unwrap()["values"]
        .as_object_mut()
        .unwrap();
    for (index, vertex) in source_mesh.vertices.iter().enumerate() {
        user_values.insert(format!("v{}", vertex.id), json!(73 + index as i64));
    }
    let hash = |doc: &SceneDoc| {
        potter_core::hash::sha256(
            &potter_core::hash::canonicalize(&serde_json::to_value(doc).unwrap()).unwrap(),
        )
    };
    let before_hash = hash(&doc);
    let mut array_only = doc.clone();
    array_only
        .nodes
        .get_mut(&Id::new("subj").unwrap())
        .unwrap()
        .modifiers
        .retain(|modifier| modifier.id == Id::new("array").unwrap());
    let before = evaluated_mesh(&array_only, "subj", None);
    let after = evaluated_mesh(&doc, "subj", None);
    assert_eq!(
        hash(&doc),
        before_hash,
        "evaluation must not mutate source history"
    );
    assert_eq!(before.vertices.len(), 16);
    assert_eq!(after.vertices.len(), before.vertices.len());
    let moved: Vec<usize> = before
        .vertices
        .iter()
        .zip(&after.vertices)
        .enumerate()
        .filter_map(|(index, (before, after))| {
            ((after.co - before.co).length() > TOLERANCE).then_some(index)
        })
        .collect();
    assert_eq!(moved, vec![0, 8]);
    for index in moved {
        assert!(
            (after.vertices[index].co - before.vertices[index].co)
                .distance(DVec3::new(1.0, 1.0, 0.0))
                < TOLERANCE
        );
    }
    assert_eq!(
        after.attributes.get("blender_origindex"),
        before.attributes.get("blender_origindex"),
        "the carrier must not overwrite or strip the user's canonical attribute"
    );
    let user_attribute = &after.attributes["blender_origindex"]["values"];
    assert_eq!(
        user_attribute.as_object().unwrap().len(),
        after.vertices.len()
    );
}
