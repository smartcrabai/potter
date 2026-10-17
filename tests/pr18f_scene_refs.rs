#![expect(clippy::unwrap_used, clippy::expect_used, reason = "integration tests")]

//! Scene-referenced deformers checked against Blender 5.2.2 headless results.
//! Blender oracles: `/tmp/pr18f/SceneRefsEval/oracle.py`, `oracle_lattice_*`, and
//! `oracle_texture_modes.py`.
//! Subjects are a size-1 box (corners at +-0.5) with non-uniform, rotated transforms and
//! reference objects (hook, cast, origin, warp, wave start, lattice) that are also transformed.

use glam::DVec3;
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
    0.9396926164627075,
    0.34202006459236145,
    0.0,
    0.0,
    -0.34202006459236145,
    0.9396926164627075,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.20000000298023224,
    -0.30000001192092896,
    0.10000000149011612,
    1.0,
];

/// Hook with a vertex group (weights 1, 0.5, 0, 0.25, 1, 0.75 by position) and a moved hook object.
const HOOK_OBJECT: [[f64; 3]; 8] = [
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.85942018, -0.699808836, -1.626155615],
    [-0.902172685, -0.847907901, -0.186463058],
    [-0.484628558, -3.201371431, -3.038493633],
    [-0.570133567, -3.497569561, -1.15910852],
    [-0.55224824, -2.409552574, -2.588542461],
    [-0.637753248, -2.705750704, -0.709157228],
];

/// Cast (SPHERE, size 2, factor 0.5) centred on a transformed control object.
const CAST_OBJECT: [[f64; 3]; 8] = [
    [-0.52557224, -0.653433323, -1.757563114],
    [-0.529200196, -0.675201237, -1.255081892],
    [-0.526529312, 0.050907016, -1.804627895],
    [-0.529740751, -0.003457427, -1.287571192],
    [0.188328981, -0.656773508, -1.784939528],
    [0.147979736, -0.677069545, -1.273797512],
    [0.177791774, 0.042730331, -1.828381538],
    [0.141878128, -0.008237481, -1.304543257],
];

/// Simple Deform TAPER along Z with a transformed origin object.
const SIMPLE_ORIGIN: [[f64; 3]; 8] = [
    [-0.411550164, 0.567631841, -0.066393852],
    [-0.432710707, 0.990750909, 1.168603897],
    [-0.409765601, 1.882573485, 0.088823318],
    [-0.419214666, 2.475524902, 1.395445824],
    [1.382965684, 1.045860887, -0.720227718],
    [1.526098371, 1.515826344, 0.35358572],
    [1.302733302, 2.551594734, -0.377626896],
    [1.457577348, 3.191392422, 0.767812252],
];

/// Warp from one transformed empty to another (falloff NONE, strength 1).
const WARP_OBJECTS: [[f64; 3]; 8] = [
    [-0.882513583, -0.714806676, 2.095781326],
    [-0.978931665, -1.048808813, 3.015033722],
    [-0.816932619, -0.209790468, 2.939809799],
    [-0.913350821, -0.543792605, 3.859064102],
    [-0.155877918, -1.349052191, 2.39343071],
    [-0.25229609, -1.683054328, 3.312685013],
    [-0.090296984, -0.844035983, 3.23746109],
    [-0.186715156, -1.17803812, 4.156713486],
];

/// Wave at frame 10 whose start position is a transformed object (the start object is (1, -1, 0)).
const WAVE_START_OBJECT: [[f64; 3]; 8] = [
    [-0.5, -0.5, -0.146254122],
    [-0.5, -0.5, 0.853745878],
    [-0.5, 0.5, -0.285607576],
    [-0.5, 0.5, 0.714392424],
    [0.5, -0.5, -0.006906986],
    [0.5, -0.5, 0.993093014],
    [0.5, 0.5, -0.401244581],
    [0.5, 0.5, 0.598755419],
];

/// Lattice cage points (deformed positions in lattice space, u fastest), bspline interpolation.
const LATTICE_POINTS: [[f64; 3]; 8] = [
    [-0.373779356, -0.400000006, -0.5],
    [0.63639462, -0.541614711, -0.620000005],
    [-0.478832006, 0.434635639, -0.5],
    [0.386479616, 0.596017003, -0.620000005],
    [-0.643838644, -0.51455003, 0.5],
    [0.458087683, -0.583907127, 0.379999995],
    [-0.401452005, 0.584385395, 0.5],
    [0.648403764, 0.513673723, 0.379999995],
];

/// Lattice strength 0.6 with a vertex group (weights 1, 0.5, 0, 0.25, 1, 0.75 by position).
const LATTICE_STRENGTH_GROUP: [[f64; 3]; 8] = [
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.449668139, 0.493081719, -0.544151008],
    [-0.452683061, 0.495519519, 0.461997896],
    [0.584085703, -0.481330276, -0.637177765],
    [0.57810998, -0.481549263, 0.376648754],
    [0.562391579, 0.51566726, -0.612684429],
    [0.578146577, 0.50897342, 0.395640135],
];

/// Same as above with `invert_vertex_group`.
const LATTICE_STRENGTH_GROUP_INVERT: [[f64; 3]; 8] = [
    [-0.377685308, -0.502437472, -0.607570171],
    [-0.407126427, -0.491691947, 0.410931975],
    [-0.449668139, 0.493081719, -0.544151008],
    [-0.452683061, 0.495519519, 0.461997896],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
];

/// Negative strength (-0.5) without a vertex group.
const LATTICE_NEGATIVE_NO_GROUP: [[f64; 3]; 8] = [
    [-0.601929188, -0.497968763, -0.41035828],
    [-0.577394783, -0.506923437, 0.574223042],
    [-0.583886445, 0.51153034, -0.426414996],
    [-0.578861952, 0.507467508, 0.563336968],
    [0.429928631, -0.515558064, -0.385685027],
    [0.434908181, -0.515375435, 0.60279268],
    [0.448007047, 0.48694396, -0.406096339],
    [0.434877813, 0.492522329, 0.586966753],
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
                            "rotation":[0.091999677163,0.126136585176,-0.054446932243,0.98623585052],
                            "scale":[1.2,0.9,1.1]}}),
    ];
    let mut params = json!({"object":"lat","strength":strength});
    if grouped {
        operations
            .push(json!({"op":"vertex_group.create","target":{"id":"subj"},"id":"g","name":"G"}));
        params["vertex_group"] = json!("g");
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
    [-0.373779327, -0.400000006, -0.5],
    [-0.643838644, -0.51455003, 0.5],
    [-0.478832006, 0.434635639, -0.5],
    [-0.401451975, 0.584385395, 0.5],
    [0.628255427, -0.428322971, -0.523999989],
    [0.376546621, -0.528421462, 0.476000011],
    [0.49423033, 0.466911912, -0.523999989],
    [0.608519137, 0.570243061, 0.476000011],
];

const CARDINAL_OUTSIDE: [[f64; 3]; 8] = [
    [-0.374010473, -0.396782517, -0.497273594],
    [-0.646154463, -0.512974203, 0.502726376],
    [-0.475771904, 0.430969089, -0.497273594],
    [-0.402584702, 0.585991979, 0.502726376],
    [0.627972186, -0.424380392, -0.520659208],
    [0.373709023, -0.52649051, 0.479340822],
    [0.497980028, 0.462419063, -0.520659208],
    [0.607131183, 0.572211683, 0.479340822],
];

const CATMULL_ROM_OUTSIDE: [[f64; 3]; 8] = [
    [-0.373942107, -0.397734135, -0.498079985],
    [-0.645469487, -0.513440251, 0.501919985],
    [-0.476677001, 0.432053506, -0.498079985],
    [-0.402249724, 0.585516751, 0.501919985],
    [0.627767086, -0.421525419, -0.518240035],
    [0.371654212, -0.525092244, 0.481760025],
    [0.500695348, 0.459165573, -0.518240035],
    [0.60612607, 0.573637187, 0.481760025],
];

const BSPLINE_OUTSIDE: [[f64; 3]; 8] = [
    [-0.426641583, -0.439427376, -0.500160038],
    [-0.567981541, -0.486408293, 0.499839962],
    [-0.458187342, 0.479920536, -0.500160038],
    [-0.445054829, 0.550225735, 0.499839962],
    [0.574460626, -0.464168251, -0.534079969],
    [0.453325391, -0.507091522, 0.465919912],
    [0.518530726, 0.502718627, -0.534079969],
    [0.563525319, 0.538846195, 0.465919912],
];

/// Blender's scale-8 B-spline result clamps cell coordinates beyond both ends to `[-2, 2]`.
const BSPLINE_EXTREME: [[f64; 3]; 8] = [
    [-0.484222412, -0.487500012, -0.5],
    [-0.517980099, -0.501818478, 0.5],
    [-0.497354001, 0.491829515, -0.5],
    [-0.487681538, 0.510547936, 0.5],
    [0.517049134, -0.505201817, -0.514999568],
    [0.49476102, -0.510488987, 0.484999865],
    [0.485810101, 0.51200217, -0.514999568],
    [0.518550336, 0.501709282, 0.484999865],
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
    [-0.978765845, -1.0, 0.0],
    [2.387259483, -1.0, 0.0],
    [-0.887259483, 1.0, 0.0],
    [2.478765965, 1.0, 0.0],
];
const BLEND_TEXTURE_OBJECT: [[f64; 3]; 4] = [
    [-0.831708074, -1.0, 0.0],
    [1.998478651, -1.0, 0.0],
    [-0.614447474, 1.0, 0.0],
    [2.21573925, 1.0, 0.0],
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
