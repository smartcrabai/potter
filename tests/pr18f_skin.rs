#![expect(
    clippy::unwrap_used,
    reason = "integration tests use concise assertions"
)]
#![expect(clippy::expect_used, reason = "integration test fixtures")]

//! Skin parity against Blender 5.2.2. Expected geometry is captured by
//! `tests/fixtures/pr18f_skin_oracle.py`; vertices match by position and faces by vertex cycle,
//! so storage order does not matter.

use std::collections::{BTreeMap, HashMap};

use glam::DVec3;
use potter_core::{
    error::ErrorCode,
    geom::{
        Mesh,
        modifiers::{evaluate_modifiers, evaluate_modifiers_with_groups},
    },
    model::{Id, Modifier, ModifierRuntime},
};
use serde_json::{Value, json};

const SUBJECT_NAME: &str = "SkinSubject";

const LINE_INPUT_VERTICES: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.5], [1.0, 0.0, 1.0]];
const LINE_EDGES: [[u32; 2]; 2] = [[0, 1], [1, 2]];
const LINE_SKIN: [([f64; 2], bool); 3] = [
    ([0.25, 0.25], true),
    ([0.25, 0.25], false),
    ([0.25, 0.25], false),
];
const LINE_NO_ROOT_SKIN: [([f64; 2], bool); 3] = [
    ([0.25, 0.25], false),
    ([0.25, 0.25], false),
    ([0.25, 0.25], false),
];

const BENT_INPUT_VERTICES: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.5],
    [2.0, 1.0, 2.0],
];
const BENT_EDGES: [[u32; 2]; 3] = [[0, 1], [1, 2], [2, 3]];
const BENT_SKIN: [([f64; 2], bool); 4] = [
    ([0.5, 0.25], true),
    ([0.2, 0.4], false),
    ([0.3, 0.3], false),
    ([0.1, 0.6], false),
];

const Y_INPUT_VERTICES: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 1.0, 0.0],
    [2.0, -1.0, 0.5],
];
const Y_EDGES: [[u32; 2]; 3] = [[0, 1], [1, 2], [1, 3]];
const Y_SKIN: [([f64; 2], bool); 4] = [
    ([0.3, 0.3], true),
    ([0.3, 0.3], false),
    ([0.2, 0.2], false),
    ([0.2, 0.2], false),
];

const SYMMETRY_INPUT_VERTICES: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.4, 0.0, -0.46],
    [-0.2, 0.822_724_133_595_216_8, 0.23],
    [-0.2, -0.606_217_782_649_106_9, -0.23],
];
const SYMMETRY_EDGES: [[u32; 2]; 3] = [[0, 1], [0, 2], [0, 3]];
const SYMMETRY_SKIN: [([f64; 2], bool); 4] = [
    ([0.23, 0.31], true),
    ([0.16, 0.19], false),
    ([0.19, 0.23], false),
    ([0.22, 0.19], false),
];
const SYMMETRY_Z_INPUT_VERTICES: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [-0.46, 0.0, 0.4],
    [0.23, 0.822_724_133_595_216_8, -0.2],
    [-0.23, -0.606_217_782_649_106_9, -0.2],
];

const MIDDLE_ROOT_INPUT_VERTICES: [[f64; 3]; 3] =
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]];
const MIDDLE_ROOT_EDGES: [[u32; 2]; 2] = [[0, 1], [1, 2]];
const MIDDLE_ROOT_SKIN: [([f64; 2], bool); 3] = [
    ([0.25, 0.25], false),
    ([0.5, 0.25], true),
    ([0.25, 0.25], false),
];

/// Blender 5.2.2 output for the 3-vertex line, default radii, root at vertex 0.
const LINE_VERTICES: [[f64; 3]; 20] = [
    [1.25, -0.25, 0.0],
    [0.75, -0.25, 0.0],
    [0.75, 0.25, 0.0],
    [1.25, 0.25, 0.0],
    [1.25, -0.25, 0.5],
    [0.75, -0.25, 0.5],
    [0.75, 0.25, 0.5],
    [1.25, 0.25, 0.5],
    [1.25, -0.25, 1.0],
    [0.75, -0.25, 1.0],
    [0.75, 0.25, 1.0],
    [1.25, 0.25, 1.0],
    [1.25, -0.25, 0.25],
    [0.75, -0.25, 0.25],
    [0.75, 0.25, 0.25],
    [1.25, 0.25, 0.25],
    [1.25, -0.25, 0.75],
    [0.75, -0.25, 0.75],
    [0.75, 0.25, 0.75],
    [1.25, 0.25, 0.75],
];
const LINE_FACES: [[usize; 4]; 18] = [
    [0, 1, 2, 3],
    [0, 3, 15, 12],
    [0, 12, 13, 1],
    [1, 13, 14, 2],
    [2, 14, 15, 3],
    [4, 5, 13, 12],
    [4, 7, 19, 16],
    [4, 12, 15, 7],
    [4, 16, 17, 5],
    [5, 6, 14, 13],
    [5, 17, 18, 6],
    [6, 7, 15, 14],
    [6, 18, 19, 7],
    [8, 9, 17, 16],
    [8, 11, 10, 9],
    [8, 16, 19, 11],
    [9, 10, 18, 17],
    [10, 11, 19, 18],
];

/// Blender 5.2.2 output for the bent chain (`BENT_SKIN`). Its branch smoothing and symmetry settings
/// change nothing: Blender only uses them while building branch hulls.
const BENT_VERTICES: [[f64; 3]; 32] = [
    [0.0, 0.5, -0.25],
    [0.0, -0.5, -0.25],
    [0.0, -0.5, 0.25],
    [0.0, 0.5, 0.25],
    [1.0, 0.200_000_018, -0.400_000_006],
    [1.252_982_14, -0.106_274_188, -0.353_137_106],
    [0.999_999_94, -0.200_000_018, 0.400_000_006],
    [0.747_017_741, 0.106_274_188, 0.353_137_106],
    [0.907_180_786, 1.357_104_421, 0.290_569_425],
    [1.388_847_351, 1.001_550_913, 0.330_307_961],
    [1.092_819_214, 0.642_895_639, 0.709_430_575],
    [0.611_152_649, 0.998_449_206, 0.669_691_98],
    [2.313_684_464, 1.477_357_03, 1.790_877_104],
    [2.425_545_216, 1.329_283_118, 1.716_303_11],
    [1.686_315_417, 0.522_642_97, 2.209_122_896],
    [1.574_454_665, 0.670_716_882, 2.283_696_89],
    [0.535_886_765, 0.339_233_965, -0.330_383_003],
    [0.535_886_765, -0.339_233_965, -0.330_383_003],
    [0.535_886_765, -0.339_233_965, 0.330_383_003],
    [0.535_886_765, 0.339_233_965, 0.330_383_003],
    [0.932_917_953, 0.690_000_057, -0.130_000_114],
    [1.380_131_602, 0.590_000_033, 0.069_999_933],
    [1.067_082_047, 0.309_999_943, 0.630_000_114],
    [0.619_868_398, 0.409_999_967, 0.430_000_067],
    [1.411_425_829, 1.440_070_391, 0.884_754_479],
    [1.678_958_654, 1.085_928_202, 0.706_399_322],
    [1.196_917_295, 0.559_929_669, 1.027_760_267],
    [0.929_384_47, 0.914_071_798, 1.206_115_365],
    [1.852_741_241, 1.458_308_22, 1.327_959_657],
    [2.044_131_279, 1.204_958_677, 1.200_366_139],
    [1.436_293_125, 0.541_691_84, 1.605_591_655],
    [1.244_903_088, 0.795_041_382, 1.733_185_172],
];
const BENT_FACES: [[usize; 4]; 30] = [
    [0, 1, 2, 3],
    [0, 3, 19, 16],
    [0, 16, 17, 1],
    [1, 17, 18, 2],
    [2, 18, 19, 3],
    [4, 5, 17, 16],
    [4, 7, 23, 20],
    [4, 16, 19, 7],
    [4, 20, 21, 5],
    [5, 6, 18, 17],
    [5, 21, 22, 6],
    [6, 7, 19, 18],
    [6, 22, 23, 7],
    [8, 9, 21, 20],
    [8, 11, 27, 24],
    [8, 20, 23, 11],
    [8, 24, 25, 9],
    [9, 10, 22, 21],
    [9, 25, 26, 10],
    [10, 11, 23, 22],
    [10, 26, 27, 11],
    [12, 13, 29, 28],
    [12, 15, 14, 13],
    [12, 28, 31, 15],
    [13, 14, 30, 29],
    [14, 15, 31, 30],
    [24, 27, 31, 28],
    [24, 28, 29, 25],
    [25, 29, 30, 26],
    [26, 30, 31, 27],
];

fn skin_modifier(params: Value) -> Modifier {
    Modifier {
        id: Id::new("modifier_skin").unwrap(),
        modifier_type: "skin".to_owned(),
        name: "skin".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: serde_json::from_value(params).unwrap(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    }
}

/// Skeleton with `skin_radii` / `skin_roots` keyed by vertex index, as Blender's skin layer.
fn skeleton(verts: &[[f64; 3]], edges: &[[u32; 2]], skin: Option<&[([f64; 2], bool)]>) -> Mesh {
    let mut mesh = Mesh::new();
    for vertex in verts {
        mesh.insert_vertex(DVec3::from_array(*vertex)).unwrap();
    }
    for edge in edges {
        mesh.insert_edge(*edge).unwrap();
    }
    if let Some(skin) = skin {
        let radii: serde_json::Map<String, Value> = skin
            .iter()
            .enumerate()
            .map(|(index, (radius, _))| (index.to_string(), json!(radius)))
            .collect();
        let roots: serde_json::Map<String, Value> = skin
            .iter()
            .enumerate()
            .map(|(index, (_, root))| (index.to_string(), json!(root)))
            .collect();
        mesh.attributes
            .insert("skin_radii".to_owned(), Value::Object(radii));
        mesh.attributes
            .insert("skin_roots".to_owned(), Value::Object(roots));
    }
    mesh
}

/// Rotates a vertex cycle so it starts at its lowest index, keeping its winding.
fn cycle_from_lowest(face: &[usize]) -> Vec<usize> {
    let start = face
        .iter()
        .enumerate()
        .min_by_key(|(_, index)| **index)
        .map_or(0, |(position, _)| position);
    face.iter()
        .cycle()
        .skip(start)
        .take(face.len())
        .copied()
        .collect()
}

/// Maps every evaluated vertex ID to its index in the Blender oracle's vertex list. Each Blender
/// vertex must have exactly one evaluated vertex at the same position.
fn blender_index_by_id(result: &Mesh, vertices: &[[f64; 3]]) -> HashMap<u32, usize> {
    let mut matched = vec![false; result.vertices.len()];
    let mut index_by_id = HashMap::with_capacity(vertices.len());
    for (index, position) in vertices.iter().enumerate() {
        let position = DVec3::from_array(*position);
        let storage = result
            .vertices
            .iter()
            .enumerate()
            .find_map(|(storage, vertex)| {
                (!matched[storage] && vertex.co.distance(position) < 1.0e-6).then_some(storage)
            })
            .expect("Blender Skin vertex is present");
        matched[storage] = true;
        index_by_id.insert(result.vertices[storage].id, index);
    }
    index_by_id
}

/// Every expected Blender vertex must have an evaluated vertex at the same position, and the faces
/// must match as vertex cycles over those vertices (winding included).
fn assert_blender_geometry<F: AsRef<[usize]>>(result: &Mesh, vertices: &[[f64; 3]], faces: &[F]) {
    assert_eq!(result.vertices.len(), vertices.len());
    assert_eq!(result.faces.len(), faces.len());
    let index_by_id = blender_index_by_id(result, vertices);
    let mut actual: Vec<Vec<usize>> = result
        .faces
        .iter()
        .map(|face| {
            let cycle: Vec<usize> = face.vertices.iter().map(|id| index_by_id[id]).collect();
            cycle_from_lowest(&cycle)
        })
        .collect();
    actual.sort();
    let mut expected: Vec<Vec<usize>> = faces
        .iter()
        .map(|face| cycle_from_lowest(face.as_ref()))
        .collect();
    expected.sort();
    assert_eq!(actual, expected);
}

/// The evaluated edges, as sorted pairs of Blender oracle vertex indices, must equal the Blender
/// oracle's edge set exactly: same pairs, no extras, no duplicates.
fn assert_blender_edges(result: &Mesh, vertices: &[[f64; 3]], pairs: &[[usize; 2]]) {
    let index_by_id = blender_index_by_id(result, vertices);
    let mut actual: Vec<[usize; 2]> = result
        .edges
        .iter()
        .map(|edge| {
            let mut pair = edge.vertices.map(|id| index_by_id[&id]);
            pair.sort_unstable();
            pair
        })
        .collect();
    actual.sort_unstable();
    let mut expected = pairs.to_vec();
    for pair in &mut expected {
        pair.sort_unstable();
    }
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

fn assert_blender_vertices(result: &Mesh, vertices: &[[f64; 3]]) {
    assert_eq!(result.vertices.len(), vertices.len());
    for expected in vertices {
        let position = DVec3::from_array(*expected);
        assert!(
            result
                .vertices
                .iter()
                .any(|vertex| vertex.co.distance(position) < 1.0e-6)
        );
    }
}

fn smooth_face_count(result: &Mesh) -> usize {
    result
        .attributes
        .get("shade_smooth")
        .and_then(|attribute| attribute["values"].as_object())
        .map_or(0, |values| {
            values
                .values()
                .filter(|flag| flag.as_bool() == Some(true))
                .count()
        })
}

fn assert_skin_oracle(result: &Mesh, oracle: &str) {
    let expected: Value = serde_json::from_str(oracle).expect("parse Blender Skin oracle");
    assert_eq!(expected["blender_version"], "5.2.2");
    let vertices: Vec<[f64; 3]> =
        serde_json::from_value(expected["vertices"].clone()).expect("Skin oracle vertices");
    let faces: Vec<Vec<usize>> =
        serde_json::from_value(expected["faces"].clone()).expect("Skin oracle faces");
    let smooth: Vec<bool> =
        serde_json::from_value(expected["face_smooth"].clone()).expect("Skin oracle shading");
    assert_blender_geometry(result, &vertices, &faces);
    assert!(result.faces.iter().all(|face| face.material_index == 0));
    let edge_pairs: Vec<[usize; 2]> =
        serde_json::from_value(expected["edge_pairs"].clone()).expect("Skin oracle edge pairs");
    assert_eq!(
        edge_pairs.len(),
        expected["edges"].as_u64().expect("Skin oracle edge count") as usize
    );
    assert_blender_edges(result, &vertices, &edge_pairs);
    assert_eq!(
        smooth_face_count(result),
        smooth.iter().filter(|value| **value).count()
    );
}

#[test]
fn line_skin_matches_blender() {
    let mesh = skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&LINE_SKIN));
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))], SUBJECT_NAME).unwrap();
    assert_blender_geometry(&result, &LINE_VERTICES, &LINE_FACES);
    assert_eq!(smooth_face_count(&result), 0);
}

#[test]
fn hook_uses_current_indices_after_skin_drops_origindex() {
    let mesh = skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&LINE_SKIN));
    let skin = skin_modifier(json!({}));
    let hook = Modifier {
        id: Id::new("hook_after_skin").unwrap(),
        modifier_type: "hook".to_owned(),
        name: "Hook after Skin".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: serde_json::from_value(json!({
            "vertex_indices": [0],
            "matrix_inverse": glam::DMat4::IDENTITY.to_cols_array()
        }))
        .unwrap(),
        binding_data: None,
        runtime: ModifierRuntime {
            object_matrix: Some(glam::DMat4::from_translation(DVec3::X).to_cols_array()),
            ..ModifierRuntime::default()
        },
    };
    let skinned = evaluate_modifiers(&mesh, std::slice::from_ref(&skin), SUBJECT_NAME).unwrap();
    let hooked = evaluate_modifiers(&mesh, &[skin, hook], SUBJECT_NAME).unwrap();
    assert_eq!(skinned.vertices.len(), hooked.vertices.len());
    for (index, (before, after)) in skinned.vertices.iter().zip(&hooked.vertices).enumerate() {
        let expected_delta = if index == 0 { DVec3::X } else { DVec3::ZERO };
        assert!(
            (after.co - before.co).distance(expected_delta) < 1.0e-6,
            "Skin output vertex {index} did not use Blender's missing-ORIGINDEX fallback"
        );
    }
}

#[test]
fn line_without_skin_layer_uses_blender_default_root() {
    // Blender adds a skin layer (radius 0.25, root on vertex 0) when the mesh has none.
    let mesh = skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, None);
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))], SUBJECT_NAME).unwrap();
    assert_blender_geometry(&result, &LINE_VERTICES, &LINE_FACES);
}

#[test]
fn line_use_smooth_shade_marks_every_face_smooth() {
    let mesh = skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&LINE_SKIN));
    let result = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({"use_smooth_shade": true}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_blender_geometry(&result, &LINE_VERTICES, &LINE_FACES);
    assert_eq!(smooth_face_count(&result), 18);
}

#[test]
fn bent_chain_skin_matches_blender() {
    let mesh = skeleton(&BENT_INPUT_VERTICES, &BENT_EDGES, Some(&BENT_SKIN));
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))], SUBJECT_NAME).unwrap();
    assert_blender_geometry(&result, &BENT_VERTICES, &BENT_FACES);
    assert_eq!(smooth_face_count(&result), 0);
}

#[test]
fn branch_smoothing_and_symmetry_leave_chains_unchanged() {
    // Blender's branch_smoothing and symmetry axes act only on branch hulls; chains have none.
    let mesh = skeleton(&BENT_INPUT_VERTICES, &BENT_EDGES, Some(&BENT_SKIN));
    for params in [
        json!({
            "branch_smoothing": 1.0,
            "use_x_symmetry": false,
            "use_y_symmetry": false,
            "use_z_symmetry": false
        }),
        json!({
            "branch_smoothing": 0.5,
            "use_x_symmetry": true,
            "use_y_symmetry": true,
            "use_z_symmetry": true
        }),
    ] {
        let result = evaluate_modifiers(&mesh, &[skin_modifier(params)], SUBJECT_NAME).unwrap();
        assert_blender_geometry(&result, &BENT_VERTICES, &BENT_FACES);
    }
}

#[test]
fn bent_chain_use_smooth_shade_marks_every_face_smooth() {
    let mesh = skeleton(&BENT_INPUT_VERTICES, &BENT_EDGES, Some(&BENT_SKIN));
    let result = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({"use_smooth_shade": true}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_blender_geometry(&result, &BENT_VERTICES, &BENT_FACES);
    assert_eq!(smooth_face_count(&result), 30);
}

#[test]
fn branch_smoothing_outside_zero_to_one_is_invalid() {
    let mesh = skeleton(&BENT_INPUT_VERTICES, &BENT_EDGES, Some(&BENT_SKIN));
    let error = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({"branch_smoothing": 1.5}))],
        SUBJECT_NAME,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidArgument);
}

#[test]
fn y_branch_matches_blender_with_default_and_disabled_symmetry() {
    let mesh = skeleton(&Y_INPUT_VERTICES, &Y_EDGES, Some(&Y_SKIN));
    let baseline = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))], SUBJECT_NAME).unwrap();
    assert_blender_geometry(&baseline, &Y_BRANCH_VERTICES, &Y_BRANCH_FACES);
    let symmetry_off = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({
            "use_x_symmetry": false,
            "use_y_symmetry": false,
            "use_z_symmetry": false
        }))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_blender_geometry(&symmetry_off, &Y_BRANCH_VERTICES, &Y_BRANCH_FACES);
}

#[test]
fn branch_x_symmetry_changes_hull_triangle_merging() {
    let mesh = skeleton(
        &SYMMETRY_INPUT_VERTICES,
        &SYMMETRY_EDGES,
        Some(&SYMMETRY_SKIN),
    );
    let symmetric = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({"use_x_symmetry": true}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_blender_geometry(
        &symmetric,
        &SYMMETRY_BRANCH_VERTICES,
        &SYMMETRY_BRANCH_ON_FACES,
    );
    let asymmetric = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({"use_x_symmetry": false}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_blender_geometry(
        &asymmetric,
        &SYMMETRY_BRANCH_VERTICES,
        &SYMMETRY_BRANCH_OFF_FACES,
    );
    assert_ne!(symmetric.faces.len(), asymmetric.faces.len());
}

#[test]
fn branch_y_symmetry_changes_hull_triangle_merging() {
    let mesh = skeleton(
        &SYMMETRY_INPUT_VERTICES,
        &SYMMETRY_EDGES,
        Some(&SYMMETRY_SKIN),
    );
    let symmetric = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({
            "use_x_symmetry": false,
            "use_y_symmetry": true,
            "use_z_symmetry": false
        }))],
        SUBJECT_NAME,
    )
    .unwrap();
    let asymmetric = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({
            "use_x_symmetry": false,
            "use_y_symmetry": false,
            "use_z_symmetry": false
        }))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_blender_vertices(&symmetric, &SYMMETRY_BRANCH_VERTICES);
    assert_blender_vertices(&asymmetric, &SYMMETRY_BRANCH_VERTICES);
    assert_eq!(symmetric.faces.len(), 26);
    assert_eq!(asymmetric.faces.len(), 22);
}

#[test]
fn branch_z_symmetry_changes_hull_triangle_merging() {
    let mesh = skeleton(
        &SYMMETRY_Z_INPUT_VERTICES,
        &SYMMETRY_EDGES,
        Some(&SYMMETRY_SKIN),
    );
    let symmetric = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({
            "use_x_symmetry": false,
            "use_y_symmetry": false,
            "use_z_symmetry": true
        }))],
        SUBJECT_NAME,
    )
    .unwrap();
    let asymmetric = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({
            "use_x_symmetry": false,
            "use_y_symmetry": false,
            "use_z_symmetry": false
        }))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_blender_vertices(&symmetric, &SYMMETRY_Z_BRANCH_VERTICES);
    assert_blender_vertices(&asymmetric, &SYMMETRY_Z_BRANCH_VERTICES);
    assert_eq!(symmetric.faces.len(), 24);
    assert_eq!(asymmetric.faces.len(), 22);
}

#[test]
fn y_branch_smoothing_matches_blender_coordinates() {
    let mesh = skeleton(&Y_INPUT_VERTICES, &Y_EDGES, Some(&Y_SKIN));
    let result = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({"branch_smoothing": 1.0}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_blender_geometry(&result, &Y_BRANCH_SMOOTH_VERTICES, &Y_BRANCH_FACES);
}

#[test]
fn detached_branch_matches_blender_after_topology_repair() {
    let vertices = [
        [0.0, 0.0, 0.0],
        [
            0.094_487_131_228_255_7,
            1.394_741_214_214_490_8,
            -0.678_946_578_260_350_7,
        ],
        [
            0.582_235_576_162_845,
            0.133_558_627_622_563,
            -1.977_934_171_189_755_8,
        ],
        [
            0.459_038_833_155_134_6,
            1.448_496_303_638_360_5,
            0.069_810_121_192_919_23,
        ],
    ];
    let edges = [[0, 1], [0, 2], [0, 3]];
    let skin = [
        ([0.461_704_528_345_290_13, 0.303_625_296_618_987_2], true),
        ([0.541_249_514_302_200_6, 0.630_267_945_303_741_3], false),
        ([0.143_448_118_220_367_52, 0.123_531_765_881_565_64], false),
        ([0.379_951_191_640_614, 0.261_341_214_477_875_95], false),
    ];
    let result = evaluate_modifiers(
        &skeleton(&vertices, &edges, Some(&skin)),
        &[skin_modifier(json!({}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_skin_oracle(
        &result,
        include_str!("fixtures/pr18f_skin_detached_branch.json"),
    );
}

#[test]
fn detached_branch_without_good_vertices_matches_blender() {
    let vertices = [
        [0.0, 0.0, 0.0],
        [
            -0.615_750_305_639_683,
            -0.249_189_935_518_786_9,
            -0.656_205_807_326_401,
        ],
        [
            0.472_990_441_586_766_95,
            0.630_204_184_756_855_8,
            -0.752_066_426_844_683_8,
        ],
        [
            0.431_616_752_093_885_2,
            0.769_742_372_176_139_3,
            -1.843_958_393_914_058,
        ],
        [
            -0.265_830_154_123_394_6,
            -0.615_534_116_543_724,
            0.237_502_074_899_434_4,
        ],
        [
            0.388_771_182_389_775_1,
            -2.367_231_987_934_589,
            1.256_233_270_860_481_3,
        ],
    ];
    let edges = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]];
    let skin = [
        ([0.594_402_594_058_547_7, 0.393_392_627_532_891_4], true),
        ([0.749_858_756_115_899_2, 0.826_390_060_367_004_9], false),
        ([0.750_462_446_341_210_4, 0.599_503_104_107_198_5], false),
        ([0.361_745_709_536_949_96, 0.327_011_298_497_759_8], false),
        ([0.866_649_107_589_648_1, 0.387_449_475_726_335_9], false),
        ([0.301_049_344_307_154_8, 0.646_648_963_948_643], false),
    ];
    let result = evaluate_modifiers(
        &skeleton(&vertices, &edges, Some(&skin)),
        &[skin_modifier(json!({}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_skin_oracle(
        &result,
        include_str!("fixtures/pr18f_skin_nogood_branch.json"),
    );
}

#[test]
fn detached_skin_repair_preserves_point_colors_and_vertex_groups() {
    let vertices = [
        [0.0, 0.0, 0.0],
        [
            0.094_487_131_228_255_7,
            1.394_741_214_214_490_8,
            -0.678_946_578_260_350_7,
        ],
        [
            0.582_235_576_162_845,
            0.133_558_627_622_563,
            -1.977_934_171_189_755_8,
        ],
        [
            0.459_038_833_155_134_6,
            1.448_496_303_638_360_5,
            0.069_810_121_192_919_23,
        ],
    ];
    let edges = [[0, 1], [0, 2], [0, 3]];
    let skin = [
        ([0.461_704_528_345_290_13, 0.303_625_296_618_987_2], true),
        ([0.541_249_514_302_200_6, 0.630_267_945_303_741_3], false),
        ([0.143_448_118_220_367_52, 0.123_531_765_881_565_64], false),
        ([0.379_951_191_640_614, 0.261_341_214_477_875_95], false),
    ];
    let mut mesh = skeleton(&vertices, &edges, Some(&skin));
    let mut weights = serde_json::Map::new();
    for vertex in &mesh.vertices {
        weights.insert(format!("v{}", vertex.id), json!(1.0));
    }
    mesh.attributes
        .insert("vertex_groups".to_owned(), json!({"Weight": weights}));
    let source_vertex_count = mesh.vertices.len();
    mesh.attributes.insert(
        "blender_attributes".to_owned(),
        json!([
            {
                "name": "SkinPoint",
                "domain": "POINT",
                "data_type": "FLOAT",
                "values": vec![json!(1.75); source_vertex_count]
            },
            {
                "name": "SkinColor",
                "domain": "POINT",
                "data_type": "FLOAT_COLOR",
                "values": vec![json!([0.25, 0.5, 0.75, 1.0]); source_vertex_count]
            }
        ]),
    );

    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))], SUBJECT_NAME).unwrap();
    let weights = result.attributes["vertex_groups"]["Weight"]
        .as_object()
        .unwrap();
    for vertex in &result.vertices {
        assert_eq!(
            weights
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64),
            Some(1.0)
        );
    }
    let attributes = result.attributes["blender_attributes"].as_array().unwrap();
    let point_values = attributes
        .iter()
        .find(|attribute| attribute["name"] == "SkinPoint")
        .unwrap()["values"]
        .as_array()
        .unwrap();
    assert_eq!(point_values.len(), result.vertices.len());
    assert!(
        point_values
            .iter()
            .all(|value| value.as_f64() == Some(1.75))
    );
    let color_values = attributes
        .iter()
        .find(|attribute| attribute["name"] == "SkinColor")
        .unwrap()["values"]
        .as_array()
        .unwrap();
    assert_eq!(color_values.len(), result.vertices.len());
    assert!(
        color_values
            .iter()
            .all(|value| value == &json!([0.25, 0.5, 0.75, 1.0]))
    );
}

#[test]
fn loose_skin_branch_skips_edge_subdivision() {
    let regular = evaluate_modifiers(
        &skeleton(&Y_INPUT_VERTICES, &Y_EDGES, Some(&Y_SKIN)),
        &[skin_modifier(json!({}))],
        SUBJECT_NAME,
    )
    .unwrap();
    let mut loose_mesh = skeleton(&Y_INPUT_VERTICES, &Y_EDGES, Some(&Y_SKIN));
    loose_mesh.attributes.insert(
        "blender_skin_vertices".to_owned(),
        Value::Array(
            Y_SKIN
                .iter()
                .enumerate()
                .map(|(index, (radius, root))| {
                    json!({"radius": radius, "root": root, "loose": index == 1})
                })
                .collect(),
        ),
    );
    let loose = evaluate_modifiers(&loose_mesh, &[skin_modifier(json!({}))], SUBJECT_NAME).unwrap();
    assert!(
        loose.vertices.len() < regular.vertices.len(),
        "loose branch should omit the intermediate frames"
    );
    assert!(!loose.attributes.contains_key("blender_skin_vertices"));
}

#[test]
fn middle_root_seam_frames_match_blender() {
    let mesh = skeleton(
        &MIDDLE_ROOT_INPUT_VERTICES,
        &MIDDLE_ROOT_EDGES,
        Some(&MIDDLE_ROOT_SKIN),
    );
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))], SUBJECT_NAME).unwrap();
    assert_blender_geometry(&result, &MIDDLE_ROOT_VERTICES, &MIDDLE_ROOT_FACES);
}

#[test]
fn isolated_skin_vertex_builds_blender_box() {
    let verts = [[0.0, 0.0, 0.0]];
    let result = evaluate_modifiers(
        &skeleton(&verts, &[], None),
        &[skin_modifier(json!({}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_blender_geometry(&result, &ISOLATED_VERTICES, &ISOLATED_FACES);
}

#[test]
fn zero_skin_radii_are_accepted() {
    let skin = [
        ([0.25, 0.25], true),
        ([0.0, 0.0], false),
        ([0.25, 0.25], false),
    ];
    let result = evaluate_modifiers(
        &skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&skin)),
        &[skin_modifier(json!({}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_eq!(result.vertices.len(), 28);
    assert_eq!(result.faces.len(), 26);
}

#[test]
fn negative_skin_radii_with_negative_subdivision_count_are_rejected() {
    let skin = [
        ([-0.25, -0.25], true),
        ([-0.25, -0.25], false),
        ([-0.25, -0.25], false),
    ];
    let error = evaluate_modifiers(
        &skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&skin)),
        &[skin_modifier(json!({}))],
        SUBJECT_NAME,
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidArgument);
}

#[test]
fn mixed_sign_skin_radii_match_blender_bounds() {
    let skin = [
        ([0.25, -0.25], true),
        ([0.25, 0.25], false),
        ([0.25, 0.25], false),
    ];
    let result = evaluate_modifiers(
        &skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&skin)),
        &[skin_modifier(json!({}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_eq!(result.vertices.len(), 24);
    assert_eq!(result.faces.len(), 22);
    let bounds = result.bounds().unwrap();
    assert!((bounds.min.x - 0.75).abs() < 1.0e-6);
    assert!((bounds.max.x - 1.25).abs() < 1.0e-6);
    assert!((bounds.min.y + 0.25).abs() < 1.0e-6);
    assert!((bounds.max.y - 0.25).abs() < 1.0e-6);
    assert!((bounds.min.z - 0.0).abs() < 1.0e-6);
    assert!((bounds.max.z - 1.0).abs() < 1.0e-6);
}

#[test]
fn zero_average_edge_radius_is_accepted_without_subdivision() {
    let verts = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.5]];
    let edges = [[0, 1]];
    let skin = [([-0.25, -0.25], true), ([0.25, 0.25], false)];
    let result = evaluate_modifiers(
        &skeleton(&verts, &edges, Some(&skin)),
        &[skin_modifier(json!({}))],
        SUBJECT_NAME,
    )
    .unwrap();
    assert_eq!(result.vertices.len(), 8);
    assert_eq!(result.faces.len(), 6);
}
#[test]
fn skeleton_without_root_reports_blender_modifier_warning_and_keeps_input() {
    let mesh = skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&LINE_NO_ROOT_SKIN));
    let result = evaluate_modifiers_with_groups(
        &mesh,
        &[skin_modifier(json!({}))],
        &BTreeMap::new(),
        SUBJECT_NAME,
    )
    .unwrap();
    assert_eq!(result.mesh, mesh);
    assert_eq!(result.warnings.len(), 1);
    assert_eq!(result.warnings[0].details["modifier_warning"], true);
    assert!(result.warnings[0].message.contains("No valid root vertex"));
}

const MIDDLE_ROOT_VERTICES: [[f64; 3]; 24] = [
    [0.25, -0.25, 0.0],
    [-0.25, -0.25, 0.0],
    [-0.25, 0.25, 0.0],
    [0.25, 0.25, 0.0],
    [0.5, -0.25, 0.625],
    [-0.5, -0.25, 0.625],
    [-0.5, 0.25, 0.625],
    [0.5, 0.25, 0.625],
    [0.5, -0.25, 1.375],
    [-0.5, -0.25, 1.375],
    [-0.5, 0.25, 1.375],
    [0.5, 0.25, 1.375],
    [0.25, -0.25, 2.0],
    [-0.25, -0.25, 2.0],
    [-0.25, 0.25, 2.0],
    [0.25, 0.25, 2.0],
    [0.355_112_076, -0.25, 0.420_448_214],
    [-0.355_112_076, -0.25, 0.420_448_214],
    [-0.355_112_076, 0.25, 0.420_448_214],
    [0.355_112_076, 0.25, 0.420_448_214],
    [0.359_692_246, -0.25, 1.561_231_017],
    [-0.359_692_246, -0.25, 1.561_231_017],
    [-0.359_692_246, 0.25, 1.561_231_017],
    [0.359_692_246, 0.25, 1.561_231_017],
];
const MIDDLE_ROOT_FACES: [[usize; 4]; 22] = [
    [0, 1, 2, 3],
    [0, 3, 19, 16],
    [0, 16, 17, 1],
    [1, 17, 18, 2],
    [2, 18, 19, 3],
    [4, 5, 17, 16],
    [4, 7, 11, 8],
    [4, 8, 9, 5],
    [4, 16, 19, 7],
    [5, 6, 18, 17],
    [5, 9, 10, 6],
    [6, 7, 19, 18],
    [6, 10, 11, 7],
    [8, 11, 23, 20],
    [8, 20, 21, 9],
    [9, 21, 22, 10],
    [10, 22, 23, 11],
    [12, 13, 21, 20],
    [12, 15, 14, 13],
    [12, 20, 23, 15],
    [13, 14, 22, 21],
    [14, 15, 23, 22],
];

const CYCLE_INPUT_VERTICES: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
];
const CYCLE_EDGES: [[u32; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];
/// Blender 5.2.2 output for the quad cycle without a skin layer (default root on vertex 0).
const CYCLE_VERTICES: [[f64; 3]; 48] = [
    [-0.176_776_648, -0.176_776_752, -0.25],
    [0.176_776_648, 0.176_776_752, -0.25],
    [0.176_776_648, 0.176_776_752, 0.25],
    [-0.176_776_648, -0.176_776_752, 0.25],
    [1.176_776_767, -0.176_776_662, -0.25],
    [0.823_223_233, 0.176_776_662, -0.25],
    [0.823_223_233, 0.176_776_662, 0.25],
    [1.176_776_767, -0.176_776_662, 0.25],
    [1.176_776_648, 1.176_776_767, -0.25],
    [0.823_223_352, 0.823_223_293, -0.25],
    [0.823_223_352, 0.823_223_293, 0.25],
    [1.176_776_648, 1.176_776_767, 0.25],
    [-0.176_776_707, 1.176_776_648, -0.25],
    [0.176_776_707, 0.823_223_293, -0.25],
    [0.176_776_707, 0.823_223_293, 0.25],
    [-0.176_776_707, 1.176_776_648, 0.25],
    [0.333_333_403, -0.25, -0.25],
    [0.333_333_284, 0.25, -0.25],
    [0.333_333_284, 0.25, 0.25],
    [0.333_333_403, -0.25, 0.25],
    [0.666_666_746, -0.25, -0.25],
    [0.666_666_627, 0.25, -0.25],
    [0.666_666_627, 0.25, 0.25],
    [0.666_666_746, -0.25, 0.25],
    [1.25, 0.333_333_373, -0.25],
    [0.75, 0.333_333_313, -0.25],
    [0.75, 0.333_333_313, 0.25],
    [1.25, 0.333_333_373, 0.25],
    [1.25, 0.666_666_746, -0.25],
    [0.75, 0.666_666_627, -0.25],
    [0.75, 0.666_666_627, 0.25],
    [1.25, 0.666_666_746, 0.25],
    [0.666_666_627, 1.25, -0.25],
    [0.666_666_627, 0.75, -0.25],
    [0.666_666_627, 0.75, 0.25],
    [0.666_666_627, 1.25, 0.25],
    [0.333_333_284, 1.25, -0.25],
    [0.333_333_343, 0.75, -0.25],
    [0.333_333_343, 0.75, 0.25],
    [0.333_333_284, 1.25, 0.25],
    [-0.25, 0.666_666_627, -0.25],
    [0.25, 0.666_666_627, -0.25],
    [0.25, 0.666_666_627, 0.25],
    [-0.25, 0.666_666_627, 0.25],
    [-0.25, 0.333_333_313, -0.25],
    [0.25, 0.333_333_313, -0.25],
    [0.25, 0.333_333_313, 0.25],
    [-0.25, 0.333_333_313, 0.25],
];
const CYCLE_FACES: [[usize; 4]; 48] = [
    [0, 1, 17, 16],
    [0, 3, 47, 44],
    [0, 16, 19, 3],
    [0, 44, 45, 1],
    [1, 2, 18, 17],
    [1, 45, 46, 2],
    [2, 3, 19, 18],
    [2, 46, 47, 3],
    [4, 5, 25, 24],
    [4, 7, 23, 20],
    [4, 20, 21, 5],
    [4, 24, 27, 7],
    [5, 6, 26, 25],
    [5, 21, 22, 6],
    [6, 7, 27, 26],
    [6, 22, 23, 7],
    [8, 9, 33, 32],
    [8, 11, 31, 28],
    [8, 28, 29, 9],
    [8, 32, 35, 11],
    [9, 10, 34, 33],
    [9, 29, 30, 10],
    [10, 11, 35, 34],
    [10, 30, 31, 11],
    [12, 13, 41, 40],
    [12, 15, 39, 36],
    [12, 36, 37, 13],
    [12, 40, 43, 15],
    [13, 14, 42, 41],
    [13, 37, 38, 14],
    [14, 15, 43, 42],
    [14, 38, 39, 15],
    [16, 17, 21, 20],
    [16, 20, 23, 19],
    [17, 18, 22, 21],
    [18, 19, 23, 22],
    [24, 25, 29, 28],
    [24, 28, 31, 27],
    [25, 26, 30, 29],
    [26, 27, 31, 30],
    [32, 33, 37, 36],
    [32, 36, 39, 35],
    [33, 34, 38, 37],
    [34, 35, 39, 38],
    [40, 41, 45, 44],
    [40, 44, 47, 43],
    [41, 42, 46, 45],
    [42, 43, 47, 46],
];

const TWO_ROOTS_LINE_INPUT_VERTICES: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 2.0],
    [0.0, 0.0, 3.0],
];
const TWO_ROOTS_LINE_EDGES: [[u32; 2]; 3] = [[0, 1], [1, 2], [2, 3]];
const TWO_ROOTS_LINE_SKIN: [([f64; 2], bool); 4] = [
    ([0.25, 0.25], true),
    ([0.25, 0.25], false),
    ([0.25, 0.25], false),
    ([0.25, 0.25], true),
];
/// Blender 5.2.2 output for a line rooted at both ends.
const TWO_ROOTS_LINE_VERTICES: [[f64; 3]; 40] = [
    [0.25, -0.25, 0.0],
    [-0.25, -0.25, 0.0],
    [-0.25, 0.25, 0.0],
    [0.25, 0.25, 0.0],
    [0.25, -0.25, 1.0],
    [-0.25, -0.25, 1.0],
    [-0.25, 0.25, 1.0],
    [0.25, 0.25, 1.0],
    [0.25, -0.25, 2.0],
    [-0.25, -0.25, 2.0],
    [-0.25, 0.25, 2.0],
    [0.25, 0.25, 2.0],
    [0.25, -0.25, 3.0],
    [-0.25, -0.25, 3.0],
    [-0.25, 0.25, 3.0],
    [0.25, 0.25, 3.0],
    [0.25, -0.25, 0.333_333_343],
    [-0.25, -0.25, 0.333_333_343],
    [-0.25, 0.25, 0.333_333_343],
    [0.25, 0.25, 0.333_333_343],
    [0.25, -0.25, 0.666_666_687],
    [-0.25, -0.25, 0.666_666_687],
    [-0.25, 0.25, 0.666_666_687],
    [0.25, 0.25, 0.666_666_687],
    [0.25, -0.25, 1.333_333_254],
    [-0.25, -0.25, 1.333_333_254],
    [-0.25, 0.25, 1.333_333_254],
    [0.25, 0.25, 1.333_333_254],
    [0.25, -0.25, 1.666_666_746],
    [-0.25, -0.25, 1.666_666_746],
    [-0.25, 0.25, 1.666_666_746],
    [0.25, 0.25, 1.666_666_746],
    [0.25, -0.25, 2.333_333_254],
    [-0.25, -0.25, 2.333_333_254],
    [-0.25, 0.25, 2.333_333_254],
    [0.25, 0.25, 2.333_333_254],
    [0.25, -0.25, 2.666_666_508],
    [-0.25, -0.25, 2.666_666_508],
    [-0.25, 0.25, 2.666_666_508],
    [0.25, 0.25, 2.666_666_508],
];
const TWO_ROOTS_LINE_FACES: [[usize; 4]; 38] = [
    [0, 1, 2, 3],
    [0, 3, 19, 16],
    [0, 16, 17, 1],
    [1, 17, 18, 2],
    [2, 18, 19, 3],
    [4, 5, 21, 20],
    [4, 7, 27, 24],
    [4, 20, 23, 7],
    [4, 24, 25, 5],
    [5, 6, 22, 21],
    [5, 25, 26, 6],
    [6, 7, 23, 22],
    [6, 26, 27, 7],
    [8, 9, 29, 28],
    [8, 11, 35, 32],
    [8, 28, 31, 11],
    [8, 32, 33, 9],
    [9, 10, 30, 29],
    [9, 33, 34, 10],
    [10, 11, 31, 30],
    [10, 34, 35, 11],
    [12, 13, 37, 36],
    [12, 15, 14, 13],
    [12, 36, 39, 15],
    [13, 14, 38, 37],
    [14, 15, 39, 38],
    [16, 19, 23, 20],
    [16, 20, 21, 17],
    [17, 21, 22, 18],
    [18, 22, 23, 19],
    [24, 27, 31, 28],
    [24, 28, 29, 25],
    [25, 29, 30, 26],
    [26, 30, 31, 27],
    [32, 35, 39, 36],
    [32, 36, 37, 33],
    [33, 37, 38, 34],
    [34, 38, 39, 35],
];

const TWO_ISLANDS_INPUT_VERTICES: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [3.0, 0.0, 0.0],
    [3.0, 0.0, 1.0],
];
const TWO_ISLANDS_EDGES: [[u32; 2]; 2] = [[0, 1], [2, 3]];
const TWO_ISLANDS_SKIN: [([f64; 2], bool); 4] = [
    ([0.25, 0.25], true),
    ([0.25, 0.25], false),
    ([0.4, 0.2], true),
    ([0.25, 0.25], false),
];
/// Blender 5.2.2 output for two separate lines, each rooted at its first vertex.
const TWO_ISLANDS_VERTICES: [[f64; 3]; 28] = [
    [0.25, -0.25, 0.0],
    [-0.25, -0.25, 0.0],
    [-0.25, 0.25, 0.0],
    [0.25, 0.25, 0.0],
    [0.25, -0.25, 1.0],
    [-0.25, -0.25, 1.0],
    [-0.25, 0.25, 1.0],
    [0.25, 0.25, 1.0],
    [3.400_000_095, -0.200_000_003, 0.0],
    [2.599_999_905, -0.200_000_003, 0.0],
    [2.599_999_905, 0.200_000_003, 0.0],
    [3.400_000_095, 0.200_000_003, 0.0],
    [3.25, -0.25, 1.0],
    [2.75, -0.25, 1.0],
    [2.75, 0.25, 1.0],
    [3.25, 0.25, 1.0],
    [0.25, -0.25, 0.333_333_343],
    [-0.25, -0.25, 0.333_333_343],
    [-0.25, 0.25, 0.333_333_343],
    [0.25, 0.25, 0.333_333_343],
    [0.25, -0.25, 0.666_666_687],
    [-0.25, -0.25, 0.666_666_687],
    [-0.25, 0.25, 0.666_666_687],
    [0.25, 0.25, 0.666_666_687],
    [3.320_540_19, -0.226_486_579, 0.529_731_572],
    [2.679_459_81, -0.226_486_579, 0.529_731_572],
    [2.679_459_81, 0.226_486_579, 0.529_731_572],
    [3.320_540_19, 0.226_486_579, 0.529_731_572],
];
const TWO_ISLANDS_FACES: [[usize; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 3, 19, 16],
    [0, 16, 17, 1],
    [1, 17, 18, 2],
    [2, 18, 19, 3],
    [4, 5, 21, 20],
    [4, 7, 6, 5],
    [4, 20, 23, 7],
    [5, 6, 22, 21],
    [6, 7, 23, 22],
    [8, 9, 10, 11],
    [8, 11, 27, 24],
    [8, 24, 25, 9],
    [9, 25, 26, 10],
    [10, 26, 27, 11],
    [12, 13, 25, 24],
    [12, 15, 14, 13],
    [12, 24, 27, 15],
    [13, 14, 26, 25],
    [14, 15, 27, 26],
    [16, 19, 23, 20],
    [16, 20, 21, 17],
    [17, 21, 22, 18],
    [18, 22, 23, 19],
];

#[test]
fn quad_cycle_without_skin_layer_matches_blender() {
    // Blender's default layer roots vertex 0, which the cycle reaches from both sides.
    let mesh = skeleton(&CYCLE_INPUT_VERTICES, &CYCLE_EDGES, None);
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))], SUBJECT_NAME).unwrap();
    assert_blender_geometry(&result, &CYCLE_VERTICES, &CYCLE_FACES);
}

#[test]
fn two_roots_on_one_skeleton_match_blender() {
    let mesh = skeleton(
        &TWO_ROOTS_LINE_INPUT_VERTICES,
        &TWO_ROOTS_LINE_EDGES,
        Some(&TWO_ROOTS_LINE_SKIN),
    );
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))], SUBJECT_NAME).unwrap();
    assert_blender_geometry(&result, &TWO_ROOTS_LINE_VERTICES, &TWO_ROOTS_LINE_FACES);
}

#[test]
fn two_islands_each_with_a_root_match_blender() {
    let mesh = skeleton(
        &TWO_ISLANDS_INPUT_VERTICES,
        &TWO_ISLANDS_EDGES,
        Some(&TWO_ISLANDS_SKIN),
    );
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))], SUBJECT_NAME).unwrap();
    assert_blender_geometry(&result, &TWO_ISLANDS_VERTICES, &TWO_ISLANDS_FACES);
}

const ISOLATED_VERTICES: [[f64; 3]; 8] = [
    [0.25, -0.25, 0.25],
    [-0.25, -0.25, 0.25],
    [-0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25],
    [0.25, -0.25, -0.25],
    [-0.25, -0.25, -0.25],
    [-0.25, 0.25, -0.25],
    [0.25, 0.25, -0.25],
];
const ISOLATED_FACES: [[usize; 4]; 6] = [
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [3, 2, 1, 0],
    [4, 5, 6, 7],
];

const Y_BRANCH_VERTICES: [[f64; 3]; 36] = [
    [0.0, 0.300_000_012, -0.300_000_012],
    [0.0, -0.300_000_012, -0.300_000_012],
    [0.0, -0.300_000_012, 0.300_000_012],
    [0.0, 0.300_000_012, 0.300_000_012],
    [1.858_578_682, 1.141_421_318, -0.200_000_003],
    [2.141_421_318, 0.858_578_622, -0.200_000_003],
    [2.141_421_318, 0.858_578_622, 0.200_000_003],
    [1.858_578_682, 1.141_421_318, 0.200_000_003],
    [2.188_561_678, -0.905_719_161, 0.311_438_203],
    [1.905_719_161, -1.188_561_797, 0.311_438_203],
    [1.811_438_203, -1.094_280_839, 0.688_561_797],
    [2.094_280_958, -0.811_438_203, 0.688_561_797],
    [0.5, 0.300_000_012, -0.300_000_012],
    [0.5, -0.300_000_012, -0.300_000_012],
    [0.5, -0.300_000_012, 0.300_000_012],
    [0.5, 0.300_000_012, 0.300_000_012],
    [1.216_486_692, 0.584_138_036, -0.259_968_787],
    [1.584_138_155, 0.216_486_648, -0.259_968_787],
    [1.584_138_155, 0.216_486_648, 0.259_968_787],
    [1.216_486_692, 0.584_138_036, 0.259_968_787],
    [1.551_579_595, 0.874_971_33, -0.228_672_46],
    [1.874_971_271, 0.551_579_654, -0.228_672_46],
    [1.874_971_271, 0.551_579_654, 0.228_672_46],
    [1.551_579_595, 0.874_971_33, 0.228_672_46],
    [1.568_126_321, -0.188_407_242, -0.095_655_948],
    [1.188_407_302, -0.568_126_38, -0.095_655_948],
    [1.061_834_216, -0.441_553_324, 0.410_636_216],
    [1.441_553_235, -0.061_834_194, 0.410_636_216],
    [1.791_160_345, -0.446_266_323, 0.050_686_151],
    [1.446_266_413, -0.791_160_405, 0.050_686_151],
    [1.331_301_689, -0.676_195_681, 0.510_544_896],
    [1.676_195_621, -0.331_301_659, 0.510_544_896],
    [1.995_495_558, -0.682_506_8, 0.184_759_08],
    [1.682_506_8, -0.995_495_439, 0.184_759_08],
    [1.578_177_214, -0.891_165_853, 0.602_077_246],
    [1.891_165_972, -0.578_177_214, 0.602_077_246],
];
const Y_BRANCH_SMOOTH_VERTICES: [[f64; 3]; 36] = [
    [0.0, 0.300_000_012, -0.300_000_012],
    [0.0, -0.300_000_012, -0.300_000_012],
    [0.0, -0.300_000_012, 0.300_000_012],
    [0.0, 0.300_000_012, 0.300_000_012],
    [1.858_578_682, 1.141_421_318, -0.200_000_003],
    [2.141_421_318, 0.858_578_622, -0.200_000_003],
    [2.141_421_318, 0.858_578_622, 0.200_000_003],
    [1.858_578_682, 1.141_421_318, 0.200_000_003],
    [2.188_561_678, -0.905_719_161, 0.311_438_203],
    [1.905_719_161, -1.188_561_797, 0.311_438_203],
    [1.811_438_203, -1.094_280_839, 0.688_561_797],
    [2.094_280_958, -0.811_438_203, 0.688_561_797],
    [0.680_062_532, 0.260_062_486, -0.231_993_765],
    [0.586_050_928, -0.258_515_805, -0.199_457_005],
    [0.570_229_292, -0.242_694_169, 0.238_829_538],
    [0.627_832_115, 0.194_258_481, 0.247_060_508],
    [1.172_882_318, 0.502_664_328, -0.199_980_497],
    [1.428_843_737, 0.154_958_129, -0.192_591_116],
    [1.520_358_562, 0.227_652_967, 0.213_810_027],
    [1.204_109_907, 0.454_361_856, 0.227_044_895],
    [1.551_579_595, 0.874_971_33, -0.228_672_46],
    [1.874_971_271, 0.551_579_654, -0.228_672_46],
    [1.874_971_271, 0.551_579_654, 0.228_672_46],
    [1.551_579_595, 0.874_971_33, 0.228_672_46],
    [1.506_841_302, -0.169_438_779, -0.052_908_532],
    [1.184_454_203, -0.412_223_22, -0.081_892_416],
    [1.000_096_56, -0.327_928_066, 0.337_879_747],
    [1.407_990_456, -0.020_034_105, 0.329_873_502],
    [1.791_160_345, -0.446_266_323, 0.050_686_151],
    [1.446_266_413, -0.791_160_405, 0.050_686_151],
    [1.331_301_689, -0.676_195_681, 0.510_544_896],
    [1.676_195_621, -0.331_301_659, 0.510_544_896],
    [1.995_495_558, -0.682_506_8, 0.184_759_08],
    [1.682_506_8, -0.995_495_439, 0.184_759_08],
    [1.578_177_214, -0.891_165_853, 0.602_077_246],
    [1.891_165_972, -0.578_177_214, 0.602_077_246],
];
const Y_BRANCH_FACES: [&[usize]; 36] = [
    &[22, 23, 19, 18],
    &[23, 20, 16, 19],
    &[18, 27, 24, 17],
    &[21, 22, 18, 17],
    &[20, 21, 17, 16],
    &[17, 24, 25],
    &[26, 14, 13, 25],
    &[15, 19, 16, 12],
    &[15, 12, 0, 3],
    &[26, 15, 14],
    &[14, 15, 3, 2],
    &[13, 14, 2, 1],
    &[16, 17, 12],
    &[12, 13, 1, 0],
    &[15, 26, 27, 19],
    &[17, 25, 13, 12],
    &[11, 10, 9, 8],
    &[18, 19, 27],
    &[7, 6, 5, 4],
    &[0, 1, 2, 3],
    &[4, 5, 21, 20],
    &[5, 6, 22, 21],
    &[6, 7, 23, 22],
    &[7, 4, 20, 23],
    &[28, 29, 25, 24],
    &[29, 30, 26, 25],
    &[30, 31, 27, 26],
    &[31, 28, 24, 27],
    &[32, 33, 29, 28],
    &[33, 34, 30, 29],
    &[34, 35, 31, 30],
    &[35, 32, 28, 31],
    &[8, 9, 33, 32],
    &[9, 10, 34, 33],
    &[10, 11, 35, 34],
    &[11, 8, 32, 35],
];
const SYMMETRY_BRANCH_ON_FACES: [&[usize]; 24] = [
    &[19, 22, 18],
    &[2, 3, 15, 14],
    &[1, 2, 14, 13],
    &[0, 1, 13, 12],
    &[5, 6, 18, 17],
    &[4, 5, 17, 16],
    &[11, 10, 9, 8],
    &[21, 22, 19],
    &[3, 0, 12, 15],
    &[12, 17, 18, 15],
    &[13, 14, 22],
    &[14, 15, 18, 22],
    &[20, 21, 19, 16],
    &[7, 6, 5, 4],
    &[12, 16, 17],
    &[12, 20, 16],
    &[3, 2, 1, 0],
    &[12, 13, 20],
    &[6, 7, 19, 18],
    &[7, 4, 16, 19],
    &[8, 9, 20, 13],
    &[9, 10, 21, 20],
    &[10, 11, 22, 21],
    &[11, 8, 13, 22],
];
const SYMMETRY_BRANCH_OFF_FACES: [&[usize]; 22] = [
    &[6, 7, 19, 18],
    &[5, 6, 18, 17],
    &[4, 5, 17, 16],
    &[3, 0, 12, 15],
    &[7, 4, 16, 19],
    &[12, 17, 18, 15],
    &[2, 3, 15, 14],
    &[1, 2, 14, 13],
    &[12, 20, 16, 17],
    &[14, 15, 18, 22],
    &[13, 14, 22],
    &[21, 22, 18, 19],
    &[20, 21, 19, 16],
    &[0, 1, 13, 12],
    &[11, 10, 9, 8],
    &[7, 6, 5, 4],
    &[3, 2, 1, 0],
    &[12, 13, 20],
    &[8, 9, 20, 13],
    &[9, 10, 21, 20],
    &[10, 11, 22, 21],
    &[11, 8, 13, 22],
];

const SYMMETRY_BRANCH_VERTICES: [[f64; 3]; 23] = [
    [0.256_624_937, 0.159_999_996, -0.584_674_001],
    [0.256_624_937, -0.159_999_996, -0.584_674_001],
    [0.543_375_075, -0.159_999_996, -0.335_326_016],
    [0.543_375_075, 0.159_999_996, -0.335_326_016],
    [-0.398_865_491, 0.836_430_848, 0.008_043_587],
    [-0.029_619_23, 0.926_192_701, 0.008_043_587],
    [-0.001_134_502, 0.809_017_36, 0.451_956_421],
    [-0.370_380_789, 0.719_255_507, 0.451_956_421],
    [0.029_101_707, -0.613_983_154, -0.408_751_547],
    [-0.388_745_546, -0.476_129_293, -0.408_751_547],
    [-0.429_101_706, -0.598_452_449, -0.051_248_476],
    [-0.011_254_467, -0.736_306_31, -0.051_248_476],
    [0.043_158_308, 0.190_460_876, -0.418_767_273],
    [0.086_855_657, -0.255_753_219, -0.387_030_751],
    [0.408_717_394, -0.190_460_876, -0.100_889_802],
    [0.408_717_394, 0.190_460_876, -0.100_889_802],
    [-0.327_028_245, 0.463_029_683, -0.133_260_25],
    [0.077_974_297, 0.561_483_741, -0.133_260_25],
    [0.111_016_281, 0.425_561_547, 0.381_673_992],
    [-0.293_986_231, 0.327_107_489, 0.381_673_992],
    [-0.295_964_45, -0.180_331_26, -0.355_294_228],
    [-0.347_955_793, -0.337_921_59, 0.105_281_04],
    [0.078_561_679, -0.478_635_877, 0.105_281_04],
];

const SYMMETRY_Z_BRANCH_VERTICES: [[f64; 3]; 23] = [
    [-0.584_674_001, -0.159_999_996, 0.256_624_937],
    [-0.584_674_001, 0.159_999_996, 0.256_624_937],
    [-0.335_326_016, 0.159_999_996, 0.543_375_075],
    [-0.335_326_016, -0.159_999_996, 0.543_375_075],
    [0.032_899_976, 0.823_385_417, -0.423_944_533],
    [0.398_868_173, 0.721_075_714, -0.423_944_533],
    [0.427_100_033, 0.822_062_79, 0.023_944_512],
    [0.061_131_835, 0.924_372_494, 0.023_944_512],
    [-0.004_440_695, -0.631_896_436, -0.381_558_806],
    [-0.415_827_096, -0.475_815_862, -0.381_558_806],
    [-0.455_559_313, -0.580_539_167, -0.018_441_2],
    [-0.044_172_883, -0.736_619_771, -0.018_441_2],
    [-0.389_664_471, -0.253_871_441, 0.084_180_601],
    [-0.418_767_273, 0.190_460_876, 0.043_158_308],
    [-0.100_889_802, 0.190_460_876, 0.408_717_394],
    [-0.100_889_802, -0.190_460_876, 0.408_717_394],
    [-0.092_870_966, 0.441_832_155, -0.367_779_285],
    [0.308_536_083, 0.329_615_146, -0.367_779_285],
    [0.341_284_722, 0.446_759_075, 0.151_767_343],
    [-0.060_122_311, 0.558_976_114, 0.151_767_343],
    [0.110_548_407, -0.341_685_116, -0.342_605_68],
    [-0.309_374_183, -0.182_365_805, -0.342_605_68],
    [0.059_360_977, -0.476_601_303, 0.125_202_894],
];
