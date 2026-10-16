#![expect(
    clippy::unwrap_used,
    reason = "integration tests use concise assertions"
)]

//! Skin parity against Blender 5.2.2. Expected geometry is captured by
//! `tests/fixtures/pr18f_skin_oracle.py`; vertices match by position and faces by vertex cycle,
//! so storage order does not matter.

use std::collections::HashMap;

use glam::DVec3;
use potter_core::{
    error::ErrorCode,
    geom::{Mesh, modifiers::evaluate_modifiers},
    model::{Id, Modifier, ModifierRuntime},
};
use serde_json::{Value, json};

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
    [-0.2, 0.8227241335952168, 0.23],
    [-0.2, -0.6062177826491069, -0.23],
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
    [0.23, 0.8227241335952168, -0.2],
    [-0.23, -0.6062177826491069, -0.2],
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
    [1.0, 0.200000018, -0.400000006],
    [1.25298214, -0.106274188, -0.353137106],
    [0.99999994, -0.200000018, 0.400000006],
    [0.747017741, 0.106274188, 0.353137106],
    [0.907180786, 1.357104421, 0.290569425],
    [1.388847351, 1.001550913, 0.330307961],
    [1.092819214, 0.642895639, 0.709430575],
    [0.611152649, 0.998449206, 0.66969198],
    [2.313684464, 1.47735703, 1.790877104],
    [2.425545216, 1.329283118, 1.71630311],
    [1.686315417, 0.52264297, 2.209122896],
    [1.574454665, 0.670716882, 2.28369689],
    [0.535886765, 0.339233965, -0.330383003],
    [0.535886765, -0.339233965, -0.330383003],
    [0.535886765, -0.339233965, 0.330383003],
    [0.535886765, 0.339233965, 0.330383003],
    [0.932917953, 0.690000057, -0.130000114],
    [1.380131602, 0.590000033, 0.069999933],
    [1.067082047, 0.309999943, 0.630000114],
    [0.619868398, 0.409999967, 0.430000067],
    [1.411425829, 1.440070391, 0.884754479],
    [1.678958654, 1.085928202, 0.706399322],
    [1.196917295, 0.559929669, 1.027760267],
    [0.92938447, 0.914071798, 1.206115365],
    [1.852741241, 1.45830822, 1.327959657],
    [2.044131279, 1.204958677, 1.200366139],
    [1.436293125, 0.54169184, 1.605591655],
    [1.244903088, 0.795041382, 1.733185172],
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

/// Every expected Blender vertex must have an evaluated vertex at the same position, and the faces
/// must match as vertex cycles over those vertices (winding included).
fn assert_blender_geometry<F: AsRef<[usize]>>(
    result: &Mesh,
    vertices: &[[f64; 3]],
    faces: &[F],
) {
    assert_eq!(result.vertices.len(), vertices.len());
    assert_eq!(result.faces.len(), faces.len());
    let mut matched = vec![false; result.vertices.len()];
    let mut expected_index = HashMap::with_capacity(vertices.len());
    for (index, position) in vertices.iter().enumerate() {
        let position = DVec3::from_array(*position);
        let storage = result
            .vertices
            .iter()
            .enumerate()
            .find_map(|(storage, vertex)| {
                (!matched[storage] && vertex.co.distance(position) < 1.0e-6)
                    .then_some(storage)
            })
            .expect("Blender Skin vertex is present");
        matched[storage] = true;
        expected_index.insert(storage, index);
    }
    let storage_of_id: HashMap<u32, usize> = result
        .vertices
        .iter()
        .enumerate()
        .map(|(storage, vertex)| (vertex.id, storage))
        .collect();
    let mut actual: Vec<Vec<usize>> = result
        .faces
        .iter()
        .map(|face| {
            let cycle: Vec<usize> = face
                .vertices
                .iter()
                .map(|id| expected_index[&storage_of_id[id]])
                .collect();
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
    assert_eq!(
        result.edges.len(),
        expected["edges"].as_u64().expect("Skin oracle edge count") as usize
    );
    assert_eq!(
        smooth_face_count(result),
        smooth.iter().filter(|value| **value).count()
    );
}


#[test]
fn line_skin_matches_blender() {
    let mesh = skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&LINE_SKIN));
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))]).unwrap();
    assert_blender_geometry(&result, &LINE_VERTICES, &LINE_FACES);
    assert_eq!(smooth_face_count(&result), 0);
}

#[test]
fn line_without_skin_layer_uses_blender_default_root() {
    // Blender adds a skin layer (radius 0.25, root on vertex 0) when the mesh has none.
    let mesh = skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, None);
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))]).unwrap();
    assert_blender_geometry(&result, &LINE_VERTICES, &LINE_FACES);
}

#[test]
fn line_use_smooth_shade_marks_every_face_smooth() {
    let mesh = skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&LINE_SKIN));
    let result =
        evaluate_modifiers(&mesh, &[skin_modifier(json!({"use_smooth_shade": true}))]).unwrap();
    assert_blender_geometry(&result, &LINE_VERTICES, &LINE_FACES);
    assert_eq!(smooth_face_count(&result), 18);
}

#[test]
fn bent_chain_skin_matches_blender() {
    let mesh = skeleton(&BENT_INPUT_VERTICES, &BENT_EDGES, Some(&BENT_SKIN));
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))]).unwrap();
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
        let result = evaluate_modifiers(&mesh, &[skin_modifier(params)]).unwrap();
        assert_blender_geometry(&result, &BENT_VERTICES, &BENT_FACES);
    }
}

#[test]
fn bent_chain_use_smooth_shade_marks_every_face_smooth() {
    let mesh = skeleton(&BENT_INPUT_VERTICES, &BENT_EDGES, Some(&BENT_SKIN));
    let result =
        evaluate_modifiers(&mesh, &[skin_modifier(json!({"use_smooth_shade": true}))]).unwrap();
    assert_blender_geometry(&result, &BENT_VERTICES, &BENT_FACES);
    assert_eq!(smooth_face_count(&result), 30);
}

#[test]
fn branch_smoothing_outside_zero_to_one_is_invalid() {
    let mesh = skeleton(&BENT_INPUT_VERTICES, &BENT_EDGES, Some(&BENT_SKIN));
    let error =
        evaluate_modifiers(&mesh, &[skin_modifier(json!({"branch_smoothing": 1.5}))]).unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidArgument);
}

#[test]
fn y_branch_matches_blender_with_default_and_disabled_symmetry() {
    let mesh = skeleton(&Y_INPUT_VERTICES, &Y_EDGES, Some(&Y_SKIN));
    let baseline = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))]).unwrap();
    assert_blender_geometry(&baseline, &Y_BRANCH_VERTICES, &Y_BRANCH_FACES);
    let symmetry_off = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({
            "use_x_symmetry": false,
            "use_y_symmetry": false,
            "use_z_symmetry": false
        }))],
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
    )
    .unwrap();
    let asymmetric = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({
            "use_x_symmetry": false,
            "use_y_symmetry": false,
            "use_z_symmetry": false
        }))],
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
    )
    .unwrap();
    let asymmetric = evaluate_modifiers(
        &mesh,
        &[skin_modifier(json!({
            "use_x_symmetry": false,
            "use_y_symmetry": false,
            "use_z_symmetry": false
        }))],
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
    )
    .unwrap();
    assert_blender_geometry(&result, &Y_BRANCH_SMOOTH_VERTICES, &Y_BRANCH_FACES);
}

#[test]
fn detached_branch_matches_blender_after_topology_repair() {
    let vertices = [
        [0.0, 0.0, 0.0],
        [0.0944871312282557, 1.3947412142144908, -0.6789465782603507],
        [0.582235576162845, 0.133558627622563, -1.9779341711897558],
        [0.4590388331551346, 1.4484963036383605, 0.06981012119291923],
    ];
    let edges = [[0, 1], [0, 2], [0, 3]];
    let skin = [
        ([0.46170452834529013, 0.3036252966189872], true),
        ([0.5412495143022006, 0.6302679453037413], false),
        ([0.14344811822036752, 0.12353176588156564], false),
        ([0.379951191640614, 0.26134121447787595], false),
    ];
    let result = evaluate_modifiers(
        &skeleton(&vertices, &edges, Some(&skin)),
        &[skin_modifier(json!({}))],
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
        [-0.615750305639683, -0.2491899355187869, -0.656205807326401],
        [0.47299044158676695, 0.6302041847568558, -0.7520664268446838],
        [0.4316167520938852, 0.7697423721761393, -1.843958393914058],
        [-0.2658301541233946, -0.615534116543724, 0.2375020748994344],
        [0.3887711823897751, -2.367231987934589, 1.2562332708604813],
    ];
    let edges = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]];
    let skin = [
        ([0.5944025940585477, 0.3933926275328914], true),
        ([0.7498587561158992, 0.8263900603670049], false),
        ([0.7504624463412104, 0.5995031041071985], false),
        ([0.36174570953694996, 0.3270112984977598], false),
        ([0.8666491075896481, 0.3874494757263359], false),
        ([0.3010493443071548, 0.646648963948643], false),
    ];
    let result = evaluate_modifiers(
        &skeleton(&vertices, &edges, Some(&skin)),
        &[skin_modifier(json!({}))],
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
        [0.0944871312282557, 1.3947412142144908, -0.6789465782603507],
        [0.582235576162845, 0.133558627622563, -1.9779341711897558],
        [0.4590388331551346, 1.4484963036383605, 0.06981012119291923],
    ];
    let edges = [[0, 1], [0, 2], [0, 3]];
    let skin = [
        ([0.46170452834529013, 0.3036252966189872], true),
        ([0.5412495143022006, 0.6302679453037413], false),
        ([0.14344811822036752, 0.12353176588156564], false),
        ([0.379951191640614, 0.26134121447787595], false),
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

    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))]).unwrap();
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
    assert!(point_values.iter().all(|value| value.as_f64() == Some(1.75)));
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
                    json!({"radius": radius, "root": root, "loose": index == 0})
                })
                .collect(),
        ),
    );
    let loose = evaluate_modifiers(&loose_mesh, &[skin_modifier(json!({}))]).unwrap();
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
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))]).unwrap();
    assert_blender_geometry(&result, &MIDDLE_ROOT_VERTICES, &MIDDLE_ROOT_FACES);
}

#[test]
fn isolated_skin_vertex_builds_blender_box() {
    let verts = [[0.0, 0.0, 0.0]];
    let result = evaluate_modifiers(
        &skeleton(&verts, &[], None),
        &[skin_modifier(json!({}))],
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
    )
    .unwrap();
    assert_eq!(result.vertices.len(), 28);
    assert_eq!(result.faces.len(), 26);
}

#[test]
fn negative_skin_radii_are_accepted() {
    let skin = [
        ([-0.25, -0.25], true),
        ([-0.25, -0.25], false),
        ([-0.25, -0.25], false),
    ];
    let result = evaluate_modifiers(
        &skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&skin)),
        &[skin_modifier(json!({}))],
    )
    .unwrap();
    assert_eq!(result.vertices.len(), 12);
    assert_eq!(result.faces.len(), 10);
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
    )
    .unwrap();
    assert_eq!(result.vertices.len(), 8);
    assert_eq!(result.faces.len(), 6);
}
#[test]
fn skeleton_without_root_is_invalid() {
    let mesh = skeleton(&LINE_INPUT_VERTICES, &LINE_EDGES, Some(&LINE_NO_ROOT_SKIN));
    let error = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))]).unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidArgument);
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
    [0.355112076, -0.25, 0.420448214],
    [-0.355112076, -0.25, 0.420448214],
    [-0.355112076, 0.25, 0.420448214],
    [0.355112076, 0.25, 0.420448214],
    [0.359692246, -0.25, 1.561231017],
    [-0.359692246, -0.25, 1.561231017],
    [-0.359692246, 0.25, 1.561231017],
    [0.359692246, 0.25, 1.561231017],
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
    [-0.176776648, -0.176776752, -0.25],
    [0.176776648, 0.176776752, -0.25],
    [0.176776648, 0.176776752, 0.25],
    [-0.176776648, -0.176776752, 0.25],
    [1.176776767, -0.176776662, -0.25],
    [0.823223233, 0.176776662, -0.25],
    [0.823223233, 0.176776662, 0.25],
    [1.176776767, -0.176776662, 0.25],
    [1.176776648, 1.176776767, -0.25],
    [0.823223352, 0.823223293, -0.25],
    [0.823223352, 0.823223293, 0.25],
    [1.176776648, 1.176776767, 0.25],
    [-0.176776707, 1.176776648, -0.25],
    [0.176776707, 0.823223293, -0.25],
    [0.176776707, 0.823223293, 0.25],
    [-0.176776707, 1.176776648, 0.25],
    [0.333333403, -0.25, -0.25],
    [0.333333284, 0.25, -0.25],
    [0.333333284, 0.25, 0.25],
    [0.333333403, -0.25, 0.25],
    [0.666666746, -0.25, -0.25],
    [0.666666627, 0.25, -0.25],
    [0.666666627, 0.25, 0.25],
    [0.666666746, -0.25, 0.25],
    [1.25, 0.333333373, -0.25],
    [0.75, 0.333333313, -0.25],
    [0.75, 0.333333313, 0.25],
    [1.25, 0.333333373, 0.25],
    [1.25, 0.666666746, -0.25],
    [0.75, 0.666666627, -0.25],
    [0.75, 0.666666627, 0.25],
    [1.25, 0.666666746, 0.25],
    [0.666666627, 1.25, -0.25],
    [0.666666627, 0.75, -0.25],
    [0.666666627, 0.75, 0.25],
    [0.666666627, 1.25, 0.25],
    [0.333333284, 1.25, -0.25],
    [0.333333343, 0.75, -0.25],
    [0.333333343, 0.75, 0.25],
    [0.333333284, 1.25, 0.25],
    [-0.25, 0.666666627, -0.25],
    [0.25, 0.666666627, -0.25],
    [0.25, 0.666666627, 0.25],
    [-0.25, 0.666666627, 0.25],
    [-0.25, 0.333333313, -0.25],
    [0.25, 0.333333313, -0.25],
    [0.25, 0.333333313, 0.25],
    [-0.25, 0.333333313, 0.25],
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
    [0.25, -0.25, 0.333333343],
    [-0.25, -0.25, 0.333333343],
    [-0.25, 0.25, 0.333333343],
    [0.25, 0.25, 0.333333343],
    [0.25, -0.25, 0.666666687],
    [-0.25, -0.25, 0.666666687],
    [-0.25, 0.25, 0.666666687],
    [0.25, 0.25, 0.666666687],
    [0.25, -0.25, 1.333333254],
    [-0.25, -0.25, 1.333333254],
    [-0.25, 0.25, 1.333333254],
    [0.25, 0.25, 1.333333254],
    [0.25, -0.25, 1.666666746],
    [-0.25, -0.25, 1.666666746],
    [-0.25, 0.25, 1.666666746],
    [0.25, 0.25, 1.666666746],
    [0.25, -0.25, 2.333333254],
    [-0.25, -0.25, 2.333333254],
    [-0.25, 0.25, 2.333333254],
    [0.25, 0.25, 2.333333254],
    [0.25, -0.25, 2.666666508],
    [-0.25, -0.25, 2.666666508],
    [-0.25, 0.25, 2.666666508],
    [0.25, 0.25, 2.666666508],
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
    [3.400000095, -0.200000003, 0.0],
    [2.599999905, -0.200000003, 0.0],
    [2.599999905, 0.200000003, 0.0],
    [3.400000095, 0.200000003, 0.0],
    [3.25, -0.25, 1.0],
    [2.75, -0.25, 1.0],
    [2.75, 0.25, 1.0],
    [3.25, 0.25, 1.0],
    [0.25, -0.25, 0.333333343],
    [-0.25, -0.25, 0.333333343],
    [-0.25, 0.25, 0.333333343],
    [0.25, 0.25, 0.333333343],
    [0.25, -0.25, 0.666666687],
    [-0.25, -0.25, 0.666666687],
    [-0.25, 0.25, 0.666666687],
    [0.25, 0.25, 0.666666687],
    [3.32054019, -0.226486579, 0.529731572],
    [2.67945981, -0.226486579, 0.529731572],
    [2.67945981, 0.226486579, 0.529731572],
    [3.32054019, 0.226486579, 0.529731572],
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
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))]).unwrap();
    assert_blender_geometry(&result, &CYCLE_VERTICES, &CYCLE_FACES);
}

#[test]
fn two_roots_on_one_skeleton_match_blender() {
    let mesh = skeleton(
        &TWO_ROOTS_LINE_INPUT_VERTICES,
        &TWO_ROOTS_LINE_EDGES,
        Some(&TWO_ROOTS_LINE_SKIN),
    );
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))]).unwrap();
    assert_blender_geometry(&result, &TWO_ROOTS_LINE_VERTICES, &TWO_ROOTS_LINE_FACES);
}

#[test]
fn two_islands_each_with_a_root_match_blender() {
    let mesh = skeleton(
        &TWO_ISLANDS_INPUT_VERTICES,
        &TWO_ISLANDS_EDGES,
        Some(&TWO_ISLANDS_SKIN),
    );
    let result = evaluate_modifiers(&mesh, &[skin_modifier(json!({}))]).unwrap();
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
    [0.0, 0.300000012, -0.300000012],
    [0.0, -0.300000012, -0.300000012],
    [0.0, -0.300000012, 0.300000012],
    [0.0, 0.300000012, 0.300000012],
    [1.858578682, 1.141421318, -0.200000003],
    [2.141421318, 0.858578622, -0.200000003],
    [2.141421318, 0.858578622, 0.200000003],
    [1.858578682, 1.141421318, 0.200000003],
    [2.188561678, -0.905719161, 0.311438203],
    [1.905719161, -1.188561797, 0.311438203],
    [1.811438203, -1.094280839, 0.688561797],
    [2.094280958, -0.811438203, 0.688561797],
    [0.5, 0.300000012, -0.300000012],
    [0.5, -0.300000012, -0.300000012],
    [0.5, -0.300000012, 0.300000012],
    [0.5, 0.300000012, 0.300000012],
    [1.216486692, 0.584138036, -0.259968787],
    [1.584138155, 0.216486648, -0.259968787],
    [1.584138155, 0.216486648, 0.259968787],
    [1.216486692, 0.584138036, 0.259968787],
    [1.551579595, 0.87497133, -0.22867246],
    [1.874971271, 0.551579654, -0.22867246],
    [1.874971271, 0.551579654, 0.22867246],
    [1.551579595, 0.87497133, 0.22867246],
    [1.568126321, -0.188407242, -0.095655948],
    [1.188407302, -0.56812638, -0.095655948],
    [1.061834216, -0.441553324, 0.410636216],
    [1.441553235, -0.061834194, 0.410636216],
    [1.791160345, -0.446266323, 0.050686151],
    [1.446266413, -0.791160405, 0.050686151],
    [1.331301689, -0.676195681, 0.510544896],
    [1.676195621, -0.331301659, 0.510544896],
    [1.995495558, -0.6825068, 0.18475908],
    [1.6825068, -0.995495439, 0.18475908],
    [1.578177214, -0.891165853, 0.602077246],
    [1.891165972, -0.578177214, 0.602077246],
];
const Y_BRANCH_SMOOTH_VERTICES: [[f64; 3]; 36] = [
    [0.0, 0.300000012, -0.300000012],
    [0.0, -0.300000012, -0.300000012],
    [0.0, -0.300000012, 0.300000012],
    [0.0, 0.300000012, 0.300000012],
    [1.858578682, 1.141421318, -0.200000003],
    [2.141421318, 0.858578622, -0.200000003],
    [2.141421318, 0.858578622, 0.200000003],
    [1.858578682, 1.141421318, 0.200000003],
    [2.188561678, -0.905719161, 0.311438203],
    [1.905719161, -1.188561797, 0.311438203],
    [1.811438203, -1.094280839, 0.688561797],
    [2.094280958, -0.811438203, 0.688561797],
    [0.680062532, 0.260062486, -0.231993765],
    [0.586050928, -0.258515805, -0.199457005],
    [0.570229292, -0.242694169, 0.238829538],
    [0.627832115, 0.194258481, 0.247060508],
    [1.172882318, 0.502664328, -0.199980497],
    [1.428843737, 0.154958129, -0.192591116],
    [1.520358562, 0.227652967, 0.213810027],
    [1.204109907, 0.454361856, 0.227044895],
    [1.551579595, 0.87497133, -0.22867246],
    [1.874971271, 0.551579654, -0.22867246],
    [1.874971271, 0.551579654, 0.22867246],
    [1.551579595, 0.87497133, 0.22867246],
    [1.506841302, -0.169438779, -0.052908532],
    [1.184454203, -0.41222322, -0.081892416],
    [1.00009656, -0.327928066, 0.337879747],
    [1.407990456, -0.020034105, 0.329873502],
    [1.791160345, -0.446266323, 0.050686151],
    [1.446266413, -0.791160405, 0.050686151],
    [1.331301689, -0.676195681, 0.510544896],
    [1.676195621, -0.331301659, 0.510544896],
    [1.995495558, -0.6825068, 0.18475908],
    [1.6825068, -0.995495439, 0.18475908],
    [1.578177214, -0.891165853, 0.602077246],
    [1.891165972, -0.578177214, 0.602077246],
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
    [0.256624937, 0.159999996, -0.584674001],
    [0.256624937, -0.159999996, -0.584674001],
    [0.543375075, -0.159999996, -0.335326016],
    [0.543375075, 0.159999996, -0.335326016],
    [-0.398865491, 0.836430848, 0.008043587],
    [-0.02961923, 0.926192701, 0.008043587],
    [-0.001134502, 0.80901736, 0.451956421],
    [-0.370380789, 0.719255507, 0.451956421],
    [0.029101707, -0.613983154, -0.408751547],
    [-0.388745546, -0.476129293, -0.408751547],
    [-0.429101706, -0.598452449, -0.051248476],
    [-0.011254467, -0.73630631, -0.051248476],
    [0.043158308, 0.190460876, -0.418767273],
    [0.086855657, -0.255753219, -0.387030751],
    [0.408717394, -0.190460876, -0.100889802],
    [0.408717394, 0.190460876, -0.100889802],
    [-0.327028245, 0.463029683, -0.13326025],
    [0.077974297, 0.561483741, -0.13326025],
    [0.111016281, 0.425561547, 0.381673992],
    [-0.293986231, 0.327107489, 0.381673992],
    [-0.29596445, -0.18033126, -0.355294228],
    [-0.347955793, -0.33792159, 0.10528104],
    [0.078561679, -0.478635877, 0.10528104],
];

const SYMMETRY_Z_BRANCH_VERTICES: [[f64; 3]; 23] = [
    [-0.584674001, -0.159999996, 0.256624937],
    [-0.584674001, 0.159999996, 0.256624937],
    [-0.335326016, 0.159999996, 0.543375075],
    [-0.335326016, -0.159999996, 0.543375075],
    [0.032899976, 0.823385417, -0.423944533],
    [0.398868173, 0.721075714, -0.423944533],
    [0.427100033, 0.82206279, 0.023944512],
    [0.061131835, 0.924372494, 0.023944512],
    [-0.004440695, -0.631896436, -0.381558806],
    [-0.415827096, -0.475815862, -0.381558806],
    [-0.455559313, -0.580539167, -0.0184412],
    [-0.044172883, -0.736619771, -0.0184412],
    [-0.389664471, -0.253871441, 0.084180601],
    [-0.418767273, 0.190460876, 0.043158308],
    [-0.100889802, 0.190460876, 0.408717394],
    [-0.100889802, -0.190460876, 0.408717394],
    [-0.092870966, 0.441832155, -0.367779285],
    [0.308536083, 0.329615146, -0.367779285],
    [0.341284722, 0.446759075, 0.151767343],
    [-0.060122311, 0.558976114, 0.151767343],
    [0.110548407, -0.341685116, -0.34260568],
    [-0.309374183, -0.182365805, -0.34260568],
    [0.059360977, -0.476601303, 0.125202894],
];
