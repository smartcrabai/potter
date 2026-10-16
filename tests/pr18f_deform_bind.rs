//! Deformation bind state and parameter parity for Surface Deform, Mesh Deform, Laplacian Deform,
//! and the Corrective Smooth bind data, including chained deforms.
//!
//! Surface Deform and Laplacian Deform positions were produced by Blender 5.2.2 (headless) on the
//! same fixtures. Mesh Deform expectations are the Blender-measured invariants: a uniform cage
//! translation moves each vertex by the translation times its group weight, and moving the cage or
//! owner object after binding has no effect.

use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    model::{Id, SceneDoc},
};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult = Result<(), Box<dyn Error>>;

const SURFACE_TOLERANCE: f64 = 2.0e-5;
const LAPLACIAN_TOLERANCE: f64 = 1.0e-4;
const MESH_TOLERANCE: f64 = 1.0e-5;

const SUBJECT_LOCATION: [f64; 3] = [0.2, -0.1, 0.5];
const SUBJECT_ROTATION: [f64; 4] = [
    0.2672612419124244,
    0.5345224838248488,
    0.8017837257372732,
    0.0,
];
const SUBJECT_SCALE: [f64; 3] = [1.0, 0.8, 1.2];
const TARGET_LOCATION: [f64; 3] = [-0.4, 0.3, 0.2];
const TARGET_ROTATION: [f64; 4] = [0.0, 0.0, -0.2474039592545229, 0.9689124217106447];
const TARGET_SCALE: [f64; 3] = [1.3, 0.7, 1.1];
const TARGET_MOVED_LOCATION: [f64; 3] = [0.1, 0.05, 0.5];
const TARGET_MOVED_ROTATION: [f64; 4] = [0.0, 0.0, 0.3826834323650898, 0.9238795325112867];
const TARGET_DELTAS: [(usize, [f64; 3]); 4] = [
    (0, [0.0, 0.0, 0.035]),
    (1, [0.018, 0.0, -0.012]),
    (2, [0.0, -0.016, 0.024]),
    (3, [-0.011, 0.008, -0.019]),
];
const CAGE_TRANSLATION: [f64; 3] = [0.3, -0.2, 0.5];
const BLENDER_GRID_EDGES: [[usize; 2]; 40] = [
    [0, 5], [0, 1], [1, 6], [5, 6], [1, 2], [2, 7], [6, 7], [2, 3], [3, 8], [7, 8],
    [3, 4], [4, 9], [8, 9], [5, 10], [6, 11], [10, 11], [7, 12], [11, 12], [8, 13],
    [12, 13], [9, 14], [13, 14], [10, 15], [11, 16], [15, 16], [12, 17], [16, 17],
    [13, 18], [17, 18], [14, 19], [18, 19], [15, 20], [16, 21], [20, 21], [17, 22],
    [21, 22], [18, 23], [22, 23], [19, 24], [23, 24],
];


const SURFACE_DEFAULT: [[f64; 3]; 25] = [
    [-1.000000119, -0.999999821, -0.248274416], [-0.499999970, -1.000000000, -0.225197464], [-0.000000030, -0.999999881, -0.099941403],
    [0.499999940, -0.999999940, 0.025314709], [0.999999881, -1.000000000, 0.048391655], [-1.000000000, -0.499999970, -0.112059832],
    [-0.500000000, -0.499999970, -0.076482914], [0.000000000, -0.500000000, 0.061273109], [0.499999940, -0.499999940, 0.199029163],
    [1.000000000, -0.499999970, 0.234606087], [-1.000000000, 0.000000000, -0.048332900], [-0.499999970, -0.000000015, -0.000256091],
    [-0.000000004, 0.000000014, 0.149999976], [0.499999970, -0.000000051, 0.300256044], [1.000000000, -0.000000060, 0.348332971],
    [-1.000000119, 0.500000060, -0.162059769], [-0.500000417, 0.500000179, -0.101482846], [-0.000085619, 0.500144899, 0.061303847],
    [0.503811836, 0.507666528, 0.222910672], [1.010961294, 0.509083688, 0.278834611], [-0.998930335, 0.999674678, -0.348586559],
    [-0.498427778, 0.998802006, -0.275387555], [0.002112350, 1.006417155, -0.100004546], [0.504332244, 1.022444606, 0.074970193],
    [1.016770959, 1.031681538, 0.143233642],
];

const SURFACE_FALLOFF_GROUP: [[f64; 3]; 25] = [
    [-1.000000000, -1.000000000, -0.248274371], [-0.500000000, -1.000000000, -0.225197479], [-0.000000018, -0.999999940, -0.099941410],
    [0.500000000, -1.000000000, 0.025314681], [0.999999940, -1.000000000, 0.048391592], [-0.999999940, -0.499999940, -0.112059832],
    [-0.500000000, -0.500000000, -0.076482959], [0.000000000, -0.500000000, 0.061273113], [0.499999970, -0.499999970, 0.199029177],
    [1.000000000, -0.500000000, 0.234606072], [-1.000000000, 0.000000000, -0.048332941], [-0.499999970, -0.000000009, -0.000256087],
    [-0.000000001, 0.000000002, 0.150000006], [0.500000000, -0.000000014, 0.300256073], [1.000000000, -0.000000036, 0.348332942],
    [-1.000000000, 0.500000000, -0.162059829], [-0.500003338, 0.500001848, -0.101482272], [-0.000077665, 0.500140011, 0.061301228],
    [0.500457108, 0.500795543, 0.223891467], [1.003879905, 0.503499985, 0.282578498], [-0.999428213, 0.999826074, -0.348441243],
    [-0.499811947, 0.999856710, -0.275220215], [0.000760893, 1.002310634, -0.099964201], [0.503028810, 1.013453603, 0.075001098],
    [1.001869559, 1.004112601, 0.147844166],
];

const SURFACE_SPARSE: [[f64; 3]; 25] = [
    [-1.000000000, -1.000000000, -0.248274371], [-0.500000000, -1.000000000, -0.225197479], [-0.000000017, -0.999999940, -0.099941410],
    [0.499999970, -0.999999940, 0.025314702], [1.000000000, -1.000000000, 0.048391558], [-0.999999940, -0.499999970, -0.112059839],
    [-0.500000000, -0.500000000, -0.076482944], [0.000000000, -0.499999970, 0.061273109], [0.500000000, -0.500000000, 0.199029192],
    [0.999999940, -0.499999940, 0.234606072], [-1.000000000, 0.000000000, -0.048332945], [-0.499999970, -0.000000006, -0.000256085],
    [0.000000000, 0.000000000, 0.150000006], [0.499999970, -0.000000042, 0.300256073], [1.000000000, -0.000000054, 0.348332942],
    [-1.000000000, 0.500000000, -0.162059829], [-0.500000000, 0.500000000, -0.101482965], [-0.000025802, 0.500043333, 0.061282363],
    [0.502674222, 0.506610215, 0.223280236], [1.009895802, 0.508068979, 0.279388577], [-1.000000000, 1.000000000, -0.348274380],
    [-0.499327719, 0.999487698, -0.275278747], [0.001235721, 1.003754139, -0.099978343], [0.503032625, 1.016670823, 0.075104713],
    [1.000000000, 1.000000000, 0.148391560],
];

const SURFACE_INVERT: [[f64; 3]; 25] = [
    [-1.000000000, -0.999999881, -0.248274401], [-0.499999970, -1.000000000, -0.225197479], [0.000000000, -1.000000000, -0.099941403],
    [0.499999970, -1.000000000, 0.025314704], [0.999999940, -1.000000000, 0.048391595], [-1.000000000, -0.500000000, -0.112059847],
    [-0.500000000, -0.499999970, -0.076482929], [0.000000000, -0.500000000, 0.061273109], [0.500000000, -0.500000000, 0.199029192],
    [0.999999940, -0.499999940, 0.234606087], [-1.000000000, 0.000000000, -0.048332941], [-0.500000000, 0.000000000, -0.000256081],
    [-0.000000004, 0.000000011, 0.149999976], [0.500000000, -0.000000018, 0.300256073], [1.000000000, 0.000000000, 0.348332971],
    [-1.000000119, 0.500000060, -0.162059784], [-0.500001311, 0.500000715, -0.101482689], [0.000000000, 0.500000000, 0.061273117],
    [0.503061891, 0.505586445, 0.223114163], [1.004350305, 0.503752351, 0.282323420], [-1.000000000, 1.000000000, -0.348274380],
    [-0.498743713, 0.999042749, -0.275349349], [0.000845022, 1.002566934, -0.099966675], [0.500000000, 1.000000000, 0.075314678],
    [1.012795806, 1.026695848, 0.144576117],
];

const SURFACE_CHAINED: [[f64; 3]; 25] = [
    [-1.000000000, -0.999999762, -0.248274416], [-0.499999940, -1.000000000, -0.225197464], [-0.000000030, -0.999999881, -0.099941388],
    [0.499999940, -0.999999881, 0.025314685], [0.999999940, -1.000000000, 0.048391655], [-0.999999881, -0.499999940, -0.112059832],
    [-0.500000000, -0.499999940, -0.076482885], [0.000000000, -0.500000000, 0.061273105], [0.499999940, -0.499999940, 0.199029177],
    [1.000000000, -0.500000000, 0.234606117], [-1.000000000, 0.000000000, -0.048332870], [-0.500000000, -0.000000022, -0.000256076],
    [-0.000000005, 0.000000022, 0.149999961], [0.499999940, -0.000000069, 0.300256044], [1.000000000, -0.000000060, 0.348332971],
    [-1.000000238, 0.500000119, -0.162059754], [-0.500001788, 0.500000954, -0.101482525], [-0.000100406, 0.500172496, 0.061309226],
    [0.503819585, 0.507324815, 0.222898036], [1.010918498, 0.509232283, 0.278867066], [-0.998966813, 0.999685764, -0.348575920],
    [-0.498428702, 0.998802781, -0.275387436], [0.002112437, 1.006417274, -0.100004613], [0.504523218, 1.022438765, 0.074922636],
    [1.016382933, 1.032525659, 0.143427968],
];

const LAPLACIAN_BORDER_ITERATIONS_3: [[f64; 3]; 25] = [
    [-0.962602556, -1.010230660, 0.378446519], [-0.455155492, -1.020076036, 0.432966530], [0.050362136, -1.021329999, 0.468472689],
    [0.542759538, -1.021855831, 0.458822817], [1.039118409, -1.016809225, 0.398531556], [-0.949688315, -0.513273656, 0.433677137],
    [-0.450488985, -0.520104766, 0.458757460], [0.049812917, -0.522060454, 0.478633612], [0.546177804, -0.522890031, 0.475435257],
    [1.042777896, -0.522862494, 0.446842164], [-0.950206935, -0.020067114, 0.468752354], [-0.450363070, -0.021789888, 0.478724420],
    [0.050283052, -0.022632083, 0.492568016], [0.550176740, -0.023372583, 0.495592475], [1.051192403, -0.025260454, 0.489051968],
    [-0.951727629, 0.476578593, 0.459142596], [-0.451742917, 0.476954371, 0.475609243], [0.049820788, 0.477466971, 0.495704174],
    [0.550422728, 0.477511138, 0.501291394], [1.048423171, 0.478734940, 0.492642641], [-0.941845059, 0.980397582, 0.398959875],
    [-0.451838851, 0.976792634, 0.447196215], [0.049832959, 0.978310049, 0.489413291], [0.550407648, 0.978127301, 0.492817372],
    [1.038190722, 0.982824326, 0.444264412],
];

const LAPLACIAN_INTERIOR_INVERT: [[f64; 3]; 25] = [
    [-0.958089411, -1.015850425, 0.506027877], [-0.454937190, -1.017725229, 0.484991729], [0.049623404, -1.018995404, 0.457452953],
    [0.555069089, -1.018325686, 0.425366133], [1.057611227, -1.017662525, 0.404932201], [-0.954402626, -0.515427589, 0.484905183],
    [-0.453130156, -0.517412663, 0.469902426], [0.049750134, -0.518606186, 0.448435068], [0.553323269, -0.517659903, 0.422862589],
    [1.054753065, -0.515958607, 0.407223731], [-0.949228406, -0.014930387, 0.457237840], [-0.449368894, -0.017579850, 0.448358923],
    [0.049630415, -0.018272044, 0.432499856], [0.549712002, -0.017446764, 0.419227540], [1.049684048, -0.014630899, 0.409688234],
    [-0.945945919, 0.484292716, 0.425167799], [-0.447116762, 0.482461810, 0.422760308], [0.050560001, 0.481545657, 0.419192880],
    [0.546640038, 0.482969880, 0.416760534], [1.044965982, 0.485659033, 0.412739545], [-0.943410337, 0.983105540, 0.404726386],
    [-0.445490777, 0.981909990, 0.407103002], [0.050698020, 0.981150150, 0.409642160], [0.544844925, 0.982504308, 0.412732035],
    [1.041645288, 0.984405160, 0.412075460],
];

const LAPLACIAN_BORDER_ITERATIONS_1: [[f64; 3]; 25] = [
    [-0.960990131, -1.013216734, 0.380200803], [-0.452735722, -1.020231128, 0.427765638], [0.051025029, -1.020626903, 0.461828411],
    [0.542703390, -1.021567225, 0.459042609], [1.040146112, -1.017656326, 0.410211474], [-0.949912190, -0.516165614, 0.428017288],
    [-0.450328499, -0.520356774, 0.448653758], [0.049593769, -0.521468222, 0.467457414], [0.545805216, -0.522483289, 0.468409628],
    [1.042858481, -0.523405969, 0.449426919], [-0.950973392, -0.020844094, 0.461866409], [-0.450979739, -0.021490179, 0.467474699],
    [0.049640723, -0.021917911, 0.479574144], [0.549567997, -0.022763496, 0.485013515], [1.050696373, -0.025038136, 0.484243244],
    [-0.952994466, 0.477129608, 0.459127069], [-0.452706724, 0.477789760, 0.468468010], [0.049087580, 0.478307724, 0.485064268],
    [0.549989641, 0.478179365, 0.492588133], [1.048330069, 0.478991836, 0.490403414], [-0.944242299, 0.982314229, 0.410344929],
    [-0.453246534, 0.978283763, 0.449567020], [0.049075015, 0.979295492, 0.484419435], [0.550497293, 0.978973269, 0.490446925],
    [1.039763093, 0.983763993, 0.453088582],
];


fn subject_xy(index: usize) -> (f64, f64) {
    (
        -1.0 + 0.5 * (index % 5) as f64,
        -1.0 + 0.5 * (index / 5) as f64,
    )
}

fn subject_height(x: f64, y: f64) -> f64 {
    0.2 * (1.7 * x).sin() + 0.15 * (2.3 * y).cos() + 0.05 * x * y
}

fn target_height(x: f64, y: f64) -> f64 {
    0.3 * (1.1 * x).sin() + 0.2 * y * y - 0.1 * x * y + 0.05 * x
}

fn wobbled_subject(index: usize) -> [f64; 3] {
    let (x, y) = subject_xy(index);
    [x, y, subject_height(x, y)]
}

fn flat_subject(index: usize) -> [f64; 3] {
    let (x, y) = subject_xy(index);
    [x, y, 0.0]
}

fn init(scene: &Path) -> TestResult {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(scene)
        .arg("--json")
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "scene initialization failed: {}",
            String::from_utf8_lossy(&output.stdout)
        )
        .into());
    }
    Ok(())
}

fn apply(scene: &Path, operations: &Value) -> Result<Output, Box<dyn Error>> {
    let batch = scene.with_extension("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))?,
    )?;
    Ok(Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .arg("--json")
        .output()?)
}

fn applied(output: &Output) -> TestResult {
    if !output.status.success() {
        return Err(format!(
            "apply failed (status={}): stdout={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(())
}

fn load_doc(scene: &Path) -> Result<SceneDoc, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(
        scene.join("scene.json"),
    )?)?)
}

fn evaluated_subject(scene: &Path) -> Result<Vec<DVec3>, Box<dyn Error>> {
    let doc = load_doc(scene)?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    let mesh = snapshot
        .meshes
        .get(&Id::new("subject")?)
        .ok_or("evaluated subject mesh is missing")?;
    Ok(mesh.vertices.iter().map(|vertex| vertex.co).collect())
}
fn evaluated_laplacian_subject(scene: &Path) -> Result<Vec<DVec3>, Box<dyn Error>> {
    let mut doc = load_doc(scene)?;
    let subject_id = Id::new("subject")?;
    let data_id = doc
        .nodes
        .get(&subject_id)
        .and_then(|node| node.data.as_ref())
        .cloned()
        .ok_or("subject data is missing")?;
    {
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .and_then(|data| data.mesh.as_mut())
            .ok_or("subject mesh is missing")?;
        let mut reordered_edges = Vec::with_capacity(mesh.edges.len());
        for [first, second] in BLENDER_GRID_EDGES {
            let first_id = mesh
                .vertices
                .get(first)
                .ok_or("Blender edge vertex is outside the mesh")?
                .id;
            let second_id = mesh
                .vertices
                .get(second)
                .ok_or("Blender edge vertex is outside the mesh")?
                .id;
            let edge_index = mesh
                .edges
                .iter()
                .position(|edge| {
                    (edge.vertices[0] == first_id && edge.vertices[1] == second_id)
                        || (edge.vertices[0] == second_id && edge.vertices[1] == first_id)
                })
                .ok_or("Blender edge does not exist in the Potter mesh")?;
            reordered_edges.push(mesh.edges.remove(edge_index));
        }
        if !mesh.edges.is_empty() {
            return Err("Blender and Potter edge counts differ".into());
        }
        mesh.edges = reordered_edges;
    }
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    let mesh = snapshot
        .meshes
        .get(&subject_id)
        .ok_or("evaluated subject mesh is missing")?;
    Ok(mesh.vertices.iter().map(|vertex| vertex.co).collect())
}


fn assert_close(actual: &[DVec3], expected: &[[f64; 3]], tolerance: f64, label: &str) -> TestResult {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: {} vertices, expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut worst = (0_usize, 0.0_f64);
    for (index, (position, reference)) in actual.iter().zip(expected).enumerate() {
        let error = position.distance(DVec3::from_array(*reference));
        if error > worst.1 {
            worst = (index, error);
        }
    }
    if worst.1 > tolerance {
        return Err(format!(
            "{label}: vertex {} off by {} (tolerance {tolerance}); actual {:?}, expected {:?}",
            worst.0, worst.1, actual[worst.0], expected[worst.0]
        )
        .into());
    }
    Ok(())
}

fn vertex_move(node: &str, index: usize, translation: [f64; 3]) -> Value {
    json!({
        "op": "mesh.transform_elements",
        "target": {"id": node},
        "elements": {"domain": "vertex", "ids": [format!("v{index}")]},
        "translation": translation,
    })
}

fn bind(modifier: &str) -> Value {
    json!({"op": "modifier.bind", "target": {"id": "subject"}, "modifier_id": modifier})
}

fn unbind(modifier: &str) -> Value {
    json!({"op": "modifier.unbind", "target": {"id": "subject"}, "modifier_id": modifier})
}

fn group_operations(weights: &[f64]) -> Vec<Value> {
    let assignments = weights
        .iter()
        .enumerate()
        .map(|(index, weight)| json!({"vertex_id": index, "weight": weight}))
        .collect::<Vec<_>>();
    vec![
        json!({"op": "vertex_group.create", "target": {"id": "subject"}, "id": "influence", "name": "Influence"}),
        json!({"op": "vertex_group.assign", "target": {"id": "subject"}, "group_id": "influence", "weights": assignments}),
    ]
}

fn partial_weights() -> Vec<f64> {
    (0..25)
        .map(|index| 0.2 + 0.8 * ((index % 3) as f64 / 2.0))
        .collect()
}

fn sparse_weights() -> Vec<f64> {
    (0..25)
        .map(|index| {
            if index % 4 == 0 {
                0.0
            } else {
                0.3 + 0.7 * ((index % 5) as f64 / 4.0)
            }
        })
        .collect()
}

fn surface_operations(
    modifiers: &[(&str, Value)],
    group: Option<Vec<f64>>,
    operations_after_create: Vec<Value>,
    target_object_move: bool,
) -> Value {
    let mut operations = vec![
        json!({"op": "node.create", "id": "subject", "kind": "grid",
            "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4},
            "transform": {"translation": SUBJECT_LOCATION, "rotation": SUBJECT_ROTATION, "scale": SUBJECT_SCALE}}),
    ];
    for index in 0..25 {
        let (x, y) = subject_xy(index);
        operations.push(vertex_move("subject", index, [0.0, 0.0, subject_height(x, y)]));
    }
    operations.push(json!({"op": "node.create", "id": "target", "kind": "grid",
        "params": {"size": 3.0, "x_subdivisions": 3, "y_subdivisions": 3},
        "transform": {"translation": TARGET_LOCATION, "rotation": TARGET_ROTATION, "scale": TARGET_SCALE}}));
    for index in 0..16 {
        let x = -1.5 + (index % 4) as f64;
        let y = -1.5 + (index / 4) as f64;
        operations.push(vertex_move("target", index, [0.0, 0.0, target_height(x, y)]));
    }
    if let Some(weights) = group {
        operations.extend(group_operations(&weights));
    }
    for (id, params) in modifiers {
        operations.push(json!({"op": "modifier.create", "target": {"id": "subject"},
            "id": id, "type": "surface_deform", "params": params}));
    }
    operations.extend(operations_after_create);
    for (index, delta) in TARGET_DELTAS {
        operations.push(vertex_move("target", index, delta));
    }
    if target_object_move {
        operations.push(json!({"op": "node.update", "target": {"id": "target"},
            "set": {"transform": {"translation": TARGET_MOVED_LOCATION,
                "rotation": TARGET_MOVED_ROTATION, "scale": TARGET_SCALE}}}));
    }
    Value::Array(operations)
}

fn surface_params(falloff: f64, strength: f64) -> Value {
    json!({"target": "target", "falloff": falloff, "strength": strength})
}

fn build_scene(
    directory: &Path,
    name: &str,
    operations: &Value,
) -> Result<(Output, std::path::PathBuf), Box<dyn Error>> {
    let scene = directory.join(name);
    init(&scene)?;
    let output = apply(&scene, operations)?;
    Ok((output, scene))
}

#[test]
fn surface_deform_default_bind_matches_blender() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[("deform", surface_params(4.0, 1.0))],
        None,
        vec![bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "default", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_DEFAULT,
        SURFACE_TOLERANCE,
        "surface default",
    )
}

#[test]
fn surface_deform_falloff_strength_and_group_match_blender() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[(
            "deform",
            json!({"target": "target", "falloff": 2.5, "strength": 0.6, "vertex_group": "Influence"}),
        )],
        Some(partial_weights()),
        vec![bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "falloff_group", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_FALLOFF_GROUP,
        SURFACE_TOLERANCE,
        "surface falloff/strength/group",
    )
}

#[test]
fn surface_deform_sparse_bind_matches_blender() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[(
            "deform",
            json!({"target": "target", "falloff": 6.0, "strength": 0.9,
                "vertex_group": "Influence", "use_sparse_bind": true}),
        )],
        Some(sparse_weights()),
        vec![bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "sparse", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_SPARSE,
        SURFACE_TOLERANCE,
        "surface sparse bind",
    )
}

#[test]
fn surface_deform_inverted_group_matches_blender() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[(
            "deform",
            json!({"target": "target", "falloff": 3.0, "strength": 1.0,
                "vertex_group": "Influence", "invert_vertex_group": true}),
        )],
        Some(partial_weights()),
        vec![bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "invert", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_INVERT,
        SURFACE_TOLERANCE,
        "surface inverted group",
    )
}
#[test]
fn surface_deform_missing_vertex_group_acts_as_no_group() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[(
            "deform",
            json!({
                "target": "target",
                "falloff": 4.0,
                "strength": 1.0,
                "vertex_group": "Missing",
            }),
        )],
        None,
        vec![bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "missing_surface_group", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_DEFAULT,
        SURFACE_TOLERANCE,
        "surface missing vertex group",
    )
}


#[test]
fn surface_deform_target_object_transform_is_frozen_at_bind() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[("deform", surface_params(4.0, 1.0))],
        None,
        vec![bind("deform")],
        true,
    );
    let (output, scene) = build_scene(directory.path(), "target_moved", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_DEFAULT,
        SURFACE_TOLERANCE,
        "surface target object moved after bind",
    )
}

#[test]
fn surface_deform_chained_binds_match_blender() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[
            ("first", surface_params(4.0, 1.0)),
            ("second", surface_params(3.0, 0.5)),
        ],
        None,
        vec![bind("first"), bind("second")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "chained", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_CHAINED,
        SURFACE_TOLERANCE,
        "surface chained",
    )
}

#[test]
fn surface_deform_sparse_bind_without_group_is_rejected() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[(
            "deform",
            json!({"target": "target", "falloff": 4.0, "strength": 1.0, "use_sparse_bind": true}),
        )],
        None,
        vec![bind("deform")],
        false,
    );
    let (output, _scene) = build_scene(directory.path(), "sparse_no_group", &operations)?;
    if output.status.success() {
        return Err("sparse surface bind without any bound vertex must fail".into());
    }
    Ok(())
}

#[test]
fn unbound_surface_deform_leaves_the_mesh_unchanged() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[("deform", surface_params(4.0, 1.0))],
        None,
        Vec::new(),
        false,
    );
    let (output, scene) = build_scene(directory.path(), "unbound", &operations)?;
    applied(&output)?;
    let rest = (0..25).map(wobbled_subject).collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &rest,
        SURFACE_TOLERANCE,
        "unbound surface deform",
    )
}

#[test]
fn surface_deform_rebind_after_unbind_matches_default_bind() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[("deform", surface_params(4.0, 1.0))],
        None,
        vec![bind("deform"), unbind("deform"), bind("deform")],
        false,
    );
    let (output, scene) = build_scene(directory.path(), "rebind", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_subject(&scene)?,
        &SURFACE_DEFAULT,
        SURFACE_TOLERANCE,
        "surface rebind after unbind",
    )
}

#[test]
fn surface_deform_second_bind_on_bound_modifier_is_rejected() -> TestResult {
    let directory = tempdir()?;
    let operations = surface_operations(
        &[("deform", surface_params(4.0, 1.0))],
        None,
        vec![bind("deform"), bind("deform")],
        false,
    );
    let (output, _scene) = build_scene(directory.path(), "double_bind", &operations)?;
    if output.status.success() {
        return Err("binding an already bound surface deform must fail".into());
    }
    Ok(())
}

fn mesh_operations(
    group: Option<Vec<f64>>,
    precision: u64,
    dynamic: bool,
    invert: bool,
    operations_after_bind: Vec<Value>,
) -> Value {
    let mut operations = vec![
        json!({"op": "node.create", "id": "subject", "kind": "grid",
            "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4}}),
        json!({"op": "node.create", "id": "cage", "kind": "box", "params": {"size": 4.0}}),
    ];
    let mut params = json!({"object": "cage", "precision": precision,
        "use_dynamic_bind": dynamic, "invert_vertex_group": invert});
    if let Some(weights) = group {
        operations.extend(group_operations(&weights));
        params["vertex_group"] = json!("Influence");
    }
    operations.push(json!({"op": "modifier.create", "target": {"id": "subject"},
        "id": "deform", "type": "mesh_deform", "params": params}));
    operations.push(bind("deform"));
    operations.extend(operations_after_bind);
    Value::Array(operations)
}

fn cage_translation_operations() -> Vec<Value> {
    (0..8)
        .map(|index| vertex_move("cage", index, CAGE_TRANSLATION))
        .collect()
}
fn asymmetric_mesh_operations(precision: u64, dynamic: bool, invert: bool) -> Value {
    let mut operations = vec![
        json!({"op": "node.create", "id": "subject", "kind": "grid",
            "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4},
            "transform": {"translation": [0.2, -0.1, 0.5],
                "rotation": [0.0, 0.0, 0.14943813247359922, 0.9887710779360422],
                "scale": [0.9, 1.2, 0.75]}}),
        json!({"op": "node.create", "id": "cage", "kind": "box",
            "params": {"size": 4.0},
            "transform": {"translation": [-0.4, 0.3, -0.2],
                "rotation": [0.0, 0.0, -0.09983341664682815, 0.9950041652780258],
                "scale": [1.3, 0.8, 1.1]}}),
    ];
    let base = [
        [-2.0, -2.0, -2.0],
        [2.0, -2.0, -2.0],
        [2.0, 2.0, -2.0],
        [-2.0, 2.0, -2.0],
        [-2.0, -2.0, 2.0],
        [2.0, -2.0, 2.0],
        [2.0, 2.0, 2.0],
        [-2.0, 2.0, 2.0],
    ];
    let asymmetric = [
        [-2.2, -1.8, -2.0],
        [2.1, -1.7, -2.1],
        [2.1, 1.9, -2.2],
        [-2.0, 1.7, -1.8],
        [-1.8, -1.9, 2.0],
        [2.1, -1.7, 2.2],
        [2.2, 1.8, 2.3],
        [-2.1, 1.6, 1.7],
    ];
    for (index, (from, to)) in base.iter().zip(asymmetric).enumerate() {
        operations.push(vertex_move(
            "cage",
            index,
            [to[0] - from[0], to[1] - from[1], to[2] - from[2]],
        ));
    }
    let weights = (0..25)
        .map(|index| 0.15 + 0.2 * (index % 5) as f64)
        .collect::<Vec<_>>();
    operations.extend(group_operations(&weights));
    operations.push(json!({"op": "modifier.create", "target": {"id": "subject"},
        "id": "deform", "type": "mesh_deform",
        "params": {"object": "cage", "precision": precision,
            "use_dynamic_bind": dynamic, "vertex_group": "Influence",
            "invert_vertex_group": invert}}));
    operations.push(bind("deform"));
    operations.push(vertex_move("cage", 0, [0.0, 0.0, 0.12]));
    operations.push(vertex_move("cage", 2, [0.1, -0.03, 0.0]));
    operations.push(vertex_move("cage", 6, [-0.04, 0.08, -0.06]));
    Value::Array(operations)
}


#[test]
fn mesh_deform_uniform_cage_translation_moves_every_vertex() -> TestResult {
    let directory = tempdir()?;
    let operations = mesh_operations(None, 3, false, false, cage_translation_operations());
    let (output, scene) = build_scene(directory.path(), "mesh_static", &operations)?;
    applied(&output)?;
    let expected = (0..25)
        .map(|index| {
            let rest = flat_subject(index);
            [
                rest[0] + CAGE_TRANSLATION[0],
                rest[1] + CAGE_TRANSLATION[1],
                rest[2] + CAGE_TRANSLATION[2],
            ]
        })
        .collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &expected,
        MESH_TOLERANCE,
        "mesh deform static translation",
    )
}
#[test]
fn mesh_deform_nonuniform_bind_transform_matches_blender_translation() -> TestResult {
    let directory = tempdir()?;
    let mut operations = mesh_operations(None, 3, false, false, cage_translation_operations());
    operations[0]["transform"] = json!({
        "translation": [0.2, -0.1, 0.5],
        "rotation": [0.0, 0.0, 0.14943813247359922, 0.9887710779360422],
        "scale": [0.9, 1.2, 0.75],
    });
    operations[1]["params"]["size"] = json!(6.0);
    operations[1]["transform"] = json!({
        "translation": [-0.4, 0.3, -0.2],
        "rotation": [0.0, 0.0, -0.09983341664682815, 0.9950041652780258],
        "scale": [1.3, 0.8, 1.1],
    });
    let (output, scene) = build_scene(directory.path(), "mesh_transformed", &operations)?;
    applied(&output)?;
    let expected = (0..25)
        .map(|index| {
            let rest = flat_subject(index);
            [
                rest[0] + 0.3830646276473999,
                rest[1] - 0.18392157554626465,
                rest[2] + 0.7455119490623474,
            ]
        })
        .collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &expected,
        MESH_TOLERANCE,
        "mesh deform nonuniform bind transform",
    )
}
#[test]
fn mesh_deform_precision_dynamic_weights_and_nonuniform_transforms_match_blender() -> TestResult {
    let cases = [
        (
            "mesh_asymmetric_static_p2",
            2,
            false,
            false,
            [
                [-0.999769687653, -0.999865770340, 0.003038096707],
                [-0.499045044184, -0.499446064234, 0.005308256950],
                [0.003932299092, 0.002456886228, 0.000047756672],
                [0.509214341640, 0.505858659744, -0.012196095660],
                [1.015522599220, 1.009907603264, -0.026900172234],
            ],
        ),
        (
            "mesh_asymmetric_static_p3",
            3,
            false,
            false,
            [
                [-1.0, -1.0, 0.0],
                [-0.499141335487, -0.499453783036, 0.005734985229],
                [0.003923542798, 0.002484298311, 0.000121743491],
                [0.509186029434, 0.505920290947, -0.012218339369],
                [1.016380310059, 1.010796308517, -0.030081091449],
            ],
        ),
        (
            "mesh_asymmetric_dynamic_p3_invert",
            3,
            true,
            true,
            [
                [-1.0, -1.0, 0.0],
                [-0.498405337334, -0.498985618353, 0.010650688782],
                [0.003210171591, 0.002032608027, 0.000099608587],
                [0.503062009811, 0.501973450184, -0.004072779324],
                [1.000862121582, 1.000568270683, -0.001583216246],
            ],
        ),
    ];
    let directory = tempdir()?;
    let sample_indices = [0, 6, 12, 18, 24];
    for (name, precision, dynamic, invert, expected) in cases {
        let operations = asymmetric_mesh_operations(precision, dynamic, invert);
        let (output, scene) = build_scene(directory.path(), name, &operations)?;
        applied(&output)?;
        let positions = evaluated_subject(&scene)?;
        let samples = sample_indices
            .iter()
            .map(|index| {
                positions
                    .get(*index)
                    .copied()
                    .ok_or_else(|| -> Box<dyn Error> {
                        Box::new(std::io::Error::other(format!(
                            "mesh deform sample vertex {index} is missing"
                        )))
                    })
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        assert_close(&samples, &expected, 1.0e-3, name)?;
    }
    Ok(())
}



#[test]
fn mesh_deform_inverted_group_scales_the_cage_motion() -> TestResult {
    let directory = tempdir()?;
    let weights = (0..25)
        .map(|index| 0.15 + 0.2 * (index % 5) as f64)
        .collect::<Vec<_>>();
    let operations = mesh_operations(
        Some(weights.clone()),
        4,
        false,
        true,
        cage_translation_operations(),
    );
    let (output, scene) = build_scene(directory.path(), "mesh_invert", &operations)?;
    applied(&output)?;
    let expected = (0..25)
        .map(|index| {
            let rest = flat_subject(index);
            let scale = 1.0 - weights[index];
            [
                rest[0] + CAGE_TRANSLATION[0] * scale,
                rest[1] + CAGE_TRANSLATION[1] * scale,
                rest[2] + CAGE_TRANSLATION[2] * scale,
            ]
        })
        .collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &expected,
        MESH_TOLERANCE,
        "mesh deform inverted group",
    )
}

#[test]
fn mesh_deform_dynamic_bind_scales_the_cage_motion_by_group_weight() -> TestResult {
    let directory = tempdir()?;
    let weights = (0..25)
        .map(|index| 0.15 + 0.2 * (index % 5) as f64)
        .collect::<Vec<_>>();
    let operations = mesh_operations(
        Some(weights.clone()),
        2,
        true,
        false,
        cage_translation_operations(),
    );
    let (output, scene) = build_scene(directory.path(), "mesh_dynamic", &operations)?;
    applied(&output)?;
    let expected = (0..25)
        .map(|index| {
            let rest = flat_subject(index);
            let scale = weights[index];
            [
                rest[0] + CAGE_TRANSLATION[0] * scale,
                rest[1] + CAGE_TRANSLATION[1] * scale,
                rest[2] + CAGE_TRANSLATION[2] * scale,
            ]
        })
        .collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &expected,
        MESH_TOLERANCE,
        "mesh deform dynamic group",
    )
}
#[test]
fn mesh_deform_missing_vertex_group_acts_as_no_group() -> TestResult {
    let directory = tempdir()?;
    let mut operations = mesh_operations(None, 3, false, false, cage_translation_operations());
    operations[2]["params"]["vertex_group"] = json!("Missing");
    let (output, scene) = build_scene(directory.path(), "missing_mesh_group", &operations)?;
    applied(&output)?;
    let expected = (0..25)
        .map(|index| {
            let rest = flat_subject(index);
            [
                rest[0] + CAGE_TRANSLATION[0],
                rest[1] + CAGE_TRANSLATION[1],
                rest[2] + CAGE_TRANSLATION[2],
            ]
        })
        .collect::<Vec<_>>();
    assert_close(
        &evaluated_subject(&scene)?,
        &expected,
        MESH_TOLERANCE,
        "mesh deform missing vertex group",
    )
}


#[test]
fn mesh_deform_ignores_cage_and_owner_object_motion_after_bind() -> TestResult {
    let rest = (0..25).map(flat_subject).collect::<Vec<_>>();
    let moves = [
        (
            "static cage object",
            false,
            vec![json!({"op": "node.update", "target": {"id": "cage"},
                "set": {"transform": {"translation": [0.6, -0.3, 0.4],
                    "rotation": [0.0, 0.0, 0.2474039592545229, 0.9689124217106447]}}})],
        ),
        (
            "dynamic cage object",
            true,
            vec![json!({"op": "node.update", "target": {"id": "cage"},
                "set": {"transform": {"translation": [0.6, -0.3, 0.4],
                    "rotation": [0.0, 0.0, 0.2474039592545229, 0.9689124217106447]}}})],
        ),
        (
            "owner object",
            false,
            vec![json!({"op": "node.update", "target": {"id": "subject"},
                "set": {"transform": {"translation": [0.5, 0.0, 0.0]}}})],
        ),
    ];
    let directory = tempdir()?;
    for (label, dynamic, after_bind) in moves {
        let operations = mesh_operations(None, 3, dynamic, false, after_bind);
        let name = label.replace(' ', "_");
        let (output, scene) = build_scene(directory.path(), &name, &operations)?;
        applied(&output)?;
        let actual = evaluated_subject(&scene)?;
        assert_close(&actual, &rest, MESH_TOLERANCE, label)?;
    }
    Ok(())
}

fn laplacian_operations(
    group_indices: &[usize],
    iterations: u64,
    invert: bool,
    moved: &[usize],
) -> Value {
    let weights = (0..25)
        .map(|index| if group_indices.contains(&index) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let mut operations = vec![json!({"op": "node.create", "id": "subject", "kind": "grid",
        "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4}})];
    operations.extend(group_operations(&weights));
    operations.push(json!({"op": "modifier.create", "target": {"id": "subject"}, "id": "laplace",
        "type": "laplacian_deform",
        "params": {"vertex_group": "Influence", "iterations": iterations, "invert_vertex_group": invert}}));
    operations.push(bind("laplace"));
    for &index in moved {
        operations.push(vertex_move(
            "subject",
            index,
            [
                0.05 * (index % 3) as f64,
                -0.04 * (index % 2) as f64,
                0.3 + 0.1 * (index % 4) as f64,
            ],
        ));
    }
    Value::Array(operations)
}

fn border_indices() -> Vec<usize> {
    (0..25)
        .filter(|index| index % 5 == 0 || index % 5 == 4 || *index < 5 || *index >= 20)
        .collect()
}

fn interior_indices() -> Vec<usize> {
    (0..25)
        .filter(|index| !border_indices().contains(index))
        .collect()
}

#[test]
fn laplacian_deform_border_anchors_with_three_iterations_match_blender() -> TestResult {
    let directory = tempdir()?;
    let border = border_indices();
    let operations = laplacian_operations(&border, 3, false, &border);
    let (output, scene) = build_scene(directory.path(), "laplacian_iterations", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_laplacian_subject(&scene)?,
        &LAPLACIAN_BORDER_ITERATIONS_3,
        LAPLACIAN_TOLERANCE,
        "laplacian border anchors, 3 iterations",
    )
}

#[test]
fn laplacian_deform_single_iteration_matches_blender() -> TestResult {
    let directory = tempdir()?;
    let border = border_indices();
    let operations = laplacian_operations(&border, 1, false, &border);
    let (output, scene) = build_scene(directory.path(), "laplacian_single", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_laplacian_subject(&scene)?,
        &LAPLACIAN_BORDER_ITERATIONS_1,
        LAPLACIAN_TOLERANCE,
        "laplacian border anchors, 1 iteration",
    )
}

#[test]
fn laplacian_deform_inverted_group_anchors_interior_vertices() -> TestResult {
    let directory = tempdir()?;
    let border = border_indices();
    let interior = interior_indices();
    let operations = laplacian_operations(&border, 1, true, &interior);
    let (output, scene) = build_scene(directory.path(), "laplacian_invert", &operations)?;
    applied(&output)?;
    assert_close(
        &evaluated_laplacian_subject(&scene)?,
        &LAPLACIAN_INTERIOR_INVERT,
        LAPLACIAN_TOLERANCE,
        "laplacian inverted group",
    )
}

#[test]
fn corrective_smooth_bind_stores_rest_coordinates() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("corrective_bind");
    init(&scene)?;
    let operations = json!([
        {"op": "node.create", "id": "subject", "kind": "grid",
            "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4}},
        {"op": "modifier.create", "target": {"id": "subject"}, "id": "smooth",
            "type": "corrective_smooth", "params": {}},
        {"op": "modifier.bind", "target": {"id": "subject"}, "modifier_id": "smooth"},
    ]);
    applied(&apply(&scene, &operations)?)?;
    let doc = load_doc(&scene)?;
    let node = doc
        .nodes
        .get(&Id::new("subject")?)
        .ok_or("subject node is missing")?;
    let binding = node.modifiers[0]
        .binding_data
        .clone()
        .ok_or("corrective smooth bind data is missing")?;
    if binding["format"] != "blender_corrective_bind_v1" {
        return Err(format!("unexpected corrective bind format: {}", binding["format"]).into());
    }
    let coords = binding["coords"]
        .as_array()
        .ok_or("corrective bind coords are missing")?;
    if coords.len() != 25
        || coords[0] != json!([-1.0, -1.0, 0.0])
        || coords[24] != json!([1.0, 1.0, 0.0])
    {
        return Err(format!("unexpected corrective rest coordinates: {coords:?}").into());
    }
    Ok(())
}

#[test]
fn corrective_smooth_unbind_clears_bind_data() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("corrective_unbind");
    init(&scene)?;
    let operations = json!([
        {"op": "node.create", "id": "subject", "kind": "grid",
            "params": {"size": 2.0, "x_subdivisions": 4, "y_subdivisions": 4}},
        {"op": "modifier.create", "target": {"id": "subject"}, "id": "smooth",
            "type": "corrective_smooth", "params": {}},
        {"op": "modifier.bind", "target": {"id": "subject"}, "modifier_id": "smooth"},
        {"op": "modifier.unbind", "target": {"id": "subject"}, "modifier_id": "smooth"},
    ]);
    applied(&apply(&scene, &operations)?)?;
    let doc = load_doc(&scene)?;
    let node = doc
        .nodes
        .get(&Id::new("subject")?)
        .ok_or("subject node is missing")?;
    if node.modifiers[0].binding_data.is_some() {
        return Err("unbind must clear corrective smooth bind data".into());
    }
    Ok(())
}
