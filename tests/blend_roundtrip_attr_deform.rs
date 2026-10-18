#![recursion_limit = "512"]
#![expect(
    clippy::unwrap_used,
    reason = "Blender adapter fixtures use fixed valid paths and values"
)]

use std::{collections::HashMap, error::Error, fs, path::Path, process::Command};

use glam::DVec3;
use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/blender_script_guarded.rs"]
mod blender_script_guarded;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use blender_script_guarded::run_blender_script;
use process::run_guarded;

fn pot_json(arguments: &[&str]) -> Result<Value, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(arguments).arg("--json");
    let output = run_guarded(command)?;
    assert!(
        output.status.success(),
        "pot command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    serde_json::from_slice(&output.stdout).map_err(|error| {
        format!("pot command {arguments:?} returned malformed JSON: {error}").into()
    })
}

fn as_points(values: &[Value]) -> Vec<DVec3> {
    values
        .iter()
        .map(|point| {
            DVec3::new(
                point[0].as_f64().unwrap(),
                point[1].as_f64().unwrap(),
                point[2].as_f64().unwrap(),
            )
        })
        .collect()
}

fn assert_nearest_vertices(
    actual: &[Value],
    expected: &[Value],
    context: &str,
    tolerance: f64,
) -> f64 {
    let actual = as_points(actual);
    let mut unmatched = as_points(expected);
    let bounds = |points: &[DVec3]| {
        points.iter().fold(
            (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
            |(minimum, maximum), point| (minimum.min(*point), maximum.max(*point)),
        )
    };
    let (actual_minimum, actual_maximum) = bounds(&actual);
    let (expected_minimum, expected_maximum) = bounds(&unmatched);
    let mut maximum_error = 0.0_f64;
    let mut worst_actual = DVec3::ZERO;
    let mut worst_expected = DVec3::ZERO;
    for point in actual {
        let (nearest_index, nearest_distance) = unmatched
            .iter()
            .enumerate()
            .map(|(index, candidate)| (index, point.distance(*candidate)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap();
        let nearest = unmatched.swap_remove(nearest_index);
        if nearest_distance > maximum_error {
            maximum_error = nearest_distance;
            worst_actual = point;
            worst_expected = nearest;
        }
    }
    assert!(
        maximum_error <= tolerance,
        "{context}: nearest-matched vertex error {maximum_error} exceeds {tolerance}; \
         actual {worst_actual:?} in {actual_minimum:?}..{actual_maximum:?}, \
         nearest expected {worst_expected:?} in {expected_minimum:?}..{expected_maximum:?}"
    );
    eprintln!("{context}: max vertex error={maximum_error:.9e}");
    maximum_error
}

fn assert_vertex_motion(before: &[Value], after: &[Value], context: &str, minimum: f64) {
    let before = as_points(before);
    let after = as_points(after);
    assert_eq!(before.len(), after.len(), "{context}: vertex count");
    let maximum_motion = before
        .iter()
        .zip(&after)
        .map(|(before, after)| before.distance(*after))
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_motion > minimum,
        "{context}: maximum vertex motion {maximum_motion} did not exceed {minimum}"
    );
}

const REMESH_OCEAN_FIXTURE: &str = r"
import bpy, json, math, os, sys
root = sys.argv[-1]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
positions = []
polygons = []
def append_sphere(segments, rings, radius, center, indent):
    start = len(positions)
    local = [(0.0, 0.0, radius)]
    for ring in range(1, rings):
        latitude = math.pi * ring / rings
        radial = radius * math.sin(latitude)
        z = radius * math.cos(latitude)
        for segment in range(segments):
            longitude = 2.0 * math.pi * segment / segments
            local.append((radial * math.cos(longitude), radial * math.sin(longitude), z))
    bottom = len(local)
    local.append((0.0, 0.0, -radius))
    def ring_id(ring, segment):
        return start + 1 + ring * segments + segment % segments
    for segment in range(segments):
        polygons.append((start, ring_id(0, segment), ring_id(0, segment+1)))
    for ring in range(rings-2):
        for segment in range(segments):
            polygons.append((ring_id(ring, segment), ring_id(ring+1, segment),
                             ring_id(ring+1, segment+1), ring_id(ring, segment+1)))
    for segment in range(segments):
        polygons.append((ring_id(rings-2, segment), start+bottom,
                         ring_id(rings-2, segment+1)))
    for x, y, z in local:
        if indent and x > 1.0 and abs(y) < 0.65 and abs(z) < 0.65:
            x, y, z = 0.4*x, 0.4*y, 0.4*z
        positions.append((x+center[0], y+center[1], z+center[2]))
append_sphere(16, 12, 1.5, (0.0, 0.0, 0.0), True)
append_sphere(12, 8, 0.4, (2.6, 0.1, 0.15), False)
mesh = bpy.data.meshes.new('RemeshInput')
mesh.from_pydata(positions, [], polygons)
mesh.update()
source = bpy.data.objects.new('RemeshSource', mesh)
bpy.context.scene.collection.objects.link(source)
def snapshot(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    result = evaluated.to_mesh()
    try:
        return {'positions':[list(vertex.co) for vertex in result.vertices],
                'polygons':[list(poly.vertices) for poly in result.polygons],
                'faces':len(result.polygons)}
    finally:
        evaluated.to_mesh_clear()
results = {}
for mode in ('VOXEL','BLOCKS','SMOOTH','SHARP'):
    obj = bpy.data.objects.new('OwnerRemesh'+mode, mesh.copy())
    bpy.context.scene.collection.objects.link(obj)
    modifier = obj.modifiers.new('Remesh', 'REMESH')
    modifier.mode = mode
    modifier.use_remove_disconnected = mode != 'VOXEL'
    modifier.threshold = 0.15
    if mode == 'VOXEL':
        modifier.voxel_size = 0.3
        modifier.adaptivity = 0.0
    else:
        modifier.octree_depth = 4
        modifier.scale = 0.9
        modifier.sharpness = 1.0
    results['Remesh'+mode] = snapshot(obj)
# Generated Ocean geometry exercises the spectral deformation and emitted topology.
bpy.ops.mesh.primitive_grid_add(x_subdivisions=5, y_subdivisions=5, size=4.0, location=(0,0,0))
ocean_owner = bpy.context.object
ocean_owner.name = 'OwnerOcean'
ocean = ocean_owner.modifiers.new('Ocean', 'OCEAN')
ocean.geometry_mode = 'GENERATE'
ocean.resolution = 5
ocean.viewport_resolution = 5
ocean.spatial_size = 16
ocean.wave_scale = 0.8
ocean.wave_scale_min = 0.15
ocean.choppiness = 1.25
ocean.wind_velocity = 18.0
ocean.random_seed = 17
ocean.wave_alignment = 0.35
ocean.wave_direction = 0.6
ocean.time = 2.25
ocean.use_foam = True
ocean.foam_layer_name = 'Foam'
bpy.context.view_layer.update()
results['Ocean'] = snapshot(ocean_owner)
with open(os.path.join(root,'remesh_ocean_before.json'),'w',encoding='utf-8') as output:
    json.dump(results,output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root,'remesh_ocean_source.blend'))
";

fn match_vertex_positions(
    actual: &[DVec3],
    expected: &[DVec3],
    context: &str,
) -> (Vec<usize>, f64) {
    const TOLERANCE: f64 = 1.0e-5;
    fn assign(
        actual_index: usize,
        candidates: &[Vec<(usize, f64)>],
        expected_to_actual: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for &(expected_index, _) in &candidates[actual_index] {
            if visited[expected_index] {
                continue;
            }
            visited[expected_index] = true;
            if let Some(previous_actual) = expected_to_actual[expected_index] {
                if assign(previous_actual, candidates, expected_to_actual, visited) {
                    expected_to_actual[expected_index] = Some(actual_index);
                    return true;
                }
            } else {
                expected_to_actual[expected_index] = Some(actual_index);
                return true;
            }
        }
        false
    }

    assert_eq!(actual.len(), expected.len(), "{context}: vertex count");
    let candidates = actual
        .iter()
        .map(|actual_position| {
            let mut matches = expected
                .iter()
                .enumerate()
                .filter_map(|(index, expected_position)| {
                    let distance = actual_position.distance(*expected_position);
                    (distance <= TOLERANCE).then_some((index, distance))
                })
                .collect::<Vec<_>>();
            matches.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            matches
        })
        .collect::<Vec<_>>();
    let mut actual_order = (0..actual.len()).collect::<Vec<_>>();
    actual_order.sort_by_key(|index| (candidates[*index].len(), *index));

    let mut expected_to_actual = vec![None; expected.len()];
    for actual_index in actual_order {
        let mut visited = vec![false; expected.len()];
        let (nearest_index, nearest_distance) = expected
            .iter()
            .enumerate()
            .map(|(index, position)| (index, actual[actual_index].distance(*position)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap();
        assert!(
            assign(
                actual_index,
                &candidates,
                &mut expected_to_actual,
                &mut visited
            ),
            "{context}: no bijective vertex correspondence within {TOLERANCE}; \
             vertex {actual_index} has {} candidates, actual={:?}, nearest expected[{nearest_index}]={:?} (distance {nearest_distance})",
            candidates[actual_index].len(),
            actual[actual_index],
            expected[nearest_index]
        );
    }

    let mut actual_to_expected = vec![usize::MAX; actual.len()];
    for (expected_index, actual_index) in expected_to_actual.into_iter().enumerate() {
        let Some(actual_index) = actual_index else {
            panic!("{context}: perfect matching did not cover expected vertex {expected_index}");
        };
        actual_to_expected[actual_index] = expected_index;
    }
    let maximum_error = actual
        .iter()
        .zip(&actual_to_expected)
        .map(|(actual_position, expected_index)| {
            actual_position.distance(expected[*expected_index])
        })
        .fold(0.0, f64::max);
    assert!(
        maximum_error <= TOLERANCE,
        "{context}: maximum matched vertex error {maximum_error} exceeds {TOLERANCE}"
    );
    (actual_to_expected, maximum_error)
}

fn assert_ordered_vertices(
    actual: &[Value],
    expected: &[Value],
    context: &str,
    tolerance: f64,
) -> f64 {
    let actual = as_points(actual);
    let expected = as_points(expected);
    assert_eq!(actual.len(), expected.len(), "{context}: vertex count");
    let (worst_index, maximum_error) = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, actual.distance(*expected)))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    assert!(
        maximum_error <= tolerance,
        "{context}: vertex {worst_index} error {maximum_error} exceeds {tolerance}"
    );
    eprintln!("{context}: max vertex error={maximum_error:.9e}");
    maximum_error
}

fn assert_values(actual: &Value, expected: &Value, context: &str, tolerance: f64) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            assert!(
                (actual - expected).abs() <= tolerance,
                "{context}: expected {expected}, got {actual}"
            );
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{context}: array size");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_values(actual, expected, &format!("{context}[{index}]"), tolerance);
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{context}: object size");
            for (key, expected) in expected {
                let actual = actual
                    .get(key)
                    .unwrap_or_else(|| panic!("{context}: missing object field {key}"));
                assert_values(actual, expected, &format!("{context}.{key}"), tolerance);
            }
        }
        _ => assert_eq!(actual, expected, "{context}"),
    }
}

fn canonical_polygon_cycles(polygons: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut cycles = polygons
        .iter()
        .map(|polygon| {
            let mut best: Option<Vec<usize>> = None;
            for reversed in [false, true] {
                for start in 0..polygon.len() {
                    let candidate = (0..polygon.len())
                        .map(|offset| {
                            let index = if reversed {
                                (start + polygon.len() - offset) % polygon.len()
                            } else {
                                (start + offset) % polygon.len()
                            };
                            polygon[index]
                        })
                        .collect::<Vec<_>>();
                    if best
                        .as_ref()
                        .is_none_or(|current| candidate.as_slice() < current.as_slice())
                    {
                        best = Some(candidate);
                    }
                }
            }
            best.unwrap_or_default()
        })
        .collect::<Vec<_>>();
    cycles.sort();
    cycles
}

fn remesh_polygons(value: &Value) -> Vec<Vec<usize>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|polygon| {
            polygon
                .as_array()
                .unwrap()
                .iter()
                .map(|index| index.as_u64().unwrap() as usize)
                .collect()
        })
        .collect()
}

fn assert_remesh_geometry(mesh: &potter_core::geom::Mesh, expected: &Value, mode: &str) {
    let expected_positions = as_points(expected["positions"].as_array().unwrap());
    let expected_polygons = remesh_polygons(&expected["polygons"]);
    assert_eq!(
        mesh.vertices.len(),
        expected_positions.len(),
        "{mode}: vertex count"
    );
    assert_eq!(
        mesh.faces.len(),
        expected_polygons.len(),
        "{mode}: face count"
    );
    let actual_positions = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let (mesh_to_expected, maximum_vertex_error) =
        match_vertex_positions(&actual_positions, &expected_positions, mode);

    let vertex_ordinals = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let actual_polygons = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|vertex| mesh_to_expected[vertex_ordinals[vertex]])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        canonical_polygon_cycles(&actual_polygons),
        canonical_polygon_cycles(&expected_polygons),
        "{mode}: polygon topology"
    );
    eprintln!(
        "{mode}: exact topology, vertices={}, faces={}, max matched vertex error={maximum_vertex_error:.9e}",
        mesh.vertices.len(),
        mesh.faces.len()
    );
}

fn assert_blender_remesh_geometry(actual: &Value, expected: &Value, mode: &str) {
    let actual_positions = as_points(actual["positions"].as_array().unwrap());
    let expected_positions = as_points(expected["positions"].as_array().unwrap());
    let actual_polygons = remesh_polygons(&actual["polygons"]);
    let expected_polygons = remesh_polygons(&expected["polygons"]);
    assert_eq!(
        actual_positions.len(),
        expected_positions.len(),
        "{mode}: vertex count"
    );
    assert_eq!(
        actual_polygons.len(),
        expected_polygons.len(),
        "{mode}: face count"
    );
    let (actual_to_expected, maximum_vertex_error) =
        match_vertex_positions(&actual_positions, &expected_positions, mode);
    let actual_polygons = actual_polygons
        .iter()
        .map(|polygon| {
            polygon
                .iter()
                .map(|vertex| actual_to_expected[*vertex])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        canonical_polygon_cycles(&actual_polygons),
        canonical_polygon_cycles(&expected_polygons),
        "{mode}: reopened polygon topology"
    );
    eprintln!(
        "{mode}: reopened exact topology, vertices={}, faces={}, max matched vertex error={maximum_vertex_error:.9e}",
        actual_positions.len(),
        actual_polygons.len()
    );
}

const SIMPLE_REOPEN: &str = r"
import bpy, json, sys
path = sys.argv[-1]
names = ['RemeshVOXEL', 'RemeshBLOCKS', 'RemeshSMOOTH', 'RemeshSHARP', 'Ocean']
result = {}
for suffix in names:
    obj = bpy.data.objects['Owner' + suffix]
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        result[suffix] = {
            'positions': [list(vertex.co) for vertex in mesh.vertices],
            'polygons': [list(poly.vertices) for poly in mesh.polygons],
            'faces': len(mesh.polygons),
        }
    finally:
        evaluated.to_mesh_clear()
with open(path, 'w', encoding='utf-8') as output:
    json.dump(result, output)
";

#[test]
fn blender_dual_contour_remesh_modes_and_ocean_round_trip_depsgraph_geometry()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping Blender remesh/ocean parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_remesh_ocean.py",
        REMESH_OCEAN_FIXTURE,
        root,
        &[],
        "Blender fixture failed",
        run_guarded,
    )?;
    let before: Value = serde_json::from_slice(&fs::read(root.join("remesh_ocean_before.json"))?)?;
    let project = root.join("remesh_ocean_project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("remesh_ocean_source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert!(
        imported["result"]["losses"].as_array().unwrap().is_empty(),
        "remesh/ocean import losses: {:?}",
        imported["result"]["losses"]
    );
    let mappings = imported["result"]["id_mappings"].as_object().unwrap();
    let scene: potter_core::model::SceneDoc =
        serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let evaluation_scene = scene.clone();
    for mode in ["VOXEL", "BLOCKS", "SMOOTH", "SHARP"] {
        let name = format!("OwnerRemesh{mode}");
        let id = potter_core::model::Id::new(
            mappings[&format!("Object:{name}")]
                .as_str()
                .unwrap()
                .to_owned(),
        )?;
        let modifier = scene.nodes[&id].modifiers.first().unwrap();
        assert_eq!(
            modifier.modifier_type, "remesh",
            "{name}: imported modifier type"
        );
        assert_eq!(modifier.params["mode"], json!(mode), "{name}: remesh mode");
    }
    let snapshot = potter_core::eval::Snapshot::evaluate(
        &evaluation_scene,
        &potter_core::eval::EvaluationContext::default(),
    )?;
    for mode in ["VOXEL", "BLOCKS", "SMOOTH", "SHARP"] {
        let name = format!("OwnerRemesh{mode}");
        let id = potter_core::model::Id::new(
            mappings[&format!("Object:{name}")]
                .as_str()
                .unwrap()
                .to_owned(),
        )?;
        let mesh = snapshot
            .meshes
            .get(&id)
            .ok_or("Remesh evaluated mesh missing")?;
        assert_remesh_geometry(mesh, &before[&format!("Remesh{mode}")], mode);
    }
    let ocean_id =
        potter_core::model::Id::new(mappings["Object:OwnerOcean"].as_str().unwrap().to_owned())?;
    let ocean_modifier = scene.nodes[&ocean_id].modifiers.first().unwrap();
    assert_eq!(ocean_modifier.modifier_type, "ocean");
    assert_eq!(ocean_modifier.params["geometry_mode"], json!("GENERATE"));
    assert_eq!(ocean_modifier.params["spatial_size"], json!(16));
    assert_eq!(ocean_modifier.params["viewport_resolution"], json!(5));
    assert_eq!(ocean_modifier.params["random_seed"], json!(17));
    assert_eq!(
        ocean_modifier.params["wave_direction"],
        json!(f64::from(0.6_f32))
    );
    let ocean_mesh = snapshot
        .meshes
        .get(&ocean_id)
        .ok_or("Ocean evaluated mesh missing")?;
    assert_eq!(
        ocean_mesh.vertices.len(),
        before["Ocean"]["positions"].as_array().unwrap().len(),
        "Ocean evaluated vertex count"
    );
    let actual_ocean_faces = ocean_mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|vertex| u64::from(*vertex))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_values(
        &json!(actual_ocean_faces),
        &before["Ocean"]["polygons"],
        "Ocean generated topology",
        0.0,
    );
    let actual_ocean_positions = ocean_mesh
        .vertices
        .iter()
        .map(|vertex| json!(vertex.co.to_array()))
        .collect::<Vec<_>>();
    assert_ordered_vertices(
        &actual_ocean_positions,
        before["Ocean"]["positions"].as_array().unwrap(),
        "Ocean depsgraph output",
        1.0e-4,
    );
    let exported = root.join("remesh_ocean_roundtrip.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let after_path = root.join("remesh_ocean_after.json");
    let reopen_path = root.join("reopen_remesh_ocean.py");
    fs::write(&reopen_path, SIMPLE_REOPEN)?;
    let mut command = Command::new(&blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&reopen_path)
        .arg("--")
        .arg(&after_path);
    let output = run_guarded(command)?;
    assert!(
        output.status.success(),
        "Blender remesh/ocean reopen failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let after: Value = serde_json::from_slice(&fs::read(after_path)?)?;
    for mode in ["BLOCKS", "SMOOTH", "SHARP"] {
        let name = format!("Remesh{mode}");
        assert_blender_remesh_geometry(&after[&name], &before[&name], &name);
    }
    assert_eq!(
        after["Ocean"]["polygons"], before["Ocean"]["polygons"],
        "Ocean: reopened topology"
    );
    assert_ordered_vertices(
        after["Ocean"]["positions"].as_array().unwrap(),
        before["Ocean"]["positions"].as_array().unwrap(),
        "Ocean reopened depsgraph",
        1.0e-5,
    );
    Ok(())
}

const BOUND_DEFORM_FIXTURE: &str = r"
import bpy, json, os, sys
root = sys.argv[-1]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
def make_grid(name):
    points = [((x-2)*0.5, (y-2)*0.5, 0.05*x*y) for y in range(5) for x in range(5)]
    faces = []
    for y in range(4):
        for x in range(4):
            index = y*5+x
            faces.append((index,index+1,index+6,index+5))
    mesh = bpy.data.meshes.new(name+'Mesh')
    mesh.from_pydata(points,[],faces)
    mesh.update()
    return mesh
def make_cube(name):
    points = [(-2,-2,-2),(2,-2,-2),(2,2,-2),(-2,2,-2),
              (-2,-2,2),(2,-2,2),(2,2,2),(-2,2,2)]
    faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    mesh = bpy.data.meshes.new(name+'Mesh')
    mesh.from_pydata(points,[],faces)
    mesh.update()
    return mesh
def make_surface(name):
    mesh = bpy.data.meshes.new(name+'Mesh')
    mesh.from_pydata([(-5,-4,0),(5,-4,0),(0,6,0)],[],[(0,1,2)])
    mesh.update()
    return mesh
def bind(obj, modifier, operator):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    with bpy.context.temp_override(object=obj, active_object=obj,
                                   selected_objects=[obj], selected_editable_objects=[obj]):
        result = operator(modifier=modifier.name)
    if 'FINISHED' not in result:
        raise RuntimeError('bind failed for '+obj.name+': '+repr(result))
def shape_key_at_bind(obj, name, indices, z_offset):
    if obj.data.shape_keys is None:
        obj.shape_key_add(name='Basis')
    key = obj.shape_key_add(name=name)
    for index in indices:
        key.data[index].co.z += z_offset
    key.value = 1.0
    return key
def owner(name):
    obj = bpy.data.objects.new(name, make_grid(name))
    bpy.context.scene.collection.objects.link(obj)
    return obj
def target(name):
    obj = bpy.data.objects.new(name, make_cube(name))
    bpy.context.scene.collection.objects.link(obj)
    return obj
surface_target = bpy.data.objects.new('SurfaceTarget', make_surface('SurfaceTarget'))
bpy.context.scene.collection.objects.link(surface_target)
surface_bind_shape = shape_key_at_bind(surface_target, 'BindShape', [2], 1.25)
surface_target.location.x = 0.35
obj = owner('OwnerSurfaceDeform')
modifier = obj.modifiers.new('SurfaceDeform','SURFACE_DEFORM')
modifier.target = surface_target
modifier.falloff = 4.0
modifier.strength = 0.85
bind(obj, modifier, bpy.ops.object.surfacedeform_bind)
surface_bind_shape.value = 0.0

mesh_target = target('MeshTarget')
mesh_bind_shape = shape_key_at_bind(mesh_target, 'BindShape', [4,5,6,7], 0.7)
mesh_target.location = (0.35,-0.2,0.15)
obj = owner('OwnerMeshDeform')
modifier = obj.modifiers.new('MeshDeform','MESH_DEFORM')
modifier.object = mesh_target
modifier.precision = 2
modifier.use_dynamic_bind = False
bind(obj, modifier, bpy.ops.object.meshdeform_bind)
mesh_bind_shape.value = 0.0

obj = owner('OwnerLaplacianDeform')
laplacian_bind_shape = shape_key_at_bind(obj, 'BindShape', [20,24], 0.6)
anchors = obj.vertex_groups.new(name='Anchors')
anchors.add([0,4,20,24],1.0,'REPLACE')
modifier = obj.modifiers.new('LaplacianDeform','LAPLACIANDEFORM')
modifier.vertex_group = anchors.name
modifier.iterations = 3
bind(obj, modifier, bpy.ops.object.laplaciandeform_bind)
laplacian_bind_shape.value = 0.0
bpy.context.view_layer.update()
def summarize(name):
    obj = bpy.data.objects[name]
    modifier = obj.modifiers[0]
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return {'type':modifier.type, 'is_bound':bool(getattr(modifier,'is_bound',getattr(modifier,'is_bind',False))),
                'positions':[list(vertex.co) for vertex in mesh.vertices],
                'polygons':[list(poly.vertices) for poly in mesh.polygons]}
    finally:
        evaluated.to_mesh_clear()
names = ['OwnerSurfaceDeform','OwnerMeshDeform','OwnerLaplacianDeform']
before = {name:summarize(name) for name in names}
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root,'bound_deform_source.blend'))
surface_target.data.shape_keys.key_blocks['Basis'].data[2].co.z += 0.45
# Perturb the bound deformers after the source fixture is saved. The changed
# depsgraph output proves the saved binding responds to later cage/anchor edits.
for vertex in surface_target.data.shape_keys.key_blocks['Basis'].data:
    if vertex.co.y > 0.0:
        vertex.co.z += 0.45
        vertex.co.x += 0.12
for vertex in mesh_target.data.shape_keys.key_blocks['Basis'].data:
    if vertex.co.y > 0.0:
        vertex.co.y += 0.35
        vertex.co.z += 0.18
laplacian_obj = bpy.data.objects['OwnerLaplacianDeform']
laplacian_basis = laplacian_obj.data.shape_keys.key_blocks['Basis'].data
for index in (0,4,20,24):
    laplacian_basis[index].co.z += 0.45
bpy.context.view_layer.update()
moved = {name:summarize(name) for name in names}
with open(os.path.join(root,'bound_deform_before.json'),'w',encoding='utf-8') as output:
    json.dump({'before':before,'moved':moved},output)
";

const BOUND_DEFORM_REOPEN: &str = r"
import bpy, json, sys
path = sys.argv[-1]
names = ['OwnerSurfaceDeform','OwnerMeshDeform','OwnerLaplacianDeform']
result = {}
for name in names:
    obj = bpy.data.objects[name]
    modifier = obj.modifiers[0]
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        result[name] = {'type':modifier.type, 'is_bound':bool(getattr(modifier,'is_bound',getattr(modifier,'is_bind',False))),
                        'positions':[list(vertex.co) for vertex in mesh.vertices],
                        'polygons':[list(poly.vertices) for poly in mesh.polygons]}
    finally:
        evaluated.to_mesh_clear()
with open(path,'w',encoding='utf-8') as output:
    json.dump(result,output)
";

fn move_shape_key_basis(
    scene: &mut potter_core::model::SceneDoc,
    mappings: &serde_json::Map<String, Value>,
    object_name: &str,
    vertex_index: u32,
    offset: [f64; 3],
) -> Result<(), Box<dyn Error>> {
    let object_id = potter_core::model::Id::new(
        mappings[&format!("Object:{object_name}")]
            .as_str()
            .ok_or("object mapping did not contain a string ID")?
            .to_owned(),
    )?;
    let data_id = scene.nodes[&object_id]
        .data
        .as_ref()
        .ok_or("object has no mesh data")?
        .clone();
    let data = scene
        .data_blocks
        .get_mut(&data_id)
        .ok_or("mesh data block missing")?;
    let vertex_index = usize::try_from(vertex_index)?;
    let vertex_id = data
        .mesh
        .as_ref()
        .and_then(|mesh| mesh.vertices.get(vertex_index))
        .map(|vertex| vertex.id)
        .ok_or("shape key basis vertex missing")?;
    let shape_keys = data.shape_keys.as_mut().ok_or("mesh has no shape keys")?;
    if let Some(basis) = shape_keys.basis.get_mut(&vertex_id) {
        for (coordinate, delta) in basis.iter_mut().zip(offset) {
            *coordinate += delta;
        }
    } else {
        let vertex = data
            .mesh
            .as_mut()
            .and_then(|mesh| mesh.vertices.get_mut(vertex_index))
            .ok_or("shape key basis vertex missing")?;
        for (axis, delta) in offset.into_iter().enumerate() {
            vertex.co[axis] += delta;
        }
    }
    Ok(())
}

fn moved_bound_scene(
    scene: &potter_core::model::SceneDoc,
    mappings: &serde_json::Map<String, Value>,
) -> Result<potter_core::model::SceneDoc, Box<dyn Error>> {
    let mut moved = scene.clone();
    move_shape_key_basis(&mut moved, mappings, "SurfaceTarget", 2, [0.0, 0.0, 0.45])?;
    move_shape_key_basis(&mut moved, mappings, "SurfaceTarget", 2, [0.12, 0.0, 0.45])?;
    for index in [2, 3, 6, 7] {
        move_shape_key_basis(&mut moved, mappings, "MeshTarget", index, [0.0, 0.35, 0.18])?;
    }
    for index in [0, 4, 20, 24] {
        move_shape_key_basis(
            &mut moved,
            mappings,
            "OwnerLaplacianDeform",
            index,
            [0.0, 0.0, 0.45],
        )?;
    }
    Ok(moved)
}

#[test]
fn blender_native_bound_deform_payloads_follow_deformed_bind_targets_and_anchors()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping Blender bound-deform parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_bound_deform.py",
        BOUND_DEFORM_FIXTURE,
        root,
        &[],
        "Blender fixture failed",
        run_guarded,
    )?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("bound_deform_before.json"))?)?;
    let project = root.join("bound_deform_project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("bound_deform_source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert!(
        imported["result"]["losses"].as_array().unwrap().is_empty(),
        "bound deform import losses: {:?}",
        imported["result"]["losses"]
    );
    let mappings = imported["result"]["id_mappings"].as_object().unwrap();
    let scene_bytes = fs::read(project.join("scene.json"))?;
    assert!(
        !String::from_utf8_lossy(&scene_bytes).contains("<bpy_"),
        "bound-deform imported project contains an unnormalized Blender RNA repr",
    );
    let scene: potter_core::model::SceneDoc = serde_json::from_slice(&scene_bytes)?;
    let baseline = potter_core::eval::Snapshot::evaluate(
        &scene,
        &potter_core::eval::EvaluationContext::default(),
    )?;
    for (name, target_name, expected_type) in [
        ("OwnerSurfaceDeform", "SurfaceTarget", "surface_deform"),
        ("OwnerMeshDeform", "MeshTarget", "mesh_deform"),
        (
            "OwnerLaplacianDeform",
            "OwnerLaplacianDeform",
            "laplacian_deform",
        ),
    ] {
        let owner_id = potter_core::model::Id::new(
            mappings[&format!("Object:{name}")]
                .as_str()
                .unwrap()
                .to_owned(),
        )?;
        let modifier = scene.nodes[&owner_id]
            .modifiers
            .first()
            .ok_or("bound modifier missing")?;
        assert_eq!(
            modifier.modifier_type, expected_type,
            "{name}: imported modifier type"
        );
        let modifier_state = serde_json::to_value(modifier)?;
        assert!(
            modifier_state.get("binding_data").is_some(),
            "{name}: import did not restore bound state"
        );
        assert_eq!(
            modifier_state["binding_data"]["format"], "blender_native_bind_v1",
            "{name}: imported native bind representation"
        );
        if name != "OwnerLaplacianDeform" {
            let expected_target = mappings[&format!("Object:{target_name}")].as_str().unwrap();
            let field = if expected_type == "surface_deform" {
                "target"
            } else {
                "object"
            };
            assert_eq!(
                modifier.params[field],
                json!(expected_target),
                "{name}: bind target"
            );
        }
        let actual = baseline
            .meshes
            .get(&owner_id)
            .ok_or("bound evaluated mesh missing")?;
        assert_nearest_vertices(
            &actual
                .vertices
                .iter()
                .map(|vertex| json!(vertex.co.to_array()))
                .collect::<Vec<_>>(),
            expected["before"][name]["positions"].as_array().unwrap(),
            &format!("{name} imported bind output"),
            if name == "OwnerLaplacianDeform" {
                1.0e-5
            } else {
                1.0e-4
            },
        );
    }
    assert_vertex_motion(
        expected["before"]["OwnerSurfaceDeform"]["positions"]
            .as_array()
            .unwrap(),
        expected["moved"]["OwnerSurfaceDeform"]["positions"]
            .as_array()
            .unwrap(),
        "Surface Deform moved target stimulus",
        1.0e-3,
    );
    assert_vertex_motion(
        expected["before"]["OwnerMeshDeform"]["positions"]
            .as_array()
            .unwrap(),
        expected["moved"]["OwnerMeshDeform"]["positions"]
            .as_array()
            .unwrap(),
        "Mesh Deform moved cage stimulus",
        1.0e-3,
    );
    assert_vertex_motion(
        expected["before"]["OwnerLaplacianDeform"]["positions"]
            .as_array()
            .unwrap(),
        expected["moved"]["OwnerLaplacianDeform"]["positions"]
            .as_array()
            .unwrap(),
        "Laplacian Deform moved anchor stimulus",
        1.0e-3,
    );
    let moved = moved_bound_scene(&scene, mappings)?;
    let moved_snapshot = potter_core::eval::Snapshot::evaluate(
        &moved,
        &potter_core::eval::EvaluationContext::default(),
    )?;
    for name in [
        "OwnerSurfaceDeform",
        "OwnerMeshDeform",
        "OwnerLaplacianDeform",
    ] {
        let id = potter_core::model::Id::new(
            mappings[&format!("Object:{name}")]
                .as_str()
                .unwrap()
                .to_owned(),
        )?;
        let actual = moved_snapshot
            .meshes
            .get(&id)
            .ok_or("moved bound evaluated mesh missing")?;
        assert_nearest_vertices(
            &actual
                .vertices
                .iter()
                .map(|vertex| json!(vertex.co.to_array()))
                .collect::<Vec<_>>(),
            expected["moved"][name]["positions"].as_array().unwrap(),
            &format!("{name} after target/cage/anchor movement"),
            if name == "OwnerLaplacianDeform" {
                1.0e-5
            } else {
                1.0e-4
            },
        );
    }
    fs::write(
        project.join("scene.json"),
        serde_json::to_vec_pretty(&moved)?,
    )?;
    let exported = root.join("bound_deform_roundtrip.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let after_path = root.join("bound_deform_after.json");
    let reopen_path = root.join("reopen_bound_deform.py");
    fs::write(&reopen_path, BOUND_DEFORM_REOPEN)?;
    let mut command = Command::new(&blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&reopen_path)
        .arg("--")
        .arg(&after_path);
    let output = run_guarded(command)?;
    assert!(
        output.status.success(),
        "Blender bound-deform reopen failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let after: Value = serde_json::from_slice(&fs::read(after_path)?)?;
    for name in [
        "OwnerLaplacianDeform",
        "OwnerSurfaceDeform",
        "OwnerMeshDeform",
    ] {
        assert_eq!(
            after[name]["is_bound"], true,
            "{name}: exported binding state"
        );
        assert_nearest_vertices(
            after[name]["positions"].as_array().unwrap(),
            expected["moved"][name]["positions"].as_array().unwrap(),
            &format!("{name} reopened after movement"),
            if name == "OwnerLaplacianDeform" {
                1.0e-5
            } else {
                1.0e-4
            },
        );
    }
    Ok(())
}

const CORRECTIVE_BIND_FIXTURE: &str = r"
import bpy, json, os, sys
root = sys.argv[-1]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
def make_grid(name):
    points = [((x-2)*0.5, (y-2)*0.5, 0.05*x*y) for y in range(5) for x in range(5)]
    faces = []
    for y in range(4):
        for x in range(4):
            index = y*5+x
            faces.append((index,index+1,index+6,index+5))
    mesh = bpy.data.meshes.new(name+'Mesh')
    mesh.from_pydata(points,[],faces)
    mesh.update()
    return mesh
obj = bpy.data.objects.new('OwnerCorrectiveSmooth', make_grid('OwnerCorrectiveSmooth'))
bpy.context.scene.collection.objects.link(obj)
subdivision = obj.modifiers.new('Subdivision','SUBSURF')
subdivision.levels = 1
subdivision.render_levels = 1
modifier = obj.modifiers.new('CorrectiveSmooth','CORRECTIVE_SMOOTH')
modifier.rest_source = 'BIND'
modifier.factor = 0.5
modifier.iterations = 5
modifier.scale = 1.0
modifier.smooth_type = 'SIMPLE'
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
# Blender binds the modifier input, which is the Subdivision output. Capture that input with the
# Corrective Smooth hidden, before the bind.
modifier.show_viewport = False
bpy.context.view_layer.update()
evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
input_mesh = evaluated.to_mesh()
try:
    bind_coords = [list(vertex.co) for vertex in input_mesh.vertices]
finally:
    evaluated.to_mesh_clear()
if len(bind_coords) == len(obj.data.vertices):
    raise RuntimeError('subdivision prefix did not change the corrective smooth bind input')
modifier.show_viewport = True
bpy.context.view_layer.update()
with bpy.context.temp_override(object=obj, active_object=obj,
                               selected_objects=[obj], selected_editable_objects=[obj]):
    result = bpy.ops.object.correctivesmooth_bind(modifier=modifier.name)
if 'FINISHED' not in result:
    raise RuntimeError('corrective smooth bind failed: '+repr(result))
bpy.context.view_layer.update()
if not modifier.is_bind:
    raise RuntimeError('corrective smooth modifier is not bound after the bind operator')
# Asymmetric base-vertex moves after the bind; the bound rest positions keep the bind-time coordinates.
moves = {0:(0.0,0.0,0.12), 6:(-0.04,0.08,-0.06), 12:(0.0,0.0,0.25), 18:(0.1,-0.03,0.0)}
for index, (dx, dy, dz) in moves.items():
    vertex = obj.data.vertices[index]
    vertex.co.x += dx
    vertex.co.y += dy
    vertex.co.z += dz
obj.data.update()
bpy.context.view_layer.update()
evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
mesh = evaluated.to_mesh()
try:
    moved = [list(vertex.co) for vertex in mesh.vertices]
finally:
    evaluated.to_mesh_clear()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root,'corrective_bind_source.blend'))
with open(os.path.join(root,'corrective_bind_expected.json'),'w',encoding='utf-8') as output:
    json.dump({'bind_coords':bind_coords,'moved':moved},output)
";

const CORRECTIVE_BIND_REOPEN: &str = r"
import bpy, json, sys
path = sys.argv[-1]
obj = bpy.data.objects['OwnerCorrectiveSmooth']
modifier = next(item for item in obj.modifiers if item.type == 'CORRECTIVE_SMOOTH')
evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
mesh = evaluated.to_mesh()
try:
    result = {'type':modifier.type, 'is_bind':bool(modifier.is_bind),
              'positions':[list(vertex.co) for vertex in mesh.vertices]}
finally:
    evaluated.to_mesh_clear()
with open(path,'w',encoding='utf-8') as output:
    json.dump(result,output)
";

fn remove_original_blend_blobs(
    scene: &potter_core::model::SceneDoc,
    project: &Path,
) -> Result<usize, Box<dyn Error>> {
    let mut removed = 0;
    for entry in scene.compatibility.values() {
        let blobs = entry
            .get("blobs")
            .and_then(Value::as_array)
            .into_iter()
            .flatten();
        for blob in blobs {
            if blob.get("name").and_then(Value::as_str) != Some("original.blend") {
                continue;
            }
            let path = blob
                .get("path")
                .and_then(Value::as_str)
                .ok_or("original.blend compatibility blob has no path")?;
            fs::remove_file(project.join(path))?;
            removed += 1;
        }
    }
    Ok(removed)
}

fn export_corrective_bind(
    blender: &Path,
    project: &Path,
    root: &Path,
    label: &str,
    moved: &[Value],
) -> Result<(), Box<dyn Error>> {
    let exported = root.join(format!("corrective_bind_{label}.blend"));
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let after_path = root.join(format!("corrective_bind_{label}_after.json"));
    let reopen_path = root.join("reopen_corrective_bind.py");
    fs::write(&reopen_path, CORRECTIVE_BIND_REOPEN)?;
    let mut command = Command::new(blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&reopen_path)
        .arg("--")
        .arg(&after_path);
    let output = run_guarded(command)?;
    assert!(
        output.status.success(),
        "Blender corrective smooth reopen ({label}) failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let after: Value = serde_json::from_slice(&fs::read(after_path)?)?;
    assert_eq!(
        after["type"], "CORRECTIVE_SMOOTH",
        "{label}: reopened modifier type"
    );
    assert_eq!(
        after["is_bind"], true,
        "{label}: exported corrective smooth binding state"
    );
    assert_ordered_vertices(
        after["positions"].as_array().unwrap(),
        moved,
        &format!("corrective smooth {label} reopened evaluation"),
        1.0e-4,
    );
    Ok(())
}

#[test]
fn blender_corrective_smooth_bind_round_trips_bind_coords_and_evaluation()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping corrective smooth bind parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_corrective_bind.py",
        CORRECTIVE_BIND_FIXTURE,
        root,
        &[],
        "Blender fixture failed",
        run_guarded,
    )?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("corrective_bind_expected.json"))?)?;
    assert_eq!(
        expected["bind_coords"].as_array().unwrap().len(),
        81,
        "Blender Subdivision-prefix corrective smooth bind input vertex count"
    );
    let moved = expected["moved"].as_array().unwrap();
    let project = root.join("corrective_bind_project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("corrective_bind_source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert!(
        imported["result"]["losses"].as_array().unwrap().is_empty(),
        "corrective smooth bind import losses: {:?}",
        imported["result"]["losses"]
    );
    let mappings = imported["result"]["id_mappings"].as_object().unwrap();
    let scene_bytes = fs::read(project.join("scene.json"))?;
    assert!(
        !String::from_utf8_lossy(&scene_bytes).contains("<bpy_"),
        "corrective smooth imported project contains an unnormalized Blender RNA repr",
    );
    let scene: potter_core::model::SceneDoc = serde_json::from_slice(&scene_bytes)?;
    let owner_id = potter_core::model::Id::new(
        mappings["Object:OwnerCorrectiveSmooth"]
            .as_str()
            .unwrap()
            .to_owned(),
    )?;
    let modifier = scene.nodes[&owner_id]
        .modifiers
        .iter()
        .find(|modifier| modifier.modifier_type == "corrective_smooth")
        .ok_or("corrective smooth modifier missing")?;
    assert_eq!(modifier.modifier_type, "corrective_smooth");
    assert_eq!(modifier.params["rest_source"], json!("BIND"));
    assert!(
        modifier.params.get("is_bind").is_none(),
        "bind state must come from binding_data, not a parameter"
    );
    let modifier_state = serde_json::to_value(modifier)?;
    assert_eq!(
        modifier_state["binding_data"]["format"], "blender_corrective_bind_v1",
        "imported corrective smooth bind representation"
    );
    assert_eq!(
        modifier_state["binding_data"]["coords"]
            .as_array()
            .map(Vec::len),
        Some(81),
        "imported corrective smooth bind coordinate count"
    );
    assert_values(
        &modifier_state["binding_data"]["coords"],
        &expected["bind_coords"],
        "corrective smooth imported bind coords",
        1.0e-6,
    );
    let baseline = potter_core::eval::Snapshot::evaluate(
        &scene,
        &potter_core::eval::EvaluationContext::default(),
    )?;
    let actual = baseline
        .meshes
        .get(&owner_id)
        .ok_or("corrective smooth evaluated mesh missing")?;
    let actual_positions = actual
        .vertices
        .iter()
        .map(|vertex| json!(vertex.co.to_array()))
        .collect::<Vec<_>>();
    assert_vertex_motion(
        expected["bind_coords"].as_array().unwrap(),
        moved,
        "corrective smooth post-bind motion",
        1.0e-3,
    );
    assert_ordered_vertices(
        &actual_positions,
        moved,
        "corrective smooth imported evaluation",
        1.0e-4,
    );
    // The compatibility original.blend is present: the source copy keeps Blender's bind state.
    export_corrective_bind(&blender, &project, root, "source", moved)?;
    // Without the compatibility original.blend the bind must be rebuilt from the serialized coords.
    let removed = remove_original_blend_blobs(&scene, &project)?;
    assert!(
        removed > 0,
        "corrective smooth project has no compatibility original.blend to remove"
    );
    export_corrective_bind(&blender, &project, root, "fallback", moved)?;
    Ok(())
}

const CACHE_FIXTURE: &str = r"
import bpy, json, math, os, struct, sys
root = sys.argv[-1]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
def grid(name):
    points = [(float(x),float(y),0.08*math.sin(x+y)) for y in range(3) for x in range(3)]
    faces = [(0,1,4,3),(1,2,5,4),(3,4,7,6),(4,5,8,7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points,[],faces)
    mesh.update()
    return mesh, points
def make_owner(name):
    mesh, points = grid(name+'Mesh')
    obj = bpy.data.objects.new(name,mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj, points
base_mesh, base_points = grid('CacheBase')
cache_points = [(x,y,z+0.15*x+0.35) for x,y,z in base_points]
def write_pc2(path):
    with open(path,'wb') as stream:
        stream.write(struct.pack('<12sii ffi',b'POINTCACHE2',1,len(base_points),1.0,1.0,2))
        for sample in (base_points,cache_points):
            for point in sample: stream.write(struct.pack('<3f',*point))
def write_mdd(path):
    with open(path,'wb') as stream:
        stream.write(struct.pack('>ii',2,len(base_points)))
        stream.write(struct.pack('>2f',0.0,1.0))
        for sample in (base_points,cache_points):
            for point in sample: stream.write(struct.pack('>3f',*point))
pc2_path = os.path.join(root,'cache_points.pc2')
mdd_path = os.path.join(root,'cache_points.mdd')
write_pc2(pc2_path)
write_mdd(mdd_path)
for name, cache_format, path in [
    ('OwnerMeshCachePC2','PC2',pc2_path),
    ('OwnerMeshCacheMDD','MDD',mdd_path),
]:
    obj, points = make_owner(name)
    modifier = obj.modifiers.new('MeshCache','MESH_CACHE')
    modifier.cache_format = cache_format
    modifier.filepath = path
    modifier.play_mode = 'CUSTOM'
    modifier.time_mode = 'FRAME' if cache_format == 'PC2' else 'TIME'
    modifier.interpolation = 'LINEAR'
    modifier.deform_mode = 'OVERWRITE'
    modifier.frame_start = 1.0
    modifier.frame_scale = 1.0
    modifier.factor = 1.0
    modifier.eval_frame = 2.0
    modifier.eval_time = 0.75
    if hasattr(modifier, 'flip_axis'):
        modifier.flip_axis = (True, False, True)

# An animated mesh is written to a real Alembic file and consumed by the
# Blender Mesh Sequence Cache modifier on an independent matching mesh.
source_mesh, source_points = grid('AlembicSourceMesh')
source = bpy.data.objects.new('AlembicSource',source_mesh)
bpy.context.scene.collection.objects.link(source)
basis = source.shape_key_add(name='Basis')
key = source.shape_key_add(name='CacheTarget')
for vertex in key.data:
    vertex.co.z += 0.55
key.value = 0.0
key.keyframe_insert(data_path='value',frame=1)
key.value = 1.0
key.keyframe_insert(data_path='value',frame=2)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 2
abc_path = os.path.join(root,'mesh_sequence.abc')
bpy.ops.object.select_all(action='DESELECT')
source.select_set(True)
bpy.context.view_layer.objects.active = source
bpy.ops.wm.alembic_export(filepath=abc_path,start=1,end=2,selected=True,flatten=True,
                          uvs=True,normals=True)
bpy.ops.cachefile.open(filepath=abc_path)
cache_file = bpy.data.cache_files.get(os.path.basename(abc_path))
if cache_file is None:
    raise RuntimeError('Alembic cache file was not opened')
sequence_owner, _ = make_owner('OwnerMeshSequenceCache')
modifier = sequence_owner.modifiers.new('SequenceCache','MESH_SEQUENCE_CACHE')
modifier.cache_file = cache_file
modifier.object_path = '/' + source.name + '/' + source.data.name
modifier.read_data = {'VERT','POLY','UV','COLOR'}
modifier.use_vertex_interpolation = True
cache_file.frame_offset = 0.35
cache_file.scale = 1.25
cache_file.override_frame = False
scene.frame_set(2)
bpy.context.view_layer.update()
names = ['OwnerMeshCachePC2','OwnerMeshCacheMDD','OwnerMeshSequenceCache']
def summarize(name):
    obj = bpy.data.objects[name]
    modifier = obj.modifiers[0]
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        params = {}
        for key_name in ('cache_format','play_mode','time_mode','interpolation','deform_mode',
                         'factor','frame_start','frame_scale','eval_frame','eval_time',
                         'eval_factor','forward_axis','up_axis','flip_axis','object_path',
                         'read_data','velocity_scale','use_vertex_interpolation'):
            if hasattr(modifier,key_name):
                value = getattr(modifier,key_name)
                if hasattr(value,'to_list'): value = value.to_list()
                elif key_name == 'flip_axis': value = [bool(value[index]) for index in range(3)]
                elif isinstance(value,set): value = sorted(value)
                elif isinstance(value,tuple): value = list(value)
                elif not isinstance(value,(str,int,float,bool,list)):
                    try: value = list(value)
                    except TypeError: value = str(value)
                params[key_name] = value
        cache_settings = {}
        if modifier.type == 'MESH_CACHE':
            resource_path = os.path.basename(modifier.filepath)
        else:
            resource_path = os.path.basename(modifier.cache_file.filepath)
            cache_settings = {
                'frame_offset': modifier.cache_file.frame_offset,
                'scale': modifier.cache_file.scale,
                'override_frame': modifier.cache_file.override_frame,
            }
        return {'type':modifier.type,'params':params,'resource_path':resource_path,
                'cache_settings':cache_settings,
                'positions':[list(vertex.co) for vertex in mesh.vertices],
                'polygons':[list(poly.vertices) for poly in mesh.polygons]}
    finally:
        evaluated.to_mesh_clear()
before = {name:summarize(name) for name in names}
with open(os.path.join(root,'cache_before.json'),'w',encoding='utf-8') as output:
    json.dump(before,output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root,'cache_source.blend'))
";

const CACHE_REOPEN: &str = r"
import bpy, json, os, sys
path = sys.argv[-1]
names = ['OwnerMeshCachePC2','OwnerMeshCacheMDD','OwnerMeshSequenceCache']
result = {}
for name in names:
    obj = bpy.data.objects[name]
    modifier = obj.modifiers[0]
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        params = {}
        for key in ('cache_format','play_mode','time_mode','interpolation','deform_mode',
                    'factor','frame_start','frame_scale','eval_frame','eval_time',
                    'eval_factor','forward_axis','up_axis','flip_axis','object_path',
                    'read_data','velocity_scale','use_vertex_interpolation'):
            if hasattr(modifier,key):
                value = getattr(modifier,key)
                if hasattr(value,'to_list'): value = value.to_list()
                elif key == 'flip_axis': value = [bool(value[index]) for index in range(3)]
                elif isinstance(value,set): value = sorted(value)
                elif isinstance(value,tuple): value = list(value)
                elif not isinstance(value,(str,int,float,bool,list)):
                    try: value = list(value)
                    except TypeError: value = str(value)
                params[key] = value
        cache_settings = {}
        if modifier.type == 'MESH_CACHE':
            resource_path = os.path.basename(modifier.filepath)
        else:
            resource_path = os.path.basename(modifier.cache_file.filepath)
            cache_settings = {
                'frame_offset': modifier.cache_file.frame_offset,
                'scale': modifier.cache_file.scale,
                'override_frame': modifier.cache_file.override_frame,
            }
        result[name] = {'type':modifier.type,'params':params,'resource_path':resource_path,
                        'cache_settings':cache_settings,
                        'positions':[list(vertex.co) for vertex in mesh.vertices],
                        'polygons':[list(poly.vertices) for poly in mesh.polygons]}
    finally:
        evaluated.to_mesh_clear()
with open(path,'w',encoding='utf-8') as output:
    json.dump(result,output)
";

#[test]
fn blender_mesh_cache_pc2_mdd_and_alembic_sequence_round_trip_resources()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping Blender cache modifier parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_cache_fixture.py",
        CACHE_FIXTURE,
        root,
        &[],
        "Blender fixture failed",
        run_guarded,
    )?;
    let before_bytes = fs::read(root.join("cache_before.json"))?;
    let before: Value =
        serde_json::from_slice(&before_bytes).map_err(|error| -> Box<dyn Error> {
            format!("cache_before.json malformed: {error}").into()
        })?;
    let project = root.join("cache_project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("cache_source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert!(
        imported["result"]["losses"].as_array().unwrap().is_empty(),
        "cache modifier import losses: {:?}",
        imported["result"]["losses"]
    );
    let mappings = imported["result"]["id_mappings"].as_object().unwrap();
    let scene_bytes = fs::read(project.join("scene.json"))?;
    assert!(
        !String::from_utf8_lossy(&scene_bytes).contains("<bpy_"),
        "cache imported project contains an unnormalized Blender RNA repr",
    );
    let scene: potter_core::model::SceneDoc =
        serde_json::from_slice(&scene_bytes).map_err(|error| -> Box<dyn Error> {
            format!("imported cache scene.json malformed: {error}").into()
        })?;
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        &scene,
        &potter_core::eval::EvaluationContext {
            frame: Some(2.0),
            ..Default::default()
        },
        Some(project.as_path()),
        Some(&project.as_path().join(".potter/cache")),
    )?;
    for (name, expected_type) in [
        ("OwnerMeshCachePC2", "mesh_cache"),
        ("OwnerMeshCacheMDD", "mesh_cache"),
        ("OwnerMeshSequenceCache", "mesh_sequence_cache"),
    ] {
        let id = potter_core::model::Id::new(
            mappings[&format!("Object:{name}")]
                .as_str()
                .unwrap()
                .to_owned(),
        )?;
        let modifier = scene.nodes[&id]
            .modifiers
            .first()
            .ok_or("cache modifier missing")?;
        assert_eq!(
            modifier.modifier_type, expected_type,
            "{name}: imported type"
        );
        let resource_id = modifier.params["resource"]
            .as_str()
            .ok_or("cache resource ID missing")?;
        assert!(
            scene
                .resources
                .contains_key(&potter_core::model::Id::new(resource_id.to_owned())?),
            "{name}: registered cache resource missing"
        );
        if expected_type == "mesh_cache" {
            assert_eq!(
                modifier.params["cache_format"], before[name]["params"]["cache_format"],
                "{name}: cache format"
            );
        } else {
            assert_eq!(
                modifier.params["object_path"], before[name]["params"]["object_path"],
                "{name}: Alembic object path"
            );
            assert_eq!(
                modifier.params["use_vertex_interpolation"],
                json!(true),
                "{name}: vertex interpolation"
            );
            let mut read_flags = modifier.params["read_data"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>();
            read_flags.sort_unstable();
            assert_eq!(
                read_flags,
                vec!["COLOR", "POLY", "UV", "VERT"],
                "{name}: Alembic read flags",
            );
            assert_eq!(
                modifier.params["frame_offset"],
                json!(f64::from(0.35_f32)),
                "{name}: Alembic frame offset"
            );
            assert_eq!(
                modifier.params["scale"],
                json!(1.25),
                "{name}: Alembic time scale"
            );
        }
        let actual = snapshot
            .meshes
            .get(&id)
            .ok_or("cache evaluated mesh missing")?;
        assert_nearest_vertices(
            &actual
                .vertices
                .iter()
                .map(|vertex| json!(vertex.co.to_array()))
                .collect::<Vec<_>>(),
            before[name]["positions"].as_array().unwrap(),
            &format!("{name} imported resource evaluation"),
            1.0e-5,
        );
    }
    for name in [
        "OwnerMeshCachePC2",
        "OwnerMeshCacheMDD",
        "OwnerMeshSequenceCache",
    ] {
        let id = potter_core::model::Id::new(
            mappings[&format!("Object:{name}")]
                .as_str()
                .unwrap()
                .to_owned(),
        )?;
        let original = scene.nodes[&id].data.as_ref().unwrap();
        let base = &scene.data_blocks[original].mesh.as_ref().unwrap().vertices;
        let evaluated = snapshot.meshes.get(&id).unwrap();
        assert!(
            evaluated
                .vertices
                .iter()
                .zip(base)
                .any(|(actual, source)| actual.co.distance(source.co) > 0.1),
            "{name}: cache fixture did not materially deform the input mesh"
        );
    }
    let exported = root.join("cache_roundtrip.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let after_path = root.join("cache_after.json");
    let reopen_path = root.join("reopen_cache.py");
    fs::write(&reopen_path, CACHE_REOPEN)?;
    let mut command = Command::new(&blender);
    command
        .args(["--background", "--python-exit-code", "1"])
        .arg(&exported)
        .arg("--python")
        .arg(&reopen_path)
        .arg("--")
        .arg(&after_path);
    let output = run_guarded(command)?;
    assert!(
        output.status.success(),
        "Blender cache reopen failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let after_bytes = fs::read(after_path)?;
    let after: Value = serde_json::from_slice(&after_bytes).map_err(|error| -> Box<dyn Error> {
        format!(
            "cache_after.json malformed ({} bytes): {error}; {}",
            after_bytes.len(),
            String::from_utf8_lossy(&after_bytes)
        )
        .into()
    })?;
    for name in [
        "OwnerMeshCachePC2",
        "OwnerMeshCacheMDD",
        "OwnerMeshSequenceCache",
    ] {
        assert_eq!(
            after[name]["type"], before[name]["type"],
            "{name}: reopened modifier type"
        );
        assert_values(
            &after[name]["params"],
            &before[name]["params"],
            &format!("{name}: reopened modifier parameters"),
            1.0e-6,
        );
        assert_eq!(
            after[name]["resource_path"], before[name]["resource_path"],
            "{name}: reopened cache resource path"
        );
        assert_values(
            &after[name]["cache_settings"],
            &before[name]["cache_settings"],
            &format!("{name}: reopened CacheFile settings"),
            1.0e-6,
        );
        assert_eq!(
            after[name]["polygons"], before[name]["polygons"],
            "{name}: reopened topology"
        );
        assert_nearest_vertices(
            after[name]["positions"].as_array().unwrap(),
            before[name]["positions"].as_array().unwrap(),
            name,
            1.0e-5,
        );
    }
    Ok(())
}
