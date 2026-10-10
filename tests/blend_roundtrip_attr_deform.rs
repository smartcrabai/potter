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

const ATTRIBUTE_FIXTURE: &str = r"
import bpy, json, math, os, sys
root = sys.argv[-1]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

def make_mesh(name, offset=0.0, uv_shift=(0.0, 0.0)):
    points = []
    for y in range(4):
        for x in range(4):
            px = float(x) - 1.5
            py = float(y) - 1.5
            pz = 0.12 * math.sin(px * 1.3) * math.cos(py * 0.8) + offset
            points.append((px, py, pz))
    faces = []
    for y in range(3):
        for x in range(3):
            i = y * 4 + x
            faces.append((i, i + 1, i + 5, i + 4))
    mesh = bpy.data.meshes.new(name + 'Mesh')
    mesh.from_pydata(points, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name='UVMap')
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
            uv.data[loop_index].uv = (0.15 * vertex.co.x + 0.5 + uv_shift[0],
                                      0.15 * vertex.co.y + 0.5 + uv_shift[1])
    return mesh

def owner(name, offset=0.0, uv_shift=(0.0, 0.0)):
    obj = bpy.data.objects.new(name, make_mesh(name, offset, uv_shift))
    bpy.context.scene.collection.objects.link(obj)
    return obj
def quad_owner(name):
    mesh = bpy.data.meshes.new(name + 'Mesh')
    mesh.from_pydata([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],[],[(0,1,2,3)])
    mesh.update()
    uv = mesh.uv_layers.new(name='UVMap')
    for loop in mesh.loops:
        uv.data[loop.index].uv = [(0,0),(1,0),(1,1),(0,1)][loop.vertex_index]
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj

def groups(obj, first='Weight', second='WeightB', bias=0.0):
    a = obj.vertex_groups.new(name=first)
    b = obj.vertex_groups.new(name=second)
    for index, vertex in enumerate(obj.data.vertices):
        a.add([index], max(0.05, min(0.95, 0.08 + index * 0.055 + bias)), 'REPLACE')
        b.add([index], max(0.05, min(0.95, 0.85 - index * 0.04 - bias)), 'REPLACE')

obj = owner('OwnerWeightEdit')
groups(obj)
mod = obj.modifiers.new('WeightEdit', 'VERTEX_WEIGHT_EDIT')
mod.vertex_group = 'Weight'
mod.mask_vertex_group = 'WeightB'
mod.falloff_type = 'SMOOTH'
mod.use_add = True
mod.default_weight = 0.42
mod.add_threshold = 0.01
mod.use_remove = False
mod.normalize = False

obj = owner('OwnerWeightMix')
groups(obj)
mod = obj.modifiers.new('WeightMix', 'VERTEX_WEIGHT_MIX')
mod.vertex_group_a = 'Weight'
mod.vertex_group_b = 'WeightB'
mod.mix_mode = 'MUL'
mod.mix_set = 'ALL'
mod.default_weight_a = 0.17
mod.default_weight_b = 0.29
mod.mask_vertex_group = 'WeightB'
mod.normalize = False

obj = owner('OwnerWeightProximity')
groups(obj)
target = bpy.data.objects.new('ProximityTarget', make_mesh('ProximityTarget', 0.75))
bpy.context.scene.collection.objects.link(target)
mod = obj.modifiers.new('WeightProximity', 'VERTEX_WEIGHT_PROXIMITY')
mod.vertex_group = 'Weight'
mod.target = target
mod.proximity_mode = 'GEOMETRY'
mod.proximity_geometry = {'FACE'}
mod.min_dist = 0.15
mod.max_dist = 2.4
mod.falloff_type = 'LINEAR'
mod.mask_vertex_group = 'WeightB'

weighted_vertices = [(0,0,0),(4,0,0),(0,1,0),(0,0,3),(-2,0,0)]
weighted_faces = [(0,1,2),(0,2,3),(0,3,4)]
weighted_mesh = bpy.data.meshes.new('OwnerWeightedNormalMesh')
weighted_mesh.from_pydata(weighted_vertices,[],weighted_faces)
weighted_mesh.update()
obj = bpy.data.objects.new('OwnerWeightedNormal',weighted_mesh)
bpy.context.scene.collection.objects.link(obj)
for index, polygon in enumerate(obj.data.polygons):
    polygon.use_smooth = True
    polygon.material_index = 1 if index == 2 else 0
groups(obj)
mod = obj.modifiers.new('WeightedNormal', 'WEIGHTED_NORMAL')
mod.mode = 'FACE_AREA'
mod.weight = 50
mod.keep_sharp = False
mod.use_face_influence = False

obj = quad_owner('OwnerNormalEdit')
groups(obj)
target = bpy.data.objects.new('NormalTarget', None)
bpy.context.scene.collection.objects.link(target)
target.location = (2.0, -1.0, 0.0)
mod = obj.modifiers.new('NormalEdit', 'NORMAL_EDIT')
mod.mode = 'DIRECTIONAL'
mod.target = target
mod.offset = (0.5, 0.1, 0.0)
mod.mix_mode = 'ADD'
mod.mix_factor = 0.65
mod.mix_limit = math.pi
mod.use_direction_parallel = True

obj = owner('OwnerUVProject')
groups(obj)
obj.location = (0.2, -0.3, 0.15)
obj.rotation_euler = (0.03, -0.04, 0.15)
obj.scale = (1.1, 0.9, 1.0)
camera_data = bpy.data.cameras.new('ProjectorCameraData')
camera = bpy.data.objects.new('ProjectorCamera', camera_data)
bpy.context.scene.collection.objects.link(camera)
camera.location = (0.1, -0.1, 8.0)
camera.rotation_euler = (0.01, -0.02, 0.0)
camera_data.type = 'ORTHO'
camera_data.ortho_scale = 5.0
camera_data.sensor_fit = 'VERTICAL'
camera_data.shift_x = 0.07
camera_data.shift_y = -0.03
side_camera_data = bpy.data.cameras.new('ProjectorCameraSideData')
side_camera_data.type = 'ORTHO'
side_camera = bpy.data.objects.new('ProjectorCameraSide', side_camera_data)
bpy.context.scene.collection.objects.link(side_camera)
side_camera.location = (8.0, 0.0, 0.0)
side_camera.rotation_euler = (0.0, math.pi / 2.0, 0.0)
mod = obj.modifiers.new('UVProject', 'UV_PROJECT')
mod.projectors[0].object = camera
mod.projector_count = 2
mod.projectors[1].object = side_camera
mod.aspect_x = 1.4
mod.aspect_y = 1.1
mod.scale_x = 1.25
mod.scale_y = 0.85
mod.uv_layer = 'UVMap'

obj = owner('OwnerUVWarp')
groups(obj)
from_obj = bpy.data.objects.new('WarpFrom', None)
to_obj = bpy.data.objects.new('WarpTo', None)
bpy.context.scene.collection.objects.link(from_obj)
bpy.context.scene.collection.objects.link(to_obj)
from_obj.location = (-0.4, 0.2, 0.1)
to_obj.location = (0.7, -0.25, 0.35)
mod = obj.modifiers.new('UVWarp', 'UV_WARP')
mod.object_from = from_obj
mod.object_to = to_obj
mod.uv_layer = 'UVMap'
mod.center = (0.35, 0.65)
mod.offset = (0.12, -0.08)
mod.scale = (1.2, 0.75)
mod.rotation = 0.23
mod.axis_u = 'X'
mod.axis_v = 'Y'
mod.vertex_group = 'Weight'

obj = owner('OwnerDataTransfer')
groups(obj)
target = owner('DataTransferTarget', 0.2, (0.33, -0.21))
groups(target, 'Weight', 'WeightB', 0.18)
mod = obj.modifiers.new('DataTransfer', 'DATA_TRANSFER')
mod.object = target
mod.use_object_transform = False
mod.use_vert_data = True
mod.data_types_verts = {'VGROUP_WEIGHTS'}
mod.use_loop_data = True
mod.data_types_loops = {'UV'}
mod.vert_mapping = 'TOPOLOGY'
mod.loop_mapping = 'TOPOLOGY'
mod.mix_mode = 'REPLACE'
mod.mix_factor = 1.0
mod.layers_uv_select_src = 'ALL'
mod.layers_uv_select_dst = 'NAME'
mod.vertex_group = 'Weight'

bpy.context.view_layer.update()

def plain(value):
    if isinstance(value, bpy.types.ID):
        return value.name_full
    if hasattr(value, 'to_list'):
        return value.to_list()
    if hasattr(value, 'to_tuple'):
        return [plain(item) for item in value.to_tuple()]
    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
        try: return [plain(item) for item in value]
        except Exception: pass
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)

PARAMS = {
    'OwnerWeightEdit': ['vertex_group','mask_vertex_group','falloff_type','use_add','default_weight','add_threshold','use_remove','normalize'],
    'OwnerWeightMix': ['vertex_group_a','vertex_group_b','mask_vertex_group','mix_mode','mix_set','default_weight_a','default_weight_b','normalize'],
    'OwnerWeightProximity': ['vertex_group','mask_vertex_group','target','proximity_mode','proximity_geometry','min_dist','max_dist','falloff_type'],
    'OwnerWeightedNormal': ['mode','weight','keep_sharp','use_face_influence','vertex_group'],
'OwnerNormalEdit': ['mode','target','offset','mix_mode','mix_factor','mix_limit','use_direction_parallel','vertex_group'],
    'OwnerUVProject': ['aspect_x','aspect_y','scale_x','scale_y','uv_layer'],
'OwnerUVWarp': ['object_from','object_to','uv_layer','center','offset','scale','rotation','axis_u','axis_v','vertex_group'],
'OwnerDataTransfer': ['object','use_object_transform','use_vert_data','data_types_verts','use_loop_data','data_types_loops','vert_mapping','loop_mapping','mix_mode','mix_factor','layers_uv_select_src','layers_uv_select_dst','vertex_group'],
}

def summarize(name):
    obj = bpy.data.objects[name]
    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(dg)
    mesh = evaluated.to_mesh()
    try:
        modifier = obj.modifiers[0]
        params = {key: plain(getattr(modifier, key)) for key in PARAMS[name]}
        if name == 'OwnerUVProject':
            params['projectors'] = [plain(slot.object) for slot in modifier.projectors if slot.object]
        groups = {}
        for group_name in ('Weight', 'WeightB'):
            group = evaluated.vertex_groups.get(group_name)
            if group:
                values = []
                for vertex in mesh.vertices:
                    try: values.append(float(group.weight(vertex.index)))
                    except RuntimeError: values.append(0.0)
                groups[group_name] = values
        uv = {}
        for layer in mesh.uv_layers:
            uv[layer.name] = [list(item.uv) for item in layer.data]
        normals = [list(corner.vector) for corner in mesh.corner_normals]
        return {'type': modifier.type, 'params': params,
                'positions': [list(vertex.co) for vertex in mesh.vertices],
                'faces': [list(poly.vertices) for poly in mesh.polygons],
                'uv': uv, 'normals': normals, 'groups': groups}
    finally:
        evaluated.to_mesh_clear()

names = list(PARAMS)
with open(os.path.join(root, 'attribute_before.json'), 'w', encoding='utf-8') as output:
    json.dump({name: summarize(name) for name in names}, output)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, 'attribute_source.blend'))
";

const ATTRIBUTE_REOPEN: &str = r"
import bpy, json, sys
output_path = sys.argv[-1]
PARAMS = {
    'OwnerWeightEdit': ['vertex_group','mask_vertex_group','falloff_type','use_add','default_weight','add_threshold','use_remove','normalize'],
    'OwnerWeightMix': ['vertex_group_a','vertex_group_b','mask_vertex_group','mix_mode','mix_set','default_weight_a','default_weight_b','normalize'],
    'OwnerWeightProximity': ['vertex_group','mask_vertex_group','target','proximity_mode','proximity_geometry','min_dist','max_dist','falloff_type'],
    'OwnerWeightedNormal': ['mode','weight','keep_sharp','use_face_influence','vertex_group'],
'OwnerNormalEdit': ['mode','target','offset','mix_mode','mix_factor','mix_limit','use_direction_parallel','vertex_group'],
    'OwnerUVProject': ['aspect_x','aspect_y','scale_x','scale_y','uv_layer'],
'OwnerUVWarp': ['object_from','object_to','uv_layer','center','offset','scale','rotation','axis_u','axis_v','vertex_group'],
'OwnerDataTransfer': ['object','use_object_transform','use_vert_data','data_types_verts','use_loop_data','data_types_loops','vert_mapping','loop_mapping','mix_mode','mix_factor','layers_uv_select_src','layers_uv_select_dst','vertex_group'],
}
def plain(value):
    if isinstance(value, bpy.types.ID): return value.name_full
    if hasattr(value, 'to_list'): return value.to_list()
    if hasattr(value, 'to_tuple'): return [plain(item) for item in value.to_tuple()]
    if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
        try: return [plain(item) for item in value]
        except Exception: pass
    if isinstance(value, (bool, int, float, str)) or value is None: return value
    return str(value)
def summarize(name):
    obj = bpy.data.objects[name]
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        modifier = obj.modifiers[0]
        params = {key: plain(getattr(modifier, key)) for key in PARAMS[name]}
        if name == 'OwnerUVProject':
            params['projectors'] = [plain(slot.object) for slot in modifier.projectors if slot.object]
        groups = {}
        for group_name in ('Weight', 'WeightB'):
            group = evaluated.vertex_groups.get(group_name)
            if group:
                values = []
                for vertex in mesh.vertices:
                    try: values.append(float(group.weight(vertex.index)))
                    except RuntimeError: values.append(0.0)
                groups[group_name] = values
        uv = {layer.name:[list(item.uv) for item in layer.data] for layer in mesh.uv_layers}
        normals = [list(corner.vector) for corner in mesh.corner_normals]
        return {'type': modifier.type, 'params': params,
                'positions':[list(vertex.co) for vertex in mesh.vertices],
                'faces':[list(poly.vertices) for poly in mesh.polygons],
                'uv':uv, 'normals':normals, 'groups':groups}
    finally:
        evaluated.to_mesh_clear()
with open(output_path, 'w', encoding='utf-8') as output:
    json.dump({name:summarize(name) for name in PARAMS}, output)
";

const NORMAL_QUANTIZATION: &str = r"
import bpy, json, os, sys
root = sys.argv[-1]
with open(os.path.join(root, 'normal_quantization_input.json'), encoding='utf-8') as stream:
    data = json.load(stream)
mesh = bpy.data.meshes.new('NormalQuantization')
mesh.from_pydata(data['vertices'], [], data['faces'])
mesh.update()
for index, polygon in enumerate(mesh.polygons):
    polygon.use_smooth = True
    polygon.material_index = data['materials'][index]
sharp_edges = {tuple(sorted(edge)) for edge in data['sharp_edges']}
for edge in mesh.edges:
    edge.use_edge_sharp = tuple(sorted(edge.vertices)) in sharp_edges
mesh.update()
mesh.normals_split_custom_set(data['normals'])
obj = bpy.data.objects.new('NormalQuantization', mesh)
bpy.context.collection.objects.link(obj)
evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
evaluated_mesh = evaluated.to_mesh()
normals = [list(loop.normal) for loop in evaluated_mesh.loops]
evaluated.to_mesh_clear()
with open(os.path.join(root, 'normal_quantization_output.json'), 'w', encoding='utf-8') as stream:
    json.dump(normals, stream)
";

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

fn assert_flat_float_arrays(actual: &[Value], expected: &[Value], context: &str, tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{context}: value count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_values(actual, expected, &format!("{context}[{index}]"), tolerance);
    }
}

fn modifier_attribute(mesh: &potter_core::geom::Mesh, group: &str) -> Vec<Value> {
    let values = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .and_then(|groups| groups.get(group))
        .and_then(Value::as_object);
    mesh.vertices
        .iter()
        .map(|vertex| {
            values
                .and_then(|values| values.get(&format!("v{}", vertex.id)))
                .cloned()
                .unwrap_or_else(|| json!(0.0))
        })
        .collect()
}

fn modifier_uvs(mesh: &potter_core::geom::Mesh, layer: &str) -> Vec<Value> {
    let entries = mesh.attributes.get("uv_map").and_then(Value::as_array);
    let mut result = Vec::new();
    for face in &mesh.faces {
        if let Some(uv) = entries
            .into_iter()
            .flatten()
            .find(|entry| {
                entry["face_id"].as_u64() == Some(u64::from(face.id))
                    && entry["layer"].as_str().unwrap_or("UVMap") == layer
            })
            .and_then(|entry| entry["uv"].as_array())
        {
            result.extend(uv.iter().cloned());
        }
    }
    result
}

fn modifier_normals(mesh: &potter_core::geom::Mesh) -> Vec<Value> {
    let Some(values) = mesh
        .attributes
        .get("custom_normal")
        .and_then(|attribute| attribute["values"].as_object())
    else {
        return Vec::new();
    };
    let mut result = Vec::new();
    for face in &mesh.faces {
        if let Some(corners) = values
            .get(&format!("f{}", face.id))
            .and_then(Value::as_array)
        {
            result.extend(corners.iter().cloned());
        }
    }
    result
}
fn blender_quantize_loop_normals(
    blender: &Path,
    root: &Path,
    mesh: &potter_core::geom::Mesh,
    normals: &[Value],
) -> Result<Vec<Value>, Box<dyn Error>> {
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let vertices = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect::<Vec<_>>();
    let faces = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|vertex_id| {
                    vertex_indices
                        .get(vertex_id)
                        .copied()
                        .ok_or("normal quantization face vertex missing")
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sharp_edges = mesh
        .edges
        .iter()
        .filter(|edge| {
            mesh.attributes
                .get("sharp_edges")
                .and_then(Value::as_array)
                .is_some_and(|ids| ids.iter().any(|id| id.as_u64() == Some(u64::from(edge.id))))
        })
        .map(|edge| {
            edge.vertices
                .iter()
                .map(|vertex_id| {
                    vertex_indices
                        .get(vertex_id)
                        .copied()
                        .ok_or("normal quantization sharp-edge vertex missing")
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let materials = mesh
        .faces
        .iter()
        .map(|face| face.material_index)
        .collect::<Vec<_>>();
    let input = json!({
        "vertices": vertices,
        "faces": faces,
        "materials": materials,
        "sharp_edges": sharp_edges,
        "normals": normals,
    });
    fs::write(
        root.join("normal_quantization_input.json"),
        serde_json::to_vec(&input)?,
    )?;
    run_blender_script(
        blender,
        "quantize_loop_normals.py",
        NORMAL_QUANTIZATION,
        root,
        &[],
        "Blender fixture failed",
        run_guarded,
    )?;
    Ok(serde_json::from_slice(&fs::read(
        root.join("normal_quantization_output.json"),
    )?)?)
}

fn assert_attribute_case(
    blender: &Path,
    root: &Path,
    scene: &potter_core::model::SceneDoc,
    snapshot: &potter_core::eval::Snapshot,
    mappings: &serde_json::Map<String, Value>,
    expected: &Value,
    name: &str,
) -> Result<(), Box<dyn Error>> {
    let owner_id = mappings[&format!("Object:{name}")]
        .as_str()
        .ok_or("object mapping did not contain a string ID")?;
    let owner_key = potter_core::model::Id::new(owner_id.to_owned())?;
    let node = &scene.nodes[&owner_key];
    let modifier = node
        .modifiers
        .first()
        .ok_or("modifier missing after import")?;
    let expected_item = &expected[name];
    let expected_type = expected_item["type"]
        .as_str()
        .ok_or("fixture modifier type missing")?;
    assert_eq!(
        modifier.modifier_type.to_ascii_uppercase(),
        expected_type,
        "{name}"
    );
    let evaluated = snapshot
        .meshes
        .get(&owner_key)
        .ok_or("evaluated mesh missing")?;
    assert_eq!(
        evaluated.faces.len(),
        expected_item["faces"].as_array().unwrap().len(),
        "{name}: face count"
    );
    let eval_max_error = assert_nearest_vertices(
        &evaluated
            .vertices
            .iter()
            .map(|vertex| json!(vertex.co.to_array()))
            .collect::<Vec<_>>(),
        expected_item["positions"].as_array().unwrap(),
        &format!("{name} import evaluation"),
        1.0e-5,
    );

    match name {
        "OwnerWeightEdit" | "OwnerWeightMix" | "OwnerWeightProximity" => {}
        "OwnerWeightedNormal" | "OwnerNormalEdit" => {
            let normals = modifier_normals(evaluated);
            let quantized = blender_quantize_loop_normals(blender, root, evaluated, &normals)?;
            assert_flat_float_arrays(
                &quantized,
                expected_item["normals"].as_array().unwrap(),
                &format!("{name} output loop normals after Blender short2 storage"),
                1.0e-5,
            );
        }
        "OwnerUVProject" | "OwnerUVWarp" | "OwnerDataTransfer" => assert_flat_float_arrays(
            &modifier_uvs(evaluated, "UVMap"),
            expected_item["uv"]["UVMap"].as_array().unwrap(),
            &format!("{name} output loop UVs"),
            1.0e-5,
        ),
        _ => return Err(format!("unexpected attribute fixture {name}").into()),
    }
    for group in ["Weight", "WeightB"] {
        assert_flat_float_arrays(
            &modifier_attribute(evaluated, group),
            expected_item["groups"][group].as_array().unwrap(),
            &format!("{name} output vertex group {group}"),
            1.0e-5,
        );
    }
    let expected_params = &expected_item["params"];
    for key in [
        "vertex_group",
        "vertex_group_a",
        "vertex_group_b",
        "mask_vertex_group",
        "falloff_type",
        "use_add",
        "default_weight",
        "add_threshold",
        "use_remove",
        "normalize",
        "mix_mode",
        "mix_set",
        "default_weight_a",
        "default_weight_b",
        "proximity_mode",
        "proximity_geometry",
        "min_dist",
        "max_dist",
        "mode",
        "weight",
        "keep_sharp",
        "use_face_influence",
        "offset",
        "mix_factor",
        "mix_limit",
        "use_direction_parallel",
        "aspect_x",
        "aspect_y",
        "scale_x",
        "scale_y",
        "uv_layer",
        "center",
        "scale",
        "rotation",
        "axis_u",
        "axis_v",
        "use_object_transform",
        "use_vert_data",
        "data_types_verts",
        "use_loop_data",
        "data_types_loops",
        "vert_mapping",
        "loop_mapping",
        "layers_uv_select_src",
        "layers_uv_select_dst",
    ] {
        if let Some(value) = expected_params.get(key) {
            if matches!(
                key,
                "target" | "object" | "object_from" | "object_to" | "projectors"
            ) {
                continue;
            }
            let actual = modifier
                .params
                .get(key)
                .ok_or_else(|| format!("{name}.params.{key} is missing after import"))?;
            assert_values(actual, value, &format!("{name}.params.{key}"), 1.0e-6);
        }
    }
    let reference_names: &[(&str, &str)] = match name {
        "OwnerWeightProximity" => &[("target", "ProximityTarget")],
        "OwnerNormalEdit" => &[("target", "NormalTarget")],
        "OwnerUVWarp" => &[("object_from", "WarpFrom"), ("object_to", "WarpTo")],
        "OwnerDataTransfer" => &[("object", "DataTransferTarget")],
        _ => &[],
    };
    for (parameter, object_name) in reference_names {
        let id = mappings[&format!("Object:{object_name}")].as_str().unwrap();
        assert_eq!(
            modifier.params[*parameter],
            json!(id),
            "{name}.params.{parameter}"
        );
    }
    if name == "OwnerUVProject" {
        let camera_id = mappings["Object:ProjectorCamera"].as_str().unwrap();
        let side_camera_id = mappings["Object:ProjectorCameraSide"].as_str().unwrap();
        assert_eq!(
            modifier.params["projectors"],
            json!([camera_id, side_camera_id]),
            "{name}.params.projectors",
        );
    }
    eprintln!("{name}: import/loss-free params=ok eval_max_error={eval_max_error:.9e}");
    Ok(())
}

#[test]
fn blender_attribute_deformation_modifiers_round_trip_attributes_and_settings()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping Blender attribute modifier parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_attribute_fixture.py",
        ATTRIBUTE_FIXTURE,
        root,
        &[],
        "Blender fixture failed",
        run_guarded,
    )?;
    let before: Value = serde_json::from_slice(&fs::read(root.join("attribute_before.json"))?)?;
    let project = root.join("attribute_project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("attribute_source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let losses = imported["result"]["losses"].as_array().unwrap();
    assert!(
        losses.is_empty(),
        "attribute modifier fixture import reported losses: {losses:?}"
    );
    let mappings = imported["result"]["id_mappings"].as_object().unwrap();
    let scene: potter_core::model::SceneDoc =
        serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let snapshot = potter_core::eval::Snapshot::evaluate(
        &scene,
        &potter_core::eval::EvaluationContext::default(),
    )?;
    for name in [
        "OwnerWeightEdit",
        "OwnerWeightMix",
        "OwnerWeightProximity",
        "OwnerWeightedNormal",
        "OwnerNormalEdit",
        "OwnerUVProject",
        "OwnerUVWarp",
        "OwnerDataTransfer",
    ] {
        assert_attribute_case(&blender, root, &scene, &snapshot, mappings, &before, name)?;
    }

    let exported = root.join("attribute_roundtrip.blend");
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
    let after_path = root.join("attribute_after.json");
    let reopen_path = root.join("reopen_attribute.py");
    fs::write(&reopen_path, ATTRIBUTE_REOPEN)?;
    let mut command = Command::new(&blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&reopen_path)
        .arg("--")
        .arg(&after_path);
    let output = run_guarded(command)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "Blender attribute reopen failed: stdout={stdout} stderr={stderr}"
    );
    let after: Value = serde_json::from_slice(&fs::read(after_path)?)?;
    for name in [
        "OwnerWeightEdit",
        "OwnerWeightMix",
        "OwnerWeightProximity",
        "OwnerWeightedNormal",
        "OwnerNormalEdit",
        "OwnerUVProject",
        "OwnerUVWarp",
        "OwnerDataTransfer",
    ] {
        let source = &before[name];
        let reopened = &after[name];
        assert_eq!(
            reopened["type"], source["type"],
            "{name}: modifier type after export"
        );
        for (key, expected) in source["params"].as_object().unwrap() {
            assert_values(
                &reopened["params"][key],
                expected,
                &format!("{name}.exported.{key}"),
                1.0e-6,
            );
        }
        assert_eq!(
            reopened["faces"], source["faces"],
            "{name}: evaluated face connectivity after export"
        );
        let reopened_error = assert_nearest_vertices(
            reopened["positions"].as_array().unwrap(),
            source["positions"].as_array().unwrap(),
            &format!("{name} export/reopen"),
            1.0e-5,
        );
        assert_values(
            &reopened["uv"],
            &source["uv"],
            &format!("{name}: evaluated UV loops after export"),
            1.0e-5,
        );
        assert_values(
            &reopened["groups"],
            &source["groups"],
            &format!("{name}: evaluated weights after export"),
            1.0e-5,
        );
        if matches!(name, "OwnerWeightedNormal" | "OwnerNormalEdit") {
            assert_values(
                &reopened["normals"],
                &source["normals"],
                &format!("{name}: evaluated loop normals after export"),
                1.0e-5,
            );
        }
        eprintln!("{name}: export/reopen params/attributes=ok eval_max_error={reopened_error:.9e}");
    }
    Ok(())
}

// Remaining fixture families share the same child-process guard and comparison helpers above.
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
        .args(["--background"])
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
