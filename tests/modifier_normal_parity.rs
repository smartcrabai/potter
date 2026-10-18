use std::{collections::BTreeMap, error::Error, fs, path::Path, process::Command};

use glam::{DMat4, DVec3};
use potter_core::{
    geom::{
        Mesh,
        modifiers::attributes::{AttributeLayerSelection, SceneOperand, SceneOperandMatricesF32, evaluate_with_context},
    },
    model::{Modifier, ModifierRuntime},
};
use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/blender_script.rs"]
mod blender_script;

use blender_file::blender_executable;

const VERTICES: [[f64; 3]; 9] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.1, 0.25],
    [2.2, -0.1, 0.1],
    [0.05, 1.1, 0.3],
    [1.3, 1.2, 0.9],
    [2.5, 0.9, -0.2],
    [-0.3, 2.2, 0.1],
    [1.1, 2.4, 0.4],
    [2.9, 2.0, 0.6],
];

const BLENDER_NORMAL_PROBE: &str = r#"
import bpy, json, os, sys
root = os.path.realpath(sys.argv[sys.argv.index('--') + 1])
cases = json.load(open(os.path.join(root, 'cases.json'), encoding='utf-8'))
results = {}
for case in cases:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    mesh = bpy.data.meshes.new('ProbeMesh')
    mesh.from_pydata(case['vertices'], [], case['faces'])
    mesh.update()
    obj = bpy.data.objects.new('ProbeSubject', mesh)
    bpy.context.collection.objects.link(obj)
    subject_transform = case.get('subject_transform')
    if subject_transform is not None:
        obj.location = subject_transform['location']
        obj.rotation_euler = subject_transform['rotation_euler']
        obj.scale = subject_transform['scale']
    for i, polygon in enumerate(mesh.polygons):
        polygon.use_smooth = case['smooth'][i]
    for edge_pair in case['sharp_edges']:
        edge = {tuple(sorted(edge_pair))}
        for item in mesh.edges:
            if tuple(sorted(item.vertices)) in edge:
                item.use_edge_sharp = True
    for name, group_weights in case['groups'].items():
        group = obj.vertex_groups.new(name=name)
        for vertex_index, weight in group_weights.items():
            group.add([int(vertex_index)], float(weight), 'REPLACE')
    if case['face_strength'] is not None:
        layer = mesh.attributes.new(name='__mod_weightednormals_faceweight', type='INT', domain='FACE')
        for i, strength in enumerate(case['face_strength']):
            layer.data[i].value = strength
    if case['custom_normals'] is not None:
        mesh.normals_split_custom_set(case['custom_normals'])
    mesh.attributes.new(name='corner_tag', type='FLOAT', domain='CORNER')
    tag = mesh.attributes['corner_tag']
    for i in range(len(tag.data)):
        tag.data[i].value = float(i + 1)
    color = mesh.attributes.new(name='corner_color', type='FLOAT_COLOR', domain='CORNER')
    for i in range(len(color.data)):
        color.data[i].color = (float(i) * 0.03125, 0.25, 0.5, 1.0)
    uv = mesh.uv_layers.new(name='UVMap')
    for loop in mesh.loops:
        uv.data[loop.index].uv = (float(loop.index), float(loop.index + 100))
    target = None
    if case['target'] is not None:
        target = bpy.data.objects.new('ProbeTarget', None)
        bpy.context.collection.objects.link(target)
        target.location = case['target']['location']
        target.scale = case['target']['scale']
        target.rotation_euler = case['target'].get('rotation_euler', (0.0, 0.0, 0.0))
    bpy.context.view_layer.update()
    if target is not None:
        columns = lambda matrix: [[float(matrix[row][column]) for row in range(4)] for column in range(4)]
        relative = obj.matrix_world.inverted() @ target.matrix_world
        subject_to_target = relative.inverted()
        if '_transform_matrices' not in results:
            results['_transform_matrices'] = {}
        results['_transform_matrices'][case['name']] = {
            'local_to_subject': columns(relative),
            'subject_to_local': columns(subject_to_target),
            'world': columns(target.matrix_world),
            'subject_world': columns(obj.matrix_world),
        }
    for name, value in case['params'].items():
        if name == 'target':
            modifier.target = target
        else:
            setattr(modifier, name, value)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    out = evaluated.to_mesh()
    faces = []
    for polygon in out.polygons:
        loops = range(polygon.loop_start, polygon.loop_start + polygon.loop_total)
        faces.append({
            'vertices': [out.loops[i].vertex_index for i in loops],
            'normals': [list(out.loops[i].normal) for i in loops],
            'tags': [float(out.attributes['corner_tag'].data[i].value) for i in loops],
            'colors': [list(out.attributes['corner_color'].data[i].color) for i in loops],
            'uvs': [list(out.uv_layers['UVMap'].data[i].uv) for i in loops],
        })
    evaluated.to_mesh_clear()
    results[case['name']] = faces
json.dump(results, open(os.path.join(root, 'results.json'), 'w'), separators=(',', ':'))
print('NORMAL_PROBE_DONE', len(results))
"#;

fn parameter_case(
    name: &str,
    modifier_type: &str,
    params: Value,
    target: Option<Value>,
    groups: Value,
    face_strength: Option<Vec<i32>>,
    sharp_edges: Vec<[usize; 2]>,
    smooth: Vec<bool>,
    custom_normals: Option<Vec<Vec<[f64; 3]>>>,
) -> Value {
    json!({
        "name":name,
        "vertices":VERTICES,
        "faces":[[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8],[4,8,7]],
        "modifier_type":modifier_type,
        "params":params,
        "target":target,
        "groups":groups,
        "face_strength":face_strength,
        "sharp_edges":sharp_edges,
        "smooth":smooth,
        "custom_normals":custom_normals,
        "subject_transform":None,
    })
}

fn target_operand(location: [f64; 3], scale: [f64; 3]) -> Value {
    json!({"location":location,"scale":scale})
}

fn normal_edit_cases() -> Vec<Value> {
    let radial_target = target_operand([2.5, -1.5, 1.0], [2.0, 0.5, 1.5]);
    let group = json!({"G":{"0":1.0,"2":0.3,"4":0.6,"7":0.9}});
    let smooth = vec![true; 5];
    let mut cases = Vec::new();
    for (mix_mode, factor, limit) in [
        ("COPY", 0.48, 0.72),
        ("ADD", 0.65, 3.141592653589793),
        ("SUB", 0.4, 0.35),
        ("MUL", 0.7, 0.9),
    ] {
        cases.push(parameter_case(
            &format!("radial_{mix_mode}"),
            "NORMAL_EDIT",
            json!({"mode":"RADIAL","mix_mode":mix_mode,"mix_factor":factor,"mix_limit":limit,"target":"target","offset":[0.5,-0.2,0.1],"no_polynors_fix":true}),
            Some(radial_target.clone()),
            json!({}),
            None,
            vec![],
            smooth.clone(),
            None,
        ));
    }
    cases.push(parameter_case(
        "radial_noop_ellipsoid_scale",
        "NORMAL_EDIT",
        json!({"mode":"RADIAL","mix_mode":"COPY","mix_factor":1.0,"mix_limit":std::f64::consts::PI,"target":"target","no_polynors_fix":true}),
        Some(radial_target.clone()),
        json!({}),
        None,
        vec![],
        smooth.clone(),
        None,
    ));
    cases.push(parameter_case(
        "radial_group_inverted",
        "NORMAL_EDIT",
        json!({"mode":"RADIAL","mix_mode":"COPY","mix_factor":0.55,"mix_limit":std::f64::consts::PI,"target":"target","vertex_group":"G","invert_vertex_group":true,"no_polynors_fix":true}),
        Some(radial_target),
        group.clone(),
        None,
        vec![],
        smooth.clone(),
        None,
    ));
    for (name, scale) in [
        ("target_zero_scale", [0.0, 0.0, 0.0]),
        ("target_negative_scale", [-2.0, 0.5, 1.5]),
    ] {
        cases.push(parameter_case(
            name,
            "NORMAL_EDIT",
            json!({"mode":"RADIAL","mix_mode":"ADD","mix_factor":0.55,"mix_limit":1.2,"target":"target","no_polynors_fix":true}),
            Some(target_operand([2.5, -1.5, 1.0], scale)),
            json!({}),
            None,
            vec![],
            smooth.clone(),
            None,
        ));
    }
    let source_custom_normals = vec![
        vec![[0.2, 0.1, 0.9746794344808963]; 4],
        vec![[0.2, 0.1, 0.9746794344808963]; 4],
        vec![[0.2, 0.1, 0.9746794344808963]; 4],
        vec![[0.2, 0.1, 0.9746794344808963]; 3],
        vec![[0.2, 0.1, 0.9746794344808963]; 3],
    ];
    cases.push(parameter_case(
        "directional_mixes_from_existing_corner_normals",
        "NORMAL_EDIT",
        json!({"mode":"DIRECTIONAL","mix_mode":"ADD","mix_factor":0.38,"mix_limit":0.8,"target":"target","use_direction_parallel":false,"no_polynors_fix":true}),
        Some(target_operand([2.5, -1.5, 1.0], [1.0, 1.0, 1.0])),
        json!({}),
        None,
        vec![],
        smooth.clone(),
        Some(source_custom_normals),
    ));
    for parallel in [false, true] {
        cases.push(parameter_case(
            &format!("directional_parallel_{parallel}"),
            "NORMAL_EDIT",
            json!({"mode":"DIRECTIONAL","mix_mode":"ADD","mix_factor":0.63,"mix_limit":1.25,"target":"target","offset":[0.5,0.1,-0.3],"use_direction_parallel":parallel,"no_polynors_fix":true}),
            Some(target_operand([2.5, -1.5, 1.0], [1.0, 1.0, 1.0])),
            json!({}),
            None,
            vec![],
            smooth.clone(),
            None,
        ));
    }
    cases.push(parameter_case(
        "directional_group",
        "NORMAL_EDIT",
        json!({"mode":"DIRECTIONAL","mix_mode":"SUB","mix_factor":0.7,"mix_limit":std::f64::consts::PI,"target":"target","vertex_group":"G","use_direction_parallel":false,"no_polynors_fix":true}),
        Some(target_operand([2.5, -1.5, 1.0], [1.0, 1.0, 1.0])),
        group,
        None,
        vec![],
        smooth.clone(),
        None,
    ));
    let mut transformed_target = target_operand([2.35, -1.65, 1.2], [2.0, 0.55, 1.35]);
    transformed_target["rotation_euler"] = json!([0.23, -0.37, 0.41]);
    let subject_transform = json!({
        "location":[-1.1,0.65,0.4],
        "rotation_euler":[0.31,-0.43,0.17],
        "scale":[1.3,0.7,1.6]
    });
    for (mode, parallel) in [("RADIAL", false), ("DIRECTIONAL", false), ("DIRECTIONAL", true)] {
        let mut case = parameter_case(
            &format!("f32_owner_target_{mode}_{parallel}"),
            "NORMAL_EDIT",
            json!({"mode":mode,"mix_mode":"COPY","mix_factor":1.0,"mix_limit":std::f64::consts::PI,"target":"target","offset":[0.2,-0.35,0.15],"use_direction_parallel":parallel,"no_polynors_fix":true}),
            Some(transformed_target.clone()),
            json!({}),
            None,
            vec![],
            smooth.clone(),
            None,
        );
        case["subject_transform"] = subject_transform.clone();
        cases.push(case);
    }
    cases.extend([
        json!({"name":"flip_fix_on","vertices":[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],"faces":[[0,1,2,3]],"modifier_type":"NORMAL_EDIT","params":{"mode":"DIRECTIONAL","mix_mode":"COPY","mix_factor":1.0,"mix_limit":std::f64::consts::PI,"target":"target","use_direction_parallel":false,"no_polynors_fix":false},"target":{"location":[0.5,0.5,-2.0],"scale":[1,1,1]},"groups":{},"face_strength":null,"sharp_edges":[],"smooth":[true],"custom_normals":null}),
        json!({"name":"flip_fix_off","vertices":[[0,0,0],[1,0,0],[1,1,0],[0,1,0]],"faces":[[0,1,2,3]],"modifier_type":"NORMAL_EDIT","params":{"mode":"DIRECTIONAL","mix_mode":"COPY","mix_factor":1.0,"mix_limit":std::f64::consts::PI,"target":"target","use_direction_parallel":false,"no_polynors_fix":true},"target":{"location":[0.5,0.5,-2.0],"scale":[1,1,1]},"groups":{},"face_strength":null,"sharp_edges":[],"smooth":[true],"custom_normals":null}),
    ]);
    cases.push(parameter_case(
        "radial_bounds_offset",
        "NORMAL_EDIT",
        json!({"mode":"RADIAL","mix_mode":"COPY","mix_factor":0.45,"mix_limit":std::f64::consts::PI,"offset":[0.3,-0.2,0.1],"no_polynors_fix":true}),
        None,
        json!({}),
        None,
        vec![],
        smooth,
        None,
    ));
    cases
}

fn weighted_normal_cases() -> Result<Vec<Value>, Box<dyn Error>> {
    let group = json!({"G":{"0":1.0,"2":0.3,"4":0.6,"7":0.9}});
    let smooth = vec![true; 5];
    let mut cases = Vec::new();
    for (mode, weight) in [
        ("FACE_AREA", 80),
        ("CORNER_ANGLE", 65),
        ("FACE_AREA_WITH_ANGLE", 70),
    ] {
        cases.push(parameter_case(
            &format!("{mode}_weight_{weight}"),
            "WEIGHTED_NORMAL",
            json!({"mode":mode,"weight":weight,"thresh":0.01,"keep_sharp":false,"use_face_influence":false}),
            None,
            json!({}),
            None,
            vec![],
            smooth.clone(),
            None,
        ));
    }
    for weight in [1, 20, 50, 52, 100] {
        cases.push(parameter_case(
            &format!("FACE_AREA_weight_{weight}"),
            "WEIGHTED_NORMAL",
            json!({"mode":"FACE_AREA","weight":weight,"thresh":0.01}),
            None,
            json!({}),
            None,
            vec![],
            smooth.clone(),
            None,
        ));
    }
    cases.push(parameter_case(
        "FACE_AREA_thresh_0_5",
        "WEIGHTED_NORMAL",
        json!({"mode":"FACE_AREA","weight":60,"thresh":0.5}),
        None,
        json!({}),
        None,
        vec![],
        smooth.clone(),
        None,
    ));
    cases.push(parameter_case(
        "FACE_AREA_keep_sharp",
        "WEIGHTED_NORMAL",
        json!({"mode":"FACE_AREA","weight":50,"thresh":0.01,"keep_sharp":true}),
        None,
        json!({}),
        None,
        vec![[1,4],[4,7]],
        smooth.clone(),
        None,
    ));
    cases.push(parameter_case(
        "CORNER_ANGLE_keep_sharp",
        "WEIGHTED_NORMAL",
        json!({"mode":"CORNER_ANGLE","weight":65,"thresh":0.01,"keep_sharp":true}),
        None,
        json!({}),
        None,
        vec![[1,4],[4,7]],
        smooth.clone(),
        None,
    ));
    cases.push(parameter_case(
        "FACE_AREA_with_face_influence",
        "WEIGHTED_NORMAL",
        json!({"mode":"FACE_AREA","weight":50,"thresh":0.01,"use_face_influence":true}),
        None,
        json!({}),
        Some(vec![0, 1, 2, 1, 0]),
        vec![],
        smooth.clone(),
        None,
    ));
    cases.push(parameter_case(
        "FACE_AREA_WITH_ANGLE_strength_keep_sharp",
        "WEIGHTED_NORMAL",
        json!({"mode":"FACE_AREA_WITH_ANGLE","weight":70,"thresh":0.01,"use_face_influence":true,"keep_sharp":true}),
        None,
        json!({}),
        Some(vec![0, 3, 1, 2, 2]),
        vec![[1,4],[4,7]],
        smooth.clone(),
        None,
    ));
    for invert in [false, true] {
        cases.push(parameter_case(
            &format!("vertex_group_invert_{invert}"),
            "WEIGHTED_NORMAL",
            json!({"mode":"FACE_AREA","weight":80,"thresh":0.01,"vertex_group":"G","invert_vertex_group":invert}),
            None,
            group.clone(),
            None,
            vec![],
            smooth.clone(),
            None,
        ));
    }
    cases.push(parameter_case(
        "mixed_smooth_faces",
        "WEIGHTED_NORMAL",
        json!({"mode":"CORNER_ANGLE","weight":30,"thresh":0.01}),
        None,
        json!({}),
        None,
        vec![],
        vec![false, true, true, false, true],
        None,
    ));
    cases.push(weighted_threshold_boundary_case());
    cases.push(weighted_custom_nonmanifold_case());
    cases.push(weighted_tied_area_case()?);
    Ok(cases)
}

fn weighted_threshold_boundary_case() -> Value {
    json!({
        "name":"FACE_AREA_thresh_exact_boundary",
        "vertices":[[0,0,0],[2,0,0],[0,2,0],[0,1,0],[0,0,3.5],[0,0,3],[1,0,0]],
        "faces":[[0,1,2],[0,3,4],[0,5,6]],
        "modifier_type":"WEIGHTED_NORMAL",
        "params":{"mode":"FACE_AREA","weight":80,"thresh":0.25},
        "target":null,
        "groups":{},
        "face_strength":null,
        "sharp_edges":[],
        "smooth":[true,true,true],
        "custom_normals":null
    })
}

fn weighted_custom_nonmanifold_case() -> Value {
    json!({
        "name":"FACE_AREA_KEEP_SHARP_custom_normals_nonmanifold",
        "vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,-1,1],[0,0,2],[1,1,0],[-1,0,0]],
        "faces":[[0,1,2],[1,0,3],[0,1,4],[0,5,6],[5,0,7]],
        "modifier_type":"WEIGHTED_NORMAL",
        "params":{"mode":"FACE_AREA","weight":65,"thresh":0.01,"keep_sharp":true},
        "target":null,
        "groups":{},
        "face_strength":null,
        "sharp_edges":[],
        "smooth":[true,true,true,true,true],
        "custom_normals":[
            [[0,0,1],[0,0,1],[0,0,1]],
            [[0,1,0],[0,1,0],[0,1,0]],
            [[1,0,0],[1,0,0],[1,0,0]],
            [[0,0,1],[0,0,1],[0,0,1]],
            [[0,1,0],[0,1,0],[0,1,0]]
        ]
    })
}

/// Signed-permutation faces share exact areas; an outlier makes Blender's qsort tie stream asymmetric.
fn tied_area_mesh() -> Result<Mesh, Box<dyn Error>> {
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut positions = vec![DVec3::ZERO];
    let mut faces = Vec::with_capacity(6 * 8 * 128);
    for permutation in permutations {
        for signs in 0..8 {
            let sign = [
                if signs & 4 == 0 { -1.0 } else { 1.0 },
                if signs & 2 == 0 { -1.0 } else { 1.0 },
                if signs & 1 == 0 { -1.0 } else { 1.0 },
            ];
            for _ in 0..128 {
                let rotate = |source: [f64; 3]| {
                    let mut point = [0.0; 3];
                    for axis in 0..3 {
                        point[permutation[axis]] = source[axis] * sign[axis];
                    }
                    DVec3::from_array(point)
                };
                let first = positions.len();
                positions.push(rotate([2.0, 0.0, 0.0]));
                positions.push(rotate([0.0, 1.0, 1.0]));
                faces.push(vec![0, first, first + 1]);
            }
        }
    }
    let first = positions.len();
    positions.push(DVec3::X);
    positions.push(DVec3::new(0.0, 0.5, 0.0));
    faces.push(vec![0, first, first + 1]);
    Ok(Mesh::from_positions_and_faces(positions, faces)?)
}

fn weighted_tied_area_case() -> Result<Value, Box<dyn Error>> {
    let mesh = tied_area_mesh()?;
    let vertices = mesh.vertices.iter().map(|vertex| vertex.co.to_array()).collect::<Vec<_>>();
    let faces = mesh.faces.iter().map(|face| face.vertices.clone()).collect::<Vec<_>>();
    Ok(json!({
        "name":"FACE_AREA_tied_qsort_stream",
        "vertices":vertices,
        "faces":faces,
        "modifier_type":"WEIGHTED_NORMAL",
        "params":{"mode":"FACE_AREA","weight":80,"thresh":0.01},
        "target":null,
        "groups":{},
        "face_strength":null,
        "sharp_edges":[],
        "smooth":vec![true;mesh.faces.len()],
        "custom_normals":null
    }))
}
fn blender_cases() -> Result<Vec<Value>, Box<dyn Error>> {
    let mut cases = normal_edit_cases();
    cases.extend(weighted_normal_cases()?);
    Ok(cases)
}

fn blender_probe(blender: &Path, root: &Path, cases: &[Value]) -> Result<Value, Box<dyn Error>> {
    fs::write(root.join("cases.json"), serde_json::to_vec(cases)?)?;
    blender_script::run_blender_script(
        blender,
        "normal_parity.py",
        BLENDER_NORMAL_PROBE,
        root,
        "Blender normal modifier probe failed",
    )?;
    Ok(serde_json::from_slice(&fs::read(root.join("results.json"))?)?)
}

fn pot_mesh(case: &Value) -> Result<Mesh, Box<dyn Error>> {
    let positions = case["vertices"]
        .as_array()
        .ok_or("case vertices are missing")?
        .iter()
        .map(|vertex| {
            let [x, y, z] = vertex.as_array().ok_or("vertex is not a vector")?.as_slice() else {
                return Err("vertex must have three components".into());
            };
            Ok(DVec3::new(
                x.as_f64().ok_or("vertex x is not numeric")?,
                y.as_f64().ok_or("vertex y is not numeric")?,
                z.as_f64().ok_or("vertex z is not numeric")?,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let faces = case["faces"]
        .as_array()
        .ok_or("case faces are missing")?
        .iter()
        .map(|face| {
            face.as_array()
                .ok_or("face is not an array")?
                .iter()
                .map(|vertex| {
                    usize::try_from(vertex.as_u64().ok_or("face vertex is not an index")?)
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let mut mesh = Mesh::from_positions_and_faces(positions, faces)?;
    let smooth = case["smooth"].as_array().ok_or("smooth flags are missing")?;
    let smooth_values = mesh
        .faces
        .iter()
        .zip(smooth)
        .map(|(face, flag)| (format!("f{}", face.id), flag.clone()))
        .collect::<serde_json::Map<_, _>>();
    mesh.attributes.insert(
        "shade_smooth".to_owned(),
        json!({"domain":"face","type":"bool","values":smooth_values}),
    );
    let mut sharp_values = serde_json::Map::new();
    for sharp_edge in case["sharp_edges"].as_array().ok_or("sharp edges missing")? {
        let [first, second] = sharp_edge.as_array().ok_or("sharp edge is malformed")?.as_slice() else {
            return Err("sharp edge must have two endpoints".into());
        };
        let first = u32::try_from(first.as_u64().ok_or("sharp edge endpoint is malformed")?)?;
        let second = u32::try_from(second.as_u64().ok_or("sharp edge endpoint is malformed")?)?;
        if let Some(edge) = mesh.edges.iter().find(|edge| {
            (edge.vertices[0] == first && edge.vertices[1] == second)
                || (edge.vertices[0] == second && edge.vertices[1] == first)
        }) {
            sharp_values.insert(format!("e{}", edge.id), json!(true));
        }
    }
    mesh.attributes.insert(
        "sharp".to_owned(),
        json!({"domain":"edge","type":"bool","values":sharp_values}),
    );
    if let Some(groups) = case["groups"].as_object() {
        let mut group_values = serde_json::Map::new();
        for (name, values) in groups {
            let weights = values.as_object().ok_or("vertex group data is malformed")?;
            let potter_weights = weights
                .iter()
                .map(|(vertex, weight)| {
                    let index = vertex.strip_prefix('v').unwrap_or(vertex);
                    (format!("v{index}"), weight.clone())
                })
                .collect::<serde_json::Map<_, _>>();
            group_values.insert(name.clone(), Value::Object(potter_weights));
        }
        mesh.attributes.insert("vertex_groups".to_owned(), Value::Object(group_values));
    }
    if let Some(strengths) = case["face_strength"].as_array() {
        let values = mesh
            .faces
            .iter()
            .zip(strengths)
            .map(|(face, strength)| (format!("f{}", face.id), strength.clone()))
            .collect::<serde_json::Map<_, _>>();
        mesh.attributes.insert(
            "__mod_weightednormals_faceweight".to_owned(),
            json!({"domain":"face","type":"int","values":values}),
        );
    }
    let mut corner_tag_values = serde_json::Map::new();
    let mut corner_color_values = serde_json::Map::new();
    let mut uv_entries = Vec::new();
    let mut next_tag = 1.0;
    let mut next_corner = 0_usize;
    for face in &mesh.faces {
        let tags = (0..face.vertices.len())
            .map(|_| {
                let tag = next_tag;
                next_tag += 1.0;
                json!(tag)
            })
            .collect::<Vec<_>>();
        let colors = (0..face.vertices.len())
            .map(|corner| json!([(next_corner + corner) as f64 * 0.03125, 0.25, 0.5, 1.0]))
            .collect::<Vec<_>>();
        let uvs = (0..face.vertices.len())
            .map(|corner| {
                let loop_index = next_corner + corner;
                json!([loop_index as f64, (loop_index + 100) as f64])
            })
            .collect::<Vec<_>>();
        corner_tag_values.insert(format!("f{}", face.id), Value::Array(tags));
        corner_color_values.insert(format!("f{}", face.id), Value::Array(colors));
        uv_entries.push(json!({"face_id":face.id,"uv":uvs}));
        next_corner += face.vertices.len();
    }
    mesh.attributes.insert(
        "corner_tag".to_owned(),
        json!({"domain":"corner","type":"float","values":corner_tag_values}),
    );
    mesh.attributes.insert(
        "corner_color".to_owned(),
        json!({"domain":"corner","type":"color","values":corner_color_values}),
    );
    mesh.attributes.insert(
        "uv_layers".to_owned(),
        json!({"layers":[{"name":"UVMap","faces":uv_entries}],"active":"UVMap","render":"UVMap"}),
    );
    if let Some(normals) = case["custom_normals"].as_array() {
        let values = mesh
            .faces
            .iter()
            .zip(normals)
            .map(|(face, normals)| (format!("f{}", face.id), normals.clone()))
            .collect::<serde_json::Map<_, _>>();
        let ids = mesh
            .faces
            .iter()
            .map(|face| format!("f{}", face.id))
            .collect::<Vec<_>>();
        potter_core::geom::edit::apply(
            &mut mesh,
            "mesh.set_custom_normals",
            &json!({
                "elements":{"domain":"face","ids":ids},
                "normals":values,
            }),
        )?;
    }
    Ok(mesh)
}

fn modifier(case: &Value) -> Result<Modifier, Box<dyn Error>> {
    Ok(Modifier {
        id: potter_core::model::Id::new("normal_modifier")?,
        modifier_type: case["modifier_type"].as_str().ok_or("modifier type missing")?.to_owned(),
        name: "Normal modifier parity".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: case["params"].as_object().ok_or("modifier params missing")?.clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

fn evaluate_case(case: &Value, transform: Option<&Value>) -> Result<Mesh, Box<dyn Error>> {
    let mesh = pot_mesh(case)?;
    let mut operands = BTreeMap::new();
    if let Some(target) = case["target"].as_object() {
        let [sx, sy, sz] = target["scale"].as_array().ok_or("target scale missing")?.as_slice() else {
            return Err("target scale must be a vector".into());
        };
        let transform = transform.ok_or("Blender target transform matrix is missing")?;
        let local_to_subject_f32 = f32_matrix(&transform["local_to_subject"])?;
        let subject_to_local_f32 = f32_matrix(&transform["subject_to_local"])?;
        operands.insert(
            "target".to_owned(),
            vec![SceneOperand {
                mesh: Mesh::default(),
                local_to_subject: None,
                blender_matrices_f32: Some(SceneOperandMatricesF32 {
                    world: None,
                    local_to_subject: Some(local_to_subject_f32),
                    subject_to_local: Some(subject_to_local_f32),
                }),
                camera: None,
                armature: false,
                object_scale: DVec3::new(
                    sx.as_f64().ok_or("target scale x is not numeric")?,
                    sy.as_f64().ok_or("target scale y is not numeric")?,
                    sz.as_f64().ok_or("target scale z is not numeric")?,
                ),
                selected_bones: Default::default(),
                deforming_bones: Default::default(),
                layer_selection: Default::default(),
            }],
        );
    }
    Ok(evaluate_with_context(&mesh, &modifier(case)?, &operands, None, DMat4::IDENTITY, "TestObject", &AttributeLayerSelection::default())?)
}

fn f32_matrix(value: &Value) -> Result<[f32; 16], Box<dyn Error>> {
    let columns = value.as_array().ok_or("transform matrix columns are missing")?;
    if columns.len() != 4 {
        return Err("transform matrix must have four columns".into());
    }
    let mut result = [0.0_f32; 16];
    for (column, values) in columns.iter().enumerate() {
        let values = values.as_array().ok_or("transform matrix column is malformed")?;
        if values.len() != 4 {
            return Err("transform matrix column must have four rows".into());
        }
        for (row, component) in values.iter().enumerate() {
            result[column * 4 + row] = component
                .as_f64()
                .ok_or("transform matrix component is not numeric")? as f32;
        }
    }
    Ok(result)
}

fn result_faces(mesh: &Mesh) -> Result<Vec<Value>, Box<dyn Error>> {
    let state = mesh.attributes.get("custom_normal").ok_or("canonical custom-normal state is missing")?;
    if state.get("type").and_then(Value::as_str) != Some("normal_code")
        || state.get("values").is_none()
    {
        return Err("modifier did not write allocated canonical custom-normal codes".into());
    }
    let normals = potter_core::geom::corner_normal_vectors(mesh)?;
    let tags = mesh
        .attributes
        .get("corner_tag")
        .and_then(|value| value.get("values"))
        .and_then(Value::as_object)
        .ok_or("corner tag layer is missing")?;
    let colors = mesh
        .attributes
        .get("corner_color")
        .and_then(|value| value.get("values"))
        .and_then(Value::as_object)
        .ok_or("corner color layer is missing")?;
    let uvs = crate_uv_values(mesh);
    let mut corner = 0;
    mesh.faces
        .iter()
        .map(|face| {
            let end = corner + face.vertices.len();
            let face_normals = normals[corner..end]
                .iter()
                .map(|normal| json!(normal.to_array()))
                .collect::<Vec<_>>();
            corner = end;
            Ok(json!({
                "vertices":face.vertices,
                "normals":face_normals,
                "tags":tags.get(&format!("f{}",face.id)).ok_or("face tag row is missing")?,
                "colors":colors.get(&format!("f{}",face.id)).ok_or("face color row is missing")?,
                "uvs":uvs.get(&format!("f{}",face.id)).ok_or("face UV row is missing")?,
            }))
        })
        .collect()
}

fn crate_uv_values(mesh: &Mesh) -> BTreeMap<String, Value> {
    let mut values = BTreeMap::new();
    for layer in mesh
        .attributes
        .get("uv_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for entry in layer.get("faces").and_then(Value::as_array).into_iter().flatten() {
            let Some(face_id) = entry.get("face_id").and_then(Value::as_u64) else { continue };
            if let Some(uv) = entry.get("uv") {
                values.insert(format!("f{face_id}"), uv.clone());
            }
        }
    }
    values
}

fn assert_faces_match(case_name: &str, actual: &[Value], expected: &[Value]) -> Result<(), Box<dyn Error>> {
    assert_eq!(actual.len(), expected.len(), "{case_name}: face count");
    let mut max_angle = 0.0_f64;
    for (face_index, (actual_face, expected_face)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual_face["vertices"], expected_face["vertices"], "{case_name}: face {face_index} corner order");
        assert_eq!(actual_face["tags"], expected_face["tags"], "{case_name}: face {face_index} corner attributes");
        assert_eq!(actual_face["colors"], expected_face["colors"], "{case_name}: face {face_index} color corner order");
        assert_eq!(actual_face["uvs"], expected_face["uvs"], "{case_name}: face {face_index} UV corner order");
        let actual_normals = actual_face["normals"].as_array().ok_or("actual normals are missing")?;
        let expected_normals = expected_face["normals"].as_array().ok_or("oracle normals are missing")?;
        assert_eq!(actual_normals.len(), expected_normals.len(), "{case_name}: face {face_index} normal count");
        for (actual, expected) in actual_normals.iter().zip(expected_normals) {
            let actual = vector(actual)?;
            let expected = vector(expected)?;
            assert!((actual.length() - 1.0).abs() < 1.0e-3, "{case_name}: non-unit native normal {actual:?}");
            assert!((expected.length() - 1.0).abs() < 1.0e-3, "{case_name}: non-unit Blender normal {expected:?}");
            let angle = actual.normalize().angle_between(expected.normalize());
            max_angle = max_angle.max(angle);
        }
    }
    eprintln!("Blender normal parity {case_name}: max corner-normal angle {max_angle:.8e} rad");
    assert!(max_angle <= 2.0e-4, "{case_name}: maximum corner-normal angular error {max_angle}");
    Ok(())
}

fn vector(value: &Value) -> Result<DVec3, Box<dyn Error>> {
    let [x, y, z] = value.as_array().ok_or("normal is not a vector")?.as_slice() else {
        return Err("normal must have three components".into());
    };
    Ok(DVec3::new(
        x.as_f64().ok_or("normal x is not numeric")?,
        y.as_f64().ok_or("normal y is not numeric")?,
        z.as_f64().ok_or("normal z is not numeric")?,
    ))
}

#[test]
fn normal_modifiers_match_blender_522_per_corner_cases() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender 5.2.2 normal parity: Blender unavailable");
        return Ok(());
    };
    let temp = tempdir()?;
    let root = fs::canonicalize(temp.path())?;
    let cases = blender_cases()?;
    let oracle = blender_probe(&blender, &root, &cases)?;
    assert_eq!(oracle.as_object().map(serde_json::Map::len), Some(cases.len() + 1));
    let transform_matrices = oracle
        .get("_transform_matrices")
        .and_then(Value::as_object)
        .ok_or("Blender target transform matrices are missing")?;
    for case in &cases {
        let name = case["name"].as_str().ok_or("case name missing")?;
        let actual = result_faces(&evaluate_case(case, transform_matrices.get(name))?)?;
        let expected = oracle[name]
            .as_array()
            .ok_or("Blender oracle face rows are missing")?;
        assert_faces_match(name, &actual, expected)?;
    }
    Ok(())
}

#[test]
fn weighted_normal_uses_blenders_ranked_weight_curve() -> Result<(), Box<dyn Error>> {
    let mesh = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::new(2.0, 0.0, 0.0), DVec3::new(0.0, 2.0, 0.0), DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 0.0, 2.0)],
        vec![vec![0, 1, 2], vec![0, 3, 4]],
    )?;
    let modifier = Modifier {
        id: potter_core::model::Id::new("weighted_normal")?,
        modifier_type: "weighted_normal".to_owned(),
        name: "Ranked weight curve".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: json!({"mode":"FACE_AREA","weight":80,"thresh":0.01}).as_object().ok_or("params missing")?.clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    };
    let result = evaluate_with_context(&mesh, &modifier, &BTreeMap::new(), None, DMat4::IDENTITY, "TestObject", &AttributeLayerSelection::default())?;
    let normals = potter_core::geom::corner_normal_vectors(&result)?;
    let actual = normals[0];
    let expected = DVec3::new(1.0 / 15.0, 0.0, 2.0).normalize();
    assert!(actual.angle_between(expected) < 2.0e-4, "expected {expected:?}, got {actual:?}");
    Ok(())
}

#[test]
fn weighted_normal_threshold_includes_exact_compare_boundary() -> Result<(), Box<dyn Error>> {
    let mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 3.5),
            DVec3::new(0.0, 0.0, 3.0),
            DVec3::new(1.0, 0.0, 0.0),
        ],
        vec![vec![0, 1, 2], vec![0, 3, 4], vec![0, 5, 6]],
    )?;
    let modifier = Modifier {
        id: potter_core::model::Id::new("weighted_threshold")?,
        modifier_type: "weighted_normal".to_owned(),
        name: "Inclusive threshold boundary".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: json!({"mode":"FACE_AREA","weight":80,"thresh":0.25}).as_object().ok_or("params missing")?.clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    };
    let result = evaluate_with_context(&mesh, &modifier, &BTreeMap::new(), None, DMat4::IDENTITY, "TestObject", &AttributeLayerSelection::default())?;
    let actual = potter_core::geom::corner_normal_vectors(&result)?[0];
    let expected = DVec3::new(1.75, 0.1, 2.0).normalize();
    assert!(
        actual.angle_between(expected) < 2.0e-4,
        "inclusive compare_ff boundary: expected {expected:?}, got {actual:?}"
    );
    Ok(())
}


#[test]
fn normal_edit_direction_uses_target_minus_vertex() -> Result<(), Box<dyn Error>> {
    let mesh = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::Y],
        vec![vec![0, 1, 2]],
    )?;
    let modifier = Modifier {
        id: potter_core::model::Id::new("directional")?,
        modifier_type: "normal_edit".to_owned(),
        name: "Directional sign".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: json!({"mode":"DIRECTIONAL","mix_mode":"COPY","mix_factor":1.0,"mix_limit":std::f64::consts::PI,"target":"target","use_direction_parallel":false,"no_polynors_fix":true}).as_object().ok_or("params missing")?.clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    };
    let local_to_subject_f32 = DMat4::from_translation(DVec3::new(0.0, 0.0, -2.0))
        .to_cols_array()
        .map(|value| value as f32);
    let subject_to_local_f32 = DMat4::from_translation(DVec3::new(0.0, 0.0, 2.0))
        .to_cols_array()
        .map(|value| value as f32);
    let mut operands = BTreeMap::new();
    operands.insert("target".to_owned(), vec![SceneOperand {
        mesh: Mesh::default(),
        local_to_subject: None,
        blender_matrices_f32: Some(SceneOperandMatricesF32 {
            world: None,
            local_to_subject: Some(local_to_subject_f32),
            subject_to_local: Some(subject_to_local_f32),
        }),
        camera: None,
        armature: false,
        object_scale: DVec3::ONE,
        selected_bones: Default::default(),
        deforming_bones: Default::default(),
        layer_selection: Default::default(),
    }]);
    let result = evaluate_with_context(&mesh, &modifier, &operands, None, DMat4::IDENTITY, "TestObject", &AttributeLayerSelection::default())?;
    let normal = potter_core::geom::corner_normal_vectors(&result)?[0];
    assert!(normal.z < -0.999_999, "direction must point from vertex toward target: {normal:?}");
    Ok(())
}

#[test]
fn cli_applies_and_inspects_both_normal_modifiers() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let operations = directory.path().join("operations.json");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .args(["--json"])
        .output()?;
    assert!(initialized.status.success(), "{}", String::from_utf8_lossy(&initialized.stdout));
    fs::write(&operations, serde_json::to_vec(&json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            {"op":"node.create","id":"body","kind":"plane","params":{"size":2.0}},
            {"op":"node.create","id":"target","kind":"empty","transform":{"translation":[0.0,0.0,-2.0],"scale":[2.0,0.5,1.5]}},
            {"op":"modifier.create","target":{"id":"body"},"id":"weighted","type":"weighted_normal","params":{"mode":"FACE_AREA","weight":80,"keep_sharp":false,"use_face_influence":false}},
            {"op":"modifier.create","target":{"id":"body"},"id":"normal_edit","type":"normal_edit","params":{"mode":"DIRECTIONAL","target":"target","mix_mode":"ADD","mix_factor":0.5,"mix_limit":1.0,"use_direction_parallel":false,"no_polynors_fix":true}}
        ]
    }))?)?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&operations)
        .args(["--json"])
        .output()?;
    assert!(applied.status.success(), "{}", String::from_utf8_lossy(&applied.stdout));
    let inspected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id", "body", "--json"])
        .output()?;
    assert!(inspected.status.success(), "{}", String::from_utf8_lossy(&inspected.stdout));
    let inspected: Value = serde_json::from_slice(&inspected.stdout)?;
    let item = &inspected["result"]["items"][0];
    let geometry = &item["evaluated_geometry"];
    let modifiers = item["modifiers"].as_array().ok_or("inspected modifier stack missing")?;
    assert_eq!(modifiers.len(), 2);
    assert!(modifiers.iter().any(|modifier| modifier["id"] == "weighted"));
    assert!(modifiers.iter().any(|modifier| modifier["id"] == "normal_edit"));
    assert_eq!(geometry["face_count"], 1);
    assert_eq!(geometry["vertex_count"], 4);
    Ok(())
}
