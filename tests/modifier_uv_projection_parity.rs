use std::{collections::BTreeMap, error::Error, fs, process::Command};

use glam::{DMat4, DVec3};
use potter_core::{
    error::ErrorCode,
    geom::{
        Mesh,
        modifiers::attributes::{SceneOperand, SceneOperandMatricesF32, evaluate_with_context},
    },
    model::{CameraData, CameraProjection, Id, Modifier, ModifierRuntime},
};
use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/blender_script.rs"]
mod blender_script;

use blender_file::blender_executable;
use blender_script::run_blender_script;

const UV_PROJECTION_ORACLE: &str = r#"
import bpy, json, math, sys
from mathutils import Matrix
root = sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
vertices = [(0,0,0),(2,0,0.2),(2,1,0),(0,1,-0.4),(3,0,0),(3,2,0),(3,2,1),(3,0,1)]
faces = [(0,1,2,3),(4,5,6,7)]
uv_values = [(0.1,0.2),(0.8,0.15),(0.9,0.7),(0.25,0.85),(0.5,0.6),(0.2,0.3),(0.4,0.9),(0.7,0.4)]
def matrix(location, rotation, scale):
    return Matrix.Translation(location) @ Matrix.Rotation(rotation[0], 4, 'X') @ Matrix.Rotation(rotation[1], 4, 'Y') @ Matrix.Rotation(rotation[2], 4, 'Z') @ Matrix.Diagonal((*scale, 1.0))
def mat_cols(mat):
    return [float(mat[row][col]) for col in range(4) for row in range(4)]
subject_transform = matrix((0.4,-0.7,0.6),(0.2,-0.3,0.4),(1.2,0.8,1.1))
def make_subject(name):
    mesh = bpy.data.meshes.new(name + 'Mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name='UVMap')
    for loop, value in zip(mesh.loops, uv_values):
        uv.data[loop.index].uv = value
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = subject_transform.copy()
    return obj
def make_camera(name, projection, location, rotation, scale, lens=42.0, ortho=4.1, fit='AUTO', shift=(0.12,-0.08)):
    data = bpy.data.cameras.new(name + 'Data')
    data.type = projection
    data.lens = lens
    data.ortho_scale = ortho
    data.sensor_fit = fit
    data.sensor_width = 36.0
    data.sensor_height = 24.0
    data.shift_x, data.shift_y = shift
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = matrix(location, rotation, scale)
    return obj
def make_empty(name, location, rotation, scale):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = matrix(location, rotation, scale)
    return obj
def read_uv(obj, name='UVMap'):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    layer = mesh.uv_layers.get(name)
    values = [[float(layer.data[loop.index].uv.x), float(layer.data[loop.index].uv.y)] for loop in mesh.loops]
    evaluated.to_mesh_clear()
    return values
def project_case(key, projectors, count, aspect, scale, layer='UVMap'):
    obj = make_subject('Subject_' + key)
    mod = obj.modifiers.new('Project', 'UV_PROJECT')
    mod.uv_layer = layer
    mod.projector_count = count
    mod.aspect_x, mod.aspect_y = aspect
    mod.scale_x, mod.scale_y = scale
    for index, projector in enumerate(projectors):
        if projector is not None:
            mod.projectors[index].object = projector
    return read_uv(obj, layer)

persp = make_camera('Perspective', 'PERSP', (1.1,0.4,4.3),(0.1,0.12,-0.2),(1.4,0.7,1.2),lens=42.0,fit='AUTO',shift=(0.12,-0.08))
ortho = make_camera('Orthographic', 'ORTHO', (-0.5,0.8,3.2),(0.15,0.1,0.25),(0.8,1.3,1.0),ortho=4.1,fit='VERTICAL',shift=(-0.07,0.11))
pano = make_camera('Panorama', 'PANO', (0.3,0.5,3.0),(-0.2,0.25,0.3),(1.5,0.7,1.2),ortho=5.5,fit='AUTO',shift=(0.06,-0.09))
plain = make_empty('PlainProjector', (3.0,1.0,0.2),(0.0,math.pi/2,0.0),(1.1,0.9,1.0))
normal_angle = math.radians(75.0)
normal_tangent = (math.cos(normal_angle), 0.0, -math.sin(normal_angle))
normal_mesh = bpy.data.meshes.new('ScaledNormalMesh')
normal_mesh.from_pydata(
    [(0.0,0.0,0.0), normal_tangent,
     (normal_tangent[0],1.0,normal_tangent[2]), (0.0,1.0,0.0)],
    [], [(0,1,2,3)])
normal_mesh.update()
normal_uv = normal_mesh.uv_layers.new(name='UVMap')
for loop, value in zip(normal_mesh.loops, [(0.1,0.2),(0.8,0.15),(0.9,0.7),(0.25,0.85)]):
    normal_uv.data[loop.index].uv = value
normal_subject = bpy.data.objects.new('ScaledNormalSubject', normal_mesh)
bpy.context.scene.collection.objects.link(normal_subject)
high_normal = make_empty('HighScaleNormal',(0.0,0.0,3.0),(0.0,0.0,0.0),(1.0,1.0,4.0))
aligned_normal = make_empty('AlignedLowScaleNormal',(2.0,0.0,1.0),(0.0,normal_angle,0.0),(1.0,1.0,0.5))
normal_project = normal_subject.modifiers.new('ScaledNormalProject','UV_PROJECT')
normal_project.projector_count = 2
normal_project.projectors[0].object = high_normal
normal_project.projectors[1].object = aligned_normal
normal_project.uv_layer = 'UVMap'
scaled_normal_selection = read_uv(normal_subject)

result = {
    'perspective': project_case('perspective',[persp],1,(1.7,1.2),(0.8,1.3)),
    'orthographic': project_case('orthographic',[ortho],1,(1.2,2.0),(0.7,1.4)),
    'panorama': project_case('panorama',[pano],1,(2.1,1.2),(1.3,0.7)),
    'projector_count_with_null_slot': project_case('multi',[pano,None,plain],3,(1.0,1.0),(1.0,1.0)),
    'created_layer': project_case('created',[persp],1,(1.0,1.0),(1.0,1.0),'CreatedUV'),
    'scaled_projector_normal_selection': scaled_normal_selection,
    'matrices': {
        'subject':mat_cols(subject_transform),
        'perspective':mat_cols(persp.matrix_world),
        'orthographic':mat_cols(ortho.matrix_world),
        'panorama':mat_cols(pano.matrix_world),
        'plain':mat_cols(plain.matrix_world),
        'high_normal':mat_cols(high_normal.matrix_world),
        'aligned_normal':mat_cols(aligned_normal.matrix_world),
    },
}
with open(root + '/expected_uv_project.json','w') as file:
    json.dump(result,file)
"#;

const UV_WARP_ORACLE: &str = r#"
import bpy, json, math, sys
from mathutils import Matrix
root = sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
vertices = [(0,0,0),(2,0,0),(2,1,0.5),(0,1,0),(1,2,-0.3)]
faces = [(0,1,2,3),(0,3,4)]
uv_values = [(0.1,0.2),(0.8,0.15),(0.9,0.7),(0.25,0.85),(0.5,0.6),(0.2,0.3),(0.4,0.9)]
def matrix(location, rotation, scale):
    return Matrix.Translation(location) @ Matrix.Rotation(rotation[0], 4, 'X') @ Matrix.Rotation(rotation[1], 4, 'Y') @ Matrix.Rotation(rotation[2], 4, 'Z') @ Matrix.Diagonal((*scale, 1.0))
def make_mesh(name, extra_layer=False):
    mesh = bpy.data.meshes.new(name + 'Mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name='UVMap')
    other = mesh.uv_layers.new(name='Secondary') if extra_layer else None
    for index, loop in enumerate(mesh.loops):
        uv.data[loop.index].uv = uv_values[index]
        if other:
            other.data[loop.index].uv = (uv_values[index][0] * 0.3 + 0.1, uv_values[index][1] * 0.7 - 0.2)
    mesh.uv_layers.active = uv
    return mesh
def subject(name, extra_layer=False, empty_group=False):
    obj = bpy.data.objects.new(name, make_mesh(name, extra_layer))
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = matrix((0.3,-0.2,0.5),(0.1,-0.2,0.4),(1.1,0.9,1.2))
    group = obj.vertex_groups.new(name='W')
    if not empty_group:
        for i, weight in enumerate((0.0,0.25,0.5,0.75,1.0)):
            group.add([i], weight, 'REPLACE')
    return obj
def read_uv(obj, layer='UVMap'):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    values = [[float(mesh.uv_layers[layer].data[loop.index].uv.x),float(mesh.uv_layers[layer].data[loop.index].uv.y)] for loop in mesh.loops]
    evaluated.to_mesh_clear()
    return values
def mat_cols(mat):
    return [float(mat[row][col]) for col in range(4) for row in range(4)]
def apply_case(name, values, from_obj=None, to_obj=None, layer='UVMap', empty_group=False):
    obj = subject(name, extra_layer=True, empty_group=empty_group)
    mod = obj.modifiers.new('Warp', 'UV_WARP')
    mod.uv_layer = layer
    for key, value in values.items():
        setattr(mod, key, value)
    mod.object_from = from_obj
    mod.object_to = to_obj
    return read_uv(obj, layer)
def apply_active_selector_case(name, selector, values):
    obj = subject(name, extra_layer=True)
    render_layer = obj.data.uv_layers.new(name='RenderOnly')
    render_values = [(0.91,0.02),(0.82,0.04),(0.73,0.06),(0.64,0.08),(0.55,0.10),(0.46,0.12),(0.37,0.14)]
    for loop, value in zip(obj.data.loops, render_values):
        render_layer.data[loop.index].uv = value
    obj.data.uv_layers.active = obj.data.uv_layers['Secondary']
    render_layer.active_render = True
    mod = obj.modifiers.new('Warp', 'UV_WARP')
    mod.uv_layer = selector
    for key, value in values.items():
        setattr(mod, key, value)
    return {layer: read_uv(obj, layer) for layer in ('UVMap','Secondary','RenderOnly')}
source = bpy.data.objects.new('Source', None); bpy.context.scene.collection.objects.link(source)
source.matrix_world = matrix((1.2,0.4,-0.3),(0.9,0.0,0.0),(1.3,0.8,1.1))
dest = bpy.data.objects.new('Destination', None); bpy.context.scene.collection.objects.link(dest)
dest.matrix_world = matrix((-0.7,1.1,0.2),(0.0,-0.5,0.0),(0.9,1.2,1.0))
arm_data = bpy.data.armatures.new('ArmatureData')
arm = bpy.data.objects.new('Armature', arm_data); bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
for name, y in (('BoneA',0.0),('BoneB',1.0)):
    bone = arm_data.edit_bones.new(name); bone.head = (0,y,0); bone.tail = (0.5,y+0.5,0.2)
bpy.ops.object.mode_set(mode='OBJECT')
arm.matrix_world = matrix((0.5,-0.4,0.1),(0.0,0.0,0.3),(1.0,1.0,1.0))
pose_a = arm.pose.bones['BoneA']; pose_a.rotation_mode='XYZ'; pose_a.location=(0.4,-0.3,0.2); pose_a.rotation_euler=(0.2,-0.5,0.7); pose_a.scale=(1.2,0.9,1.0)
pose_b = arm.pose.bones['BoneB']; pose_b.location=(-0.2,0.5,0.1); pose_b.rotation_euler=(0.1,0.3,-0.4)
bpy.context.view_layer.update()
subject_matrix = matrix((0.3,-0.2,0.5),(0.1,-0.2,0.4),(1.1,0.9,1.2))
base = {'center':(0.3,-0.2),'offset':(0.1,0.4),'scale':(1.5,0.7),'rotation':0.6,'axis_u':'X','axis_v':'Y'}
def warp_matrix(src_world, dst_world, params):
    axis_u = {'X':0,'Y':1,'Z':2}[params.get('axis_u','X')]
    axis_v = {'X':0,'Y':1,'Z':2}[params.get('axis_v','Y')]
    center = params.get('center',(0.0,0.0))
    offset = params.get('offset',(0.0,0.0))
    scale = params.get('scale',(1.0,1.0))
    rotation = params.get('rotation',0.0)
    if src_world is not None and dst_world is not None:
        result = dst_world.inverted() @ src_world
        if center != (0.0,0.0):
            center3 = [0.0,0.0,0.0]
            center3[axis_u] = center[0]
            center3[axis_v] = center[1]
            center_matrix = Matrix.Translation(center3)
            result = center_matrix @ result @ center_matrix.inverted()
        shuffle = [[0.0]*4 for _ in range(4)]
        shuffle[0][axis_u] = 1.0
        shuffle[1][axis_v] = 1.0
        shuffle[3][3] = 1.0
        shuffle = Matrix(shuffle)
        result = shuffle @ result @ shuffle.transposed()
    else:
        result = Matrix.Identity(4)
    result = (result @ Matrix.Translation((center[0],center[1],0.0))
              @ Matrix.Diagonal((scale[0],scale[1],1.0,1.0))
              @ Matrix.Rotation(rotation,4,'Z')
              @ Matrix.Translation((offset[0],offset[1],0.0))
              @ Matrix.Translation((-center[0],-center[1],0.0)))
    return mat_cols(result)

bone_a_world = arm.matrix_world @ pose_a.matrix
bone_b_world = arm.matrix_world @ pose_b.matrix
result = {
    'transform_only': apply_case('TransformOnly',base),
    'object_pair': apply_case('ObjectPair',base,source,dest),
    'object_group': apply_case('ObjectGroup',{**base,'vertex_group':'W'},source,dest),
    'object_group_inverted': apply_case('ObjectGroupInvert',{**base,'vertex_group':'W','invert_vertex_group':True},source,dest),
    'empty_group': apply_case('EmptyGroup',{**base,'vertex_group':'W'},source,dest,empty_group=True),
    'empty_group_inverted': apply_case('EmptyGroupInvert',{**base,'vertex_group':'W','invert_vertex_group':True},source,dest,empty_group=True),
    'axes_and_named_layer': apply_case('AxesLayer',{'center':(0.1,0.25),'offset':(-0.2,0.15),'scale':(0.6,1.4),'rotation':-0.8,'axis_u':'Y','axis_v':'Z'},source,dest,'Secondary'),
    'active_layer_empty_selector': apply_active_selector_case('ActiveEmptySelector','',base),
    'active_layer_missing_selector': apply_active_selector_case('ActiveMissingSelector','Missing',base),
    'bone_pair': apply_case('BonePair',base,arm,arm),
    'bone_pair_axes': apply_case('BonePairAxes',{'center':(0.1,0.25),'offset':(-0.2,0.15),'scale':(0.6,1.4),'rotation':-0.8,'axis_u':'Y','axis_v':'Z'},arm,arm),
    'bone_from_missing': apply_case('BoneFromMissing',{**base,'bone_from':'Missing'},arm,dest),
    'bone_to_missing': apply_case('BoneToMissing',{**base,'bone_to':'Missing'},source,arm),
    'bone_from_empty': apply_case('BoneFromEmpty',{**base,'bone_from':''},arm,dest),
    'warp_matrices': {
        'transform_only':warp_matrix(None,None,base),
        'object_pair':warp_matrix(source.matrix_world,dest.matrix_world,base),
        'object_group':warp_matrix(source.matrix_world,dest.matrix_world,base),
        'object_group_inverted':warp_matrix(source.matrix_world,dest.matrix_world,base),
        'empty_group':warp_matrix(source.matrix_world,dest.matrix_world,base),
        'empty_group_inverted':warp_matrix(source.matrix_world,dest.matrix_world,base),
        'axes_and_named_layer':warp_matrix(source.matrix_world,dest.matrix_world,{'center':(0.1,0.25),'offset':(-0.2,0.15),'scale':(0.6,1.4),'rotation':-0.8,'axis_u':'Y','axis_v':'Z'}),
        'active_layer_empty_selector':warp_matrix(None,None,base),
        'active_layer_missing_selector':warp_matrix(None,None,base),
        'bone_pair':warp_matrix(bone_a_world,bone_b_world,base),
        'bone_pair_axes':warp_matrix(bone_a_world,bone_b_world,{'center':(0.1,0.25),'offset':(-0.2,0.15),'scale':(0.6,1.4),'rotation':-0.8,'axis_u':'Y','axis_v':'Z'}),
        'bone_from_missing':warp_matrix(arm.matrix_world,dest.matrix_world,base),
        'bone_to_missing':warp_matrix(source.matrix_world,arm.matrix_world,base),
        'bone_from_empty':warp_matrix(arm.matrix_world,dest.matrix_world,base),
    },
    'matrices': {
        'subject':mat_cols(subject_matrix),
        'source':mat_cols(source.matrix_world),
        'destination':mat_cols(dest.matrix_world),
        'armature':mat_cols(arm.matrix_world),
        'source_subject':mat_cols(subject_matrix.inverted() @ source.matrix_world),
        'destination_subject':mat_cols(subject_matrix.inverted() @ dest.matrix_world),
        'armature_subject':mat_cols(subject_matrix.inverted() @ arm.matrix_world),
    },
}
with open(root + '/expected_uv_warp.json','w') as file:
    json.dump(result,file)
"#;

fn test_modifier(modifier_type: &str, params: Value) -> Result<Modifier, Box<dyn Error>> {
    Ok(Modifier {
        id: Id::new("uv_parity_modifier")?,
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        params: params.as_object().ok_or("modifier parameters must be an object")?.clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

fn set_uv_layers(
    mesh: &mut Mesh,
    layers: &[(&str, Vec<[f64; 2]>)],
    active: &str,
    render: &str,
) {
    let stored = layers
        .iter()
        .map(|(name, blender_values)| {
            let mut offset = 0;
            let faces = mesh
                .faces
                .iter()
                .map(|face| {
                    let end = offset + face.vertices.len();
                    let values = blender_values[offset..end]
                        .iter()
                        .map(|[u, v]| [*u, 1.0 - *v])
                        .collect::<Vec<_>>();
                    offset = end;
                    json!({"face_id":face.id,"uv":values})
                })
                .collect::<Vec<_>>();
            assert_eq!(offset, blender_values.len());
            json!({"name":name,"faces":faces})
        })
        .collect::<Vec<_>>();
    mesh.attributes.insert(
        "uv_layers".to_owned(),
        json!({"layers":stored,"active":active,"render":render}),
    );
}


fn test_mesh() -> Result<Mesh, Box<dyn Error>> {
    let vertices = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(2.0, 0.0, 0.2),
        DVec3::new(2.0, 1.0, 0.0),
        DVec3::new(0.0, 1.0, -0.4),
        DVec3::new(3.0, 0.0, 0.0),
        DVec3::new(3.0, 2.0, 0.0),
        DVec3::new(3.0, 2.0, 1.0),
        DVec3::new(3.0, 0.0, 1.0),
    ];
    let mut mesh = Mesh::from_positions_and_faces(vertices, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]])?;
    set_uv_layers(
        &mut mesh,
        &[(
            "UVMap",
            vec![
                [0.1,0.2],[0.8,0.15],[0.9,0.7],[0.25,0.85],
                [0.5,0.6],[0.2,0.3],[0.4,0.9],[0.7,0.4],
            ],
        )],
        "UVMap",
        "UVMap",
    );
    Ok(mesh)
}

fn operand(world_matrix: DMat4, camera: Option<CameraData>, armature: bool) -> SceneOperand {
    SceneOperand {
        mesh: Mesh::default(),
        local_to_subject: None,
        blender_matrices_f32: Some(SceneOperandMatricesF32 {
            world: Some(world_matrix.to_cols_array().map(|value| value as f32)),
            local_to_subject: None,
            subject_to_local: None,
        }),
        camera,
        armature,
        object_scale: DVec3::ONE,
        selected_bones: Default::default(),
        deforming_bones: Default::default(),
        layer_selection: Default::default(),
    }
}

fn camera_data(
    projection: CameraProjection,
    sensor_fit: &str,
    ortho_scale: f64,
    lens_mm: f64,
    shift: [f64; 2],
) -> CameraData {
    let mut data = CameraData::default();
    data.projection = projection;
    data.sensor_fit = sensor_fit.to_owned();
    data.ortho_scale = ortho_scale;
    data.lens_mm = lens_mm;
    data.shift = shift;
    data
}

fn matrix_from_oracle(value: &Value) -> Result<DMat4, Box<dyn Error>> {
    let values = value.as_array().ok_or("Blender matrix is not an array")?;
    let columns = values
        .iter()
        .map(|value| value.as_f64().ok_or("Blender matrix component missing"))
        .collect::<Result<Vec<_>, _>>()?;
    let columns: [f64; 16] = columns
        .try_into()
        .map_err(|_| "Blender matrix does not have sixteen components")?;
    Ok(DMat4::from_cols_array(&columns))
}
fn runtime_uv_warp_matrix(value: &Value) -> Result<[[f32; 4]; 4], Box<dyn Error>> {
    let values = value.as_array().ok_or("Blender UV Warp matrix is not an array")?;
    let flat = values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or("Blender UV Warp matrix component missing")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let flat: [f32; 16] = flat
        .try_into()
        .map_err(|_| "Blender UV Warp matrix does not have sixteen components")?;
    Ok(std::array::from_fn(|column| {
        std::array::from_fn(|row| flat[column * 4 + row])
    }))
}

fn evaluate_project_case(
    mesh: &Mesh,
    subject_world: DMat4,
    projectors: Vec<(&str, DMat4, Option<CameraData>)>,
    params: Value,
) -> Result<Mesh, Box<dyn Error>> {
    let mut operands = Vec::with_capacity(projectors.len());
    for (id, world, camera) in projectors {
        let resolved = operand(world, camera, false);
        operands.push((id.to_owned(), resolved));
    }
    let mut map = BTreeMap::new();
    map.insert(
        "projectors".to_owned(),
        operands.into_iter().map(|(_, value)| value).collect(),
    );
    let modifier = test_modifier("uv_project", params)?;
    Ok(evaluate_with_context(mesh, &modifier, &map, None, subject_world)?)
}

fn assert_uv_parity(name: &str, actual_mesh: &Mesh, expected: &Value, layer_name: &str) -> Result<f64, Box<dyn Error>> {
    let layer = actual_mesh
        .attributes
        .get("uv_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .and_then(|layers| {
            layers
                .iter()
                .find(|layer| layer.get("name").and_then(Value::as_str) == Some(layer_name))
        })
        .ok_or("native canonical UV layer missing")?;
    let rows = layer
        .get("faces")
        .and_then(Value::as_array)
        .ok_or("native canonical UV face rows missing")?;
    let mut actual = Vec::new();
    for face in &actual_mesh.faces {
        let face_row = rows
            .iter()
            .find(|row| row.get("face_id").and_then(Value::as_u64) == Some(u64::from(face.id)))
            .ok_or("native canonical UV face row missing")?;
        let values = face_row
            .get("uv")
            .and_then(Value::as_array)
            .ok_or("native canonical UV row missing")?;
        assert_eq!(values.len(), face.vertices.len(), "{name}: native UV corner count differs");
        for pair in values {
            let pair = pair.as_array().ok_or("native UV pair missing")?;
            actual.push([
                pair.first().and_then(Value::as_f64).ok_or("native U missing")?,
                pair.get(1).and_then(Value::as_f64).ok_or("native V missing")?,
            ]);
        }
    }
    let expected = expected
        .as_array()
        .ok_or("Blender UV values missing")?
        .iter()
        .map(|pair| {
            let pair = pair.as_array().ok_or("Blender UV pair missing")?;
            Ok::<_, Box<dyn Error>>([
                pair.first().and_then(Value::as_f64).ok_or("Blender U missing")?,
                1.0 - pair.get(1).and_then(Value::as_f64).ok_or("Blender V missing")?,
            ])
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(actual.len(), expected.len(), "{name}: per-loop count differs");
    let max_error = actual
        .iter()
        .zip(&expected)
        .flat_map(|(a, b)| [(a[0] - b[0]).abs(), (a[1] - b[1]).abs()])
        .fold(0.0_f64, f64::max);
    assert!(max_error <= 3.0e-5, "{name}: max per-loop UV error {max_error}");
    Ok(max_error)
}

fn read_oracle(blender: &std::path::Path, script: &str, filename: &str) -> Result<Value, Box<dyn Error>> {
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    run_blender_script(blender, "modifier_uv_oracle.py", script, &root, "Blender UV oracle failed")?;
    Ok(serde_json::from_slice(&fs::read(root.join(filename))?)?)
}

#[test]
fn uv_project_matches_blender_camera_modes_aspect_scale_and_projector_list() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping UV modifier Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let expected = read_oracle(&blender, UV_PROJECTION_ORACLE, "expected_uv_project.json")?;
    let mesh = test_mesh()?;
    let subject = matrix_from_oracle(&expected["matrices"]["subject"])?;
    let persp_world = matrix_from_oracle(&expected["matrices"]["perspective"])?;
    let ortho_world = matrix_from_oracle(&expected["matrices"]["orthographic"])?;
    let pano_world = matrix_from_oracle(&expected["matrices"]["panorama"])?;
    let plain_world = matrix_from_oracle(&expected["matrices"]["plain"])?;
    let cases = [
        (
            "perspective",
            vec![("persp", persp_world, Some(camera_data(CameraProjection::Perspective,"AUTO",4.1,42.0,[0.12,-0.08])))],
            json!({"projectors":["persp"],"uv_layer":"UVMap","aspect_x":1.7,"aspect_y":1.2,"scale_x":0.8,"scale_y":1.3}),
            "UVMap",
        ),
        (
            "orthographic",
            vec![("ortho", ortho_world, Some(camera_data(CameraProjection::Orthographic,"VERTICAL",4.1,42.0,[-0.07,0.11])))],
            json!({"projectors":["ortho"],"uv_layer":"UVMap","aspect_x":1.2,"aspect_y":2.0,"scale_x":0.7,"scale_y":1.4}),
            "UVMap",
        ),
        (
            "panorama",
            vec![("pano", pano_world, Some(camera_data(CameraProjection::Panorama,"AUTO",5.5,42.0,[0.06,-0.09])))],
            json!({"projectors":["pano"],"uv_layer":"UVMap","aspect_x":2.1,"aspect_y":1.2,"scale_x":1.3,"scale_y":0.7}),
            "UVMap",
        ),
        (
            "projector_count_with_null_slot",
            vec![
                ("pano", pano_world, Some(camera_data(CameraProjection::Panorama,"AUTO",5.5,42.0,[0.06,-0.09]))),
                ("plain", plain_world, None),
            ],
            json!({"projectors":["pano",null,"plain"],"projector_count":3,"uv_layer":"UVMap","aspect_x":1.0,"aspect_y":1.0,"scale_x":1.0,"scale_y":1.0}),
            "UVMap",
        ),
        (
            "created_layer",
            vec![("persp", persp_world, Some(camera_data(CameraProjection::Perspective,"AUTO",4.1,42.0,[0.12,-0.08])))],
            json!({"projectors":["persp"],"uv_layer":"CreatedUV","aspect_x":1.0,"aspect_y":1.0,"scale_x":1.0,"scale_y":1.0}),
            "CreatedUV",
        ),
    ];
    for (name, projectors, params, layer) in cases {
        let actual = evaluate_project_case(&mesh, subject, projectors, params)?;
        assert_uv_parity(name, &actual, &expected[name], layer)?;
    }
    // Blender resizes the projectors collection from `projector_count`; once imported,
    // the explicit projector array is the canonical slot list, including null slots.
    for count in [1, 5] {
        let actual = evaluate_project_case(
            &mesh,
            subject,
            vec![
                (
                    "pano",
                    pano_world,
                    Some(camera_data(
                        CameraProjection::Panorama,
                        "AUTO",
                        5.5,
                        42.0,
                        [0.06, -0.09],
                    )),
                ),
                ("plain", plain_world, None),
            ],
            json!({
                "projectors":["pano",null,"plain"],"projector_count":count,
                "uv_layer":"UVMap","aspect_x":1.0,"aspect_y":1.0,"scale_x":1.0,"scale_y":1.0
            }),
        )?;
        let name = if count < 3 {
            "projector_count_smaller_than_array"
        } else {
            "projector_count_larger_than_array"
        };
        assert_uv_parity(
            name,
            &actual,
            &expected["projector_count_with_null_slot"],
            "UVMap",
        )?;
    }
    let angle = 75.0_f64.to_radians();
    let tangent = DVec3::new(angle.cos(), 0.0, -angle.sin());
    let mut scaled_normal_mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            tangent,
            tangent + DVec3::Y,
            DVec3::Y,
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    set_uv_layers(
        &mut scaled_normal_mesh,
        &[("UVMap", vec![[0.1,0.2],[0.8,0.15],[0.9,0.7],[0.25,0.85]])],
        "UVMap",
        "UVMap",
    );
    let high_scale_projector = matrix_from_oracle(&expected["matrices"]["high_normal"])?;
    let aligned_projector = matrix_from_oracle(&expected["matrices"]["aligned_normal"])?;
    let scaled_normal = evaluate_project_case(
        &scaled_normal_mesh,
        DMat4::IDENTITY,
        vec![
            ("high_scale", high_scale_projector, None),
            ("aligned", aligned_projector, None),
        ],
        json!({"projectors":["high_scale","aligned"],"uv_layer":"UVMap","aspect_x":1.0,"aspect_y":1.0,"scale_x":1.0,"scale_y":1.0}),
    )?;
    assert_uv_parity(
        "scaled_projector_normal_selection",
        &scaled_normal,
        &expected["scaled_projector_normal_selection"],
        "UVMap",
    )?;
    Ok(())
}

#[test]
fn uv_warp_matches_blender_object_bone_group_and_axis_cases_per_loop() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping UV modifier Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let expected = read_oracle(&blender, UV_WARP_ORACLE, "expected_uv_warp.json")?;
    let source_world = matrix_from_oracle(&expected["matrices"]["source"])?;
    let dest_world = matrix_from_oracle(&expected["matrices"]["destination"])?;
    let subject_world = matrix_from_oracle(&expected["matrices"]["subject"])?;
    let arm_world = matrix_from_oracle(&expected["matrices"]["armature"])?;
    let mesh = {
        let mut mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO,DVec3::new(2.0,0.0,0.0),DVec3::new(2.0,1.0,0.5),DVec3::new(0.0,1.0,0.0),DVec3::new(1.0,2.0,-0.3)],
            vec![vec![0,1,2,3],vec![0,3,4]],
        )?;
        set_uv_layers(
            &mut mesh,
            &[("UVMap", vec![
                [0.1,0.2],[0.8,0.15],[0.9,0.7],[0.25,0.85],
                [0.5,0.6],[0.2,0.3],[0.4,0.9],
            ])],
            "UVMap",
            "UVMap",
        );
        mesh.attributes.insert("vertex_groups".to_owned(), json!({"W":{"v0":0.0,"v1":0.25,"v2":0.5,"v3":0.75,"v4":1.0}}));
        mesh
    };
    let base_params = json!({"center":[0.3,-0.2],"offset":[0.1,0.4],"scale":[1.5,0.7],"rotation":0.6,"axis_u":"X","axis_v":"Y","uv_layer":"UVMap"});
    let mut cases = Vec::new();
    cases.push(("transform_only", base_params.clone(), BTreeMap::new(), None, None));
    for (name, vgroup, invert, expected_key) in [
        ("object_pair",None,false,"object_pair"),
        ("object_group",Some("W"),false,"object_group"),
        ("object_group_inverted",Some("W"),true,"object_group_inverted"),
    ] {
        let mut params = base_params.as_object().ok_or("base UV Warp parameters are invalid")?.clone();
        params.insert("object_from".to_owned(), json!("source"));
        params.insert("object_to".to_owned(), json!("destination"));
        if let Some(group) = vgroup { params.insert("vertex_group".to_owned(), json!(group)); }
        params.insert("invert_vertex_group".to_owned(), json!(invert));
        let mut operands = BTreeMap::new();
        operands.insert("object_from".to_owned(), vec![operand(source_world, None, false)]);
        operands.insert("object_to".to_owned(), vec![operand(dest_world, None, false)]);
        cases.push((expected_key, Value::Object(params), operands, None, None));
        let _ = name;
    }
    let mut empty_group_mesh = mesh.clone();
    empty_group_mesh
        .attributes
        .insert("vertex_groups".to_owned(), json!({"W":{}}));
    let mut empty_group_operands = BTreeMap::new();
    empty_group_operands.insert(
        "object_from".to_owned(),
        vec![operand(source_world, None, false)],
    );
    empty_group_operands.insert(
        "object_to".to_owned(),
        vec![operand(dest_world, None, false)],
    );
    for (name, invert) in [("empty_group", false), ("empty_group_inverted", true)] {
        let mut params = base_params.as_object().ok_or("base UV Warp parameters are invalid")?.clone();
        params.insert("object_from".to_owned(), json!("source"));
        params.insert("object_to".to_owned(), json!("destination"));
        params.insert("vertex_group".to_owned(), json!("W"));
        params.insert("invert_vertex_group".to_owned(), json!(invert));
        cases.push((
            name,
            Value::Object(params),
            empty_group_operands.clone(),
            Some(empty_group_mesh.clone()),
            None,
        ));
    }
    let alt_params = json!({"center":[0.1,0.25],"offset":[-0.2,0.15],"scale":[0.6,1.4],"rotation":-0.8,"axis_u":"Y","axis_v":"Z","uv_layer":"Secondary"});
    let mut alt_mesh = mesh.clone();
    set_uv_layers(
        &mut alt_mesh,
        &[
            ("UVMap", vec![
                [0.1,0.2],[0.8,0.15],[0.9,0.7],[0.25,0.85],
                [0.5,0.6],[0.2,0.3],[0.4,0.9],
            ]),
            ("Secondary", vec![
                [0.13,-0.06],[0.34,-0.095],[0.37,0.29],[0.175,0.395],
                [0.25,0.22],[0.16,0.01],[0.22,0.43],
            ]),
        ],
        "UVMap",
        "UVMap",
    );
    let mut alt_operands = BTreeMap::new();
    alt_operands.insert("object_from".to_owned(), vec![operand(source_world, None, false)]);
    alt_operands.insert("object_to".to_owned(), vec![operand(dest_world, None, false)]);
    cases.push(("axes_and_named_layer", alt_params, alt_operands, Some(alt_mesh), Some("Secondary")));
    let mut active_layer_mesh = mesh.clone();
    set_uv_layers(
        &mut active_layer_mesh,
        &[
            ("UVMap", vec![
                [0.1,0.2],[0.8,0.15],[0.9,0.7],[0.25,0.85],
                [0.5,0.6],[0.2,0.3],[0.4,0.9],
            ]),
            ("Secondary", vec![
                [0.13,-0.06],[0.34,-0.095],[0.37,0.29],[0.175,0.395],
                [0.25,0.22],[0.16,0.01],[0.22,0.43],
            ]),
            ("RenderOnly", vec![
                [0.91,0.02],[0.82,0.04],[0.73,0.06],[0.64,0.08],
                [0.55,0.10],[0.46,0.12],[0.37,0.14],
            ]),
        ],
        "Secondary",
        "RenderOnly",
    );
    for (name, selector) in [
        ("active_layer_empty_selector", ""),
        ("active_layer_missing_selector", "Missing"),
    ] {
        let mut params = base_params.as_object().ok_or("base UV Warp parameters are invalid")?.clone();
        params.insert("uv_layer".to_owned(), json!(selector));
        let mut modifier = test_modifier("uv_warp", Value::Object(params))?;
        modifier.runtime.uv_warp_matrix = Some(runtime_uv_warp_matrix(
            &expected["warp_matrices"][name],
        )?);
        let result = evaluate_with_context(
            &active_layer_mesh,
            &modifier,
            &BTreeMap::new(),
            None,
            subject_world,
        )?;
        for layer in ["UVMap", "Secondary", "RenderOnly"] {
            assert_uv_parity(name, &result, &expected[name][layer], layer)?;
        }
    }

    let mut bone_operands = BTreeMap::new();
    bone_operands.insert("object_from".to_owned(), vec![operand(arm_world, None, true)]);
    bone_operands.insert("object_to".to_owned(), vec![operand(arm_world, None, true)]);
    let mut bone_params = base_params.as_object().ok_or("base UV Warp parameters are invalid")?.clone();
    bone_params.insert("object_from".to_owned(), json!("armature"));
    bone_params.insert("object_to".to_owned(), json!("armature"));
    bone_params.insert("bone_from".to_owned(), json!("BoneA"));
    bone_params.insert("bone_to".to_owned(), json!("BoneB"));
    cases.push(("bone_pair", Value::Object(bone_params), bone_operands.clone(), None, None));
    let mut missing_from_params = base_params.as_object().ok_or("base UV Warp parameters are invalid")?.clone();
    missing_from_params.insert("object_from".to_owned(), json!("armature"));
    missing_from_params.insert("object_to".to_owned(), json!("destination"));
    missing_from_params.insert("bone_from".to_owned(), json!("Missing"));
    let mut missing_from_operands = BTreeMap::new();
    missing_from_operands.insert("object_from".to_owned(), vec![operand(arm_world, None, true)]);
    missing_from_operands.insert("object_to".to_owned(), vec![operand(dest_world, None, false)]);
    cases.push(("bone_from_missing", Value::Object(missing_from_params), missing_from_operands.clone(), None, None));

    let mut missing_to_params = base_params.as_object().ok_or("base UV Warp parameters are invalid")?.clone();
    missing_to_params.insert("object_from".to_owned(), json!("source"));
    missing_to_params.insert("object_to".to_owned(), json!("armature"));
    missing_to_params.insert("bone_to".to_owned(), json!("Missing"));
    let mut missing_to_operands = BTreeMap::new();
    missing_to_operands.insert("object_from".to_owned(), vec![operand(source_world, None, false)]);
    missing_to_operands.insert("object_to".to_owned(), vec![operand(arm_world, None, true)]);
    cases.push(("bone_to_missing", Value::Object(missing_to_params), missing_to_operands, None, None));

    let mut empty_bone_params = base_params.as_object().ok_or("base UV Warp parameters are invalid")?.clone();
    empty_bone_params.insert("object_from".to_owned(), json!("armature"));
    empty_bone_params.insert("object_to".to_owned(), json!("destination"));
    empty_bone_params.insert("bone_from".to_owned(), json!(""));
    cases.push(("bone_from_empty", Value::Object(empty_bone_params), missing_from_operands.clone(), None, None));
    let mut bone_alt_params = json!({"center":[0.1,0.25],"offset":[-0.2,0.15],"scale":[0.6,1.4],"rotation":-0.8,"axis_u":"Y","axis_v":"Z","uv_layer":"UVMap","object_from":"armature","object_to":"armature","bone_from":"BoneA","bone_to":"BoneB"});
    cases.push(("bone_pair_axes", bone_alt_params.take(), bone_operands.clone(), None, None));

    for (name, params, operands, test_mesh_override, layer_override) in cases {
        let subject_mesh = test_mesh_override.as_ref().unwrap_or(&mesh);
        let layer = layer_override.unwrap_or("UVMap");
        let mut modifier = test_modifier("uv_warp", params)?;
        modifier.runtime.uv_warp_matrix = Some(runtime_uv_warp_matrix(
            &expected["warp_matrices"][name],
        )?);
        let result = evaluate_with_context(subject_mesh, &modifier, &operands, None, subject_world)?;
        assert_uv_parity(name, &result, &expected[name], layer)?;
    }
    let mut missing_runtime_params = base_params.as_object().ok_or("base UV Warp parameters are invalid")?.clone();
    missing_runtime_params.insert("object_from".to_owned(), json!("armature"));
    missing_runtime_params.insert("object_to".to_owned(), json!("armature"));
    missing_runtime_params.insert("bone_from".to_owned(), json!("BoneA"));
    missing_runtime_params.insert("bone_to".to_owned(), json!("BoneB"));
    let error = evaluate_with_context(
        &mesh,
        &test_modifier("uv_warp", Value::Object(missing_runtime_params))?,
        &bone_operands,
        None,
        subject_world,
    )
    .err()
    .ok_or("UV Warp accepted missing resolved bone matrices")?;
    assert_eq!(error.code, ErrorCode::EvaluationFailed);
    Ok(())
}

#[test]
fn cli_apply_inspect_runs_uv_project_uv_warp_and_mask() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let batch = directory.path().join("operations.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(init.status.success(), "{}", String::from_utf8_lossy(&init.stdout));
    let operations = json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            {"op":"node.create","id":"body","kind":"plane","params":{"size":2.0}},
            {"op":"node.create","id":"projector","kind":"empty","transform":{"translation":[0.0,0.0,4.0]}},
            {"op":"node.create","id":"from","kind":"empty","transform":{"translation":[0.3,0.2,0.0],"rotation_deg":[0.0,0.0,15.0]}},
            {"op":"node.create","id":"to","kind":"empty","transform":{"translation":[-0.2,0.1,0.0],"rotation_deg":[0.0,0.0,-10.0]}},
            {"op":"modifier.create","target":{"id":"body"},"id":"project","type":"uv_project","params":{"projectors":["projector",null],"projector_count":2,"uv_layer":"CLIUV","aspect_x":1.0,"aspect_y":1.0,"scale_x":1.0,"scale_y":1.0}},
            {"op":"modifier.apply","target":{"id":"body"},"id":"project"},
            {"op":"modifier.create","target":{"id":"body"},"id":"warp","type":"uv_warp","params":{"object_from":"from","object_to":"to","uv_layer":"CLIUV","center":[0.5,0.5],"offset":[0.1,-0.2],"scale":[1.1,0.8],"rotation":0.25}},
            {"op":"modifier.apply","target":{"id":"body"},"id":"warp"},
            {"op":"vertex_group.create","target":{"id":"body"},"id":"mask_group","name":"Mask"},
            {"op":"vertex_group.assign","target":{"id":"body"},"group_id":"mask_group","weights":[{"vertex_id":0,"weight":1.0},{"vertex_id":1,"weight":1.0},{"vertex_id":2,"weight":1.0},{"vertex_id":3,"weight":0.0}]},
            {"op":"modifier.create","target":{"id":"body"},"id":"mask","type":"mask","params":{"mode":"VERTEX_GROUP","vertex_group":"Mask","threshold":0.5}},
            {"op":"modifier.apply","target":{"id":"body"},"id":"mask"},
            {"op":"node.create","id":"render_body","kind":"plane","params":{"size":1.5},"transform":{"translation":[0.0,0.0,0.5]}},
            {"op":"camera.create","id":"render_camera","transform":{"translation":[0.0,0.0,4.0]},"projection":"orthographic","ortho_scale":4.0},
            {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"render_camera"}},
            {"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":16,"resolution_y":16,"samples":1,"seed":0,"max_bounces":0}},
            {"op":"modifier.create","target":{"id":"render_body"},"id":"render_project","type":"uv_project","params":{"projectors":["projector",null],"projector_count":2,"uv_layer":"RenderUV","aspect_x":1.0,"aspect_y":1.0,"scale_x":1.0,"scale_y":1.0}},
            {"op":"modifier.apply","target":{"id":"render_body"},"id":"render_project"},
            {"op":"modifier.create","target":{"id":"render_body"},"id":"render_warp","type":"uv_warp","params":{"object_from":"from","object_to":"to","uv_layer":"RenderUV","center":[0.5,0.5],"offset":[0.1,-0.2],"scale":[1.1,0.8],"rotation":0.25}},
            {"op":"modifier.apply","target":{"id":"render_body"},"id":"render_warp"}
        ]
    });
    fs::write(&batch, serde_json::to_vec(&operations)?)?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(applied.status.success(), "{}", String::from_utf8_lossy(&applied.stdout));
    let inspected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id","body","--json"])
        .output()?;
    assert!(inspected.status.success(), "{}", String::from_utf8_lossy(&inspected.stdout));
    let envelope: Value = serde_json::from_slice(&inspected.stdout)?;
    let geometry = &envelope["result"]["items"][0]["evaluated_geometry"];
    assert_eq!(geometry["vertex_count"], 3);
    assert_eq!(geometry["face_count"], 0);
    let render_body_inspect = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id", "render_body", "--json"])
        .output()?;
    assert!(
        render_body_inspect.status.success(),
        "{}",
        String::from_utf8_lossy(&render_body_inspect.stdout)
    );
    let render_body: Value = serde_json::from_slice(&render_body_inspect.stdout)?;
    let render_geometry = &render_body["result"]["items"][0]["evaluated_geometry"];
    assert!(render_geometry["vertex_count"].as_u64().unwrap_or(0) > 0);
    assert!(render_geometry["face_count"].as_u64().unwrap_or(0) > 0);

    let render_directory = directory.path().join("rendered");
    let rendered = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("render")
        .arg(&scene)
        .args(["--camera", "render_camera", "--format", "png", "--out"])
        .arg(&render_directory)
        .arg("--json")
        .output()?;
    assert!(
        rendered.status.success(),
        "{}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    assert!(render_directory.join("frame_0001.png").is_file());

    let glb_path = directory.path().join("uv_modifier_parity.glb");
    let exported = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("export")
        .arg(&scene)
        .args(["--format", "glb", "--out"])
        .arg(&glb_path)
        .arg("--json")
        .output()?;
    assert!(
        exported.status.success(),
        "{}",
        String::from_utf8_lossy(&exported.stdout)
    );
    assert!(glb_path.is_file());
    Ok(())
}
