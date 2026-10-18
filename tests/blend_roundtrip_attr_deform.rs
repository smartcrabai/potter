#![recursion_limit = "512"]
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "Blender adapter fixtures use fixed valid paths and values"
)]

use std::{error::Error, fs, process::Command};

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
