use std::{collections::BTreeMap, error::Error, fs, process::Command};

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::{Mesh, modifiers::evaluate_modifiers},
    model::{
        ArmatureData, Bone, DataBlock, Id, Modifier, ModifierRuntime, Node, SceneDoc,
    },
};
use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/blender_script.rs"]
mod blender_script;

use blender_file::blender_executable;
use blender_script::run_blender_script;

const BLENDER_MASK_ORACLE: &str = r#"
import bpy, json, sys
root = sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
def grid(name):
    vertices = [(float(x),float(y),0.0) for y in range(3) for x in range(3)]
    faces = [(0,1,4,3),(1,2,5,4),(3,4,7,6),(4,5,8,7)]
    mesh = bpy.data.meshes.new(name + 'Mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj
def dump(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    vertices = [[float(v.co.x),float(v.co.y),float(v.co.z)] for v in mesh.vertices]
    faces = [[vertices[i] for i in poly.vertices] for poly in mesh.polygons]
    result = {'vertices':vertices,'faces':faces,'vertex_count':len(vertices),'face_count':len(faces)}
    evaluated.to_mesh_clear()
    return result
weights = [0.0,0.2,0.4,0.5,0.6,0.8,1.0,0.3,0.7]
def add_group(obj, name, values):
    group = obj.vertex_groups.new(name=name)
    for index, weight in enumerate(values):
        group.add([index], weight, 'REPLACE')
def group_case(key, threshold, invert=False, smooth=False):
    obj = grid('Group_' + key)
    add_group(obj, 'G', weights)
    mod = obj.modifiers.new('Mask','MASK')
    mod.mode = 'VERTEX_GROUP'
    mod.vertex_group = 'G'
    mod.threshold = threshold
    mod.invert_vertex_group = invert
    mod.use_smooth = smooth
    return dump(obj)
result = {
    'vg_threshold':group_case('threshold',0.5),
    'vg_inverted':group_case('inverted',0.5,True),
    'vg_smooth':group_case('smooth',0.5,False,True),
    'vg_smooth_inverted':group_case('smooth_inverted',0.5,True,True),
    'vg_threshold_other':group_case('threshold_other',0.4),
}
# A missing group leaves the input unchanged once deform-vertex data exists.
missing = grid('Missing')
add_group(missing,'Other',weights)
mod = missing.modifiers.new('Mask','MASK'); mod.mode='VERTEX_GROUP'; mod.vertex_group='MISSING'; mod.threshold=0.5
result['missing_group'] = dump(missing)
mod.invert_vertex_group = True
result['missing_group_inverted'] = dump(missing)
# No deform-vertex data produces an empty mesh unless inverted.
no_data = grid('NoData')
mod = no_data.modifiers.new('Mask','MASK'); mod.mode='VERTEX_GROUP'; mod.vertex_group='G'; mod.threshold=0.5
result['no_weights'] = dump(no_data)
mod.invert_vertex_group = True
result['no_weights_inverted'] = dump(no_data)
empty_unrelated = grid('EmptyUnrelated'); empty_unrelated.vertex_groups.new(name='Other')
mod = empty_unrelated.modifiers.new('Mask','MASK'); mod.mode='VERTEX_GROUP'; mod.vertex_group='MISSING'; mod.threshold=0.5
result['empty_unrelated_missing_group'] = dump(empty_unrelated)
mod.invert_vertex_group = True
result['empty_unrelated_missing_group_inverted'] = dump(empty_unrelated)
# An empty target group reads as zero weights when another group has created deform-vertex data.
empty_group = grid('EmptyGroup'); empty_group.vertex_groups.new(name='G'); add_group(empty_group,'Other',weights)
mod = empty_group.modifiers.new('Mask','MASK'); mod.mode='VERTEX_GROUP'; mod.vertex_group='G'; mod.threshold=0.5
result['empty_group'] = dump(empty_group)
mod.invert_vertex_group = True
result['empty_group_inverted'] = dump(empty_group)

arm_data = bpy.data.armatures.new('ArmatureData')
arm = bpy.data.objects.new('Armature', arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
for name, y in (('B1',0.0),('B2',1.0)):
    bone = arm_data.edit_bones.new(name); bone.head=(0.0,y,0.0); bone.tail=(0.0,y+0.5,0.0)
bpy.ops.object.mode_set(mode='OBJECT')
arm_obj = grid('ArmatureSubject')
add_group(arm_obj,'B1',weights)
add_group(arm_obj,'B2',[1.0-w for w in weights])
def arm_case(key, selected, threshold=0.5, invert=False, armature=True):
    obj = grid('Arm_' + key)
    add_group(obj,'B1',weights)
    add_group(obj,'B2',[1.0-w for w in weights])
    for pose_bone in arm.pose.bones:
        pose_bone.select = pose_bone.name in selected
    mod = obj.modifiers.new('Mask','MASK'); mod.mode='ARMATURE'; mod.armature=arm if armature else None
    mod.threshold=threshold; mod.invert_vertex_group=invert
    return dump(obj)
result['armature_b1'] = arm_case('b1',['B1'])
arm.data.bones['B1'].hide = True
result['armature_b1_hidden'] = arm_case('b1_hidden',['B1'])
arm.data.bones['B1'].hide = False
result['armature_b1_inverted'] = arm_case('b1_inverted',['B1'],invert=True)
result['armature_both'] = arm_case('both',['B1','B2'])
result['armature_none'] = arm_case('none',[])
result['armature_none_inverted'] = arm_case('none_inverted',[],invert=True)
result['armature_missing_object'] = arm_case('missing_object',['B1'],armature=False)
with open(root + '/expected_mask.json','w') as file:
    json.dump(result,file)
"#;

fn mask_modifier(params: Value) -> Result<Modifier, Box<dyn Error>> {
    Ok(Modifier {
        id: Id::new("mask_parity_modifier")?,
        modifier_type: "mask".to_owned(),
        name: "Mask".to_owned(),
        enabled: true,
        params: params.as_object().ok_or("mask params must be an object")?.clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

fn weighted_grid(with_group: bool) -> Result<Mesh, Box<dyn Error>> {
    let positions = (0..3)
        .flat_map(|y| (0..3).map(move |x| DVec3::new(f64::from(x), f64::from(y), 0.0)))
        .collect();
    let mut mesh = Mesh::from_positions_and_faces(
        positions,
        vec![vec![0,1,4,3],vec![1,2,5,4],vec![3,4,7,6],vec![4,5,8,7]],
    )?;
    if with_group {
        mesh.attributes.insert("vertex_groups".to_owned(), json!({
            "G":{"v0":0.0,"v1":0.2,"v2":0.4,"v3":0.5,"v4":0.6,"v5":0.8,"v6":1.0,"v7":0.3,"v8":0.7},
            "B1":{"v0":0.0,"v1":0.2,"v2":0.4,"v3":0.5,"v4":0.6,"v5":0.8,"v6":1.0,"v7":0.3,"v8":0.7},
            "B2":{"v0":1.0,"v1":0.8,"v2":0.6,"v3":0.5,"v4":0.4,"v5":0.2,"v6":0.0,"v7":0.7,"v8":0.3}
        }));
    }
    Ok(mesh)
}

fn read_expected(blender: &std::path::Path) -> Result<Value, Box<dyn Error>> {
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    run_blender_script(blender, "mask_oracle.py", BLENDER_MASK_ORACLE, &root, "Blender mask oracle failed")?;
    Ok(serde_json::from_slice(&fs::read(root.join("expected_mask.json"))?)?)
}

fn vector(value: &Value) -> Result<DVec3, Box<dyn Error>> {
    let values = value.as_array().ok_or("expected position is not an array")?;
    Ok(DVec3::new(
        values.first().and_then(Value::as_f64).ok_or("position x missing")?,
        values.get(1).and_then(Value::as_f64).ok_or("position y missing")?,
        values.get(2).and_then(Value::as_f64).ok_or("position z missing")?,
    ))
}

fn assert_geometry_parity(name: &str, actual: &Mesh, expected: &Value) -> Result<(), Box<dyn Error>> {
    let expected_vertices = expected["vertices"].as_array().ok_or("Blender vertices missing")?.iter().map(vector).collect::<Result<Vec<_>,_>>()?;
    assert_eq!(actual.vertices.len(), expected_vertices.len(), "{name}: vertex count differs");
    let mut actual_vertices = actual.vertices.iter().map(|vertex| vertex.co).collect::<Vec<_>>();
    let mut expected_vertices = expected_vertices;
    let sort_points = |points: &mut Vec<DVec3>| points.sort_by(|a,b| {
        a.x.total_cmp(&b.x).then_with(||a.y.total_cmp(&b.y)).then_with(||a.z.total_cmp(&b.z))
    });
    sort_points(&mut actual_vertices); sort_points(&mut expected_vertices);
    for (actual, expected) in actual_vertices.iter().zip(&expected_vertices) {
        assert!((*actual - *expected).length() <= 2.0e-5, "{name}: vertex {actual:?} differs from {expected:?}");
    }
    let expected_faces = expected["faces"].as_array().ok_or("Blender faces missing")?;
    assert_eq!(actual.faces.len(), expected_faces.len(), "{name}: face count differs");
    let positions = actual.vertices.iter().map(|vertex| (vertex.id, vertex.co)).collect::<BTreeMap<_,_>>();
    let mut actual_faces = actual.faces.iter().map(|face| {
        let mut points = face.vertices.iter().filter_map(|id| positions.get(id).copied()).collect::<Vec<_>>();
        sort_points(&mut points);
        points
    }).collect::<Vec<_>>();
    let mut expected_faces = expected_faces.iter().map(|face| {
        let mut points = face.as_array().into_iter().flatten().map(vector).collect::<Result<Vec<_>,_>>()?;
        sort_points(&mut points);
        Ok::<_,Box<dyn Error>>(points)
    }).collect::<Result<Vec<_>,_>>()?;
    let sort_faces = |faces: &mut Vec<Vec<DVec3>>| {
        faces.sort_by(|a, b| {
            a.len().cmp(&b.len()).then_with(|| {
                a.iter()
                    .zip(b)
                    .find_map(|(first, second)| {
                        let order = first
                            .x
                            .total_cmp(&second.x)
                            .then_with(|| first.y.total_cmp(&second.y))
                            .then_with(|| first.z.total_cmp(&second.z));
                        (order != std::cmp::Ordering::Equal).then_some(order)
                    })
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
    };
    sort_faces(&mut actual_faces); sort_faces(&mut expected_faces);
    for (actual_face, expected_face) in actual_faces.iter().zip(&expected_faces) {
        assert_eq!(actual_face.len(), expected_face.len(), "{name}: polygon corner count differs");
        for (actual, expected) in actual_face.iter().zip(expected_face) {
            assert!((*actual - *expected).length() <= 2.0e-5, "{name}: face corner {actual:?} differs from {expected:?}");
        }
    }
    Ok(())
}

fn armature_scene(
    mesh: Mesh,
    selected_bones: &[&str],
    invert: bool,
    has_armature: bool,
) -> Result<SceneDoc, Box<dyn Error>> {
    let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000001".to_owned());
    let mesh_id = Id::new("mask_subject")?;
    let mesh_data_id = Id::new("mask_mesh_data")?;
    let armature_id = Id::new("mask_armature")?;
    let armature_data_id = Id::new("mask_armature_data")?;
    let mut mesh_data = DataBlock::default();
    mesh_data.mesh = Some(mesh);
    doc.data_blocks.insert(mesh_data_id.clone(), mesh_data);
    let params = json!({
        "mode":"ARMATURE",
        "armature":has_armature.then_some(armature_id.as_str()),
        "threshold":0.5,
        "invert_vertex_group":invert,
        "use_smooth":false
    });
    let mut mesh_node = Node { name:"MaskSubject".to_owned(), kind:"mesh".to_owned(), data:Some(mesh_data_id), ..Node::default() };
    mesh_node.modifiers.push(mask_modifier(params)?);
    doc.nodes.insert(mesh_id.clone(), mesh_node);
    if has_armature {
        let mut armature = ArmatureData::default();
        for (name, bone_id) in [("B1","mask_bone_b1"),("B2","mask_bone_b2")] {
            let mut bone = Bone::default();
            bone.name = name.to_owned();
            bone.head = [0.0,0.0,0.0];
            bone.tail = [0.0,0.5,0.0];
            bone.select = selected_bones.contains(&name);
            armature.bones.insert(Id::new(bone_id)?, bone);
        }
        let mut armature_data = DataBlock::default();
        armature_data.data_type = "armature".to_owned();
        armature_data.mesh = None;
        armature_data.armature = Some(armature);
        doc.data_blocks.insert(armature_data_id.clone(), armature_data);
        doc.nodes.insert(armature_id.clone(), Node {
            name:"MaskArmature".to_owned(), kind:"armature".to_owned(), data:Some(armature_data_id), ..Node::default()
        });
    }
    let root = Id::from_static("collection_root");
    let objects = &mut doc.collections.get_mut(&root).ok_or("root collection missing")?.objects;
    objects.push(mesh_id);
    if has_armature {
        objects.push(armature_id);
    }
    Ok(doc)
}

#[test]
fn mask_vertex_group_and_armature_modes_match_blender_threshold_invert_and_smooth() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping mask Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let expected = read_expected(&blender)?;
    let cases = [
        ("vg_threshold", json!({"mode":"VERTEX_GROUP","vertex_group":"G","threshold":0.5,"invert_vertex_group":false,"use_smooth":false}), true),
        ("vg_inverted", json!({"mode":"VERTEX_GROUP","vertex_group":"G","threshold":0.5,"invert_vertex_group":true,"use_smooth":false}), true),
        ("vg_smooth", json!({"mode":"VERTEX_GROUP","vertex_group":"G","threshold":0.5,"invert_vertex_group":false,"use_smooth":true}), true),
        ("vg_smooth_inverted", json!({"mode":"VERTEX_GROUP","vertex_group":"G","threshold":0.5,"invert_vertex_group":true,"use_smooth":true}), true),
        ("vg_threshold_other", json!({"mode":"VERTEX_GROUP","vertex_group":"G","threshold":0.4,"invert_vertex_group":false,"use_smooth":false}), true),
        ("missing_group", json!({"mode":"VERTEX_GROUP","vertex_group":"MISSING","threshold":0.5}), true),
        ("missing_group_inverted", json!({"mode":"VERTEX_GROUP","vertex_group":"MISSING","threshold":0.5,"invert_vertex_group":true}), true),
    ];
    for (name, params, with_group) in cases {
        let mesh = weighted_grid(with_group)?;
        let actual = evaluate_modifiers(&mesh, &[mask_modifier(params)?])?;
        assert_geometry_parity(name, &actual, &expected[name])?;
    }
    let empty_group_mesh = {
        let mut mesh = weighted_grid(false)?;
        mesh.attributes.insert("vertex_groups".to_owned(), json!({"G":{},"Other":{"v0":1.0}}));
        mesh
    };
    for (name, invert) in [("empty_group",false),("empty_group_inverted",true),("no_weights",false),("no_weights_inverted",true)] {
        let mesh = if name.starts_with("empty_group") { empty_group_mesh.clone() } else { weighted_grid(false)? };
        let actual = evaluate_modifiers(&mesh, &[mask_modifier(json!({"mode":"VERTEX_GROUP","vertex_group":"G","threshold":0.5,"invert_vertex_group":invert}))?])?;
        assert_geometry_parity(name, &actual, &expected[name])?;
    }
    let empty_unrelated_mesh = {
        let mut mesh = weighted_grid(false)?;
        mesh.attributes.insert("vertex_groups".to_owned(), json!({"Other":{}}));
        mesh
    };
    for (name, invert) in [
        ("empty_unrelated_missing_group", false),
        ("empty_unrelated_missing_group_inverted", true),
    ] {
        let actual = evaluate_modifiers(
            &empty_unrelated_mesh,
            &[mask_modifier(json!({
                "mode":"VERTEX_GROUP","vertex_group":"MISSING","threshold":0.5,
                "invert_vertex_group":invert
            }))?],
        )?;
        assert_geometry_parity(name, &actual, &expected[name])?;
    }
    for (name, selected, invert) in [
        ("armature_b1", &["B1"][..], false),
        ("armature_b1_hidden", &["B1"][..], false),
        ("armature_b1_inverted", &["B1"][..], true),
        ("armature_both", &["B1", "B2"][..], false),
        ("armature_none", &[][..], false),
        ("armature_none_inverted", &[][..], true),
    ] {
        let doc = armature_scene(weighted_grid(true)?, selected, invert, true)?;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let actual = snapshot
            .meshes
            .get(&Id::new("mask_subject")?)
            .ok_or("evaluated mask mesh missing")?;
        assert_geometry_parity(name, actual, &expected[name])?;
    }
    let no_armature_doc = armature_scene(weighted_grid(true)?, &["B1"], false, false)?;
    let snapshot = Snapshot::evaluate(&no_armature_doc, &EvaluationContext::default())?;
    let actual = snapshot
        .meshes
        .get(&Id::new("mask_subject")?)
        .ok_or("evaluated mask mesh missing")?;
    assert_geometry_parity("armature_missing_object", actual, &expected["armature_missing_object"])?;
    Ok(())
}
#[test]
fn cli_armature_mask_and_blend_roundtrip_preserve_selected_bone() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping armature-mask blend roundtrip: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let operations_file = directory.path().join("operations.json");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stdout)
    );
    let operations = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [
            {"op":"node.create","id":"body","name":"Body","kind":"plane","params":{"size":2.0}},
            {"op":"node.create","id":"rig","name":"Rig","kind":"armature"},
            {"op":"bone.create","target":{"id":"rig"},"id":"mask_bone","name":"MaskBone",
                "head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0],"select":true},
            {"op":"vertex_group.create","target":{"id":"body"},"id":"mask_group","name":"MaskBone"},
            {"op":"vertex_group.assign","target":{"id":"body"},"group_id":"mask_group",
                "weights":[{"vertex_id":0,"weight":1.0},{"vertex_id":1,"weight":1.0},
                           {"vertex_id":2,"weight":1.0},{"vertex_id":3,"weight":0.0}]},
            {"op":"modifier.create","target":{"id":"body"},"id":"armature_mask","type":"mask",
                "params":{"mode":"ARMATURE","armature":"rig","threshold":0.5}},
            {"op":"modifier.apply","target":{"id":"body"},"id":"armature_mask"}
        ]
    });
    fs::write(&operations_file, serde_json::to_vec(&operations)?)?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .arg("--file")
        .arg(&operations_file)
        .arg("--json")
        .output()?;
    assert!(applied.status.success(), "{}", String::from_utf8_lossy(&applied.stdout));
    let inspected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id", "body", "--json"])
        .output()?;
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stdout)
    );
    let output: Value = serde_json::from_slice(&inspected.stdout)?;
    let geometry = &output["result"]["items"][0]["evaluated_geometry"];
    assert_eq!(geometry["vertex_count"], 3);
    assert_eq!(geometry["face_count"], 0);

    let blend_file = directory.path().join("mask-selected.blend");
    let exported = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["export"])
        .arg(&scene)
        .args(["--format", "blend", "--out"])
        .arg(&blend_file)
        .args(["--blender"])
        .arg(&blender)
        .arg("--json")
        .output()?;
    assert!(exported.status.success(), "{}", String::from_utf8_lossy(&exported.stdout));
    let blender_probe = Command::new(&blender)
        .args(["--background", "--disable-autoexec", "--python-exit-code", "1", "--python-expr"])
        .arg("import bpy; bone=bpy.data.objects['Rig'].data.bones['MaskBone']; assert bone.select; print('selected mask bone roundtrip OK')")
        .output()?;
    assert!(
        blender_probe.status.success(),
        "{}",
        String::from_utf8_lossy(&blender_probe.stdout)
    );
    let imported = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["import"])
        .arg(&scene)
        .args(["--file"])
        .arg(&blend_file)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "1",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json")
        .output()?;
    assert!(imported.status.success(), "{}", String::from_utf8_lossy(&imported.stdout));
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let rig_data = document["nodes"]["rig"]["data"]
        .as_str()
        .ok_or("armature Data-Block reference missing")?;
    assert_eq!(
        document["data_blocks"][rig_data]["armature"]["bones"]["mask_bone"]["select"],
        json!(true)
    );
    assert_eq!(
        document["nodes"]["body"]["active_vertex_group_index"],
        json!(0)
    );
    Ok(())
}
