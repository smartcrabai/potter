use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    process::Command,
};

use glam::{DMat4, DVec3, Mat4};
use potter_core::{
    geom::{
        Mesh,
        modifiers::attributes::{
            AttributeLayerSelection, SceneOperand, SceneOperandMatricesF32, evaluate_with_context,
        },
    },
    model::{CameraData, Modifier, ModifierRuntime},
};
use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/blender_script.rs"]
mod blender_script;

fn modifier(modifier_type: &str, params: &Value) -> Result<Modifier, Box<dyn Error>> {
    Ok(Modifier {
        id: potter_core::model::Id::new("transfer_test")?,
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: params
            .as_object()
            .ok_or("modifier params must be an object")?
            .clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

fn layer_selection(groups: &[&str], active_group: Option<usize>) -> AttributeLayerSelection {
    AttributeLayerSelection {
        vertex_group_names: groups.iter().map(|name| (*name).to_owned()).collect(),
        active_vertex_group_index: active_group,
    }
}

fn operand(
    mesh: Mesh,
    _local_to_subject: DMat4,
    groups: &[&str],
    active: Option<usize>,
) -> SceneOperand {
    let identity = Mat4::IDENTITY.to_cols_array();
    SceneOperand {
        mesh,
        local_to_subject: None,
        blender_matrices_f32: Some(SceneOperandMatricesF32 {
            world: None,
            local_to_subject: Some(identity),
            subject_to_local: Some(identity),
        }),
        camera: None::<CameraData>,
        armature: false,
        object_scale: DVec3::ONE,
        selected_bones: BTreeSet::new(),
        deforming_bones: BTreeSet::new(),
        layer_selection: layer_selection(groups, active),
    }
}

fn set_group(mesh: &mut Mesh, name: &str, weights: &[f64]) {
    let values = weights
        .iter()
        .enumerate()
        .map(|(index, weight)| (format!("v{index}"), json!(weight)))
        .collect::<serde_json::Map<_, _>>();
    let groups = mesh
        .attributes
        .entry("vertex_groups".to_owned())
        .or_insert_with(|| json!({}));
    groups[name] = Value::Object(values);
}

fn group_weights(mesh: &Mesh, name: &str) -> Vec<f64> {
    (0..mesh.vertices.len())
        .map(|index| {
            mesh.attributes["vertex_groups"][name][format!("v{index}")]
                .as_f64()
                .unwrap_or(0.0)
        })
        .collect()
}

fn set_uv_layers(mesh: &mut Mesh, layers: &[(&str, Vec<Vec<[f64; 2]>>)], active: &str) {
    let layers = layers
        .iter()
        .map(|(name, faces)| {
            let faces = mesh
                .faces
                .iter()
                .zip(faces)
                .map(|(face, values)| {
                    let uv = values
                        .iter()
                        .map(|[u, v]| [*u, 1.0 - *v])
                        .collect::<Vec<_>>();
                    json!({"face_id":face.id,"uv":uv})
                })
                .collect::<Vec<_>>();
            json!({"name":name,"faces":faces})
        })
        .collect::<Vec<_>>();
    mesh.attributes.insert(
        "uv_layers".to_owned(),
        json!({"layers":layers,"active":active,"render":active}),
    );
}

fn uv_layer_values(mesh: &Mesh, name: &str) -> Value {
    let Some(layer) = mesh
        .attributes
        .get("uv_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .and_then(|layers| {
            layers
                .iter()
                .find(|layer| layer.get("name").and_then(Value::as_str) == Some(name))
        })
    else {
        return json!([]);
    };
    let Some(rows) = layer.get("faces").and_then(Value::as_array) else {
        return json!([]);
    };
    let mut values = Vec::new();
    for face in &mesh.faces {
        if let Some(row) = rows
            .iter()
            .find(|row| row.get("face_id").and_then(Value::as_u64) == Some(u64::from(face.id)))
            .and_then(|row| row.get("uv"))
            .and_then(Value::as_array)
        {
            values.extend(row.iter().cloned());
        }
    }
    Value::Array(values)
}

fn flip_blender_uv_to_potter(value: &mut Value) {
    match value {
        Value::Array(values)
            if values.len() == 2 && values.iter().all(|value| value.as_f64().is_some()) =>
        {
            if let Some(v) = values[1].as_f64() {
                values[1] = json!(1.0 - v);
            }
        }
        Value::Array(values) => {
            for value in values {
                flip_blender_uv_to_potter(value);
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                flip_blender_uv_to_potter(value);
            }
        }
        _ => {}
    }
}

fn set_smooth_faces(mesh: &mut Mesh) {
    let values = mesh
        .faces
        .iter()
        .map(|face| (format!("f{}", face.id), json!(true)))
        .collect::<serde_json::Map<_, _>>();
    mesh.attributes.insert(
        "shade_smooth".to_owned(),
        json!({"domain":"face","type":"bool","values":values}),
    );
}
fn set_edge_flags(mesh: &mut Mesh, name: &str, marked_vertices: Option<(usize, usize)>) {
    let marked_ids =
        marked_vertices.map(|(first, second)| (mesh.vertices[first].id, mesh.vertices[second].id));
    let values = mesh
        .edges
        .iter()
        .map(|edge| {
            let marked = marked_ids.is_some_and(|(first, second)| {
                edge.vertices.contains(&first) && edge.vertices.contains(&second)
            });
            (format!("e{}", edge.id), json!(marked))
        })
        .collect::<serde_json::Map<_, _>>();
    mesh.attributes.insert(
        name.to_owned(),
        json!({"domain":"edge","type":"bool","values":values}),
    );
}

fn set_face_flags(mesh: &mut Mesh, name: &str, flags: &[bool]) {
    let values = mesh
        .faces
        .iter()
        .zip(flags)
        .map(|(face, flag)| (format!("f{}", face.id), json!(flag)))
        .collect::<serde_json::Map<_, _>>();
    mesh.attributes.insert(
        name.to_owned(),
        json!({"domain":"face","type":"bool","values":values}),
    );
}

fn set_domain_attribute(
    mesh: &mut Mesh,
    name: &str,
    domain: &str,
    data_type: &str,
    values: &Value,
) -> Result<(), Box<dyn Error>> {
    let items = values
        .as_array()
        .ok_or("domain attribute values must be an array")?;
    let expected_len = match domain {
        "point" => mesh.vertices.len(),
        "edge" => mesh.edges.len(),
        "face" => mesh.faces.len(),
        _ => return Err("unsupported test attribute domain".into()),
    };
    if items.len() != expected_len {
        return Err("domain attribute value count does not match mesh".into());
    }
    let mut entries = serde_json::Map::new();
    match domain {
        "point" => {
            for (vertex, value) in mesh.vertices.iter().zip(items) {
                entries.insert(format!("v{}", vertex.id), value.clone());
            }
        }
        "edge" => {
            for (edge, value) in mesh.edges.iter().zip(items) {
                entries.insert(format!("e{}", edge.id), value.clone());
            }
        }
        "face" => {
            for (face, value) in mesh.faces.iter().zip(items) {
                entries.insert(format!("f{}", face.id), value.clone());
            }
        }
        _ => return Err("unsupported test attribute domain".into()),
    }
    mesh.attributes.insert(
        name.to_owned(),
        json!({"domain":domain,"type":data_type,"values":entries}),
    );
    Ok(())
}

fn domain_attribute_values(mesh: &Mesh, name: &str) -> Value {
    let Some(attribute) = mesh.attributes.get(name) else {
        return json!([]);
    };
    let Some(values) = attribute.get("values").and_then(Value::as_object) else {
        return json!([]);
    };
    let keys = match attribute.get("domain").and_then(Value::as_str) {
        Some("point") => mesh
            .vertices
            .iter()
            .map(|item| format!("v{}", item.id))
            .collect::<Vec<_>>(),
        Some("edge") => mesh
            .edges
            .iter()
            .map(|item| format!("e{}", item.id))
            .collect::<Vec<_>>(),
        Some("face") => mesh
            .faces
            .iter()
            .map(|item| format!("f{}", item.id))
            .collect::<Vec<_>>(),
        _ => return json!([]),
    };
    Value::Array(
        keys.iter()
            .filter_map(|key| values.get(key).cloned())
            .collect(),
    )
}

/// The color values of a layer in row order: one per vertex for a point layer, one per corner for a
/// corner layer. A missing layer reads as empty.
fn color_values(mesh: &Mesh, name: &str) -> Value {
    let Some(attribute) = mesh.attributes.get(name) else {
        return json!([]);
    };
    let Some(values) = attribute.get("values").and_then(Value::as_object) else {
        return json!([]);
    };
    if attribute.get("domain").and_then(Value::as_str) == Some("point") {
        Value::Array(
            mesh.vertices
                .iter()
                .filter_map(|vertex| values.get(&format!("v{}", vertex.id)).cloned())
                .collect(),
        )
    } else {
        let mut colors = Vec::new();
        for face in &mesh.faces {
            if let Some(row) = values
                .get(&format!("f{}", face.id))
                .and_then(Value::as_array)
            {
                colors.extend(row.iter().cloned());
            }
        }
        Value::Array(colors)
    }
}

fn set_custom_normal_codes(mesh: &mut Mesh, raw_codes: &Value) -> Result<(), Box<dyn Error>> {
    let codes = raw_codes
        .as_array()
        .ok_or("Blender custom-normal codes must be an array")?;
    let corner_count = mesh
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    if codes.len() != corner_count {
        return Err("custom-normal code count does not match mesh corners".into());
    }
    let mut cursor = 0;
    let mut values = serde_json::Map::new();
    for face in &mesh.faces {
        let end = cursor + face.vertices.len();
        let row = codes
            .get(cursor..end)
            .ok_or("custom-normal face row is out of range")?;
        values.insert(format!("f{}", face.id), Value::Array(row.to_vec()));
        cursor = end;
    }
    mesh.attributes.insert(
        "custom_normal".to_owned(),
        json!({"domain":"corner","type":"normal_code","values":values}),
    );
    Ok(())
}

fn max_error(actual: &Value, expected: &Value) -> Result<f64, Box<dyn Error>> {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            Ok((a.as_f64().ok_or("non-f64 actual")? - b.as_f64().ok_or("non-f64 expected")?).abs())
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            let mut error = 0.0_f64;
            for (a, b) in a.iter().zip(b) {
                error = error.max(max_error(a, b)?);
            }
            Ok(error)
        }
        _ => Err("parity values have different shapes".into()),
    }
}

fn rgba_values(values: &Value) -> Result<Vec<[f64; 4]>, Box<dyn Error>> {
    let values = values.as_array().ok_or("oracle colors must be an array")?;
    values
        .iter()
        .map(|value| {
            let channels = value.as_array().ok_or("RGBA color must be an array")?;
            if channels.len() != 4 {
                return Err("RGBA color must have four channels".into());
            }
            Ok([
                channels[0].as_f64().ok_or("RGBA red channel missing")?,
                channels[1].as_f64().ok_or("RGBA green channel missing")?,
                channels[2].as_f64().ok_or("RGBA blue channel missing")?,
                channels[3].as_f64().ok_or("RGBA alpha channel missing")?,
            ])
        })
        .collect()
}

const BLENDER_VERTEX_MAPPING_SCRIPT: &str = r"
import bpy, json, os, sys
from mathutils import Matrix, Vector
root = os.path.realpath(sys.argv[sys.argv.index('--') + 1])
def obj(name, verts, faces):
    mesh = bpy.data.meshes.new(name + 'Mesh'); mesh.from_pydata(verts, [], faces); mesh.update()
    ob = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(ob); return ob
def group(ob, name, weights):
    vg = ob.vertex_groups.new(name=name)
    for i, weight in enumerate(weights):
        vg.add([i], float(weight), 'REPLACE')
    return vg
def read(ob):
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh = ev.to_mesh()
    vg = ev.vertex_groups.get('A'); values = []
    for v in mesh.vertices:
        value = 0.0
        if vg:
            for a in v.groups:
                if a.group == vg.index: value = a.weight; break
        values.append(value)
    ev.to_mesh_clear(); return values
src = obj('Source', [(0,0,0),(2,0,0),(2,1,0),(0,1,0)], [(0,1,2,3)])
group(src, 'A', [0.05,0.35,0.8,0.95])
src.matrix_world = Matrix.Translation(Vector((0.2,-0.1,0.0))) @ Matrix.Diagonal((1.2,0.8,1.1,1.0))
result = {}
def run(key, verts, faces, mode, max_distance=10.0, use_transform=True):
    dst = obj('Dest' + key, verts, faces); group(dst, 'A', [0.2 + 0.1*i for i in range(len(verts))])
    dst.matrix_world = Matrix.Translation(Vector((-0.1,0.15,0.0))) @ Matrix.Diagonal((0.9,1.1,1.0,1.0))
    mod = dst.modifiers.new('Transfer', 'DATA_TRANSFER'); mod.object = src
    mod.use_object_transform = use_transform; mod.use_vert_data = True; mod.data_types_verts = {'VGROUP_WEIGHTS'}
    mod.vert_mapping = mode; mod.mix_mode = 'REPLACE'; mod.mix_factor = 1.0
    mod.use_max_distance = True; mod.max_distance = max_distance; mod.ray_radius = 0.01
    bpy.context.view_layer.update(); result[key] = read(dst)
tri = [(0.15,0.12,0.18),(1.45,0.18,0.15),(0.85,0.86,0.21)]
for mode in ('NEAREST','EDGE_NEAREST','EDGEINTERP_NEAREST','POLY_NEAREST','POLYINTERP_NEAREST','POLYINTERP_VNORPROJ'):
    run(mode, tri, [(0,1,2)], mode)
run('TOPOLOGY', [(0.0,0.0,0.0),(2.0,0.0,0.0),(2.0,1.0,0.0),(0.0,1.0,0.0)], [(0,1,2,3)], 'TOPOLOGY')
run('DISTANCE_CUTOFF',[(20,20,20),(21,20,20),(20,21,20)],[(0,1,2)],'NEAREST',0.1)
run('NO_OBJECT_TRANSFORM',tri,[(0,1,2)],'NEAREST',10.0,False)
with open(os.path.join(root, 'vertex-mapping-oracle.json'), 'w') as f: json.dump(result, f)
";

#[test]
fn vertex_mapping_families_match_blender_with_asymmetric_transforms() -> Result<(), Box<dyn Error>>
{
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping data-transfer Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "vertex-mappings.py",
        BLENDER_VERTEX_MAPPING_SCRIPT,
        root,
        "Blender data-transfer vertex mapping oracle failed",
    )?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("vertex-mapping-oracle.json"))?)?;
    let source = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(2.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    let mut source = source;
    set_group(&mut source, "A", &[0.05, 0.35, 0.8, 0.95]);
    let source_world = DMat4::from_scale_rotation_translation(
        DVec3::new(1.2, 0.8, 1.1),
        glam::DQuat::IDENTITY,
        DVec3::new(0.2, -0.1, 0.0),
    );
    let destination_world = DMat4::from_scale_rotation_translation(
        DVec3::new(0.9, 1.1, 1.0),
        glam::DQuat::IDENTITY,
        DVec3::new(-0.1, 0.15, 0.0),
    );
    let source_to_destination = destination_world.inverse() * source_world;
    let local_to_subject_f32 = [
        1.333_333_5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.727_272_7,
        0.0,
        0.0,
        0.0,
        0.0,
        1.1,
        0.0,
        0.333_333_37,
        -0.227_272_72,
        0.0,
        1.0,
    ];
    let subject_to_local_f32 = [
        0.749_999_94,
        0.0,
        0.0,
        0.0,
        0.0,
        1.375_000_1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.909_090_94,
        0.0,
        -0.25,
        0.312_500_03,
        0.0,
        1.0,
    ];
    let mut source_operand = operand(source.clone(), source_to_destination, &["A"], Some(0));
    source_operand.blender_matrices_f32 = Some(SceneOperandMatricesF32 {
        world: None,
        local_to_subject: Some(local_to_subject_f32),
        subject_to_local: Some(subject_to_local_f32),
    });
    let mut source_operands = BTreeMap::new();
    source_operands.insert("object".to_owned(), vec![source_operand]);
    let mappings = [
        "NEAREST",
        "EDGE_NEAREST",
        "EDGEINTERP_NEAREST",
        "POLY_NEAREST",
        "POLYINTERP_NEAREST",
        "POLYINTERP_VNORPROJ",
    ];
    let destination_positions = [
        DVec3::new(0.15, 0.12, 0.18),
        DVec3::new(1.45, 0.18, 0.15),
        DVec3::new(0.85, 0.86, 0.21),
    ];
    for mapping in mappings {
        let mut destination =
            Mesh::from_positions_and_faces(destination_positions.to_vec(), vec![vec![0, 1, 2]])?;
        set_group(&mut destination, "A", &[0.2, 0.3, 0.4]);
        let modifier_1 = modifier(
            "data_transfer",
            &json!({
                "object":"source","use_object_transform":true,"use_vert_data":true,
                "data_types_verts":["VGROUP_WEIGHTS"],"vert_mapping":mapping,
                "mix_mode":"REPLACE","mix_factor":1.0,"use_max_distance":true,
                "max_distance":10.0,"ray_radius":0.01
            }),
        )?;
        let actual = evaluate_with_context(
            &destination,
            &modifier_1,
            &source_operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &AttributeLayerSelection::default(),
        )?;
        let error = max_error(&json!(group_weights(&actual, "A")), &expected[mapping])?;
        assert!(error < 2.0e-4, "{mapping} max error {error}");
    }
    let mut no_transform_destination =
        Mesh::from_positions_and_faces(destination_positions.to_vec(), vec![vec![0, 1, 2]])?;
    set_group(&mut no_transform_destination, "A", &[0.2, 0.3, 0.4]);
    let no_transform_modifier = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_object_transform":false,"use_vert_data":true,
            "data_types_verts":["VGROUP_WEIGHTS"],"vert_mapping":"NEAREST",
            "mix_mode":"REPLACE","mix_factor":1.0,"use_max_distance":true,"max_distance":10.0,
        }),
    )?;
    let no_transform_actual = evaluate_with_context(
        &no_transform_destination,
        &no_transform_modifier,
        &source_operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    let no_transform_error = max_error(
        &json!(group_weights(&no_transform_actual, "A")),
        &expected["NO_OBJECT_TRANSFORM"],
    )?;
    assert!(
        no_transform_error < 2.0e-4,
        "use_object_transform=false max error {no_transform_error}"
    );
    let mut cutoff = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(20.0, 20.0, 20.0),
            DVec3::new(21.0, 20.0, 20.0),
            DVec3::new(20.0, 21.0, 20.0),
        ],
        vec![vec![0, 1, 2]],
    )?;
    set_group(&mut cutoff, "A", &[0.2, 0.3, 0.4]);
    let cutoff_modifier = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_object_transform":true,"use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],
            "vert_mapping":"NEAREST","mix_mode":"REPLACE","mix_factor":1.0,"use_max_distance":true,"max_distance":0.1
        }),
    )?;
    let cutoff_actual = evaluate_with_context(
        &cutoff,
        &cutoff_modifier,
        &source_operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    let cutoff_error = max_error(
        &json!(group_weights(&cutoff_actual, "A")),
        &expected["DISTANCE_CUTOFF"],
    )?;
    assert!(
        cutoff_error < 1.0e-6,
        "max-distance cutoff error {cutoff_error}"
    );
    let topology = Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(2.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    let mut topology = topology;
    set_group(&mut topology, "A", &[0.2, 0.3, 0.4, 0.5]);
    let modifier_2 = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_object_transform":true,"use_vert_data":true,
            "data_types_verts":["VGROUP_WEIGHTS"],"vert_mapping":"TOPOLOGY",
            "mix_mode":"REPLACE","mix_factor":1.0
        }),
    )?;
    let actual = evaluate_with_context(
        &topology,
        &modifier_2,
        &source_operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    let error = max_error(&json!(group_weights(&actual, "A")), &expected["TOPOLOGY"])?;
    assert!(error < 1.0e-6, "TOPOLOGY max error {error}");
    Ok(())
}

const BLENDER_MIX_SCRIPT: &str = r"
import bpy, json, os, sys
root=os.path.realpath(sys.argv[sys.argv.index('--')+1])
def mesh(name):
    m=bpy.data.meshes.new(name+'Mesh'); m.from_pydata([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],[],[(0,1,2,3)]); m.update()
    o=bpy.data.objects.new(name,m); bpy.context.collection.objects.link(o); return o
def group(o,name,values):
    g=o.vertex_groups.new(name=name)
    for i,v in enumerate(values): g.add([i],float(v),'REPLACE')
    return g
def weights(o,name):
    ev=o.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh(); g=ev.vertex_groups[name]; out=[]
    for v in me.vertices:
        w=0.0
        for a in v.groups:
            if a.group==g.index: w=a.weight; break
        out.append(w)
    ev.to_mesh_clear(); return out
src=mesh('Source'); group(src,'B',[0.8,0.3,0.1,0.6]); group(src,'A',[0.1,0.6,0.9,0.2])
src.vertex_groups.active_index=1
dst=mesh('Destination'); group(dst,'D0',[0.2,0.4,0.6,0.8]); group(dst,'D1',[0.7,0.5,0.3,0.1]); group(dst,'Mask',[0.25,0.75,0.5,1.0]); dst.vertex_groups.active_index=1
result={}
for mode in ('REPLACE','ABOVE_THRESHOLD','BELOW_THRESHOLD','MIX','ADD','SUB','MUL'):
    out=mesh('Mode'+mode); group(out,'D0',[0.2,0.4,0.6,0.8]); group(out,'D1',[0.7,0.5,0.3,0.1]); group(out,'Mask',[0.25,0.75,0.5,1.0])
    mod=out.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=src; mod.use_object_transform=True; mod.use_vert_data=True; mod.data_types_verts={'VGROUP_WEIGHTS'}; mod.vert_mapping='TOPOLOGY'
    mod.layers_vgroup_select_src='ALL'; mod.layers_vgroup_select_dst='INDEX'; mod.mix_mode=mode; mod.mix_factor=0.4; mod.vertex_group='Mask'
    bpy.context.view_layer.update(); result[mode]={'D0':weights(out,'D0'),'D1':weights(out,'D1')}
active=mesh('Active'); group(active,'D0',[0.2]*4); group(active,'D1',[0.3]*4); active.vertex_groups.active_index=1
bpy.context.view_layer.objects.active=active; active.select_set(True)
mod=active.modifiers.new('Active','DATA_TRANSFER'); mod.object=src; mod.use_vert_data=True; mod.data_types_verts={'VGROUP_WEIGHTS'}; mod.vert_mapping='TOPOLOGY'; mod.layers_vgroup_select_src='A'; mod.layers_vgroup_select_dst='D1'; mod.mix_mode='REPLACE'; mod.mix_factor=1.0
bpy.context.view_layer.update(); result['ACTIVE']={'D1':weights(active,'D1'),'D0':weights(active,'D0')}
inverted=mesh('Inverted'); group(inverted,'D0',[0.2,0.4,0.6,0.8]); group(inverted,'D1',[0.7,0.5,0.3,0.1]); group(inverted,'Mask',[0.25,0.75,0.5,1.0])
mod=inverted.modifiers.new('Inverted','DATA_TRANSFER'); mod.object=src; mod.use_vert_data=True; mod.data_types_verts={'VGROUP_WEIGHTS'}; mod.vert_mapping='TOPOLOGY'
mod.layers_vgroup_select_src='ALL'; mod.layers_vgroup_select_dst='INDEX'; mod.mix_mode='REPLACE'; mod.mix_factor=0.4; mod.vertex_group='Mask'; mod.invert_vertex_group=True
bpy.context.view_layer.update(); result['INVERT_MASK']={'D0':weights(inverted,'D0'),'D1':weights(inverted,'D1')}
with open(os.path.join(root,'mix-oracle.json'),'w') as f: json.dump(result,f)
";

#[test]
fn vgroup_active_index_mix_and_mask_match_blender() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping data-transfer mix Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "mix.py",
        BLENDER_MIX_SCRIPT,
        root,
        "Blender data-transfer mix oracle failed",
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("mix-oracle.json"))?)?;
    let quad = || {
        Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::ONE, DVec3::Y],
            vec![vec![0, 1, 2, 3]],
        )
    };
    let mut source = quad()?;
    set_group(&mut source, "B", &[0.8, 0.3, 0.1, 0.6]);
    set_group(&mut source, "A", &[0.1, 0.6, 0.9, 0.2]);
    let source_operand = operand(source.clone(), DMat4::IDENTITY, &["B", "A"], Some(1));
    let mut operands = BTreeMap::new();
    operands.insert("object".to_owned(), vec![source_operand]);
    for mode in [
        "REPLACE",
        "ABOVE_THRESHOLD",
        "BELOW_THRESHOLD",
        "MIX",
        "ADD",
        "SUB",
        "MUL",
    ] {
        let mut destination = quad()?;
        set_group(&mut destination, "D0", &[0.2, 0.4, 0.6, 0.8]);
        set_group(&mut destination, "D1", &[0.7, 0.5, 0.3, 0.1]);
        set_group(&mut destination, "Mask", &[0.25, 0.75, 0.5, 1.0]);
        let modifier_3 = modifier(
            "data_transfer",
            &json!({"object":"source","use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],"vert_mapping":"TOPOLOGY","layers_vgroup_select_src":"ALL","layers_vgroup_select_dst":"INDEX","mix_mode":mode,"mix_factor":0.4,"vertex_group":"Mask"}),
        )?;
        let actual = evaluate_with_context(
            &destination,
            &modifier_3,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &layer_selection(&["D0", "D1", "Mask"], Some(1)),
        )?;
        for name in ["D0", "D1"] {
            let error = max_error(&json!(group_weights(&actual, name)), &expected[mode][name])?;
            assert!(error < 2.0e-6, "{mode}/{name} max error {error}");
        }
    }
    let mut inverted_destination = quad()?;
    set_group(&mut inverted_destination, "D0", &[0.2, 0.4, 0.6, 0.8]);
    set_group(&mut inverted_destination, "D1", &[0.7, 0.5, 0.3, 0.1]);
    set_group(&mut inverted_destination, "Mask", &[0.25, 0.75, 0.5, 1.0]);
    let inverted_modifier = modifier(
        "data_transfer",
        &json!({"object":"source","use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],"vert_mapping":"TOPOLOGY","layers_vgroup_select_src":"ALL","layers_vgroup_select_dst":"INDEX","mix_mode":"REPLACE","mix_factor":0.4,"vertex_group":"Mask","invert_vertex_group":true}),
    )?;
    let inverted_actual = evaluate_with_context(
        &inverted_destination,
        &inverted_modifier,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &layer_selection(&["D0", "D1", "Mask"], Some(1)),
    )?;
    for name in ["D0", "D1"] {
        let error = max_error(
            &json!(group_weights(&inverted_actual, name)),
            &expected["INVERT_MASK"][name],
        )?;
        assert!(error < 2.0e-6, "inverted mask/{name} max error {error}");
    }
    let mut destination = quad()?;
    set_group(&mut destination, "D0", &[0.2; 4]);
    set_group(&mut destination, "D1", &[0.3; 4]);
    let active_modifier = modifier(
        "data_transfer",
        &json!({"object":"source","use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],"vert_mapping":"TOPOLOGY","layers_vgroup_select_src":"ACTIVE","layers_vgroup_select_dst":"ACTIVE","mix_mode":"REPLACE","mix_factor":1.0}),
    )?;
    let actual = evaluate_with_context(
        &destination,
        &active_modifier,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &layer_selection(&["D0", "D1"], Some(1)),
    )?;
    for name in ["D0", "D1"] {
        let error = max_error(
            &json!(group_weights(&actual, name)),
            &expected["ACTIVE"][name],
        )?;
        assert!(error < 1.0e-6, "ACTIVE/{name} max error {error}");
    }
    let named_modifier = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],
            "vert_mapping":"TOPOLOGY","layers_vgroup_select_src":"A",
            "layers_vgroup_select_dst":"D1","mix_mode":"REPLACE","mix_factor":1.0
        }),
    )?;
    let named_actual = evaluate_with_context(
        &destination,
        &named_modifier,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &layer_selection(&["D0", "D1"], Some(1)),
    )?;
    assert_eq!(
        group_weights(&named_actual, "D1"),
        group_weights(&actual, "D1")
    );
    let missing_source = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],
            "vert_mapping":"TOPOLOGY","layers_vgroup_select_src":"MissingGroup",
            "layers_vgroup_select_dst":"D1"
        }),
    )?;
    assert!(
        evaluate_with_context(
            &destination,
            &missing_source,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &layer_selection(&["D0", "D1"], Some(1))
        )
        .is_err()
    );
    let missing_destination = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],
            "vert_mapping":"TOPOLOGY","layers_vgroup_select_src":"A",
            "layers_vgroup_select_dst":"MissingGroup"
        }),
    )?;
    assert!(
        evaluate_with_context(
            &destination,
            &missing_destination,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &layer_selection(&["D0", "D1"], Some(1))
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn vertex_group_bone_selectors_filter_source_layers() -> Result<(), Box<dyn Error>> {
    let positions = vec![DVec3::ZERO, DVec3::X, DVec3::Y];
    let make_mesh = || Mesh::from_positions_and_faces(positions.clone(), vec![vec![0, 1, 2]]);
    let mut source = make_mesh()?;
    let source_weights = [
        ("BoneA", vec![0.1, 0.2, 0.3]),
        ("BoneB", vec![0.4, 0.5, 0.6]),
        ("Ordinary", vec![0.7, 0.8, 0.9]),
    ];
    for (name, weights) in &source_weights {
        set_group(&mut source, name, weights);
    }
    let mut source_operand = operand(
        source,
        DMat4::IDENTITY,
        &["BoneA", "BoneB", "Ordinary"],
        None,
    );
    source_operand.selected_bones = BTreeSet::from(["BoneB".to_owned()]);
    source_operand.deforming_bones = BTreeSet::from(["BoneA".to_owned()]);
    let operands = BTreeMap::from([("source".to_owned(), vec![source_operand])]);
    let original = [
        ("D0", vec![0.2, 0.2, 0.2]),
        ("D1", vec![0.3, 0.3, 0.3]),
        ("D2", vec![0.4, 0.4, 0.4]),
    ];
    for (selection, target, source_index) in [("BONE_SELECT", "D1", 1), ("BONE_DEFORM", "D0", 0)] {
        let mut destination = make_mesh()?;
        for (name, weights) in &original {
            set_group(&mut destination, name, weights);
        }
        let transfer = modifier(
            "data_transfer",
            &json!({
                "object":"source","use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],
                "vert_mapping":"TOPOLOGY","layers_vgroup_select_src":selection,
                "layers_vgroup_select_dst":"INDEX","mix_mode":"REPLACE","mix_factor":1.0,
            }),
        )?;
        let actual = evaluate_with_context(
            &destination,
            &transfer,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &layer_selection(&["D0", "D1", "D2"], None),
        )?;
        for (index, (name, prior)) in original.iter().enumerate() {
            let expected = if *name == target {
                &source_weights[source_index].1
            } else {
                prior
            };
            assert_eq!(
                group_weights(&actual, name),
                *expected,
                "{selection} should filter source group layers"
            );
            if index == source_index {
                assert_eq!(*name, target);
            }
        }
    }
    Ok(())
}

#[test]
fn absent_transfer_source_is_an_input_mesh_noop() -> Result<(), Box<dyn Error>> {
    let destination =
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])?;
    let transfer = modifier("data_transfer", &json!({"object":null}))?;
    let actual = evaluate_with_context(
        &destination,
        &transfer,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    assert_eq!(actual.attributes, destination.attributes);
    assert_eq!(
        actual
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>(),
        destination
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        actual
            .edges
            .iter()
            .map(|edge| edge.vertices)
            .collect::<Vec<_>>(),
        destination
            .edges
            .iter()
            .map(|edge| edge.vertices)
            .collect::<Vec<_>>(),
    );
    Ok(())
}

#[test]
fn cli_applies_and_inspects_data_transfer_modifier() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let root = directory.path();
    let scene = root.join("scene");
    let operations = root.join("transfer.json");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .args(["--json"])
        .output()?;
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stdout)
    );
    let mut ops = vec![
        json!({"op":"node.create","id":"source","kind":"box","params":{"size":1.0}}),
        json!({"op":"node.create","id":"destination","kind":"box","params":{"size":1.0}}),
        json!({"op":"vertex_group.create","target":{"id":"source"},"id":"source_group","name":"A"}),
        json!({"op":"vertex_group.create","target":{"id":"destination"},"id":"destination_group","name":"A"}),
    ];
    let weights = (0..8)
        .map(|vertex_id| json!({"vertex_id":vertex_id,"weight":0.1+vertex_id as f64*0.1}))
        .collect::<Vec<_>>();
    ops.push(json!({"op":"vertex_group.assign","target":{"id":"source"},"group_id":"source_group","weights":weights}));
    ops.push(json!({"op":"vertex_group.assign","target":{"id":"destination"},"group_id":"destination_group","weights":[{"vertex_id":0,"weight":0.25}]}));
    ops.push(json!({"op":"modifier.create","target":{"id":"destination"},"id":"transfer","type":"data_transfer","params":{"object":"source","use_object_transform":true,"use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],"vert_mapping":"TOPOLOGY","mix_mode":"REPLACE","mix_factor":1.0}}));
    ops.push(json!({"op":"modifier.apply","target":{"id":"destination"},"modifier_id":"transfer"}));
    fs::write(
        &operations,
        serde_json::to_vec(&json!({"schema_version":1,"base_revision":0,"operations":ops}))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&operations)
        .args(["--json"])
        .output()?;
    fs::write(root.join("apply-output.json"), &applied.stdout)?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let inspected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id", "destination", "--json"])
        .output()?;
    fs::write(root.join("inspect-output.json"), &inspected.stdout)?;
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stdout)
    );
    let document: Value = serde_json::from_slice(&inspected.stdout)?;
    assert!(
        document["result"]["items"]
            .as_array()
            .is_some_and(|items| !items.is_empty())
    );
    Ok(())
}

#[test]
fn cli_applies_null_source_transfer_without_changing_persisted_mesh() -> Result<(), Box<dyn Error>>
{
    let directory = tempdir()?;
    let root = directory.path();
    let scene = root.join("scene");
    let init = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .args(["--json"])
        .output()?;
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stdout)
    );
    let create_file = root.join("create.json");
    fs::write(
        &create_file,
        serde_json::to_vec(&json!({
            "schema_version":1,"base_revision":0,
            "operations":[{"op":"node.create","id":"destination","kind":"box","params":{"size":1.0}}],
        }))?,
    )?;
    let created = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&create_file)
        .args(["--json"])
        .output()?;
    fs::write(root.join("create-output.json"), &created.stdout)?;
    assert!(
        created.status.success(),
        "{}",
        String::from_utf8_lossy(&created.stdout)
    );
    let before_output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id", "destination", "--json"])
        .output()?;
    fs::write(
        root.join("before-inspect-output.json"),
        &before_output.stdout,
    )?;
    assert!(
        before_output.status.success(),
        "{}",
        String::from_utf8_lossy(&before_output.stdout)
    );
    let before: Value = serde_json::from_slice(&before_output.stdout)?;
    let geometry_before = before["result"]["items"]
        .get(0)
        .and_then(|item| item.get("evaluated_geometry"))
        .ok_or("pre-transfer mesh geometry is missing")?
        .clone();
    let base_revision = before["scene"]["revision"]
        .as_u64()
        .ok_or("scene revision is missing")?;
    let transfer_file = root.join("null-source.json");
    fs::write(
        &transfer_file,
        serde_json::to_vec(&json!({
            "schema_version":1,"base_revision":base_revision,
            "operations":[
                {"op":"modifier.create","target":{"id":"destination"},"id":"null_transfer",
                 "type":"data_transfer","params":{
                    "object":null,"use_vert_data":true,"data_types_verts":["VGROUP_WEIGHTS"],
                    "vert_mapping":"TOPOLOGY","mix_mode":"REPLACE","mix_factor":1.0,
                 }},
                {"op":"modifier.apply","target":{"id":"destination"},"modifier_id":"null_transfer"},
            ],
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&transfer_file)
        .args(["--json"])
        .output()?;
    fs::write(root.join("null-apply-output.json"), &applied.stdout)?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let after_output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id", "destination", "--json"])
        .output()?;
    fs::write(root.join("null-inspect-output.json"), &after_output.stdout)?;
    assert!(
        after_output.status.success(),
        "{}",
        String::from_utf8_lossy(&after_output.stdout)
    );
    let after: Value = serde_json::from_slice(&after_output.stdout)?;
    let geometry_after = after["result"]["items"]
        .get(0)
        .and_then(|item| item.get("evaluated_geometry"))
        .ok_or("post-transfer mesh geometry is missing")?;
    assert_eq!(
        geometry_after, &geometry_before,
        "null-source DataTransfer must preserve the evaluated mesh"
    );
    Ok(())
}

const BLENDER_UV_LAYER_SCRIPT: &str = r"
import bpy, json, os, sys
root=os.path.realpath(sys.argv[sys.argv.index('--')+1])
def make(name):
    mesh=bpy.data.meshes.new(name+'Mesh')
    mesh.from_pydata([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],[],[(0,1,2,3)])
    mesh.update(); obj=bpy.data.objects.new(name,mesh); bpy.context.collection.objects.link(obj); return obj
def add_layer(obj,name,offset):
    layer=obj.data.uv_layers.new(name=name)
    for loop in obj.data.loops:
        vertex=obj.data.vertices[loop.vertex_index].co
        layer.data[loop.index].uv=(vertex.x+offset,vertex.y+offset*0.5)
def add_color(obj,name,domain,offset):
    layer=obj.data.color_attributes.new(name=name,type='FLOAT_COLOR',domain=domain)
    for i,item in enumerate(layer.data): item.color=(offset+i*0.1,0.25+offset,0.4-offset,1.0)
source=make('Source'); add_layer(source,'Source0',0.2); add_layer(source,'Source1',0.7)
source.data.uv_layers.active_index=1
add_color(source,'Color0','POINT',0.1); add_color(source,'Color1','POINT',0.5)
add_color(source,'Corner0','CORNER',0.15); add_color(source,'Corner1','CORNER',0.55)
destination=make('Destination'); add_layer(destination,'Destination0',-0.2); add_layer(destination,'Destination1',-0.6)
destination.data.uv_layers.active_index=0
mod=destination.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source
mod.use_loop_data=True; mod.data_types_loops={'UV'}; mod.loop_mapping='TOPOLOGY'
mod.layers_uv_select_src='ALL'; mod.layers_uv_select_dst='INDEX'; mod.mix_mode='ADD'; mod.mix_factor=0.25
bpy.context.view_layer.update()
evaluated=destination.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh=evaluated.to_mesh()
result={name:[list(item.uv) for item in mesh.uv_layers[name].data] for name in ('Destination0','Destination1')}
evaluated.to_mesh_clear()
active=make('Active'); add_layer(active,'Destination0',-0.2); add_layer(active,'Destination1',-0.6)
active.data.uv_layers.active_index=1
bpy.context.view_layer.objects.active=active; active.select_set(True)
mod=active.modifiers.new('ActiveTransfer','DATA_TRANSFER'); mod.object=source
mod.use_loop_data=True; mod.data_types_loops={'UV'}; mod.loop_mapping='TOPOLOGY'
mod.layers_uv_select_src='Source1'; mod.layers_uv_select_dst='Destination1'; mod.mix_mode='REPLACE'; mod.mix_factor=1.0
bpy.context.view_layer.update()
evaluated=active.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh=evaluated.to_mesh()
result['ACTIVE']={name:[list(item.uv) for item in mesh.uv_layers[name].data] for name in ('Destination0','Destination1')}
evaluated.to_mesh_clear()
point_dest=make('PointColor'); add_color(point_dest,'Dest0','POINT',-0.1); add_color(point_dest,'Dest1','POINT',-0.4)
mod=point_dest.modifiers.new('PointColorTransfer','DATA_TRANSFER'); mod.object=source
mod.use_vert_data=True; mod.data_types_verts={'COLOR_VERTEX'}; mod.vert_mapping='TOPOLOGY'
mod.layers_vcol_vert_select_src='ALL'; mod.layers_vcol_vert_select_dst='INDEX'; mod.mix_mode='REPLACE'; mod.mix_factor=1.0
bpy.context.view_layer.update(); evaluated=point_dest.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh=evaluated.to_mesh()
result['POINT_COLOR']={name:[list(item.color) for item in mesh.color_attributes[name].data] for name in ('Dest0','Dest1')}
evaluated.to_mesh_clear()
corner_dest=make('CornerColor'); add_color(corner_dest,'Dest0','CORNER',-0.1); add_color(corner_dest,'Dest1','CORNER',-0.4)
mod=corner_dest.modifiers.new('CornerColorTransfer','DATA_TRANSFER'); mod.object=source
mod.use_loop_data=True; mod.data_types_loops={'COLOR_CORNER'}; mod.loop_mapping='TOPOLOGY'
mod.layers_vcol_loop_select_src='ALL'; mod.layers_vcol_loop_select_dst='INDEX'; mod.mix_mode='REPLACE'; mod.mix_factor=1.0
bpy.context.view_layer.update(); evaluated=corner_dest.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh=evaluated.to_mesh()
result['CORNER_COLOR']={name:[list(item.color) for item in mesh.color_attributes[name].data] for name in ('Dest0','Dest1')}
evaluated.to_mesh_clear()
with open(os.path.join(root,'uv-layers-oracle.json'),'w') as f: json.dump(result,f)
";

#[test]
fn uv_layer_index_mapping_keeps_full_loop_order_and_replace_factor() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping data-transfer UV Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "uv-layers.py",
        BLENDER_UV_LAYER_SCRIPT,
        root,
        "Blender data-transfer UV oracle failed",
    )?;
    let mut expected: Value =
        serde_json::from_slice(&fs::read(root.join("uv-layers-oracle.json"))?)?;
    flip_blender_uv_to_potter(&mut expected);
    let mut source = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::ONE, DVec3::Y],
        vec![vec![0, 1, 2, 3]],
    )?;
    set_uv_layers(
        &mut source,
        &[
            (
                "Source0",
                vec![vec![[0.2, 0.1], [1.2, 0.1], [1.2, 1.1], [0.2, 1.1]]],
            ),
            (
                "Source1",
                vec![vec![[0.7, 0.35], [1.7, 0.35], [1.7, 1.35], [0.7, 1.35]]],
            ),
        ],
        "Source1",
    );
    let mut destination = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::ONE, DVec3::Y],
        vec![vec![0, 1, 2, 3]],
    )?;
    set_uv_layers(
        &mut destination,
        &[
            (
                "Destination0",
                vec![vec![[-0.2, -0.1], [0.8, -0.1], [0.8, 0.9], [-0.2, 0.9]]],
            ),
            (
                "Destination1",
                vec![vec![[-0.6, -0.3], [0.4, -0.3], [0.4, 0.7], [-0.6, 0.7]]],
            ),
        ],
        "Destination0",
    );
    set_color_layer(
        &mut source,
        "Color0",
        "point",
        "FLOAT_COLOR",
        vec![
            [0.1, 0.35, 0.3, 1.0],
            [0.2, 0.35, 0.3, 1.0],
            [0.3, 0.35, 0.3, 1.0],
            [0.4, 0.35, 0.3, 1.0],
        ],
    )?;
    set_color_layer(
        &mut source,
        "Color1",
        "point",
        "FLOAT_COLOR",
        vec![
            [0.5, 0.75, -0.1, 1.0],
            [0.6, 0.75, -0.1, 1.0],
            [0.7, 0.75, -0.1, 1.0],
            [0.8, 0.75, -0.1, 1.0],
        ],
    )?;
    set_color_layer(
        &mut source,
        "Corner0",
        "corner",
        "FLOAT_COLOR",
        vec![
            [0.15, 0.4, 0.25, 1.0],
            [0.25, 0.4, 0.25, 1.0],
            [0.35, 0.4, 0.25, 1.0],
            [0.45, 0.4, 0.25, 1.0],
        ],
    )?;
    set_color_layer(
        &mut source,
        "Corner1",
        "corner",
        "FLOAT_COLOR",
        vec![
            [0.55, 0.8, -0.15, 1.0],
            [0.65, 0.8, -0.15, 1.0],
            [0.75, 0.8, -0.15, 1.0],
            [0.85, 0.8, -0.15, 1.0],
        ],
    )?;
    let source_operand = operand(source.clone(), DMat4::IDENTITY, &[], None);
    let mut operands = BTreeMap::new();
    operands.insert("object".to_owned(), vec![source_operand]);
    let transfer = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_object_transform":true,"use_loop_data":true,"data_types_loops":["UV"],
            "loop_mapping":"TOPOLOGY","layers_uv_select_src":"ALL","layers_uv_select_dst":"INDEX",
            "mix_mode":"ADD","mix_factor":0.25
        }),
    )?;
    let actual = evaluate_with_context(
        &destination,
        &transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    let uv_values = |mesh: &Mesh, name: &str| -> Value { uv_layer_values(mesh, name) };
    for name in ["Destination0", "Destination1"] {
        let error = max_error(&uv_values(&actual, name), &expected[name])?;
        assert!(error < 1.0e-6, "UV layer {name} max error {error}");
    }
    let mut active_destination = destination.clone();
    active_destination.attributes["uv_layers"]["active"] = json!("Destination1");
    let active_transfer = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_object_transform":true,"use_loop_data":true,"data_types_loops":["UV"],
            "loop_mapping":"TOPOLOGY","layers_uv_select_src":"ACTIVE","layers_uv_select_dst":"ACTIVE",
            "mix_mode":"REPLACE","mix_factor":1.0
        }),
    )?;
    let active_actual = evaluate_with_context(
        &active_destination,
        &active_transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    for name in ["Destination0", "Destination1"] {
        let error = max_error(&uv_values(&active_actual, name), &expected["ACTIVE"][name])?;
        assert!(error < 1.0e-6, "ACTIVE UV layer {name} max error {error}");
    }
    let mut no_active_destination = active_destination.clone();
    no_active_destination.attributes["uv_layers"]["active"] = Value::Null;
    let no_active_destination_actual = evaluate_with_context(
        &no_active_destination,
        &active_transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    assert_eq!(
        uv_values(&no_active_destination_actual, "Destination1"),
        uv_values(&no_active_destination, "Destination1"),
        "ACTIVE UV destination selection must no-op without an active layer"
    );
    let mut no_active_source = source.clone();
    no_active_source.attributes["uv_layers"]["active"] = Value::Null;
    let no_active_source_operands = BTreeMap::from([(
        "object".to_owned(),
        vec![operand(no_active_source, DMat4::IDENTITY, &[], None)],
    )]);
    let no_active_source_actual = evaluate_with_context(
        &active_destination,
        &active_transfer,
        &no_active_source_operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    assert_eq!(
        uv_values(&no_active_source_actual, "Destination1"),
        uv_values(&active_destination, "Destination1"),
        "ACTIVE UV source selection must no-op without an active layer"
    );
    let named_transfer = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_object_transform":true,"use_loop_data":true,"data_types_loops":["UV"],
            "loop_mapping":"TOPOLOGY","layers_uv_select_src":"Source1",
            "layers_uv_select_dst":"Destination1","mix_mode":"REPLACE","mix_factor":1.0
        }),
    )?;
    let named_actual = evaluate_with_context(
        &active_destination,
        &named_transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    for name in ["Destination0", "Destination1"] {
        assert_eq!(uv_values(&named_actual, name), expected["ACTIVE"][name]);
    }
    let missing_uv_source = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_loop_data":true,"data_types_loops":["UV"],
            "loop_mapping":"TOPOLOGY","layers_uv_select_src":"MissingUV",
            "layers_uv_select_dst":"NAME"
        }),
    )?;
    assert!(
        evaluate_with_context(
            &active_destination,
            &missing_uv_source,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &AttributeLayerSelection::default()
        )
        .is_err()
    );
    let missing_uv_destination = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_loop_data":true,"data_types_loops":["UV"],
            "loop_mapping":"TOPOLOGY","layers_uv_select_src":"Source1",
            "layers_uv_select_dst":"MissingUV"
        }),
    )?;
    assert!(
        evaluate_with_context(
            &active_destination,
            &missing_uv_destination,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &AttributeLayerSelection::default()
        )
        .is_err()
    );
    let mut point_destination = destination.clone();
    set_color_layer(
        &mut point_destination,
        "Dest0",
        "point",
        "FLOAT_COLOR",
        vec![
            [-0.1, 0.15, 0.5, 1.0],
            [0.0, 0.15, 0.5, 1.0],
            [0.1, 0.15, 0.5, 1.0],
            [0.2, 0.15, 0.5, 1.0],
        ],
    )?;
    set_color_layer(
        &mut point_destination,
        "Dest1",
        "point",
        "FLOAT_COLOR",
        vec![
            [-0.4, -0.15, 0.8, 1.0],
            [-0.3, -0.15, 0.8, 1.0],
            [-0.2, -0.15, 0.8, 1.0],
            [-0.1, -0.15, 0.8, 1.0],
        ],
    )?;
    let point_transfer = modifier(
        "data_transfer",
        &json!({"object":"source","use_vert_data":true,"data_types_verts":["COLOR_VERTEX"],"vert_mapping":"TOPOLOGY","layers_vcol_vert_select_src":"ALL","layers_vcol_vert_select_dst":"INDEX","mix_mode":"REPLACE","mix_factor":1.0}),
    )?;
    let point_selection = layer_selection(&[], None);
    let point_actual = evaluate_with_context(
        &point_destination,
        &point_transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &point_selection,
    )?;
    for name in ["Dest0", "Dest1"] {
        let error = max_error(
            &color_values(&point_actual, name),
            &expected["POINT_COLOR"][name],
        )?;
        assert!(error < 1.0e-6, "point color layer {name} max error {error}");
    }
    let mut corner_destination = destination.clone();
    set_color_layer(
        &mut corner_destination,
        "Dest0",
        "corner",
        "FLOAT_COLOR",
        vec![
            [-0.1, 0.15, 0.5, 1.0],
            [0.0, 0.15, 0.5, 1.0],
            [0.1, 0.15, 0.5, 1.0],
            [0.2, 0.15, 0.5, 1.0],
        ],
    )?;
    set_color_layer(
        &mut corner_destination,
        "Dest1",
        "corner",
        "FLOAT_COLOR",
        vec![
            [-0.4, -0.15, 0.8, 1.0],
            [-0.3, -0.15, 0.8, 1.0],
            [-0.2, -0.15, 0.8, 1.0],
            [-0.1, -0.15, 0.8, 1.0],
        ],
    )?;
    let corner_transfer = modifier(
        "data_transfer",
        &json!({"object":"source","use_loop_data":true,"data_types_loops":["COLOR_CORNER"],"loop_mapping":"TOPOLOGY","layers_vcol_loop_select_src":"ALL","layers_vcol_loop_select_dst":"INDEX","mix_mode":"REPLACE","mix_factor":1.0}),
    )?;
    let corner_selection = layer_selection(&[], None);
    let corner_actual = evaluate_with_context(
        &corner_destination,
        &corner_transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &corner_selection,
    )?;
    for name in ["Dest0", "Dest1"] {
        let error = max_error(
            &color_values(&corner_actual, name),
            &expected["CORNER_COLOR"][name],
        )?;
        assert!(
            error < 1.0e-6,
            "corner color layer {name} max error {error}"
        );
    }
    Ok(())
}

const BLENDER_CUSTOM_NORMAL_TRANSFER_SCRIPT: &str = r"
import bpy, json, os, sys
from mathutils import Vector
root=os.path.realpath(sys.argv[sys.argv.index('--')+1])
verts=[(0,0,0),(1,0,0),(1,1,0),(0,-1,1)]
faces=[(0,1,2),(1,0,3)]
def make(name):
    data=bpy.data.meshes.new(name+'Mesh'); data.from_pydata(verts,[],faces); data.update()
    for polygon in data.polygons: polygon.use_smooth=True
    obj=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(obj); return obj
def custom(obj,vectors):
    obj.data.normals_split_custom_set([tuple(Vector(value).normalized()) for value in vectors])
    return [list(item.value) for item in obj.data.attributes['custom_normal'].data]
source=make('Source')
source_codes=custom(source,[(0.1,0.0,0.995),(0.0,0.1,0.995),(0.2,0.1,0.975),
                            (0.0,0.1,0.995),(0.1,0.0,0.995),(-0.1,0.2,0.975)])
destination=make('Destination')
destination_codes=custom(destination,[(-0.15,0.0,0.989),(0.0,-0.1,0.995),(0.1,-0.2,0.975),
                                      (0.0,-0.1,0.995),(-0.15,0.0,0.989),(0.2,-0.1,0.975)])
mod=destination.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source
mod.use_loop_data=True; mod.data_types_loops={'CUSTOM_NORMAL'}; mod.loop_mapping='TOPOLOGY'
mod.mix_mode='MIX'; mod.mix_factor=0.35
bpy.context.view_layer.update()
evaluated=destination.evaluated_get(bpy.context.evaluated_depsgraph_get()); output=evaluated.to_mesh()
with open(os.path.join(root,'custom-normal-oracle.json'),'w') as f:
    json.dump({'source_codes':source_codes,'destination_codes':destination_codes,
               'output_normals':[list(loop.normal) for loop in output.loops]},f)
evaluated.to_mesh_clear()
";

#[test]
#[expect(
    clippy::redundant_closure_for_method_calls,
    reason = "the iterator yields references, so to_array needs the closure to read the value"
)]
fn custom_normal_transfer_keeps_typed_codes_and_corner_order() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!(
            "skipping data-transfer custom-normal Blender parity test: Blender is unavailable"
        );
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "custom-normal-transfer.py",
        BLENDER_CUSTOM_NORMAL_TRANSFER_SCRIPT,
        root,
        "Blender data-transfer custom-normal oracle failed",
    )?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("custom-normal-oracle.json"))?)?;
    let positions = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(1.0, 1.0, 0.0),
        DVec3::new(0.0, -1.0, 1.0),
    ];
    let faces = vec![vec![0, 1, 2], vec![1, 0, 3]];
    let mut source = Mesh::from_positions_and_faces(positions.clone(), faces.clone())?;
    let mut destination = Mesh::from_positions_and_faces(positions.clone(), faces)?;
    set_smooth_faces(&mut source);
    set_smooth_faces(&mut destination);
    set_custom_normal_codes(&mut source, &expected["source_codes"])?;
    set_custom_normal_codes(&mut destination, &expected["destination_codes"])?;
    let operands = BTreeMap::from([(
        "object".to_owned(),
        vec![operand(source, DMat4::IDENTITY, &[], None)],
    )]);
    let transfer = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_loop_data":true,"data_types_loops":["CUSTOM_NORMAL"],
            "loop_mapping":"TOPOLOGY","mix_mode":"MIX","mix_factor":0.35
        }),
    )?;
    let actual = evaluate_with_context(
        &destination,
        &transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    let actual_normals = potter_core::geom::corner_normal_vectors(&actual)?;
    let actual_values = actual_normals
        .iter()
        .map(|normal| normal.to_array())
        .collect::<Vec<_>>();
    let error = max_error(&json!(actual_values), &expected["output_normals"])?;
    assert!(error < 2.0e-4, "CUSTOM_NORMAL max error {error}");
    assert_eq!(actual.attributes["custom_normal"]["type"], "normal_code");
    Ok(())
}

const BLENDER_COLOR_MIX_SCRIPT: &str = r"
import bpy, json, os, sys
root=os.path.realpath(sys.argv[sys.argv.index('--')+1])
def mesh(name):
    data=bpy.data.meshes.new(name+'Mesh')
    data.from_pydata([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],[],[(0,1,2,3)])
    data.update(); obj=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(obj); return obj
def colors(obj,name,data_type,values):
    layer=obj.data.color_attributes.new(name=name,type=data_type,domain='POINT')
    for item,color in zip(layer.data,values): item.color=color
    return [list(item.color) for item in layer.data]
source=mesh('Source')
source_input={
    'FloatColor':colors(source,'FloatColor','FLOAT_COLOR',[(0.85,0.2,0.1,0.25),(0.1,0.75,0.3,0.8),(0.6,0.15,0.9,0.45),(0.3,0.5,0.2,0.0)]),
    'ByteColor':colors(source,'ByteColor','BYTE_COLOR',[(0.85,0.2,0.1,0.25),(0.1,0.75,0.3,0.8),(0.6,0.15,0.9,0.45),(0.3,0.5,0.2,0.0)])
}
destination_input=None
result={}
for mode in ('REPLACE','ABOVE_THRESHOLD','BELOW_THRESHOLD','MIX','ADD','SUB','MUL'):
    destination=mesh('Destination'+mode)
    values={
        'FloatColor':colors(destination,'FloatColor','FLOAT_COLOR',[(0.1,0.2,0.3,0.35),(0.85,0.65,0.5,0.7),(0.2,0.2,0.2,0.0),(0.8,0.2,0.1,0.5)]),
        'ByteColor':colors(destination,'ByteColor','BYTE_COLOR',[(0.1,0.2,0.3,0.35),(0.85,0.65,0.5,0.7),(0.2,0.2,0.2,0.0),(0.8,0.2,0.1,0.5)])
    }
    if destination_input is None: destination_input=values
    mod=destination.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source
    mod.use_vert_data=True; mod.data_types_verts={'COLOR_VERTEX'}; mod.vert_mapping='TOPOLOGY'
    mod.layers_vcol_vert_select_src='ALL'; mod.layers_vcol_vert_select_dst='NAME'
    mod.mix_mode=mode; mod.mix_factor=0.4
    bpy.context.view_layer.update()
    evaluated=destination.evaluated_get(bpy.context.evaluated_depsgraph_get()); output=evaluated.to_mesh()
    result[mode]={name:[list(item.color) for item in output.color_attributes[name].data] for name in ('FloatColor','ByteColor')}
    evaluated.to_mesh_clear()
with open(os.path.join(root,'color-mix-oracle.json'),'w') as f:
    json.dump({'source':source_input,'destination':destination_input,'outputs':result},f)
";

#[test]
fn float_and_byte_color_mix_modes_match_blender() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping data-transfer color Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "color-mix.py",
        BLENDER_COLOR_MIX_SCRIPT,
        root,
        "Blender data-transfer color mix oracle failed",
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("color-mix-oracle.json"))?)?;
    let colors = rgba_values;
    let positions = vec![DVec3::ZERO, DVec3::X, DVec3::ONE, DVec3::Y];
    let mut source = Mesh::from_positions_and_faces(positions.clone(), vec![vec![0, 1, 2, 3]])?;
    for (name, data_type) in [("FloatColor", "FLOAT_COLOR"), ("ByteColor", "BYTE_COLOR")] {
        set_color_layer(
            &mut source,
            name,
            "point",
            data_type,
            colors(&expected["source"][name])?,
        )?;
    }
    let operands = BTreeMap::from([(
        "object".to_owned(),
        vec![operand(source, DMat4::IDENTITY, &[], None)],
    )]);
    for mode in [
        "REPLACE",
        "ABOVE_THRESHOLD",
        "BELOW_THRESHOLD",
        "MIX",
        "ADD",
        "SUB",
        "MUL",
    ] {
        let mut destination =
            Mesh::from_positions_and_faces(positions.clone(), vec![vec![0, 1, 2, 3]])?;
        for (name, data_type) in [("FloatColor", "FLOAT_COLOR"), ("ByteColor", "BYTE_COLOR")] {
            set_color_layer(
                &mut destination,
                name,
                "point",
                data_type,
                colors(&expected["destination"][name])?,
            )?;
        }
        let transfer = modifier(
            "data_transfer",
            &json!({
                "object":"source","use_vert_data":true,"data_types_verts":["COLOR_VERTEX"],
                "vert_mapping":"TOPOLOGY","layers_vcol_vert_select_src":"ALL",
                "layers_vcol_vert_select_dst":"NAME","mix_mode":mode,"mix_factor":0.4
            }),
        )?;
        let actual = evaluate_with_context(
            &destination,
            &transfer,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &AttributeLayerSelection::default(),
        )?;
        for name in ["FloatColor", "ByteColor"] {
            let error = max_error(
                &color_values(&actual, name),
                &expected["outputs"][mode][name],
            )?;
            assert!(error < 1.0e-6, "{name} {mode} max error {error}");
        }
    }
    Ok(())
}

const BLENDER_ACTIVE_COLOR_SCRIPT: &str = r"
import bpy, json, os, sys
root=os.path.realpath(sys.argv[sys.argv.index('--')+1])
def mesh(name):
    data=bpy.data.meshes.new(name+'Mesh'); data.from_pydata([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],[],[(0,1,2,3)]); data.update()
    obj=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(obj); return obj
source=mesh('Source'); source_layer=source.data.color_attributes.new(name='ActiveColor',type='FLOAT_COLOR',domain='POINT')
source_values=[(0.15,0.35,0.7,0.4),(0.65,0.1,0.25,0.8),(0.2,0.8,0.1,0.6),(0.9,0.3,0.45,1.0)]
for item,color in zip(source_layer.data,source_values): item.color=color
destination=mesh('Destination'); destination_layer=destination.data.color_attributes.new(name='ActiveColor',type='FLOAT_COLOR',domain='POINT')
destination_values=[(0.3,0.1,0.2,0.6),(0.1,0.5,0.4,0.7),(0.8,0.2,0.1,0.5),(0.4,0.6,0.2,0.3)]
for item,color in zip(destination_layer.data,destination_values): item.color=color
mod=destination.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source
mod.use_vert_data=True; mod.data_types_verts={'COLOR_VERTEX'}; mod.vert_mapping='TOPOLOGY'
mod.layers_vcol_vert_select_src='ALL'; mod.layers_vcol_vert_select_dst='NAME'
mod.mix_mode='REPLACE'; mod.mix_factor=1.0
bpy.context.view_layer.update()
evaluated=destination.evaluated_get(bpy.context.evaluated_depsgraph_get()); output=evaluated.to_mesh()
with open(os.path.join(root,'active-color-oracle.json'),'w') as f:
    json.dump({'source':[list(item.color) for item in source_layer.data],
               'destination':[list(item.color) for item in destination_layer.data],
               'output':[list(item.color) for item in output.color_attributes['ActiveColor'].data]},f)
evaluated.to_mesh_clear()
";

#[test]
fn active_color_store_selectors_match_blender_all_name_path() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!(
            "skipping data-transfer active color Blender parity test: Blender is unavailable"
        );
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "active-color.py",
        BLENDER_ACTIVE_COLOR_SCRIPT,
        root,
        "Blender active-color transfer oracle failed",
    )?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("active-color-oracle.json"))?)?;
    let positions = vec![DVec3::ZERO, DVec3::X, DVec3::ONE, DVec3::Y];
    let mut source = Mesh::from_positions_and_faces(positions.clone(), vec![vec![0, 1, 2, 3]])?;
    let mut destination = Mesh::from_positions_and_faces(positions, vec![vec![0, 1, 2, 3]])?;
    let colors = rgba_values;
    let inactive_source = vec![
        [0.91, 0.02, 0.11, 0.7],
        [0.81, 0.12, 0.21, 0.8],
        [0.71, 0.22, 0.31, 0.9],
        [0.61, 0.32, 0.41, 1.0],
    ];
    let inactive_destination = vec![
        [0.51, 0.42, 0.33, 0.4],
        [0.41, 0.32, 0.23, 0.5],
        [0.31, 0.22, 0.13, 0.6],
        [0.21, 0.12, 0.03, 0.7],
    ];
    set_color_layer(
        &mut source,
        "InactiveColor",
        "point",
        "FLOAT_COLOR",
        inactive_source,
    )?;
    set_color_layer(
        &mut source,
        "ActiveColor",
        "point",
        "FLOAT_COLOR",
        colors(&expected["source"])?,
    )?;
    source
        .attributes
        .get_mut("color_layers")
        .ok_or("source ColorStore missing")?["active"] = json!("ActiveColor");
    set_color_layer(
        &mut destination,
        "InactiveColor",
        "point",
        "FLOAT_COLOR",
        inactive_destination.clone(),
    )?;
    set_color_layer(
        &mut destination,
        "ActiveColor",
        "point",
        "FLOAT_COLOR",
        colors(&expected["destination"])?,
    )?;
    destination
        .attributes
        .get_mut("color_layers")
        .ok_or("destination ColorStore missing")?["active"] = json!("ActiveColor");
    let operands = BTreeMap::from([(
        "object".to_owned(),
        vec![operand(source.clone(), DMat4::IDENTITY, &[], None)],
    )]);
    let transfer = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_vert_data":true,"data_types_verts":["COLOR_VERTEX"],
            "vert_mapping":"TOPOLOGY","layers_vcol_vert_select_src":"ACTIVE",
            "layers_vcol_vert_select_dst":"ACTIVE","mix_mode":"REPLACE","mix_factor":1.0
        }),
    )?;
    let actual = evaluate_with_context(
        &destination,
        &transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    let error = max_error(
        &domain_attribute_values(&actual, "ActiveColor"),
        &expected["output"],
    )?;
    assert!(
        error < 1.0e-6,
        "ACTIVE color store selectors max error {error}"
    );
    assert_eq!(
        domain_attribute_values(&actual, "InactiveColor"),
        json!(inactive_destination),
        "ACTIVE selection must leave the first, inactive color layer untouched"
    );
    let mut no_active_destination = destination.clone();
    no_active_destination
        .attributes
        .get_mut("color_layers")
        .ok_or("destination ColorStore missing")?["active"] = Value::Null;
    let no_active_actual = evaluate_with_context(
        &no_active_destination,
        &transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    assert_eq!(
        domain_attribute_values(&no_active_actual, "ActiveColor"),
        json!(colors(&expected["destination"])?),
        "ACTIVE destination selection must no-op when the ColorStore has no active layer",
    );
    let mut no_active_source = source.clone();
    no_active_source
        .attributes
        .get_mut("color_layers")
        .ok_or("source ColorStore missing")?["active"] = Value::Null;
    let no_active_source_operands = BTreeMap::from([(
        "object".to_owned(),
        vec![operand(no_active_source, DMat4::IDENTITY, &[], None)],
    )]);
    let no_active_source_actual = evaluate_with_context(
        &destination,
        &transfer,
        &no_active_source_operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    assert_eq!(
        domain_attribute_values(&no_active_source_actual, "ActiveColor"),
        json!(colors(&expected["destination"])?),
        "ACTIVE source selection must no-op when the ColorStore has no active layer",
    );
    Ok(())
}

const BLENDER_UV_ISLAND_SCRIPT: &str = r"
import bpy, json, os, sys
root=os.path.realpath(sys.argv[sys.argv.index('--')+1])
def mesh(name, verts, faces):
    data=bpy.data.meshes.new(name+'Mesh'); data.from_pydata(verts,[],faces); data.update()
    obj=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(obj); return obj
verts=[(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
faces=[(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
source=mesh('Source',verts,faces)
uv=source.data.uv_layers.new(name='UVMap')
for poly in source.data.polygons:
    for loop_index in poly.loop_indices:
        uv.data[loop_index].uv=(10.0*poly.index+loop_index*0.1,3.0*poly.index+loop_index*0.03)
source.data.edges[next(i for i,e in enumerate(source.data.edges) if set(e.vertices)=={0,4})].use_seam=True
destination_vertices=[(0,0.2,0.3),(0.2,0,0.3),(0.2,0,0.8),(0,0.2,0.8)]
result={}
for precision in (0.0,1.0):
    destination=mesh('Destination'+str(precision),destination_vertices,[(0,1,2,3)])
    target=destination.data.uv_layers.new(name='UVMap')
    for i,item in enumerate(target.data): item.uv=(-1.0-i*0.1,0.5+i*0.05)
    mod=destination.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source
    mod.use_loop_data=True; mod.data_types_loops={'UV'}; mod.loop_mapping='NEAREST_POLY'
    mod.layers_uv_select_src='ALL'; mod.layers_uv_select_dst='NAME'
    mod.mix_mode='REPLACE'; mod.mix_factor=1.0; mod.islands_precision=precision
    bpy.context.view_layer.update()
    evaluated=destination.evaluated_get(bpy.context.evaluated_depsgraph_get()); output=evaluated.to_mesh()
    result[str(int(precision))]=[list(item.uv) for item in output.uv_layers['UVMap'].data]
    evaluated.to_mesh_clear()
with open(os.path.join(root,'uv-island-oracle.json'),'w') as f: json.dump(result,f)
";

#[test]
fn uv_island_selection_and_precision_match_blender() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping data-transfer UV island Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "uv-island.py",
        BLENDER_UV_ISLAND_SCRIPT,
        root,
        "Blender data-transfer UV island oracle failed",
    )?;
    let mut expected: Value =
        serde_json::from_slice(&fs::read(root.join("uv-island-oracle.json"))?)?;
    flip_blender_uv_to_potter(&mut expected);
    let positions = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(1.0, 1.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(1.0, 0.0, 1.0),
        DVec3::new(1.0, 1.0, 1.0),
        DVec3::new(0.0, 1.0, 1.0),
    ];
    let faces = vec![
        vec![0, 1, 5, 4],
        vec![1, 2, 6, 5],
        vec![2, 3, 7, 6],
        vec![3, 0, 4, 7],
    ];
    let mut source = Mesh::from_positions_and_faces(positions.clone(), faces)?;
    let mut source_uvs = Vec::new();
    let mut loop_index = 0;
    for (face_index, face) in source.faces.iter().enumerate() {
        let mut row = Vec::new();
        for _ in 0..face.vertices.len() {
            row.push([
                10.0 * face_index as f64 + loop_index as f64 * 0.1,
                3.0 * face_index as f64 + loop_index as f64 * 0.03,
            ]);
            loop_index += 1;
        }
        source_uvs.push(row);
    }
    set_uv_layers(&mut source, &[("UVMap", source_uvs)], "UVMap");
    let seam_start = source.vertices[0].id;
    let seam_end = source.vertices[4].id;
    let seam = source
        .edges
        .iter()
        .find(|edge| edge.vertices.contains(&seam_start) && edge.vertices.contains(&seam_end))
        .ok_or("UV seam edge missing")?;
    source.attributes.insert(
        "seam".to_owned(),
        json!({
            "domain":"edge","type":"bool","values":{format!("e{}",seam.id):true}
        }),
    );
    let mut operands = BTreeMap::new();
    operands.insert(
        "object".to_owned(),
        vec![operand(source, DMat4::IDENTITY, &[], None)],
    );
    let destination_positions = vec![
        DVec3::new(0.0, 0.2, 0.3),
        DVec3::new(0.2, 0.0, 0.3),
        DVec3::new(0.2, 0.0, 0.8),
        DVec3::new(0.0, 0.2, 0.8),
    ];
    let mut expected_precision_zero: Option<Value> = None;
    for (key, precision) in [("0", 0.0), ("1", 1.0)] {
        let mut destination =
            Mesh::from_positions_and_faces(destination_positions.clone(), vec![vec![0, 1, 2, 3]])?;
        set_uv_layers(
            &mut destination,
            &[(
                "UVMap",
                vec![vec![[-1.0, 0.5], [-1.1, 0.55], [-1.2, 0.6], [-1.3, 0.65]]],
            )],
            "UVMap",
        );
        let modifier_4 = modifier(
            "data_transfer",
            &json!({
                "object":"source","use_object_transform":true,"use_loop_data":true,"data_types_loops":["UV"],
                "loop_mapping":"NEAREST_POLY","layers_uv_select_src":"ALL","layers_uv_select_dst":"NAME",
                "mix_mode":"REPLACE","mix_factor":1.0,"islands_precision":precision
            }),
        )?;
        let actual = evaluate_with_context(
            &destination,
            &modifier_4,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &AttributeLayerSelection::default(),
        )?;
        let values = uv_layer_values(&actual, "UVMap");
        let error = max_error(&values, &expected[key])?;
        assert!(
            error < 2.0e-4,
            "islands_precision={precision} max error {error}"
        );
        if precision == 0.0 {
            expected_precision_zero = Some(values);
        }
    }
    assert_ne!(
        expected_precision_zero.as_ref(),
        Some(&expected["1"]),
        "the seam path case must exercise A-star correction"
    );
    Ok(())
}

const BLENDER_LOOP_MAPPING_SCRIPT: &str = r"
import bpy, json, os, sys
root=os.path.realpath(sys.argv[sys.argv.index('--')+1])
def obj(name,verts,faces):
    mesh=bpy.data.meshes.new(name+'Mesh'); mesh.from_pydata(verts,[],faces); mesh.update()
    ob=bpy.data.objects.new(name,mesh); bpy.context.collection.objects.link(ob); return ob
source=obj('Source',[(0,0,0),(2,0,0),(2,2,0),(0,2,0)],[(0,1,2,3)])
uv=source.data.uv_layers.new(name='UVMap')
for loop in source.data.loops:
    v=source.data.vertices[loop.vertex_index].co
    uv.data[loop.index].uv=(0.2+v.x*0.5+loop.index*0.01,0.3+v.y*0.4+loop.index*0.02)
destination_vertices=[(0.3,0.2,0.1),(1.2,0.2,0.1),(1.5,1.5,0.1),(0.2,1.5,0.1)]
destination_faces=[(0,1,2),(0,2,3)]
result={}
for mapping in ('NEAREST_NORMAL','NEAREST_POLYNOR','NEAREST_POLY','POLYINTERP_NEAREST','POLYINTERP_LNORPROJ'):
    dest=obj('Dest'+mapping,destination_vertices,destination_faces)
    layer=dest.data.uv_layers.new(name='UVMap')
    for item in layer.data: item.uv=(0.0,0.0)
    mod=dest.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source; mod.use_object_transform=True
    mod.use_loop_data=True; mod.data_types_loops={'UV'}; mod.loop_mapping=mapping
    mod.layers_uv_select_src='ALL'; mod.layers_uv_select_dst='NAME'; mod.mix_mode='REPLACE'; mod.mix_factor=1.0
    mod.use_max_distance=True; mod.max_distance=5.0; mod.ray_radius=0.0
    bpy.context.view_layer.update()
    ev=dest.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh()
    result[mapping]=[list(item.uv) for item in me.uv_layers['UVMap'].data]
    ev.to_mesh_clear()
with open(os.path.join(root,'loop-mapping-oracle.json'),'w') as f: json.dump(result,f)
";

#[test]
fn loop_mapping_families_match_blender_with_corner_association() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping data-transfer loop Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "loop-mappings.py",
        BLENDER_LOOP_MAPPING_SCRIPT,
        root,
        "Blender data-transfer loop oracle failed",
    )?;
    let mut expected: Value =
        serde_json::from_slice(&fs::read(root.join("loop-mapping-oracle.json"))?)?;
    flip_blender_uv_to_potter(&mut expected);
    let mut source = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(2.0, 2.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    set_uv_layers(
        &mut source,
        &[(
            "UVMap",
            vec![vec![[0.2, 0.3], [1.21, 0.32], [1.22, 1.14], [0.23, 1.16]]],
        )],
        "UVMap",
    );
    let vertices = vec![
        DVec3::new(0.3, 0.2, 0.1),
        DVec3::new(1.2, 0.2, 0.1),
        DVec3::new(1.5, 1.5, 0.1),
        DVec3::new(0.2, 1.5, 0.1),
    ];
    let mut operands = BTreeMap::new();
    operands.insert(
        "object".to_owned(),
        vec![operand(source, DMat4::IDENTITY, &[], None)],
    );
    for mapping in [
        "NEAREST_NORMAL",
        "NEAREST_POLYNOR",
        "NEAREST_POLY",
        "POLYINTERP_NEAREST",
        "POLYINTERP_LNORPROJ",
    ] {
        let mut destination =
            Mesh::from_positions_and_faces(vertices.clone(), vec![vec![0, 1, 2], vec![0, 2, 3]])?;
        set_uv_layers(
            &mut destination,
            &[("UVMap", vec![vec![[0.0, 0.0]; 3], vec![[0.0, 0.0]; 3]])],
            "UVMap",
        );
        let transfer = modifier(
            "data_transfer",
            &json!({
                "object":"source","use_object_transform":true,"use_loop_data":true,"data_types_loops":["UV"],
                "loop_mapping":mapping,"layers_uv_select_src":"ALL","layers_uv_select_dst":"NAME",
                "mix_mode":"REPLACE","mix_factor":1.0,"use_max_distance":true,"max_distance":5.0
            }),
        )?;
        let actual = evaluate_with_context(
            &destination,
            &transfer,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &AttributeLayerSelection::default(),
        )?;
        let values = uv_layer_values(&actual, "UVMap");
        let error = max_error(&values, &expected[mapping])?;
        assert!(
            error < 2.0e-4,
            "{mapping} loop max error {error}; values={values}"
        );
    }
    Ok(())
}

const BLENDER_DOMAIN_MAPPING_SCRIPT: &str = r"
import bpy, json, os, sys
root=os.path.realpath(sys.argv[sys.argv.index('--')+1])
def make(name,verts,faces):
    mesh=bpy.data.meshes.new(name+'Mesh'); mesh.from_pydata(verts,[],faces); mesh.update()
    ob=bpy.data.objects.new(name,mesh); bpy.context.collection.objects.link(ob); return ob
source=make('EdgeSource',[(0,0,0),(2,0,0),(2,2,0),(0,2,0)],[(0,1,2,3)])
crease=source.data.attributes.get('crease_edge') or source.data.attributes.new(name='crease_edge',type='FLOAT',domain='EDGE')
for edge in source.data.edges: crease.data[edge.index].value=0.1+0.2*edge.index
edge_result={}
dest_vertices=[(0.2,0.0,0.1),(1.4,0.0,0.1),(0.9,1.5,0.1)]
for mapping in ('NEAREST','VERT_NEAREST','POLY_NEAREST','EDGEINTERP_VNORPROJ'):
    dest=make('Edge'+mapping,dest_vertices,[(0,1,2)])
    attr=dest.data.attributes.get('crease_edge') or dest.data.attributes.new(name='crease_edge',type='FLOAT',domain='EDGE')
    for edge in dest.data.edges: attr.data[edge.index].value=0.0
    mod=dest.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source; mod.use_edge_data=True
    mod.data_types_edges={'CREASE'}; mod.edge_mapping=mapping; mod.mix_mode='REPLACE'; mod.mix_factor=1.0
    mod.use_max_distance=True; mod.max_distance=5.0; mod.ray_radius=0.15
    bpy.context.view_layer.update(); ev=dest.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh()
    edge_result[mapping]=[me.attributes['crease_edge'].data[e.index].value for e in me.edges]; ev.to_mesh_clear()
dest=make('EdgeRadiusZero',dest_vertices,[(0,1,2)])
attr=dest.data.attributes.get('crease_edge') or dest.data.attributes.new(name='crease_edge',type='FLOAT',domain='EDGE')
for edge in dest.data.edges: attr.data[edge.index].value=0.0
mod=dest.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source; mod.use_edge_data=True
mod.data_types_edges={'CREASE'}; mod.edge_mapping='EDGEINTERP_VNORPROJ'; mod.mix_mode='REPLACE'; mod.mix_factor=1.0
mod.use_max_distance=True; mod.max_distance=5.0; mod.ray_radius=0.0
bpy.context.view_layer.update(); ev=dest.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh()
edge_result['EDGEINTERP_VNORPROJ_R0']=[me.attributes['crease_edge'].data[e.index].value for e in me.edges]; ev.to_mesh_clear()
dest=make('EdgeTOPOLOGY',[(0,0,0.1),(2,0,0.1),(2,2,0.1),(0,2,0.1)],[(0,1,2,3)])
attr=dest.data.attributes.get('crease_edge') or dest.data.attributes.new(name='crease_edge',type='FLOAT',domain='EDGE')
for edge in dest.data.edges: attr.data[edge.index].value=0.0
mod=dest.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source; mod.use_edge_data=True
mod.data_types_edges={'CREASE'}; mod.edge_mapping='TOPOLOGY'; mod.mix_mode='REPLACE'; mod.mix_factor=1.0
bpy.context.view_layer.update(); ev=dest.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh()
edge_result['TOPOLOGY']=[me.attributes['crease_edge'].data[e.index].value for e in me.edges]; ev.to_mesh_clear()
poly_source=make('FaceSource',[(0,0,0),(1,0,0),(0,1,0),(3,0,0),(4,0,0),(3,1,0)],[(0,1,2),(3,4,5)])
freestyle=poly_source.data.attributes.new(name='freestyle_face',type='BOOLEAN',domain='FACE')
freestyle.data[0].value=False; freestyle.data[1].value=True
face_result={}
for mapping in ('TOPOLOGY','NEAREST','NORMAL','POLYINTERP_PNORPROJ'):
    dest=make('Face'+mapping,[(0,0,0.1),(1,0,0.1),(0,1,0.1),(3,0,0.1),(4,0,0.1),(3,1,0.1)],[(0,1,2),(3,4,5)])
    attr=dest.data.attributes.new(name='freestyle_face',type='BOOLEAN',domain='FACE')
    for item in attr.data: item.value=False
    mod=dest.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=poly_source; mod.use_poly_data=True
    mod.data_types_polys={'FREESTYLE_FACE'}; mod.poly_mapping=mapping; mod.mix_mode='REPLACE'; mod.mix_factor=1.0
    mod.use_max_distance=True; mod.max_distance=5.0; mod.ray_radius=0.01
    bpy.context.view_layer.update(); ev=dest.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh()
    face_result[mapping]=[me.attributes['freestyle_face'].data[i].value for i in range(len(me.polygons))]; ev.to_mesh_clear()
grid_size=8
coords=[-1.0+2.0*i/grid_size for i in range(grid_size+1)]
oblique_source_verts=[(0.0,y,z) for z in coords for y in coords]
oblique_source_faces=[]
for zi in range(grid_size):
    for yi in range(grid_size):
        a=zi*(grid_size+1)+yi; b=a+1; d=(zi+1)*(grid_size+1)+yi; c=d+1
        oblique_source_faces.append((a,b,c,d))
oblique_dest_verts=[(0.1,-0.8,-0.7),(0.1,0.9,-0.6),(0.1,0.65,0.8),(0.1,-0.7,0.9)]
for label,selected in (('PNORPROJ_OBLIQUE_HIT',19),('PNORPROJ_OBLIQUE_MISS',0)):
    source=make('ObliqueSource'+label,oblique_source_verts,oblique_source_faces)
    flags=source.data.attributes.new(name='freestyle_face',type='BOOLEAN',domain='FACE')
    for index,item in enumerate(flags.data): item.value=(index==selected)
    dest=make('ObliqueDest'+label,oblique_dest_verts,[(0,1,2,3)])
    attr=dest.data.attributes.new(name='freestyle_face',type='BOOLEAN',domain='FACE')
    attr.data[0].value=False
    mod=dest.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source; mod.use_poly_data=True
    mod.data_types_polys={'FREESTYLE_FACE'}; mod.poly_mapping='POLYINTERP_PNORPROJ'
    mod.mix_mode='REPLACE'; mod.mix_factor=1.0; mod.use_max_distance=True; mod.max_distance=5.0; mod.ray_radius=0.5
    bpy.context.view_layer.update(); ev=dest.evaluated_get(bpy.context.evaluated_depsgraph_get()); me=ev.to_mesh()
    face_result[label]=[me.attributes['freestyle_face'].data[i].value for i in range(len(me.polygons))]; ev.to_mesh_clear()
with open(os.path.join(root,'domain-mapping-oracle.json'),'w') as f: json.dump({'edge':edge_result,'face':face_result},f)
";

fn set_blender_attribute(
    mesh: &mut Mesh,
    name: &str,
    domain: &str,
    data_type: &str,
    values: &Value,
) -> Result<(), Box<dyn Error>> {
    let attributes = mesh
        .attributes
        .entry("blender_attributes".to_owned())
        .or_insert_with(|| json!([]));
    attributes
        .as_array_mut()
        .ok_or("Blender attributes fixture must be an array")?
        .push(json!({
            "name":name,"domain":domain,"data_type":data_type,"values":values
        }));
    Ok(())
}

fn set_color_layer(
    mesh: &mut Mesh,
    name: &str,
    domain: &str,
    data_type: &str,
    colors: Vec<[f64; 4]>,
) -> Result<(), Box<dyn Error>> {
    let expected_len = match domain {
        "point" => mesh.vertices.len(),
        "corner" => mesh.faces.iter().map(|face| face.vertices.len()).sum(),
        _ => return Err("color test layer has an unsupported domain".into()),
    };
    if colors.len() != expected_len {
        return Err("color layer value count does not match mesh".into());
    }
    let mut values = serde_json::Map::new();
    match domain {
        "point" => {
            for (vertex, color) in mesh.vertices.iter().zip(colors) {
                values.insert(format!("v{}", vertex.id), json!(color));
            }
        }
        "corner" => {
            let mut cursor = 0;
            for face in &mesh.faces {
                let end = cursor + face.vertices.len();
                let row = colors
                    .get(cursor..end)
                    .ok_or("corner color face row is out of range")?;
                values.insert(format!("f{}", face.id), json!(row));
                cursor = end;
            }
        }
        _ => return Err("color test layer has an unsupported domain".into()),
    }
    mesh.attributes.insert(
        name.to_owned(),
        json!({"domain":domain,"type":"color","values":values}),
    );
    let store = mesh
        .attributes
        .entry("color_layers".to_owned())
        .or_insert_with(|| json!({"layers":[],"active":null,"render":null}));
    let layers = store
        .get_mut("layers")
        .and_then(Value::as_array_mut)
        .ok_or("color layer registry must be an array")?;
    if !layers.iter().any(|layer| layer.as_str() == Some(name)) {
        layers.push(json!(name));
    }
    if store.get("active").is_none_or(Value::is_null) {
        store["active"] = json!(name);
    }
    if store.get("render").is_none_or(Value::is_null) {
        store["render"] = json!(name);
    }
    let metadata = mesh
        .attributes
        .entry("blender_metadata".to_owned())
        .or_insert_with(|| json!({}));
    let types = metadata
        .as_object_mut()
        .ok_or("Blender metadata fixture must be an object")?
        .entry("color_data_types".to_owned())
        .or_insert_with(|| json!({}));
    types
        .as_object_mut()
        .ok_or("color data types fixture must be an object")?
        .insert(name.to_owned(), json!(data_type));
    Ok(())
}

#[test]
fn edge_and_polygon_mapping_families_match_blender() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("skipping data-transfer edge/face Blender parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "domain-mappings.py",
        BLENDER_DOMAIN_MAPPING_SCRIPT,
        root,
        "Blender data-transfer edge/face oracle failed",
    )?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("domain-mapping-oracle.json"))?)?;
    let mut source = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(2.0, 2.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    set_blender_attribute(
        &mut source,
        "crease_edge",
        "EDGE",
        "FLOAT",
        &json!([0.1, 0.3, 0.5, 0.7]),
    )?;
    let mut operands = BTreeMap::new();
    operands.insert(
        "object".to_owned(),
        vec![operand(source, DMat4::IDENTITY, &[], None)],
    );
    let destination_vertices = vec![
        DVec3::new(0.2, 0.0, 0.1),
        DVec3::new(1.4, 0.0, 0.1),
        DVec3::new(0.9, 1.5, 0.1),
    ];
    for mapping in [
        "NEAREST",
        "VERT_NEAREST",
        "POLY_NEAREST",
        "EDGEINTERP_VNORPROJ",
    ] {
        let mut destination =
            Mesh::from_positions_and_faces(destination_vertices.clone(), vec![vec![0, 1, 2]])?;
        set_blender_attribute(
            &mut destination,
            "crease_edge",
            "EDGE",
            "FLOAT",
            &json!([0.0, 0.0, 0.0]),
        )?;
        let transfer = modifier(
            "data_transfer",
            &json!({
                "object":"source","use_object_transform":true,"use_edge_data":true,"data_types_edges":["CREASE"],
                "edge_mapping":mapping,"mix_mode":"REPLACE","mix_factor":1.0,
                "use_max_distance":true,"max_distance":5.0,"ray_radius":0.15
            }),
        )?;
        let actual = evaluate_with_context(
            &destination,
            &transfer,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &AttributeLayerSelection::default(),
        )?;
        let values = actual.attributes["blender_attributes"][0]["values"].clone();
        let error = max_error(&values, &expected["edge"][mapping])?;
        assert!(error < 3.0e-4, "edge {mapping} max error {error}");
    }
    let mut radius_zero_destination =
        Mesh::from_positions_and_faces(destination_vertices.clone(), vec![vec![0, 1, 2]])?;
    set_blender_attribute(
        &mut radius_zero_destination,
        "crease_edge",
        "EDGE",
        "FLOAT",
        &json!([0.0, 0.0, 0.0]),
    )?;
    let radius_zero_transfer = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_object_transform":true,"use_edge_data":true,"data_types_edges":["CREASE"],
            "edge_mapping":"EDGEINTERP_VNORPROJ","mix_mode":"REPLACE","mix_factor":1.0,
            "use_max_distance":true,"max_distance":5.0,"ray_radius":0.0
        }),
    )?;
    let radius_zero_actual = evaluate_with_context(
        &radius_zero_destination,
        &radius_zero_transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    let radius_zero_values =
        radius_zero_actual.attributes["blender_attributes"][0]["values"].clone();
    let radius_zero_error = max_error(
        &radius_zero_values,
        &expected["edge"]["EDGEINTERP_VNORPROJ_R0"],
    )?;
    assert!(
        radius_zero_error < 3.0e-4,
        "EDGEINTERP_VNORPROJ radius=0 max error {radius_zero_error}"
    );
    let mut topology_destination = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(0.0, 0.0, 0.1),
            DVec3::new(2.0, 0.0, 0.1),
            DVec3::new(2.0, 2.0, 0.1),
            DVec3::new(0.0, 2.0, 0.1),
        ],
        vec![vec![0, 1, 2, 3]],
    )?;
    set_blender_attribute(
        &mut topology_destination,
        "crease_edge",
        "EDGE",
        "FLOAT",
        &json!([0.0, 0.0, 0.0, 0.0]),
    )?;
    let transfer = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_object_transform":true,"use_edge_data":true,"data_types_edges":["CREASE"],
            "edge_mapping":"TOPOLOGY","mix_mode":"REPLACE","mix_factor":1.0
        }),
    )?;
    let actual = evaluate_with_context(
        &topology_destination,
        &transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    let values = actual.attributes["blender_attributes"][0]["values"].clone();
    let error = max_error(&values, &expected["edge"]["TOPOLOGY"])?;
    assert!(error < 1.0e-6, "edge TOPOLOGY max error {error}");
    let mut face_source = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(3.0, 0.0, 0.0),
            DVec3::new(4.0, 0.0, 0.0),
            DVec3::new(3.0, 1.0, 0.0),
        ],
        vec![vec![0, 1, 2], vec![3, 4, 5]],
    )?;
    set_blender_attribute(
        &mut face_source,
        "freestyle_face",
        "FACE",
        "BOOLEAN",
        &json!([false, true]),
    )?;
    let mut face_operands = BTreeMap::new();
    face_operands.insert(
        "object".to_owned(),
        vec![operand(face_source, DMat4::IDENTITY, &[], None)],
    );
    for mapping in ["TOPOLOGY", "NEAREST", "NORMAL", "POLYINTERP_PNORPROJ"] {
        let mut destination = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.1),
                DVec3::new(1.0, 0.0, 0.1),
                DVec3::new(0.0, 1.0, 0.1),
                DVec3::new(3.0, 0.0, 0.1),
                DVec3::new(4.0, 0.0, 0.1),
                DVec3::new(3.0, 1.0, 0.1),
            ],
            vec![vec![0, 1, 2], vec![3, 4, 5]],
        )?;
        set_blender_attribute(
            &mut destination,
            "freestyle_face",
            "FACE",
            "BOOLEAN",
            &json!([false, false]),
        )?;
        let transfer = modifier(
            "data_transfer",
            &json!({
                "object":"source","use_object_transform":true,"use_poly_data":true,"data_types_polys":["FREESTYLE_FACE"],
                "poly_mapping":mapping,"mix_mode":"REPLACE","mix_factor":1.0,
                "use_max_distance":true,"max_distance":5.0,"ray_radius":0.01
            }),
        )?;
        let actual = evaluate_with_context(
            &destination,
            &transfer,
            &face_operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &AttributeLayerSelection::default(),
        )?;
        let values = actual.attributes["blender_attributes"][0]["values"].clone();
        assert_eq!(values, expected["face"][mapping], "face {mapping} flags");
    }
    let grid_size = 8usize;
    let coordinates = (0..=grid_size)
        .map(|index| -1.0 + 2.0 * index as f64 / grid_size as f64)
        .collect::<Vec<_>>();
    let oblique_positions = coordinates
        .iter()
        .flat_map(|z| coordinates.iter().map(move |y| DVec3::new(0.0, *y, *z)))
        .collect::<Vec<_>>();
    let mut oblique_faces = Vec::new();
    for z in 0..grid_size {
        for y in 0..grid_size {
            let a = z * (grid_size + 1) + y;
            let b = a + 1;
            let d = (z + 1) * (grid_size + 1) + y;
            let c = d + 1;
            oblique_faces.push(vec![a, b, c, d]);
        }
    }
    let oblique_destination_positions = vec![
        DVec3::new(0.1, -0.8, -0.7),
        DVec3::new(0.1, 0.9, -0.6),
        DVec3::new(0.1, 0.65, 0.8),
        DVec3::new(0.1, -0.7, 0.9),
    ];
    for (label, selected) in [
        ("PNORPROJ_OBLIQUE_HIT", 19usize),
        ("PNORPROJ_OBLIQUE_MISS", 0),
    ] {
        let mut oblique_source =
            Mesh::from_positions_and_faces(oblique_positions.clone(), oblique_faces.clone())?;
        let face_flags = (0..grid_size * grid_size)
            .map(|index| index == selected)
            .collect::<Vec<_>>();
        set_blender_attribute(
            &mut oblique_source,
            "freestyle_face",
            "FACE",
            "BOOLEAN",
            &json!(face_flags),
        )?;
        let mut oblique_operands = BTreeMap::new();
        oblique_operands.insert(
            "object".to_owned(),
            vec![operand(oblique_source, DMat4::IDENTITY, &[], None)],
        );
        let mut oblique_destination = Mesh::from_positions_and_faces(
            oblique_destination_positions.clone(),
            vec![vec![0, 1, 2, 3]],
        )?;
        set_blender_attribute(
            &mut oblique_destination,
            "freestyle_face",
            "FACE",
            "BOOLEAN",
            &json!([false]),
        )?;
        let oblique_transfer = modifier(
            "data_transfer",
            &json!({
                "object":"source","use_object_transform":true,"use_poly_data":true,
                "data_types_polys":["FREESTYLE_FACE"],"poly_mapping":"POLYINTERP_PNORPROJ",
                "mix_mode":"REPLACE","mix_factor":1.0,"use_max_distance":true,
                "max_distance":5.0,"ray_radius":0.5
            }),
        )?;
        let oblique_actual = evaluate_with_context(
            &oblique_destination,
            &oblique_transfer,
            &oblique_operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &AttributeLayerSelection::default(),
        )?;
        let values = oblique_actual.attributes["blender_attributes"][0]["values"].clone();
        assert_eq!(values, expected["face"][label], "oblique {label} flags");
    }
    Ok(())
}

const BLENDER_ATTRIBUTE_DOMAINS_SCRIPT: &str = r"
import bpy, json, os, sys
root=os.path.realpath(sys.argv[sys.argv.index('--')+1])
verts=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
faces=[(0,1,2,3)]
def mesh(name, source):
    data=bpy.data.meshes.new(name+'Mesh'); data.from_pydata(verts,[],faces); data.update()
    obj=bpy.data.objects.new(name,data); bpy.context.collection.objects.link(obj)
    specifications=(
        ('bevel_weight_vert','FLOAT','POINT',[0.1,0.3,0.5,0.7] if source else [0.8,0.6,0.4,0.2]),
        ('crease_edge','FLOAT','EDGE',[0.1,0.3,0.5,0.7] if source else [0.0]*4),
        ('bevel_weight_edge','FLOAT','EDGE',[0.2,0.4,0.6,0.8] if source else [0.0]*4),
        ('freestyle_edge','BOOLEAN','EDGE',[False,True,False,True] if source else [False]*4),
        ('freestyle_face','BOOLEAN','FACE',[True] if source else [False]),
    )
    for name,type,domain,values in specifications:
        attribute=data.attributes.get(name) or data.attributes.new(name=name,type=type,domain=domain)
        for item,value in zip(attribute.data,values): item.value=value
    data.polygons[0].use_smooth=source
    if source:
        for edge in data.edges:
            if set(edge.vertices)=={0,1}: edge.use_edge_sharp=True
            if set(edge.vertices)=={1,2}: edge.use_seam=True
    return obj
source=mesh('Source',True)
result={}
for data_type,domain in (
    ('BEVEL_WEIGHT_VERT','VERT'),('SHARP_EDGE','EDGE'),('SEAM','EDGE'),('CREASE','EDGE'),
    ('BEVEL_WEIGHT_EDGE','EDGE'),('FREESTYLE_EDGE','EDGE'),('SMOOTH','POLY'),('FREESTYLE_FACE','POLY')
):
    destination=mesh('Dest'+data_type,False)
    mod=destination.modifiers.new('Transfer','DATA_TRANSFER'); mod.object=source; mod.use_object_transform=True
    if domain=='VERT':
        mod.use_vert_data=True; mod.data_types_verts={data_type}; mod.vert_mapping='TOPOLOGY'
    elif domain=='EDGE':
        mod.use_edge_data=True; mod.data_types_edges={data_type}; mod.edge_mapping='TOPOLOGY'
    else:
        mod.use_poly_data=True; mod.data_types_polys={data_type}; mod.poly_mapping='TOPOLOGY'
    mod.mix_mode='REPLACE'; mod.mix_factor=1.0
    bpy.context.view_layer.update()
    evaluated=destination.evaluated_get(bpy.context.evaluated_depsgraph_get()); output=evaluated.to_mesh()
    if data_type=='SHARP_EDGE':
        values=[bool(edge.use_edge_sharp) for edge in output.edges]
    elif data_type=='SEAM':
        values=[bool(edge.use_seam) for edge in output.edges]
    elif data_type=='SMOOTH':
        values=[bool(poly.use_smooth) for poly in output.polygons]
    else:
        name={'BEVEL_WEIGHT_VERT':'bevel_weight_vert','CREASE':'crease_edge',
              'BEVEL_WEIGHT_EDGE':'bevel_weight_edge','FREESTYLE_EDGE':'freestyle_edge',
              'FREESTYLE_FACE':'freestyle_face'}[data_type]
        values=[item.value for item in output.attributes[name].data]
    result[data_type]=values
    evaluated.to_mesh_clear()
with open(os.path.join(root,'attribute-domains-oracle.json'),'w') as f: json.dump(result,f)
";

#[test]
fn every_data_transfer_attribute_domain_matches_blender() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!(
            "skipping data-transfer attribute-domain Blender parity test: Blender is unavailable"
        );
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "attribute-domains.py",
        BLENDER_ATTRIBUTE_DOMAINS_SCRIPT,
        root,
        "Blender data-transfer attribute-domain oracle failed",
    )?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("attribute-domains-oracle.json"))?)?;
    let positions = vec![DVec3::ZERO, DVec3::X, DVec3::ONE, DVec3::Y];
    let make_mesh = || Mesh::from_positions_and_faces(positions.clone(), vec![vec![0, 1, 2, 3]]);
    let mut source = make_mesh()?;
    set_domain_attribute(
        &mut source,
        "bevel_weight_vert",
        "point",
        "float",
        &json!([0.1, 0.3, 0.5, 0.7]),
    )?;
    set_domain_attribute(
        &mut source,
        "crease_edge",
        "edge",
        "float",
        &json!([0.1, 0.3, 0.5, 0.7]),
    )?;
    set_domain_attribute(
        &mut source,
        "bevel_weight_edge",
        "edge",
        "float",
        &json!([0.2, 0.4, 0.6, 0.8]),
    )?;
    set_domain_attribute(
        &mut source,
        "freestyle_edge",
        "edge",
        "bool",
        &json!([false, true, false, true]),
    )?;
    set_domain_attribute(
        &mut source,
        "freestyle_face",
        "face",
        "bool",
        &json!([true]),
    )?;
    set_edge_flags(&mut source, "sharp", Some((0, 1)));
    set_edge_flags(&mut source, "seam", Some((1, 2)));
    set_face_flags(&mut source, "shade_smooth", &[true]);
    let operands = BTreeMap::from([(
        "object".to_owned(),
        vec![operand(source, DMat4::IDENTITY, &[], None)],
    )]);
    for data_type in [
        "BEVEL_WEIGHT_VERT",
        "SHARP_EDGE",
        "SEAM",
        "CREASE",
        "BEVEL_WEIGHT_EDGE",
        "FREESTYLE_EDGE",
        "SMOOTH",
        "FREESTYLE_FACE",
    ] {
        let mut destination = make_mesh()?;
        set_domain_attribute(
            &mut destination,
            "bevel_weight_vert",
            "point",
            "float",
            &json!([0.8, 0.6, 0.4, 0.2]),
        )?;
        set_domain_attribute(
            &mut destination,
            "crease_edge",
            "edge",
            "float",
            &json!([0.0, 0.0, 0.0, 0.0]),
        )?;
        set_domain_attribute(
            &mut destination,
            "bevel_weight_edge",
            "edge",
            "float",
            &json!([0.0, 0.0, 0.0, 0.0]),
        )?;
        set_domain_attribute(
            &mut destination,
            "freestyle_edge",
            "edge",
            "bool",
            &json!([false, false, false, false]),
        )?;
        set_domain_attribute(
            &mut destination,
            "freestyle_face",
            "face",
            "bool",
            &json!([false]),
        )?;
        set_edge_flags(&mut destination, "sharp", None);
        set_edge_flags(&mut destination, "seam", None);
        set_face_flags(&mut destination, "shade_smooth", &[false]);
        let fields = match data_type {
            "BEVEL_WEIGHT_VERT" => {
                json!({"use_vert_data":true,"data_types_verts":[data_type],"vert_mapping":"TOPOLOGY"})
            }
            "SHARP_EDGE" | "SEAM" | "CREASE" | "BEVEL_WEIGHT_EDGE" | "FREESTYLE_EDGE" => {
                json!({"use_edge_data":true,"data_types_edges":[data_type],"edge_mapping":"TOPOLOGY"})
            }
            _ => {
                json!({"use_poly_data":true,"data_types_polys":[data_type],"poly_mapping":"TOPOLOGY"})
            }
        };
        let mut params = fields
            .as_object()
            .ok_or("domain fields must be an object")?
            .clone();
        params.insert("object".to_owned(), json!("source"));
        params.insert("mix_mode".to_owned(), json!("REPLACE"));
        params.insert("mix_factor".to_owned(), json!(1.0));
        let transfer = modifier("data_transfer", &Value::Object(params))?;
        let actual = evaluate_with_context(
            &destination,
            &transfer,
            &operands,
            None,
            None,
            DMat4::IDENTITY,
            "destination",
            &AttributeLayerSelection::default(),
        )?;
        let actual_values = match data_type {
            "SHARP_EDGE" => domain_attribute_values(&actual, "sharp"),
            "SEAM" => domain_attribute_values(&actual, "seam"),
            "SMOOTH" => domain_attribute_values(&actual, "shade_smooth"),
            "BEVEL_WEIGHT_VERT" => domain_attribute_values(&actual, "bevel_weight_vert"),
            "CREASE" => domain_attribute_values(&actual, "crease_edge"),
            "BEVEL_WEIGHT_EDGE" => domain_attribute_values(&actual, "bevel_weight_edge"),
            "FREESTYLE_EDGE" => domain_attribute_values(&actual, "freestyle_edge"),
            _ => domain_attribute_values(&actual, "freestyle_face"),
        };
        if matches!(
            data_type,
            "BEVEL_WEIGHT_VERT" | "CREASE" | "BEVEL_WEIGHT_EDGE"
        ) {
            let error = max_error(&actual_values, &expected[data_type])?;
            assert!(error < 1.0e-6, "{data_type} max error {error}");
        } else {
            assert_eq!(actual_values, expected[data_type], "{data_type}");
        }
    }
    Ok(())
}
const BLENDER_SHARP_NORMAL_TRANSFER_SCRIPT: &str = r"
import bpy, json
positions=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
faces=[(0,1,2),(1,0,3)]
def make_object(name):
    mesh=bpy.data.meshes.new(name+'Mesh')
    mesh.from_pydata(positions,[],faces); mesh.update()
    for polygon in mesh.polygons: polygon.use_smooth=True
    obj=bpy.data.objects.new(name,mesh); bpy.context.collection.objects.link(obj)
    return obj
source=make_object('Source'); destination=make_object('Destination')
source_sharp=source.data.attributes.get('sharp_edge')
if source_sharp is None: source_sharp=source.data.attributes.new('sharp_edge','BOOLEAN','EDGE')
shared=next(edge.index for edge in source.data.edges if set(edge.vertices)=={0,1})
source_sharp.data[shared].value=True
modifier=destination.modifiers.new('Transfer sharp','DATA_TRANSFER')
modifier.object=source; modifier.use_edge_data=True; modifier.data_types_edges={'SHARP_EDGE'}
modifier.edge_mapping='TOPOLOGY'; modifier.mix_mode='REPLACE'; modifier.mix_factor=1.0
bpy.context.view_layer.update()
evaluated=destination.evaluated_get(bpy.context.evaluated_depsgraph_get())
mesh=evaluated.to_mesh()
json.dump({
    'corner_normals':[list(corner.vector) for corner in mesh.corner_normals],
},open('sharp-normal-oracle.json','w'))
evaluated.to_mesh_clear()
";

#[test]
fn transferred_sharp_edge_separates_derived_corner_normals() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!(
            "skipping transferred-sharp corner-normal Blender parity test: Blender is unavailable"
        );
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    blender_script::run_blender_script(
        &blender,
        "sharp-normal-transfer.py",
        BLENDER_SHARP_NORMAL_TRANSFER_SCRIPT,
        root,
        "Blender sharp-edge corner-normal oracle failed",
    )?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("sharp-normal-oracle.json"))?)?;
    let positions = vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::Z];
    let faces = vec![vec![0, 1, 2], vec![1, 0, 3]];
    let mut source = Mesh::from_positions_and_faces(positions.clone(), faces.clone())?;
    set_edge_flags(&mut source, "sharp", Some((0, 1)));
    set_face_flags(&mut source, "shade_smooth", &[true, true]);
    let operands = BTreeMap::from([(
        "source".to_owned(),
        vec![operand(source, DMat4::IDENTITY, &[], None)],
    )]);
    let mut destination = Mesh::from_positions_and_faces(positions.clone(), faces)?;
    set_edge_flags(&mut destination, "sharp", None);
    set_face_flags(&mut destination, "shade_smooth", &[true, true]);
    let transfer = modifier(
        "data_transfer",
        &json!({
            "object":"source","use_edge_data":true,"data_types_edges":["SHARP_EDGE"],
            "edge_mapping":"TOPOLOGY","mix_mode":"REPLACE","mix_factor":1.0,
        }),
    )?;
    let actual = evaluate_with_context(
        &destination,
        &transfer,
        &operands,
        None,
        None,
        DMat4::IDENTITY,
        "destination",
        &AttributeLayerSelection::default(),
    )?;
    let actual_normals = potter_core::geom::corner_normal_vectors(&actual)?;
    let actual_values = json!(
        actual_normals
            .iter()
            .map(|normal| [normal.x, normal.y, normal.z])
            .collect::<Vec<_>>()
    );
    let error = max_error(&actual_values, &expected["corner_normals"])?;
    assert!(
        error < 2.0e-4,
        "sharp-transfer derived corner-normal max error {error}"
    );
    let shared_edge = actual
        .edges
        .iter()
        .find(|edge| {
            edge.vertices.contains(&actual.vertices[0].id)
                && edge.vertices.contains(&actual.vertices[1].id)
        })
        .ok_or("native output has no shared edge")?;
    let transferred_sharp = actual
        .attributes
        .get("sharp")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
        .and_then(|values| values.get(&format!("e{}", shared_edge.id)))
        .and_then(Value::as_bool);
    assert_eq!(
        transferred_sharp,
        Some(true),
        "shared edge should receive transferred sharp state"
    );
    let normals = expected["corner_normals"]
        .as_array()
        .ok_or("oracle corner normals are missing")?;
    let normal_at = |index: usize| -> Result<DVec3, Box<dyn Error>> {
        let row = normals[index]
            .as_array()
            .ok_or("oracle corner normal is not an array")?;
        Ok(DVec3::new(
            row[0].as_f64().ok_or("normal x missing")?,
            row[1].as_f64().ok_or("normal y missing")?,
            row[2].as_f64().ok_or("normal z missing")?,
        ))
    };
    assert!(
        normal_at(0)?.dot(normal_at(4)?) < 0.99,
        "the transferred sharp edge must split normals at shared vertex 0"
    );
    assert!(
        normal_at(1)?.dot(normal_at(3)?) < 0.99,
        "the transferred sharp edge must split normals at shared vertex 1"
    );
    Ok(())
}
