use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::{BoxParams, GridParams, Mesh, UvSphereParams, modifiers::evaluate_modifiers},
    model::{Id, Modifier, SceneDoc},
};
use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;

fn init(scene: &Path) -> Result<(), Box<dyn Error>> {
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

fn apply(scene: &Path, revision: u64, operations: &Value) -> Result<Output, Box<dyn Error>> {
    let batch = scene.with_extension("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": revision,
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

fn load_doc(scene: &Path) -> Result<SceneDoc, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(
        scene.join("scene.json"),
    )?)?)
}

fn modifier(modifier_type: &str, params: Value) -> Result<Modifier, Box<dyn Error>> {
    Ok(Modifier {
        id: Id::new("deform_test")?,
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        params: serde_json::from_value(params)?,
        binding_data: None,
        runtime: potter_core::model::ModifierRuntime::default(),
    })
}

fn remesh_fixture() -> Result<Mesh, Box<dyn Error>> {
    let mut primary = Mesh::uv_sphere(UvSphereParams {
        segments: 16,
        ring_count: 12,
        radius: 1.5,
    })?;
    for vertex in &mut primary.vertices {
        let position = vertex.co;
        if position.x > 1.0 && position.y.abs() < 0.65 && position.z.abs() < 0.65 {
            vertex.co *= 0.4;
        }
    }
    let mut secondary = Mesh::uv_sphere(UvSphereParams {
        segments: 12,
        ring_count: 8,
        radius: 0.4,
    })?;
    for vertex in &mut secondary.vertices {
        vertex.co += DVec3::new(2.6, 0.1, 0.15);
    }
    let mut positions = Vec::with_capacity(primary.vertices.len() + secondary.vertices.len());
    let mut polygons = Vec::with_capacity(primary.faces.len() + secondary.faces.len());
    for mesh in [&primary, &secondary] {
        let offset = positions.len();
        let indices = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<std::collections::HashMap<_, _>>();
        positions.extend(mesh.vertices.iter().map(|vertex| vertex.co));
        for face in &mesh.faces {
            polygons.push(
                face.vertices
                    .iter()
                    .map(|vertex| offset + indices[vertex])
                    .collect(),
            );
        }
    }
    Ok(Mesh::from_positions_and_faces(positions, polygons)?)
}

fn round_position_to_blender_precision(position: DVec3) -> DVec3 {
    DVec3::new(
        f64::from(position.x as f32),
        f64::from(position.y as f32),
        f64::from(position.z as f32),
    )
}

fn round_mesh_positions_to_blender_precision(mesh: &mut Mesh) {
    for vertex in &mut mesh.vertices {
        vertex.co = round_position_to_blender_precision(vertex.co);
    }
}

fn mesh_fixture_json(mesh: &Mesh) -> Value {
    let indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<std::collections::HashMap<_, _>>();
    let positions = mesh
        .vertices
        .iter()
        .map(|vertex| [vertex.co.x, vertex.co.y, vertex.co.z])
        .collect::<Vec<_>>();
    let polygons = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|vertex| indices[vertex])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    json!({"positions":positions,"polygons":polygons})
}

#[test]
fn apply_as_shape_key_captures_topology_preserving_output_and_removes_modifier()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let output = apply(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}},
            {"op":"modifier.create","target":{"id":"subject"},"id":"wave","type":"wave","params":{"height":0.8,"width":2.0,"speed":0.0,"time_offset":0.0}},
            {"op":"modifier.apply_as_shape_key","target":{"id":"subject"},"modifier_id":"wave","id":"baked_wave","name":"Baked Wave"}
        ]),
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let doc = load_doc(&scene)?;
    let data_id = doc.nodes[&Id::new("subject")?]
        .data
        .as_ref()
        .ok_or("mesh data missing")?;
    let data = &doc.data_blocks[data_id];
    let shape_keys = data.shape_keys.as_ref().ok_or("shape key data missing")?;
    let key = &shape_keys.keys[&Id::new("baked_wave")?];
    assert_eq!(key.name, "Baked Wave");
    assert!(
        doc.nodes[&Id::new("subject")?].modifiers.is_empty(),
        "{:?}",
        doc.nodes[&Id::new("subject")?].modifiers
    );
    assert!(key.positions.iter().any(|(id, position)| {
        shape_keys.basis.get(id).is_some_and(|basis| {
            DVec3::from_array(*position).distance(DVec3::from_array(*basis)) > 1.0e-6
        })
    }));
    Ok(())
}

#[test]
fn apply_as_shape_key_can_keep_the_original_modifier_enabled() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let output = apply(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}},
            {"op":"modifier.create","target":{"id":"subject"},"id":"wave","type":"wave","params":{"height":0.8,"speed":0.0}},
            {"op":"modifier.apply_as_shape_key","target":{"id":"subject"},"modifier_id":"wave","id":"kept_key","keep_modifier":true}
        ]),
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let doc = load_doc(&scene)?;
    let node = &doc.nodes[&Id::new("subject")?];
    assert_eq!(node.modifiers.len(), 1);
    assert_eq!(node.modifiers[0].id, Id::new("wave")?);
    let data_id = node.data.as_ref().ok_or("mesh data missing")?;
    let key_id = Id::new("kept_key")?;
    assert!(
        doc.data_blocks[data_id]
            .shape_keys
            .as_ref()
            .is_some_and(|keys| keys.keys.contains_key(&key_id))
    );
    Ok(())
}

#[test]
fn apply_as_shape_key_reports_topology_changes() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let output = apply(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"subject","kind":"box","params":{"size":2.0}},
            {"op":"modifier.create","target":{"id":"subject"},"id":"subdivide","type":"subdivision","params":{"levels":1}},
            {"op":"modifier.apply_as_shape_key","target":{"id":"subject"},"modifier_id":"subdivide","id":"invalid_key"}
        ]),
    )?;
    assert!(!output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        response["error"]["details"]["feature_id"],
        "modifier.apply_as_shape_key.topology"
    );
    assert!(
        response["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("changes the mesh topology"))
    );
    Ok(())
}

#[test]
fn apply_modifier_matches_blender_shape_key_guard_for_any_modifier_type()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let output = apply(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}},
            {"op":"shape_key.create","target":{"id":"subject"},"id":"smile","name":"Smile","positions":{"0":[1.0,-1.0,0.25]}},
            {"op":"modifier.create","target":{"id":"subject"},"id":"wave","type":"wave","params":{"height":0.5}},
            {"op":"modifier.apply","target":{"id":"subject"},"modifier_id":"wave"}
        ]),
    )?;
    assert!(!output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(
        response["error"]["details"]["feature_id"],
        "modifier.apply.shape_key_guard"
    );
    assert_eq!(
        response["error"]["message"],
        "Modifier cannot be applied to a mesh with shape keys"
    );
    Ok(())
}

#[test]
fn mesh_deform_harmonic_bind_tracks_the_closed_cage() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let output = apply(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}},
            {"op":"node.create","id":"cage","kind":"box","params":{"size":4.0}},
            {"op":"modifier.create","target":{"id":"subject"},"id":"mesh_deform","type":"mesh_deform","params":{"object":"cage","precision":3}},
            {"op":"modifier.bind","target":{"id":"subject"},"modifier_id":"mesh_deform"},
            {"op":"mesh.transform_elements","target":{"id":"cage","elements":{"domain":"vertex","ids":["v0","v1","v2","v3","v4","v5","v6","v7"]}},"translation":[0.0,0.0,1.0]}
        ]),
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let doc = load_doc(&scene)?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    let subject = snapshot
        .meshes
        .get(&Id::new("subject")?)
        .ok_or("evaluated subject mesh missing")?;
    assert!(
        subject
            .vertices
            .iter()
            .all(|vertex| (vertex.co.z - 1.0).abs() < 1.0e-5)
    );
    Ok(())
}

#[test]
fn surface_deform_bind_tracks_target_mesh_changes_and_unbind_restores_source()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let output = apply(
        &scene,
        0,
        &json!([
            {"op":"node.create","id":"subject","kind":"plane","params":{"size":2.0}},
            {"op":"node.create","id":"target","kind":"plane","params":{"size":2.0}},
            {"op":"modifier.create","target":{"id":"target"},"id":"target_subdivision","type":"subdivision","params":{"levels":1}},
            {"op":"modifier.create","target":{"id":"subject"},"id":"surface","type":"surface_deform","params":{"target":"target","strength":1.0}},
            {"op":"modifier.bind","target":{"id":"subject"},"modifier_id":"surface"},
            {"op":"mesh.transform_elements","target":{"id":"target","elements":{"domain":"vertex","ids":["v0"]}},"translation":[0.0,0.0,1.0]}
        ]),
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let doc = load_doc(&scene)?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    let subject = snapshot
        .meshes
        .get(&Id::new("subject")?)
        .ok_or("evaluated subject mesh missing")?;
    assert!(
        subject
            .vertices
            .iter()
            .any(|vertex| vertex.co.z.abs() > 1.0e-3),
        "a Surface Deform bind must track the target's evaluated subdivision output"
    );

    let output = apply(
        &scene,
        1,
        &json!([{"op":"modifier.unbind","target":{"id":"subject"},"modifier_id":"surface"}]),
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let doc = load_doc(&scene)?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    let subject = snapshot
        .meshes
        .get(&Id::new("subject")?)
        .ok_or("evaluated subject mesh missing")?;
    assert!(
        subject
            .vertices
            .iter()
            .all(|vertex| vertex.co.z.abs() < 1.0e-6)
    );
    Ok(())
}

#[test]
fn remesh_octree_modes_return_valid_closed_meshes_and_support_smooth_shading()
-> Result<(), Box<dyn Error>> {
    let source = Mesh::box_mesh(BoxParams::default())?;
    for mode in ["BLOCKS"] {
        let smooth_shade = true;
        let params = json!({
            "mode":mode,"octree_depth":3,"scale":0.9,"sharpness":1.0,
            "use_remove_disconnected":false,"threshold":1.0,
            "use_smooth_shade":smooth_shade
        });
        let output = evaluate_modifiers(&source, &[modifier("remesh", params)?])?;
        assert!(output.validate().is_ok(), "{mode} output is invalid");
        assert!(!output.faces.is_empty(), "{mode} output is empty");
        for face in &output.faces {
            let smooth = output
                .attributes
                .get("shade_smooth")
                .and_then(|attribute| attribute["values"].as_object())
                .and_then(|values| values.get(&format!("f{}", face.id)))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            assert_eq!(
                smooth, smooth_shade,
                "{mode}: face {} smooth-shading flag",
                face.id
            );
        }
    }
    Ok(())
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
fn match_vertex_positions(
    actual: &[DVec3],
    expected: &[DVec3],
    context: &str,
) -> (Vec<usize>, f64) {
    // Blender's Remesh SHARP mode solves the dual-contouring QEF in float32
    // (`intern/dualcon/intern/octree.cpp` `minimize`), and its rounding differs between Blender
    // builds: SHARP vertices reach ~8e-6 on macOS arm64 and ~1.3e-5 on Linux x86-64, while
    // BLOCKS, SMOOTH, and VOXEL stay below 1e-6.
    const TOLERANCE: f64 = 2.5e-5;
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
        assert!(
            assign(
                actual_index,
                &candidates,
                &mut expected_to_actual,
                &mut visited
            ),
            "{context}: no bijective vertex correspondence within {TOLERANCE}; vertex {actual_index} has {} candidates, nearest expected error={}",
            candidates[actual_index].len(),
            expected
                .iter()
                .map(|expected_position| actual[actual_index].distance(*expected_position))
                .fold(f64::INFINITY, f64::min)
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

#[test]
fn blender_remesh_supported_modes_match_exact_geometry_and_voxel_adaptivity_is_gated()
-> Result<(), Box<dyn Error>> {
    let mut source = remesh_fixture()?;
    round_mesh_positions_to_blender_precision(&mut source);
    let voxel_adaptivity_params = json!({
        "mode":"VOXEL","voxel_size":0.3,"adaptivity":0.1,
        "use_smooth_shade":false,"use_remove_disconnected":false,"threshold":0.15
    });
    let voxel_adaptivity_error =
        evaluate_modifiers(&source, &[modifier("remesh", voxel_adaptivity_params)?])
            .err()
            .ok_or("nonzero VOXEL adaptivity unexpectedly succeeded")?;
    assert_eq!(
        voxel_adaptivity_error.code,
        potter_core::error::ErrorCode::UnsupportedFeature
    );
    assert_eq!(
        voxel_adaptivity_error
            .details
            .get("feature_id")
            .and_then(Value::as_str),
        Some("modifier.remesh.voxel_adaptivity")
    );
    let catalog = potter_core::catalog::feature_catalog();
    let features = catalog["features"]
        .as_array()
        .ok_or("feature catalog rows are missing")?;
    let voxel_feature = features
        .iter()
        .find(|feature| feature["feature_id"] == "modifier.remesh")
        .ok_or("Remesh is missing from the feature catalog")?;
    assert_eq!(voxel_feature["status"], "supported");
    let voxel_adaptivity_feature = features
        .iter()
        .find(|feature| feature["feature_id"] == "modifier.remesh.voxel_adaptivity")
        .ok_or("VOXEL adaptivity is missing from the feature catalog")?;
    assert_eq!(
        voxel_adaptivity_feature["status"], "not_supported",
        "nonzero VOXEL adaptivity catalog status"
    );
    for mode in ["SMOOTH", "SHARP"] {
        assert!(
            !features.iter().any(|feature| feature["feature_id"]
                == format!("modifier.remesh.{}", mode.to_lowercase())),
            "{mode}: supported mode must not have an unsupported catalog row"
        );
    }
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender Remesh parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let script = r#"
import bpy, json, sys
fixture = __POTTER_INPUT__
source_mesh = bpy.data.meshes.new("InputMesh")
source_mesh.from_pydata(fixture["positions"], [], fixture["polygons"])
source_mesh.update()
source = bpy.data.objects.new("Input", source_mesh)
bpy.context.scene.collection.objects.link(source)
results = {}
for mode in ('VOXEL', 'BLOCKS', 'SMOOTH', 'SHARP'):
    obj = source.copy()
    obj.data = source.data.copy()
    bpy.context.scene.collection.objects.link(obj)
    modifier = obj.modifiers.new('Remesh', 'REMESH')
    modifier.mode = mode
    modifier.use_remove_disconnected = mode != 'VOXEL'
    modifier.use_smooth_shade = (mode == 'BLOCKS')
    modifier.threshold = 0.15
    if mode == 'VOXEL':
        modifier.voxel_size = 0.3
        modifier.adaptivity = 0.0
    else:
        modifier.octree_depth = 4
        modifier.scale = 0.9
        modifier.sharpness = 1.0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    results[mode] = {
        'positions':[list(vertex.co) for vertex in mesh.vertices],
        'polygons':[list(polygon.vertices) for polygon in mesh.polygons],
        'smooth':[polygon.use_smooth for polygon in mesh.polygons]
    }
    evaluated.to_mesh_clear()
    bpy.data.objects.remove(obj, do_unlink=True)
with open(sys.argv[-1], 'w', encoding='utf-8') as output:
    json.dump(results, output)
"#
    .replace("__POTTER_INPUT__", &mesh_fixture_json(&source).to_string());
    let references = blender_script_json(&blender, directory.path(), &script)?;

    for mode in ["VOXEL", "BLOCKS", "SMOOTH", "SHARP"] {
        let smooth_shade = mode == "BLOCKS";
        let params = if mode == "VOXEL" {
            json!({
                "mode":mode,"voxel_size":0.3,"adaptivity":0.0,
                "use_smooth_shade":false,"use_remove_disconnected":false,"threshold":0.15
            })
        } else {
            json!({
                "mode":mode,"octree_depth":4,"scale":0.9,"sharpness":1.0,
                "use_smooth_shade":smooth_shade,
                "use_remove_disconnected":true,"threshold":0.15
            })
        };
        let native = evaluate_modifiers(&source, &[modifier("remesh", params)?])?;
        let reference = &references[mode];
        let blender_positions = points(&reference["positions"])?;
        let blender_polygons = reference["polygons"]
            .as_array()
            .ok_or("Blender remesh polygons missing")?
            .iter()
            .map(|polygon| {
                polygon
                    .as_array()
                    .and_then(|indices| {
                        indices
                            .iter()
                            .map(|index| {
                                index.as_u64().and_then(|index| usize::try_from(index).ok())
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .ok_or("Blender remesh polygon indices were malformed")
            })
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(
            native.vertices.len(),
            blender_positions.len(),
            "{mode}: vertex count"
        );
        assert_eq!(
            native.faces.len(),
            blender_polygons.len(),
            "{mode}: face count"
        );
        let native_positions = native
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let (native_to_blender, maximum_vertex_error) =
            match_vertex_positions(&native_positions, &blender_positions, mode);

        let native_ordinals = native
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<std::collections::HashMap<_, _>>();
        let native_polygons = native
            .faces
            .iter()
            .map(|face| {
                face.vertices
                    .iter()
                    .map(|vertex| {
                        native_ordinals
                            .get(vertex)
                            .map(|ordinal| native_to_blender[*ordinal])
                            .ok_or("native remesh face references an unknown vertex")
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            canonical_polygon_cycles(&native_polygons),
            canonical_polygon_cycles(&blender_polygons),
            "{mode}: polygon topology"
        );

        let native_smooth_values = native
            .attributes
            .get("shade_smooth")
            .and_then(|attribute| attribute["values"].as_object());
        for face in &native.faces {
            let native_smooth = native_smooth_values
                .and_then(|values| values.get(&format!("f{}", face.id)))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            assert_eq!(
                native_smooth, smooth_shade,
                "{mode}: native face {} smooth-shading flag",
                face.id
            );
        }
        let blender_smooth = reference["smooth"]
            .as_array()
            .ok_or("Blender smooth-shading flags missing")?;
        assert_eq!(
            blender_smooth.len(),
            blender_polygons.len(),
            "{mode}: Blender smooth-shading flag count"
        );
        for (index, smooth) in blender_smooth.iter().enumerate() {
            assert_eq!(
                smooth.as_bool(),
                Some(smooth_shade),
                "{mode}: Blender face {index} smooth-shading flag"
            );
        }

        let blender_max_x = blender_positions
            .iter()
            .map(|point| point.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let native_max_x = native
            .vertices
            .iter()
            .map(|vertex| vertex.co.x)
            .fold(f64::NEG_INFINITY, f64::max);
        if mode == "VOXEL" {
            assert!(
                blender_max_x > 2.8,
                "VOXEL Blender did not preserve the disconnected island"
            );
            assert!(
                native_max_x > 2.8,
                "VOXEL native remesh did not preserve the disconnected island"
            );
        } else {
            assert!(
                blender_max_x < 2.0,
                "{mode} Blender retained the disconnected island despite flood-fill removal"
            );
            assert!(
                native_max_x < 2.0,
                "{mode} native remesh retained the disconnected island"
            );
        }
        eprintln!(
            "{mode}: exact topology, vertices={}, faces={}, max matched vertex error={maximum_vertex_error:.9e}, smooth_shading={smooth_shade}",
            native.vertices.len(),
            native.faces.len()
        );
    }
    Ok(())
}

#[test]
fn blender_dual_contour_depth_and_shape_fixtures_match_exact_geometry() -> Result<(), Box<dyn Error>>
{
    struct Case {
        name: &'static str,
        depth: u32,
        scale: f64,
        remove: bool,
        threshold: f64,
        sharpness: f64,
        modes: &'static [&'static str],
    }
    let Some(blender) = blender_executable() else {
        eprintln!("skipping extended Blender Remesh parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let references = blender_script_json(
        &blender,
        directory.path(),
        r"
import bpy, json, math, sys
results = {}
def append_sphere(positions, polygons, segments, rings, radius, center, indent):
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

def make_object(kind, name):
    if kind == 'uv':
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0)
        obj = bpy.context.object
    elif kind == 'torus':
        bpy.ops.mesh.primitive_torus_add(major_segments=32, minor_segments=12,
                                         major_radius=1.0, minor_radius=0.25)
        obj = bpy.context.object
    elif kind == 'suzanne':
        bpy.ops.mesh.primitive_monkey_add()
        obj = bpy.context.object
    elif kind == 'islands':
        positions, polygons = [], []
        append_sphere(positions, polygons, 16, 12, 1.5, (0.0, 0.0, 0.0), True)
        append_sphere(positions, polygons, 12, 8, 0.4, (2.6, 0.1, 0.15), False)
        mesh = bpy.data.meshes.new(name + 'Input')
        mesh.from_pydata(positions, [], polygons)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
    else:
        raise ValueError(kind)
    obj.name = name
    if name == 'nonuniform_rotated':
        obj.scale = (1.7, 0.65, 1.2)
        obj.rotation_euler = (0.35, -0.5, 0.22)
    return obj

def run_case(name, kind, depth, scale, remove, threshold, sharpness, modes):
    source = make_object(kind, name)
    input_mesh = source.data
    result = {
        'input': {
            'positions': [list(vertex.co) for vertex in input_mesh.vertices],
            'polygons': [list(polygon.vertices) for polygon in input_mesh.polygons],
        },
        'modes': {},
    }
    for mode in modes:
        obj = bpy.data.objects.new(name + mode, input_mesh.copy())
        bpy.context.scene.collection.objects.link(obj)
        obj.location = source.location.copy()
        obj.rotation_mode = source.rotation_mode
        obj.rotation_euler = source.rotation_euler.copy()
        obj.scale = source.scale.copy()
        modifier = obj.modifiers.new('Remesh', 'REMESH')
        modifier.mode = mode
        modifier.use_remove_disconnected = remove
        modifier.use_smooth_shade = False
        modifier.threshold = threshold
        modifier.octree_depth = depth
        modifier.scale = scale
        modifier.sharpness = sharpness
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        result['modes'][mode] = {
            'positions': [list(vertex.co) for vertex in mesh.vertices],
            'polygons': [list(polygon.vertices) for polygon in mesh.polygons],
        }
        evaluated.to_mesh_clear()
        bpy.data.objects.remove(obj, do_unlink=True)
    results[name] = result
    bpy.data.objects.remove(source, do_unlink=True)

all_modes = ('BLOCKS', 'SMOOTH', 'SHARP')
run_case('uv_depth5', 'uv', 5, 0.9, False, 0.15, 1.0, all_modes)
run_case('uv_depth6', 'uv', 6, 0.9, False, 0.15, 1.0, all_modes)
run_case('torus', 'torus', 5, 0.9, False, 0.15, 1.0, all_modes)
run_case('suzanne', 'suzanne', 4, 0.9, False, 0.15, 1.0, all_modes)
run_case('nonuniform_rotated', 'uv', 4, 0.9, False, 0.15, 1.0, all_modes)
run_case('scale_0_6', 'uv', 4, 0.6, False, 0.15, 1.0, all_modes)
run_case('remove_threshold_1', 'islands', 4, 0.9, True, 1.0, 1.0, all_modes)
run_case('remove_threshold_03', 'islands', 4, 0.9, True, 0.3, 1.0, all_modes)
run_case('sharpness_0_5', 'uv', 4, 0.9, False, 0.15, 0.5, ('SHARP',))
run_case('sharpness_2', 'uv', 4, 0.9, False, 0.15, 2.0, ('SHARP',))
with open(sys.argv[-1], 'w', encoding='utf-8') as output:
    json.dump(results, output)
",
    )?;

    let all_modes = &["BLOCKS", "SMOOTH", "SHARP"];
    let cases = [
        Case {
            name: "uv_depth5",
            depth: 5,
            scale: 0.9,
            remove: false,
            threshold: 0.15,
            sharpness: 1.0,
            modes: all_modes,
        },
        Case {
            name: "uv_depth6",
            depth: 6,
            scale: 0.9,
            remove: false,
            threshold: 0.15,
            sharpness: 1.0,
            modes: all_modes,
        },
        Case {
            name: "torus",
            depth: 5,
            scale: 0.9,
            remove: false,
            threshold: 0.15,
            sharpness: 1.0,
            modes: all_modes,
        },
        Case {
            name: "suzanne",
            depth: 4,
            scale: 0.9,
            remove: false,
            threshold: 0.15,
            sharpness: 1.0,
            modes: all_modes,
        },
        Case {
            name: "nonuniform_rotated",
            depth: 4,
            scale: 0.9,
            remove: false,
            threshold: 0.15,
            sharpness: 1.0,
            modes: all_modes,
        },
        Case {
            name: "scale_0_6",
            depth: 4,
            scale: 0.6,
            remove: false,
            threshold: 0.15,
            sharpness: 1.0,
            modes: all_modes,
        },
        Case {
            name: "remove_threshold_1",
            depth: 4,
            scale: 0.9,
            remove: true,
            threshold: 1.0,
            sharpness: 1.0,
            modes: all_modes,
        },
        Case {
            name: "remove_threshold_03",
            depth: 4,
            scale: 0.9,
            remove: true,
            threshold: 0.3,
            sharpness: 1.0,
            modes: all_modes,
        },
        Case {
            name: "sharpness_0_5",
            depth: 4,
            scale: 0.9,
            remove: false,
            threshold: 0.15,
            sharpness: 0.5,
            modes: &["SHARP"],
        },
        Case {
            name: "sharpness_2",
            depth: 4,
            scale: 0.9,
            remove: false,
            threshold: 0.15,
            sharpness: 2.0,
            modes: &["SHARP"],
        },
    ];
    let parse_polygons = |value: &Value| -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
        let values = value.as_array().ok_or("Blender polygon list is missing")?;
        values
            .iter()
            .map(|polygon| {
                polygon
                    .as_array()
                    .and_then(|indices| {
                        indices
                            .iter()
                            .map(|index| {
                                index.as_u64().and_then(|index| usize::try_from(index).ok())
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .ok_or_else(|| "Blender polygon indices are malformed".into())
            })
            .collect()
    };
    let feature_catalog = potter_core::catalog::feature_catalog();
    let open_surface_feature = feature_catalog["features"]
        .as_array()
        .and_then(|features| {
            features
                .iter()
                .find(|feature| feature["feature_id"] == "modifier.remesh.open_nonmanifold")
        })
        .ok_or("open/non-manifold dual-contour feature is missing from the catalog")?;
    assert_eq!(open_surface_feature["status"], "not_supported");

    for case in cases {
        let reference = &references[case.name];
        let input = Mesh::from_positions_and_faces(
            points(&reference["input"]["positions"])?,
            parse_polygons(&reference["input"]["polygons"])?,
        )?;
        for mode in case.modes {
            let context = format!("{}/{}", case.name, mode);
            let params = json!({
                "mode":mode,
                "octree_depth":case.depth,
                "scale":case.scale,
                "sharpness":case.sharpness,
                "use_remove_disconnected":case.remove,
                "threshold":case.threshold,
                "use_smooth_shade":false
            });
            if case.name == "suzanne" {
                let error = evaluate_modifiers(&input, &[modifier("remesh", params)?])
                    .err()
                    .ok_or_else(|| format!("{context}: open Suzanne unexpectedly succeeded"))?;
                assert_eq!(
                    error.code,
                    potter_core::error::ErrorCode::UnsupportedFeature
                );
                assert_eq!(
                    error.details.get("feature_id").and_then(Value::as_str),
                    Some("modifier.remesh.open_nonmanifold"),
                    "{context}: precise open/non-manifold feature gate"
                );
                assert_eq!(
                    error.details.get("mode").and_then(Value::as_str),
                    Some(*mode),
                    "{context}: gated mode"
                );
                eprintln!("{context}: gated open/non-manifold input");
                continue;
            }
            let native = evaluate_modifiers(&input, &[modifier("remesh", params)?])?;
            let expected = &reference["modes"][mode];
            let expected_positions = points(&expected["positions"])?;
            let expected_polygons = parse_polygons(&expected["polygons"])?;
            assert_eq!(
                native.vertices.len(),
                expected_positions.len(),
                "{context}: vertex count"
            );
            assert_eq!(
                native.faces.len(),
                expected_polygons.len(),
                "{context}: face count"
            );
            let native_positions = native
                .vertices
                .iter()
                .map(|vertex| vertex.co)
                .collect::<Vec<_>>();
            let (native_to_expected, maximum_error) =
                match_vertex_positions(&native_positions, &expected_positions, &context);
            let native_ordinals = native
                .vertices
                .iter()
                .enumerate()
                .map(|(index, vertex)| (vertex.id, index))
                .collect::<std::collections::HashMap<_, _>>();
            let native_polygons = native
                .faces
                .iter()
                .map(|face| {
                    face.vertices
                        .iter()
                        .map(|vertex| {
                            native_ordinals
                                .get(vertex)
                                .map(|ordinal| native_to_expected[*ordinal])
                                .ok_or("native remesh face references an unknown vertex")
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(
                canonical_polygon_cycles(&native_polygons),
                canonical_polygon_cycles(&expected_polygons),
                "{context}: polygon topology"
            );
            eprintln!(
                "{context}: exact topology, vertices={}, faces={}, max matched vertex error={maximum_error:.9e}",
                native.vertices.len(),
                native.faces.len()
            );
        }
    }
    Ok(())
}
#[test]
fn ocean_fft_is_seeded_and_choppiness_changes_the_surface() -> Result<(), Box<dyn Error>> {
    let source = Mesh::grid(GridParams {
        size_x: 8.0,
        size_y: 8.0,
        x_subdivisions: 9,
        y_subdivisions: 9,
    })?;
    let create = |choppiness| {
        modifier(
            "ocean",
            json!({
                "geometry_mode":"DISPLACE",
                "resolution":4,
                "viewport_resolution":4,
                "spatial_size":8,
                "random_seed":17,
                "time":1.5,
                "choppiness":choppiness,
                "wave_scale":1.0,
                "wave_direction":0.0,
                "wave_alignment":0.5
            }),
        )
    };
    let first = evaluate_modifiers(&source, &[create(0.0)?])?;
    let again = evaluate_modifiers(&source, &[create(0.0)?])?;
    assert_eq!(first, again);
    let chopped = evaluate_modifiers(&source, &[create(2.0)?])?;
    assert!(
        first
            .vertices
            .iter()
            .zip(&chopped.vertices)
            .any(|(left, right)| {
                (left.co.x - right.co.x).abs() + (left.co.y - right.co.y).abs() > 1.0e-8
            })
    );
    let rms_height = |mesh: &Mesh| {
        (mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co.z.powi(2))
            .sum::<f64>()
            / mesh.vertices.len() as f64)
            .sqrt()
    };
    assert!(rms_height(&first) > 1.0e-10);
    let generated = evaluate_modifiers(
        &source,
        &[modifier(
            "ocean",
            json!({
                "geometry_mode":"GENERATE","resolution":4,"viewport_resolution":4,
                "spatial_size":8,"random_seed":17,"time":1.5,"wave_scale":1.0,
                "wave_direction":-0.3,"wave_alignment":0.65,"repeat_x":2,"repeat_y":1,
                "use_normals":true,"use_foam":true,"foam_layer_name":"foam","choppiness":0.0
            }),
        )?],
    )?;
    assert_eq!(generated.vertices.len(), 33 * 17);
    assert_eq!(generated.faces.len(), 32 * 16);
    assert!((generated.vertices[0].co.x + 8.0).abs() < 1.0e-9);
    assert!((generated.vertices[0].co.y + 4.0).abs() < 1.0e-9);
    assert!(generated.attributes.contains_key("ocean_normal"));
    assert!(generated.attributes.contains_key("foam"));
    Ok(())
}

fn blender_script_json(
    blender: &Path,
    directory: &Path,
    script: &str,
) -> Result<Value, Box<dyn Error>> {
    let script_path = directory.join("modifier_reference.py");
    fs::write(&script_path, script)?;
    let script_path = fs::canonicalize(script_path)?;
    let output_path = fs::canonicalize(directory)?.join("modifier_reference.json");
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .arg("--")
        .arg(&output_path)
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "Blender reference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    if !output_path.is_file() {
        return Err(std::io::Error::other(format!(
            "Blender did not write its reference JSON; stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(serde_json::from_slice(&fs::read(output_path)?)?)
}

fn points(value: &Value) -> Result<Vec<DVec3>, Box<dyn Error>> {
    let rows = value
        .as_array()
        .ok_or_else(|| std::io::Error::other("Blender vertex positions are missing"))?;
    rows.iter()
        .map(|row| {
            let coordinates = row
                .as_array()
                .filter(|coordinates| coordinates.len() == 3)
                .ok_or_else(|| std::io::Error::other("Blender vertex position is invalid"))?;
            Ok(DVec3::new(
                coordinates[0]
                    .as_f64()
                    .ok_or_else(|| std::io::Error::other("Blender X coordinate is invalid"))?,
                coordinates[1]
                    .as_f64()
                    .ok_or_else(|| std::io::Error::other("Blender Y coordinate is invalid"))?,
                coordinates[2]
                    .as_f64()
                    .ok_or_else(|| std::io::Error::other("Blender Z coordinate is invalid"))?,
            ))
        })
        .collect()
}

#[test]
fn blender_surface_and_mesh_deform_bind_match_native_evaluation() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender deformation parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let expected = blender_script_json(
        &blender,
        directory.path(),
        r#"
import bpy, json, sys
def make_grid(name, subdivisions, size, z):
    coords = []
    faces = []
    for y in range(subdivisions):
        for x in range(subdivisions):
            px = -size/2 + size*x/(subdivisions-1)
            py = -size/2 + size*y/(subdivisions-1)
            coords.append((px, py, z(px, py)))
    for y in range(subdivisions-1):
        for x in range(subdivisions-1):
            i = y*subdivisions+x
            faces.append((i, i+1, i+subdivisions+1, i+subdivisions))
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(coords, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj
def activate(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
def evaluated_positions(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    positions = [list(vertex.co) for vertex in mesh.vertices]
    evaluated.to_mesh_clear()
    return positions
surface_source = make_grid("SurfaceSource", 5, 2.0, lambda x, y: 0.35)
surface_target = make_grid("SurfaceTarget", 2, 3.0, lambda x, y:
    0.014*x*x + 0.008*y*y + 0.018*x*y + 0.008*x - 0.004*y)
surface = surface_source.modifiers.new("Surface", "SURFACE_DEFORM")
surface.target = surface_target
activate(surface_source)
bpy.ops.object.surfacedeform_bind(modifier=surface.name)
for index, delta in ((0, (0.0, 0.0, 0.035)), (1, (0.018, 0.0, -0.012)),
                     (2, (0.0, -0.016, 0.024)), (3, (-0.011, 0.008, -0.019))):
    surface_target.data.vertices[index].co.x += delta[0]
    surface_target.data.vertices[index].co.y += delta[1]
    surface_target.data.vertices[index].co.z += delta[2]
bpy.context.view_layer.update()
surface_baseline = [list(vertex.co) for vertex in surface_source.data.vertices]
surface_positions = evaluated_positions(surface_source)
mesh_results = {}
for dynamic, invert, precision in ((True, False, 3), (False, True, 4)):
    source = make_grid("MeshSource", 5, 2.0, lambda x, y: 0.0)
    group = source.vertex_groups.new(name="Influence")
    for index in range(25):
        group.add([index], 0.15 + 0.2*(index % 5), 'REPLACE')
    bpy.ops.mesh.primitive_cube_add(size=4.0)
    cage = bpy.context.object
    cage.name = "Cage"
    mesh_deform = source.modifiers.new("MeshDeform", "MESH_DEFORM")
    mesh_deform.object = cage
    mesh_deform.precision = precision
    mesh_deform.use_dynamic_bind = dynamic
    mesh_deform.vertex_group = group.name
    mesh_deform.invert_vertex_group = invert
    activate(source)
    bpy.ops.object.meshdeform_bind(modifier=mesh_deform.name)
    for vertex in cage.data.vertices:
        x, y, z = vertex.co.copy()
        vertex.co.x += 0.12*y
        vertex.co.y += 0.08*x
        vertex.co.z *= 1.5
    bpy.context.view_layer.update()
    mesh_results["dynamic" if dynamic else "static"] = {
        "baseline":[list(vertex.co) for vertex in source.data.vertices],
        "positions":evaluated_positions(source)
    }
with open(sys.argv[-1], 'w', encoding='utf-8') as output:
    json.dump({
        "surface":{"baseline":surface_baseline,"positions":surface_positions},
        "mesh":mesh_results
    }, output)
"#,
    )?;
    let mut surface_operations = vec![
        json!({"op":"node.create","id":"subject","kind":"grid","params":{"size":2.0,"x_subdivisions":4,"y_subdivisions":4}}),
        json!({"op":"node.create","id":"target","kind":"grid","params":{"size":3.0,"x_subdivisions":1,"y_subdivisions":1}}),
        json!({"op":"mesh.transform_elements","target":{"id":"subject"},"elements":{"domain":"vertex","ids":(0..25).map(|index| format!("v{index}")).collect::<Vec<_>>() },"translation":[0.0,0.0,0.35]}),
    ];
    for index in 0..4 {
        let x = -1.5 + 3.0 * (index % 2) as f64;
        let y = -1.5 + 3.0 * (index / 2) as f64;
        let z = 0.014 * x * x + 0.008 * y * y + 0.018 * x * y + 0.008 * x - 0.004 * y;
        surface_operations.push(json!({
            "op":"mesh.transform_elements","target":{"id":"target"},
            "elements":{"domain":"vertex","ids":[format!("v{index}")]},
            "translation":[0.0,0.0,z]
        }));
    }
    surface_operations.push(json!({
        "op":"modifier.create","target":{"id":"subject"},"id":"deform",
        "type":"surface_deform","params":{"target":"target","falloff":4.0,"strength":1.0}
    }));
    surface_operations.push(json!({
        "op":"modifier.bind","target":{"id":"subject"},"modifier_id":"deform"
    }));
    for (index, delta) in [
        (0, [0.0, 0.0, 0.035]),
        (1, [0.018, 0.0, -0.012]),
        (2, [0.0, -0.016, 0.024]),
        (3, [-0.011, 0.008, -0.019]),
    ] {
        surface_operations.push(json!({
            "op":"mesh.transform_elements","target":{"id":"target"},
            "elements":{"domain":"vertex","ids":[format!("v{index}")]},
            "translation":delta
        }));
    }
    let mut cases = vec![("surface", Value::Array(surface_operations), "surface")];
    for (name, dynamic, invert, precision) in [
        ("mesh_dynamic", true, false, 3),
        ("mesh_static", false, true, 4),
    ] {
        let weights = (0_u32..25)
            .map(|vertex_id| {
                json!({
                    "vertex_id":vertex_id,
                    "weight":0.15 + 0.2 * f64::from(vertex_id % 5)
                })
            })
            .collect::<Vec<_>>();
        let operations = json!([
            {"op":"node.create","id":"subject","kind":"grid","params":{"size":2.0,"x_subdivisions":4,"y_subdivisions":4}},
            {"op":"node.create","id":"cage","kind":"box","params":{"size":4.0}},
            {"op":"vertex_group.create","target":{"id":"subject"},"id":"influence","name":"Influence"},
            {"op":"vertex_group.assign","target":{"id":"subject"},"group_id":"influence","weights":weights},
            {"op":"modifier.create","target":{"id":"subject"},"id":"deform","type":"mesh_deform","params":{"object":"cage","precision":precision,"use_dynamic_bind":dynamic,"vertex_group":"Influence","invert_vertex_group":invert}},
            {"op":"modifier.bind","target":{"id":"subject"},"modifier_id":"deform"},
            {"op":"mesh.transform_elements","target":{"id":"cage","elements":{"domain":"vertex","ids":["v0"]}},"translation":[-0.24,-0.16,-1.0]},
            {"op":"mesh.transform_elements","target":{"id":"cage","elements":{"domain":"vertex","ids":["v1"]}},"translation":[-0.24,0.16,-1.0]},
            {"op":"mesh.transform_elements","target":{"id":"cage","elements":{"domain":"vertex","ids":["v2"]}},"translation":[0.24,0.16,-1.0]},
            {"op":"mesh.transform_elements","target":{"id":"cage","elements":{"domain":"vertex","ids":["v3"]}},"translation":[0.24,-0.16,-1.0]},
            {"op":"mesh.transform_elements","target":{"id":"cage","elements":{"domain":"vertex","ids":["v4"]}},"translation":[-0.24,-0.16,1.0]},
            {"op":"mesh.transform_elements","target":{"id":"cage","elements":{"domain":"vertex","ids":["v5"]}},"translation":[-0.24,0.16,1.0]},
            {"op":"mesh.transform_elements","target":{"id":"cage","elements":{"domain":"vertex","ids":["v6"]}},"translation":[0.24,0.16,1.0]},
            {"op":"mesh.transform_elements","target":{"id":"cage","elements":{"domain":"vertex","ids":["v7"]}},"translation":[0.24,-0.16,1.0]}
        ]);
        cases.push((name, operations, if dynamic { "dynamic" } else { "static" }));
    }
    for (case, operations, expected_key) in cases {
        let scene = directory.path().join(format!("{case}_scene"));
        init(&scene)?;
        let output = apply(&scene, 0, &operations)?;
        assert!(
            output.status.success(),
            "{case} bind apply failed (status={}): stdout={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let doc = load_doc(&scene)?;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let native = snapshot
            .meshes
            .get(&Id::new("subject")?)
            .ok_or("evaluated subject mesh missing")?
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let reference_record = if case == "surface" {
            &expected["surface"]
        } else {
            &expected["mesh"][expected_key]
        };
        let reference = points(&reference_record["positions"])?;
        let baseline = points(&reference_record["baseline"])?;
        assert_eq!(native.len(), reference.len());
        let movements = reference
            .iter()
            .zip(&baseline)
            .map(|(actual, base)| *actual - *base)
            .collect::<Vec<_>>();
        let maximum_displacement = movements
            .iter()
            .map(|movement| movement.length())
            .fold(0.0_f64, f64::max);
        let mean_displacement = movements.iter().copied().sum::<DVec3>() / movements.len() as f64;
        let nonrigid_residual = movements
            .iter()
            .map(|movement| (*movement - mean_displacement).length())
            .fold(0.0_f64, f64::max);
        assert!(
            maximum_displacement > 1.0e-3 && nonrigid_residual > 1.0e-3,
            "{case} Blender fixture did not distinguish identity/rigid translation: max={maximum_displacement}, residual={nonrigid_residual}"
        );
        let errors = native
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| actual.distance(*expected))
            .collect::<Vec<_>>();
        let maximum_error = errors.iter().copied().fold(0.0, f64::max);
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let worst_index = errors
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or("deformation error vector is empty")?;
        eprintln!(
            "{case} deformation max/mean vertex errors: {maximum_error}/{mean_error}; worst={worst_index}, native={}, reference={}",
            native[worst_index], reference[worst_index]
        );
        let tolerance = if case == "surface" { 1.0e-3 } else { 1.0e-4 };
        assert!(
            maximum_error <= tolerance,
            "{case} max/mean vertex errors were {maximum_error}/{mean_error}"
        );
    }
    Ok(())
}

#[test]
fn blender_laplacian_deform_matches_hook_moved_anchor_positions() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender Laplacian Deform parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let expected = blender_script_json(
        &blender,
        directory.path(),
        r#"
import bpy, json, math, sys
def make_grid():
    coords = [(-1.0 + x*0.5, -1.0 + y*0.5, 0.0) for y in range(5) for x in range(5)]
    faces = []
    for y in range(4):
        for x in range(4):
            i = y*5+x
            faces.append((i, i+1, i+6, i+5))
    mesh = bpy.data.meshes.new("SourceMesh")
    mesh.from_pydata(coords, [], faces)
    source = bpy.data.objects.new("Source", mesh)
    bpy.context.scene.collection.objects.link(source)
    return source
source = make_grid()
anchor_indices = [index for index in range(25) if index % 5 in (0, 4) or index < 5 or index >= 20]
anchor_group = source.vertex_groups.new(name="Anchor")
anchor_group.add(anchor_indices, 1.0, 'REPLACE')
hook_targets = {}
for index in anchor_indices:
    group = source.vertex_groups.new(name="Anchor" + str(index))
    group.add([index], 1.0, 'REPLACE')
    target = bpy.data.objects.new("HookTarget" + str(index), None)
    bpy.context.scene.collection.objects.link(target)
    hook = source.modifiers.new("Hook" + str(index), "HOOK")
    hook.object = target
    hook.vertex_group = group.name
    hook.matrix_inverse = target.matrix_world.inverted() @ source.matrix_world
    hook_targets[index] = target
laplacian = source.modifiers.new("Laplacian", "LAPLACIANDEFORM")
laplacian.vertex_group = anchor_group.name
laplacian.iterations = 8
bpy.ops.object.select_all(action='DESELECT')
source.select_set(True)
bpy.context.view_layer.objects.active = source
bpy.ops.object.laplaciandeform_bind(modifier=laplacian.name)
angle = 0.35
for index, target in hook_targets.items():
    x, y = source.data.vertices[index].co[:2]
    target.location = (math.cos(angle)*x-math.sin(angle)*y-x,
                       math.sin(angle)*x+math.cos(angle)*y-y, 0.5)
bpy.context.view_layer.update()
laplacian.show_viewport = False
depsgraph = bpy.context.evaluated_depsgraph_get()
hook_evaluated = source.evaluated_get(depsgraph)
hook_mesh = hook_evaluated.to_mesh()
hook_positions = [list(vertex.co) for vertex in hook_mesh.vertices]
hook_evaluated.to_mesh_clear()
laplacian.show_viewport = True
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = source.evaluated_get(depsgraph)
mesh = evaluated.to_mesh()
baseline = [list(vertex.co) for vertex in source.data.vertices]
positions = [list(vertex.co) for vertex in mesh.vertices]
evaluated.to_mesh_clear()
with open(sys.argv[-1], 'w', encoding='utf-8') as output:
    json.dump({'baseline':baseline, 'hook_positions':hook_positions, 'positions':positions,
               'edges':[list(edge.vertices) for edge in source.data.edges]}, output)
"#,
    )?;
    let scene = directory.path().join("laplacian_scene");
    init(&scene)?;
    let anchor_indices = (0_u32..25)
        .filter(|vertex_id| {
            vertex_id % 5 == 0 || vertex_id % 5 == 4 || *vertex_id < 5 || *vertex_id >= 20
        })
        .collect::<Vec<_>>();
    let weights = anchor_indices
        .iter()
        .map(|vertex_id| json!({"vertex_id":vertex_id,"weight":1.0}))
        .collect::<Vec<_>>();
    let angle = 0.35_f64;
    let mut operations = vec![
        json!({"op":"node.create","id":"subject","kind":"grid","params":{"size":2.0,"x_subdivisions":4,"y_subdivisions":4}}),
        json!({"op":"vertex_group.create","target":{"id":"subject"},"id":"anchor","name":"Anchor"}),
        json!({"op":"vertex_group.assign","target":{"id":"subject"},"group_id":"anchor","weights":weights}),
    ];
    for &vertex_id in &anchor_indices {
        let group_id = format!("anchor_{vertex_id}");
        let group_name = format!("Anchor{vertex_id}");
        let target_id = format!("hook_target_{vertex_id}");
        operations.push(json!({"op":"node.create","id":target_id,"kind":"empty"}));
        operations.push(json!({"op":"vertex_group.create","target":{"id":"subject"},"id":group_id,"name":group_name}));
        operations.push(json!({"op":"vertex_group.assign","target":{"id":"subject"},"group_id":group_id,"weights":[{"vertex_id":vertex_id,"weight":1.0}]}));
        operations.push(json!({"op":"modifier.create","target":{"id":"subject"},"id":format!("hook_{vertex_id}"),"type":"hook","params":{"object":target_id,"vertex_group":group_name,"strength":1.0}}));
    }
    operations.push(json!({
        "op":"modifier.create","target":{"id":"subject"},"id":"laplace",
        "type":"laplacian_deform","params":{"vertex_group":"Anchor","iterations":8}
    }));
    operations
        .push(json!({"op":"modifier.bind","target":{"id":"subject"},"modifier_id":"laplace"}));
    for &vertex_id in &anchor_indices {
        let x = -1.0 + 0.5 * f64::from(vertex_id % 5);
        let y = -1.0 + 0.5 * f64::from(vertex_id / 5);
        let translation = [
            angle.cos() * x - angle.sin() * y - x,
            angle.sin() * x + angle.cos() * y - y,
            0.5,
        ];
        operations.push(json!({
            "op":"node.update","target":{"id":format!("hook_target_{vertex_id}")},
            "set":{"transform":{"translation":translation}}
        }));
    }
    let output = apply(&scene, 0, &Value::Array(operations))?;
    assert!(
        output.status.success(),
        "Laplacian Deform bind failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let mut doc = load_doc(&scene)?;
    let subject_id = Id::new("subject")?;
    let data_id = doc.nodes[&subject_id]
        .data
        .as_ref()
        .ok_or("subject data missing")?
        .clone();
    let mesh = doc
        .data_blocks
        .get_mut(&data_id)
        .and_then(|data| data.mesh.as_mut())
        .ok_or("subject mesh missing")?;
    let mut blender_edge_order = Vec::with_capacity(mesh.edges.len());
    for edge in expected["edges"]
        .as_array()
        .ok_or("Blender edge order missing")?
    {
        let vertices = edge
            .as_array()
            .filter(|vertices| vertices.len() == 2)
            .ok_or("Blender edge has invalid vertex indices")?;
        let first = usize::try_from(vertices[0].as_u64().ok_or("invalid edge endpoint")?)?;
        let second = usize::try_from(vertices[1].as_u64().ok_or("invalid edge endpoint")?)?;
        let first_id = mesh
            .vertices
            .get(first)
            .ok_or("Blender edge endpoint is outside the mesh")?
            .id;
        let second_id = mesh
            .vertices
            .get(second)
            .ok_or("Blender edge endpoint is outside the mesh")?
            .id;
        let edge_index = mesh
            .edges
            .iter()
            .position(|edge| {
                (edge.vertices[0] == first_id && edge.vertices[1] == second_id)
                    || (edge.vertices[0] == second_id && edge.vertices[1] == first_id)
            })
            .ok_or("Blender edge does not exist in the Potter mesh")?;
        blender_edge_order.push(mesh.edges.swap_remove(edge_index));
    }
    if !mesh.edges.is_empty() {
        return Err("Blender and Potter edge counts differ".into());
    }
    mesh.edges = blender_edge_order;
    let anchor_id = Id::new("anchor")?;
    assert!(
        doc.data_blocks[&data_id]
            .vertex_weights
            .get(&4)
            .and_then(|groups| groups.get(&anchor_id))
            .is_some_and(|weight| (*weight - 1.0).abs() < f64::EPSILON)
    );
    let mut hook_only_doc = doc.clone();
    let hook_only_node = hook_only_doc
        .nodes
        .get_mut(&subject_id)
        .ok_or("subject node missing")?;
    let laplace_id = Id::new("laplace")?;
    let laplace = hook_only_node
        .modifiers
        .iter_mut()
        .find(|modifier| modifier.id == laplace_id)
        .ok_or("Laplacian Deform modifier missing")?;
    laplace.enabled = false;
    let hook_only = Snapshot::evaluate(&hook_only_doc, &EvaluationContext::default())?;
    let hook_mesh = hook_only
        .meshes
        .get(&subject_id)
        .ok_or("Hook-only mesh missing")?;
    let hook_reference = points(&expected["hook_positions"])?;
    let hook_native = hook_mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let maximum_hook_error = hook_native
        .iter()
        .zip(&hook_reference)
        .map(|(actual, expected)| actual.distance(*expected))
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_hook_error <= 1.0e-5,
        "Hook-only anchor transform max error was {maximum_hook_error}"
    );
    let baseline = points(&expected["baseline"])?;
    assert!(
        hook_mesh
            .vertices
            .iter()
            .filter(|vertex| anchor_indices.contains(&vertex.id))
            .all(|vertex| {
                usize::try_from(vertex.id)
                    .ok()
                    .and_then(|index| baseline.get(index))
                    .is_some_and(|position| vertex.co.distance(*position) > 0.1)
            })
    );
    assert!(
        hook_mesh
            .vertices
            .iter()
            .filter(|vertex| !anchor_indices.contains(&vertex.id))
            .all(|vertex| {
                usize::try_from(vertex.id)
                    .ok()
                    .and_then(|index| baseline.get(index))
                    .is_some_and(|position| vertex.co.distance(*position) < 1.0e-6)
            })
    );
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    let native = snapshot
        .meshes
        .get(&Id::new("subject")?)
        .ok_or("evaluated subject mesh missing")?
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let reference = points(&expected["positions"])?;
    assert_eq!(native.len(), reference.len());
    let movements = reference
        .iter()
        .zip(&baseline)
        .map(|(position, original)| *position - *original)
        .collect::<Vec<_>>();
    let maximum_displacement = movements
        .iter()
        .map(|movement| movement.length())
        .fold(0.0_f64, f64::max);
    let mean_displacement = movements.iter().copied().sum::<DVec3>() / movements.len() as f64;
    let nonrigid_residual = movements
        .iter()
        .map(|movement| (*movement - mean_displacement).length())
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_displacement > 1.0e-2 && nonrigid_residual > 1.0e-2,
        "Laplacian reference did not distinguish identity/rigid translation: max={maximum_displacement}, residual={nonrigid_residual}"
    );
    let errors = native
        .iter()
        .zip(&reference)
        .map(|(actual, expected)| actual.distance(*expected))
        .collect::<Vec<_>>();
    let maximum_error = errors.iter().copied().fold(0.0, f64::max);
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
    let worst_index = errors
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or("Laplacian error vector is empty")?;
    eprintln!(
        "Laplacian max/mean vertex errors: {maximum_error}/{mean_error}; worst={worst_index}, native={}, Blender={}",
        native[worst_index], reference[worst_index]
    );
    assert!(
        maximum_error <= 1.0e-5,
        "Laplacian Deform max/mean vertex errors were {maximum_error}/{mean_error}"
    );
    Ok(())
}

#[test]
fn blender_ocean_direction_and_choppiness_match_displacement_fields() -> Result<(), Box<dyn Error>>
{
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender ocean parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let references = blender_script_json(
        &blender,
        directory.path(),
        r#"
import bpy, json, math, sys
def make_grid(name):
    coords = [(-4.0 + 0.5*x, -4.0 + 0.5*y, 0.0) for y in range(17) for x in range(17)]
    faces = []
    for y in range(16):
        for x in range(16):
            i = y*17+x
            faces.append((i, i+1, i+18, i+17))
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(coords, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj
def evaluated_positions(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    positions = [list(vertex.co) for vertex in mesh.vertices]
    evaluated.to_mesh_clear()
    return positions
results = {}
for choppiness in (0.0, 1.0):
    obj = make_grid("Ocean" + str(choppiness))
    modifier = obj.modifiers.new("Ocean", "OCEAN")
    modifier.geometry_mode = 'DISPLACE'
    modifier.resolution = 4
    modifier.viewport_resolution = 4
    modifier.spatial_size = 8
    modifier.wave_scale = 1.0
    modifier.wind_velocity = 20.0
    modifier.random_seed = 17
    modifier.time = 1.5
    modifier.wave_alignment = 1.0
    modifier.wave_direction = 0.4
    modifier.choppiness = choppiness
    results[str(choppiness)] = evaluated_positions(obj)
with open(sys.argv[-1], 'w', encoding='utf-8') as output:
    json.dump(results, output)
"#,
    )?;
    let native_positions = |choppiness: f64| -> Result<Vec<DVec3>, Box<dyn Error>> {
        let source = Mesh::grid(GridParams {
            size_x: 8.0,
            size_y: 8.0,
            x_subdivisions: 17,
            y_subdivisions: 17,
        })?;
        let modifier = Modifier {
            id: Id::new("sea")?,
            modifier_type: "ocean".to_owned(),
            name: "Ocean".to_owned(),
            enabled: true,
            params: serde_json::from_value(json!({
                "geometry_mode":"DISPLACE","resolution":4,"viewport_resolution":4,
                "spatial_size":8.0,"wave_scale":1.0,"wind_velocity":20.0,
                "random_seed":17,"time":1.5,"wave_alignment":1.0,
                "wave_direction":0.4,"choppiness":choppiness
            }))?,
            binding_data: None,
            runtime: potter_core::model::ModifierRuntime::default(),
        };
        let output = evaluate_modifiers(&source, &[modifier])?;
        Ok(output.vertices.iter().map(|vertex| vertex.co).collect())
    };
    let native_still = native_positions(0.0)?;
    let native_chopped = native_positions(1.0)?;
    let reference_still = points(&references["0.0"])?;
    let reference_chopped = points(&references["1.0"])?;
    assert_eq!(native_still.len(), reference_still.len());
    assert_eq!(native_chopped.len(), reference_chopped.len());
    let baseline_delta = reference_still
        .iter()
        .zip((0..17).flat_map(|y| {
            (0..17).map(move |x| DVec3::new(-4.0 + 0.5 * x as f64, -4.0 + 0.5 * y as f64, 0.0))
        }))
        .map(|(actual, base)| actual.distance(base))
        .fold(0.0_f64, f64::max);
    assert!(
        baseline_delta > 1.0e-6,
        "Blender ocean baseline was not displaced: {baseline_delta}"
    );
    let reference_rms = |positions: &[DVec3]| -> DVec3 {
        let mut sum = DVec3::ZERO;
        for (index, point) in positions.iter().enumerate() {
            let x = -4.0 + 0.5 * (index % 17) as f64;
            let y = -4.0 + 0.5 * (index / 17) as f64;
            let delta = *point - DVec3::new(x, y, 0.0);
            sum += delta * delta;
        }
        let mean_square = sum / positions.len() as f64;
        DVec3::new(
            mean_square.x.sqrt(),
            mean_square.y.sqrt(),
            mean_square.z.sqrt(),
        )
    };
    let still_rms = reference_rms(&reference_still);
    let chopped_rms = reference_rms(&reference_chopped);
    let native_still_rms = reference_rms(&native_still);
    let native_chopped_rms = reference_rms(&native_chopped);
    assert!(
        still_rms.z > 1.0e-6,
        "Blender height field was trivial: {still_rms}"
    );
    assert!(chopped_rms.x.hypot(chopped_rms.y) > still_rms.x.hypot(still_rms.y) + 1.0e-6);
    let still_height_ratio = native_still_rms.z / still_rms.z;
    let height_ratio = native_chopped_rms.z / chopped_rms.z;
    let horizontal_ratio =
        native_chopped_rms.x.hypot(native_chopped_rms.y) / chopped_rms.x.hypot(chopped_rms.y);
    assert!(
        (0.85..=1.15).contains(&still_height_ratio) && (0.85..=1.15).contains(&height_ratio),
        "Ocean still/chopped height RMS ratios were {still_height_ratio}/{height_ratio}"
    );
    assert!(
        (0.85..=1.15).contains(&horizontal_ratio),
        "Ocean choppiness displacement RMS ratio was {horizontal_ratio}"
    );
    let direction = |positions: &[DVec3]| {
        let mut xx = 0.0;
        let mut yy = 0.0;
        let mut xy = 0.0;
        for y in 1..16 {
            for x in 1..16 {
                let index = y * 17 + x;
                let gx = positions[index + 1].z - positions[index - 1].z;
                let gy = positions[index + 17].z - positions[index - 17].z;
                xx += gx * gx;
                yy += gy * gy;
                xy += gx * gy;
            }
        }
        0.5 * (2.0 * xy).atan2(xx - yy)
    };
    let direction_error = (direction(&native_chopped) - direction(&reference_chopped))
        .abs()
        .rem_euclid(std::f64::consts::PI);
    let direction_error = direction_error.min(std::f64::consts::PI - direction_error);
    assert!(
        direction_error <= 0.12,
        "dominant wave-axis difference was {direction_error} radians"
    );
    // Blender hashes each wave coordinate for its seeded phases; compare the
    // deterministic spatial power distribution rather than unrelated phases.
    let spectral_bands = |positions: &[DVec3]| {
        let resolution = 16;
        let mut bands = [0.0; 3];
        for frequency_y in 0..resolution {
            for frequency_x in 0..resolution {
                let signed_x = if frequency_x <= resolution / 2 {
                    frequency_x as f64
                } else {
                    frequency_x as f64 - resolution as f64
                };
                let signed_y = if frequency_y <= resolution / 2 {
                    frequency_y as f64
                } else {
                    frequency_y as f64 - resolution as f64
                };
                let radius = signed_x.hypot(signed_y);
                if radius == 0.0 {
                    continue;
                }
                let mut real = 0.0;
                let mut imaginary = 0.0;
                for y in 0..resolution {
                    for x in 0..resolution {
                        let phase = std::f64::consts::TAU
                            * (signed_x * x as f64 + signed_y * y as f64)
                            / resolution as f64;
                        let height = positions[y * 17 + x].z;
                        real += height * phase.cos();
                        imaginary -= height * phase.sin();
                    }
                }
                let band = if radius <= 2.0 {
                    0
                } else if radius <= 5.0 {
                    1
                } else {
                    2
                };
                bands[band] += real * real + imaginary * imaginary;
            }
        }
        let total = bands.iter().sum::<f64>();
        bands.map(|power| power / total)
    };
    let native_spectrum = spectral_bands(&native_chopped);
    let blender_spectrum = spectral_bands(&reference_chopped);
    for band in 0..3 {
        assert!(
            (native_spectrum[band] - blender_spectrum[band]).abs() <= 0.25,
            "Ocean frequency-band {band} power fractions differed: {} vs {}",
            native_spectrum[band],
            blender_spectrum[band]
        );
    }
    eprintln!(
        "Ocean field parity still/chopped height RMS ratios={still_height_ratio:.4}/{height_ratio:.4}, horizontal RMS ratio={horizontal_ratio:.4}, dominant-axis error={direction_error:.4} rad, power bands native={native_spectrum:?}, Blender={blender_spectrum:?}"
    );
    Ok(())
}

fn voxel_remesh_fixture() -> Result<Mesh, Box<dyn Error>> {
    let mut monkey = Mesh::uv_sphere(UvSphereParams {
        segments: 16,
        ring_count: 12,
        radius: 1.5,
    })?;
    for vertex in &mut monkey.vertices {
        let position = vertex.co;
        if position.x > 1.0 && position.y.abs() < 0.65 && position.z.abs() < 0.65 {
            vertex.co *= 0.4;
        }
    }
    let mut cube = Mesh::box_mesh(BoxParams {
        size: DVec3::splat(0.8),
    })?;
    for vertex in &mut cube.vertices {
        vertex.co += DVec3::new(2.6, 0.1, 0.15);
    }
    let mut positions = monkey
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let mut polygons = Vec::new();
    for face in &monkey.faces {
        let indices = monkey
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<std::collections::HashMap<_, _>>();
        polygons.push(face.vertices.iter().map(|vertex| indices[vertex]).collect());
    }
    let cube_offset = positions.len();
    let cube_indices = cube
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, cube_offset + index))
        .collect::<std::collections::HashMap<_, _>>();
    positions.extend(cube.vertices.iter().map(|vertex| vertex.co));
    for face in &cube.faces {
        polygons.push(
            face.vertices
                .iter()
                .map(|vertex| cube_indices[vertex])
                .collect(),
        );
    }
    Ok(Mesh::from_positions_and_faces(positions, polygons)?)
}

fn assert_voxel_remesh_geometry(
    actual: &Mesh,
    expected: &Value,
    context: &str,
) -> Result<(), Box<dyn Error>> {
    let expected_positions = points(&expected["positions"])?;
    let expected_polygons = expected["polygons"]
        .as_array()
        .ok_or("Blender voxel remesh polygons missing")?
        .iter()
        .map(|polygon| {
            polygon
                .as_array()
                .and_then(|indices| {
                    indices
                        .iter()
                        .map(|index| index.as_u64().and_then(|index| usize::try_from(index).ok()))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or("Blender voxel polygon indices were malformed")
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        actual.vertices.len(),
        expected_positions.len(),
        "{context}: vertex count"
    );
    assert_eq!(
        actual.faces.len(),
        expected_polygons.len(),
        "{context}: face count"
    );
    let actual_positions = actual
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let (actual_to_expected, maximum_vertex_error) =
        match_vertex_positions(&actual_positions, &expected_positions, context);
    let actual_ordinals = actual
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<std::collections::HashMap<_, _>>();
    let actual_polygons = actual
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|vertex| {
                    actual_ordinals
                        .get(vertex)
                        .map(|ordinal| actual_to_expected[*ordinal])
                        .ok_or("voxel remesh face references an unknown vertex")
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        canonical_polygon_cycles(&actual_polygons),
        canonical_polygon_cycles(&expected_polygons),
        "{context}: polygon topology"
    );
    eprintln!(
        "{context}: exact topology, vertices={}, faces={}, max matched vertex error={maximum_vertex_error:.9e}",
        actual.vertices.len(),
        actual.faces.len()
    );
    Ok(())
}

#[test]
fn blender_voxel_remesh_matches_exact_geometry_at_two_voxel_sizes() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("Skipping Blender VOXEL parity: no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let mut source = voxel_remesh_fixture()?;
    round_mesh_positions_to_blender_precision(&mut source);
    let script = r#"
import bpy, json, sys
fixture = __POTTER_INPUT__
positions = fixture["positions"]
polygons = fixture["polygons"]
source_mesh = bpy.data.meshes.new('VoxelInputMesh')
source_mesh.from_pydata(positions, [], polygons)
source_mesh.update()
source = bpy.data.objects.new('VoxelInput', source_mesh)
bpy.context.scene.collection.objects.link(source)
results = {}
for voxel_size, smooth_shade in ((0.1, False), (0.05, True)):
    obj = source.copy()
    obj.data = source.data.copy()
    bpy.context.scene.collection.objects.link(obj)
    modifier = obj.modifiers.new('VoxelRemesh', 'REMESH')
    modifier.mode = 'VOXEL'
    modifier.voxel_size = voxel_size
    modifier.adaptivity = 0.0
    modifier.use_smooth_shade = smooth_shade
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    results[str(voxel_size)] = {
        'positions':[list(vertex.co) for vertex in mesh.vertices],
        'polygons':[list(polygon.vertices) for polygon in mesh.polygons],
        'smooth':[polygon.use_smooth for polygon in mesh.polygons]
    }
    evaluated.to_mesh_clear()
    bpy.data.objects.remove(obj, do_unlink=True)
with open(sys.argv[-1], 'w', encoding='utf-8') as output:
    json.dump(results, output)
    "#
    .replace("__POTTER_INPUT__", &mesh_fixture_json(&source).to_string());
    let references = blender_script_json(&blender, directory.path(), &script)?;
    for (voxel_size, smooth_shade) in [(0.1_f64, false), (0.05, true)] {
        let key = voxel_size.to_string();
        let native = evaluate_modifiers(
            &source,
            &[modifier(
                "remesh",
                json!({
                    "mode":"VOXEL",
                    "voxel_size":voxel_size,
                    "adaptivity":0.0,
                    "use_smooth_shade":smooth_shade
                }),
            )?],
        )?;
        assert_voxel_remesh_geometry(
            &native,
            &references[&key],
            &format!("VOXEL voxel_size={key}"),
        )?;
        let blender_smooth = references[&key]["smooth"]
            .as_array()
            .ok_or("Blender voxel smooth-shading flags missing")?;
        assert_eq!(blender_smooth.len(), native.faces.len());
        for (index, face) in native.faces.iter().enumerate() {
            let actual_smooth = native
                .attributes
                .get("shade_smooth")
                .and_then(|attribute| attribute["values"].as_object())
                .and_then(|values| values.get(&format!("f{}", face.id)))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            assert_eq!(
                actual_smooth, smooth_shade,
                "native face {} smooth flag",
                face.id
            );
            assert_eq!(
                blender_smooth[index].as_bool(),
                Some(smooth_shade),
                "Blender face {index} smooth flag"
            );
        }
    }
    Ok(())
}
