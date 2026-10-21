//! Blender 5.2.2 parity for VOXEL Remesh with adaptivity.
//!
//! The checked-in Blender 5.2.2 (`OpenVDB` 13.0.0) fixture covers cube, open box, closed cylinder,
//! and cube plus plane at voxel sizes 0.1 and 0.3 with adaptivity 0.0 through 1.0. When Blender is
//! available, the live oracle also covers a plane, open cylinder, non-manifold book, and empty mesh.

use std::{collections::HashMap, error::Error, fs, path::Path, process::Command};

use glam::DVec3;
use potter_core::{
    geom::{Mesh, modifiers::evaluate_modifiers},
    model::{Id, Modifier, ModifierRuntime},
};
#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;
use serde_json::{Value, json};
use tempfile::tempdir;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/remesh_voxel_adaptivity/blender_5_2_2.json"
);
/// Blender's float32 vertex positions compared against Potter's float32 positions.
const POSITION_TOLERANCE: f64 = 2.5e-5;

fn load_fixture() -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(FIXTURE)?)?)
}

fn supplemental_inputs() -> Value {
    let cylinder_positions = (0..24)
        .map(|index| {
            let ring = index / 12;
            let angle = std::f64::consts::TAU * (index % 12) as f64 / 12.0;
            json!([angle.cos(), angle.sin(), if ring == 0 { -1.0 } else { 1.0 }])
        })
        .collect::<Vec<_>>();
    let cylinder_faces = (0..12)
        .map(|index| json!([index, (index + 1) % 12, 12 + (index + 1) % 12, 12 + index]))
        .collect::<Vec<_>>();
    json!({
        "plane": {
            "positions":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],
            "polygons":[[0,1,2,3]]
        },
        "cyl_open": {
            "positions":cylinder_positions,
            "polygons":cylinder_faces
        },
        "book_nonmanifold": {
            "positions":[[0,0,0],[0,0,1],[1,0,0],[1,0,1],[0,1,0],[0,1,1],[-1,0,0],[-1,0,1]],
            "polygons":[[0,1,3,2],[0,1,5,4],[0,1,7,6]]
        },
        "empty":{"positions":[],"polygons":[]}
    })
}

/// Generate live Blender references when available; the checked-in Blender fixture is the fallback.
fn blender_reference_cases(
    blender: &Path,
    inputs: &Value,
    base_cases: &Value,
) -> Result<Value, Box<dyn Error>> {
    let mut cases = base_cases
        .as_array()
        .ok_or("fixture cases are missing")?
        .clone();
    for kind in ["plane", "cyl_open", "book_nonmanifold", "empty"] {
        for voxel_size in [0.1, 0.3] {
            for adaptivity in [0.0, 0.05, 0.2, 0.5, 1.0] {
                cases.push(json!({
                    "kind":kind,
                    "voxel_size":voxel_size,
                    "adaptivity":adaptivity
                }));
            }
        }
    }
    let input = json!({"inputs":inputs,"cases":cases});
    let directory = tempdir()?;
    let input_path = directory.path().join("voxel_adaptivity_input.json");
    let script_path = directory.path().join("voxel_adaptivity_reference.py");
    let output_path = directory.path().join("voxel_adaptivity_reference.json");
    fs::write(&input_path, serde_json::to_vec(&input)?)?;
    fs::write(
        &script_path,
        r"
import bpy, json, sys
data = json.load(open(sys.argv[-2]))
results = []
for case in data['cases']:
    source = data['inputs'][case['kind']]
    mesh = bpy.data.meshes.new('Input_' + case['kind'])
    mesh.from_pydata(source['positions'], [], source['polygons'])
    mesh.update()
    obj = bpy.data.objects.new('Owner', mesh)
    bpy.context.scene.collection.objects.link(obj)
    modifier = obj.modifiers.new('Remesh', 'REMESH')
    modifier.mode = 'VOXEL'
    modifier.voxel_size = case['voxel_size']
    modifier.adaptivity = case['adaptivity']
    modifier.use_smooth_shade = False
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    output = evaluated.to_mesh()
    positions = [[float(c) for c in vertex.co] for vertex in output.vertices]
    polygons = [[int(index) for index in polygon.vertices] for polygon in output.polygons]
    results.append({
        'kind': case['kind'], 'voxel_size': case['voxel_size'],
        'adaptivity': case['adaptivity'],
        'vertices': len(positions), 'faces': len(polygons),
        'quads': sum(len(polygon) == 4 for polygon in polygons),
        'tris': sum(len(polygon) == 3 for polygon in polygons),
        'positions': positions, 'polygons': polygons,
    })
    evaluated.to_mesh_clear()
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh)
with open(sys.argv[-1], 'w') as output_file:
    json.dump(results, output_file)
",
    )?;
    let output = Command::new(blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script_path)
        .arg("--")
        .arg(&input_path)
        .arg(&output_path)
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "Blender VOXEL reference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(serde_json::from_slice(&fs::read(output_path)?)?)
}

fn remesh_modifier(params: Value) -> Result<Modifier, Box<dyn Error>> {
    Ok(Modifier {
        id: Id::new("remesh_voxel_adaptivity")?,
        modifier_type: "remesh".to_owned(),
        name: "remesh".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: serde_json::from_value(params)?,
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

fn points(value: &Value) -> Result<Vec<DVec3>, Box<dyn Error>> {
    value
        .as_array()
        .ok_or("fixture positions are missing")?
        .iter()
        .map(|position| {
            let coords = position
                .as_array()
                .ok_or("fixture position is malformed")?
                .iter()
                .map(|coord| coord.as_f64().ok_or("fixture coordinate is malformed"))
                .collect::<Result<Vec<_>, _>>()?;
            match coords.as_slice() {
                [x, y, z] => Ok(DVec3::new(*x, *y, *z)),
                _ => Err("fixture position must have three coordinates".into()),
            }
        })
        .collect()
}

fn polygons(value: &Value) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    value
        .as_array()
        .ok_or("fixture polygons are missing")?
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
                .ok_or_else(|| "fixture polygon indices are malformed".into())
        })
        .collect()
}

fn mesh_from_fixture(value: &Value) -> Result<Mesh, Box<dyn Error>> {
    Ok(Mesh::from_positions_and_faces(
        points(&value["positions"])?,
        polygons(&value["polygons"])?,
    )?)
}

/// Rotation-canonical polygon cycles; orientation is significant.
fn canonical_cycles(polygons: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut cycles = polygons
        .iter()
        .map(|polygon| {
            let count = polygon.len();
            (0..count)
                .map(|start| {
                    (0..count)
                        .map(|offset| polygon[(start + offset) % count])
                        .collect::<Vec<_>>()
                })
                .min()
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    cycles.sort();
    cycles
}

/// One-to-one nearest matching of Potter vertices to Blender vertices.
fn match_positions(actual: &[DVec3], expected: &[DVec3], context: &str) -> (Vec<usize>, f64) {
    let mut used = vec![false; expected.len()];
    let mut mapping = Vec::with_capacity(actual.len());
    let mut maximum_error: f64 = 0.0;
    for (index, position) in actual.iter().enumerate() {
        let Some((best, error)) = expected
            .iter()
            .enumerate()
            .filter(|(candidate, _)| !used[*candidate])
            .map(|(candidate, target)| (candidate, position.distance(*target)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
        else {
            panic!("{context}: Potter produced more vertices than Blender");
        };
        assert!(
            error <= POSITION_TOLERANCE,
            "{context}: vertex {index} at {position:?} is {error:e} from its nearest Blender vertex"
        );
        used[best] = true;
        mapping.push(best);
        maximum_error = maximum_error.max(error);
    }
    (mapping, maximum_error)
}

fn assert_matches_blender(
    actual: &Mesh,
    expected: &Value,
    context: &str,
) -> Result<(), Box<dyn Error>> {
    let expected_positions = points(&expected["positions"])?;
    let expected_polygons = polygons(&expected["polygons"])?;
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
    let expected_quads = expected_polygons
        .iter()
        .filter(|polygon| polygon.len() == 4)
        .count();
    let actual_quads = actual
        .faces
        .iter()
        .filter(|face| face.vertices.len() == 4)
        .count();
    assert_eq!(actual_quads, expected_quads, "{context}: quad count");
    assert_eq!(
        actual.faces.len() - actual_quads,
        expected_polygons.len() - expected_quads,
        "{context}: triangle count"
    );

    let actual_positions = actual
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let (actual_to_expected, maximum_error) =
        match_positions(&actual_positions, &expected_positions, context);
    let order_error = actual_positions
        .iter()
        .zip(&expected_positions)
        .map(|(actual, expected)| actual.distance(*expected))
        .fold(0.0_f64, f64::max);
    assert!(
        order_error <= POSITION_TOLERANCE,
        "{context}: vertex generation order differs from Blender by {order_error:e}"
    );
    eprintln!("{context}: vertex-order max error={order_error:.3e}");
    let ordinals = actual
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let mapped_faces = actual
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| actual_to_expected[ordinals[id]])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        canonical_cycles(&mapped_faces),
        canonical_cycles(&expected_polygons),
        "{context}: polygon cycles"
    );
    eprintln!(
        "{context}: vertices={} faces={} max vertex error={maximum_error:.3e}",
        actual.vertices.len(),
        actual.faces.len()
    );
    Ok(())
}

#[test]
fn voxel_remesh_cases_match_blender_counts_positions_and_polygon_cycles()
-> Result<(), Box<dyn Error>> {
    let fixture = load_fixture()?;
    let mut extended_inputs = fixture["inputs"].clone();
    let additional_inputs = supplemental_inputs();
    let supplemental = additional_inputs
        .as_object()
        .ok_or("supplemental inputs are malformed")?;
    extended_inputs
        .as_object_mut()
        .ok_or("fixture inputs are malformed")?
        .extend(supplemental.clone());
    let (cases, expected_count) = if let Some(blender) = blender_executable() {
        (
            blender_reference_cases(&blender, &extended_inputs, &fixture["cases"])?,
            80,
        )
    } else {
        (fixture["cases"].clone(), 40)
    };
    let cases = cases.as_array().ok_or("Blender remesh cases are missing")?;
    assert_eq!(cases.len(), expected_count);
    for case in cases {
        let kind = case["kind"]
            .as_str()
            .ok_or("fixture case kind is missing")?;
        let voxel_size = case["voxel_size"]
            .as_f64()
            .ok_or("fixture voxel_size is missing")?;
        let adaptivity = case["adaptivity"]
            .as_f64()
            .ok_or("fixture adaptivity is missing")?;
        let context = format!("{kind} voxel_size={voxel_size} adaptivity={adaptivity}");
        let source = mesh_from_fixture(&extended_inputs[kind])?;
        let params = json!({
            "mode":"VOXEL","voxel_size":voxel_size,"adaptivity":adaptivity,
            "use_smooth_shade":false,"use_remove_disconnected":false,"threshold":0.15
        });
        let native = evaluate_modifiers(&source, &[remesh_modifier(params)?], "Owner")?;
        assert_matches_blender(&native, case, &context)?;
    }
    Ok(())
}

#[test]
fn voxel_remesh_without_a_surface_passes_the_input_through() -> Result<(), Box<dyn Error>> {
    let fixture = load_fixture()?;
    let entries = fixture["passthrough"]
        .as_array()
        .ok_or("fixture passthrough cases are missing")?;
    assert!(
        !entries.is_empty(),
        "fixture must contain a passthrough case"
    );
    for entry in entries {
        let source = mesh_from_fixture(entry)?;
        let params = json!({
            "mode":"VOXEL","voxel_size":entry["voxel_size"],"adaptivity":entry["adaptivity"],
            "use_smooth_shade":false,"use_remove_disconnected":false,"threshold":0.15
        });
        let native = evaluate_modifiers(&source, &[remesh_modifier(params)?], "Owner")?;
        assert!(
            native.faces.is_empty(),
            "Blender returns nullptr without faces; the input passes through"
        );
        assert_eq!(
            native
                .vertices
                .iter()
                .map(|vertex| vertex.co)
                .collect::<Vec<_>>(),
            points(&entry["positions"])?,
            "passthrough vertex positions"
        );
    }
    Ok(())
}
