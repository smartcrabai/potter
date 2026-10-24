//! Blender 5.2.2 round-trip coverage for curve exchange edits.
//!
//! When a taper Data-Block has no node, `blend_bridge.py` creates a marked `_Potter Taper Proxy`
//! object (`taper_objects_by_data`) and excludes marked proxies from the imported `objects` dump
//! (`object_dump`). This test checks the Blender-side proxy and that it never becomes a Potter node.

use std::{env, error::Error, path::PathBuf, process::Command};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/blender_script.rs"]
mod blender_script;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use blender_script::run_blender_script;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    exchange::{
        ExportOptions,
        blend::{export_blend, import_blend},
    },
    model::{
        CurveData, CurveDimensions, CurveFillMode, CurveFillRule, CurveFillSolver,
        CurveTaperRadiusMode, Id, SceneDoc,
    },
    ops,
    store::Project,
};
use serde_json::json;
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";
const TAPER_NODE_ID: &str = "taper_profile";
const TAPER_DATA_ID: &str = "taper_profile_data";
const HOST_NODE_ID: &str = "tapered_host";
const HOST_DATA_ID: &str = "tapered_host_data";
const HOST_NAME: &str = "Tapered Host";
const FILLED_NODE_ID: &str = "filled_profile";
const FILLED_NAME: &str = "Filled Profile";
const SPLINE_RESOLUTIONS: [u32; 2] = [3, 9];

const BLENDER_READBACK: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"curve exchange round-trip requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.open_mainfile(filepath=str(root / "curves.blend"))
host = bpy.data.objects.get("Tapered Host")
assert host is not None and host.type == "CURVE", "tapered host was not exported"
assert host.data.taper_object is not None, "taper object reference was not exported"
proxy = host.data.taper_object
assert bpy.context.scene.collection.objects.get(proxy.name) is proxy, "taper proxy is not linked to the scene"
assert proxy.get("_potter.taper_proxy") is True, proxy.name
assert proxy.name.startswith("_Potter Taper Proxy "), proxy.name
assert host.data.taper_radius_mode == "MULTIPLY", host.data.taper_radius_mode
assert host.data.use_map_taper is True
filled = bpy.data.objects.get("Filled Profile")
assert filled is not None and filled.type == "CURVE", "filled curve was not exported"
assert filled.data.dimensions == "2D", filled.data.dimensions
assert filled.data.fill_mode == "FRONT", filled.data.fill_mode
assert filled.data.resolution_u == 6, filled.data.resolution_u
assert filled.data.fill_solver == "SWEEP_LINE", filled.data.fill_solver
assert filled.data.fill_rule == "NONZERO", filled.data.fill_rule
resolutions = [spline.resolution_u for spline in filled.data.splines]
assert resolutions == [3, 9], resolutions
"#;

const BLENDER_NONCURVE_TAPER: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"non-curve taper test requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

mesh_data = bpy.data.meshes.new("Mesh Taper Data")
mesh_data.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
mesh_taper = bpy.data.objects.new("Mesh Taper", mesh_data)
scene.collection.objects.link(mesh_taper)

def make_curve(name):
    data = bpy.data.curves.new(name + " Data", "CURVE")
    data.dimensions = "3D"
    data.bevel_depth = 0.2
    data.bevel_resolution = 2
    data.resolution_u = 1
    spline = data.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
    spline.points[1].co = (2.0, 0.0, 0.0, 1.0)
    spline.resolution_u = 1
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    return obj

host = make_curve("Noncurve Taper")
host.data.taper_object = mesh_taper
baseline = make_curve("No Taper Baseline")
bpy.context.view_layer.update()

def sample(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return (
            [tuple(float(value) for value in vertex.co) for vertex in mesh.vertices],
            [tuple(int(index) for index in polygon.vertices) for polygon in mesh.polygons],
        )
    finally:
        evaluated.to_mesh_clear()

assert sample(host) == sample(baseline), "Blender non-curve taper must use factor 1.0"
bpy.ops.wm.save_as_mainfile(filepath=str(root / "noncurve_taper.blend"))
"#;

fn require_blender() -> bool {
    env::var_os("POTTER_REQUIRE_BLENDER").is_some_and(|value| !value.is_empty())
}

fn blender_522() -> TestResult<Option<PathBuf>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping curve exchange round-trip: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(&blender);
    command.arg("--version");
    let output = process::run_guarded(command)?;
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if version.starts_with("Blender 5.2.2") {
        return Ok(Some(blender));
    }
    assert!(
        !require_blender(),
        "POTTER_REQUIRE_BLENDER is set but curve exchange round-trip requires Blender 5.2.2, found `{version}`"
    );
    eprintln!("skipping curve exchange round-trip: found `{version}`, need Blender 5.2.2");
    Ok(None)
}

fn curve_doc() -> TestResult<SceneDoc> {
    let doc = SceneDoc::new(SCENE_ID.to_owned());
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": [
            {
                "op": "curve.create",
                "id": TAPER_NODE_ID,
                "data_id": TAPER_DATA_ID,
                "name": "Taper Profile",
                "dimensions": "two_d",
                "splines": [{
                    "type": "poly",
                    "points": [
                        {"co": [0.0, 0.2, 0.0]},
                        {"co": [0.5, 0.55, 0.0]},
                        {"co": [1.0, 1.0, 0.0]}
                    ],
                    "order": 2,
                    "resolution": 4,
                    "use_endpoint": true
                }]
            },
            {
                "op": "curve.create",
                "id": HOST_NODE_ID,
                "data_id": HOST_DATA_ID,
                "name": HOST_NAME,
                "dimensions": "three_d",
                "bevel_depth": 0.15,
                "bevel_resolution": 2,
                "taper": TAPER_DATA_ID,
                "taper_radius_mode": "multiply",
                "use_map_taper": true,
                "splines": [{
                    "type": "poly",
                    "points": [
                        {"co": [0.0, 0.0, 0.0]},
                        {"co": [1.0, 0.0, 0.0]},
                        {"co": [3.0, 1.0, 0.0]}
                    ],
                    "order": 2,
                    "resolution": 5,
                    "use_endpoint": true
                }]
            },
            {
                "op": "curve.create",
                "id": FILLED_NODE_ID,
                "name": FILLED_NAME,
                "dimensions": "two_d",
                "resolution_u": 6,
                "fill_mode": "front",
                "fill_solver": "sweep_line",
                "fill_rule": "nonzero",
                "splines": [
                    {
                        "type": "poly",
                        "points": [
                            {"co": [0.0, 0.0, 0.0]},
                            {"co": [3.0, 0.0, 0.0]},
                            {"co": [3.0, 3.0, 0.0]},
                            {"co": [0.0, 3.0, 0.0]}
                        ],
                        "order": 2,
                        "cyclic": true,
                        "resolution": SPLINE_RESOLUTIONS[0],
                        "use_endpoint": true
                    },
                    {
                        "type": "poly",
                        "points": [
                            {"co": [5.0, 0.0, 0.0]},
                            {"co": [7.0, 0.0, 0.0]},
                            {"co": [7.0, 2.0, 0.0]},
                            {"co": [5.0, 2.0, 0.0]}
                        ],
                        "order": 2,
                        "cyclic": true,
                        "resolution": SPLINE_RESOLUTIONS[1],
                        "use_endpoint": true
                    }
                ]
            }
        ]
    });
    let mut doc = ops::apply_batch(&doc, &batch)?.doc;
    let taper_node_id = Id::new(TAPER_NODE_ID)?;
    doc.nodes
        .remove(&taper_node_id)
        .ok_or("taper proxy source node was not created")?;
    let root_collection = doc.scenes[&doc.active_scene].root_collection.clone();
    doc.collections
        .get_mut(&root_collection)
        .ok_or("scene root collection is missing")?
        .objects
        .retain(|id| id != &taper_node_id);
    Ok(doc)
}

fn curve_for_node<'a>(doc: &'a SceneDoc, name: &str) -> TestResult<&'a CurveData> {
    let node = doc
        .nodes
        .values()
        .find(|node| node.name == name)
        .ok_or_else(|| format!("curve node `{name}` is missing"))?;
    let data_id = node
        .data
        .as_ref()
        .ok_or_else(|| format!("curve node `{name}` has no data-block"))?;
    doc.data_blocks
        .get(data_id)
        .and_then(|data| data.curve.as_ref())
        .ok_or_else(|| format!("curve node `{name}` has no curve data").into())
}

#[test]
fn curve_exchange_edits_round_trip_without_importing_taper_proxy_node() -> TestResult {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let project = Project::init(root.join("project"))?;
    let doc = curve_doc()?;

    let host_curve = curve_for_node(&doc, HOST_NAME)?;
    let taper_id = host_curve
        .taper
        .as_ref()
        .ok_or("tapered curve has no taper reference")?;
    assert_eq!(taper_id.as_str(), TAPER_DATA_ID);
    assert!(
        doc.data_blocks
            .get(taper_id)
            .and_then(|data| data.curve.as_ref())
            .is_some()
    );
    assert!(
        !doc.nodes
            .values()
            .any(|node| node.data.as_ref() == Some(taper_id))
    );

    let blend = root.join("curves.blend");
    let context = EvaluationContext::default();
    export_blend(
        &doc,
        &project,
        &blend,
        &ExportOptions {
            allow_lossy: false,
            pack: false,
            blender: Some(blender.as_path()),
            context: &context,
        },
    )?;
    run_blender_script(
        &blender,
        "read_curve_exchange.py",
        BLENDER_READBACK,
        root,
        "Blender curve exchange readback failed",
    )?;

    let imported = import_blend(&blend, Some(blender.as_path()))?;
    let imported_host = curve_for_node(&imported.doc, HOST_NAME)?;
    let imported_taper_id = imported_host
        .taper
        .as_ref()
        .ok_or("imported tapered curve has no taper reference")?;
    assert_eq!(imported_taper_id.as_str(), TAPER_DATA_ID);
    assert!(
        imported
            .doc
            .data_blocks
            .get(imported_taper_id)
            .and_then(|data| data.curve.as_ref())
            .is_some(),
        "imported taper reference does not resolve to curve data"
    );
    assert_eq!(imported_host.dimensions, CurveDimensions::ThreeD);
    assert_eq!(
        imported_host.taper_radius_mode,
        CurveTaperRadiusMode::Multiply
    );
    assert!(imported_host.use_map_taper);

    let imported_fill = curve_for_node(&imported.doc, FILLED_NAME)?;
    assert_eq!(imported_fill.dimensions, CurveDimensions::TwoD);
    assert_eq!(imported_fill.resolution_u, 6);
    assert_eq!(imported_fill.fill_mode, CurveFillMode::Front);
    assert_eq!(imported_fill.fill_solver, CurveFillSolver::SweepLine);
    assert_eq!(imported_fill.fill_rule, CurveFillRule::Nonzero);
    assert_eq!(
        imported_fill
            .splines
            .iter()
            .map(|spline| spline.resolution)
            .collect::<Vec<_>>(),
        SPLINE_RESOLUTIONS
    );

    assert_eq!(
        imported.doc.nodes.len(),
        2,
        "bridge-only taper proxy became a Potter node"
    );
    assert!(!imported.doc.nodes.values().any(|node| {
        node.name.starts_with("_Potter Taper Proxy")
            || node.data.as_ref() == Some(imported_taper_id)
    }));
    Ok(())
}

#[test]
fn non_curve_taper_blend_import_uses_blenders_unit_factor() -> TestResult {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "build_noncurve_taper.py",
        BLENDER_NONCURVE_TAPER,
        root,
        "Blender non-curve taper oracle generation failed",
    )?;
    let imported = import_blend(&root.join("noncurve_taper.blend"), Some(blender.as_path()))?;
    assert!(
        curve_for_node(&imported.doc, "Noncurve Taper")?
            .taper
            .is_none()
    );

    let node_id = |name: &str| -> TestResult<Id> {
        imported
            .doc
            .nodes
            .iter()
            .find(|(_, node)| node.name == name)
            .map(|(id, _)| id.clone())
            .ok_or_else(|| format!("curve node `{name}` is missing").into())
    };
    let host_id = node_id("Noncurve Taper")?;
    let baseline_id = node_id("No Taper Baseline")?;
    let snapshot = Snapshot::evaluate(&imported.doc, &EvaluationContext::default())?;
    let host = &snapshot.meshes[&host_id];
    let baseline = &snapshot.meshes[&baseline_id];
    assert_eq!(
        host.vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>(),
        baseline
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>()
    );
    assert_eq!(host.faces, baseline.faces);
    Ok(())
}
