#![expect(clippy::unwrap_used, reason = "integration tests")]

use std::{env, error::Error, path::PathBuf, process::Command};

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/blender_script.rs"]
mod blender_script;
#[path = "common/process.rs"]
mod process;

use blender_file::blender_executable;
use blender_script::run_blender_script;
use potter_core::{
    catalog::feature_catalog,
    error::ErrorCode,
    eval::{EvaluationContext, Snapshot},
    exchange::{
        ExportOptions,
        blend::{export_blend, import_blend},
    },
    model::{CurveData, CurveFillMode, CurveFillRule, CurveFillSolver, DataBlock, Id, SceneDoc},
    ops::apply_batch,
    schema::schema,
    store::Project,
};
use serde_json::json;
use tempfile::tempdir;

fn filled_curve_doc() -> SceneDoc {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    apply_batch(
        &base,
        &json!({
            "schema_version":1,
            "base_revision":base.revision,
            "operations":[{
                "op":"curve.create",
                "id":"fill",
                "data_id":"fill_data",
                "name":"Filled curve",
                "dimensions":"two_d",
                "fill_solver":"sweep_line",
                "fill_rule":"even_odd",
                "fill_mode":"both",
                "splines":[{
                    "type":"poly",
                    "points":[
                        {"co":[0.0,0.0,0.0]},
                        {"co":[2.0,0.0,0.0]},
                        {"co":[2.0,2.0,0.0]},
                        {"co":[0.0,2.0,0.0]}
                    ],
                    "order":2,
                    "cyclic":true,
                    "resolution":1,
                    "use_endpoint":true
                }]
            }]
        }),
    )
    .unwrap()
    .doc
}

fn with_fill_settings(solver: CurveFillSolver, rule: CurveFillRule) -> SceneDoc {
    let doc = filled_curve_doc();
    apply_batch(
        &doc,
        &json!({
            "schema_version":1,
            "base_revision":doc.revision,
            "operations":[{
                "op":"curve.update",
                "target":{"id":"fill"},
                "set":{"fill_solver":solver,"fill_rule":rule}
            }]
        }),
    )
    .unwrap()
    .doc
}

#[test]
fn curve_fill_settings_default_serialize_and_appear_in_schema() {
    let defaults = serde_json::to_value(CurveData::default()).unwrap();
    assert_eq!(defaults["fill_solver"], "sweep_line");
    assert_eq!(defaults["fill_rule"], "even_odd");

    let created = filled_curve_doc();
    let created_curve = curve_named(&created, "Filled curve");
    assert_eq!(created_curve.fill_solver, CurveFillSolver::SweepLine);
    assert_eq!(created_curve.fill_rule, CurveFillRule::EvenOdd);

    let legacy: CurveData = serde_json::from_value(json!({})).unwrap();
    assert_eq!(legacy.fill_solver, CurveFillSolver::SweepLine);
    assert_eq!(legacy.fill_rule, CurveFillRule::EvenOdd);
    assert!(serde_json::from_value::<CurveData>(json!({"fill_solver":"unknown"})).is_err());

    let scene = schema("scene", None).unwrap();
    let curve = &scene["$defs"]["curve_data"];
    assert_eq!(
        curve["properties"]["fill_solver"]["enum"],
        json!(["sweep_line", "cdt"])
    );
    assert_eq!(
        curve["properties"]["fill_rule"]["enum"],
        json!(["even_odd", "nonzero"])
    );
    assert!(
        curve["required"]
            .as_array()
            .unwrap()
            .iter()
            .all(|field| { field != "fill_solver" && field != "fill_rule" })
    );

    let catalog = feature_catalog();
    let feature = catalog["features"]
        .as_array()
        .unwrap()
        .iter()
        .find(|feature| feature["feature_id"] == "curve.fill.multiple_contours")
        .unwrap();
    assert_eq!(feature["status"], "supported");
    assert!(
        feature["name"]
            .as_str()
            .unwrap()
            .contains("CDT NONZERO is rejected")
    );
}

#[test]
fn cdt_nonzero_fill_is_rejected_as_unsupported() {
    let doc = with_fill_settings(CurveFillSolver::Cdt, CurveFillRule::Nonzero);
    let Err(error) = Snapshot::evaluate(&doc, &EvaluationContext::default()) else {
        panic!("CDT NONZERO fill must be refused");
    };
    assert_eq!(error.code, ErrorCode::UnsupportedFeature);
    assert_eq!(error.details["feature_id"], "curve.fill.multiple_contours");
    assert_eq!(error.details["fill_solver"], "cdt");
    assert_eq!(error.details["fill_rule"], "nonzero");
}

#[test]
fn supported_solver_rule_combinations_still_evaluate() {
    for (solver, rule) in [
        (CurveFillSolver::SweepLine, CurveFillRule::EvenOdd),
        (CurveFillSolver::SweepLine, CurveFillRule::Nonzero),
        (CurveFillSolver::Cdt, CurveFillRule::EvenOdd),
    ] {
        let doc = with_fill_settings(solver, rule);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
        assert!(
            !snapshot.meshes[&Id::new("fill").unwrap()].faces.is_empty(),
            "{solver:?} + {rule:?} should produce a fill"
        );
    }
}

#[test]
fn cdt_nonzero_without_fill_mode_is_inert() {
    let mut doc = with_fill_settings(CurveFillSolver::Cdt, CurveFillRule::Nonzero);
    doc.data_blocks
        .get_mut(&Id::new("fill_data").unwrap())
        .and_then(|data| data.curve.as_mut())
        .unwrap()
        .fill_mode = CurveFillMode::None;
    Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap();
}

#[test]
fn cdt_nonzero_taper_data_without_a_node_is_refused() {
    let mut doc = filled_curve_doc();
    let taper_id = Id::new("taper_data").unwrap();
    let taper_curve = CurveData {
        fill_mode: CurveFillMode::Both,
        fill_solver: CurveFillSolver::Cdt,
        fill_rule: CurveFillRule::Nonzero,
        ..CurveData::default()
    };
    doc.data_blocks.insert(
        taper_id.clone(),
        DataBlock {
            data_type: "curve".to_owned(),
            curve: Some(taper_curve),
            ..DataBlock::default()
        },
    );
    doc.data_blocks
        .get_mut(&Id::new("fill_data").unwrap())
        .and_then(|data| data.curve.as_mut())
        .unwrap()
        .taper = Some(taper_id);

    let Err(error) = Snapshot::evaluate(&doc, &EvaluationContext::default()) else {
        panic!("CDT NONZERO taper fill must be refused");
    };
    assert_eq!(error.code, ErrorCode::UnsupportedFeature);
    assert_eq!(error.details["feature_id"], "curve.fill.multiple_contours");
}

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const BLENDER_FIXTURE: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"curve fill bridge test requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

def add_curve(name, fill_rule):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "2D"
    data.fill_mode = "BOTH"
    data.fill_solver = "CDT"
    data.fill_rule = fill_rule
    spline = data.splines.new("POLY")
    spline.points.add(3)
    for point, co in zip(spline.points, ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))):
        point.co = (*co, 1.0)
    spline.use_cyclic_u = True
    scene.collection.objects.link(bpy.data.objects.new(name, data))

add_curve("ActiveFill", "EVEN_ODD")
add_curve("NonzeroFill", "NONZERO")
bpy.ops.wm.save_as_mainfile(filepath=str(root / "source.blend"))
"#;

const BLENDER_READBACK: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"curve fill bridge test requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.open_mainfile(filepath=str(root / "roundtrip.blend"))
for name, rule, mode in (
        ("ActiveFill", "EVEN_ODD", "BOTH"),
        ("NonzeroFill", "NONZERO", "NONE")):
    obj = bpy.data.objects.get(name)
    assert obj is not None and obj.type == "CURVE", name
    assert obj.data.fill_solver == "CDT", (name, obj.data.fill_solver)
    assert obj.data.fill_rule == rule, (name, obj.data.fill_rule)
    assert obj.data.fill_mode == mode, (name, obj.data.fill_mode)
"#;

fn blender_522() -> TestResult<Option<PathBuf>> {
    let require_blender =
        env::var_os("POTTER_REQUIRE_BLENDER").is_some_and(|value| !value.is_empty());
    let Some(blender) = blender_executable() else {
        if require_blender {
            return Err(std::io::Error::other(
                "POTTER_REQUIRE_BLENDER is set but Blender was not found",
            )
            .into());
        }
        eprintln!("skipping curve fill bridge test: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(&blender);
    command.arg("--version");
    let output = process::run_guarded(command)?;
    if !output.status.success() {
        return Err(std::io::Error::other("Blender --version failed").into());
    }
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if version.starts_with("Blender 5.2.2") {
        return Ok(Some(blender));
    }
    if require_blender {
        return Err(
            std::io::Error::other(format!(
                "POTTER_REQUIRE_BLENDER is set but curve fill bridge test requires Blender 5.2.2, found `{version}`"
            ))
            .into(),
        );
    }
    eprintln!("skipping curve fill bridge test: found `{version}`, need Blender 5.2.2");
    Ok(None)
}

fn curve_named<'a>(doc: &'a SceneDoc, name: &str) -> &'a CurveData {
    let node = doc.nodes.values().find(|node| node.name == name).unwrap();
    let data_id = node.data.as_ref().unwrap();
    doc.data_blocks
        .get(data_id)
        .and_then(|data| data.curve.as_ref())
        .unwrap()
}

fn curve_data_id(doc: &SceneDoc, name: &str) -> Id {
    doc.nodes
        .values()
        .find(|node| node.name == name)
        .and_then(|node| node.data.clone())
        .unwrap()
}

#[test]
fn blender_fill_solver_and_rule_import_dump_round_trip() -> TestResult {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_curve_fill_settings.py",
        BLENDER_FIXTURE,
        root,
        "Blender curve fill fixture failed",
    )?;

    let imported = import_blend(&root.join("source.blend"), Some(blender.as_path()))?;
    let active_fill = curve_named(&imported.doc, "ActiveFill");
    assert_eq!(active_fill.fill_solver, CurveFillSolver::Cdt);
    assert_eq!(active_fill.fill_rule, CurveFillRule::EvenOdd);
    assert_eq!(active_fill.fill_mode, CurveFillMode::Both);
    let nonzero_fill = curve_named(&imported.doc, "NonzeroFill");
    assert_eq!(nonzero_fill.fill_solver, CurveFillSolver::Cdt);
    assert_eq!(nonzero_fill.fill_rule, CurveFillRule::Nonzero);
    assert_eq!(nonzero_fill.fill_mode, CurveFillMode::Both);

    // Preserve CDT/NONZERO, but disable its fill before export because active use is rejected.
    let mut doc = imported.doc;
    let nonzero_data_id = curve_data_id(&doc, "NonzeroFill");
    doc.data_blocks
        .get_mut(&nonzero_data_id)
        .and_then(|data| data.curve.as_mut())
        .unwrap()
        .fill_mode = CurveFillMode::None;
    let project = Project::init(root.join("project"))?;
    let roundtrip_blend = root.join("roundtrip.blend");
    let context = EvaluationContext::default();
    export_blend(
        &doc,
        &project,
        &roundtrip_blend,
        &ExportOptions {
            allow_lossy: false,
            pack: false,
            blender: Some(blender.as_path()),
            context: &context,
        },
    )?;
    run_blender_script(
        &blender,
        "read_curve_fill_settings.py",
        BLENDER_READBACK,
        root,
        "Blender curve fill readback failed",
    )?;

    let roundtripped = import_blend(&roundtrip_blend, Some(blender.as_path()))?;
    assert_eq!(
        curve_named(&roundtripped.doc, "ActiveFill").fill_solver,
        CurveFillSolver::Cdt
    );
    assert_eq!(
        curve_named(&roundtripped.doc, "ActiveFill").fill_rule,
        CurveFillRule::EvenOdd
    );
    assert_eq!(
        curve_named(&roundtripped.doc, "NonzeroFill").fill_solver,
        CurveFillSolver::Cdt
    );
    assert_eq!(
        curve_named(&roundtripped.doc, "NonzeroFill").fill_rule,
        CurveFillRule::Nonzero
    );
    assert_eq!(
        curve_named(&roundtripped.doc, "NonzeroFill").fill_mode,
        CurveFillMode::None
    );
    Ok(())
}
