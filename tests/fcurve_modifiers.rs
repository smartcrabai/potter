#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/mod.rs"]
mod common;
#[path = "common/process.rs"]
mod process;

use std::{fs, path::Path};

use potter_core::{
    eval::{EvaluationContext, animation::animated_transform},
    exchange::{
        ExportOptions,
        blend::{export_blend, import_blend},
    },
    ops,
    store::Project,
};
use serde_json::{Value, json};

use common::TestResult;

#[test]
fn fcurve_modifiers_match_blender_oracle_and_roundtrip() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        eprintln!("skipping F-Curve modifier oracle: Blender is unavailable");
        return Ok(());
    };
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fcurve_modifiers");
    let expected: Value = serde_json::from_slice(&fs::read(fixture.join("expected.json"))?)?;
    assert_eq!(expected["blender_version"], "5.2.2 LTS");
    // Blender stores and evaluates F-Curve inputs and modifier outputs as f32; 2e-6 is about four ULPs at the largest sample value.
    let tolerance = expected["tolerance"]
        .as_f64()
        .ok_or("oracle tolerance is missing")?;
    let source = fixture.join("fcurve_modifiers.blend");
    let imported = import_blend(&source, Some(&blender))?;
    assert!(
        imported
            .losses
            .iter()
            .all(|loss| loss.feature_id != "blender.animation.fcurve_modifier"),
        "implemented F-Curve modifier types must not be reported as import losses: {:?}",
        imported.losses
    );

    for case in expected["cases"]
        .as_array()
        .ok_or("oracle cases are missing")?
    {
        let case_name = case["name"].as_str().ok_or("oracle case name is missing")?;
        let node = imported
            .doc
            .nodes
            .values()
            .find(|node| node.name == case_name)
            .ok_or_else(|| format!("imported Blender object `{case_name}` is missing"))?;
        let action_id = node
            .action
            .as_ref()
            .ok_or_else(|| format!("Blender object `{case_name}` has no Action"))?;
        let action = imported
            .doc
            .actions
            .get(action_id)
            .ok_or_else(|| format!("Action for `{case_name}` is missing"))?;
        let curve = action
            .fcurves
            .iter()
            .find(|curve| curve.path == "transform.translation" && curve.index == 0)
            .ok_or_else(|| format!("location F-Curve for `{case_name}` is missing"))?;
        assert!(
            !curve.modifiers.is_empty(),
            "{case_name} lost its modifiers"
        );
        for sample in case["samples"]
            .as_array()
            .ok_or("oracle samples are missing")?
        {
            let frame = sample["frame"].as_f64().ok_or("oracle frame is missing")?;
            let expected_value = sample["value"].as_f64().ok_or("oracle value is missing")?;
            let actual = animated_transform(node, &imported.doc, frame)?.translation[0];
            assert!(
                (actual - expected_value).abs() <= tolerance,
                "{case_name} at frame {frame}: expected {expected_value:.9}, got {actual:.9}, tolerance {tolerance}; modifiers={:?}",
                curve.modifiers
            );
        }
    }
    let default_node_id = imported
        .doc
        .nodes
        .iter()
        .find(|(_, node)| node.name == "generator_factorised_defaults")
        .map(|(id, _)| id)
        .ok_or("factorized-default oracle object is missing")?;
    let updated = ops::apply_batch(
        &imported.doc,
        &json!({
            "schema_version": 1,
            "base_revision": imported.doc.revision,
            "operations": [{
                "op": "fcurve.update",
                "target": {"id": default_node_id},
                "path": "transform.translation",
                "index": 0,
                "set": {"modifiers": [{
                    "type": "GENERATOR",
                    "properties": {
                        "mode": "POLYNOMIAL_FACTORISED",
                        "poly_order": 2
                    }
                }]}
            }]
        }),
    )?
    .doc;
    let expected_default = expected["cases"]
        .as_array()
        .and_then(|cases| {
            cases
                .iter()
                .find(|case| case["name"] == "generator_factorised_defaults")
        })
        .and_then(|case| case["samples"].as_array())
        .and_then(|samples| {
            samples
                .iter()
                .find(|sample| sample["frame"].as_f64() == Some(2.25))
        })
        .and_then(|sample| sample["value"].as_f64())
        .ok_or("factorized-default oracle sample is missing")?;
    let updated_node = updated
        .nodes
        .get(default_node_id)
        .ok_or("updated factorized-default object is missing")?;
    let actual_default = animated_transform(updated_node, &updated, 2.25)?.translation[0];
    assert!(
        (actual_default - expected_default).abs() <= tolerance,
        "factorized Generator defaults at frame 2.25: expected {expected_default}, got {actual_default}"
    );

    let temporary = tempfile::tempdir()?;
    let project = Project::init(temporary.path().join("project"))?;
    let exported = temporary.path().join("roundtrip.blend");
    export_blend(
        &imported.doc,
        &project,
        &exported,
        &ExportOptions {
            allow_lossy: false,
            pack: false,
            blender: Some(&blender),
            context: &EvaluationContext::default(),
        },
    )?;
    let roundtrip = import_blend(&exported, Some(&blender))?;
    for action in imported.doc.actions.values() {
        let restored = roundtrip
            .doc
            .actions
            .values()
            .find(|candidate| candidate.name == action.name)
            .ok_or_else(|| format!("round-tripped Blender Action `{}` is missing", action.name))?;
        for curve in &action.fcurves {
            let restored_curve = restored
                .fcurves
                .iter()
                .find(|candidate| candidate.path == curve.path && candidate.index == curve.index)
                .ok_or_else(|| format!("round-tripped F-Curve `{}` is missing", curve.path))?;
            assert_eq!(restored_curve.modifiers.len(), curve.modifiers.len());
            for (modifier_index, (restored, original)) in restored_curve
                .modifiers
                .iter()
                .zip(&curve.modifiers)
                .enumerate()
            {
                assert_eq!(restored.modifier_type, original.modifier_type);
                for (name, value) in &original.properties {
                    assert_eq!(
                        restored.properties.get(name),
                        Some(value),
                        "round-tripped modifier property `{name}` changed on {}[{}] modifier {modifier_index}",
                        curve.path,
                        curve.index
                    );
                }
            }
        }
    }
    Ok(())
}
