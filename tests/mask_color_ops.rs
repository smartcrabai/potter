#![expect(clippy::float_cmp, reason = "asserting stable scene color settings")]
#![expect(clippy::unwrap_used, reason = "small operation error assertions")]

use std::error::Error;

use potter::{
    error::ErrorCode,
    model::{Id, SceneDoc},
    ops::apply_batch,
};
use serde_json::json;

fn polygon(points: [[f64; 2]; 4]) -> serde_json::Value {
    json!({
        "splines": [{
            "feather": 0.0,
            "points": points.into_iter().map(|position| json!({"position": position, "keyframes": []})).collect::<Vec<_>>()
        }]
    })
}

#[test]
fn mask_update_changes_raster_coverage_and_delete_removes_the_mask() -> Result<(), Box<dyn Error>> {
    let document = SceneDoc::default();
    let created = apply_batch(
        &document,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [{
                "op": "mask.create",
                "id": "subject",
                "splines": polygon([[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]])["splines"].clone()
            }]
        }),
    )?;
    let id = Id::new("subject")?;
    let original = created.doc.masks[&id].rasterize(10, 10, 1.0)?;
    assert_eq!(original.get(5, 5), Some(1.0));

    let updated = apply_batch(
        &created.doc,
        &json!({
            "schema_version": 1,
            "base_revision": created.doc.revision,
            "operations": [{
                "op": "mask.update",
                "id": "subject",
                "set": polygon([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]])
            }]
        }),
    )?;
    let changed = updated.doc.masks[&id].rasterize(10, 10, 1.0)?;
    assert_eq!(changed.get(5, 5), Some(0.0));
    assert_ne!(original.pixels, changed.pixels);

    let deleted = apply_batch(
        &updated.doc,
        &json!({
            "schema_version": 1,
            "base_revision": updated.doc.revision,
            "operations": [{"op": "mask.delete", "id": "subject"}]
        }),
    )?;
    assert!(!deleted.doc.masks.contains_key(&id));
    Ok(())
}

#[test]
fn invalid_mask_update_is_rejected_without_changing_stored_coverage() -> Result<(), Box<dyn Error>>
{
    let document = SceneDoc::default();
    let created = apply_batch(
        &document,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [{
                "op": "mask.create",
                "id": "subject",
                "splines": polygon([[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]])["splines"].clone()
            }]
        }),
    )?;
    let id = Id::new("subject")?;
    let invalid = apply_batch(
        &created.doc,
        &json!({
            "schema_version": 1,
            "base_revision": created.doc.revision,
            "operations": [{
                "op": "mask.update",
                "id": "subject",
                "set": {"splines": [{
                    "feather": -1.0,
                    "points": [
                        {"position": [2.0, 2.0], "keyframes": []},
                        {"position": [8.0, 2.0], "keyframes": []},
                        {"position": [8.0, 8.0], "keyframes": []},
                        {"position": [2.0, 8.0], "keyframes": []}
                    ]
                }]}
            }]
        }),
    );
    assert_eq!(invalid.unwrap_err().code, ErrorCode::InvalidOperation);
    assert_eq!(
        created.doc.masks[&id].rasterize(10, 10, 1.0)?.get(5, 5),
        Some(1.0)
    );
    Ok(())
}

#[test]
fn color_operations_apply_transform_settings_and_validate_gamma_and_curve()
-> Result<(), Box<dyn Error>> {
    let document = SceneDoc::default();
    let updated = apply_batch(
        &document,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [{
                "op": "color.update",
                "target": {"id": "scene_main"},
                "set": {
                    "view_transform": "raw",
                    "exposure": 1.0,
                    "gamma": 2.0,
                    "curve": [[0.0, 0.0], [1.0, 1.0]]
                }
            }]
        }),
    )?;
    let scene = &updated.doc.scenes[&updated.doc.active_scene];
    assert_eq!(
        scene.color_management.view_transform,
        potter::color::ViewTransform::Raw
    );
    assert_eq!(scene.color_management.exposure, 1.0);
    assert_eq!(scene.color_management.gamma, 2.0);
    let transformed = scene.color_management.transform_linear([0.25; 3]);
    assert!((transformed[0] - 0.5_f64.sqrt()).abs() < 1.0e-12);

    let invalid_gamma = apply_batch(
        &updated.doc,
        &json!({
            "schema_version": 1,
            "base_revision": updated.doc.revision,
            "operations": [{
                "op": "color.update",
                "target": {"id": "scene_main"},
                "set": {"gamma": 0.0}
            }]
        }),
    );
    assert_eq!(invalid_gamma.unwrap_err().code, ErrorCode::InvalidOperation);

    let invalid_curve = apply_batch(
        &updated.doc,
        &json!({
            "schema_version": 1,
            "base_revision": updated.doc.revision,
            "operations": [{
                "op": "color.update",
                "target": {"id": "scene_main"},
                "set": {"curve": [[1.0, 0.0], [0.0, 1.0]]}
            }]
        }),
    );
    assert_eq!(invalid_curve.unwrap_err().code, ErrorCode::InvalidOperation);
    Ok(())
}
