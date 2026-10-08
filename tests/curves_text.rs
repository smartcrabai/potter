#![expect(clippy::unwrap_used, reason = "integration tests")]
#![expect(clippy::float_cmp, reason = "geometry assertions use analytic values")]

use potter::{
    eval::{EvaluationContext, Snapshot},
    geom::{curve::evaluate_hair_curves, text::evaluate_text},
    model::{DataBlock, HairCurve, HairCurvesData, Id, Node, SceneDoc, TextObjectData},
    ops::apply_batch,
};
use proptest::prelude::*;
use serde_json::{Value, json};

fn apply(doc: &SceneDoc, operations: &Value) -> SceneDoc {
    apply_batch(
        doc,
        &json!({"schema_version":1,"base_revision":doc.revision,"operations":operations}),
    )
    .unwrap()
    .doc
}

fn evaluated(doc: &SceneDoc) -> Snapshot {
    Snapshot::evaluate(doc, &EvaluationContext::default()).unwrap()
}

#[test]
fn bezier_matches_analytic_cubic_and_order_two_nurbs_matches_polyline() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let bezier_points = json!([
        {"co":[0.0,0.0,0.0],"handle_left":[0.0,0.0,0.0],"handle_right":[1.0,0.0,0.0],"handle_type":"aligned","weight":1.0,"radius":1.0,"tilt":0.0},
        {"co":[0.0,1.0,0.0],"handle_left":[1.0,1.0,0.0],"handle_right":[0.0,1.0,0.0],"handle_type":"aligned","weight":1.0,"radius":1.0,"tilt":0.0}
    ]);
    let line_points = json!([
        {"co":[0.0,0.0,0.0],"weight":1.0,"radius":1.0,"tilt":0.0},
        {"co":[1.0,0.0,0.0],"weight":1.0,"radius":1.0,"tilt":0.0},
        {"co":[1.0,1.0,0.0],"weight":1.0,"radius":1.0,"tilt":0.0}
    ]);
    let doc = apply(
        &base,
        &json!([
            {"op":"curve.create","id":"bezier","name":"Cubic","dimensions":"three_d","splines":[{"type":"bezier","points":bezier_points,"order":3,"cyclic":false,"resolution":2,"use_endpoint":false}]},
            {"op":"curve.create","id":"poly","name":"Polyline","dimensions":"three_d","splines":[{"type":"poly","points":line_points,"order":2,"cyclic":false,"resolution":2,"use_endpoint":true}]},
            {"op":"curve.create","id":"nurbs","name":"NURBS line","dimensions":"three_d","splines":[{"type":"nurbs","points":line_points,"order":2,"cyclic":false,"resolution":2,"use_endpoint":true}]},
            {"op":"surface.create","id":"patch","name":"Patch","points":[[{"co":[0.0,0.0,0.0],"weight":1.0},{"co":[1.0,0.0,0.0],"weight":1.0}],[{"co":[0.0,1.0,0.0],"weight":1.0},{"co":[1.0,1.0,0.0],"weight":1.0}]],"order_u":2,"order_v":2,"resolution":[1,1]}
        ]),
    );
    let snapshot = evaluated(&doc);
    let bezier_id = Id::new("bezier").unwrap();
    let bezier_mesh = &snapshot.meshes[&bezier_id];
    assert_eq!(bezier_mesh.vertices.len(), 3);
    let midpoint = bezier_mesh.vertices[1].co;
    assert!((midpoint.x - 0.75).abs() < 1.0e-6);
    assert!((midpoint.y - 0.5).abs() < 1.0e-6);
    assert!(midpoint.z.abs() < 1.0e-6);

    let poly = &snapshot.meshes[&Id::new("poly").unwrap()];
    let nurbs = &snapshot.meshes[&Id::new("nurbs").unwrap()];
    assert_eq!(poly.vertices.len(), nurbs.vertices.len());
    for (poly_vertex, nurbs_vertex) in poly.vertices.iter().zip(&nurbs.vertices) {
        assert!((poly_vertex.co - nurbs_vertex.co).length() < 1.0e-6);
    }
    let surface = &snapshot.meshes[&Id::new("patch").unwrap()];
    assert_eq!(surface.vertices.len(), 4);
    assert_eq!(surface.faces.len(), 1);
}

#[test]
fn cyclic_bevel_is_a_closed_tube_and_text_bounds_scale_with_size() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let circle = json!([
        {"co":[1.0,0.0,0.0],"handle_left":[1.0,0.0,0.0],"handle_right":[1.0,0.0,0.0],"handle_type":"vector","weight":1.0,"radius":1.0,"tilt":0.0},
        {"co":[0.0,1.0,0.0],"handle_left":[0.0,1.0,0.0],"handle_right":[0.0,1.0,0.0],"handle_type":"vector","weight":1.0,"radius":1.0,"tilt":0.0},
        {"co":[-1.0,0.0,0.0],"handle_left":[-1.0,0.0,0.0],"handle_right":[-1.0,0.0,0.0],"handle_type":"vector","weight":1.0,"radius":1.0,"tilt":0.0},
        {"co":[0.0,-1.0,0.0],"handle_left":[0.0,-1.0,0.0],"handle_right":[0.0,-1.0,0.0],"handle_type":"vector","weight":1.0,"radius":1.0,"tilt":0.0}
    ]);
    let doc = apply(
        &base,
        &json!([
            {"op":"curve.create","id":"tube","name":"Tube","dimensions":"three_d","bevel_depth":0.1,"bevel_resolution":8,"splines":[{"type":"poly","points":circle,"order":2,"cyclic":true,"resolution":1,"use_endpoint":true}]},
            {"op":"text_object.create","id":"text_small","name":"Small","body":"AB","font":"builtin","size":1.0,"align_x":"left","align_y":"baseline","extrude":0.0,"bevel_depth":0.0},
            {"op":"text_object.create","id":"text_large","name":"Large","body":"AB","font":"builtin","size":2.0,"align_x":"left","align_y":"baseline","extrude":0.0,"bevel_depth":0.0}
        ]),
    );
    let snapshot = evaluated(&doc);
    let tube = &snapshot.meshes[&Id::new("tube").unwrap()];
    assert_eq!(tube.vertices.len(), 80);
    assert_eq!(tube.faces.len(), 80);
    assert!(tube.validate().is_ok());
    for edge in &tube.edges {
        let incident_faces = tube
            .faces
            .iter()
            .filter(|face| {
                face.vertices
                    .iter()
                    .copied()
                    .zip(face.vertices.iter().cycle().skip(1).copied())
                    .any(|(first, second)| {
                        (first == edge.vertices[0] && second == edge.vertices[1])
                            || (first == edge.vertices[1] && second == edge.vertices[0])
                    })
            })
            .count();
        assert_eq!(incident_faces, 2);
    }
    let small = snapshot.nodes[&Id::new("text_small").unwrap()]
        .bounds
        .unwrap()
        .size()
        .x;
    let large = snapshot.nodes[&Id::new("text_large").unwrap()]
        .bounds
        .unwrap()
        .size()
        .x;
    assert!((large - 2.0 * small).abs() < 1.0e-12);
}
#[test]
fn curve_editing_and_conversion_report_production_data_loss() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let doc = apply(
        &base,
        &json!([
            {"op":"curve.create","id":"editable","data_id":"editable_data","dimensions":"three_d","splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[2.0,0.0,0.0]}],"order":2,"cyclic":false,"resolution":1,"use_endpoint":true}]},
            {"op":"curve.point_add","target":{"id":"editable"},"spline_index":0,"index":1,"point":{"co":[1.0,1.0,0.0]}},
            {"op":"curve.point_update","target":{"id":"editable"},"spline_index":0,"point_index":1,"set":{"co":[1.0,0.5,0.0]}},
            {"op":"curve.point_delete","target":{"id":"editable"},"spline_index":0,"point_index":0},
            {"op":"curve.update","target":{"id":"editable"},"set":{"bevel_depth":0.1,"bevel_resolution":8}},
            {"op":"curve.convert","target":{"id":"editable"},"to":"mesh"}
        ]),
    );

    assert_eq!(
        doc.data_blocks[&Id::new("editable_data").unwrap()].data_type,
        "mesh"
    );
    let losses = doc.compatibility["losses"].as_array().unwrap();
    assert_eq!(losses[0]["feature_id"], "curve.production_data");
    let mesh = &evaluated(&doc).meshes[&Id::new("editable").unwrap()];
    assert_eq!(mesh.vertices.len(), 40);
    assert_eq!(mesh.faces.len(), 20);
    assert!(mesh.validate().is_ok());
}

#[test]
fn text_update_uses_variable_width_hershey_glyphs() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let doc = apply(
        &base,
        &json!([
            {"op":"text_object.create","id":"label","body":"I","size":1.0},
            {"op":"text_object.update","target":{"id":"label"},"set":{"body":"M","size":2.0,"extrude":0.25,"bevel_depth":0.05}}
        ]),
    );
    let snapshot = evaluated(&doc);
    let label_id = Id::new("label").unwrap();
    let bounds = snapshot.nodes[&label_id].bounds.unwrap();
    let mesh = &snapshot.meshes[&label_id];
    assert!(bounds.size().x > 1.0);
    assert!(!mesh.faces.is_empty(), "text mesh has no faces");
    assert!(
        mesh.vertices
            .iter()
            .all(|vertex| vertex.co.z == 0.0 || vertex.co.z == 0.25)
    );
}
#[test]
fn external_text_font_reference_is_preserved_with_builtin_geometry_fallback() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let external = apply(
        &base,
        &json!([{
            "op": "text_object.create",
            "id": "external_font",
            "body": "A",
            "font": "custom"
        }]),
    );
    let builtin = apply(
        &base,
        &json!([{
            "op": "text_object.create",
            "id": "external_font",
            "body": "A"
        }]),
    );
    let text = external.data_blocks[&Id::new("external_font_text").unwrap()]
        .text
        .as_ref()
        .unwrap();
    assert_eq!(text.font, "custom");
    let text_node_id = Id::new("external_font").unwrap();
    let external_snapshot = evaluated(&external);
    let builtin_snapshot = evaluated(&builtin);
    assert_eq!(
        external_snapshot.meshes[&text_node_id],
        builtin_snapshot.meshes[&text_node_id]
    );
}

#[test]
fn built_in_hershey_font_covers_printable_ascii() {
    let body = (b' '..=b'~').map(char::from).collect::<String>();
    let text = TextObjectData {
        body,
        ..TextObjectData::default()
    };
    let mesh = evaluate_text(&text).unwrap();
    assert!(!mesh.faces.is_empty(), "font generated no faces");
    assert!(mesh.validate().is_ok());
}

#[test]
fn hair_curves_evaluate_to_loose_line_edges() {
    let hair = HairCurvesData {
        curves: vec![
            HairCurve {
                points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                radius: 0.02,
            },
            HairCurve {
                points: vec![[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
                radius: 0.01,
            },
        ],
    };
    let mesh = evaluate_hair_curves(&hair).unwrap();
    assert_eq!(mesh.vertices.len(), 5);
    assert_eq!(mesh.edges.len(), 3);
    assert!(mesh.faces.is_empty(), "{:?}", mesh.faces);
    assert!(mesh.validate().is_ok());
}
#[test]
fn snapshot_evaluates_hair_curves_as_lines() {
    let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let node_id = Id::new("hair").unwrap();
    let data_id = Id::new("hair_data").unwrap();
    let hair_curves = HairCurvesData {
        curves: vec![HairCurve {
            points: vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            radius: 0.02,
        }],
    };
    doc.nodes.insert(
        node_id.clone(),
        Node {
            kind: "hair_curves".to_owned(),
            data: Some(data_id.clone()),
            ..Node::default()
        },
    );
    doc.collections
        .get_mut(&Id::new("collection_root").unwrap())
        .unwrap()
        .objects
        .push(node_id.clone());
    doc.data_blocks.insert(
        data_id,
        DataBlock {
            data_type: "hair_curves".to_owned(),
            mesh: None,
            hair_curves: Some(hair_curves),
            ..DataBlock::default()
        },
    );

    let snapshot = evaluated(&doc);
    let mesh = &snapshot.meshes[&node_id];
    assert_eq!(mesh.vertices.len(), 2);
    assert_eq!(mesh.edges.len(), 1);
    assert!(mesh.faces.is_empty(), "{:?}", mesh.faces);
}

#[test]
fn unsupported_curve_taper_reports_feature_id() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let error = apply_batch(
        &base,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [{
                "op": "curve.create",
                "id": "tapered",
                "dimensions": "three_d",
                "taper": "profile",
                "splines": [{
                    "type": "poly",
                    "points": [{"co": [0.0, 0.0, 0.0]}, {"co": [1.0, 0.0, 0.0]}],
                    "order": 2,
                    "resolution": 1
                }]
            }]
        }),
    )
    .unwrap_err();
    assert_eq!(error.code, potter::error::ErrorCode::UnsupportedFeature);
    assert_eq!(error.details["feature_id"], "curve.taper_object");
}
#[test]
fn unsupported_multiple_fill_contours_report_feature_id() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let contour = json!([
        {"co":[0.0,0.0,0.0]},
        {"co":[1.0,0.0,0.0]},
        {"co":[0.0,1.0,0.0]}
    ]);
    let error = apply_batch(
        &base,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [{
                "op": "curve.create",
                "id": "multi_fill",
                "dimensions": "two_d",
                "fill_mode": "both",
                "splines": [
                    {"type":"poly","points":contour,"order":2,"cyclic":true,"resolution":1},
                    {"type":"poly","points":contour,"order":2,"cyclic":true,"resolution":1}
                ]
            }]
        }),
    )
    .unwrap_err();
    assert_eq!(error.code, potter::error::ErrorCode::UnsupportedFeature);
    assert_eq!(error.details["feature_id"], "curve.fill.multiple_contours");
}

proptest! {
    #[test]
    fn cubic_endpoints_are_preserved(p0 in prop::array::uniform3(-10.0_f64..10.0), p3 in prop::array::uniform3(-10.0_f64..10.0)) {
        let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let points = json!([
            {"co":p0,"handle_left":p0,"handle_right":p0,"handle_type":"free","weight":1.0,"radius":1.0,"tilt":0.0},
            {"co":p3,"handle_left":p3,"handle_right":p3,"handle_type":"free","weight":1.0,"radius":1.0,"tilt":0.0}
        ]);
        let doc = apply(&base, &json!([{"op":"curve.create","id":"cubic","dimensions":"three_d","splines":[{"type":"bezier","points":points,"order":3,"resolution":4,"cyclic":false,"use_endpoint":false}]}]));
        let mesh = &evaluated(&doc).meshes[&Id::new("cubic").unwrap()];
        prop_assert!((mesh.vertices.first().unwrap().co.to_array() == p0));
        prop_assert!((mesh.vertices.last().unwrap().co.to_array() == p3));
    }
}
