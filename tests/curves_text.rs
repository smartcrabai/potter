#![expect(clippy::unwrap_used, reason = "integration tests")]
#![expect(clippy::float_cmp, reason = "geometry assertions use analytic values")]

use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::{MeshError, curve::evaluate_hair_curves, text::evaluate_text},
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
fn nurbs_bezier_knots_evaluate_and_custom_knots_are_unsupported() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let points = json!([
        {"co":[0.0,0.0,0.0]},
        {"co":[1.0,0.0,0.0]},
        {"co":[2.0,0.0,0.0]},
        {"co":[3.0,0.0,0.0]},
        {"co":[4.0,0.0,0.0]},
        {"co":[5.0,0.0,0.0]},
        {"co":[6.0,0.0,0.0]}
    ]);
    let cyclic_points = json!([
        {"co":[1.0,0.0,0.0]},
        {"co":[0.5,0.866_025_403_784_438_6,0.0]},
        {"co":[-0.5,0.866_025_403_784_438_6,0.0]},
        {"co":[-1.0,0.0,0.0]},
        {"co":[-0.5,-0.866_025_403_784_438_6,0.0]},
        {"co":[0.5,-0.866_025_403_784_438_6,0.0]}
    ]);
    let mut doc = apply(
        &base,
        &json!([
            {
                "op":"curve.create",
                "id":"bezier_nurbs",
                "data_id":"bezier_nurbs_data",
                "dimensions":"three_d",
                "splines":[{
                    "type":"nurbs",
                    "points":points.clone(),
                    "order":3,
                    "cyclic":false,
                    "resolution":2,
                    "use_endpoint":true,
                    "use_bezier":true
                }]
            },
            {
                "op":"curve.create",
                "id":"open_bezier_nurbs",
                "data_id":"open_bezier_nurbs_data",
                "dimensions":"three_d",
                "splines":[{
                    "type":"nurbs",
                    "points":points,
                    "order":3,
                    "cyclic":false,
                    "resolution":2,
                    "use_endpoint":false,
                    "use_bezier":true
                }]
            },
            {
                "op":"curve.create",
                "id":"cyclic_bezier_nurbs",
                "data_id":"cyclic_bezier_nurbs_data",
                "dimensions":"three_d",
                "splines":[{
                    "type":"nurbs",
                    "points":cyclic_points,
                    "order":3,
                    "cyclic":true,
                    "resolution":2,
                    "use_endpoint":false,
                    "use_bezier":true
                }]
            }
        ]),
    );
    let snapshot = evaluated(&doc);
    let mesh = &snapshot.meshes[&Id::new("bezier_nurbs").unwrap()];
    assert_eq!(mesh.vertices.len(), 7);
    let open_mesh = &snapshot.meshes[&Id::new("open_bezier_nurbs").unwrap()];
    assert_eq!(open_mesh.vertices.len(), 5);
    let cyclic_mesh = &snapshot.meshes[&Id::new("cyclic_bezier_nurbs").unwrap()];
    assert_eq!(cyclic_mesh.vertices.len(), 6);
    let expected_x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for (vertex, expected_x) in mesh.vertices.iter().zip(expected_x) {
        assert_eq!(vertex.co.to_array(), [expected_x, 0.0, 0.0]);
    }

    let data = doc
        .data_blocks
        .get_mut(&Id::new("bezier_nurbs_data").unwrap())
        .unwrap()
        .curve
        .as_mut()
        .unwrap();
    data.splines[0].custom_knots = true;
    let error = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap_err();
    assert_eq!(
        error.code,
        potter_core::error::ErrorCode::UnsupportedFeature
    );
    assert_eq!(error.details["feature_id"], "curve.nurbs.knot_mode");
    assert_eq!(
        error.message,
        "curve geometry is invalid: unsupported feature `curve.nurbs.knot_mode`: custom NURBS knot vector values are not represented"
    );
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
fn handle_type_updates_replace_stale_per_side_overrides() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let doc = apply(
        &base,
        &json!([
            {
                "op":"curve.create",
                "id":"handles",
                "data_id":"handles_data",
                "dimensions":"three_d",
                "splines":[{
                    "type":"bezier",
                    "points":[
                        {"co":[0.0,0.0,0.0],"handle_left":[-0.2,0.0,0.0],"handle_right":[0.2,0.0,0.0],"handle_type":"auto","handle_left_type":"aligned","handle_right_type":"free"},
                        {"co":[1.0,0.0,0.0],"handle_left":[0.8,0.0,0.0],"handle_right":[1.2,0.0,0.0],"handle_type":"auto","handle_left_type":"aligned","handle_right_type":"free"}
                    ],
                    "order":3,
                    "cyclic":false,
                    "resolution":4,
                    "use_endpoint":false
                }]
            },
            {"op":"curve.point_update","target":{"id":"handles"},"spline_index":0,"point_index":0,"set":{"handle_type":"vector"}},
            {"op":"curve.point_update","target":{"id":"handles"},"spline_index":0,"point_index":1,"set":{"handle_type":"vector","handle_left_type":"free"}},
            {"op":"curve.convert","target":{"id":"handles"},"to":"curves"}
        ]),
    );
    let attributes = &doc.data_blocks[&Id::new("handles_data").unwrap()]
        .hair_curves
        .as_ref()
        .unwrap()
        .attributes;
    assert_eq!(
        attributes["handle_type_left"].values.as_slice(),
        &[json!(2), json!(0)]
    );
    assert_eq!(
        attributes["handle_type_right"].values.as_slice(),
        &[json!(2), json!(2)]
    );
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
fn missing_external_font_without_scene_resources_is_unsupported() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let doc = apply(
        &base,
        &json!([{
            "op": "text_object.create",
            "id": "external_font",
            "body": "A",
            "font": "missing-font.ttf"
        }]),
    );
    let error = Snapshot::evaluate(&doc, &EvaluationContext::default()).unwrap_err();
    assert_eq!(
        error.code,
        potter_core::error::ErrorCode::UnsupportedFeature
    );
    assert_eq!(error.details["feature_id"], "text.font.external.resource");
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
fn built_in_hershey_rejects_layout_settings() {
    type ChangeSetting = fn(&mut TextObjectData);
    let settings: [(&str, ChangeSetting); 7] = [
        ("character_spacing", |text| text.character_spacing = 2.0),
        ("word_spacing", |text| text.word_spacing = 2.0),
        ("line_spacing", |text| text.line_spacing = 2.0),
        ("shear", |text| text.shear = 1.0),
        ("offset_x", |text| text.offset_x = 1.0),
        ("offset_y", |text| text.offset_y = 1.0),
        ("small_caps_scale", |text| text.small_caps_scale = 0.5),
    ];

    for (setting, change) in settings {
        let mut text = TextObjectData {
            body: "A".to_owned(),
            ..TextObjectData::default()
        };
        change(&mut text);
        assert!(
            matches!(
                evaluate_text(&text),
                Err(MeshError::UnsupportedFeature {
                    feature_id: "text.font.builtin.layout_settings",
                    ..
                })
            ),
            "non-default {setting} should be unsupported"
        );
    }
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
        ..HairCurvesData::default()
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
        ..HairCurvesData::default()
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
fn curve_taper_missing_data_block_is_rejected_during_create() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let error = apply_batch(
        &base,
        &json!({
            "schema_version": 1,
            "base_revision": base.revision,
            "operations": [{
                "op":"curve.create",
                "id":"tapered",
                "dimensions":"three_d",
                "bevel_depth":0.1,
                "taper":"missing_profile",
                "splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[1.0,0.0,0.0]}],"order":2,"resolution":1}]
            }]
        }),
    )
    .unwrap_err();
    assert_eq!(error.code, potter_core::error::ErrorCode::DependencyMissing);
    assert_eq!(error.details["operation_index"], 0);
    assert_eq!(error.details["pointer"], "/operations/0/taper");
}

#[test]
fn curve_taper_dependency_cycles_are_rejected_during_update() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let doc = apply(
        &base,
        &json!([
            {
                "op":"curve.create",
                "id":"first",
                "data_id":"first_data",
                "dimensions":"three_d",
                "bevel_depth":0.1,
                "splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[1.0,0.0,0.0]}],"order":2,"resolution":1}]
            },
            {
                "op":"curve.create",
                "id":"second",
                "data_id":"second_data",
                "dimensions":"three_d",
                "bevel_depth":0.1,
                "splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[1.0,0.0,0.0]}],"order":2,"resolution":1}]
            }
        ]),
    );
    let error = apply_batch(
        &doc,
        &json!({
            "schema_version": 1,
            "base_revision": doc.revision,
            "operations": [
                {"op":"curve.update","target":{"id":"first"},"set":{"taper":"second_data"}},
                {"op":"curve.update","target":{"id":"second"},"set":{"taper":"first_data"}}
            ]
        }),
    )
    .unwrap_err();
    assert_eq!(error.code, potter_core::error::ErrorCode::InvalidOperation);
    assert_eq!(error.details["operation_index"], 1);
    assert!(error.message.contains("curve taper dependency cycle"));
}

#[test]
fn non_curve_taper_profile_uses_blenders_unit_factor() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let doc = apply(
        &base,
        &json!([
            {"op":"node.create","id":"profile_mesh","kind":"box","params":{"size":1.0}},
            {
                "op":"curve.create",
                "id":"host",
                "data_id":"host_data",
                "dimensions":"three_d",
                "bevel_depth":0.2,
                "bevel_resolution":1,
                "splines":[{"type":"poly","points":[{"co":[0.0,0.0,0.0]},{"co":[1.0,0.0,0.0]}],"order":2,"resolution":1}]
            }
        ]),
    );
    let mut with_non_curve_taper = doc.clone();
    let taper_id = with_non_curve_taper.nodes[&Id::new("profile_mesh").unwrap()]
        .data
        .clone()
        .unwrap();
    with_non_curve_taper
        .data_blocks
        .get_mut(&Id::new("host_data").unwrap())
        .unwrap()
        .curve
        .as_mut()
        .unwrap()
        .taper = Some(taper_id);

    let baseline = evaluated(&doc).meshes[&Id::new("host").unwrap()].clone();
    let tapered = evaluated(&with_non_curve_taper).meshes[&Id::new("host").unwrap()].clone();
    assert_eq!(
        baseline
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>(),
        tapered
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>()
    );
    assert_eq!(baseline.faces, tapered.faces);
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
