#![expect(clippy::unwrap_used, reason = "integration tests")]

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::{
        Mesh,
        lattice::{self, LatticeData},
    },
    model::{Id, SceneDoc},
    ops::apply_batch,
};
use proptest::prelude::*;
use serde_json::{Value, json};

fn apply(doc: &SceneDoc, operations: &Value) -> SceneDoc {
    apply_batch(
        doc,
        &json!({
            "schema_version": 1,
            "base_revision": doc.revision,
            "operations": operations,
        }),
    )
    .unwrap()
    .doc
}

fn evaluated(doc: &SceneDoc) -> Snapshot {
    Snapshot::evaluate(doc, &EvaluationContext::default()).unwrap()
}

proptest! {
    #[test]
    fn identity_lattice_preserves_every_mesh_position(
        positions in prop::collection::vec(
            ( -10.0_f64..10.0, -10.0_f64..10.0, -10.0_f64..10.0),
            0..64,
        ),
    ) {
        let mesh = Mesh::from_positions_and_faces(
            positions
                .iter()
                .map(|&(x, y, z)| DVec3::new(x, y, z))
                .collect(),
            Vec::new(),
        ).unwrap();
        let output = lattice::deform_mesh(&mesh, &LatticeData::default(), None).unwrap();
        prop_assert_eq!(output, mesh);
    }
}

#[test]
fn lattice_modifier_blends_by_vertex_group_in_object_space() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let source = apply(
        &base,
        &json!([{"op":"node.create","id":"body","kind":"box","params":{"size":2.0}}]),
    );
    let mesh_data = source.nodes[&Id::new("body").unwrap()]
        .data
        .as_ref()
        .unwrap();
    let mesh = source.data_blocks[mesh_data].mesh.as_ref().unwrap();
    let weights = mesh
        .vertices
        .iter()
        .map(|vertex| json!({"vertex_id":vertex.id,"weight":0.5}))
        .collect::<Vec<_>>();
    let points = LatticeData::default()
        .points
        .iter()
        .map(|point| (*point + DVec3::X).to_array())
        .collect::<Vec<_>>();
    let doc = apply(
        &source,
        &json!([
            {"op":"vertex_group.create","target":{"id":"body"},"id":"region","name":"Region"},
            {"op":"vertex_group.assign","target":{"id":"body"},"group_id":"region","weights":weights},
            {"op":"lattice.create","id":"cage","points":points,"transform":{"translation":[0.0,0.0,0.0],"rotation":[0.0,0.0,std::f64::consts::FRAC_1_SQRT_2,std::f64::consts::FRAC_1_SQRT_2],"scale":[1.0,1.0,1.0]}},
            {"op":"modifier.create","target":{"id":"body"},"id":"deform","type":"lattice","params":{"object":"cage","vertex_group":"region"}}
        ]),
    );
    let result = &evaluated(&doc).meshes[&Id::new("body").unwrap()];
    let original = source.data_blocks[mesh_data].mesh.as_ref().unwrap();
    for (before, after) in original.vertices.iter().zip(&result.vertices) {
        assert!((after.co.x - before.co.x).abs() <= 1.0e-12);
        assert!((after.co.y - before.co.y - 0.5).abs() <= 1.0e-12);
        assert!((after.co.z - before.co.z).abs() <= 1.0e-12);
    }
}
