#![expect(clippy::unwrap_used, reason = "integration tests")]

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::{
        BoxParams, Mesh,
        lattice::{self, LatticeData},
        metaball::{self, ElementType, MetaballData, MetaballElement},
        volume::{self, VolumeData, VolumeGrid},
    },
    model::{Id, Modifier, SceneDoc},
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

#[test]
fn metaball_ball_surface_bounds_approximate_its_diameter() {
    let data = MetaballData {
        elements: vec![MetaballElement {
            element_type: ElementType::Ball,
            radius: 1.25,
            ..MetaballElement::default()
        }],
        resolution: 0.05,
        render_resolution: 0.05,
        threshold: 0.5,
    };
    let mesh = metaball::to_mesh(&data).unwrap();
    let bounds = mesh.bounds().unwrap();
    for axis in 0..3 {
        assert!((bounds.size()[axis] - 2.5).abs() <= 0.1);
    }
    assert!(!mesh.faces.is_empty(), "metaball produced no surface");
}

#[test]
fn little_endian_f32_blob_samples_match_inline_density() {
    let samples = [0.0_f32, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0];
    let mut blob = Vec::with_capacity(samples.len() * std::mem::size_of::<f32>());
    for sample in samples {
        blob.extend_from_slice(&sample.to_le_bytes());
    }
    let volume = VolumeData {
        grids: vec![VolumeGrid {
            dims: [2, 2, 2],
            voxel_size: 1.0,
            origin: DVec3::ZERO,
            values: None,
            blob_f32: Some(blob),
            content_ref: Some("sha256:sample-grid".to_owned()),
        }],
        ..VolumeData::default()
    };
    assert!((volume::sample_density(DVec3::new(0.25, 0.5, 0.75), &volume) - 3.5).abs() < 1.0e-12);
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
fn mesh_volume_round_trip_stays_within_one_voxel_of_source_bounds() {
    let source = Mesh::box_mesh(BoxParams {
        size: DVec3::new(1.7, 2.2, 2.5),
    })
    .unwrap();
    let voxel_size = 0.2;
    let volume = volume::mesh_to_volume(&source, voxel_size, 1).unwrap();
    let round_trip = volume::volume_to_mesh(&volume, 0.0).unwrap();
    let source_bounds = source.bounds().unwrap();
    let result_bounds = round_trip.bounds().unwrap();
    let modifier = Modifier {
        id: Id::new("voxelize").unwrap(),
        modifier_type: "mesh_to_volume".to_owned(),
        name: "Mesh to Volume".to_owned(),
        enabled: true,
        params: serde_json::Map::from_iter([
            ("voxel_size".to_owned(), json!(voxel_size)),
            ("padding".to_owned(), json!(1)),
        ]),
        binding_data: None,
        runtime: potter_core::model::ModifierRuntime::default(),
    };
    let modified = potter_core::geom::modifiers::evaluate_modifiers(&source, &[modifier]).unwrap();
    let modifier_bounds = modified.bounds().unwrap();
    for axis in 0..3 {
        assert!((result_bounds.min[axis] - source_bounds.min[axis]).abs() <= voxel_size);
        assert!((result_bounds.max[axis] - source_bounds.max[axis]).abs() <= voxel_size);
        assert!((modifier_bounds.min[axis] - source_bounds.min[axis]).abs() <= voxel_size);
        assert!((modifier_bounds.max[axis] - source_bounds.max[axis]).abs() <= voxel_size);
    }
}

#[test]
fn geometry_operations_evaluate_metaballs_points_and_volume_bounds() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let doc = apply(
        &base,
        &json!([
            {
                "op":"metaball.create",
                "id":"blob",
                "resolution":0.1,
                "elements":[{"type":"ball","co":[0.0,0.0,0.0],"radius":1.0,"stiffness":1.0}]
            },
            {"op":"lattice.create","id":"cage"},
            {
                "op":"pointcloud.create",
                "id":"points",
                "points":[{"id":17,"position":[1.0,2.0,3.0],"radius":0.25}],
                "attributes":{"color":{"domain":"points","values":{"p17":[0.2,0.4,0.8,1.0]}}}
            },
            {
                "op":"volume.create",
                "id":"density",
                "grids":[{"dims":[2,2,2],"voxel_size":0.5,"origin":[0.0,0.0,0.0],"values":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]}]
            }
        ]),
    );
    let snapshot = evaluated(&doc);
    let metaball = &snapshot.meshes[&Id::new("blob").unwrap()];
    assert!(!metaball.faces.is_empty(), "metaball produced no surface");

    let points = &snapshot.meshes[&Id::new("points").unwrap()];
    assert_eq!(points.vertices.len(), 1);
    assert!(points.faces.is_empty(), "{:?}", points.faces);
    assert_eq!(points.vertices[0].id, 17);
    assert_eq!(
        points.attributes["point_radius"]["values"]["v17"],
        json!(0.25)
    );
    assert_eq!(
        points.attributes["color"]["values"]["v17"],
        json!([0.2, 0.4, 0.8, 1.0])
    );

    let volume_bounds = snapshot.nodes[&Id::new("density").unwrap()].bounds.unwrap();
    assert_eq!(volume_bounds.min, DVec3::ZERO);
    assert_eq!(volume_bounds.max, DVec3::ONE * 0.5);
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

#[test]
fn volume_object_modifier_extracts_a_mesh() {
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let doc = apply(
        &base,
        &json!([
            {"op":"node.create","id":"surface","kind":"box","params":{}},
            {
                "op":"volume.create",
                "id":"density",
                "grids":[{"dims":[2,2,2],"voxel_size":1.0,"origin":[0.0,0.0,0.0],"values":[0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0]}],
                "transform":{"translation":[0.0,0.0,0.0],"rotation":[0.0,0.0,std::f64::consts::FRAC_1_SQRT_2,std::f64::consts::FRAC_1_SQRT_2],"scale":[1.0,1.0,1.0]}
            },
            {"op":"modifier.create","target":{"id":"surface"},"id":"extract","type":"volume_to_mesh","params":{"object":"density","threshold":0.5}}
        ]),
    );
    let extracted = &evaluated(&doc).meshes[&Id::new("surface").unwrap()];
    assert!(
        !extracted.faces.is_empty(),
        "volume conversion produced no faces"
    );
    assert!((extracted.bounds().unwrap().size().y).abs() < 1.0e-12);
}
