#![expect(
    clippy::unwrap_used,
    reason = "integration test fixtures use fixed valid inputs"
)]

use std::{fs, path::Path, process::Output};

use assert_cmd::Command;
use proptest::prelude::*;
use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap()
}

fn init_scene(scene: &Path) {
    let output = pot().arg("init").arg(scene).arg("--json").output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

fn apply(scene: &Path, batch: &Value) {
    let schema = potter_core::schema::schema("operations", None).unwrap();
    let validator = jsonschema::validator_for(&schema).unwrap();
    assert!(
        validator.is_valid(batch),
        "physics operation batch does not match its schema: {batch}"
    );
    let directory = tempdir().unwrap();
    let operations = directory.path().join("physics.json");
    fs::write(&operations, serde_json::to_vec(batch).unwrap()).unwrap();
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(operations)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

fn create_drop_scene(scene: &Path, height: f64, mass: f64, seed: u32) {
    init_scene(scene);
    apply(
        scene,
        &json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"floor","kind":"plane","params":{}},
                {"op":"node.create","id":"body","kind":"box","params":{"size":1.0},"transform":{"translation":[0.0,0.0,height]}},
                {"op":"physics.world.update","target":{"id":"scene_main"},"set":{"enabled":true,"gravity":[0.0,0.0,-9.81],"substeps":4,"solver_iterations":8,"frame_start":1,"frame_end":120,"seed":seed}},
                {"op":"physics.rigid_body.create","target":{"id":"floor"},"type":"passive","mass":1.0,"shape":"mesh","restitution":0.0},
                {"op":"physics.rigid_body.create","target":{"id":"body"},"type":"active","mass":mass,"shape":"box","friction":0.5,"restitution":0.0,"linear_damping":0.0,"angular_damping":0.0}
            ]
        }),
    );
}

fn bake(scene: &Path, out: &Path, frames: &str) -> (Output, Value) {
    let output = pot()
        .arg("bake")
        .arg(scene)
        .arg("--kind")
        .arg("simulation")
        .arg("--frames")
        .arg(frames)
        .arg("--out")
        .arg(out)
        .arg("--json")
        .output()
        .unwrap();
    let envelope = json(&output);
    (output, envelope)
}

fn frame_energy(out: &Path, record: &Value, mass: f64) -> f64 {
    let frame_file = record["file"].as_str().unwrap();
    let data: Value = serde_json::from_slice(&fs::read(out.join(frame_file)).unwrap()).unwrap();
    let body = &data["transforms"]["body"];
    let matrix = body["world_matrix"].as_array().unwrap();
    let velocity = body["linear_velocity"].as_array().unwrap();
    let height = matrix[14].as_f64().unwrap();
    let speed_squared = velocity
        .iter()
        .map(|component| component.as_f64().unwrap().powi(2))
        .sum::<f64>();
    mass * 9.81 * height + 0.5 * mass * speed_squared
}

#[test]
fn box_drop_rests_on_passive_plane_and_bake_keeps_scene_unchanged() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("drop_scene");
    let out = directory.path().join("simulation");
    create_drop_scene(&scene, 5.0, 1.0, 23);
    let scene_file = scene.join("scene.json");
    let before: Value = serde_json::from_slice(&fs::read(&scene_file).unwrap()).unwrap();

    let (output, envelope) = bake(&scene, &out, "1:120");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    let last = manifest["frames"].as_array().unwrap().last().unwrap();
    let frame: Value =
        serde_json::from_slice(&fs::read(out.join(last["file"].as_str().unwrap())).unwrap())
            .unwrap();
    let final_height = frame["transforms"]["body"]["world_matrix"][14]
        .as_f64()
        .unwrap();
    assert!(
        (final_height - 0.5).abs() <= 1.0e-3,
        "body ended at z={final_height}"
    );
    assert_eq!(envelope["result"]["scene_modified"], false);
    let inspect = pot()
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("body")
        .arg("--frame")
        .arg("120")
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        inspect.status.success(),
        "{}",
        String::from_utf8_lossy(&inspect.stdout)
    );
    let inspect_height = json(&inspect)["result"]["items"][0]["transform"]["world"]["matrix"][14]
        .as_f64()
        .unwrap();
    assert!(
        (inspect_height - 0.5).abs() <= 1.0e-3,
        "inspect returned simulated z={inspect_height}"
    );
    let after: Value = serde_json::from_slice(&fs::read(&scene_file).unwrap()).unwrap();
    assert_eq!(before["revision"], after["revision"]);
    assert!(
        fs::read_dir(scene.join(".potter/cache"))
            .unwrap()
            .next()
            .is_some()
    );
}
#[test]
fn wind_force_field_accelerates_a_rigid_body() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("wind_scene");
    let out = directory.path().join("wind_bake");
    create_drop_scene(&scene, 2.0, 1.0, 3);
    apply(
        &scene,
        &json!({
            "schema_version":1,
            "base_revision":1,
            "operations":[
                {"op":"node.create","id":"wind","kind":"group","transform":{"translation":[0.0,0.0,0.0]}},
                {"op":"physics.world.update","target":{"id":"scene_main"},"set":{"gravity":[0.0,0.0,0.0]}},
                {"op":"physics.force_field.create","target":{"id":"wind"},"type":"wind","strength":20.0,"falloff":0.0}
            ]
        }),
    );
    let (output, _) = bake(&scene, &out, "1:10");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    let last = manifest["frames"].as_array().unwrap().last().unwrap();
    let frame: Value =
        serde_json::from_slice(&fs::read(out.join(last["file"].as_str().unwrap())).unwrap())
            .unwrap();
    let final_height = frame["transforms"]["body"]["world_matrix"][14]
        .as_f64()
        .unwrap();
    assert!(
        final_height > 2.1,
        "wind did not move the body: z={final_height}"
    );
}

#[test]
fn off_center_contact_updates_body_orientation() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("rotation_scene");
    let out = directory.path().join("rotation_bake");
    create_drop_scene(&scene, 2.0, 1.0, 5);
    apply(
        &scene,
        &json!({
            "schema_version":1,
            "base_revision":1,
            "operations":[{"op":"node.update","target":{"id":"body"},"set":{"transform":{"rotation_deg":[35.0,0.0,0.0]}}}]
        }),
    );
    let (output, _) = bake(&scene, &out, "1:120");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let manifest: Value =
        serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
    let frames = manifest["frames"].as_array().unwrap();
    let first: Value =
        serde_json::from_slice(&fs::read(out.join(frames[0]["file"].as_str().unwrap())).unwrap())
            .unwrap();
    let last: Value = serde_json::from_slice(
        &fs::read(out.join(frames.last().unwrap()["file"].as_str().unwrap())).unwrap(),
    )
    .unwrap();
    let first_matrix = first["transforms"]["body"]["world_matrix"]
        .as_array()
        .unwrap();
    let last_matrix = last["transforms"]["body"]["world_matrix"]
        .as_array()
        .unwrap();
    let rotation_delta = [0_usize, 1, 2, 4, 5, 6, 8, 9, 10]
        .iter()
        .map(|index| {
            (first_matrix[*index].as_f64().unwrap() - last_matrix[*index].as_f64().unwrap()).abs()
        })
        .sum::<f64>();
    assert!(
        rotation_delta > 1.0e-3,
        "off-center collision did not rotate the body"
    );
}

#[test]
fn same_seed_has_same_bake_hash_and_settings_change_the_cache_key() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("deterministic_scene");
    create_drop_scene(&scene, 3.0, 2.0, 7);
    let first = directory.path().join("first_bake");
    let second = directory.path().join("second_bake");
    let (first_output, _) = bake(&scene, &first, "1:8");
    let (second_output, _) = bake(&scene, &second, "1:8");
    assert!(first_output.status.success());
    assert!(second_output.status.success());
    let first_manifest = fs::read(first.join("manifest.json")).unwrap();
    let second_manifest = fs::read(second.join("manifest.json")).unwrap();
    assert_eq!(
        potter_core::hash::sha256(&first_manifest),
        potter_core::hash::sha256(&second_manifest)
    );
    let first_key = json(&first_output)["result"]["cache_keys"][0]
        .as_str()
        .unwrap()
        .to_owned();

    apply(
        &scene,
        &json!({
            "schema_version":1,"base_revision":1,
            "operations":[{"op":"physics.world.update","target":{"id":"scene_main"},"set":{"seed":8}}]
        }),
    );
    let changed = directory.path().join("changed_settings_bake");
    let (changed_output, _) = bake(&scene, &changed, "1:8");
    assert!(changed_output.status.success());
    let changed_response = json(&changed_output);
    let changed_key = changed_response["result"]["cache_keys"][0]
        .as_str()
        .unwrap();
    assert_ne!(first_key, changed_key);
}
#[test]
fn physics_system_operations_persist_update_and_delete_settings() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("physics_systems");
    init_scene(&scene);
    apply(
        &scene,
        &json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"body","kind":"box","params":{"size":2.0}}
            ]
        }),
    );
    apply(
        &scene,
        &json!({
            "schema_version":1,
            "base_revision":1,
            "operations":[
                {"op":"physics.cloth.create","target":{"id":"body"},"settings":{"pin_group":"pinned","quality":4}},
                {"op":"physics.soft_body.create","target":{"id":"body"},"settings":{"goal_group":"goal","volume_stiffness":0.9}},
                {"op":"physics.particle_emitter.create","target":{"id":"body"},"settings":{"rate":3,"lifetime":20},"seed":17},
                {"op":"physics.fluid.create","target":{"id":"body"},"settings":{"type":"liquid","resolution":8}},
                {"op":"physics.dynamic_paint.create","target":{"id":"body"},"settings":{"role":"canvas","radius":0.5,"color":[0.2,0.4,0.8,1.0]}}
            ]
        }),
    );
    let stored: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert_eq!(
        stored["nodes"]["body"]["properties"]["physics_cloth"]["pin_group"],
        "pinned"
    );
    assert_eq!(
        stored["nodes"]["body"]["properties"]["physics_soft_body"]["volume_stiffness"],
        0.9
    );
    assert_eq!(
        stored["nodes"]["body"]["properties"]["physics_particle_emitter"]["seed"],
        17
    );
    assert_eq!(
        stored["nodes"]["body"]["properties"]["physics_fluid"]["type"],
        "liquid"
    );
    assert_eq!(
        stored["nodes"]["body"]["properties"]["physics_dynamic_paint"]["role"],
        "canvas"
    );

    apply(
        &scene,
        &json!({
            "schema_version":1,
            "base_revision":2,
            "operations":[
                {"op":"physics.cloth.update","target":{"id":"body"},"set":{"stiffness":0.7}},
                {"op":"physics.soft_body.update","target":{"id":"body"},"set":{"volume_stiffness":0.8}},
                {"op":"physics.particle_emitter.update","target":{"id":"body"},"set":{"rate":6}},
                {"op":"physics.fluid.update","target":{"id":"body"},"set":{"viscosity":0.1}},
                {"op":"physics.dynamic_paint.update","target":{"id":"body"},"set":{"strength":0.5}}
            ]
        }),
    );
    let updated: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert_eq!(
        updated["nodes"]["body"]["properties"]["physics_cloth"]["stiffness"],
        0.7
    );
    assert_eq!(
        updated["nodes"]["body"]["properties"]["physics_soft_body"]["volume_stiffness"],
        0.8
    );
    assert_eq!(
        updated["nodes"]["body"]["properties"]["physics_particle_emitter"]["rate"],
        6
    );
    assert_eq!(
        updated["nodes"]["body"]["properties"]["physics_fluid"]["viscosity"],
        0.1
    );
    assert_eq!(
        updated["nodes"]["body"]["properties"]["physics_dynamic_paint"]["strength"],
        0.5
    );

    apply(
        &scene,
        &json!({
            "schema_version":1,
            "base_revision":3,
            "operations":[
                {"op":"physics.cloth.delete","target":{"id":"body"}},
                {"op":"physics.soft_body.delete","target":{"id":"body"}},
                {"op":"physics.particle_emitter.delete","target":{"id":"body"}},
                {"op":"physics.fluid.delete","target":{"id":"body"}},
                {"op":"physics.dynamic_paint.delete","target":{"id":"body"}}
            ]
        }),
    );
    let deleted: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    for property in [
        "physics_cloth",
        "physics_soft_body",
        "physics_particle_emitter",
        "physics_fluid",
        "physics_dynamic_paint",
    ] {
        assert!(
            deleted["nodes"]["body"]["properties"]
                .get(property)
                .is_none()
        );
    }
}

#[test]
fn fluid_smoke_and_fire_remain_honestly_unsupported() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("unsupported_smoke");
    init_scene(&scene);
    apply(
        &scene,
        &json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"domain","kind":"box","params":{"size":2.0}}
            ]
        }),
    );
    for kind in ["smoke", "fire"] {
        let batch = json!({
            "schema_version":1,
            "base_revision":1,
            "operations":[
                {"op":"physics.fluid.create","target":{"id":"domain"},"settings":{"type":kind}}
            ]
        });
        let operations = directory.path().join(format!("unsupported_{kind}.json"));
        fs::write(&operations, serde_json::to_vec(&batch).unwrap()).unwrap();
        let output = pot()
            .arg("apply")
            .arg(&scene)
            .arg("--file")
            .arg(&operations)
            .arg("--json")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(4));
        let response = json(&output);
        assert_eq!(
            response["error"]["details"]["feature_id"],
            "physics.fluid.smoke_fire"
        );
        let unchanged: Value =
            serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
        assert_eq!(unchanged["revision"], 1);
    }
}

#[test]
fn cloth_keeps_a_pinned_corner_and_bounds_spring_lengths() {
    use glam::DVec3;
    use potter_core::geom::{GridParams, Mesh};

    let mut mesh = Mesh::grid(GridParams {
        size_x: 2.0,
        size_y: 2.0,
        x_subdivisions: 3,
        y_subdivisions: 3,
    })
    .unwrap();
    mesh.attributes
        .insert("vertex_groups".to_owned(), json!({"pin":{"0":1.0}}));
    let initial = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let output = potter_core::sim::cloth::simulate(
        &mesh,
        &json!({"pin_group":"pin","substeps":4,"iterations":12}),
        8.0,
        24.0,
        DVec3::new(0.0, 0.0, -9.81),
        &[],
    )
    .unwrap();
    assert_eq!(output[0], initial[0]);
    assert!(output.iter().skip(1).any(|position| position.z < 0.0));
    for edge in &mesh.edges {
        let rest = initial[edge.vertices[0] as usize].distance(initial[edge.vertices[1] as usize]);
        let length = output[edge.vertices[0] as usize].distance(output[edge.vertices[1] as usize]);
        assert!(
            length <= rest * 1.5,
            "spring length grew from {rest} to {length}"
        );
    }
}

#[test]
fn soft_body_sphere_rebounds_while_preserving_volume() {
    use glam::DVec3;
    use potter_core::geom::{Mesh, UvSphereParams};

    fn volume(mesh: &Mesh, positions: &[DVec3]) -> f64 {
        let indices = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<std::collections::BTreeMap<_, _>>();
        mesh.faces
            .iter()
            .map(|face| {
                let first = positions[indices[&face.vertices[0]]];
                (1..face.vertices.len() - 1)
                    .map(|index| {
                        let second = positions[indices[&face.vertices[index]]];
                        let third = positions[indices[&face.vertices[index + 1]]];
                        first.dot(second.cross(third)) / 6.0
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
    }

    let sphere = Mesh::uv_sphere(UvSphereParams {
        segments: 8,
        ring_count: 6,
        radius: 1.0,
    })
    .unwrap();
    let rest = sphere
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let rest_volume = volume(&sphere, &rest).abs();
    let collider = [[
        DVec3::new(-10.0, -10.0, -2.0),
        DVec3::new(10.0, -10.0, -2.0),
        DVec3::new(0.0, 10.0, -2.0),
    ]];
    let settings = json!({"substeps":4,"iterations":12,"volume_stiffness":0.9,"restitution":0.4});
    let frame_12 = potter_core::sim::softbody::simulate(
        &sphere,
        &settings,
        12.0,
        24.0,
        DVec3::new(0.0, 0.0, -9.81),
        &collider,
    )
    .unwrap();
    let frame_13 = potter_core::sim::softbody::simulate(
        &sphere,
        &settings,
        13.0,
        24.0,
        DVec3::new(0.0, 0.0, -9.81),
        &collider,
    )
    .unwrap();
    let frame_12_volume = volume(&sphere, &frame_12).abs();
    assert!(((frame_12_volume / rest_volume) - 1.0).abs() <= 0.1);
    let center_z = |positions: &[DVec3]| {
        positions.iter().map(|position| position.z).sum::<f64>()
            / f64::from(u32::try_from(positions.len()).unwrap())
    };
    assert!(
        center_z(&frame_13) > center_z(&frame_12),
        "soft-body sphere did not rebound: {} -> {}",
        center_z(&frame_12),
        center_z(&frame_13)
    );
}

#[test]
fn particles_are_seeded_and_emitted_at_rate_times_frames() {
    use glam::{DMat4, DVec3};
    use potter_core::geom::Mesh;

    let emitter =
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])
            .unwrap();
    let settings = json!({"rate":3,"lifetime":20,"speed":1,"source":"faces"});
    let first = potter_core::sim::particles::simulate(
        &emitter,
        &settings,
        DMat4::IDENTITY,
        4.0,
        24.0,
        DVec3::new(0.0, 0.0, -9.81),
        71,
        &[],
        &[],
    )
    .unwrap();
    let second = potter_core::sim::particles::simulate(
        &emitter,
        &settings,
        DMat4::IDENTITY,
        4.0,
        24.0,
        DVec3::new(0.0, 0.0, -9.81),
        71,
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(first.len(), 12);
    assert_eq!(first, second);
    assert_eq!(
        potter_core::hash::sha256(&serde_json::to_vec(&first).unwrap()),
        potter_core::hash::sha256(&serde_json::to_vec(&second).unwrap())
    );
}

#[test]
fn sph_liquid_column_collapses_inside_its_domain() {
    use glam::{DMat4, DVec3};
    use potter_core::geom::{BoxParams, Mesh};

    let domain = Mesh::box_mesh(BoxParams::default()).unwrap();
    let bounds = domain.bounds().unwrap();
    let settings = json!({"resolution":4,"particle_radius":0.1});
    let initial = potter_core::sim::fluid::simulate(
        &domain,
        &settings,
        DMat4::IDENTITY,
        0.0,
        24.0,
        DVec3::new(0.0, 0.0, -9.81),
        5,
        &[],
    )
    .unwrap();
    let collapsed = potter_core::sim::fluid::simulate(
        &domain,
        &settings,
        DMat4::IDENTITY,
        8.0,
        24.0,
        DVec3::new(0.0, 0.0, -9.81),
        5,
        &[],
    )
    .unwrap();
    let average_height = |positions: &[DVec3]| {
        positions.iter().map(|position| position.z).sum::<f64>()
            / f64::from(u32::try_from(positions.len()).unwrap())
    };
    assert!(average_height(&collapsed) < average_height(&initial));
    assert!(
        collapsed.iter().all(|position| {
            position.cmpge(bounds.min).all() && position.cmple(bounds.max).all()
        })
    );
}

#[test]
fn dynamic_paint_colors_only_vertices_inside_brush_radius() {
    use glam::{DMat4, DVec3};
    use potter_core::geom::{Mesh, Vertex};

    let canvas = Mesh {
        vertices: vec![
            Vertex {
                id: 8,
                co: DVec3::ZERO,
            },
            Vertex {
                id: 2,
                co: DVec3::X,
            },
            Vertex {
                id: 9,
                co: DVec3::new(2.0, 0.0, 0.0),
            },
        ],
        ..Mesh::default()
    };
    let colors = potter_core::sim::dynamic_paint::paint(
        &canvas,
        DMat4::IDENTITY,
        &[DVec3::ZERO],
        1.5,
        [0.2, 0.4, 0.8, 1.0],
        1.0,
    )
    .unwrap();
    assert_eq!(colors.len(), 3);
    assert!(
        colors[0]
            .iter()
            .zip([0.2, 0.4, 0.8, 1.0])
            .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-12)
    );
    assert!(colors[1][3] > 0.0);
    assert!(colors[2].iter().all(|component| component.abs() <= 1.0e-12));
}
#[test]
fn simulation_bake_contains_deformed_meshes_particles_liquid_and_paint() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("physics_bake_scene");
    let out = directory.path().join("physics_bake");
    init_scene(&scene);
    apply(
        &scene,
        &json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"node.create","id":"cloth","kind":"grid","params":{"size":2.0,"x_subdivisions":2,"y_subdivisions":2}},
                {"op":"node.create","id":"emitter","kind":"plane","params":{}},
                {"op":"node.create","id":"domain","kind":"box","params":{"size":2.0}},
                {"op":"node.create","id":"canvas","kind":"grid","params":{"size":2.0,"x_subdivisions":2,"y_subdivisions":2}},
                {"op":"node.create","id":"weight_canvas","kind":"grid","params":{"size":2.0,"x_subdivisions":2,"y_subdivisions":2}},
                {"op":"node.create","id":"brush","kind":"box","params":{"size":0.1}},
                {"op":"physics.cloth.create","target":{"id":"cloth"}},
                {"op":"physics.particle_emitter.create","target":{"id":"emitter"},"settings":{"rate":2,"lifetime":10,"speed":0.5},"seed":17},
                {"op":"physics.fluid.create","target":{"id":"domain"},"settings":{"type":"liquid","resolution":4}},
                {"op":"physics.dynamic_paint.create","target":{"id":"brush"},"settings":{"role":"brush","radius":1.0,"color":[0.1,0.8,0.3,1.0]}},
                {"op":"physics.dynamic_paint.create","target":{"id":"canvas"},"settings":{"role":"canvas","brushes":["brush"]}},
                {"op":"physics.dynamic_paint.create","target":{"id":"weight_canvas"},"settings":{"role":"canvas","surface_format":"weight","brushes":["brush"]}}
            ]
        }),
    );
    let (output, response) = bake(&scene, &out, "1:2");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(response["result"]["frame_count"], 2);
    let frame: Value =
        serde_json::from_slice(&fs::read(out.join("frame_00000000.json")).unwrap()).unwrap();
    assert!(
        !frame["deformed_meshes"]["cloth"]["vertices"]
            .as_array()
            .unwrap()
            .is_empty(),
        "cloth deformation produced no vertices"
    );
    assert_eq!(frame["particles"]["emitter"].as_array().unwrap().len(), 2);
    assert_eq!(
        frame["fluid_particles"]["domain"].as_array().unwrap().len(),
        64
    );
    assert!(
        !frame["deformed_meshes"]["domain"]["faces"]
            .as_array()
            .unwrap()
            .is_empty(),
        "fluid conversion produced no faces"
    );
    let colors = frame["paint_colors"]["canvas"].as_array().unwrap();
    assert!(colors.iter().any(|color| color[3].as_f64().unwrap() > 0.0));
    let weights = frame["paint_weights"]["weight_canvas"].as_array().unwrap();
    assert!(weights.iter().any(|weight| weight.as_f64().unwrap() > 0.0));
    let weight_attribute = &frame["deformed_meshes"]["weight_canvas"]["attributes"]["weight"];
    assert_eq!(weight_attribute["domain"], "vertices");
    assert_eq!(weight_attribute["type"], "float");
    assert!(weight_attribute["values"]["v4"].as_f64().unwrap() > 0.0);
    assert!(frame["cache_key"].as_str().unwrap().starts_with("sha256:"));
    let first_key = frame["cache_key"].as_str().unwrap();
    let second_out = directory.path().join("physics_bake_repeat");
    let (repeat_output, _) = bake(&scene, &second_out, "1:2");
    assert!(repeat_output.status.success());
    let repeated: Value =
        serde_json::from_slice(&fs::read(second_out.join("frame_00000000.json")).unwrap()).unwrap();
    assert_eq!(repeated["cache_key"], first_key);
    assert_eq!(
        repeated["paint_weights"]["weight_canvas"],
        frame["paint_weights"]["weight_canvas"]
    );

    apply(
        &scene,
        &json!({
            "schema_version":1,
            "base_revision":1,
            "operations":[
                {"op":"physics.particle_emitter.update","target":{"id":"emitter"},"set":{"rate":3}}
            ]
        }),
    );
    let changed_out = directory.path().join("physics_bake_changed");
    let (changed_output, _) = bake(&scene, &changed_out, "1:2");
    assert!(changed_output.status.success());
    let changed: Value =
        serde_json::from_slice(&fs::read(changed_out.join("frame_00000000.json")).unwrap())
            .unwrap();
    assert_ne!(changed["cache_key"], frame["cache_key"]);
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 8, .. ProptestConfig::default() })]

    #[test]
    fn zero_restitution_drop_does_not_increase_mechanical_energy(
        height in 2.0_f64..=6.0,
        mass in 0.25_f64..=8.0,
    ) {
        let directory = tempdir().unwrap();
        let scene = directory.path().join("energy_scene");
        let out = directory.path().join("energy_bake");
        create_drop_scene(&scene, height, mass, 17);
        let (output, _) = bake(&scene, &out, "1:40");
        prop_assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stdout));
        let manifest: Value = serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap();
        let records = manifest["frames"].as_array().unwrap();
        let energies = records.iter().map(|record| frame_energy(&out, record, mass)).collect::<Vec<_>>();
        prop_assert!(energies.windows(2).all(|pair| pair[1] <= pair[0] + 1.0e-4), "energy increased for mass={mass}, height={height}: {energies:?}");
    }
}

#[test]
fn physics_operations_reject_invalid_settings_with_parameter_pointers() {
    let scenarios = [
        (
            json!([{
                "op":"physics.world.update",
                "target":{"id":"scene_main"},
                "set":{"frame_start":12,"frame_end":11}
            }]),
            "/set",
        ),
        (
            json!([
                {"op":"node.create","id":"body","kind":"box","params":{}},
                {"op":"physics.rigid_body.create","target":{"id":"body"},"type":"active","mass":1.0},
                {"op":"physics.rigid_body.update","target":{"id":"body"},"set":{"mass":0.0}}
            ]),
            "/set",
        ),
    ];

    for (index, (operations, pointer)) in scenarios.into_iter().enumerate() {
        let directory = tempdir().unwrap();
        let scene = directory.path().join("invalid_physics");
        init_scene(&scene);
        let batch = json!({
            "schema_version":1,
            "base_revision":0,
            "operations":operations
        });
        let schema = potter_core::schema::schema("operations", None).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        assert!(validator.is_valid(&batch), "{batch}");

        let batch_path = directory.path().join(format!("operations_{index}.json"));
        fs::write(&batch_path, serde_json::to_vec(&batch).unwrap()).unwrap();
        let output = pot()
            .arg("apply")
            .arg(&scene)
            .arg("--file")
            .arg(batch_path)
            .arg("--json")
            .output()
            .unwrap();
        assert!(!output.status.success());
        let envelope = json(&output);
        assert_eq!(envelope["error"]["code"], "INVALID_OPERATION");
        assert_eq!(envelope["error"]["details"]["pointer"], pointer);
    }
}
