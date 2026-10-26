//! Purpose-resolved modifier evaluation through the public `Snapshot` API.
//!
//! Blender 5.2.2 reference (`/tmp/pr18l-fix/PurposeRender/probe.py`; factory cube of size 2 with a
//! Subdivision modifier of `levels=1`, `render_levels=2`): the viewport depsgraph and
//! `DAG_EVAL_VIEWPORT` OBJ export give 26 vertices, `DAG_EVAL_RENDER` gives 98. With
//! `show_render=False` the render export gives the base 8 vertices; with `show_viewport=False` the
//! viewport gives 8 and the render gives 98.
//!
//! The Boolean, Decimate, and union counts and volumes are Blender 5.2.2 results committed in
//! `tests/fixtures/purpose_eval/expected.json`, written by `generate_purpose_eval_expectations.py`.

use std::{collections::HashMap, error::Error, fmt::Debug, fs, path::Path};

use glam::DVec3;
use potter_core::{
    error::{ErrorCode, PotError},
    eval::{EvaluationContext, EvaluationPurpose, Snapshot},
    geom::Mesh,
    model::{Id, SceneDoc},
    ops,
};
use serde_json::{Value, json};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";
const BASE_VERTICES: usize = 8;
const LEVEL_ONE_VERTICES: usize = 26;
const LEVEL_TWO_VERTICES: usize = 98;
const EXPECTATIONS: &str = "tests/fixtures/purpose_eval/expected.json";
// Mirror `VOLUME_RELATIVE_TOLERANCE` and `VOLUME_ABSOLUTE_TOLERANCE` in
// `fixtures/boolean_parity/comparator.rs`; keep the values equal.
const VOLUME_RELATIVE_TOLERANCE: f64 = 1.0e-6;
const VOLUME_ABSOLUTE_TOLERANCE: f64 = 1.0e-9;

fn ensure_eq<T: PartialEq + Debug>(actual: &T, expected: &T, what: &str) -> TestResult {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{what}: expected {expected:?}, got {actual:?}").into())
    }
}

fn apply(doc: &SceneDoc, operations: &[Value]) -> Result<SceneDoc, PotError> {
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": operations,
    });
    Ok(ops::apply_batch(doc, &batch)?.doc)
}

/// A box named `cube` followed by `modifiers`, applied in order.
fn cube_with(modifiers: &[Value]) -> TestResult<SceneDoc> {
    let mut operations = vec![json!({
        "op": "node.create",
        "id": "cube",
        "kind": "box",
        "params": {"size": 2.0},
    })];
    operations.extend(modifiers.iter().cloned());
    Ok(apply(&SceneDoc::new(SCENE_ID.to_owned()), &operations)?)
}

fn subdivision(
    id: &str,
    levels: u64,
    render_levels: u64,
    show_viewport: bool,
    show_render: bool,
) -> Value {
    json!({
        "op": "modifier.create",
        "target": {"id": "cube"},
        "id": id,
        "type": "subdivision",
        "params": {"levels": levels, "render_levels": render_levels},
        "show_viewport": show_viewport,
        "show_render": show_render,
    })
}

fn purpose_context(purpose: EvaluationPurpose) -> EvaluationContext {
    EvaluationContext {
        purpose,
        ..EvaluationContext::default()
    }
}

fn cube_vertices(doc: &SceneDoc, purpose: EvaluationPurpose) -> TestResult<Vec<DVec3>> {
    let snapshot = Snapshot::evaluate(doc, &purpose_context(purpose))?;
    let mesh = snapshot
        .meshes
        .get(&Id::new("cube")?)
        .ok_or("evaluated cube mesh is missing")?;
    Ok(mesh.vertices.iter().map(|vertex| vertex.co).collect())
}

fn ensure(condition: bool, what: &str) -> TestResult {
    if condition { Ok(()) } else { Err(what.into()) }
}

/// The Boolean comparator's volume rule: relative tolerance, or absolute for an expected zero.
fn ensure_volume(actual: f64, expected: f64, what: &str) -> TestResult {
    let matches = if expected.abs() <= VOLUME_ABSOLUTE_TOLERANCE {
        actual.abs() <= VOLUME_ABSOLUTE_TOLERANCE
    } else {
        ((actual - expected) / expected).abs() <= VOLUME_RELATIVE_TOLERANCE
    };
    ensure(
        matches,
        &format!("{what}: expected {expected}, got {actual}"),
    )
}

/// Blender's per-purpose numbers, written by the generator in `fixtures/purpose_eval`.
fn expectations() -> TestResult<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(EXPECTATIONS);
    let bytes = fs::read(&path).map_err(|error| {
        format!(
            "{}: {error}; run the purpose_eval generator",
            path.display()
        )
    })?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn evaluated_mesh<'a>(snapshot: &'a Snapshot, id: &Id) -> TestResult<&'a Mesh> {
    snapshot
        .meshes
        .get(id)
        .ok_or_else(|| "evaluated mesh is missing".into())
}

/// Blender-comparable counts and signed volume of one evaluated mesh.
struct Counts {
    vertices: usize,
    edges: usize,
    faces: usize,
    volume: f64,
}

/// Counts, and signed volume by the divergence theorem over the tessellation. The volume is
/// positive for a closed mesh with outward winding.
fn summary(mesh: &Mesh) -> TestResult<Counts> {
    let positions: HashMap<u32, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let corner = |id: u32| {
        positions
            .get(&id)
            .copied()
            .ok_or("tessellated vertex is missing")
    };
    let mut six_volume = 0.0;
    for [a, b, c] in mesh.tessellate()? {
        six_volume += corner(a)?.dot(corner(b)?.cross(corner(c)?));
    }
    Ok(Counts {
        vertices: mesh.vertices.len(),
        edges: mesh.edges.len(),
        faces: mesh.faces.len(),
        volume: six_volume / 6.0,
    })
}

/// Reads one `{vertices, edges, faces, volume}` entry from the Blender expectations.
fn expected_counts(entry: &Value) -> TestResult<Counts> {
    let count = |key: &str| -> TestResult<usize> {
        let raw = entry[key]
            .as_u64()
            .ok_or_else(|| format!("expected `{key}` is not a count"))?;
        Ok(usize::try_from(raw)?)
    };
    Ok(Counts {
        vertices: count("vertices")?,
        edges: count("edges")?,
        faces: count("faces")?,
        volume: entry["volume"]
            .as_f64()
            .ok_or("expected `volume` is not a number")?,
    })
}

/// Counts must match exactly; the volume uses the Boolean comparator's tolerance.
fn ensure_counts(actual: &Counts, expected: &Counts, what: &str) -> TestResult {
    ensure_eq(
        &actual.vertices,
        &expected.vertices,
        &format!("{what} vertex count"),
    )?;
    ensure_eq(
        &actual.edges,
        &expected.edges,
        &format!("{what} edge count"),
    )?;
    ensure_eq(
        &actual.faces,
        &expected.faces,
        &format!("{what} face count"),
    )?;
    ensure_volume(actual.volume, expected.volume, &format!("{what} volume"))
}

#[test]
fn render_levels_apply_only_to_render_purpose() -> TestResult {
    let doc = cube_with(&[subdivision("subdivide", 1, 2, true, true)])?;
    let preview = cube_vertices(&doc, EvaluationPurpose::Preview)?;
    let render = cube_vertices(&doc, EvaluationPurpose::Render)?;
    ensure_eq(&preview.len(), &LEVEL_ONE_VERTICES, "preview vertex count")?;
    ensure_eq(&render.len(), &LEVEL_TWO_VERTICES, "render vertex count")?;

    let level_two = cube_with(&[subdivision("subdivide", 2, 2, true, true)])?;
    ensure_eq(
        &render,
        &cube_vertices(&level_two, EvaluationPurpose::Preview)?,
        "render equals subdivision level 2",
    )?;
    Ok(())
}

#[test]
fn modifier_hidden_for_render_is_omitted_from_render_only() -> TestResult {
    let doc = cube_with(&[subdivision("viewport_only", 1, 2, true, false)])?;
    ensure_eq(
        &cube_vertices(&doc, EvaluationPurpose::Preview)?.len(),
        &LEVEL_ONE_VERTICES,
        "preview keeps the render-hidden modifier",
    )?;
    ensure_eq(
        &cube_vertices(&doc, EvaluationPurpose::Render)?.len(),
        &BASE_VERTICES,
        "render omits the render-hidden modifier",
    )?;
    Ok(())
}

#[test]
fn modifier_hidden_for_viewport_is_render_only() -> TestResult {
    let doc = cube_with(&[subdivision("render_only", 1, 2, false, true)])?;
    ensure_eq(
        &cube_vertices(&doc, EvaluationPurpose::Preview)?.len(),
        &BASE_VERTICES,
        "preview omits the viewport-hidden modifier",
    )?;
    ensure_eq(
        &cube_vertices(&doc, EvaluationPurpose::Render)?.len(),
        &LEVEL_TWO_VERTICES,
        "render applies the render-only modifier at render levels",
    )?;
    Ok(())
}

#[test]
fn hidden_physics_sources_respect_purpose_and_legacy_properties() -> TestResult {
    let particle_settings = json!({
        "emit_from": "VERT",
        "count": 4,
        "frame_start": 1.0,
        "frame_end": 1.0,
        "lifetime": 12.0,
        "physics_type": "NEWTON",
        "normal_factor": 0.0,
        "render_type": "HALO",
        "seed": 11,
    });
    let mut doc = apply(
        &SceneDoc::new(SCENE_ID.to_owned()),
        &[
            json!({"op":"node.create","id":"emitter_hidden","kind":"plane","params":{}}),
            json!({"op":"node.create","id":"emitter_legacy","kind":"plane","params":{},"transform":{"translation":[3.0,0.0,0.0]}}),
            json!({"op":"node.create","id":"collider_hidden","kind":"plane","params":{},"transform":{"translation":[10.0,0.0,0.0]}}),
            json!({"op":"node.create","id":"collision_emitter_hidden","kind":"plane","params":{},"transform":{"translation":[10.0,0.0,0.1]}}),
            json!({"op":"node.create","id":"collider_legacy","kind":"plane","params":{},"transform":{"translation":[20.0,0.0,0.0]}}),
            json!({"op":"node.create","id":"collision_emitter_legacy","kind":"plane","params":{},"transform":{"translation":[20.0,0.0,0.1]}}),
            json!({"op":"node.create","id":"brush_hidden","kind":"plane","params":{},"transform":{"translation":[30.0,0.0,0.0]}}),
            json!({"op":"node.create","id":"canvas_hidden","kind":"plane","params":{},"transform":{"translation":[30.0,0.0,0.0]}}),
            json!({"op":"node.create","id":"brush_legacy","kind":"plane","params":{},"transform":{"translation":[40.0,0.0,0.0]}}),
            json!({"op":"node.create","id":"canvas_legacy","kind":"plane","params":{},"transform":{"translation":[40.0,0.0,0.0]}}),
            json!({"op":"physics.particle_emitter.create","target":{"id":"emitter_hidden"},"settings":particle_settings.clone()}),
            json!({"op":"physics.particle_emitter.create","target":{"id":"emitter_legacy"},"settings":particle_settings.clone()}),
            json!({"op":"physics.particle_emitter.create","target":{"id":"collision_emitter_hidden"},"settings":particle_settings.clone()}),
            json!({"op":"physics.particle_emitter.create","target":{"id":"collision_emitter_legacy"},"settings":particle_settings}),
            json!({"op":"physics.collision.create","target":{"id":"collider_hidden"}}),
            json!({"op":"physics.collision.create","target":{"id":"collider_legacy"}}),
            json!({"op":"physics.dynamic_paint.create","target":{"id":"brush_hidden"},"settings":{"role":"brush","paint_source":"POINT","radius":2.0,"strength":1.0}}),
            json!({"op":"physics.dynamic_paint.create","target":{"id":"brush_legacy"},"settings":{"role":"brush","paint_source":"POINT","radius":2.0,"strength":1.0}}),
            json!({"op":"physics.dynamic_paint.create","target":{"id":"canvas_hidden"},"settings":{"role":"canvas","brushes":["brush_hidden"],"radius":2.0,"strength":1.0}}),
            json!({"op":"physics.dynamic_paint.create","target":{"id":"canvas_legacy"},"settings":{"role":"canvas","brushes":["brush_legacy"],"radius":2.0,"strength":1.0}}),
        ],
    )?;
    for (node_id, modifier_type) in [
        ("emitter_hidden", "particle_system"),
        ("collider_hidden", "collision"),
        ("brush_hidden", "dynamic_paint"),
    ] {
        let node = doc
            .nodes
            .get_mut(&Id::new(node_id)?)
            .ok_or("linked physics node is missing")?;
        node.modifiers
            .iter_mut()
            .find(|modifier| modifier.modifier_type == modifier_type)
            .ok_or("linked physics modifier is missing")?
            .show_viewport = false;
    }
    for (node_id, modifier_type) in [
        ("emitter_legacy", "particle_system"),
        ("collider_legacy", "collision"),
        ("brush_legacy", "dynamic_paint"),
    ] {
        let node = doc
            .nodes
            .get_mut(&Id::new(node_id)?)
            .ok_or("legacy physics node is missing")?;
        node.modifiers
            .retain(|modifier| modifier.modifier_type != modifier_type);
        if node_id == "emitter_legacy" {
            node.properties
                .remove("physics_particle_emitter_modifier_id");
        }
    }

    let mut legacy_scene = serde_json::to_value(&doc)?;
    legacy_scene["schema_version"] = json!(9);
    let legacy_settings =
        legacy_scene["nodes"]["brush_legacy"]["properties"]["physics_dynamic_paint"]
            .as_object_mut()
            .ok_or("legacy Dynamic Paint settings are missing")?;
    let _ = legacy_settings.remove("paint_source");
    let _ = legacy_settings.insert("radius".to_owned(), json!(2.0));
    doc = potter_core::model::parse_scene(&serde_json::to_vec(&legacy_scene)?, Path::new("."))?.doc;
    let context = |purpose| EvaluationContext {
        purpose,
        frame: Some(5.0),
        ..EvaluationContext::default()
    };
    let mut unsupported_render_type_doc = doc.clone();
    unsupported_render_type_doc
        .nodes
        .get_mut(&Id::new("emitter_hidden")?)
        .and_then(|node| node.properties.get_mut("physics_particle_emitter"))
        .and_then(Value::as_object_mut)
        .ok_or("linked particle settings are missing")?
        .insert("render_type".to_owned(), json!("UNSUPPORTED"));
    let hidden_unsupported_preview = Snapshot::evaluate(
        &unsupported_render_type_doc,
        &context(EvaluationPurpose::Preview),
    )?;
    ensure_eq(
        &hidden_unsupported_preview
            .simulated_particles
            .contains_key(&Id::new("emitter_hidden")?),
        &false,
        "viewport-hidden unsupported particle render type is not simulated",
    )?;
    let active_render_error = Snapshot::evaluate(
        &unsupported_render_type_doc,
        &context(EvaluationPurpose::Render),
    )
    .err()
    .ok_or("active unsupported particle render type unexpectedly evaluated")?;
    ensure_eq(
        &active_render_error.code,
        &ErrorCode::UnsupportedFeature,
        "active unsupported particle render type error",
    )?;
    ensure_eq(
        &active_render_error.details["feature_id"],
        &json!("physics.particle_emitter.render_type"),
        "active unsupported particle render type feature",
    )?;

    let preview = Snapshot::evaluate(&doc, &context(EvaluationPurpose::Preview))?;
    let render = Snapshot::evaluate(&doc, &context(EvaluationPurpose::Render))?;
    ensure_eq(
        &preview
            .simulated_particles
            .contains_key(&Id::new("emitter_hidden")?),
        &false,
        "viewport-hidden linked particle emitter is not simulated",
    )?;
    ensure_eq(
        &render
            .simulated_particles
            .contains_key(&Id::new("emitter_hidden")?),
        &true,
        "render-visible linked particle emitter is simulated",
    )?;
    ensure_eq(
        &preview
            .simulated_particles
            .contains_key(&Id::new("emitter_legacy")?),
        &true,
        "legacy property-only particle emitter remains simulated",
    )?;

    let collision_z = |snapshot: &Snapshot, emitter: &str| -> TestResult<Vec<f64>> {
        Ok(snapshot
            .simulated_particles
            .get(&Id::new(emitter)?)
            .ok_or("collision emitter particle states are missing")?
            .iter()
            .map(|state| state.position[2])
            .collect())
    };
    let hidden_preview_z = collision_z(&preview, "collision_emitter_hidden")?;
    let hidden_render_z = collision_z(&render, "collision_emitter_hidden")?;
    let legacy_z = collision_z(&preview, "collision_emitter_legacy")?;
    ensure_eq(
        &hidden_preview_z.iter().all(|height| *height < 0.0),
        &true,
        "viewport-hidden linked collision does not deflect particles",
    )?;
    ensure_eq(
        &hidden_render_z.iter().all(|height| *height >= -1.0e-8),
        &true,
        "render-visible linked collision deflects particles",
    )?;
    ensure_eq(
        &legacy_z.iter().all(|height| *height >= -1.0e-8),
        &true,
        "legacy property-only collision remains active",
    )?;

    let color_alphas = |snapshot: &Snapshot, canvas: &str| -> TestResult<Vec<f64>> {
        let mesh = snapshot
            .meshes
            .get(&Id::new(canvas)?)
            .ok_or("dynamic-paint canvas mesh is missing")?;
        let values = mesh
            .attributes
            .get("color")
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object)
            .ok_or("dynamic-paint colors are missing")?;
        let mut alphas = Vec::with_capacity(values.len());
        for color in values.values() {
            alphas.push(
                color
                    .get(3)
                    .and_then(Value::as_f64)
                    .ok_or("dynamic-paint alpha is missing")?,
            );
        }
        Ok(alphas)
    };
    let hidden_preview_alpha = color_alphas(&preview, "canvas_hidden")?;
    let hidden_render_alpha = color_alphas(&render, "canvas_hidden")?;
    let legacy_alpha = color_alphas(&preview, "canvas_legacy")?;
    ensure_eq(
        &hidden_preview_alpha.iter().all(|alpha| *alpha == 0.0),
        &true,
        "viewport-hidden linked brush does not paint",
    )?;
    ensure_eq(
        &hidden_render_alpha.iter().any(|alpha| *alpha > 0.0),
        &true,
        "render-visible linked brush paints",
    )?;
    ensure_eq(
        &legacy_alpha.iter().any(|alpha| *alpha > 0.0),
        &true,
        "legacy property-only brush remains active",
    )?;
    Ok(())
}

#[test]
fn boolean_results_feed_dependent_modifiers_for_both_purposes() -> TestResult {
    // `host` runs Boolean then Decimate; `follower` consumes its evaluated output.
    let doc = apply(
        &SceneDoc::new(SCENE_ID.to_owned()),
        &[
            json!({"op": "node.create", "id": "host", "kind": "box", "params": {"size": 2.0}}),
            json!({"op": "node.create", "id": "cutter", "kind": "box", "params": {"size": 1.0}}),
            json!({
                "op": "modifier.create",
                "target": {"id": "host"},
                "id": "boolean",
                "type": "boolean",
                "params": {"object": "cutter", "operation": "DIFFERENCE", "solver": "EXACT"},
            }),
            json!({
                "op": "modifier.create",
                "target": {"id": "host"},
                "id": "decimate",
                "type": "decimate",
                "params": {"ratio": 0.6},
            }),
            json!({"op": "node.create", "id": "follower", "kind": "box", "params": {"size": 1.0}}),
            json!({
                "op": "modifier.create",
                "target": {"id": "follower"},
                "id": "follow_boolean",
                "type": "boolean",
                "params": {"object": "host", "operation": "UNION", "solver": "FLOAT"},
            }),
        ],
    )?;
    let host = Id::new("host")?;
    let follower = Id::new("follower")?;
    let cutter = Id::new("cutter")?;
    let expected = expectations()?;

    // The same host without its Decimate: the Boolean's own output, as Blender computes it.
    let mut boolean_only = doc.clone();
    let host_node = boolean_only
        .nodes
        .get_mut(&host)
        .ok_or("host node is missing")?;
    for modifier in &mut host_node.modifiers {
        if modifier.modifier_type == "decimate" {
            modifier.enabled = false;
        }
    }

    for (purpose, key) in [
        (EvaluationPurpose::Preview, "preview"),
        (EvaluationPurpose::Render, "render"),
    ] {
        let blender = &expected[key];
        let (snapshot, node_errors) =
            Snapshot::evaluate_available_with_cache(&doc, &purpose_context(purpose), None, None)?;
        ensure_eq(&node_errors.is_empty(), &true, "no node evaluation errors")?;
        for (id, name) in [
            (&host, "host"),
            (&follower, "follower"),
            (&cutter, "cutter"),
        ] {
            ensure_eq(&snapshot.meshes.contains_key(id), &true, name)?;
        }

        let boolean_snapshot = Snapshot::evaluate(&boolean_only, &purpose_context(purpose))?;
        let checks = [
            ("cutter", summary(evaluated_mesh(&snapshot, &cutter)?)?),
            (
                "boolean_host",
                summary(evaluated_mesh(&boolean_snapshot, &host)?)?,
            ),
            (
                "decimated_host",
                summary(evaluated_mesh(&snapshot, &host)?)?,
            ),
            ("follower", summary(evaluated_mesh(&snapshot, &follower)?)?),
        ];
        for (name, actual) in &checks {
            ensure_counts(
                actual,
                &expected_counts(&blender[*name])?,
                &format!("{key} {name}"),
            )?;
        }
    }
    Ok(())
}
