//! Blender-derived regression expectations for the EXACT Boolean. Every expected value comes from a
//! Blender 5.2.2 run (`tests/fixtures/boolean_parity/regressions/generate_regression_expectations.py`,
//! committed as `expected/<case>.json`), never from Potter output.
//!
//! Cases:
//! - b: an unset OBJECT Boolean leaves closed monkey and cube meshes unchanged.
//! - c: an empty COLLECTION operand self-cleans a self-overlapping subject without error.
//! - d: the subject is a member of its own operand collection; no cycle, and the subject is skipped.
//! - e: INDEX remap of operand slots onto the subject's slots (one slot, same material, last match).
//! - f: TRANSFER into a subject without material slots; `modifier.apply` persists the pool with the
//!   empty slot as `null`, and the cutter materials after it.
//! - g: touching cubes rotated about Z, in world space (the rot.py reviewer probe).
//! - h: a cavity wall keeps the cutter's custom normal, flipped: (-1,0,0) appears as (1,0,0).
//! - i: open and closed UV spheres with `use_hole_tolerant` (counts, face sizes, volume).
//!
//! The six reviewer-verified corpus cases are asserted by `blend_boolean_parity.rs`.
//!
//! Custom normals for (h) are authored in Potter with `mesh.set_custom_normals` (each corner carries
//! its face normal, as in Blender), so the case does not depend on how the Blender importer carries
//! custom normals. The (f) `modifier.apply` runs in the same batch.
//!
//! Comparison is the shared canonical comparator (`fixtures/boolean_parity/comparator.rs`). Sphere
//! expectations omit rings, so they compare counts, face sizes, and volume. A missing Blender
//! executable fails the test.

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

#[path = "fixtures/boolean_parity/comparator.rs"]
mod comparator;
use comparator::{
    TestResult, evaluate_boolean, expected_canonical, find_node, import_blend_scene, mismatches,
    read_scene, run_generator, run_pot,
};

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use glam::DVec3;
use potter_core::model::{Id, SceneDoc};
use serde_json::{Value, json};
use tempfile::tempdir;

const BLENDER_REQUIRED: &str = "Blender is required for the Boolean regression expectations: install Blender 5.2.2, or set POTTER_BLENDER to its executable";

/// The regression cases, one expectation file each.
const REGRESSION_CASES: [&str; 23] = [
    "unset_object_monkey",
    "unset_object_cube",
    "empty_collection_no_self",
    "empty_collection_use_self",
    "subject_in_collection_difference",
    "subject_in_collection_union",
    "index_remap_one_slot",
    "index_remap_same_material_keeps_slot",
    "index_remap_last_matching_slot",
    "transfer_empty_subject",
    "rotated_touch_10",
    "rotated_touch_17",
    "rotated_touch_23",
    "rotated_touch_30",
    "rotated_touch_37",
    "rotated_touch_41",
    "rotated_touch_53",
    "rotated_touch_61",
    "cavity_wall_normal",
    "sphere_open64_hole_tolerant_on",
    "sphere_open64_hole_tolerant_off",
    "sphere_open128_hole_tolerant_on",
    "sphere_closed64_hole_tolerant_on",
];

/// Hosts whose custom normals are authored in Potter before evaluation.
const CUSTOM_NORMAL_HOSTS: [&str; 2] = ["cavity_wall_normal", "cavity_wall_normal.cutter"];

/// The host whose Boolean is applied and whose persisted material slots are checked.
const APPLIED_CASE: &str = "transfer_empty_subject";

/// Every `expected/<case>.json` of the regression corpus, keyed by case name.
fn read_expectations(directory: &Path) -> TestResult<BTreeMap<String, Value>> {
    let mut expectations = BTreeMap::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
            let case = value["case"]
                .as_str()
                .ok_or("Blender case name is missing")?
                .to_owned();
            expectations.insert(case, value);
        }
    }
    Ok(expectations)
}

/// Newell's polygon normal, unit length; `None` for a degenerate ring.
fn newell_normal(ring: &[DVec3]) -> Option<DVec3> {
    let mut normal = DVec3::ZERO;
    for (index, current) in ring.iter().enumerate() {
        let next = ring[(index + 1) % ring.len()];
        normal += DVec3::new(
            (current.y - next.y) * (current.z + next.z),
            (current.z - next.z) * (current.x + next.x),
            (current.x - next.x) * (current.y + next.y),
        );
    }
    (normal != DVec3::ZERO).then(|| normal.normalize())
}

/// A `mesh.set_custom_normals` operation over every face of `node_id`, each corner carrying its
/// face normal.
fn face_normal_operation(document: &SceneDoc, node_id: &Id) -> TestResult<Value> {
    let node = document
        .nodes
        .get(node_id)
        .ok_or("custom normal host is missing")?;
    let data_id = node
        .data
        .as_ref()
        .ok_or("custom normal host has no mesh data")?;
    let mesh = document
        .data_blocks
        .get(data_id)
        .and_then(|data| data.mesh.as_ref())
        .ok_or("custom normal host has no mesh")?;
    let positions: HashMap<u32, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut ids = Vec::new();
    let mut normals = serde_json::Map::new();
    for face in &mesh.faces {
        let ring = face
            .vertices
            .iter()
            .map(|id| {
                positions
                    .get(id)
                    .copied()
                    .ok_or("custom normal face references a missing vertex")
            })
            .collect::<Result<Vec<DVec3>, _>>()?;
        let normal = newell_normal(&ring).ok_or("custom normal face has no normal")?;
        let key = format!("f{}", face.id);
        normals.insert(key.clone(), json!(vec![normal.to_array(); ring.len()]));
        ids.push(key);
    }
    Ok(json!({
        "op": "mesh.set_custom_normals",
        "target": {"id": node_id.as_str()},
        "elements": {"domain": "face", "ids": ids},
        "normals": normals,
    }))
}

/// Material names of node slots in slot order; `None` for an empty slot.
fn material_names(document: &SceneDoc, slots: &[Option<Id>]) -> TestResult<Vec<Option<String>>> {
    let mut names = Vec::with_capacity(slots.len());
    for slot in slots {
        names.push(match slot {
            Some(id) => Some(
                document
                    .materials
                    .get(id)
                    .ok_or_else(|| format!("material `{}` is missing", id.as_str()))?
                    .name
                    .clone(),
            ),
            None => None,
        });
    }
    Ok(names)
}

/// The applied material slots of a Blender expectation, by name; `null` for an empty slot.
fn expected_material_names(expected: &Value) -> TestResult<Vec<Option<String>>> {
    let mut names = Vec::new();
    for slot in expected["materials"]
        .as_array()
        .ok_or("Blender applied material slots are missing")?
    {
        names.push(match slot {
            Value::Null => None,
            Value::String(name) => Some(name.clone()),
            _ => return Err("Blender material slot is invalid".into()),
        });
    }
    Ok(names)
}

#[test]
fn exact_boolean_regressions_match_blender() -> TestResult {
    let blender = blender_executable().ok_or(BLENDER_REQUIRED)?;
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/boolean_parity/regressions");
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    run_generator(
        &blender,
        &fixture.join("generate_regression_expectations.py"),
        &root,
    )?;
    let project = root.join("project");
    let mut document = import_blend_scene(&blender, &root.join("scene.blend"), &project)?;

    let expectations = read_expectations(&fixture.join("expected"))?;
    assert_eq!(
        expectations.len(),
        REGRESSION_CASES.len(),
        "the regression corpus must hold exactly the {} expected cases",
        REGRESSION_CASES.len()
    );
    let mut failures = Vec::new();
    for case in REGRESSION_CASES {
        if !expectations.contains_key(case) {
            failures.push(format!("{case}: Blender expectation is missing"));
        }
    }

    // One batch: author the cavity custom normals, then apply the TRANSFER Boolean.
    let mut operations = Vec::new();
    for name in CUSTOM_NORMAL_HOSTS {
        let node_id = find_node(&document, name)?;
        operations.push(face_normal_operation(&document, &node_id)?);
    }
    let applied_id = find_node(&document, APPLIED_CASE)?;
    let modifier_id = document.nodes[&applied_id]
        .modifiers
        .first()
        .ok_or("the TRANSFER host has no Boolean modifier")?
        .id
        .as_str()
        .to_owned();
    operations.push(json!({
        "op": "modifier.apply",
        "target": {"id": applied_id.as_str()},
        "modifier_id": modifier_id,
    }));
    let batch = root.join("regression_batch.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 1,
            "operations": operations,
        }))?,
    )?;
    let project_path = project.to_str().ok_or("project path is not UTF-8")?;
    let batch_path = batch.to_str().ok_or("batch path is not UTF-8")?;
    run_pot(&["apply", project_path, "--file", batch_path, "--json"])?;
    document = read_scene(&project)?;

    for (case, expected_json) in &expectations {
        let node_id = find_node(&document, case)?;
        if expected_json["kind"] == "applied" {
            let actual_slots = material_names(&document, &document.nodes[&node_id].materials)?;
            let expected_slots = expected_material_names(expected_json)?;
            if actual_slots != expected_slots {
                failures.push(format!(
                    "{case}: applied material slots {actual_slots:?} differ from Blender {expected_slots:?}"
                ));
            }
        }
        match evaluate_boolean(&document, &node_id) {
            Ok(actual) => {
                let expected = expected_canonical(expected_json)?;
                failures.extend(
                    mismatches(&actual, &expected)
                        .into_iter()
                        .map(|difference| format!("{case}: {difference}")),
                );
            }
            Err(error) => failures.push(format!("{case}: {error}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} Boolean regression mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}
