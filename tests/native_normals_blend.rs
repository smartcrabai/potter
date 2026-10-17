//! PR9 custom normals against Blender 5.2.2.
//!
//! Blender stores custom normals as one raw code pair per loop, in fan space. The oracle authors
//! some fans and leaves the rest automatic, so a face can mix authored and automatic corners. Potter
//! must store those pairs as `normal_code` state: authored corners as `[i16, i16]`, automatic corners
//! as `null` (never frozen). The second test applies a Subdivision and a Displace modifier to the
//! `Regress` object, exports the result, and checks every corner's normal against Blender's own
//! evaluation of the same modifiers.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use native_oracle::{
    NORMAL_TOLERANCE, StoredCorner, TestResult, apply_operations, blender, build_oracle, code_pair,
    dump_blend, dumped_object, export, has_normal_code_values, import_blend, mesh_of_node, node_id,
    normal_code_rows, read_scene, run_blender_arguments, vec3, vectors_close, write_script,
};
use serde_json::{Value, json};
use tempfile::tempdir;

/// Checks one mesh's stored `normal_code` layer against the oracle's raw code pairs and its
/// Auto/authored statuses. Returns (authored corners, automatic corners).
fn check_stored_codes(
    mesh: &Value,
    codes: &[Value],
    statuses: &[Value],
    label: &str,
) -> TestResult<(usize, usize)> {
    let rows = normal_code_rows(mesh)?;
    assert_eq!(rows.len(), codes.len(), "{label} face count");
    assert_eq!(rows.len(), statuses.len(), "{label} status face count");
    let mut authored = 0;
    let mut automatic = 0;
    for (face, ((row, raw_row), status_row)) in rows.iter().zip(codes).zip(statuses).enumerate() {
        let raw_row = raw_row.as_array().ok_or("oracle code row is missing")?;
        let status_row = status_row
            .as_array()
            .ok_or("oracle status row is missing")?;
        assert_eq!(row.len(), raw_row.len(), "{label} face {face} corner count");
        assert_eq!(
            row.len(),
            status_row.len(),
            "{label} face {face} status count"
        );
        for (corner, stored) in row.iter().enumerate() {
            let raw = code_pair(&raw_row[corner])?;
            let oracle_authored = !status_row[corner].is_null();
            if raw == [0, 0] {
                assert_eq!(
                    *stored,
                    StoredCorner::Auto,
                    "{label} face {face} corner {corner} is automatic in Blender"
                );
                assert!(
                    !oracle_authored,
                    "{label} face {face} corner {corner}: status says authored, code says automatic"
                );
                automatic += 1;
            } else {
                assert_eq!(
                    *stored,
                    StoredCorner::Code(raw),
                    "{label} face {face} corner {corner}"
                );
                assert!(
                    oracle_authored,
                    "{label} face {face} corner {corner}: code says authored, status says automatic"
                );
                authored += 1;
            }
        }
    }
    Ok((authored, automatic))
}

#[test]
fn import_stores_raw_blender_code_pairs_as_normal_code_state() -> TestResult {
    let Some(blender) = blender() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, expected) = build_oracle(&blender, root.path())?;
    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;

    // `normal_code_rows` also requires the layer to be present, with type `normal_code`.
    let oracle_codes = expected["oracle"]["custom_normal_codes"]
        .as_array()
        .ok_or("oracle custom normal codes are missing")?;
    let oracle_statuses = expected["oracle"]["custom_normal"]
        .as_array()
        .ok_or("oracle custom normal statuses are missing")?;
    let (authored, automatic) = check_stored_codes(
        mesh_of_node(&scene, "Oracle")?,
        oracle_codes,
        oracle_statuses,
        "Oracle",
    )?;
    assert!(
        authored > 0 && automatic > 0,
        "the oracle must mix both statuses"
    );

    let regress_codes = expected["regress"]["custom_normal_codes"]
        .as_array()
        .ok_or("regress custom normal codes are missing")?;
    let regress_statuses = expected["regress"]["custom_normal"]
        .as_array()
        .ok_or("regress custom normal statuses are missing")?;
    let (authored, automatic) = check_stored_codes(
        mesh_of_node(&scene, "Regress")?,
        regress_codes,
        regress_statuses,
        "Regress",
    )?;
    assert!(
        authored > 0 && automatic > 0,
        "the regress object must mix both statuses"
    );
    Ok(())
}

fn corner_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    (0..3)
        .map(|axis| (left[axis] - right[axis]).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// One corner of a dumped mesh in world space: vertex position, face center, Blender normal.
struct DumpedCorner {
    co: [f64; 3],
    center: [f64; 3],
    normal: [f64; 3],
}

fn dumped_corners(object: &Value) -> TestResult<Vec<DumpedCorner>> {
    let vertices = object["vertices"]
        .as_array()
        .ok_or("dumped vertices are missing")?;
    let loops = object["loops"]
        .as_array()
        .ok_or("dumped loops are missing")?;
    let normals = object["corner_normals"]
        .as_array()
        .ok_or("dumped corner normals are missing")?;
    let mut corners = Vec::new();
    for face in object["faces"]
        .as_array()
        .ok_or("dumped faces are missing")?
    {
        let center = vec3(&face["center"])?;
        let start = face["loop_start"].as_u64().ok_or("loop start is missing")? as usize;
        let total = face["loop_total"].as_u64().ok_or("loop total is missing")? as usize;
        for loop_index in start..start + total {
            let vertex = loops[loop_index]["vertex"]
                .as_u64()
                .ok_or("loop vertex is missing")? as usize;
            corners.push(DumpedCorner {
                co: vec3(&vertices[vertex])?,
                center,
                normal: vec3(&normals[loop_index])?,
            });
        }
    }
    Ok(corners)
}

#[test]
fn subdivision_then_displace_keeps_every_corner_normal_as_blender_computes_it() -> TestResult {
    let Some(blender) = blender() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, expected) = build_oracle(&blender, root.path())?;
    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let node = node_id(&read_scene(&project)?, "Regress")?;
    let applied = apply_operations(
        &project,
        root.path(),
        json!([
            {"op": "modifier.create", "target": {"id": node}, "id": "subdivide", "type": "subdivision", "params": {"levels": 1, "use_custom_normals": true}},
            {"op": "modifier.apply", "target": {"id": node}, "modifier_id": "subdivide"},
            {"op": "modifier.create", "target": {"id": node}, "id": "displace", "type": "displace", "params": {"strength": 0.1, "mid_level": 0.0}},
            {"op": "modifier.apply", "target": {"id": node}, "modifier_id": "displace"}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");

    // Automatic corners must stay automatic after the topology change: `null`, not frozen codes.
    let scene = read_scene(&project)?;
    let mut nulls = 0;
    let mut codes = 0;
    for row in normal_code_rows(mesh_of_node(&scene, "Regress")?)? {
        for stored in row {
            match stored {
                StoredCorner::Auto => nulls += 1,
                StoredCorner::Code(_) => codes += 1,
                StoredCorner::ExplicitZero => {
                    panic!("a Blender import has no literal zero vectors")
                }
            }
        }
    }
    assert!(nulls > 0, "automatic corners were frozen into codes");
    assert!(codes > 0, "authored corners were lost");

    let exported_blend = root.path().join("regress.blend");
    export(&blender, &project, "blend", &exported_blend)?;
    let dump = dump_blend(&blender, root.path(), &exported_blend, "regress")?;
    let object = dumped_object(&dump, "Regress")?;
    let actual = dumped_corners(object)?;

    let reference = expected["regress"]["reference_corners"]
        .as_array()
        .ok_or("reference corners are missing")?;
    assert_eq!(
        actual.len(),
        reference.len(),
        "corner count after subdivision"
    );
    for (index, wanted) in reference.iter().enumerate() {
        let co = vec3(&wanted["co"])?;
        let center = vec3(&wanted["center"])?;
        let normal = vec3(&wanted["normal"])?;
        let (best, distance) = actual
            .iter()
            .map(|corner| {
                (
                    corner,
                    corner_distance(corner.co, co) + corner_distance(corner.center, center),
                )
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or("no dumped corners")?;
        assert!(
            distance < 1.0e-4,
            "corner {index} has no matching position in the Blender dump"
        );
        assert!(
            vectors_close(best.normal, normal, NORMAL_TOLERANCE),
            "corner {index} at {co:?}: Potter-exported normal {:?} differs from Blender {normal:?}",
            best.normal
        );
    }
    Ok(())
}

/// Blender's measured answer to whether a Subdivision keeps the `custom_normal` layer on the oracle's
/// `Regress` object: the default, an explicit `false`, and an explicit `true`.
const SUBSURF_CUSTOM_NORMAL_PROBE: &str = r#"import json
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1]
regress = bpy.data.objects["Regress"]


def has_custom_normal_layer(mesh):
    return "custom_normal" in [attribute.name for attribute in mesh.attributes]


def evaluated_layer(**settings):
    regress.modifiers.clear()
    modifier = regress.modifiers.new("Subdivision", "SUBSURF")
    modifier.levels = 1
    modifier.render_levels = 1
    for name, value in settings.items():
        setattr(modifier, name, value)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = regress.evaluated_get(depsgraph).to_mesh()
    present = has_custom_normal_layer(evaluated)
    regress.evaluated_get(depsgraph).to_mesh_clear()
    regress.modifiers.clear()
    return present


result = {
    "base": has_custom_normal_layer(regress.data),
    "default": evaluated_layer(),
    "explicit_false": evaluated_layer(use_custom_normals=False),
    "explicit_true": evaluated_layer(use_custom_normals=True),
}
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"#;

#[test]
fn subdivision_without_use_custom_normals_drops_the_custom_normal_layer() -> TestResult {
    let Some(blender) = blender() else {
        return Ok(());
    };
    let root = tempdir()?;
    let (blend, _) = build_oracle(&blender, root.path())?;

    // Blender's own evaluation is the reference for what Potter must do.
    let probe_script = write_script(root.path(), "subsurf_probe.py", SUBSURF_CUSTOM_NORMAL_PROBE)?;
    let probe_json = root.path().join("subsurf_probe.json");
    run_blender_arguments(
        &blender,
        &probe_script,
        Some(blend.as_path()),
        &[&probe_json],
    )?;
    let probe: Value = serde_json::from_slice(&std::fs::read(&probe_json)?)?;
    assert_eq!(
        probe["base"], true,
        "the oracle's Regress must start with custom normals"
    );
    assert_eq!(
        probe["default"], false,
        "Blender drops custom normals from a Subdivision without use_custom_normals"
    );
    assert_eq!(
        probe["explicit_false"], false,
        "Blender drops custom normals from a Subdivision with use_custom_normals=False"
    );
    assert_eq!(
        probe["explicit_true"], true,
        "Blender keeps custom normals through a Subdivision with use_custom_normals=True"
    );

    let cases = [
        ("default", json!({"levels": 1})),
        (
            "explicit_false",
            json!({"levels": 1, "use_custom_normals": false}),
        ),
    ];
    for (name, params) in cases {
        let (project, _) = import_blend(&blender, root.path(), name, &blend)?;
        let node = node_id(&read_scene(&project)?, "Regress")?;
        let applied = apply_operations(
            &project,
            root.path(),
            json!([
                {"op": "modifier.create", "target": {"id": node}, "id": "subdivide", "type": "subdivision", "params": params},
                {"op": "modifier.apply", "target": {"id": node}, "modifier_id": "subdivide"}
            ]),
        )?;
        assert_eq!(applied["ok"], true, "{applied}");
        let scene = read_scene(&project)?;
        assert!(
            !has_normal_code_values(mesh_of_node(&scene, "Regress")?),
            "{name}: Potter kept allocated custom normal values after a Subdivision without use_custom_normals"
        );
    }
    Ok(())
}
