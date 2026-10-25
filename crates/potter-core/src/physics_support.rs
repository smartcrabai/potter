//! Blender 5.2.2 physics settings contract shared by evaluation, ops, and the `.blend` loss catalog.
//!
//! Defaults come from `physics_defaults.json`, which `tests/fixtures/physics_defaults/generate.py`
//! reads from a headless Blender 5.2.2. A Blender property that is not listed as honoured for its
//! system is only accepted while it stays at its Blender default; any other value is an
//! `UnsupportedFeature` rejection that names `physics.<system>.<property>`.

use std::sync::LazyLock;

use serde_json::{Map, Value, json};

use std::collections::BTreeSet;

use crate::error::{ErrorCode, PotError, Result};
use crate::model::SceneDoc;

/// Physics settings blocks Potter evaluates or reports. Names are Blender 5.2.2 RNA struct roles.
///
/// `fluid_domain` is the GAS table and `fluid_domain_liquid` the LIQUID table, because Blender's
/// domain defaults depend on `domain_type`. Use [`fluid_domain_system`] to pick one.
/// The Blender physics block nested under `key` of a `system` block (`collision_settings` of cloth,
/// the surfaces of a canvas, and any `point_cache`), named by the system it holds.
pub(crate) fn nested_physics_system(system: &str, key: &str) -> Option<&'static str> {
    match (system, key) {
        ("cloth", "collision_settings") => Some("cloth_collision"),
        ("dynamic_paint_canvas", "canvas_surfaces") => Some("dynamic_paint_surface"),
        (_, "point_cache") => Some("point_cache"),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) const PHYSICS_SYSTEMS: [&str; 12] = [
    "cloth",
    "cloth_collision",
    "soft_body",
    "collision",
    "fluid_domain",
    "fluid_domain_liquid",
    "fluid_flow",
    "fluid_effector",
    "dynamic_paint_canvas",
    "dynamic_paint_surface",
    "dynamic_paint_brush",
    "point_cache",
];

/// Collision properties that point-sampled consumers (particles and liquid) honour.
///
/// Cloth and soft bodies honour [`honoured_properties`] for `collision`, which adds the culling and
/// normal properties their solvers implement. `cloth_friction` is cloth's own friction
/// (`collision.cc`, `pdef_cfrict`) and `thickness_inner` is the soft body's inner thickness
/// (`softbody.cc`, `pdef_sbift`), so a particle system has no use for either.
pub(crate) const POINT_COLLISION_HONOURED: [&str; 4] = [
    "use",
    "thickness_outer",
    "cloth_friction",
    "thickness_inner",
];

static DEFAULTS: LazyLock<Value> =
    LazyLock::new(
        || match serde_json::from_str(include_str!("physics_defaults.json")) {
            Ok(value) => value,
            Err(error) => panic!("invalid compiled Blender physics defaults: {error}"),
        },
    );

/// RNA properties of `system` that Potter evaluates with Blender's meaning.
///
/// Anything else that Blender defines for `system` must stay at its Blender default, or the
/// setting is rejected with [`reject_unhonoured`].
pub(crate) fn honoured_properties(system: &str) -> &'static [&'static str] {
    match system {
        // `quality` sets the fixed substep count per frame. `iterations` is a Potter-native
        // solver knob, not a Blender property, so it is not listed.
        "cloth" => &[
            "quality",
            "mass",
            "air_damping",
            "vertex_group_mass",
            "goal_default",
            "goal_spring",
            "pin_stiffness",
            "goal_friction",
            "tension_stiffness",
            "compression_stiffness",
            "shear_stiffness",
            "bending_stiffness",
            "bending_model",
        ],
        "cloth_collision" => &[
            "use_collision",
            "distance_min",
            "use_self_collision",
            "self_distance_min",
            "collection",
        ],
        "soft_body" => &[
            "mass",
            "friction",
            "goal_spring",
            "goal_default",
            "use_goal",
            "goal_min",
            "goal_max",
            "vertex_group_goal",
            "use_edges",
            "speed",
            "collision_collection",
        ],
        "collision" => &[
            "use",
            "thickness_outer",
            "thickness_inner",
            "use_culling",
            "cloth_friction",
        ],
        "fluid_domain" | "fluid_domain_liquid" => &[
            "domain_type",
            "particle_radius",
            "resolution_max",
            "cache_frame_start",
            "cache_frame_end",
            "effector_group",
        ],
        "fluid_flow" => &["flow_type", "flow_behavior", "flow_source", "density"],
        "fluid_effector" => &["effector_type", "use_effector"],
        "point_cache" => &["frame_start", "frame_end"],
        "dynamic_paint_surface" => &[
            "surface_type",
            "surface_format",
            "frame_start",
            "frame_end",
            "output_name_a",
            "use_output_a",
        ],
        "dynamic_paint_brush" => &["paint_color", "paint_alpha", "paint_distance"],
        _ => &[],
    }
}

/// Blender 5.2.2 default for `property` of `system`, as generated from a fresh instance.
pub(crate) fn blender_default(system: &str, property: &str) -> Option<&'static Value> {
    DEFAULTS.get("systems")?.get(system)?.get(property)
}

/// Blender 5.2.2 property names and defaults of `system`, or `None` when the system is not generated.
pub(crate) fn blender_properties(system: &str) -> Option<&'static Map<String, Value>> {
    DEFAULTS.get("systems")?.get(system)?.as_object()
}

/// True when `property` is a Blender 5.2.2 writable RNA property of `system`.
pub(crate) fn is_blender_property(system: &str, property: &str) -> bool {
    blender_default(system, property).is_some()
}

/// Defaults table of a fluid domain with `domain_type`.
///
/// Blender's domain defaults depend on the type, so LIQUID uses `fluid_domain_liquid`. GAS uses
/// `fluid_domain`. Any other value also maps to `fluid_domain`, so callers must validate
/// `domain_type` before they look up defaults.
pub(crate) fn fluid_domain_system(domain_type: &str) -> &'static str {
    if domain_type == "LIQUID" {
        "fluid_domain_liquid"
    } else {
        "fluid_domain"
    }
}

/// Blender properties that hold runtime or host state, not user settings.
///
/// These are never reported and never honoured. Soft-body estimates are recomputed each evaluation.
/// Fluid cache flags describe bake state on disk. Paths, cache indices, and names identify the
/// file system or datablock.
pub(crate) fn is_runtime_state(system: &str, property: &str) -> bool {
    match system {
        "soft_body" => {
            property.ends_with("_estimate")
                || matches!(property, "use_estimate_matrix" | "location_mass_center")
        }
        "fluid_domain" | "fluid_domain_liquid" => {
            property == "cache_directory"
                || property.starts_with("has_cache_baked_")
                || property.starts_with("is_cache_baking_")
        }
        "dynamic_paint_surface" => matches!(property, "image_output_path" | "name"),
        "point_cache" => matches!(property, "filepath" | "index" | "name" | "use_disk_cache"),
        _ => false,
    }
}

/// Whether a non-default value of `property` changes Potter's result, given its block `setting`.
///
/// False for runtime state (see [`is_runtime_state`]). A derived stiffness clamp takes effect only
/// through its vertex group, so it is semantic only when that group is set. The `canvas_surfaces`
/// list is never semantic at canvas level, because each surface is checked as
/// `dynamic_paint_surface`.
pub(crate) fn is_semantic_when(system: &str, property: &str, setting: &Map<String, Value>) -> bool {
    if is_runtime_state(system, property) {
        return false;
    }
    let group = match (system, property) {
        ("cloth", "tension_stiffness_max" | "compression_stiffness_max") => {
            "vertex_group_structural_stiffness"
        }
        ("cloth", "shear_stiffness_max") => "vertex_group_shear_stiffness",
        ("cloth", "bending_stiffness_max") => "vertex_group_bending",
        ("dynamic_paint_canvas", "canvas_surfaces") => return false,
        _ => return true,
    };
    setting
        .get(group)
        .and_then(Value::as_str)
        .is_some_and(|name| !name.is_empty())
}

/// True when the generated defaults contain `system`, even if it has no writable properties.
#[cfg(test)]
pub(crate) fn has_generated_defaults(system: &str) -> bool {
    DEFAULTS
        .get("systems")
        .and_then(|systems| systems.get(system))
        .is_some()
}

/// True when `value` equals the Blender default of `property` in `system`.
///
/// Numbers compare with a tolerance covering float32 rounding, and nested structs compare key by
/// key, so a key absent from `value` counts as its default.
pub(crate) fn is_default_value(system: &str, property: &str, value: &Value) -> bool {
    blender_default(system, property).is_some_and(|default| same_value(value, default))
}

fn same_value(value: &Value, default: &Value) -> bool {
    match (value, default) {
        (Value::Number(actual), Value::Number(expected)) => {
            match (actual.as_f64(), expected.as_f64()) {
                (Some(actual), Some(expected)) => {
                    (actual - expected).abs() <= 1.0e-6 * (1.0 + actual.abs().max(expected.abs()))
                }
                _ => false,
            }
        }
        (Value::Array(actual), Value::Array(expected)) => {
            actual.len() == expected.len()
                && actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| same_value(actual, expected))
        }
        (Value::Object(actual), Value::Object(expected)) => actual.iter().all(|(key, actual)| {
            expected
                .get(key)
                .is_some_and(|expected| same_value(actual, expected))
        }),
        _ => value == default,
    }
}

/// Rejects every Blender property of `system` that is set to a non-default value and is not in
/// `honoured`. Keys that are not Blender properties of `system` (Potter-native keys) are left to
/// their own validators.
pub(crate) fn reject_unhonoured(system: &str, settings: &Value, honoured: &[&str]) -> Result<()> {
    let Some(settings) = settings.as_object() else {
        return Ok(());
    };
    for (property, value) in settings {
        if !is_blender_property(system, property)
            || honoured.contains(&property.as_str())
            || !is_semantic_when(system, property, settings)
            || is_default_value(system, property, value)
        {
            continue;
        }
        return Err(unsupported_property(system, property));
    }
    Ok(())
}

/// Error for a Blender property that Potter does not model at a non-default value.
pub(crate) fn unsupported_property(system: &str, property: &str) -> PotError {
    let feature_id = format!("physics.{system}.{property}");
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("Blender {system} property `{property}` is not supported at a non-default value"),
        json!({"feature_id":feature_id,"property":property,"status":"not_supported"}),
    )
}

/// Removes the physics settings that an accepted `--allow-lossy` import reported (loss ids
/// `blender.physics.<system>.<property>` in `SceneDoc::accepted_losses`) from a copy of the document,
/// so evaluation does not refuse them. Each removed value gives one warning. `None` when no physics
/// loss was accepted, so the common path does not copy the document.
pub(crate) fn without_accepted_losses(doc: &SceneDoc) -> Option<(SceneDoc, Vec<PotError>)> {
    let accepted: BTreeSet<&str> = doc
        .accepted_losses
        .iter()
        .map(String::as_str)
        .filter(|id| id.starts_with("blender.physics."))
        .collect();
    if accepted.is_empty() {
        return None;
    }
    let mut copy = doc.clone();
    let mut warnings = Vec::new();
    for (node_id, node) in &mut copy.nodes {
        let fluid_type = node
            .modifiers
            .iter()
            .find(|modifier| modifier.modifier_type == "fluid")
            .and_then(|modifier| modifier.params.get("fluid_type"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        for (property, value) in &mut node.properties {
            let Value::Object(block) = value else {
                continue;
            };
            let system = match property.as_str() {
                "physics_cloth" => Some("cloth"),
                "physics_soft_body" => Some("soft_body"),
                "physics_collision" => Some("collision"),
                "physics_dynamic_paint" => match block.get("role").and_then(Value::as_str) {
                    Some("canvas") => Some("dynamic_paint_canvas"),
                    Some("brush") => Some("dynamic_paint_brush"),
                    _ => None,
                },
                "physics_fluid" => match fluid_type.as_deref() {
                    Some("DOMAIN") => Some(fluid_domain_system(
                        block
                            .get("domain_type")
                            .and_then(Value::as_str)
                            .unwrap_or("GAS"),
                    )),
                    Some("FLOW") => Some("fluid_flow"),
                    Some("EFFECTOR") => Some("fluid_effector"),
                    _ => None,
                },
                _ => None,
            };
            if let Some(system) = system {
                strip_accepted_block(block, system, &accepted, node_id.as_str(), &mut warnings);
            }
        }
    }
    Some((copy, warnings))
}

fn strip_accepted_block(
    block: &mut Map<String, Value>,
    system: &str,
    accepted: &BTreeSet<&str>,
    node_id: &str,
    warnings: &mut Vec<PotError>,
) {
    let keys = block.keys().cloned().collect::<Vec<_>>();
    for key in keys {
        if key == "role" {
            continue;
        }
        if let Some(nested) = nested_physics_system(system, &key) {
            match block.get_mut(&key) {
                Some(Value::Object(inner)) => {
                    strip_accepted_block(inner, nested, accepted, node_id, warnings);
                }
                Some(Value::Array(items)) => {
                    for item in items {
                        if let Value::Object(inner) = item {
                            strip_accepted_block(inner, nested, accepted, node_id, warnings);
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        let id = format!("blender.physics.{system}.{key}");
        if accepted.contains(id.as_str()) {
            block.remove(&key);
            warnings.push(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!(
                    "accepted lossy import: Blender {system} property `{key}` is not evaluated and is ignored"
                ),
                json!({"feature_id": id, "node_id": node_id, "status": "accepted_loss"}),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::error::ErrorCode;

    use super::{
        PHYSICS_SYSTEMS, blender_default, fluid_domain_system, has_generated_defaults,
        honoured_properties, is_blender_property, is_default_value, is_runtime_state,
        is_semantic_when, reject_unhonoured,
    };

    #[test]
    fn every_system_has_generated_defaults() {
        for system in PHYSICS_SYSTEMS {
            assert!(
                has_generated_defaults(system),
                "{system} has no generated defaults"
            );
        }
        assert_eq!(
            blender_default("collision", "thickness_outer"),
            Some(&json!(0.02))
        );
        assert!(is_blender_property("cloth", "air_damping"));
        assert!(!is_blender_property("cloth", "stiffness"));
    }

    #[test]
    fn defaults_match_with_float32_tolerance() {
        assert!(is_default_value(
            "collision",
            "thickness_outer",
            &json!(0.019_999_999_552_965_164)
        ));
        assert!(!is_default_value(
            "collision",
            "thickness_outer",
            &json!(0.03)
        ));
        assert!(is_default_value(
            "cloth",
            "bending_model",
            &json!("ANGULAR")
        ));
    }

    #[test]
    fn unhonoured_non_default_property_is_rejected_with_feature_id() {
        let rejected = reject_unhonoured(
            "cloth",
            &json!({"use_pressure": true}),
            honoured_properties("cloth"),
        );
        let Err(error) = rejected else {
            panic!("non-default unhonoured property must be rejected");
        };
        assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(
            error.details["feature_id"],
            json!("physics.cloth.use_pressure")
        );
        assert!(
            reject_unhonoured(
                "cloth",
                &json!({"tension_damping": 5.0, "stiffness": 0.5}),
                honoured_properties("cloth"),
            )
            .is_ok()
        );
    }

    #[test]
    fn liquid_domain_defaults_come_from_the_liquid_table() {
        assert_eq!(fluid_domain_system("LIQUID"), "fluid_domain_liquid");
        assert_eq!(fluid_domain_system("GAS"), "fluid_domain");
        assert!(is_default_value(
            "fluid_domain_liquid",
            "use_collision_border_back",
            &json!(true)
        ));
        assert!(is_default_value(
            "fluid_domain",
            "use_collision_border_back",
            &json!(false)
        ));
    }

    #[test]
    fn runtime_state_is_never_reported() {
        let estimate = json!([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert!(is_runtime_state("soft_body", "rotation_estimate"));
        assert!(is_runtime_state("soft_body", "location_estimate"));
        assert!(
            reject_unhonoured(
                "soft_body",
                &json!({"rotation_estimate": estimate, "use_estimate_matrix": true}),
                honoured_properties("soft_body"),
            )
            .is_ok()
        );
        assert!(is_runtime_state("fluid_domain_liquid", "cache_directory"));
    }

    #[test]
    fn derived_stiffness_clamp_is_semantic_only_with_its_vertex_group() {
        let without_group = json!({"tension_stiffness_max": 5.0});
        let with_group = json!({
            "tension_stiffness_max": 5.0,
            "vertex_group_structural_stiffness": "struct"
        });
        let (Some(without), Some(with)) = (without_group.as_object(), with_group.as_object())
        else {
            panic!("test settings must be objects");
        };
        assert!(!is_semantic_when("cloth", "tension_stiffness_max", without));
        assert!(is_semantic_when("cloth", "tension_stiffness_max", with));
        assert!(reject_unhonoured("cloth", &without_group, honoured_properties("cloth")).is_ok());
        assert!(reject_unhonoured("cloth", &with_group, honoured_properties("cloth")).is_err());
    }

    #[test]
    fn canvas_surface_list_is_checked_per_surface_not_at_canvas_level() {
        assert!(!is_semantic_when(
            "dynamic_paint_canvas",
            "canvas_surfaces",
            &serde_json::Map::new()
        ));
    }
}
