use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, anyhow};
use jcode_sdk::SessionToolDefinition;
use potter_core::cli::{
    ApplyArgs, Context, InspectArgs, PreviewArgs, PreviewMode, ScenePath, SchemaArgs, SchemaKind,
    UndoArgs,
};
use potter_core::error::{PotError, Result as PotterResult};
use potter_core::response::{Envelope, SceneInfo};
use serde_json::{Value, json};

/// Initializes a scene through `pot init <scene>`.
///
/// # Errors
///
/// Returns an error if Potter refuses to initialize the path, for example because it already
/// contains a project.
pub fn init_scene(scene: &Path) -> anyhow::Result<()> {
    potter_core::commands::init::run(ScenePath {
        scene: scene.to_path_buf(),
    })
    .map(|_| ())
    .with_context(|| format!("initializing scene at {}", scene.display()))?;
    Ok(())
}

pub struct RenderOutput {
    /// Rendered PNG paths, in requested view order.
    pub pngs: Vec<PathBuf>,
    /// Deduplicated warnings reported by Potter.
    pub warnings: Vec<Value>,
}

/// Renders the requested views with Potter's deterministic CPU preview renderer.
///
/// # Errors
///
/// Returns an error if the output directory cannot be created, Potter cannot render the scene, or
/// a requested view has no existing PNG in the command result.
pub fn render_views(
    scene: &Path,
    out_dir: &Path,
    views: &[String],
    mode: PreviewMode,
    size: u32,
) -> anyhow::Result<RenderOutput> {
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("creating preview directory {}", out_dir.display()))?;
    let (_, result) = potter_core::commands::preview::run(PreviewArgs {
        scene: scene.to_path_buf(),
        views: Some(views.join(",")),
        camera: None,
        mode,
        size,
        out: Some(out_dir.to_path_buf()),
        overwrite: true,
        context: Context {
            scene_id: None,
            view_layer: None,
            frame: None,
        },
    })
    .map_err(anyhow::Error::from)?;
    let previews = result
        .get("previews")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("pot preview result is missing its previews array"))?;
    let mut pngs = Vec::with_capacity(views.len());
    for requested_view in views {
        let preview = previews
            .iter()
            .find(|preview| {
                preview.get("view").and_then(Value::as_str) == Some(requested_view.as_str())
            })
            .ok_or_else(|| anyhow!("pot preview returned no PNG for view `{requested_view}`"))?;
        let image_path = preview
            .get("image")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .ok_or_else(|| {
                anyhow!("pot preview returned no PNG path for view `{requested_view}`")
            })?;
        if !image_path.is_file() {
            return Err(anyhow!(
                "pot preview PNG for view `{requested_view}` does not exist: {}",
                image_path.display()
            ));
        }
        pngs.push(image_path);
    }
    let warnings = match result.get("warnings").and_then(Value::as_array) {
        Some(warnings) => warnings.clone(),
        None => Vec::new(),
    };
    Ok(RenderOutput { pngs, warnings })
}

/// A mesh part connected neither to the ground plane (z = 0) nor to any part that is; `gap` is its
/// distance in meters to that grounded assembly.
#[derive(Debug, PartialEq, serde::Serialize)]
pub struct FloatingPart {
    pub id: String,
    pub gap: f64,
}

/// Lists mesh parts disconnected from the grounded model by comparing the world bounds reported by
/// `pot inspect`. Bounds closer than 0.5% of the model's bounding diagonal count as touching.
///
/// # Errors
///
/// Returns an error if Potter cannot inspect the scene.
pub fn floating_parts(scene: &Path) -> anyhow::Result<Vec<FloatingPart>> {
    let (_, result) = potter_core::commands::inspect::run(InspectArgs {
        scene: scene.to_path_buf(),
        id: None,
        tag: None,
        features: false,
        context: Context {
            scene_id: None,
            view_layer: None,
            frame: None,
        },
    })
    .map_err(anyhow::Error::from)?;
    let parts: Vec<Part> = result
        .get("items")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| item.get("kind").and_then(Value::as_str) == Some("mesh"))
        // The preview does not render hidden nodes, so they neither float nor ground anything.
        .filter(|item| item.get("visible").and_then(Value::as_bool) != Some(false))
        .filter_map(|item| {
            Some(Part {
                id: item.get("id")?.as_str()?.to_owned(),
                min: vec3(item.pointer("/bounds/min")?)?,
                max: vec3(item.pointer("/bounds/max")?)?,
            })
        })
        .collect();
    Ok(disconnected_parts(&parts))
}

struct Part {
    id: String,
    min: [f64; 3],
    max: [f64; 3],
}

fn vec3(value: &Value) -> Option<[f64; 3]> {
    let [x, y, z] = value.as_array()?.as_slice() else {
        return None;
    };
    Some([x.as_f64()?, y.as_f64()?, z.as_f64()?])
}

fn length(vector: [f64; 3]) -> f64 {
    vector
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

fn gap_between(a: &Part, b: &Part) -> f64 {
    length(std::array::from_fn(|axis| {
        (a.min[axis] - b.max[axis])
            .max(b.min[axis] - a.max[axis])
            .max(0.0)
    }))
}

fn gap_to_ground(part: &Part) -> f64 {
    part.min[2].max(-part.max[2]).max(0.0)
}

fn disconnected_parts(parts: &[Part]) -> Vec<FloatingPart> {
    let Some(first) = parts.first() else {
        return Vec::new();
    };
    let (low, high) = parts
        .iter()
        .fold((first.min, first.max), |(low, high), part| {
            (
                std::array::from_fn(|axis| low[axis].min(part.min[axis])),
                std::array::from_fn(|axis| high[axis].max(part.max[axis])),
            )
        });
    let tolerance = 0.005 * length(std::array::from_fn(|axis| high[axis] - low[axis]));

    let mut grounded: Vec<bool> = parts
        .iter()
        .map(|part| gap_to_ground(part) <= tolerance)
        .collect();
    while let Some(index) = (0..parts.len()).find(|&index| {
        !grounded[index]
            && parts
                .iter()
                .zip(&grounded)
                .any(|(other, &done)| done && gap_between(&parts[index], other) <= tolerance)
    }) {
        grounded[index] = true;
    }

    parts
        .iter()
        .zip(&grounded)
        .filter(|&(_, &done)| !done)
        .map(|(part, _)| {
            let gap = parts
                .iter()
                .zip(&grounded)
                .filter(|&(_, &done)| done)
                .map(|(other, _)| gap_between(part, other))
                .fold(gap_to_ground(part), f64::min);
            FloatingPart {
                id: part.id.clone(),
                gap: (gap * 1000.0).round() / 1000.0,
            }
        })
        .collect()
}

pub struct ToolOutcome {
    pub output: String,
    pub error: Option<String>,
}

/// Returns custom jcode session tools for the modeler agent.
///
/// # Errors
///
/// Returns an error if a JSON Schema cannot be converted to the jcode tool parameter type.
pub fn tool_definitions() -> anyhow::Result<Vec<SessionToolDefinition>> {
    let tools = [
        (
            "pot_apply",
            "Apply a Potter operation batch using `pot apply`. Returns the Potter JSON envelope.",
            json!({
                "type":"object",
                "properties":{"batch":{"type":"object","description":"The complete Potter operation batch, including schema_version, base_revision, and operations."}},
                "required":["batch"],
                "additionalProperties":false
            }),
        ),
        (
            "pot_inspect",
            "Inspect a Potter scene using `pot inspect`. Returns the Potter JSON envelope.",
            json!({
                "type":"object",
                "properties":{
                    "id":{"type":"string","description":"Inspect one entity by ID. Omit to inspect the whole scene."},
                    "tag":{"type":"string","description":"Inspect entities with this tag. Do not combine with id."}
                },
                "additionalProperties":false
            }),
        ),
        (
            "pot_schema",
            "Read an operation schema using `pot schema --op`. Returns the Potter JSON envelope.",
            json!({
                "type":"object",
                "properties":{"op":{"type":"string","description":"Operation kind, such as node.create."}},
                "required":["op"],
                "additionalProperties":false
            }),
        ),
        (
            "pot_undo",
            "Undo scene history using `pot undo`. Returns the Potter JSON envelope.",
            json!({
                "type":"object",
                "properties":{
                    "base_revision":{"type":"integer","minimum":0},
                    "steps":{"type":"integer","minimum":1}
                },
                "required":["base_revision"],
                "additionalProperties":false
            }),
        ),
    ];
    tools
        .into_iter()
        .map(|(name, description, parameters)| {
            Ok(SessionToolDefinition {
                name: name.to_owned(),
                description: description.to_owned(),
                parameters: serde_json::from_value(parameters)?,
            })
        })
        .collect()
}

/// Runs one custom tool call against a scene; malformed input and Potter errors are reported without panicking.
pub fn execute_tool(scene: &Path, name: &str, input: &Value) -> ToolOutcome {
    match name {
        "pot_apply" => execute_apply(scene, input),
        "pot_inspect" => execute_inspect(scene, input),
        "pot_schema" => execute_schema(input),
        "pot_undo" => execute_undo(scene, input),
        _ => invalid_input(format!("unknown Potter tool `{name}`")),
    }
}

fn execute_apply(scene: &Path, input: &Value) -> ToolOutcome {
    let batch = match object_field(input, "batch", &["batch"]) {
        Ok(object) => &object["batch"],
        Err(message) => return invalid_input(message),
    };
    if !batch.is_object() {
        return invalid_input("`batch` must be an object".to_owned());
    }
    let result = (|| {
        let mut file = tempfile::NamedTempFile::new().map_err(|error| PotError::io(&error))?;
        let batch_bytes = serde_json::to_vec(batch).map_err(PotError::internal_json)?;
        file.write_all(&batch_bytes)
            .map_err(|error| PotError::io(&error))?;
        potter_core::commands::apply::run(ApplyArgs {
            scene: scene.to_path_buf(),
            file: file.path().to_path_buf(),
            preview: None,
            size: None,
            dry_run: false,
        })
    })();
    tool_outcome("apply", result)
}

fn execute_inspect(scene: &Path, input: &Value) -> ToolOutcome {
    let object = match input_object(input, &["id", "tag"]) {
        Ok(object) => object,
        Err(message) => return invalid_input(message),
    };
    let id = match optional_string(object, "id") {
        Ok(value) => value,
        Err(message) => return invalid_input(message),
    };
    let tag = match optional_string(object, "tag") {
        Ok(value) => value,
        Err(message) => return invalid_input(message),
    };
    if id.is_some() && tag.is_some() {
        return invalid_input("`id` and `tag` cannot both be provided".to_owned());
    }
    tool_outcome(
        "inspect",
        potter_core::commands::inspect::run(InspectArgs {
            scene: scene.to_path_buf(),
            id,
            tag,
            features: false,
            context: Context {
                scene_id: None,
                view_layer: None,
                frame: None,
            },
        }),
    )
}

fn execute_schema(input: &Value) -> ToolOutcome {
    let object = match object_field(input, "op", &["op"]) {
        Ok(object) => object,
        Err(message) => return invalid_input(message),
    };
    let operation = match object.get("op").and_then(Value::as_str) {
        Some(operation) => operation.to_owned(),
        None => return invalid_input("`op` must be a string".to_owned()),
    };
    tool_outcome(
        "schema",
        potter_core::commands::schema::run(&SchemaArgs {
            kind: SchemaKind::Operations,
            op: Some(operation),
        }),
    )
}

fn execute_undo(scene: &Path, input: &Value) -> ToolOutcome {
    let object = match object_field(input, "base_revision", &["base_revision", "steps"]) {
        Ok(object) => object,
        Err(message) => return invalid_input(message),
    };
    let Some(base_revision) = object.get("base_revision").and_then(Value::as_u64) else {
        return invalid_input("`base_revision` must be a non-negative integer".to_owned());
    };
    // OpenAI-style strict tool calls send `null` for omitted optional fields.
    let steps = match object.get("steps").filter(|value| !value.is_null()) {
        Some(value) => match value.as_u64().and_then(|value| usize::try_from(value).ok()) {
            Some(steps) if steps >= 1 => steps,
            _ => return invalid_input("`steps` must be an integer of at least 1".to_owned()),
        },
        None => 1,
    };
    tool_outcome(
        "undo",
        potter_core::commands::undo::run(UndoArgs {
            scene: scene.to_path_buf(),
            base_revision,
            steps,
        }),
    )
}

fn input_object<'a>(
    input: &'a Value,
    allowed: &[&str],
) -> Result<&'a serde_json::Map<String, Value>, String> {
    let object = input
        .as_object()
        .ok_or_else(|| "tool input must be an object".to_owned())?;
    if let Some(unknown) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("unknown tool input field `{unknown}`"));
    }
    Ok(object)
}

fn object_field<'a>(
    input: &'a Value,
    required: &str,
    allowed: &[&str],
) -> Result<&'a serde_json::Map<String, Value>, String> {
    let object = input_object(input, allowed)?;
    if !object.contains_key(required) {
        return Err(format!("missing required tool input field `{required}`"));
    }
    Ok(object)
}

fn optional_string(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<String>, String> {
    match object.get(field).filter(|value| !value.is_null()) {
        Some(value) => value
            .as_str()
            .map(|text| Some(text.to_owned()))
            .ok_or_else(|| format!("`{field}` must be a string")),
        None => Ok(None),
    }
}

fn invalid_input(message: String) -> ToolOutcome {
    ToolOutcome {
        output: message.clone(),
        error: Some(message),
    }
}

fn tool_outcome(command: &str, result: PotterResult<(Option<SceneInfo>, Value)>) -> ToolOutcome {
    match result {
        Ok((scene, result)) => serialize_envelope(&Envelope::success(command, scene, result), None),
        Err(error) => {
            let message = error.to_string();
            serialize_envelope(
                &Envelope::failure(Some(command.to_owned()), error),
                Some(message),
            )
        }
    }
}

fn serialize_envelope(envelope: &Envelope, error: Option<String>) -> ToolOutcome {
    match serde_json::to_string(&envelope) {
        Ok(output) => ToolOutcome { output, error },
        Err(serialization_error) => {
            let message = serialization_error.to_string();
            ToolOutcome {
                output: message.clone(),
                error: Some(message),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use anyhow::Result;
    use potter_core::response::Envelope;
    use serde_json::{Value, json};

    use super::{
        FloatingPart, PreviewMode, execute_tool, floating_parts, init_scene, render_views,
    };

    fn box_batch(base_revision: u64) -> Value {
        json!({
            "schema_version": 1,
            "base_revision": base_revision,
            "operations": [{
                "op": "node.create",
                "id": "box",
                "kind": "box",
                "params": {"size": 2.0}
            }]
        })
    }

    fn decode_envelope(output: &str) -> Result<Envelope> {
        Ok(serde_json::from_str(output)?)
    }

    #[test]
    fn floating_parts_reports_only_parts_detached_from_the_grounded_assembly() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let scene = directory.path().join("scene");
        init_scene(&scene)?;
        let part = |id: &str, translation: [f64; 3], scale: [f64; 3]| {
            json!({"op": "node.create", "id": id, "kind": "box", "params": {"size": 1},
                   "transform": {"translation": translation, "scale": scale}})
        };
        let batch = json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                part("top", [0.0, 0.0, 0.725], [2.0, 1.0, 0.05]),
                part("leg", [-0.925, -0.425, 0.35], [0.05, 0.05, 0.7]),
                part("cube", [0.1, 0.1, 0.85], [0.2, 0.2, 0.2]),
                part("rail", [0.0, -0.475, 0.625], [1.6, 0.05, 0.05]),
                // Hidden like a modeling helper: not rendered, so not reported.
                json!({"op": "node.create", "id": "hidden_helper", "kind": "box",
                       "params": {"size": 1}, "visible": false,
                       "transform": {"translation": [0.0, 0.0, 2.0], "scale": [0.1, 0.1, 0.1]}}),
            ]
        });
        let applied = execute_tool(&scene, "pot_apply", &json!({"batch": batch}));
        assert!(applied.error.is_none(), "{}", applied.output);

        assert_eq!(
            floating_parts(&scene)?,
            [FloatingPart {
                id: "rail".to_owned(),
                gap: 0.05
            }]
        );
        Ok(())
    }

    #[test]
    fn apply_creates_box_and_inspect_finds_it() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let scene = directory.path().join("scene");
        init_scene(&scene)?;

        let applied = execute_tool(&scene, "pot_apply", &json!({"batch":box_batch(0)}));
        assert!(applied.error.is_none(), "{}", applied.output);
        let envelope = decode_envelope(&applied.output)?;
        assert!(envelope.ok);
        assert_eq!(envelope.scene.as_ref().map(|info| info.revision), Some(1));

        let inspected = execute_tool(&scene, "pot_inspect", &json!({"id":"box"}));
        assert!(inspected.error.is_none(), "{}", inspected.output);
        let envelope = decode_envelope(&inspected.output)?;
        assert!(envelope.ok);
        let items = envelope.result.get("items").and_then(Value::as_array);
        assert!(items.is_some_and(|items| items.iter().any(|item| item["id"] == "box")));
        Ok(())
    }

    #[test]
    fn apply_revision_conflict_returns_failure_envelope() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let scene = directory.path().join("scene");
        init_scene(&scene)?;

        let applied = execute_tool(&scene, "pot_apply", &json!({"batch":box_batch(3)}));
        assert!(
            applied
                .error
                .as_deref()
                .is_some_and(|message| message.contains("REVISION_CONFLICT"))
        );
        let envelope = decode_envelope(&applied.output)?;
        assert!(!envelope.ok);
        Ok(())
    }

    #[test]
    fn unknown_tool_and_invalid_input_are_reported() {
        let unknown = execute_tool(Path::new("unused"), "pot_unknown", &json!({}));
        assert!(unknown.error.is_some());
        assert!(unknown.output.contains("unknown Potter tool"));

        let invalid = execute_tool(Path::new("unused"), "pot_inspect", &json!({"id":7}));
        assert!(invalid.error.is_some());
        assert!(invalid.output.contains("`id` must be a string"));
    }

    #[test]
    fn null_optional_fields_count_as_omitted() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let scene = directory.path().join("scene");
        init_scene(&scene)?;
        let applied = execute_tool(&scene, "pot_apply", &json!({"batch":box_batch(0)}));
        assert!(applied.error.is_none(), "{}", applied.output);

        let inspected = execute_tool(&scene, "pot_inspect", &json!({"id":"box","tag":null}));
        assert!(inspected.error.is_none(), "{}", inspected.output);
        let envelope = decode_envelope(&inspected.output)?;
        let items = envelope.result.get("items").and_then(Value::as_array);
        assert!(items.is_some_and(|items| items.iter().any(|item| item["id"] == "box")));

        let undone = execute_tool(&scene, "pot_undo", &json!({"base_revision":1,"steps":null}));
        assert!(undone.error.is_none(), "{}", undone.output);
        let envelope = decode_envelope(&undone.output)?;
        assert_eq!(envelope.scene.as_ref().map(|info| info.revision), Some(2));
        Ok(())
    }

    #[test]
    fn renders_requested_views_and_creates_png_files() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let scene = directory.path().join("scene");
        let output = directory.path().join("previews");
        init_scene(&scene)?;
        let applied = execute_tool(&scene, "pot_apply", &json!({"batch":box_batch(0)}));
        assert!(applied.error.is_none(), "{}", applied.output);

        let views = vec!["front".to_owned(), "iso".to_owned()];
        let rendered = render_views(&scene, &output, &views, PreviewMode::Solid, 64)?;
        assert_eq!(rendered.pngs.len(), 2);
        assert!(rendered.pngs[0].ends_with("front.png"));
        assert!(rendered.pngs[1].ends_with("iso.png"));
        assert!(rendered.pngs.iter().all(|path| path.is_file()));
        Ok(())
    }
}
