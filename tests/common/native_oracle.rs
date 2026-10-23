//! Shared helpers for the PR9 native-data oracle tests.
//!
//! The oracle is built by `tests/fixtures/native_oracle/build_oracle.py` inside Blender 5.2.2. That
//! script writes the explicit numbers the tests compare against (`expected.json`); Potter's output
//! is never copied into an expectation. `dump_blend.py` reads any `.blend` back with Blender's own
//! API. Tests skip when Blender is absent, and fail when `POTTER_REQUIRE_BLENDER` is set.

use std::{
    collections::BTreeSet,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::Value;

pub type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Float32 storage in Blender and Potter's f64 store: values agree to this bound.
pub const FLOAT_TOLERANCE: f64 = 1.0e-6;
/// `BYTE_COLOR` layers: Blender stores bytes, so one step is the bound.
pub const BYTE_TOLERANCE: f64 = 1.0 / 255.0;
/// Blender quantizes custom normals to 16-bit codes; the oracle's own check uses 1e-3.
/// Same value as `exchange_normals::BLENDER_NORMAL_TOLERANCE`; the native and exchange test roots
/// share no module, so the number is repeated here.
pub const NORMAL_TOLERANCE: f64 = 2.0e-4;

const BUILD_SCRIPT: &str = include_str!("../fixtures/native_oracle/build_oracle.py");
const DUMP_SCRIPT: &str = include_str!("../fixtures/native_oracle/dump_blend.py");
const RENDER_SCRIPT: &str = include_str!("../fixtures/native_oracle/render_texture.py");

pub fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

pub fn checked(command: &mut Command, label: &str) -> TestResult<Output> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(output)
}

/// Runs `pot` with the arguments added by `configure`, appends `--json`, and returns the envelope.
pub fn pot_json(configure: impl FnOnce(&mut Command)) -> TestResult<Value> {
    let mut command = pot();
    configure(&mut command);
    command.arg("--json");
    let output = checked(&mut command, "pot")?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Runs `pot export` without asserting success, so callers can observe the loss gate.
pub fn export_raw(
    blender: &Path,
    project: &Path,
    format: &str,
    out: &Path,
    allow_lossy: bool,
) -> TestResult<Output> {
    let mut command = pot();
    command
        .arg("export")
        .arg(project)
        .args(["--format", format, "--out"])
        .arg(out)
        .arg("--blender")
        .arg(blender)
        .arg("--json");
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    Ok(command.output()?)
}

/// Exports `project` to `out` in `format`, asserting success, and returns the envelope.
pub fn export(blender: &Path, project: &Path, format: &str, out: &Path) -> TestResult<Value> {
    let output = export_raw(blender, project, format, out, false)?;
    if !output.status.success() {
        return Err(format!(
            "strict {format} export failed: {}",
            String::from_utf8_lossy(&output.stdout)
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// The `losses` array of an import or export envelope (`result.losses`).
pub fn losses(report: &Value) -> Vec<&Value> {
    report["result"]["losses"]
        .as_array()
        .map_or_else(Vec::new, |losses| losses.iter().collect())
}

/// The feature ids of the losses in `report`.
pub fn loss_features(report: &Value) -> Vec<String> {
    losses(report)
        .into_iter()
        .filter_map(|loss| loss["feature_id"].as_str().map(str::to_owned))
        .collect()
}

pub fn write_script(root: &Path, name: &str, source: &str) -> TestResult<PathBuf> {
    let path = root.join(name);
    fs::write(&path, source)?;
    Ok(path)
}

/// Runs Blender in the background. `file` is opened first when given; `arguments` follow `--`.
pub fn run_blender_arguments(
    blender: &Path,
    script: &Path,
    file: Option<&Path>,
    arguments: &[&Path],
) -> TestResult {
    let mut command = Command::new(blender);
    command.args([
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "3",
    ]);
    if let Some(file) = file {
        command.arg(file);
    }
    command.arg("--python").arg(script).arg("--");
    for argument in arguments {
        command.arg(argument);
    }
    checked(&mut command, "Blender")?;
    Ok(())
}

/// Runs Blender in the background. `file` is opened first when given; `argument` follows `--`.
pub fn run_blender(
    blender: &Path,
    script: &Path,
    file: Option<&Path>,
    argument: &Path,
) -> TestResult {
    run_blender_arguments(blender, script, file, &[argument])
}

/// Renders the texture plane of the `.blend` at `blend` (see `render_texture.py`). The render's
/// own EXR is written beside the JSON. Returns the JSON: the image's pixels and the render, both
/// bottom row first.
pub fn render_texture(blender: &Path, root: &Path, blend: &Path, name: &str) -> TestResult<Value> {
    let script = write_script(root, "render_texture.py", RENDER_SCRIPT)?;
    let json_path = root.join(format!("{name}.json"));
    let exr_path = root.join(format!("{name}.exr"));
    run_blender_arguments(blender, &script, Some(blend), &[&json_path, &exr_path])?;
    Ok(serde_json::from_slice(&fs::read(json_path)?)?)
}

/// Builds the oracle under `root/oracle`. Returns the `.blend` path and the expected numbers.
pub fn build_oracle(blender: &Path, root: &Path) -> TestResult<(PathBuf, Value)> {
    let script = write_script(root, "build_oracle.py", BUILD_SCRIPT)?;
    let out = root.join("oracle");
    run_blender(blender, &script, None, &out)?;
    let expected = serde_json::from_slice(&fs::read(out.join("expected.json"))?)?;
    Ok((out.join("oracle.blend"), expected))
}

/// Opens `blend` in a fresh Blender session and returns Blender's dump of every mesh object.
pub fn dump_blend(blender: &Path, root: &Path, blend: &Path, name: &str) -> TestResult<Value> {
    let script = write_script(root, "dump_blend.py", DUMP_SCRIPT)?;
    let out = root.join(format!("{name}.json"));
    run_blender(blender, &script, Some(blend), &out)?;
    Ok(serde_json::from_slice(&fs::read(out)?)?)
}

/// Initializes a project at `root/name` and replaces its contents with the `.blend` at `source`.
/// Returns the project path and the import envelope.
pub fn import_blend(
    blender: &Path,
    root: &Path,
    name: &str,
    source: &Path,
) -> TestResult<(PathBuf, Value)> {
    let project = root.join(name);
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    let report = pot_json(|command| {
        command
            .arg("import")
            .arg(&project)
            .arg("--file")
            .arg(source)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(blender);
    })?;
    Ok((project, report))
}

pub fn read_scene(project: &Path) -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(
        project.join("scene.json"),
    )?)?)
}

/// The ID of the node whose Blender name is `name`.
pub fn node_id(scene: &Value, name: &str) -> TestResult<String> {
    scene["nodes"]
        .as_object()
        .ok_or("scene nodes are not an object")?
        .iter()
        .find(|(_, node)| node["name"].as_str() == Some(name))
        .map(|(id, _)| id.clone())
        .ok_or_else(|| format!("no node named {name}").into())
}

/// The mesh of the node named `name`.
pub fn mesh_of_node<'a>(scene: &'a Value, name: &str) -> TestResult<&'a Value> {
    let id = node_id(scene, name)?;
    let data = scene["nodes"][&id]["data"]
        .as_str()
        .ok_or("node has no data block")?;
    let mesh = &scene["data_blocks"][data]["mesh"];
    if mesh.is_object() {
        Ok(mesh)
    } else {
        Err(format!("node {name} is not a mesh").into())
    }
}

/// A mesh attribute by name (`mesh.attributes[name]`).
pub fn attribute<'a>(mesh: &'a Value, name: &str) -> TestResult<&'a Value> {
    let value = &mesh["attributes"][name];
    if value.is_null() {
        Err(format!("mesh attribute {name} is missing").into())
    } else {
        Ok(value)
    }
}

/// One entry of an attribute's `values` object.
pub fn value_at<'a>(attribute: &'a Value, key: &str) -> TestResult<&'a Value> {
    attribute["values"]
        .get(key)
        .ok_or_else(|| format!("attribute value {key} is missing").into())
}

/// One corner of a stored `normal_code` layer (`custom_normal.values`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredCorner {
    /// `[0, 0]` (or `null` from older output): Auto, Blender's "use the loop's normal space" code.
    Auto,
    /// `[i16, i16]`: an authored, nonzero Blender fan-space code pair.
    Code([i64; 2]),
    /// `"explicit_zero"`: a literal zero vector, distinct from Auto.
    ExplicitZero,
}

/// A Blender code pair (`[i16, i16]`) from JSON.
pub fn code_pair(value: &Value) -> TestResult<[i64; 2]> {
    let items = value
        .as_array()
        .filter(|items| items.len() == 2)
        .ok_or_else(|| format!("code pair {value} must hold two integers"))?;
    Ok([code_component(&items[0])?, code_component(&items[1])?])
}

fn code_component(value: &Value) -> TestResult<i64> {
    let number = value
        .as_i64()
        .ok_or_else(|| format!("code component {value} is not an integer"))?;
    if i16::try_from(number).is_err() {
        return Err(format!("code component {number} is outside i16").into());
    }
    Ok(number)
}

fn stored_corner(value: &Value) -> TestResult<StoredCorner> {
    match value {
        Value::Null => Ok(StoredCorner::Auto),
        Value::String(name) if name == "explicit_zero" => Ok(StoredCorner::ExplicitZero),
        Value::Array(_) => {
            let code = code_pair(value)?;
            Ok(if code == [0, 0] {
                StoredCorner::Auto
            } else {
                StoredCorner::Code(code)
            })
        }
        other => Err(format!("unexpected stored corner {other}").into()),
    }
}

/// Whether `mesh` holds allocated `normal_code` values. A missing `custom_normal` object, or one
/// that keeps only basis metadata (`domain`, `type`, `corner_order`), means no allocated layer.
pub fn has_normal_code_values(mesh: &Value) -> bool {
    !mesh["attributes"]["custom_normal"]["values"].is_null()
}

/// Validates the `custom_normal` object of `mesh`. Returns `None` when no values are allocated.
/// Otherwise returns one row per face in `mesh.faces` order, each with one entry per face corner.
/// A present `corner_order` is validated against the mesh faces in both cases.
pub fn normal_code_layer(mesh: &Value) -> TestResult<Option<Vec<Vec<StoredCorner>>>> {
    let layer = &mesh["attributes"]["custom_normal"];
    if layer.is_null() {
        return Ok(None);
    }
    if layer["domain"] != "corner" || layer["type"] != "normal_code" {
        return Err(
            format!("custom_normal must be a corner normal_code layer, found {layer}").into(),
        );
    }
    let faces = mesh["faces"].as_array().ok_or("mesh faces are missing")?;
    let mut face_corners = Vec::with_capacity(faces.len());
    for face in faces {
        let id = face["id"].as_u64().ok_or("face id is missing")?;
        let corners = face["v"]
            .as_array()
            .ok_or("face vertices are missing")?
            .len();
        face_corners.push((format!("f{id}"), corners));
    }
    if let Some(order) = layer.get("corner_order") {
        check_corner_order(order, &face_corners)?;
    }
    if layer["values"].is_null() {
        return Ok(None);
    }
    let values = layer["values"]
        .as_object()
        .ok_or("custom_normal values must be an object")?;
    if values.len() != faces.len() {
        return Err(format!(
            "custom_normal has {} rows for {} faces",
            values.len(),
            faces.len()
        )
        .into());
    }
    let mut rows = Vec::with_capacity(faces.len());
    for (key, corners) in &face_corners {
        let row = values
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("custom_normal has no row for {key}"))?;
        if row.len() != *corners {
            return Err(format!(
                "{key} has {corners} corners but {} stored entries",
                row.len()
            )
            .into());
        }
        rows.push(
            row.iter()
                .map(stored_corner)
                .collect::<TestResult<Vec<_>>>()?,
        );
    }
    Ok(Some(rows))
}

/// The allocated `normal_code` rows of `mesh`; an error when no values are allocated.
pub fn normal_code_rows(mesh: &Value) -> TestResult<Vec<Vec<StoredCorner>>> {
    normal_code_layer(mesh)?.ok_or_else(|| "custom_normal has no allocated values".into())
}

/// `corner_order` names exactly the mesh faces, with one entry per face corner, and its ordinals
/// form one unique permutation of `0..corner_count`.
fn check_corner_order(order: &Value, faces: &[(String, usize)]) -> TestResult {
    let order = order
        .as_object()
        .ok_or("corner_order must be an object keyed by face")?;
    let expected: BTreeSet<&str> = faces.iter().map(|(key, _)| key.as_str()).collect();
    let named: BTreeSet<&str> = order.keys().map(String::as_str).collect();
    if expected != named {
        return Err("corner_order must name exactly the mesh faces".into());
    }
    let mut ordinals = Vec::new();
    for (key, corners) in faces {
        let row = order
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("corner_order for {key} is not an array"))?;
        if row.len() != *corners {
            return Err(format!(
                "corner_order for {key} has {} entries for {corners} corners",
                row.len()
            )
            .into());
        }
        for ordinal in row {
            ordinals.push(
                ordinal
                    .as_u64()
                    .ok_or("corner_order ordinal is not an unsigned integer")?,
            );
        }
    }
    ordinals.sort_unstable();
    if ordinals.iter().copied().eq(0..ordinals.len() as u64) {
        Ok(())
    } else {
        Err("corner_order must be a unique permutation of 0..corner_count".into())
    }
}

pub fn number(value: &Value) -> TestResult<f64> {
    value.as_f64().ok_or_else(|| "value is not a number".into())
}

pub fn floats(value: &Value) -> TestResult<Vec<f64>> {
    value
        .as_array()
        .ok_or("value is not an array")?
        .iter()
        .map(number)
        .collect()
}

pub fn vec3(value: &Value) -> TestResult<[f64; 3]> {
    let values = floats(value)?;
    match values.as_slice() {
        [x, y, z] => Ok([*x, *y, *z]),
        _ => Err("value is not a vec3".into()),
    }
}

pub fn vec4(value: &Value) -> TestResult<[f64; 4]> {
    let values = floats(value)?;
    match values.as_slice() {
        [r, g, b, a] => Ok([*r, *g, *b, *a]),
        _ => Err("value is not a vec4".into()),
    }
}

/// Blender's dump of the object named `name`.
pub fn dumped_object<'a>(dump: &'a Value, name: &str) -> TestResult<&'a Value> {
    dump["objects"]
        .as_array()
        .ok_or("dump has no objects")?
        .iter()
        .find(|object| object["object"].as_str() == Some(name))
        .ok_or_else(|| format!("Blender dump has no object {name}").into())
}

pub fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: {actual} differs from {expected} by more than {tolerance}"
    );
}

pub fn assert_floats(actual: &[f64], expected: &[f64], tolerance: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_close(*actual, *expected, tolerance, &format!("{label}[{index}]"));
    }
}

/// `ulps` units in the last place of f32 at `value`'s magnitude: `ulps * 2^(e - 23)` with
/// `e = floor(log2 |value|)`. A value stored as f32 is exact to half an ulp, so a comparison that
/// crosses f32 storage and a transfer function (decode, store, encode) is bounded in ulps, not in
/// absolute terms: an absolute bound that is right near 0 is far too strict near 1. The 8-bit
/// samples used here decode to f32 values in [0.5, 1) for the upper half of the byte range, where
/// one ulp is 2^-24 (about 6e-8); the sRGB encode has slope about 0.6 there, so one ulp of linear
/// storage moves the encoded value by about 0.3 ulp of the encoded value.
pub fn f32_ulp_tolerance(value: f64, ulps: f64) -> f64 {
    let magnitude = value.abs().max(f64::from(f32::MIN_POSITIVE));
    ulps * f64::from(f32::EPSILON) * magnitude.log2().floor().exp2()
}

/// Whether two unit-length-ish vectors agree within `tolerance` in every component.
pub fn vectors_close(actual: [f64; 3], expected: [f64; 3], tolerance: f64) -> bool {
    (0..3).all(|axis| (actual[axis] - expected[axis]).abs() <= tolerance)
}

/// Linear RGBA of one `BYTE_COLOR` value must match within one byte step.
pub fn assert_rgba(actual: [f64; 4], expected: [f64; 4], tolerance: f64, label: &str) {
    for (channel, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_close(*a, e, tolerance, &format!("{label} channel {channel}"));
    }
}

/// The expected JSON's `oracle` section.
pub fn oracle_section(expected: &Value) -> &Value {
    &expected["oracle"]
}

/// Face loop indices (`loop_start + k`) of every face, from the face sizes in order.
pub fn face_loop_starts(face_sizes: &[usize]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(face_sizes.len());
    let mut next = 0;
    for size in face_sizes {
        starts.push(next);
        next += size;
    }
    starts
}

/// Applies `operations` to the project at its current revision and returns the apply envelope.
/// `root` holds the operations file.
#[expect(
    clippy::needless_pass_by_value,
    reason = "callers pass literal operation arrays; changing 49 call sites adds no safety"
)]
pub fn apply_operations(project: &Path, root: &Path, operations: Value) -> TestResult<Value> {
    let inspected = pot_json(|command| {
        command.arg("inspect").arg(project);
    })?;
    let revision = inspected["scene"]["revision"]
        .as_u64()
        .ok_or("scene revision is missing")?;
    let ops = root.join("operations.json");
    fs::write(
        &ops,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "base_revision": revision,
            "operations": operations,
        }))?,
    )?;
    pot_json(|command| {
        command.arg("apply").arg(project).arg("--file").arg(&ops);
    })
}
