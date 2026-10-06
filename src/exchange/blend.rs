use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

use glam::{DMat4, DQuat, DVec3, EulerRot};
use serde_json::{Map, Value, json};

use crate::{
    color::{ColorManagement, ViewTransform},
    error::{ErrorCode, PotError, Result},
    geom::{Edge, Face, IdCounters, Mesh, Vertex},
    graph::{GraphInterface, GraphKind, GraphLink, GraphNode, GraphSocket, NodeGroup},
    model::{
        Action, ActionSlot, ArmatureData, Bone, CameraData, CameraProjection, Collection,
        Constraint, ConstraintType, CurveData, CurveDimensions, CurveFillMode, CurveHandleType,
        CurvePoint, CurveSpline, CurveSplineType, DataBlock, Driver, DriverType, DriverVariable,
        DriverVariableType, Extrapolation, FCurve, ForceField, ForceFieldType, GreasePencilData,
        GreasePencilFrame, GreasePencilLayer, GreasePencilPoint, GreasePencilStroke, Id, Image,
        ImageAlphaMode, ImageColorspace, ImageSource, Interpolation, Keyframe, Library,
        LibraryKind, LibraryOverride, LibraryOverrideProperty, LibraryStatus, LightData, LightType,
        Material, Modifier, MovieClip, MovieTracking, NlaBlendType, NlaExtrapolation, NlaStrip,
        NlaTrack, Node, ParentType, Profile, Registry, RenderSettings, RigidBody, RigidBodyShape,
        RigidBodyType, RigidBodyWorld, Scene, SceneDoc, ShapeKey, ShapeKeyData, SurfaceData,
        SurfacePoint, TextObjectData, TimelineMarker, Transform, UnitSettings, VertexGroup,
        ViewLayer, World,
    },
    sequencer::{
        EffectType as SequenceEffectType, RetimingKey, Strip as SequenceStrip, StripBlendType,
        StripModifier, StripType as SequenceStripType, TransitionType as SequenceTransitionType,
    },
    store::Project,
};

use super::{ExportOptions, ExportReport, ImportedGraph, Loss};

type ImportedResourceAssets = (BTreeMap<String, String>, Vec<(String, Vec<u8>)>);

const BRIDGE: &str = include_str!("blend_bridge.py");
const MAX_BLEND_VERSION: u32 = 502;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Resolve the Blender executable by the documented precedence order.
///
/// # Errors
///
/// Returns `BLENDER_NOT_FOUND` when the explicit path, environment setting, PATH, and
/// macOS application location do not identify an executable file.
pub fn resolve_blender(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return executable_path(path).ok_or_else(|| {
            PotError::new(
                ErrorCode::BlenderNotFound,
                format!("Blender executable not found: {}", path.display()),
            )
        });
    }
    if let Some(path) = std::env::var_os("POTTER_BLENDER") {
        let path = PathBuf::from(path);
        return executable_path(&path).ok_or_else(|| {
            PotError::new(
                ErrorCode::BlenderNotFound,
                format!("POTTER_BLENDER executable not found: {}", path.display()),
            )
        });
    }
    if let Some(path) = find_in_path("blender") {
        return Ok(path);
    }
    let macos = Path::new("/Applications/Blender.app/Contents/MacOS/Blender");
    executable_path(macos).ok_or_else(|| {
        PotError::new(
            ErrorCode::BlenderNotFound,
            "Blender was not found; set --blender or POTTER_BLENDER",
        )
    })
}

fn executable_path(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        Some(path.to_path_buf())
    } else {
        None
    }
}

fn find_in_path(executable: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).find_map(|directory| {
        let candidate = directory.join(executable);
        candidate.is_file().then_some(candidate)
    })
}

/// Export the supported portion of a scene graph through a clean Blender process.
/// Unsupported source structures are reported, and fail closed unless lossy export was
/// explicitly requested. Their JSON compatibility payload is included as an inert Text.
///
/// # Errors
///
/// Returns a typed adapter, Blender, I/O, or representability error.
pub fn export_blend(
    doc: &SceneDoc,
    project: &Project,
    out: &Path,
    opts: &ExportOptions<'_>,
) -> Result<ExportReport> {
    doc.validate()?;
    let blender = resolve_blender(opts.blender)?;
    check_blender_version(&blender)?;
    let mut losses = export_losses(doc);
    losses.extend(native_bind_export_losses(doc, project));
    if !losses.is_empty() && !opts.allow_lossy {
        return Err(PotError::with_details(
            ErrorCode::UnrepresentableFeature,
            "scene contains features the Blender adapter cannot translate losslessly; rerun with --allow-lossy",
            json!({"losses": losses}),
        ));
    }
    let parent = out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(|error| PotError::io(&error))?;
    let document = serde_json::to_value(doc).map_err(PotError::internal_json)?;
    let scene_hash = crate::hash::sha256(&crate::hash::canonicalize(&document)?);
    let (document, asset_files) = prepare_blend_export_resources(document, project, out)?;
    let input = temporary_path("blend-export-input.json")?;
    let script = temporary_path("blend-bridge.py")?;
    let blend_temp = temporary_blend_path(parent)?;
    let validation = temporary_path("blend-export-validation.json")?;
    let operation = (|| {
        write_new(
            &input,
            &serde_json::to_vec(&json!({
                "doc": document,
                "pack": opts.pack,
                "scene_hash": scene_hash,
                "context": opts.context,
            }))
            .map_err(PotError::internal_json)?,
        )?;
        write_new(&script, BRIDGE.as_bytes())?;
        let exported = run_bridge(&blender, &script, "export", &input, &blend_temp, false)?;
        let validation_child =
            match run_bridge(&blender, &script, "import", &blend_temp, &validation, true) {
                Ok(output) => output,
                Err(error) => {
                    forward_child_output(&exported);
                    return Err(PotError::with_details(
                        ErrorCode::ExportFailed,
                        "Blender could not reopen the file it just exported",
                        json!({"cause": error}),
                    ));
                }
            };
        let bytes = match fs::read(&validation) {
            Ok(bytes) => bytes,
            Err(error) => {
                forward_child_output(&exported);
                forward_child_output(&validation_child);
                return Err(PotError::io(&error));
            }
        };
        let intermediate: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => {
                forward_child_output(&exported);
                forward_child_output(&validation_child);
                return Err(PotError::with_details(
                    ErrorCode::ExportFailed,
                    format!("Blender reopen produced invalid intermediate JSON: {error}"),
                    json!({"line": error.line(), "column": error.column()}),
                ));
            }
        };
        if let Err(error) = validate_intermediate(&intermediate) {
            forward_child_output(&exported);
            forward_child_output(&validation_child);
            return Err(PotError::with_details(
                ErrorCode::ExportFailed,
                "Blender export did not pass reopen validation",
                json!({"cause": error}),
            ));
        }
        if let Err(error) = fs::rename(&blend_temp, out) {
            forward_child_output(&exported);
            forward_child_output(&validation_child);
            return Err(PotError::io(&error));
        }
        Ok(())
    })();
    cleanup(&[input, script, blend_temp, validation]);
    operation?;

    let (scene_count, object_count, mesh_count, material_count, action_count) = (
        doc.scenes.len(),
        doc.nodes.len(),
        doc.data_blocks
            .values()
            .filter(|data| data.data_type == "mesh")
            .count(),
        doc.materials.len(),
        doc.actions.len(),
    );
    Ok(ExportReport {
        files: std::iter::once(out.to_path_buf())
            .chain(asset_files)
            .collect(),
        counts: json!({"scenes": scene_count, "objects": object_count, "meshes": mesh_count,
                       "materials": material_count, "actions": action_count}),
        conversions: doc
            .nodes
            .iter()
            .filter(|(_, node)| root_matrix_requires_approximation(node))
            .map(|(id, _)| {
                json!({
                    "feature_id": "blend.root_matrix_shear",
                    "node_id": id.as_str(),
                    "reason": "Blender root loc/rot/scale display is approximated; exact parent_inverse and local transform are retained in potter.root_matrix_json."
                })
            })
            .collect(),
        losses,
    })
}

fn prepare_blend_export_resources(
    mut document: Value,
    project: &Project,
    out: &Path,
) -> Result<(Value, Vec<PathBuf>)> {
    let original_blend_path = document
        .get("compatibility")
        .and_then(Value::as_object)
        .and_then(|entries| {
            entries.values().find_map(|entry| {
                if entry.get("format").and_then(Value::as_str) != Some("blend") {
                    return None;
                }
                entry
                    .get("blobs")
                    .and_then(Value::as_array)?
                    .iter()
                    .find(|blob| blob.get("name").and_then(Value::as_str) == Some("original.blend"))
                    .and_then(|blob| blob.get("path").and_then(Value::as_str))
            })
        })
        .map(|path| project.path().join(path))
        .filter(|path| path.is_file());
    if let Some(path) = original_blend_path {
        document["blender_original_blend_path"] = json!(path.to_string_lossy());
    }
    let resource_ids = document
        .get("resources")
        .and_then(Value::as_object)
        .map(|resources| resources.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    let parent = out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stem = out
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| PotError::new(ErrorCode::InvalidArgument, "output filename is invalid"))?;
    let sidecar = parent.join(format!("{stem}_assets"));
    let mut sidecar_created = false;
    let mut files = Vec::new();
    let library_base = library_layout_base(&document);
    for id in resource_ids {
        let resource = document
            .get("resources")
            .and_then(Value::as_object)
            .and_then(|resources| resources.get(&id))
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "resource disappeared"))?;
        let uri = resource
            .get("uri")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let source_path = resolve_export_resource_path(project, uri);
        let bytes = if let Some(packed) = resource.get("bytes").and_then(Value::as_array) {
            let mut bytes = Vec::with_capacity(packed.len());
            for value in packed {
                let byte = value
                    .as_u64()
                    .filter(|value| *value <= u8::MAX.into())
                    .and_then(|value| u8::try_from(value).ok())
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "packed resource contains a byte outside 0..255",
                        )
                    })?;
                bytes.push(byte);
            }
            Some(bytes)
        } else if let Some(path) = &source_path {
            match fs::read(path) {
                Ok(bytes) => Some(bytes),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(PotError::io(&error)),
            }
        } else {
            None
        };
        let Some(bytes) = bytes else {
            if let Some(path) = source_path {
                if let Some(resources) =
                    document.get_mut("resources").and_then(Value::as_object_mut)
                    && let Some(resource) = resources.get_mut(&id)
                {
                    resource["export_path"] = json!(path.to_string_lossy());
                }
                continue;
            }
            if uri.is_empty() {
                continue;
            }
            return Err(PotError::with_details(
                ErrorCode::ExportFailed,
                "resource URI cannot be resolved as a local file",
                json!({"resource_id":id,"uri":uri}),
            ));
        };
        let digest = crate::hash::sha256(&bytes);
        if resource
            .get("hash")
            .or_else(|| resource.get("expected_hash"))
            .and_then(Value::as_str)
            .is_some_and(|expected| expected != digest)
        {
            return Err(PotError::with_details(
                ErrorCode::ValidationFailed,
                "resource content does not match its expected hash",
                json!({"resource_id":id,"hash":digest}),
            ));
        }
        if !sidecar_created {
            fs::create_dir_all(&sidecar).map_err(|error| PotError::io(&error))?;
            sidecar_created = true;
        }
        let filename = resource
            .get("filename")
            .and_then(Value::as_str)
            .or_else(|| {
                source_path
                    .as_ref()
                    .and_then(|path| path.file_name())
                    .and_then(|name| name.to_str())
            })
            .filter(|name| {
                Path::new(name).file_name().and_then(|value| value.to_str()) == Some(*name)
            })
            .unwrap_or("resource");
        let mirrored = (resource.get("kind").and_then(Value::as_str) == Some("library"))
            .then(|| {
                let base = library_base.as_ref()?;
                let original = resource.get("original_path").and_then(Value::as_str)?;
                Path::new(original)
                    .strip_prefix(base)
                    .ok()
                    .map(|relative| sidecar.join(relative))
            })
            .flatten()
            .filter(|path| {
                !path.exists() || fs::read(path).is_ok_and(|existing| existing == bytes)
            });
        let destination = match mirrored {
            Some(path) => {
                if let Some(dir) = path.parent() {
                    fs::create_dir_all(dir).map_err(|error| PotError::io(&error))?;
                }
                path
            }
            None => sidecar_destination(&sidecar, filename, &digest, &bytes)?,
        };
        if !destination.is_file() {
            write_asset_sidecar(&destination, &bytes)?;
        }
        let absolute = fs::canonicalize(&destination).map_err(|error| PotError::io(&error))?;
        if let Some(resources) = document.get_mut("resources").and_then(Value::as_object_mut)
            && let Some(resource) = resources.get_mut(&id)
        {
            resource["export_path"] = json!(absolute);
        }
        if !files.contains(&destination) {
            files.push(destination);
        }
    }
    Ok((document, files))
}

/// Libraries reference each other with paths relative to their own file, so
/// their sidecar copies keep the original layout below the deepest directory
/// shared by every library's source path.
fn library_layout_base(document: &Value) -> Option<PathBuf> {
    let mut directories = document
        .get("resources")
        .and_then(Value::as_object)?
        .values()
        .filter(|resource| resource.get("kind").and_then(Value::as_str) == Some("library"))
        .filter_map(|resource| resource.get("original_path").and_then(Value::as_str))
        .filter_map(|path| Path::new(path).parent().map(Path::to_path_buf));
    let mut base = directories.next()?;
    for directory in directories {
        while !directory.starts_with(&base) {
            base = base.parent()?.to_path_buf();
        }
    }
    Some(base)
}

fn resolve_export_resource_path(project: &Project, uri: &str) -> Option<PathBuf> {
    if uri.is_empty() || (uri.contains("://") && !uri.starts_with("file://")) {
        return None;
    }
    let path = PathBuf::from(uri.strip_prefix("file://").unwrap_or(uri));
    Some(if path.is_absolute() {
        path
    } else {
        project.path().join(path)
    })
}

fn sidecar_destination(
    sidecar: &Path,
    filename: &str,
    digest: &str,
    bytes: &[u8],
) -> Result<PathBuf> {
    let original = sidecar.join(filename);
    if !original.exists() {
        return Ok(original);
    }
    if original.is_file() && fs::read(&original).map_err(|error| PotError::io(&error))? == bytes {
        return Ok(original);
    }
    let path = Path::new(filename);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(filename);
    let extension = path.extension().and_then(|value| value.to_str());
    let digest = digest.strip_prefix("sha256:").unwrap_or_default();
    let mut collision = 1_u32;
    loop {
        let suffix = if collision == 1 {
            digest[..12].to_owned()
        } else {
            format!("{}_{collision}", &digest[..12])
        };
        let name = extension.map_or_else(
            || format!("{stem}_{suffix}"),
            |extension| format!("{stem}_{suffix}.{extension}"),
        );
        let candidate = sidecar.join(name);
        if !candidate.exists() {
            return Ok(candidate);
        }
        if candidate.is_file()
            && fs::read(&candidate).map_err(|error| PotError::io(&error))? == bytes
        {
            return Ok(candidate);
        }
        collision = collision.checked_add(1).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "sidecar filename collision limit reached",
            )
        })?;
    }
}

/// Import a Blender file through Blender's own reader, preserving its exact original bytes
/// in `compat_blobs` in addition to the editable graph projection.
///
/// # Errors
///
/// Returns a typed adapter, Blender, I/O, schema, or unsupported-version error.
pub fn import_blend(file: &Path, blender: Option<&Path>) -> Result<ImportedGraph> {
    if !file.is_file() {
        return Err(PotError::new(
            ErrorCode::FileNotFound,
            format!("Blender input file not found: {}", file.display()),
        ));
    }
    check_file_version(file)?;
    let blender = resolve_blender(blender)?;
    check_blender_version(&blender)?;
    let script = temporary_path("blend-bridge.py")?;
    let output = temporary_path("blend-import-output.json")?;
    let operation = (|| {
        write_new(&script, BRIDGE.as_bytes())?;
        let child = run_bridge(&blender, &script, "import", file, &output, true)?;
        let bytes = match fs::read(&output) {
            Ok(bytes) => bytes,
            Err(error) => {
                forward_child_output(&child);
                return Err(PotError::io(&error));
            }
        };
        let mut raw: Value = match serde_json::from_slice(&bytes) {
            Ok(raw) => raw,
            Err(error) => {
                forward_child_output(&child);
                return Err(PotError::with_details(
                    ErrorCode::ImportFailed,
                    format!("Blender bridge produced invalid intermediate JSON: {error}"),
                    json!({"line": error.line(), "column": error.column()}),
                ));
            }
        };
        if let Err(error) = validate_intermediate(&raw) {
            forward_child_output(&child);
            return Err(error);
        }
        discover_nested_blend_libraries(&mut raw, file, &blender, &script, &output)?;
        let original = match fs::read(file) {
            Ok(original) => original,
            Err(error) => {
                forward_child_output(&child);
                return Err(PotError::io(&error));
            }
        };
        let mut native_bindings =
            super::blend_dna::read_modifier_bindings(&original).map_err(|error| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    format!("could not read native Blender modifier bind data: {error}"),
                    json!({"file": file.display().to_string()}),
                )
            })?;
        attach_native_target_vertices(&mut raw, &mut native_bindings)?;
        build_imported_graph(&raw, file, original, &native_bindings)
    })();
    cleanup(&[script, output]);
    operation
}

fn attach_native_target_vertices(
    raw: &mut Value,
    native_bindings: &mut BTreeMap<(String, String), Value>,
) -> Result<()> {
    let Some(objects) = raw.get_mut("objects").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for object in objects {
        let Some(object_name) = object
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            continue;
        };
        let Some(modifiers) = object.get_mut("modifiers").and_then(Value::as_array_mut) else {
            continue;
        };
        for modifier in modifiers {
            let target_mesh = modifier
                .get("native_target_mesh")
                .filter(|value| !value.is_null())
                .cloned();
            let Some(properties) = modifier.as_object_mut() else {
                continue;
            };
            properties.remove("native_target_mesh");
            let Some(target_mesh) = target_mesh else {
                continue;
            };
            let modifier_name = modifier
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| {
                    invalid_intermediate("native Surface Deform modifier has no name")
                })?;
            let vertices = target_mesh
                .get("vertices")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid_intermediate("native target mesh vertices are invalid"))?;
            if let Some(binding) = native_bindings
                .get_mut(&(object_name.clone(), modifier_name))
                .and_then(Value::as_object_mut)
            {
                binding.insert(
                    "target_evaluated_vertices".to_owned(),
                    Value::Array(vertices.clone()),
                );
            }
        }
    }
    Ok(())
}
fn check_blender_version(blender: &Path) -> Result<()> {
    let output = Command::new(blender)
        .arg("--version")
        .output()
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                PotError::new(
                    ErrorCode::BlenderNotFound,
                    format!("Blender was not found: {}", blender.display()),
                )
            } else {
                PotError::io(&error)
            }
        })?;
    if !output.status.success() {
        forward_child_output(&output);
        return Err(PotError::with_details(
            ErrorCode::BlenderVersionUnsupported,
            "could not determine Blender version",
            child_details(&output),
        ));
    }
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let version = text.lines().find_map(parse_version_line);
    if version.is_none_or(|(major, minor, _patch)| (major, minor) != (5, 2)) {
        forward_child_output(&output);
        return Err(PotError::with_details(
            ErrorCode::BlenderVersionUnsupported,
            "the Blender adapter requires Blender 5.2.x",
            json!({"reported": text.lines().next().unwrap_or_default()}),
        ));
    }
    Ok(())
}

fn parse_version_line(line: &str) -> Option<(u32, u32, u32)> {
    let marker = line.find("Blender ")? + "Blender ".len();
    let mut values = line[marker..].split(|ch: char| !ch.is_ascii_digit() && ch != '.');
    let mut version = values.next()?.split('.');
    Some((
        version.next()?.parse().ok()?,
        version.next()?.parse().ok()?,
        version.next()?.parse().ok()?,
    ))
}

fn check_file_version(file: &Path) -> Result<()> {
    let mut bytes = [0_u8; 12];
    let mut source = fs::File::open(file).map_err(|error| PotError::io(&error))?;
    let count = source
        .read(&mut bytes)
        .map_err(|error| PotError::io(&error))?;
    if count >= 12 && &bytes[..7] == b"BLENDER" {
        let version_text = std::str::from_utf8(&bytes[9..12]).ok();
        if let Some(version) = version_text.and_then(|text| text.parse::<u32>().ok())
            && version > MAX_BLEND_VERSION
        {
            return Err(PotError::with_details(
                ErrorCode::BlenderVersionUnsupported,
                "the .blend file was written by a version newer than Blender 5.2",
                json!({"file_version": version, "maximum_supported": MAX_BLEND_VERSION}),
            ));
        }
    }
    Ok(())
}

fn run_bridge(
    blender: &Path,
    script: &Path,
    mode: &str,
    input: &Path,
    output: &Path,
    importing: bool,
) -> Result<Output> {
    let result = Command::new(blender)
        .arg("--background")
        .arg("--factory-startup")
        .arg("--disable-depsgraph-on-file-load")
        .arg("--disable-autoexec")
        .arg("--python-exit-code")
        .arg("3")
        .arg("--python")
        .arg(script)
        .arg("--")
        .arg(mode)
        .arg(input)
        .arg(output)
        .output();
    let child = result.map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PotError::new(
                ErrorCode::BlenderNotFound,
                format!("Blender was not found: {}", blender.display()),
            )
        } else {
            PotError::io(&error)
        }
    })?;
    if !child.status.success() {
        forward_child_output(&child);
        return Err(PotError::with_details(
            if importing {
                ErrorCode::ImportFailed
            } else {
                ErrorCode::ExportFailed
            },
            format!("Blender {mode} bridge failed with status {}", child.status),
            child_details(&child),
        ));
    }
    Ok(child)
}

fn child_details(output: &Output) -> Value {
    json!({"status": output.status.code(), "stdout": String::from_utf8_lossy(&output.stdout),
           "stderr": String::from_utf8_lossy(&output.stderr)})
}

fn forward_child_output(output: &Output) {
    let mut stderr = std::io::stderr().lock();
    let _ = stderr.write_all(&output.stdout);
    let _ = stderr.write_all(&output.stderr);
}

fn temporary_path(stem: &str) -> Result<PathBuf> {
    temporary_path_in(&std::env::temp_dir(), stem)
}

fn temporary_path_in(directory: &Path, stem: &str) -> Result<PathBuf> {
    for _ in 0..100 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(".{stem}-{}-{sequence}", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => {
                drop(file);
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(PotError::io(&error)),
        }
    }
    Err(PotError::new(
        ErrorCode::IoError,
        "could not reserve a temporary adapter path",
    ))
}
fn temporary_blend_path(directory: &Path) -> Result<PathBuf> {
    for _ in 0..100 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(
            ".blend-output-{}-{sequence}.blend",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => {
                drop(file);
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(PotError::io(&error)),
        }
    }
    Err(PotError::new(
        ErrorCode::IoError,
        "could not reserve a temporary Blender output path",
    ))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|error| PotError::io(&error))?;
    file.write_all(bytes)
        .map_err(|error| PotError::io(&error))?;
    file.sync_all().map_err(|error| PotError::io(&error))
}

fn write_asset_sidecar(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| PotError::io(&error))?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = fs::remove_file(path);
        return Err(PotError::io(&error));
    }
    Ok(())
}

fn cleanup(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn validate_intermediate(raw: &Value) -> Result<()> {
    let object = raw
        .as_object()
        .ok_or_else(|| invalid_intermediate("top-level value is not an object"))?;
    if object.get("bridge_version").and_then(Value::as_u64) != Some(1) {
        return Err(invalid_intermediate("unsupported bridge_version"));
    }
    let version = object
        .get("blender_version")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !version.starts_with("5.2.") {
        return Err(PotError::with_details(
            ErrorCode::BlenderVersionUnsupported,
            "intermediate data was not produced by Blender 5.2.x",
            json!({"blender_version": version}),
        ));
    }
    let file_version = object
        .get("file_version")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_intermediate("missing or invalid `file_version`"))?;
    if file_version.len() < 2 {
        return Err(invalid_intermediate(
            "`file_version` must contain major and minor values",
        ));
    }
    let file_major = file_version[0]
        .as_u64()
        .ok_or_else(|| invalid_intermediate("invalid `file_version` major value"))?;
    let file_minor = file_version[1]
        .as_u64()
        .ok_or_else(|| invalid_intermediate("invalid `file_version` minor value"))?;
    if file_major > 5 || (file_major == 5 && file_minor > 2) {
        return Err(PotError::with_details(
            ErrorCode::BlenderVersionUnsupported,
            "the .blend file was written by a version newer than Blender 5.2",
            json!({"file_version": file_version}),
        ));
    }
    for key in [
        "scenes",
        "collections",
        "objects",
        "meshes",
        "materials",
        "cameras",
        "lights",
        "actions",
        "texts",
        "node_groups",
        "other_datablocks",
        "unused_datablocks",
    ] {
        if !object.get(key).is_some_and(Value::is_array) {
            return Err(invalid_intermediate(&format!(
                "missing or invalid `{key}` array"
            )));
        }
    }
    if object["scenes"].as_array().is_none_or(Vec::is_empty) {
        return Err(invalid_intermediate("intermediate data contains no scenes"));
    }
    for collection in [
        "scenes",
        "collections",
        "objects",
        "meshes",
        "materials",
        "cameras",
        "lights",
        "armatures",
        "grease_pencils",
        "actions",
        "texts",
        "node_groups",
        "other_datablocks",
        "unused_datablocks",
    ] {
        if let Some(items) = object.get(collection).and_then(Value::as_array) {
            for item in items {
                if !item.is_object() {
                    return Err(invalid_intermediate(&format!(
                        "`{collection}` contains a non-object item"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn invalid_intermediate(message: &str) -> PotError {
    PotError::new(
        ErrorCode::ImportFailed,
        format!("invalid Blender intermediate JSON: {message}"),
    )
}

fn discover_nested_blend_libraries(
    raw: &mut Value,
    main_file: &Path,
    blender: &Path,
    script: &Path,
    output: &Path,
) -> Result<()> {
    let mut queue = optional_array(raw, "libraries")?.to_vec();
    let mut known_libraries = queue
        .iter()
        .filter_map(|library| library.get("path").and_then(Value::as_str))
        .map(|path| fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path)))
        .collect::<BTreeSet<_>>();
    let mut processed = BTreeSet::new();
    let main_file = fs::canonicalize(main_file).unwrap_or_else(|_| main_file.to_path_buf());
    let mut known_resources = optional_array(raw, "resources")?
        .iter()
        .filter_map(|resource| resource.get("path").and_then(Value::as_str))
        .map(|path| fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path)))
        .collect::<BTreeSet<_>>();
    let mut cursor = 0;
    while cursor < queue.len() {
        let library = queue[cursor].clone();
        cursor += 1;
        let Some(path) = library.get("path").and_then(Value::as_str) else {
            continue;
        };
        let path = fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path));
        if path == main_file || !path.is_file() || !processed.insert(path.clone()) {
            continue;
        }
        let child = run_bridge(blender, script, "import", &path, output, false)?;
        let bytes = fs::read(output).map_err(|error| {
            forward_child_output(&child);
            PotError::io(&error)
        })?;
        let nested: Value = serde_json::from_slice(&bytes).map_err(|error| {
            forward_child_output(&child);
            PotError::with_details(
                ErrorCode::ImportFailed,
                format!("nested Blender library produced invalid intermediate JSON: {error}"),
                json!({"path":path,"line":error.line(),"column":error.column()}),
            )
        })?;
        validate_intermediate(&nested)?;
        let nested_libraries = optional_array(&nested, "libraries")?.to_vec();
        let nested_resources = optional_array(&nested, "resources")?.to_vec();
        for nested_library in nested_libraries {
            let nested_path = nested_library
                .get("path")
                .and_then(Value::as_str)
                .map(|path| fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path)));
            let Some(nested_path) = nested_path else {
                continue;
            };
            if known_libraries.insert(nested_path.clone()) {
                if let Some(libraries) = raw.get_mut("libraries").and_then(Value::as_array_mut) {
                    libraries.push(nested_library.clone());
                }
                queue.push(nested_library.clone());
            }
            if let Some(resource) = nested_resources.iter().find(|resource| {
                resource.get("kind").and_then(Value::as_str) == Some("library")
                    && resource
                        .get("path")
                        .and_then(Value::as_str)
                        .is_some_and(|resource_path| {
                            fs::canonicalize(resource_path)
                                .unwrap_or_else(|_| PathBuf::from(resource_path))
                                == nested_path
                        })
            }) && known_resources.insert(nested_path)
                && let Some(resources) = raw.get_mut("resources").and_then(Value::as_array_mut)
            {
                resources.push(resource.clone());
            }
        }
    }
    Ok(())
}

fn build_imported_graph(
    raw: &Value,
    file: &Path,
    original: Vec<u8>,
    native_bindings: &BTreeMap<(String, String), Value>,
) -> Result<ImportedGraph> {
    let mut doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
    doc.scenes.clear();
    doc.collections.clear();
    doc.nodes.clear();
    doc.data_blocks.clear();
    doc.materials.clear();
    doc.actions.clear();
    doc.worlds.clear();

    let collections = array(raw, "collections")?;
    let scenes = array(raw, "scenes")?;
    let objects = array(raw, "objects")?;
    let meshes = array(raw, "meshes")?;
    let materials = array(raw, "materials")?;
    let cameras = array(raw, "cameras")?;
    let lights = array(raw, "lights")?;
    let armatures = optional_array(raw, "armatures")?;
    let grease_pencils = if raw.get("grease_pencils").is_some() {
        optional_array(raw, "grease_pencils")?
    } else {
        optional_array(raw, "grease_pencil")?
    };
    let actions = array(raw, "actions")?;
    let images = optional_array(raw, "images")?;
    let volumes = optional_array(raw, "volumes")?;
    let mut mappings = BTreeMap::<String, String>::new();
    let (resource_ids, resource_assets) = import_blender_resources(raw, file, &mut doc)?;
    let movie_clip_ids = import_movie_clips(raw, &resource_ids, &mut doc, &mut mappings)?;
    let mut collection_ids = BTreeMap::new();
    for item in collections {
        let name = string_field(item, "name")?;
        let id = mapped_id(
            item,
            "potter_id",
            "collection",
            name,
            &mut collection_ids,
            &mut mappings,
            "Collection",
        )?;
        doc.collections.insert(
            id.clone(),
            Collection {
                name: name.to_owned(),
                children: Vec::new(),
                objects: Vec::new(),
            },
        );
    }
    let mut scene_ids = BTreeMap::new();
    for item in scenes {
        let name = string_field(item, "name")?;
        let id = mapped_id(
            item,
            "potter_id",
            "scene",
            name,
            &mut scene_ids,
            &mut mappings,
            "Scene",
        )?;
        let root_name = string_field(item, "root_collection")?;
        let root = collection_ids
            .get(root_name)
            .cloned()
            .ok_or_else(|| invalid_intermediate("scene references an unknown root collection"))?;
        let mut view_layers = Registry::new();
        let mut view_layer_ids = BTreeMap::new();
        if let Some(raw_layers) = item.get("view_layers").and_then(Value::as_array) {
            for view_layer in raw_layers {
                let layer_name = string_field(view_layer, "name")?;
                let layer_id = mapped_id(
                    view_layer,
                    "potter_id",
                    "view",
                    layer_name,
                    &mut view_layer_ids,
                    &mut mappings,
                    &format!("ViewLayer:{id}"),
                )?;
                let excluded_collections = view_layer
                    .get("excluded_collections")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .filter_map(|collection_name| collection_ids.get(collection_name).cloned())
                    .collect();
                view_layers.insert(
                    layer_id,
                    ViewLayer {
                        name: layer_name.to_owned(),
                        excluded_collections,
                    },
                );
            }
        }
        if view_layers.is_empty() {
            view_layers.insert(
                Id::new("view_main")?,
                ViewLayer {
                    name: "View Layer".to_owned(),
                    excluded_collections: Vec::new(),
                },
            );
        }
        let unit = item.get("unit").and_then(Value::as_object);
        let frame_start = i32::try_from(integer_field(item, "frame_start", 1))
            .map_err(|_| invalid_intermediate("scene frame_start is out of range"))?;
        let frame_end = i32::try_from(integer_field(item, "frame_end", 250))
            .map_err(|_| invalid_intermediate("scene frame_end is out of range"))?;
        let fps = u32::try_from(integer_field(item, "fps", 24))
            .map_err(|_| invalid_intermediate("scene fps is out of range"))?;
        let render = item.get("render").cloned().unwrap_or(Value::Null);
        let render_settings = RenderSettings {
            resolution_x: u32::try_from(integer_field(&render, "resolution_x", 1920))
                .map_err(|_| invalid_intermediate("render resolution_x is out of range"))?,
            resolution_y: u32::try_from(integer_field(&render, "resolution_y", 1080))
                .map_err(|_| invalid_intermediate("render resolution_y is out of range"))?,
            resolution_percentage: u32::try_from(integer_field(
                &render,
                "resolution_percentage",
                100,
            ))
            .map_err(|_| invalid_intermediate("render resolution_percentage is out of range"))?,
            samples: u32::try_from(integer_field(&render, "samples", 64))
                .map_err(|_| invalid_intermediate("render samples is out of range"))?,
            seed: u32::try_from(integer_field(&render, "seed", 0))
                .map_err(|_| invalid_intermediate("render seed is out of range"))?,
            max_bounces: u32::try_from(integer_field(&render, "max_bounces", 4))
                .map_err(|_| invalid_intermediate("render max_bounces is out of range"))?,
            film_transparent: render
                .get("film_transparent")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            engine: render
                .get("engine")
                .and_then(Value::as_str)
                .unwrap_or("path")
                .to_owned(),
            use_sequencer: render
                .get("use_sequencer")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            audio_codec: render
                .get("audio_codec")
                .and_then(Value::as_str)
                .unwrap_or("wav")
                .to_owned(),
            passes: render.get("passes").and_then(Value::as_array).map_or_else(
                || vec!["combined".to_owned()],
                |values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                },
            ),
            motion_blur: render
                .get("motion_blur")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            shutter: render
                .get("shutter")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value > 0.0 && *value <= 2.0)
                .unwrap_or(0.5),
            motion_blur_samples: u32::try_from(integer_field(&render, "motion_blur_samples", 8))
                .map_err(|_| invalid_intermediate("motion_blur_samples is out of range"))?,
        };
        let mut marker_ids = BTreeMap::new();
        let mut markers = Vec::new();
        for marker in optional_array(item, "markers")? {
            let marker_name = string_field(marker, "name")?;
            let marker_id = mapped_id(
                marker,
                "potter_id",
                "marker",
                marker_name,
                &mut marker_ids,
                &mut mappings,
                &format!("TimelineMarker:{name}"),
            )?;
            markers.push(TimelineMarker {
                id: marker_id,
                name: marker_name.to_owned(),
                frame: number_field(marker, "frame", 1.0),
            });
        }
        let rigid_body_world = item
            .get("rigid_body_world")
            .filter(|value| !value.is_null())
            .map(parse_rigid_body_world)
            .transpose()?;
        let scene = Scene {
            name: name.to_owned(),
            root_collection: root,
            view_layers,
            frame_current: number_field(item, "frame_current", 1.0),
            frame_start,
            frame_end,
            fps,
            fps_base: number_field(item, "fps_base", 1.0),
            camera: None,
            world: None,
            unit: UnitSettings {
                system: unit
                    .and_then(|value| value.get("system"))
                    .and_then(Value::as_str)
                    .unwrap_or("none")
                    .to_lowercase(),
                scale_length: unit
                    .and_then(|value| value.get("scale_length"))
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0),
            },
            render: render_settings,
            use_compositing: item
                .get("use_compositing")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            compositor: None,
            color_management: parse_color_management(item.get("color_management"))?,
            markers,
            rigid_body_world,
            sequencer: parse_sequencer(item, &mut mappings)?,
            // Active Movie Clip is mapped by the adapter pass that maps tracking data.
            active_clip: None,
        };
        doc.scenes.insert(id, scene);
    }
    let mut world_ids = BTreeMap::<String, Id>::new();
    for item in scenes {
        let scene_name = string_field(item, "name")?;
        let Some(world) = item.get("world").filter(|world| !world.is_null()) else {
            continue;
        };
        let name = string_field(world, "name")?;
        let world_id = if let Some(id) = world_ids.get(name) {
            id.clone()
        } else {
            mapped_id(
                world,
                "potter_id",
                "world",
                name,
                &mut world_ids,
                &mut mappings,
                "World",
            )?
        };
        if !doc.worlds.contains_key(&world_id) {
            doc.worlds.insert(
                world_id.clone(),
                World {
                    color: vec3_field(world, "color", [1.0; 3]),
                    background_color: world.get("background_color").and_then(parse_vec3_value),
                    strength: number_field(world, "strength", 1.0),
                    ..World::default()
                },
            );
        }
        if let Some(scene_id) = scene_ids.get(scene_name)
            && let Some(scene) = doc.scenes.get_mut(scene_id)
        {
            scene.world = Some(world_id);
        }
    }
    let active_name = raw
        .get("active_scene")
        .and_then(Value::as_str)
        .unwrap_or_default();
    doc.active_scene = scene_ids
        .get(active_name)
        .cloned()
        .or_else(|| doc.scenes.keys().next().cloned())
        .ok_or_else(|| invalid_intermediate("scene registry is empty"))?;
    for item in scenes {
        let Some(clip_name) = item.get("active_clip").and_then(Value::as_str) else {
            continue;
        };
        let scene_name = string_field(item, "name")?;
        let Some(scene_id) = scene_ids.get(scene_name) else {
            continue;
        };
        let clip_id = movie_clip_ids
            .get(clip_name)
            .cloned()
            .ok_or_else(|| invalid_intermediate("scene references an unknown active movie clip"))?;
        if let Some(scene) = doc.scenes.get_mut(scene_id) {
            scene.active_clip = Some(clip_id);
        }
    }

    let (armature_ids, armature_bones_by_data) =
        import_armatures(armatures, &mut doc, &mut mappings)?;
    let mut mesh_ids = BTreeMap::new();
    let mut camera_ids = BTreeMap::new();
    let mut light_ids = BTreeMap::new();
    let mut shape_key_vertex_groups = BTreeMap::<Id, BTreeMap<Id, String>>::new();
    for item in meshes {
        let name = string_field(item, "name")?;
        let id = mapped_id(
            item,
            "potter_id",
            "mesh",
            name,
            &mut mesh_ids,
            &mut mappings,
            "Mesh",
        )?;
        let mesh = convert_mesh(item)?;
        let (shape_keys, key_group_names) = parse_shape_keys(item, &mesh, name, &mut mappings)?;
        if !key_group_names.is_empty() {
            shape_key_vertex_groups.insert(id.clone(), key_group_names);
        }
        let mut attributes = mesh.attributes.clone();
        attributes.insert(
            "blender_metadata".to_owned(),
            json!({"custom_properties": item.get("custom_properties"),
                "fake_user": item.get("fake_user"), "users": item.get("users"),
                "rna_properties": item.get("rna_properties")}),
        );
        doc.data_blocks.insert(
            id,
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: item
                    .get("descriptor")
                    .filter(|value| !value.is_null())
                    .cloned()
                    .and_then(|value| {
                        serde_json::from_value(value).map_or_else(
                            |_| {
                                eprintln!(
                                    "potter: ignoring unparseable primitive descriptor on mesh {name}"
                                );
                                None
                            },
                            Some,
                        )
                    }),
                mesh: Some(Mesh { attributes, ..mesh }),
                camera: None,
                light: None,
                shape_keys,
                ..DataBlock::default()
            },
        );
    }
    for item in cameras {
        let name = string_field(item, "name")?;
        let id = mapped_id(
            item,
            "potter_id",
            "camera",
            name,
            &mut camera_ids,
            &mut mappings,
            "Camera",
        )?;
        let projection = if item.get("type").and_then(Value::as_str) == Some("ORTHO") {
            CameraProjection::Orthographic
        } else {
            CameraProjection::Perspective
        };
        doc.data_blocks.insert(
            id,
            DataBlock {
                data_type: "camera".to_owned(),
                descriptor: None,
                mesh: None,
                camera: Some(CameraData {
                    projection,
                    lens_mm: number_field(item, "lens", 50.0),
                    sensor_height_mm: number_field(item, "sensor_height", 24.0),
                    sensor_fit: item
                        .get("sensor_fit")
                        .and_then(Value::as_str)
                        .unwrap_or("AUTO")
                        .to_owned(),
                    sensor_width_mm: number_field(item, "sensor_width", 36.0),
                    ortho_scale: number_field(item, "ortho_scale", 6.0),
                    clip_start: number_field(item, "clip_start", 0.1),
                    clip_end: number_field(item, "clip_end", 1000.0),
                    shift: [
                        number_field(item, "shift_x", 0.0),
                        number_field(item, "shift_y", 0.0),
                    ],
                    ..CameraData::default()
                }),
                grease_pencil: None,
                ..DataBlock::default()
            },
        );
    }
    for item in lights {
        let name = string_field(item, "name")?;
        let id = mapped_id(
            item,
            "potter_id",
            "light",
            name,
            &mut light_ids,
            &mut mappings,
            "Light",
        )?;
        let light_type = match item.get("type").and_then(Value::as_str).unwrap_or("POINT") {
            "SUN" => LightType::Sun,
            "SPOT" => LightType::Spot,
            "AREA" => LightType::Area,
            _ => LightType::Point,
        };
        doc.data_blocks.insert(
            id,
            DataBlock {
                data_type: "light".to_owned(),
                descriptor: None,
                mesh: None,
                camera: None,
                light: Some(LightData {
                    light_type,
                    color: vec3_field(item, "color", [1.0; 3]),
                    energy: number_field(item, "energy", 1000.0),
                    radius: number_field(item, "shadow_soft_size", 0.1),
                    spot_size: number_field(item, "spot_size", std::f64::consts::FRAC_PI_4),
                    spot_blend: number_field(item, "spot_blend", 0.15),
                    area_shape: item
                        .get("shape")
                        .and_then(Value::as_str)
                        .unwrap_or("SQUARE")
                        .to_owned(),
                    area_size: number_field(item, "size", 0.25),
                    area_size_y: number_field(item, "size_y", 0.25),
                }),
                grease_pencil: None,
                ..DataBlock::default()
            },
        );
    }

    let curve_ids = import_curves(optional_array(raw, "curves")?, &mut doc, &mut mappings)?;
    let volume_ids = import_volumes(volumes, objects, &resource_ids, &mut doc, &mut mappings)?;
    let mut material_ids = BTreeMap::new();
    for item in materials {
        let name = string_field(item, "name")?;
        let id = mapped_id(
            item,
            "potter_id",
            "material",
            name,
            &mut material_ids,
            &mut mappings,
            "Material",
        )?;
        let color = vec4_field(item, "base_color", [0.6, 0.6, 0.6, 1.0]);
        doc.materials.insert(
            id,
            Material {
                name: name.to_owned(),
                base_color: color,
                metallic: number_field(item, "metallic", 0.0),
                roughness: number_field(item, "roughness", 0.8),
                emission_color: vec3_field(item, "emission_color", [0.0; 3]),
                emission_strength: number_field(item, "emission_strength", 0.0),
                transmission: number_field(item, "transmission", 0.0),
                ior: number_field(item, "ior", 1.45),
                double_sided: item
                    .get("double_sided")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                ..Material::default()
            },
        );
    }
    let grease_pencil_ids =
        import_grease_pencils(grease_pencils, &material_ids, &mut doc, &mut mappings)?;
    let mut image_ids = BTreeMap::new();
    for item in images {
        let name = string_field(item, "name")?;
        let id = mapped_id(
            item,
            "potter_id",
            "image",
            name,
            &mut image_ids,
            &mut mappings,
            "Image",
        )?;
        if let Some(image_model) = item.get("potter_image").filter(|value| value.is_object()) {
            let image: Image = serde_json::from_value(image_model.clone())
                .map_err(|_| invalid_intermediate("image metadata is invalid"))?;
            doc.images.insert(id, image);
        } else if let Some(image) =
            imported_image_from_resource(item, &resource_ids, &resource_assets, &doc)?
        {
            doc.images.insert(id, image);
        }
    }
    let mut graph_sources = array(raw, "node_groups")?.to_vec();
    let mut graph_names = graph_sources
        .iter()
        .filter_map(|source| {
            source
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect::<BTreeSet<_>>();
    let mut shader_group_names = BTreeMap::new();
    for material in materials {
        let Some(tree) = material.get("nodes").filter(|tree| {
            tree.get("nodes")
                .and_then(Value::as_array)
                .is_some_and(|nodes| !nodes.is_empty())
        }) else {
            continue;
        };
        let material_name = string_field(material, "name")?;
        let mut graph_name = tree
            .get("name")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map_or_else(|| format!("{material_name} Shader"), str::to_owned);
        let base_name = graph_name.clone();
        let mut suffix = 2_u32;
        while !graph_names.insert(graph_name.clone()) {
            graph_name = format!("{base_name} {suffix}");
            suffix = suffix.saturating_add(1);
        }
        graph_sources.push(json!({
            "name": graph_name,
            "potter_id": tree.get("potter_id"),
            "type": "ShaderNodeTree",
            "tree": tree,
        }));
        shader_group_names.insert(material_name.to_owned(), graph_name);
    }
    let mut world_shader_group_names = BTreeMap::new();
    for scene in scenes {
        let Some(world) = scene.get("world").filter(|world| !world.is_null()) else {
            continue;
        };
        let world_name = string_field(world, "name")?;
        if world_shader_group_names.contains_key(world_name) {
            continue;
        }
        let Some(tree) = world.get("nodes").filter(|tree| {
            tree.get("nodes")
                .and_then(Value::as_array)
                .is_some_and(|nodes| !nodes.is_empty())
        }) else {
            continue;
        };
        let mut graph_name = tree
            .get("name")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map_or_else(|| format!("{world_name} Shader"), str::to_owned);
        let base_name = graph_name.clone();
        let mut suffix = 2_u32;
        while !graph_names.insert(graph_name.clone()) {
            graph_name = format!("{base_name} {suffix}");
            suffix = suffix.saturating_add(1);
        }
        graph_sources.push(json!({
            "name": graph_name,
            "potter_id": tree.get("potter_id"),
            "type": "ShaderNodeTree",
            "tree": tree,
        }));
        world_shader_group_names.insert(world_name.to_owned(), graph_name);
    }
    resolve_image_node_ids(&mut graph_sources, &image_ids);
    let node_group_ids = import_node_groups(&graph_sources, &mut doc, &mut mappings)?;
    for (material_name, graph_name) in shader_group_names {
        if let (Some(material_id), Some(graph_id)) = (
            material_ids.get(&material_name),
            node_group_ids.get(&graph_name),
        ) && let Some(material) = doc.materials.get_mut(material_id)
        {
            material.node_tree = Some(graph_id.clone());
        }
    }
    for (world_name, graph_name) in world_shader_group_names {
        if let (Some(world_id), Some(graph_id)) =
            (world_ids.get(&world_name), node_group_ids.get(&graph_name))
            && let Some(world) = doc.worlds.get_mut(world_id)
        {
            world.node_tree = Some(graph_id.clone());
        }
    }
    for source_scene in scenes {
        let scene_name = string_field(source_scene, "name")?;
        let compositor_name = source_scene
            .get("compositor")
            .and_then(|compositor| compositor.get("name"))
            .and_then(Value::as_str);
        let Some(compositor_id) = compositor_name.and_then(|name| node_group_ids.get(name)) else {
            continue;
        };
        if let Some(scene_id) = scene_ids.get(scene_name)
            && let Some(scene) = doc.scenes.get_mut(scene_id)
        {
            scene.compositor = Some(compositor_id.clone());
        }
    }
    let mut action_ids = BTreeMap::new();
    let mut action_by_name = BTreeMap::new();
    let mut raw_action_slots = BTreeMap::<String, Vec<Value>>::new();
    for item in actions {
        let name = string_field(item, "name")?;
        raw_action_slots.insert(name.to_owned(), optional_array(item, "slots")?.to_vec());
        let id = mapped_id(
            item,
            "potter_id",
            "action",
            name,
            &mut action_ids,
            &mut mappings,
            "Action",
        )?;
        let fcurves = item
            .get("fcurves")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|curve| convert_fcurve(curve, &mappings))
            .collect::<Result<Vec<_>>>()?;
        doc.actions.insert(
            id.clone(),
            Action {
                name: name.to_owned(),
                fcurves,
                slots: Vec::new(),
            },
        );
        action_by_name.insert(name.to_owned(), id);
    }
    for item in meshes {
        let Some(action_name) = item
            .get("shape_keys")
            .and_then(|shape_keys| shape_keys.get("action"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        let mesh_name = string_field(item, "name")?;
        let data_id = mesh_ids
            .get(mesh_name)
            .ok_or_else(|| invalid_intermediate("shape-key mesh ID mapping is missing"))?;
        let action_id = action_by_name.get(action_name).cloned().ok_or_else(|| {
            invalid_intermediate("shape keys reference an unknown animation action")
        })?;
        let shape_keys = doc
            .data_blocks
            .get_mut(data_id)
            .and_then(|data| data.shape_keys.as_mut())
            .ok_or_else(|| invalid_intermediate("shape-key data block is missing"))?;
        shape_keys.action = Some(action_id);
    }
    let mut object_ids = BTreeMap::new();
    for item in objects {
        let name = string_field(item, "name")?;
        mapped_id(
            item,
            "potter_id",
            "node",
            name,
            &mut object_ids,
            &mut mappings,
            "Object",
        )?;
    }
    for (action_name, action_id) in &action_by_name {
        let mut slot_ids = BTreeMap::new();
        let mut slots = Vec::new();
        if let Some(slot_sources) = raw_action_slots.get(action_name) {
            for source in slot_sources {
                let slot_name = source
                    .get("name")
                    .and_then(Value::as_str)
                    .or_else(|| source.get("identifier").and_then(Value::as_str))
                    .unwrap_or("Action Slot");
                let node_name = source
                    .get("node")
                    .and_then(Value::as_str)
                    .or_else(|| {
                        objects.iter().find_map(|object| {
                            let same_action =
                                object.get("action").and_then(Value::as_str) == Some(action_name);
                            let reference = object.get("action_slot");
                            let same_slot = reference.is_some_and(|reference| {
                                reference.get("identifier").and_then(Value::as_str)
                                    == source.get("identifier").and_then(Value::as_str)
                                    || reference.get("name").and_then(Value::as_str)
                                        == source.get("name").and_then(Value::as_str)
                            });
                            (same_action && same_slot)
                                .then(|| object.get("name").and_then(Value::as_str))
                                .flatten()
                        })
                    })
                    .or_else(|| {
                        if source.get("target_type").and_then(Value::as_str) != Some("KEY") {
                            return None;
                        }
                        let shape_mesh_name = meshes.iter().find_map(|mesh| {
                            let shape_keys = mesh.get("shape_keys")?;
                            let same_action = shape_keys.get("action").and_then(Value::as_str)
                                == Some(action_name);
                            let shape_slot = shape_keys.get("action_slot").and_then(Value::as_str);
                            let same_slot = shape_slot
                                == source.get("name").and_then(Value::as_str)
                                || shape_slot == source.get("identifier").and_then(Value::as_str);
                            (same_action && same_slot)
                                .then(|| mesh.get("name").and_then(Value::as_str))
                                .flatten()
                        })?;
                        objects.iter().find_map(|object| {
                            (object.get("data_name").and_then(Value::as_str)
                                == Some(shape_mesh_name))
                            .then(|| object.get("name").and_then(Value::as_str))
                            .flatten()
                        })
                    });
                let Some(node_name) = node_name else {
                    continue;
                };
                let Some(node) = object_ids.get(node_name) else {
                    continue;
                };
                let slot_id = mapped_id(
                    source,
                    "potter_id",
                    "action_slot",
                    slot_name,
                    &mut slot_ids,
                    &mut mappings,
                    &format!("ActionSlot:{action_id}"),
                )?;
                slots.push(ActionSlot {
                    id: slot_id,
                    node: node.clone(),
                });
            }
        }
        if let Some(action) = doc.actions.get_mut(action_id) {
            action.slots = slots;
        }
    }
    for item in objects {
        let name = string_field(item, "name")?;
        let id = object_ids
            .get(name)
            .cloned()
            .ok_or_else(|| invalid_intermediate("object ID mapping is missing"))?;
        let object_type = item.get("type").and_then(Value::as_str).unwrap_or("EMPTY");
        let instance_collection_name = item
            .get("instance_collection")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty());
        let kind = if instance_collection_name.is_some() {
            "collection_instance"
        } else {
            match object_type {
                "MESH" => "mesh",
                "CAMERA" => "camera",
                "LIGHT" => "light",
                "ARMATURE" => "armature",
                "GREASE_PENCIL" | "GREASE_PENCIL_V3" => "grease_pencil",
                "CURVE" => "curve",
                "SURFACE" => "surface",
                "FONT" => "text",
                "VOLUME" => "volume",
                _ => "empty",
            }
        };
        let data_name = item.get("data_name").and_then(Value::as_str);
        let data_id = match object_type {
            "MESH" => data_name.and_then(|value| mesh_ids.get(value)),
            "CAMERA" => data_name.and_then(|value| camera_ids.get(value)),
            "LIGHT" => data_name.and_then(|value| light_ids.get(value)),
            "ARMATURE" => data_name.and_then(|value| armature_ids.get(value)),
            "GREASE_PENCIL" | "GREASE_PENCIL_V3" => {
                data_name.and_then(|value| grease_pencil_ids.get(value))
            }
            "CURVE" | "SURFACE" | "FONT" => data_name.and_then(|value| curve_ids.get(value)),
            "VOLUME" => data_name.and_then(|value| volume_ids.get(value)),
            _ => None,
        }
        .cloned();
        let rotation_mode = item
            .get("rotation_mode")
            .and_then(Value::as_str)
            .unwrap_or("XYZ")
            .to_owned();
        let rotation = blender_rotation(item, &rotation_mode);
        let mut properties = object_properties(item);
        if let Some(instance_name) = instance_collection_name {
            let collection_id = collection_ids.get(instance_name).ok_or_else(|| {
                invalid_intermediate("collection instance references an unknown collection")
            })?;
            properties.insert("instance_collection".to_owned(), json!(collection_id));
        }
        if !matches!(
            object_type,
            "MESH"
                | "EMPTY"
                | "CAMERA"
                | "LIGHT"
                | "ARMATURE"
                | "GREASE_PENCIL"
                | "GREASE_PENCIL_V3"
                | "CURVE"
                | "SURFACE"
                | "FONT"
                | "VOLUME"
        ) {
            properties.insert("blender_object_type".to_owned(), json!(object_type));
        }
        let mut modifier_ids = BTreeSet::new();
        let modifiers = item
            .get("modifiers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(index, modifier)| {
                let modifier_name = modifier
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("modifier");
                let modifier_type = modifier
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("UNKNOWN");
                let model_type = potter_modifier_type(modifier_type)
                    .map_or_else(|| modifier_type.to_lowercase(), str::to_owned);
                let mut params = modifier
                    .get("properties")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                if modifier_type == "MESH_SEQUENCE_CACHE" {
                    params = parse_mesh_sequence_cache_params(&params, &resource_ids)?;
                } else if modifier_type == "MESH_CACHE" {
                    params = parse_mesh_cache_params(&params, &resource_ids)?;
                }
                if modifier_type == "NODES"
                    && let Some(group_name) = params.get("node_group").and_then(Value::as_str)
                {
                    let group_id = node_group_ids.get(group_name).ok_or_else(|| {
                        invalid_intermediate("modifier references an unknown node group")
                    })?;
                    params.insert("node_group".to_owned(), json!(group_id.as_str()));
                }
                if crate::params::type_spec(crate::params::ParameterFamily::Modifier, &model_type)
                    .is_some()
                {
                    for (parameter, value) in &mut params {
                        if crate::params::is_id_reference_parameter(
                            crate::params::ParameterFamily::Modifier,
                            &model_type,
                            parameter,
                        ) {
                            canonicalize_blender_id_references(
                                value,
                                &object_ids,
                                &movie_clip_ids,
                                &action_ids,
                            )?;
                        }
                    }
                } else {
                    for value in params.values_mut() {
                        canonicalize_blender_id_references(
                            value,
                            &object_ids,
                            &movie_clip_ids,
                            &action_ids,
                        )?;
                    }
                }
                if modifier_type == "UV_PROJECT" {
                    let projectors = params
                        .get("projectors")
                        .and_then(Value::as_array)
                        .ok_or_else(|| invalid_intermediate("UV Project projectors are invalid"))?
                        .iter()
                        .filter_map(|projector| {
                            projector
                                .get("object")
                                .filter(|object| !object.is_null())
                                .cloned()
                        })
                        .collect::<Vec<_>>();
                    params.insert("projectors".to_owned(), Value::Array(projectors));
                }
                if modifier_type == "BOOLEAN"
                    && params.get("operand_type").and_then(Value::as_str) == Some("COLLECTION")
                {
                    let collection_name = params
                        .get("collection")
                        .and_then(|value| value.get("name").and_then(Value::as_str))
                        .ok_or_else(|| {
                            invalid_intermediate(
                                "Boolean modifier references an invalid collection",
                            )
                        })?;
                    let collection_id = collection_ids.get(collection_name).ok_or_else(|| {
                        invalid_intermediate("Boolean modifier references an unknown collection")
                    })?;
                    params.insert("collection".to_owned(), json!(collection_id.as_str()));
                    params.remove("object");
                }
                if params.get("collection").is_some_and(Value::is_null) {
                    params.remove("collection");
                }
                if modifier_type == "SIMPLE_DEFORM"
                    && params.get("origin").is_some_and(Value::is_null)
                {
                    params.remove("origin");
                }
                let modifier_id = modifier
                    .get("potter_id")
                    .and_then(Value::as_str)
                    .filter(|candidate| {
                        crate::model::is_valid_id(candidate) && !modifier_ids.contains(*candidate)
                    })
                    .map_or_else(
                        || {
                            unique_id(
                                &format!("modifier_{name}_{modifier_name}_{index}"),
                                "modifier",
                                &mut modifier_ids,
                            )
                        },
                        str::to_owned,
                    );
                let physics_property = match model_type.as_str() {
                    "cloth" => Some("physics_cloth"),
                    "soft_body" => Some("physics_soft_body"),
                    "collision" => Some("physics_collision"),
                    "dynamic_paint" => Some("physics_dynamic_paint"),
                    "fluid" => Some("physics_fluid"),
                    "particle_system" => Some("physics_particle_emitter"),
                    _ => None,
                };
                if let Some(property) = physics_property {
                    params.insert("settings_id".to_owned(), json!(modifier_id));
                    let mut settings = modifier
                        .get("physics_settings")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    canonicalize_blender_id_references(
                        &mut settings,
                        &object_ids,
                        &movie_clip_ids,
                        &action_ids,
                    )?;
                    if model_type == "particle_system" {
                        if !properties.contains_key(property) {
                            properties.insert(property.to_owned(), settings.clone());
                            properties.insert(
                                "physics_particle_emitter_modifier_id".to_owned(),
                                json!(modifier_id),
                            );
                        }
                        let registry = properties
                            .entry("physics_particle_systems".to_owned())
                            .or_insert_with(|| json!({}))
                            .as_object_mut()
                            .ok_or_else(|| {
                                invalid_intermediate("particle settings registry is not an object")
                            })?;
                        registry.insert(modifier_id.clone(), settings);
                    } else {
                        properties.insert(property.to_owned(), settings);
                    }
                }
                modifier_ids.insert(modifier_id.clone());
                mappings.insert(
                    format!("Modifier:{}:{modifier_name}", id.as_str()),
                    modifier_id.clone(),
                );
                Ok(Modifier {
                    id: Id::new(modifier_id)?,
                    modifier_type: potter_modifier_type(modifier_type)
                        .map_or_else(|| modifier_type.to_lowercase(), str::to_owned),
                    name: modifier_name.to_owned(),
                    enabled: modifier
                        .get("enabled")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                    params,
                    binding_data: native_bindings
                        .get(&(name.to_owned(), modifier_name.to_owned()))
                        .cloned(),
                    runtime: crate::model::ModifierRuntime::default(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let material_slots = item
            .get("materials")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter_map(|material_name| material_ids.get(material_name).cloned())
            .collect();
        let parent = item
            .get("parent")
            .and_then(Value::as_str)
            .and_then(|parent_name| object_ids.get(parent_name))
            .cloned();
        let parent_bone = if item.get("parent_type").and_then(Value::as_str) == Some("BONE") {
            let parent_name = item
                .get("parent")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_intermediate("bone parent has no parent object"))?;
            let parent_object = objects
                .iter()
                .find(|parent| parent.get("name").and_then(Value::as_str) == Some(parent_name))
                .ok_or_else(|| invalid_intermediate("bone parent object was not imported"))?;
            let armature_name = parent_object
                .get("data_name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_intermediate("bone parent has no armature data"))?;
            let armature_id = armature_ids
                .get(armature_name)
                .ok_or_else(|| invalid_intermediate("bone parent armature was not imported"))?;
            let bone_name = item
                .get("parent_bone")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| invalid_intermediate("bone parent has no bone name"))?;
            Some(
                armature_bones_by_data
                    .get(armature_id)
                    .and_then(|bones| bones.get(bone_name))
                    .cloned()
                    .ok_or_else(|| invalid_intermediate("bone parent bone was not imported"))?,
            )
        } else {
            None
        };
        let mut transform = Transform {
            translation: vec3_field(item, "location", [0.0; 3]),
            rotation,
            scale: vec3_field(item, "scale", [1.0; 3]),
            rotation_mode,
        };
        let mut parent_inverse = matrix_column_major(item.get("matrix_parent_inverse"));
        if let Some((stored_transform, stored_parent_inverse)) = root_matrix_override(item) {
            transform = stored_transform;
            parent_inverse = Some(stored_parent_inverse);
        }
        let mut constraints = parse_constraints(
            item,
            &id,
            &object_ids,
            &movie_clip_ids,
            &action_ids,
            &resource_ids,
            &mut mappings,
            None,
        )?;
        if kind == "armature"
            && let Some(armature_data_id) = data_id.as_ref()
            && let Some(bones_by_name) = armature_bones_by_data.get(armature_data_id)
        {
            for pose_item in optional_array(item, "pose")? {
                let Some(bone_name) = pose_item.get("bone").and_then(Value::as_str) else {
                    continue;
                };
                let Some(owner_bone) = bones_by_name.get(bone_name) else {
                    continue;
                };
                let pose_constraints = optional_array(pose_item, "constraints")?;
                if !pose_constraints.is_empty() {
                    let wrapped = json!({"constraints": pose_constraints});
                    constraints.extend(parse_constraints(
                        &wrapped,
                        &id,
                        &object_ids,
                        &movie_clip_ids,
                        &action_ids,
                        &resource_ids,
                        &mut mappings,
                        Some(owner_bone),
                    )?);
                }
            }
        }
        let drivers = parse_drivers(item, &id, &object_ids, &mut mappings)?;
        let nla_tracks = parse_nla_tracks(item, &id, &action_by_name, &mut mappings)?;
        let rigid_body = item
            .get("rigid_body")
            .filter(|value| !value.is_null())
            .map(parse_rigid_body);
        let force_field = item
            .get("force_field")
            .filter(|value| !value.is_null())
            .map(parse_force_field);
        let pose_data_id = item
            .get("data_name")
            .and_then(Value::as_str)
            .and_then(|name| armature_ids.get(name));
        let mut pose = BTreeMap::new();
        for pose_item in optional_array(item, "pose")? {
            let Some(bone_name) = pose_item.get("bone").and_then(Value::as_str) else {
                continue;
            };
            let Some(bone_id) = pose_data_id
                .and_then(|armature_id| armature_bones_by_data.get(armature_id))
                .and_then(|bones| bones.get(bone_name))
            else {
                continue;
            };
            let rotation = vec4_field(pose_item, "rotation_quaternion", [1.0, 0.0, 0.0, 0.0]);
            pose.insert(
                bone_id.clone(),
                crate::model::PoseBone {
                    translation: vec3_field(pose_item, "location", [0.0; 3]),
                    rotation: [rotation[1], rotation[2], rotation[3], rotation[0]],
                    scale: vec3_field(pose_item, "scale", [1.0; 3]),
                    lock_ik: bool3_field(pose_item, "lock_ik", [false; 3]),
                    use_ik_limit: bool3_field(pose_item, "use_ik_limit", [false; 3]),
                    ik_min: vec3_field(pose_item, "ik_min", [-std::f64::consts::PI; 3]),
                    ik_max: vec3_field(pose_item, "ik_max", [std::f64::consts::PI; 3]),
                    ik_stiffness: vec3_field(pose_item, "ik_stiffness", [0.0; 3]),
                    ik_stretch: number_field(pose_item, "ik_stretch", 0.0),
                },
            );
        }
        let node = Node {
            name: name.to_owned(),
            kind: kind.to_owned(),
            primitive: None,
            tags: item
                .get("custom_properties")
                .and_then(Value::as_object)
                .and_then(|props| props.get("potter.tags"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
            parent,
            parent_type: if item.get("parent_type").and_then(Value::as_str) == Some("BONE") {
                ParentType::Bone
            } else {
                ParentType::Object
            },
            parent_bone,
            parent_inverse,
            transform,
            data: data_id,
            materials: material_slots,
            modifiers,
            visible: !item
                .get("hide_viewport")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            render_visible: !item
                .get("hide_render")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            selectable: !item
                .get("hide_select")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            action: item
                .get("action")
                .and_then(Value::as_str)
                .and_then(|action_name| action_by_name.get(action_name))
                .cloned(),
            nla_tracks,
            properties,
            rigid_body,
            force_field,
            constraints,
            drivers,
            pose,
        };
        doc.nodes.insert(id.clone(), node);
        if let Some(mesh_id) = data_name.and_then(|name| mesh_ids.get(name)) {
            let mut group_ids = BTreeMap::new();
            for group_source in optional_array(item, "vertex_groups")? {
                let group_name = group_source
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("Vertex Group");
                let group_id = mapped_id(
                    group_source,
                    "potter_id",
                    "vertex_group",
                    group_name,
                    &mut group_ids,
                    &mut mappings,
                    &format!("VertexGroup:{mesh_id}"),
                )?;
                let Some(data) = doc.data_blocks.get_mut(mesh_id) else {
                    continue;
                };
                if !data.vertex_groups.iter().any(|group| group.id == group_id) {
                    data.vertex_groups.push(VertexGroup {
                        id: group_id.clone(),
                        name: group_name.to_owned(),
                    });
                }
                for weight in optional_array(group_source, "weights")? {
                    let vertex = weight.get("vertex_index").ok_or_else(|| {
                        invalid_intermediate("vertex-group weight has no vertex index")
                    })?;
                    let vertex_id = imported_vertex_id(vertex)?;
                    let amount = number_field(weight, "weight", 0.0);
                    data.vertex_weights
                        .entry(vertex_id)
                        .or_default()
                        .insert(group_id.clone(), amount);
                }
            }
            if let (Some(group_names), Some(data)) = (
                shape_key_vertex_groups.get(mesh_id),
                doc.data_blocks.get_mut(mesh_id),
            ) && let Some(shape_keys) = data.shape_keys.as_mut()
            {
                for (key_id, group_name) in group_names {
                    let Some(group_id) = data
                        .vertex_groups
                        .iter()
                        .find(|group| group.name == *group_name)
                        .map(|group| group.id.clone())
                    else {
                        continue;
                    };
                    if let Some(shape_key) = shape_keys.keys.get_mut(key_id) {
                        shape_key.vertex_group = Some(group_id);
                    }
                }
            }
        }
    }
    for item in collections {
        let name = string_field(item, "name")?;
        if let Some(collection_id) = collection_ids.get(name).cloned()
            && let Some(collection) = doc.collections.get_mut(&collection_id)
        {
            collection.children = item
                .get("children")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .filter_map(|child| collection_ids.get(child).cloned())
                .collect();
            collection.objects = item
                .get("objects")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .filter_map(|object| object_ids.get(object).cloned())
                .collect();
        }
    }
    for item in scenes {
        let name = string_field(item, "name")?;
        if let Some(scene_id) = scene_ids.get(name).cloned()
            && let Some(scene) = doc.scenes.get_mut(&scene_id)
        {
            scene.camera = item
                .get("camera")
                .and_then(Value::as_str)
                .and_then(|camera_name| object_ids.get(camera_name))
                .cloned();
        }
    }
    let mut generic_ids = BTreeMap::<String, BTreeMap<String, Id>>::new();
    for (collection, kind, prefix) in [("texts", "Text", "text")] {
        for item in array(raw, collection)? {
            let name = string_field(item, "name")?;
            let ids = generic_ids.entry(kind.to_owned()).or_default();
            mapped_id(item, "potter_id", prefix, name, ids, &mut mappings, kind)?;
        }
    }
    for item in array(raw, "other_datablocks")? {
        let name = string_field(item, "name")?;
        let block_type = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let kind = format!("DataBlock:{block_type}");
        let ids = generic_ids.entry(kind.clone()).or_default();
        mapped_id(
            item,
            "potter_id",
            block_type,
            name,
            ids,
            &mut mappings,
            &kind,
        )?;
    }
    let mut library_registry_ids = BTreeMap::new();
    library_registry_ids.insert("nodes".to_owned(), object_ids.clone());
    library_registry_ids.insert("collections".to_owned(), collection_ids.clone());
    library_registry_ids.insert("materials".to_owned(), material_ids.clone());
    library_registry_ids.insert("actions".to_owned(), action_ids.clone());
    library_registry_ids.insert("node_groups".to_owned(), node_group_ids.clone());
    library_registry_ids.insert("worlds".to_owned(), world_ids.clone());
    let mut data_block_ids = mesh_ids.clone();
    data_block_ids.extend(camera_ids.clone());
    data_block_ids.extend(light_ids.clone());
    data_block_ids.extend(armature_ids.clone());
    data_block_ids.extend(grease_pencil_ids.clone());
    data_block_ids.extend(curve_ids.clone());
    data_block_ids.extend(volume_ids.clone());
    library_registry_ids.insert("data_blocks".to_owned(), data_block_ids);
    let mut linked_resource_ids = BTreeMap::new();
    for image in images {
        let name = string_field(image, "name")?;
        let Some(filepath) = image.get("filepath").and_then(Value::as_str) else {
            continue;
        };
        if let Some(resource_id) = resource_ids.get(filepath) {
            linked_resource_ids.insert(name.to_owned(), Id::new(resource_id.clone())?);
        }
    }
    library_registry_ids.insert("resources".to_owned(), linked_resource_ids);
    register_blend_libraries(
        raw,
        &resource_ids,
        &library_registry_ids,
        &mut doc,
        &mut mappings,
    )?;
    validate_native_modifier_bindings(&mut doc)?;
    doc.profile = Profile {
        blender: raw
            .get("blender_version")
            .and_then(Value::as_str)
            .unwrap_or("5.2.0")
            .to_owned(),
    };
    let losses = import_losses(raw);
    let id_mappings = serde_json::to_value(&mappings).map_err(PotError::internal_json)?;
    let source = json!({
        "format": "blend",
        "path": file.display().to_string(),
        "blender_version": doc.profile.blender,
        "file_version": raw.get("file_version"),
        "active_scene": active_name,
        "id_mappings": id_mappings.clone(),
    });
    doc.compatibility.insert(
        "blender_adapter".to_owned(),
        json!({"intermediate": raw, "id_mappings": id_mappings.clone()}),
    );
    doc.validate()?;
    attach_native_target_vertex_orders(&mut doc);
    Ok(ImportedGraph {
        doc,
        losses,
        id_mappings,
        compat_blobs: vec![("original.blend".to_owned(), original)],
        assets: resource_assets,
        source,
    })
}
fn imported_image_from_resource(
    item: &Value,
    resource_ids: &BTreeMap<String, String>,
    resource_assets: &[(String, Vec<u8>)],
    doc: &SceneDoc,
) -> Result<Option<Image>> {
    let name = string_field(item, "name")?;
    if item
        .get("source")
        .and_then(Value::as_str)
        .is_some_and(|source| source != "FILE")
    {
        return Ok(None);
    }
    let resource_id = item
        .get("filepath")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty())
        .and_then(|path| resource_ids.get(path))
        .or_else(|| resource_ids.get(name));
    let Some(resource_id) = resource_id else {
        return Ok(None);
    };
    let resource_id = Id::new(resource_id.clone())?;
    let Some(resource) = doc.resources.get(&resource_id) else {
        return Ok(None);
    };
    let Some(source_hash) = resource.get("hash").and_then(Value::as_str) else {
        return Ok(None);
    };
    let Some((_, bytes)) = resource_assets
        .iter()
        .find(|(digest, _)| digest.as_str() == source_hash)
    else {
        return Ok(None);
    };
    let colorspace = match item
        .get("colorspace")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_ascii_lowercase)
    {
        Some(name) if name.contains("non-color") || name.contains("non_color") || name == "raw" => {
            Some(ImageColorspace::NonColor)
        }
        Some(name) if name.contains("linear") => Some(ImageColorspace::Linear),
        Some(_) => Some(ImageColorspace::Srgb),
        None => None,
    };
    let (colorspace, width, height, _) =
        crate::image::storage::decode_pixels_with_default_colorspace(bytes, colorspace)?;
    Ok(Some(Image {
        name: name.to_owned(),
        source: ImageSource::Packed,
        colorspace,
        width,
        height,
        tiles: Vec::new(),
        blob: Some(source_hash.to_owned()),
        source_path: None,
        source_hash: Some(source_hash.to_owned()),
        alpha_mode: ImageAlphaMode::Straight,
    }))
}

fn resolve_image_node_ids(graph_sources: &mut [Value], image_ids: &BTreeMap<String, Id>) {
    for source in graph_sources {
        let Some(nodes) = source
            .get_mut("tree")
            .and_then(|tree| tree.get_mut("nodes"))
            .and_then(Value::as_array_mut)
        else {
            continue;
        };
        for node in nodes {
            let Some(properties) = node.get_mut("properties").and_then(Value::as_object_mut) else {
                continue;
            };
            for key in ["image", "image_id"] {
                let image_name = properties.get(key).and_then(|reference| match reference {
                    Value::Object(reference)
                        if reference.get("id_type").and_then(Value::as_str) == Some("Image") =>
                    {
                        reference.get("name").and_then(Value::as_str)
                    }
                    Value::String(name) => Some(name.as_str()),
                    _ => None,
                });
                if let Some(image_id) = image_name.and_then(|name| image_ids.get(name)) {
                    properties.insert(key.to_owned(), json!(image_id.as_str()));
                }
            }
        }
    }
}

fn import_grease_pencils(
    items: &[Value],
    material_ids: &BTreeMap<String, Id>,
    doc: &mut SceneDoc,
    mappings: &mut BTreeMap<String, String>,
) -> Result<BTreeMap<String, Id>> {
    let mut data_ids = BTreeMap::new();
    for item in items {
        let name = string_field(item, "name")?;
        let data_id = mapped_id(
            item,
            "potter_id",
            "grease_pencil",
            name,
            &mut data_ids,
            mappings,
            "GreasePencil",
        )?;
        let mut layer_ids = BTreeMap::new();
        let mut stroke_ids = BTreeMap::new();
        let mut layers = Vec::new();
        for raw_layer in optional_array(item, "layers")? {
            let layer_name = string_field(raw_layer, "name")?;
            let layer_id = mapped_id(
                raw_layer,
                "potter_id",
                "gp_layer",
                layer_name,
                &mut layer_ids,
                mappings,
                &format!("GreasePencilLayer:{name}"),
            )?;
            let mut frames = Vec::new();
            for (frame_index, raw_frame) in optional_array(raw_layer, "frames")?.iter().enumerate()
            {
                let mut strokes = Vec::new();
                for (stroke_index, raw_stroke) in
                    optional_array(raw_frame, "strokes")?.iter().enumerate()
                {
                    let generated_name =
                        format!("{name}_{layer_name}_frame_{frame_index}_stroke_{stroke_index}");
                    let stroke_name = raw_stroke
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or(&generated_name);
                    let stroke_id = mapped_id(
                        raw_stroke,
                        "potter_id",
                        "gp_stroke",
                        stroke_name,
                        &mut stroke_ids,
                        mappings,
                        &format!("GreasePencilStroke:{name}:{layer_name}"),
                    )?;
                    let points = optional_array(raw_stroke, "points")?
                        .iter()
                        .map(|point| GreasePencilPoint {
                            position: vec3_field(point, "position", [0.0; 3]),
                            pressure: number_field(point, "pressure", 1.0),
                            radius: number_field(point, "radius", 0.01),
                            opacity: number_field(point, "opacity", 1.0),
                            time: number_field(point, "time", 0.0),
                        })
                        .collect();
                    let material = match raw_stroke.get("material") {
                        None | Some(Value::Null) => None,
                        Some(Value::String(material_name)) => {
                            Some(material_ids.get(material_name).cloned().ok_or_else(|| {
                                invalid_intermediate(
                                    "Grease Pencil stroke references an unknown material",
                                )
                            })?)
                        }
                        Some(_) => {
                            return Err(invalid_intermediate(
                                "Grease Pencil stroke material must be a material name or null",
                            ));
                        }
                    };
                    let fill = match raw_stroke.get("fill") {
                        None | Some(Value::Null) => None,
                        Some(Value::Array(_)) => Some(vec4_field(raw_stroke, "fill", [0.0; 4])),
                        Some(_) => {
                            return Err(invalid_intermediate(
                                "Grease Pencil stroke fill must be a color or null",
                            ));
                        }
                    };
                    strokes.push(GreasePencilStroke {
                        id: stroke_id,
                        points,
                        material,
                        cyclic: raw_stroke
                            .get("cyclic")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                        fill,
                    });
                }
                frames.push(GreasePencilFrame {
                    frame: number_field(raw_frame, "frame", 1.0),
                    strokes,
                });
            }
            layers.push(GreasePencilLayer {
                id: layer_id,
                name: layer_name.to_owned(),
                opacity: number_field(raw_layer, "opacity", 1.0),
                visible: raw_layer
                    .get("visible")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                frames,
            });
        }
        doc.data_blocks.insert(
            data_id,
            DataBlock {
                data_type: "grease_pencil".to_owned(),
                mesh: None,
                grease_pencil: Some(GreasePencilData { layers }),
                ..DataBlock::default()
            },
        );
    }
    Ok(data_ids)
}

fn import_node_groups(
    items: &[Value],
    doc: &mut SceneDoc,
    mappings: &mut BTreeMap<String, String>,
) -> Result<BTreeMap<String, Id>> {
    let mut group_ids = BTreeMap::new();
    for item in items {
        let name = string_field(item, "name")?;
        let group_id = mapped_id(
            item,
            "potter_id",
            "node_group",
            name,
            &mut group_ids,
            mappings,
            "NodeGroup",
        )?;
        let tree = item
            .get("tree")
            .filter(|tree| !tree.is_null())
            .unwrap_or(item);
        let tree_type = tree
            .get("type")
            .or_else(|| item.get("type"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let kind = match tree_type {
            "GeometryNodeTree" => GraphKind::Geometry,
            "ShaderNodeTree" => GraphKind::Shader,
            "CompositorNodeTree" => GraphKind::Compositor,
            _ => {
                return Err(invalid_intermediate(&format!(
                    "unsupported Blender node graph type `{tree_type}`"
                )));
            }
        };
        let interface = tree.get("interface");
        let inputs = interface
            .map(|value| optional_array(value, "inputs"))
            .transpose()?
            .unwrap_or(&[]);
        let outputs = interface
            .map(|value| optional_array(value, "outputs"))
            .transpose()?
            .unwrap_or(&[]);
        let mut group = NodeGroup::new(name, kind);
        group.interface = GraphInterface {
            inputs: convert_graph_sockets(inputs),
            outputs: convert_graph_sockets(outputs),
        };
        let mut node_ids = BTreeMap::new();
        for raw_node in optional_array(tree, "nodes")? {
            let node_name = string_field(raw_node, "name")?;
            let node_id = mapped_id(
                raw_node,
                "potter_id",
                "graph_node",
                node_name,
                &mut node_ids,
                mappings,
                &format!("Node:{name}"),
            )?;
            let mut properties = raw_node
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            if let Some(custom_properties) = raw_node.get("custom_properties")
                && custom_properties
                    .as_object()
                    .is_some_and(|properties| !properties.is_empty())
            {
                properties.insert(
                    "_blender_custom_properties".to_owned(),
                    custom_properties.clone(),
                );
            }
            let mut location = [0.0; 2];
            if let Some(values) = raw_node.get("location").and_then(Value::as_array) {
                for (target, source) in location.iter_mut().zip(values) {
                    if let Some(value) = source.as_f64() {
                        *target = value;
                    }
                }
            }
            group.nodes.insert(
                node_id,
                GraphNode {
                    node_type: string_field(raw_node, "type")?.to_owned(),
                    name: node_name.to_owned(),
                    location,
                    properties,
                    inputs: graph_node_inputs(raw_node)?,
                },
            );
        }
        for raw_link in optional_array(tree, "links")? {
            let from_name = string_field(raw_link, "from_node")?;
            let to_name = string_field(raw_link, "to_node")?;
            let from_node = node_ids.get(from_name).cloned().ok_or_else(|| {
                invalid_intermediate("node graph link references an unknown source node")
            })?;
            let to_node = node_ids.get(to_name).cloned().ok_or_else(|| {
                invalid_intermediate("node graph link references an unknown destination node")
            })?;
            group.links.push(GraphLink {
                from_node,
                from_socket: string_field(raw_link, "from_socket")?.to_owned(),
                to_node,
                to_socket: string_field(raw_link, "to_socket")?.to_owned(),
            });
        }
        doc.node_groups.insert(group_id, group);
    }
    Ok(group_ids)
}

fn convert_graph_sockets(values: &[Value]) -> Vec<GraphSocket> {
    values
        .iter()
        .map(|socket| {
            let id = socket
                .get("id")
                .and_then(Value::as_str)
                .or_else(|| socket.get("name").and_then(Value::as_str))
                .unwrap_or_default()
                .to_owned();
            let name = socket
                .get("name")
                .and_then(Value::as_str)
                .map_or_else(|| id.clone(), str::to_owned);
            let socket_type = socket
                .get("socket_type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            GraphSocket {
                id,
                name,
                socket_type: blender_graph_socket_type(socket_type).to_owned(),
                default: socket.get("default").cloned().unwrap_or(Value::Null),
            }
        })
        .collect()
}
fn blender_graph_socket_type(value: &str) -> &str {
    match value {
        "NodeSocketGeometry" => "geometry",
        "NodeSocketFloat" => "float",
        "NodeSocketInt" => "integer",
        "NodeSocketBool" => "boolean",
        "NodeSocketVector" => "vector",
        "NodeSocketRotation" => "rotation",
        "NodeSocketColor" => "color",
        "NodeSocketString" => "string",
        "NodeSocketObject" => "object",
        "NodeSocketCollection" => "collection",
        "NodeSocketMaterial" => "material",
        other => other,
    }
}

fn graph_node_inputs(raw_node: &Value) -> Result<BTreeMap<String, Value>> {
    match raw_node.get("inputs") {
        None | Some(Value::Null) => Ok(BTreeMap::new()),
        Some(Value::Object(inputs)) => Ok(inputs
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()),
        Some(Value::Array(inputs)) => {
            let mut converted = BTreeMap::new();
            for input in inputs {
                let name = input
                    .get("identifier")
                    .and_then(Value::as_str)
                    .or_else(|| input.get("name").and_then(Value::as_str))
                    .ok_or_else(|| invalid_intermediate("node graph input has no socket name"))?;
                converted.insert(
                    name.to_owned(),
                    input.get("default").cloned().unwrap_or(Value::Null),
                );
            }
            Ok(converted)
        }
        Some(_) => Err(invalid_intermediate(
            "node graph inputs must be an object or array",
        )),
    }
}

fn array<'a>(raw: &'a Value, key: &str) -> Result<&'a [Value]> {
    raw.get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_intermediate(&format!("invalid `{key}` array")))
}
fn optional_array<'a>(raw: &'a Value, key: &str) -> Result<&'a [Value]> {
    match raw.get(key) {
        None => Ok(&[]),
        Some(Value::Array(values)) => Ok(values),
        Some(_) => Err(invalid_intermediate(&format!("invalid `{key}` array"))),
    }
}
fn parse_rigid_body_world(value: &Value) -> Result<RigidBodyWorld> {
    Ok(RigidBodyWorld {
        enabled: value
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        gravity: vec3_field(value, "gravity", [0.0, 0.0, -9.81]),
        substeps: u32::try_from(integer_field(value, "substeps", 4))
            .map_err(|_| invalid_intermediate("rigid-body substeps are out of range"))?,
        solver_iterations: u32::try_from(integer_field(value, "solver_iterations", 10))
            .map_err(|_| invalid_intermediate("rigid-body solver iterations are out of range"))?,
        frame_start: i32::try_from(integer_field(value, "frame_start", 1))
            .map_err(|_| invalid_intermediate("rigid-body frame_start is out of range"))?,
        frame_end: i32::try_from(integer_field(value, "frame_end", 250))
            .map_err(|_| invalid_intermediate("rigid-body frame_end is out of range"))?,
        seed: u32::try_from(integer_field(value, "seed", 0))
            .map_err(|_| invalid_intermediate("rigid-body seed is out of range"))?,
    })
}

fn parse_rigid_body(value: &Value) -> RigidBody {
    let body_type = match value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("ACTIVE")
    {
        "PASSIVE" => RigidBodyType::Passive,
        _ => RigidBodyType::Active,
    };
    let shape = match value.get("shape").and_then(Value::as_str).unwrap_or("BOX") {
        "SPHERE" => RigidBodyShape::Sphere,
        "CONVEX_HULL" => RigidBodyShape::ConvexHull,
        "MESH" => RigidBodyShape::Mesh,
        _ => RigidBodyShape::Box,
    };
    let velocity = value
        .get("initial_velocity")
        .and_then(Value::as_array)
        .filter(|items| items.len() == 3)
        .map_or([0.0; 3], |items| {
            [
                items[0].as_f64().unwrap_or(0.0),
                items[1].as_f64().unwrap_or(0.0),
                items[2].as_f64().unwrap_or(0.0),
            ]
        });
    RigidBody {
        body_type,
        mass: number_field(value, "mass", 1.0),
        friction: number_field(value, "friction", 0.5),
        restitution: number_field(value, "restitution", 0.0),
        shape,
        linear_damping: number_field(value, "linear_damping", 0.04),
        angular_damping: number_field(value, "angular_damping", 0.1),
        initial_velocity: velocity,
    }
}

fn parse_force_field(value: &Value) -> ForceField {
    let field_type = match value.get("type").and_then(Value::as_str).unwrap_or("FORCE") {
        "WIND" => ForceFieldType::Wind,
        "VORTEX" => ForceFieldType::Vortex,
        _ => ForceFieldType::Force,
    };
    ForceField {
        field_type,
        strength: number_field(value, "strength", 1.0),
        falloff: number_field(value, "falloff", 0.0),
    }
}
fn parse_sequencer(
    scene: &Value,
    mappings: &mut BTreeMap<String, String>,
) -> Result<crate::sequencer::Sequencer> {
    let Some(raw) = scene.get("sequencer").filter(|value| !value.is_null()) else {
        return Ok(crate::sequencer::Sequencer::default());
    };
    let raw_strips = optional_array(raw, "strips")?;
    let mut strip_ids = BTreeMap::new();
    let mut strip_names = BTreeMap::new();
    for strip in raw_strips {
        let name = strip.get("name").and_then(Value::as_str).unwrap_or("Strip");
        let id = mapped_id(
            strip,
            "potter_id",
            "sequence_strip",
            name,
            &mut strip_ids,
            mappings,
            "SequencerStrip",
        )?;
        strip_names.insert(name.to_owned(), id.as_str().to_owned());
    }
    let strips = raw_strips
        .iter()
        .map(|strip| {
            let name = strip.get("name").and_then(Value::as_str).unwrap_or("Strip");
            let id = strip_ids
                .get(name)
                .ok_or_else(|| invalid_intermediate("sequencer strip ID mapping is missing"))?
                .as_str()
                .to_owned();
            let kind = match strip.get("type").and_then(Value::as_str).unwrap_or("image") {
                "image_sequence" => SequenceStripType::ImageSequence,
                "movie" => SequenceStripType::Movie,
                "sound" => SequenceStripType::Sound,
                "scene" => SequenceStripType::Scene,
                "color" => SequenceStripType::Color,
                "text" => SequenceStripType::Text,
                "meta" => SequenceStripType::Meta,
                "transition" => SequenceStripType::Transition,
                "effect" => SequenceStripType::Effect,
                _ => SequenceStripType::Image,
            };
            let blend_type = match strip
                .get("blend_type")
                .and_then(Value::as_str)
                .unwrap_or("replace")
            {
                "alpha_over" => StripBlendType::AlphaOver,
                "add" => StripBlendType::Add,
                "subtract" => StripBlendType::Subtract,
                "multiply" => StripBlendType::Multiply,
                _ => StripBlendType::Replace,
            };
            let transition = match strip.get("transition").and_then(Value::as_str) {
                Some("gamma_cross") => Some(SequenceTransitionType::GammaCross),
                Some("wipe") => Some(SequenceTransitionType::Wipe),
                Some("cross") => Some(SequenceTransitionType::Cross),
                _ => None,
            };
            let effect = match strip.get("effect").and_then(Value::as_str) {
                Some("add") => Some(SequenceEffectType::Add),
                Some("subtract") => Some(SequenceEffectType::Subtract),
                Some("multiply") => Some(SequenceEffectType::Multiply),
                Some("alpha_over") => Some(SequenceEffectType::AlphaOver),
                Some("transform") => Some(SequenceEffectType::Transform),
                Some("speed") => Some(SequenceEffectType::Speed),
                Some("glow") => Some(SequenceEffectType::Glow),
                Some("gaussian_blur") => Some(SequenceEffectType::GaussianBlur),
                _ => None,
            };
            let modifiers = optional_array(strip, "modifiers")?
                .iter()
                .map(|modifier| {
                    let curves = optional_array(modifier, "curves")?
                        .iter()
                        .map(|point| {
                            let values = point
                                .as_array()
                                .filter(|values| values.len() == 2)
                                .ok_or_else(|| {
                                    invalid_intermediate("sequencer curve point is invalid")
                                })?;
                            Ok([
                                values[0].as_f64().unwrap_or(0.0),
                                values[1].as_f64().unwrap_or(0.0),
                            ])
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(StripModifier {
                        brightness: number_field(modifier, "brightness", 0.0),
                        contrast: number_field(modifier, "contrast", 0.0),
                        color_balance: vec3_field(modifier, "color_balance", [0.0; 3]),
                        curves,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let retiming_keys = optional_array(strip, "retiming_keys")?
                .iter()
                .map(|key| RetimingKey {
                    frame: number_field(key, "frame", 0.0),
                    source_frame: number_field(key, "source_frame", 0.0),
                })
                .collect();
            let inputs = optional_array(strip, "inputs")?
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|name| strip_names.get(name).cloned())
                .collect();
            let color =
                vec4_field(strip, "color", [0.0, 0.0, 0.0, 1.0]).map(|component| component as f32);
            Ok(SequenceStrip {
                id,
                name: name.to_owned(),
                kind,
                channel: u32::try_from(integer_field(strip, "channel", 1))
                    .map_err(|_| invalid_intermediate("sequencer channel is out of range"))?,
                frame_start: number_field(strip, "frame_start", 1.0),
                frame_offset_start: number_field(strip, "frame_offset_start", 0.0),
                frame_offset_end: number_field(strip, "frame_offset_end", 0.0),
                length: number_field(strip, "length", 1.0),
                blend_type,
                opacity: number_field(strip, "opacity", 1.0),
                mute: strip.get("mute").and_then(Value::as_bool).unwrap_or(false),
                retiming_keys,
                modifiers,
                sound_volume: number_field(strip, "sound_volume", 1.0),
                sound_pan: number_field(strip, "sound_pan", 0.0),
                sound_pitch: number_field(strip, "sound_pitch", 1.0),
                source: strip
                    .get("source")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                color,
                text: strip.get("text").and_then(Value::as_str).map(str::to_owned),
                transition,
                inputs,
                effect,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(crate::sequencer::Sequencer {
        channels: u32::try_from(integer_field(raw, "channels", 32))
            .map_err(|_| invalid_intermediate("sequencer channel count is out of range"))?,
        strips,
    })
}
fn parse_color_management(value: Option<&Value>) -> Result<ColorManagement> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(ColorManagement::default());
    };
    let view_transform = match value
        .get("view_transform")
        .and_then(Value::as_str)
        .unwrap_or("standard")
        .to_ascii_lowercase()
        .as_str()
    {
        "standard" => ViewTransform::Standard,
        "agx" | "ag_x" => ViewTransform::AgX,
        "filmic" => ViewTransform::Filmic,
        "raw" => ViewTransform::Raw,
        "false_color" | "false color" => ViewTransform::FalseColor,
        other => {
            return Err(invalid_intermediate(&format!(
                "unsupported Blender view transform `{other}`"
            )));
        }
    };
    let curve = value
        .get("curve")
        .map(|curve| {
            curve
                .as_array()
                .ok_or_else(|| invalid_intermediate("color curve must be an array"))?
                .iter()
                .map(|pair| {
                    let values = pair
                        .as_array()
                        .filter(|values| values.len() == 2)
                        .ok_or_else(|| {
                            invalid_intermediate("color curve point must have two values")
                        })?;
                    let x = values[0]
                        .as_f64()
                        .filter(|number| number.is_finite())
                        .ok_or_else(|| invalid_intermediate("color curve input is invalid"))?;
                    let y = values[1]
                        .as_f64()
                        .filter(|number| number.is_finite())
                        .ok_or_else(|| invalid_intermediate("color curve output is invalid"))?;
                    Ok([x, y])
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    Ok(ColorManagement {
        display_device: value
            .get("display_device")
            .and_then(Value::as_str)
            .unwrap_or("sRGB")
            .to_owned(),
        view_transform,
        look: value
            .get("look")
            .and_then(Value::as_str)
            .unwrap_or("none")
            .to_owned(),
        exposure: number_field(value, "exposure", 0.0),
        gamma: number_field(value, "gamma", 1.0),
        curve,
    })
}

type ImportedArmatureIds = BTreeMap<String, Id>;
type ImportedBoneIds = BTreeMap<Id, BTreeMap<String, Id>>;

fn import_armatures(
    items: &[Value],
    doc: &mut SceneDoc,
    mappings: &mut BTreeMap<String, String>,
) -> Result<(ImportedArmatureIds, ImportedBoneIds)> {
    let mut armature_ids = BTreeMap::new();
    let mut bones_by_data = BTreeMap::new();
    for item in items {
        let armature_name = string_field(item, "name")?;
        let armature_id = mapped_id(
            item,
            "potter_id",
            "armature",
            armature_name,
            &mut armature_ids,
            mappings,
            "Armature",
        )?;
        let raw_bones = optional_array(item, "bones")?;
        let mut bone_ids = BTreeMap::new();
        let mut bone_id_map = BTreeMap::new();
        for raw_bone in raw_bones {
            let bone_name = string_field(raw_bone, "name")?;
            let bone_id = mapped_id(
                raw_bone,
                "potter_id",
                "bone",
                bone_name,
                &mut bone_id_map,
                mappings,
                &format!("Bone:{armature_name}"),
            )?;
            bone_ids.insert(bone_name.to_owned(), bone_id);
        }
        let mut bones = Registry::new();
        for raw_bone in raw_bones {
            let bone_name = string_field(raw_bone, "name")?;
            let bone_id = bone_ids
                .get(bone_name)
                .cloned()
                .ok_or_else(|| invalid_intermediate("armature bone ID mapping is missing"))?;
            let parent = raw_bone
                .get("parent")
                .and_then(Value::as_str)
                .map(|parent_name| {
                    bone_ids.get(parent_name).cloned().ok_or_else(|| {
                        invalid_intermediate("armature bone parent was not imported")
                    })
                })
                .transpose()?;
            let bbone_settings = match raw_bone.get("bbone_settings") {
                None | Some(Value::Null) => BTreeMap::new(),
                Some(Value::Object(settings)) => settings
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
                Some(_) => {
                    return Err(invalid_intermediate(
                        "armature bone B-Bone settings are invalid",
                    ));
                }
            };
            bones.insert(
                bone_id,
                Bone {
                    name: bone_name.to_owned(),
                    parent,
                    head: vec3_field(raw_bone, "head", [0.0; 3]),
                    tail: vec3_field(raw_bone, "tail", [0.0, 0.0, 1.0]),
                    roll: number_field(raw_bone, "roll", 0.0),
                    deform: raw_bone
                        .get("deform")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                    inherit_rotation: raw_bone
                        .get("inherit_rotation")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                    use_connect: raw_bone
                        .get("use_connect")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    custom_shape: None,
                    envelope_distance: 0.25,
                    envelope_weight: 1.0,
                    head_radius: 0.1,
                    tail_radius: 0.1,
                    bbone_settings,
                },
            );
        }
        doc.data_blocks.insert(
            armature_id.clone(),
            DataBlock {
                data_type: "armature".to_owned(),
                descriptor: None,
                mesh: None,
                camera: None,
                light: None,
                armature: Some(ArmatureData {
                    bones,
                    ..ArmatureData::default()
                }),
                ..DataBlock::default()
            },
        );
        bones_by_data.insert(armature_id, bone_ids);
    }
    Ok((armature_ids, bones_by_data))
}

fn import_curves(
    items: &[Value],
    doc: &mut SceneDoc,
    mappings: &mut BTreeMap<String, String>,
) -> Result<BTreeMap<String, Id>> {
    let mut data_ids = BTreeMap::new();
    for item in items {
        let name = string_field(item, "name")?;
        let id = mapped_id(
            item,
            "potter_id",
            "curve",
            name,
            &mut data_ids,
            mappings,
            "Curve",
        )?;
        let kind = item.get("type").and_then(Value::as_str).unwrap_or("CURVE");
        let data = match kind {
            "FONT" => {
                let text = item.get("text").unwrap_or(&Value::Null);
                DataBlock {
                    data_type: "text".to_owned(),
                    mesh: None,
                    curve: None,
                    text: Some(TextObjectData {
                        body: text
                            .get("body")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        font: text
                            .get("font")
                            .and_then(Value::as_str)
                            .unwrap_or("builtin")
                            .to_owned(),
                        font_name: text
                            .get("font_name")
                            .and_then(Value::as_str)
                            .unwrap_or("Bfont")
                            .to_owned(),
                        size: number_field(text, "size", 1.0),
                        align_x: text
                            .get("align_x")
                            .and_then(Value::as_str)
                            .unwrap_or("left")
                            .to_owned(),
                        align_y: text
                            .get("align_y")
                            .and_then(Value::as_str)
                            .unwrap_or("baseline")
                            .to_owned(),
                        extrude: number_field(text, "extrude", 0.0),
                        bevel_depth: number_field(text, "bevel_depth", 0.0),
                        character_spacing: number_field(text, "character_spacing", 1.0),
                        word_spacing: number_field(text, "word_spacing", 1.0),
                        line_spacing: number_field(text, "line_spacing", 1.0),
                        shear: number_field(text, "shear", 0.0),
                        offset_x: number_field(text, "offset_x", 0.0),
                        offset_y: number_field(text, "offset_y", 0.0),
                        small_caps_scale: number_field(text, "small_caps_scale", 0.75),
                    }),
                    ..DataBlock::default()
                }
            }
            "SURFACE" => {
                let surface = optional_array(item, "surface")?
                    .first()
                    .unwrap_or(&Value::Null);
                let points = optional_array(surface, "points")?
                    .iter()
                    .map(|row| {
                        row.as_array()
                            .into_iter()
                            .flatten()
                            .map(|point| SurfacePoint {
                                co: vec3_field(point, "co", [0.0; 3]),
                                weight: number_field(point, "weight", 1.0),
                            })
                            .collect()
                    })
                    .collect();
                let resolution = surface.get("resolution").and_then(Value::as_array);
                DataBlock {
                    data_type: "surface".to_owned(),
                    mesh: None,
                    surface: Some(SurfaceData {
                        points,
                        order_u: u32::try_from(integer_field(surface, "order_u", 3))
                            .map_err(|_| invalid_intermediate("surface order_u is out of range"))?,
                        order_v: u32::try_from(integer_field(surface, "order_v", 3))
                            .map_err(|_| invalid_intermediate("surface order_v is out of range"))?,
                        resolution: [
                            resolution
                                .and_then(|values| values.first())
                                .and_then(Value::as_u64)
                                .and_then(|value| u32::try_from(value).ok())
                                .unwrap_or(12),
                            resolution
                                .and_then(|values| values.get(1))
                                .and_then(Value::as_u64)
                                .and_then(|value| u32::try_from(value).ok())
                                .unwrap_or(12),
                        ],
                        cyclic_u: surface
                            .get("cyclic_u")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                        cyclic_v: surface
                            .get("cyclic_v")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                        use_endpoint_u: surface
                            .get("use_endpoint_u")
                            .and_then(Value::as_bool)
                            .unwrap_or(true),
                        use_endpoint_v: surface
                            .get("use_endpoint_v")
                            .and_then(Value::as_bool)
                            .unwrap_or(true),
                    }),
                    ..DataBlock::default()
                }
            }
            _ => {
                let splines = optional_array(item, "splines")?
                    .iter()
                    .map(|spline| {
                        let spline_type = match spline
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or("poly")
                            .to_ascii_lowercase()
                            .as_str()
                        {
                            "bezier" => CurveSplineType::Bezier,
                            "nurbs" => CurveSplineType::Nurbs,
                            _ => CurveSplineType::Poly,
                        };
                        let points = optional_array(spline, "points")?
                            .iter()
                            .map(|point| {
                                let handle_type = match point
                                    .get("handle_type")
                                    .and_then(Value::as_str)
                                    .unwrap_or("AUTO")
                                    .to_ascii_lowercase()
                                    .as_str()
                                {
                                    "vector" => CurveHandleType::Vector,
                                    "aligned" => CurveHandleType::Aligned,
                                    "free" => CurveHandleType::Free,
                                    _ => CurveHandleType::Auto,
                                };
                                Ok(CurvePoint {
                                    co: vec3_field(point, "co", [0.0; 3]),
                                    handle_left: vec3_field(point, "handle_left", [0.0; 3]),
                                    handle_right: vec3_field(point, "handle_right", [0.0; 3]),
                                    handle_type,
                                    weight: number_field(point, "weight", 1.0),
                                    radius: number_field(point, "radius", 1.0),
                                    tilt: number_field(point, "tilt", 0.0),
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(CurveSpline {
                            spline_type,
                            points,
                            order: u32::try_from(integer_field(spline, "order", 3)).map_err(
                                |_| invalid_intermediate("curve spline order is out of range"),
                            )?,
                            cyclic: spline
                                .get("cyclic")
                                .and_then(Value::as_bool)
                                .unwrap_or(false),
                            resolution: u32::try_from(integer_field(spline, "resolution", 12))
                                .map_err(|_| {
                                    invalid_intermediate("curve spline resolution is out of range")
                                })?,
                            use_endpoint: spline
                                .get("use_endpoint")
                                .and_then(Value::as_bool)
                                .unwrap_or(false),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                DataBlock {
                    data_type: "curve".to_owned(),
                    mesh: None,
                    curve: Some(CurveData {
                        resolution_u: u32::try_from(integer_field(item, "resolution_u", 12))
                            .map_err(|_| {
                                invalid_intermediate("curve resolution_u is out of range")
                            })?,
                        splines,
                        dimensions: match item
                            .get("dimensions")
                            .and_then(Value::as_str)
                            .unwrap_or("3D")
                        {
                            "2D" => CurveDimensions::TwoD,
                            _ => CurveDimensions::ThreeD,
                        },
                        bevel_depth: number_field(item, "bevel_depth", 0.0),
                        bevel_resolution: u32::try_from(integer_field(item, "bevel_resolution", 0))
                            .map_err(|_| {
                                invalid_intermediate("curve bevel resolution is out of range")
                            })?,
                        extrude: number_field(item, "extrude", 0.0),
                        taper: None,
                        fill_mode: match item
                            .get("fill_mode")
                            .and_then(Value::as_str)
                            .unwrap_or("NONE")
                        {
                            "FRONT" => CurveFillMode::Front,
                            "BACK" => CurveFillMode::Back,
                            "BOTH" => CurveFillMode::Both,
                            _ => CurveFillMode::None,
                        },
                        twist_mode: item
                            .get("twist_mode")
                            .and_then(Value::as_str)
                            .unwrap_or("MINIMUM")
                            .to_owned(),
                        use_path: item
                            .get("use_path")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                        path_duration: u32::try_from(integer_field(item, "path_duration", 100))
                            .map_err(|_| {
                                invalid_intermediate("curve path duration is out of range")
                            })?,
                        eval_time: number_field(item, "eval_time", 0.0),
                        eval_time_fcurves: optional_array(item, "eval_time_fcurves")?
                            .iter()
                            .map(|curve| convert_fcurve(curve, mappings))
                            .collect::<Result<Vec<_>>>()?,
                    }),
                    ..DataBlock::default()
                }
            }
        };
        doc.data_blocks.insert(id, data);
    }
    Ok(data_ids)
}

fn register_blend_libraries(
    raw: &Value,
    resource_ids: &BTreeMap<String, String>,
    registry_ids: &BTreeMap<String, BTreeMap<String, Id>>,
    doc: &mut SceneDoc,
    mappings: &mut BTreeMap<String, String>,
) -> Result<()> {
    let mut library_ids = BTreeMap::new();
    let mut used_ids = BTreeSet::new();
    for source in optional_array(raw, "libraries")? {
        let path = source
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let name = source.get("name").and_then(Value::as_str).unwrap_or(path);
        let seed = format!("library_{}", &crate::hash::sha256(path.as_bytes())[7..19]);
        let id = Id::new(unique_id(&seed, "library", &mut used_ids))?;
        let resource = resource_ids
            .get(path)
            .and_then(|resource| Id::new(resource.clone()).ok());
        let resource_record = resource
            .as_ref()
            .and_then(|resource_id| doc.resources.get(resource_id));
        let status = if resource_record
            .and_then(|record| record.get("status"))
            .and_then(Value::as_str)
            == Some("missing")
        {
            LibraryStatus::Missing
        } else {
            LibraryStatus::Ok
        };
        let hash = resource_record
            .and_then(|record| record.get("hash"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let uri = source
            .get("uri")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        doc.libraries.insert(
            id.clone(),
            Library {
                name: name.to_owned(),
                kind: LibraryKind::Blend,
                uri,
                resolved_path: path.to_owned(),
                resource,
                hash,
                status,
                source_project: None,
                ..Library::default()
            },
        );
        library_ids.insert(path.to_owned(), id.clone());
        mappings.insert(format!("Library:{name}"), id.to_string());
    }

    for (registry, key) in [
        ("nodes", "objects"),
        ("collections", "collections"),
        ("materials", "materials"),
        ("actions", "actions"),
        ("node_groups", "node_groups"),
        ("resources", "images"),
        ("data_blocks", "meshes"),
        ("data_blocks", "armatures"),
        ("data_blocks", "grease_pencil"),
        ("data_blocks", "curves"),
        ("data_blocks", "volumes"),
        ("data_blocks", "cameras"),
        ("data_blocks", "lights"),
    ] {
        let Some(ids) = registry_ids.get(registry) else {
            continue;
        };
        for item in optional_array(raw, key)? {
            let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
            let Some(id) = ids.get(name) else {
                continue;
            };
            register_blend_linked_item(item, registry, id, &library_ids, doc)?;
        }
    }
    for scene in optional_array(raw, "scenes")? {
        let Some(world) = scene.get("world").filter(|value| value.is_object()) else {
            continue;
        };
        let name = world
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let Some(id) = registry_ids.get("worlds").and_then(|ids| ids.get(name)) else {
            continue;
        };
        register_blend_linked_item(world, "worlds", id, &library_ids, doc)?;
    }
    import_blend_library_overrides(raw, &library_ids, registry_ids, doc)?;
    Ok(())
}

fn register_blend_linked_item(
    source: &Value,
    registry: &str,
    id: &Id,
    library_ids: &BTreeMap<String, Id>,
    doc: &mut SceneDoc,
) -> Result<()> {
    let Some(reference) = source.get("library").filter(|value| value.is_object()) else {
        return Ok(());
    };
    let path = reference
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let library_id = library_ids.get(path).ok_or_else(|| {
        invalid_intermediate("linked Blender ID references an unregistered library")
    })?;
    let library = doc
        .libraries
        .get_mut(library_id)
        .ok_or_else(|| invalid_intermediate("linked Blender library disappeared"))?;
    let library_name = library.name.clone();
    let uri = library.uri.clone();
    let source_id = reference
        .get("source_id")
        .and_then(Value::as_str)
        .filter(|value| crate::model::is_valid_id(value))
        .and_then(|value| Id::new(value.to_owned()).ok())
        .unwrap_or_else(|| id.clone());
    let ids = library.linked_ids.entry(registry.to_owned()).or_default();
    if !ids.contains(id) {
        ids.push(id.clone());
        ids.sort();
    }
    library.items.insert(
        crate::library::linked_key(registry, id),
        source_id.to_string(),
    );
    let marker = json!({
        "registry":registry,
        "library_id":library_id,
        "library_name":library_name,
        "source_id":source_id,
        "uri":uri,
        "editable":false,
    });
    let linked = doc
        .compatibility
        .entry("linked_ids".to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| invalid_intermediate("linked-ID metadata is not an object"))?;
    linked.insert(crate::library::linked_key(registry, id), marker);
    if registry == "nodes"
        && let Some(node) = doc.nodes.get_mut(id)
    {
        node.properties
            .insert("library".to_owned(), json!(library_id));
        node.properties
            .insert("library_name".to_owned(), json!(library_name));
        node.properties
            .insert("editable".to_owned(), Value::Bool(false));
        node.properties
            .insert("source".to_owned(), json!(source_id));
    }
    Ok(())
}

fn import_blend_library_overrides(
    raw: &Value,
    library_ids: &BTreeMap<String, Id>,
    registry_ids: &BTreeMap<String, BTreeMap<String, Id>>,
    doc: &mut SceneDoc,
) -> Result<()> {
    let objects = optional_array(raw, "objects")?;
    let object_ids = registry_ids
        .get("nodes")
        .ok_or_else(|| invalid_intermediate("object ID registry is missing"))?;
    for source in objects {
        let Some(override_data) = source
            .get("override_library")
            .filter(|value| value.is_object())
        else {
            continue;
        };
        let source_name = string_field(source, "name")?;
        let local_id = object_ids
            .get(source_name)
            .ok_or_else(|| invalid_intermediate("library override object is not mapped"))?;
        let reference_name = override_data
            .get("reference")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_intermediate("library override has no reference object"))?;
        let reference_id = object_ids
            .get(reference_name)
            .ok_or_else(|| invalid_intermediate("library override reference is not mapped"))?;
        let linked_reference = objects
            .iter()
            .find(|item| item.get("name").and_then(Value::as_str) == Some(reference_name))
            .ok_or_else(|| invalid_intermediate("library override reference object is missing"))?;
        let path = linked_reference
            .get("library")
            .and_then(|library| library.get("path"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let library_id = library_ids.get(path).ok_or_else(|| {
            invalid_intermediate("library override reference has no registered library")
        })?;
        let mut properties = Vec::new();
        let mut operations = Vec::new();
        for action in optional_array(override_data, "properties")? {
            let path = string_field(action, "path")?;
            let operation = string_field(action, "operation")?;
            let value = action.get("value").cloned().unwrap_or(Value::Null);
            properties.push(LibraryOverrideProperty {
                path: path.to_owned(),
                operation: operation.to_owned(),
                value: value.clone(),
            });
            operations.push(json!({"op":operation,"path":path,"value":value}));
        }
        if let Some(node) = doc.nodes.get_mut(local_id) {
            node.properties.insert(
                "library_override".to_owned(),
                json!({
                    "library_id":library_id,
                    "reference_id":reference_id,
                    "operations":operations,
                }),
            );
        }
        let library = doc
            .libraries
            .get_mut(library_id)
            .ok_or_else(|| invalid_intermediate("override library disappeared"))?;
        library.overrides.push(LibraryOverride {
            registry: "nodes".to_owned(),
            id: local_id.clone(),
            reference_id: reference_id.clone(),
            properties,
        });
    }
    Ok(())
}

fn import_blender_resources(
    raw: &Value,
    blend_file: &Path,
    doc: &mut SceneDoc,
) -> Result<ImportedResourceAssets> {
    let mut path_ids = BTreeMap::new();
    let mut assets = Vec::new();
    let mut used_ids = BTreeSet::new();
    for source in optional_array(raw, "resources")? {
        let source_path = source
            .get("path")
            .and_then(Value::as_str)
            .filter(|path| !path.is_empty());
        let packed_data = source
            .get("packed_data")
            .and_then(Value::as_str)
            .filter(|data| !data.is_empty());
        if source_path.is_none() && packed_data.is_none() {
            continue;
        }
        let resolved_path = source_path.map(|path| resolve_blend_resource_path(path, blend_file));
        let packed_bytes = packed_data.map(decode_resource_base64).transpose()?;
        let bytes = if let Some(bytes) = packed_bytes {
            Some(bytes)
        } else if let Some(path) = &resolved_path {
            match fs::read(path) {
                Ok(bytes) => Some(bytes),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(PotError::io(&error)),
            }
        } else {
            None
        };
        let digest = bytes.as_deref().map(crate::hash::sha256);
        let source_packed = source
            .get("packed")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && packed_data.is_some();
        let kind = source
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("resource");
        let owner = source
            .get("owner")
            .and_then(Value::as_str)
            .unwrap_or("Blender resource");
        let identity = resolved_path.as_ref().map_or_else(
            || owner.to_owned(),
            |path| path.to_string_lossy().into_owned(),
        );
        let seed = format!(
            "resource_{}",
            &crate::hash::sha256(format!("{kind}:{identity}").as_bytes())[7..19]
        );
        let id = Id::new(unique_id(&seed, "resource", &mut used_ids))?;
        let filename = resolved_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .map_or_else(
                || format!("{}.bin", safe_resource_stem(owner)),
                str::to_owned,
            );
        let uri = resolved_path
            .as_ref()
            .map_or_else(String::new, |path| path.to_string_lossy().into_owned());
        let original_path = (!uri.is_empty()).then(|| uri.clone());
        let status = if bytes.is_some() {
            "available"
        } else {
            "missing"
        };
        let hash = digest.as_deref().map_or(Value::Null, |value| json!(value));
        let mut resource = json!({
            "uri": uri,
            "original_path": original_path,
            "hash": hash.clone(),
            "expected_hash": hash,
            "kind": kind,
            "owner": owner,
            "filename": filename,
            "packed": false,
            "source_packed": source_packed,
            "status": status,
        });
        if let Some(cache_file) = source.get("cache_file") {
            resource["cache_file"] = cache_file.clone();
        }
        doc.resources.insert(id.clone(), resource);
        let identity_key = |path: &str| format!("{path}\0{owner}");
        if let Some(path) = source_path {
            path_ids.insert(path.to_owned(), id.as_str().to_owned());
            path_ids.insert(identity_key(path), id.as_str().to_owned());
        }
        if let Some(path) = &resolved_path {
            let path = path.to_string_lossy().into_owned();
            path_ids.insert(path.clone(), id.as_str().to_owned());
            path_ids.insert(identity_key(&path), id.as_str().to_owned());
        }
        if kind == "image" {
            path_ids
                .entry(owner.to_owned())
                .or_insert_with(|| id.as_str().to_owned());
        }
        if let (Some(digest), Some(bytes)) = (digest, bytes) {
            assets.push((digest, bytes));
        }
    }
    Ok((path_ids, assets))
}

fn import_movie_clips(
    raw: &Value,
    resource_ids: &BTreeMap<String, String>,
    doc: &mut SceneDoc,
    mappings: &mut BTreeMap<String, String>,
) -> Result<BTreeMap<String, Id>> {
    let mut ids = BTreeMap::new();
    for item in optional_array(raw, "movie_clips")? {
        let name = string_field(item, "name")?;
        let id = mapped_id(
            item,
            "potter_id",
            "movie_clip",
            name,
            &mut ids,
            mappings,
            "MovieClip",
        )?;
        let filepath = item
            .get("filepath")
            .and_then(Value::as_str)
            .filter(|path| !path.is_empty());
        let source = filepath.and_then(|path| resource_ids.get(path)).cloned();
        let source_hash = source
            .as_ref()
            .and_then(|resource_id| Id::new(resource_id.clone()).ok())
            .and_then(|resource_id| doc.resources.get(&resource_id))
            .and_then(|resource| resource.get("hash"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let tracking = serde_json::from_value::<MovieTracking>(
            item.get("tracking").cloned().unwrap_or_else(|| json!({})),
        )
        .map_err(|_| invalid_intermediate("movie clip tracking data is invalid"))?;
        let clip = MovieClip {
            name: name.to_owned(),
            source,
            source_hash,
            frame_start: i32::try_from(integer_field(item, "frame_start", 1))
                .map_err(|_| invalid_intermediate("movie clip frame_start is out of range"))?,
            width: u32::try_from(integer_field(item, "width", 0))
                .map_err(|_| invalid_intermediate("movie clip width is out of range"))?,
            height: u32::try_from(integer_field(item, "height", 0))
                .map_err(|_| invalid_intermediate("movie clip height is out of range"))?,
            fps: number_field(item, "fps", 24.0),
            tracking,
        };
        doc.movie_clips.insert(id, clip);
    }
    Ok(ids)
}

fn resolve_blend_resource_path(path: &str, blend_file: &Path) -> PathBuf {
    let path = path.strip_prefix("//").map_or_else(
        || PathBuf::from(path),
        |relative| {
            blend_file
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(relative)
        },
    );
    let candidate = if path.is_absolute() {
        path
    } else {
        blend_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    };
    fs::canonicalize(&candidate).unwrap_or(candidate)
}

fn decode_resource_base64(input: &str) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(input.len().saturating_mul(3) / 4);
    let mut accumulator = 0_u32;
    let mut bits = 0_u8;
    for byte in input.bytes() {
        if byte == b'=' {
            break;
        }
        if byte.is_ascii_whitespace() {
            continue;
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => {
                return Err(invalid_intermediate(
                    "packed resource data is invalid base64",
                ));
            }
        };
        accumulator = (accumulator << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((accumulator >> bits) as u8);
        }
    }
    Ok(output)
}

fn safe_resource_stem(value: &str) -> String {
    let stem = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if stem.is_empty() {
        "resource".to_owned()
    } else {
        stem
    }
}

fn import_volumes(
    items: &[Value],
    objects: &[Value],
    resource_ids: &BTreeMap<String, String>,
    doc: &mut SceneDoc,
    mappings: &mut BTreeMap<String, String>,
) -> Result<BTreeMap<String, Id>> {
    let mut data_ids = BTreeMap::new();
    for item in items {
        let name = string_field(item, "name")?;
        let id = mapped_id(
            item,
            "potter_id",
            "volume",
            name,
            &mut data_ids,
            mappings,
            "Volume",
        )?;
        let filepath = item
            .get("filepath")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let content_ref = resource_ids.get(filepath).cloned();
        let object = objects.iter().find(|object| {
            object.get("type").and_then(Value::as_str) == Some("VOLUME")
                && object.get("data_name").and_then(Value::as_str) == Some(name)
        });
        let bounds = object
            .and_then(|object| object.get("volume_bounds"))
            .and_then(Value::as_array)
            .filter(|bounds| bounds.len() == 2)
            .and_then(|bounds| {
                Some((parse_vec3_value(&bounds[0])?, parse_vec3_value(&bounds[1])?))
            });
        let grid_names = item
            .get("grids")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        let source = if filepath.is_empty() {
            crate::geom::volume::VolumeSource::Generated(
                crate::geom::volume::VolumeGeneratedSource {
                    algorithm: "empty".to_owned(),
                },
            )
        } else {
            let format = Path::new(filepath)
                .extension()
                .and_then(|extension| extension.to_str())
                .filter(|extension| !extension.is_empty())
                .unwrap_or("vdb")
                .to_ascii_lowercase();
            crate::geom::volume::VolumeSource::File(crate::geom::volume::VolumeFileSource {
                format,
                content_ref,
                grid_names,
                bounds_min: bounds.map(|bounds| bounds.0),
                bounds_max: bounds.map(|bounds| bounds.1),
            })
        };
        doc.data_blocks.insert(
            id,
            DataBlock {
                data_type: "volume".to_owned(),
                volume: Some(crate::geom::volume::VolumeData {
                    source,
                    ..crate::geom::volume::VolumeData::default()
                }),
                ..DataBlock::default()
            },
        );
    }
    Ok(data_ids)
}

fn parse_vec3_value(value: &Value) -> Option<[f64; 3]> {
    let values = value.as_array().filter(|values| values.len() == 3)?;
    Some([
        values[0].as_f64()?,
        values[1].as_f64()?,
        values[2].as_f64()?,
    ])
}

fn string_field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_intermediate(&format!("missing string `{key}`")))
}

fn number_field(value: &Value, key: &str, fallback: f64) -> f64 {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .unwrap_or(fallback)
}

fn integer_field(value: &Value, key: &str, fallback: i64) -> i64 {
    value.get(key).and_then(Value::as_i64).unwrap_or(fallback)
}

fn vec3_field(value: &Value, key: &str, fallback: [f64; 3]) -> [f64; 3] {
    value
        .get(key)
        .and_then(Value::as_array)
        .filter(|array| array.len() == 3)
        .and_then(|array| Some([array[0].as_f64()?, array[1].as_f64()?, array[2].as_f64()?]))
        .unwrap_or(fallback)
}

fn bool3_field(value: &Value, key: &str, fallback: [bool; 3]) -> [bool; 3] {
    value
        .get(key)
        .and_then(Value::as_array)
        .filter(|array| array.len() == 3)
        .and_then(|array| {
            Some([
                array[0].as_bool()?,
                array[1].as_bool()?,
                array[2].as_bool()?,
            ])
        })
        .unwrap_or(fallback)
}

fn vec4_field(value: &Value, key: &str, fallback: [f64; 4]) -> [f64; 4] {
    value
        .get(key)
        .and_then(Value::as_array)
        .filter(|array| array.len() == 4)
        .and_then(|array| {
            Some([
                array[0].as_f64()?,
                array[1].as_f64()?,
                array[2].as_f64()?,
                array[3].as_f64()?,
            ])
        })
        .unwrap_or(fallback)
}

fn matrix_column_major(value: Option<&Value>) -> Option<[f64; 16]> {
    let values = value?.as_array()?;
    if values.len() == 16 {
        let mut result = [0.0; 16];
        for (index, value) in values.iter().enumerate() {
            result[index] = value.as_f64()?;
        }
        return Some(result);
    }
    if values.len() != 4 {
        return None;
    }
    let mut result = [0.0; 16];
    for (row, values) in values.iter().enumerate() {
        let values = values.as_array()?;
        if values.len() != 4 {
            return None;
        }
        for (column, value) in values.iter().enumerate() {
            result[column * 4 + row] = value.as_f64()?;
        }
    }
    Some(result)
}

fn mapped_id(
    value: &Value,
    id_field: &str,
    prefix: &str,
    name: &str,
    by_name: &mut BTreeMap<String, Id>,
    mappings: &mut BTreeMap<String, String>,
    kind: &str,
) -> Result<Id> {
    let used: BTreeSet<String> = by_name.values().map(|id| id.as_str().to_owned()).collect();
    let candidate = value
        .get(id_field)
        .and_then(Value::as_str)
        .filter(|id| crate::model::is_valid_id(id) && !used.contains(*id))
        .map_or_else(
            || unique_id(name, prefix, &mut used.into_iter().collect()),
            str::to_owned,
        );
    let id = Id::new(candidate)?;
    by_name.insert(name.to_owned(), id.clone());
    mappings.insert(format!("{kind}:{name}"), id.as_str().to_owned());
    Ok(id)
}

fn unique_id(name: &str, prefix: &str, used: &mut BTreeSet<String>) -> String {
    let mut slug = String::new();
    for character in name.to_lowercase().chars() {
        if character.is_ascii_lowercase() || character.is_ascii_digit() {
            slug.push(character);
        } else if !slug.ends_with('_') {
            slug.push('_');
        }
    }
    while slug.ends_with('_') {
        slug.pop();
    }
    if slug.is_empty() {
        slug.push_str("unnamed");
    }
    let prefix = if crate::model::is_valid_id(prefix) {
        prefix
    } else {
        "item"
    };
    let mut base = format!("{prefix}_{slug}");
    base.truncate(64);
    while base.ends_with('_') || base.ends_with('-') {
        base.pop();
    }
    if base.is_empty() {
        "item".clone_into(&mut base);
    }
    let mut value = base.clone();
    let mut suffix = 2_u32;
    while used.contains(&value) {
        let ending = format!("_{suffix}");
        let keep = 64_usize.saturating_sub(ending.len());
        value = format!("{}{ending}", &base[..base.len().min(keep)]);
        suffix = suffix.saturating_add(1);
    }
    used.insert(value.clone());
    value
}

fn convert_mesh(item: &Value) -> Result<Mesh> {
    let vertices_raw = item
        .get("vertices")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_intermediate("mesh vertices are missing"))?;
    let mut vertices = Vec::with_capacity(vertices_raw.len());
    for (index, vertex) in vertices_raw.iter().enumerate() {
        let co = required_vec3(vertex, "co")?;
        vertices.push(Vertex {
            id: checked_element_id(index)?,
            co: glam::DVec3::from_array(co),
        });
    }
    let edges_raw = item
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_intermediate("mesh edges are missing"))?;
    let mut edges = Vec::with_capacity(edges_raw.len());
    let mut edge_metadata = Vec::with_capacity(edges_raw.len());
    for (index, edge) in edges_raw.iter().enumerate() {
        let vertices_array = edge
            .get("v")
            .and_then(Value::as_array)
            .filter(|items| items.len() == 2)
            .ok_or_else(|| invalid_intermediate("mesh edge vertex reference is invalid"))?;
        let a = imported_vertex_id(&vertices_array[0])?;
        let b = imported_vertex_id(&vertices_array[1])?;
        edges.push(Edge {
            id: checked_element_id(index)?,
            vertices: [a, b],
        });
        edge_metadata.push(json!({"sharp": edge.get("sharp"), "seam": edge.get("seam")}));
    }
    let polygons_raw = item
        .get("polygons")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_intermediate("mesh polygons are missing"))?;
    let mut faces = Vec::with_capacity(polygons_raw.len());
    let mut smooth = Vec::with_capacity(polygons_raw.len());
    for (index, polygon) in polygons_raw.iter().enumerate() {
        let refs = polygon
            .get("v")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_intermediate("mesh polygon vertices are missing"))?;
        let vertices = refs
            .iter()
            .map(imported_vertex_id)
            .collect::<Result<Vec<_>>>()?;
        faces.push(Face {
            id: checked_element_id(index)?,
            vertices,
            material_index: u32::try_from(integer_field(polygon, "material_index", 0))
                .map_err(|_| invalid_intermediate("mesh material index is out of range"))?,
        });
        smooth.push(
            polygon
                .get("smooth")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        );
    }
    let mut attributes = Map::new();
    attributes.insert(
        "blender_uv_layers".to_owned(),
        item.get("uv_layers").cloned().unwrap_or_else(|| json!([])),
    );
    attributes.insert("blender_edge_flags".to_owned(), json!(edge_metadata));
    attributes.insert("blender_polygon_smooth".to_owned(), json!(smooth));
    attributes.insert(
        "blender_attributes".to_owned(),
        item.get("attributes").cloned().unwrap_or_else(|| json!([])),
    );
    if let Some(skin_vertices) = item
        .get("skin_vertices")
        .and_then(Value::as_array)
        .filter(|skin_vertices| !skin_vertices.is_empty())
    {
        let mut radii = Map::new();
        let mut roots = Map::new();
        for (index, skin_vertex) in skin_vertices.iter().enumerate() {
            let id = index.to_string();
            radii.insert(
                id.clone(),
                skin_vertex
                    .get("radius")
                    .cloned()
                    .unwrap_or_else(|| json!([0.25, 0.25])),
            );
            roots.insert(id, skin_vertex.get("root").cloned().unwrap_or(json!(false)));
        }
        attributes.insert("skin_radii".to_owned(), Value::Object(radii));
        attributes.insert("skin_roots".to_owned(), Value::Object(roots));
        attributes.insert(
            "blender_skin_vertices".to_owned(),
            Value::Array(skin_vertices.clone()),
        );
    }
    let next_id = IdCounters {
        vertex: next_counter(vertices.len())?,
        edge: next_counter(edges.len())?,
        face: next_counter(faces.len())?,
    };
    let mesh = Mesh {
        vertices,
        edges,
        faces,
        attributes,
        next_id,
    };
    mesh.validate()
        .map_err(|error| invalid_intermediate(&format!("invalid mesh data: {error}")))?;
    Ok(mesh)
}

fn parse_shape_keys(
    item: &Value,
    mesh: &Mesh,
    mesh_name: &str,
    mappings: &mut BTreeMap<String, String>,
) -> Result<(Option<ShapeKeyData>, BTreeMap<Id, String>)> {
    let Some(raw) = item.get("shape_keys").filter(|value| !value.is_null()) else {
        return Ok((None, BTreeMap::new()));
    };
    let key_items = optional_array(raw, "keys")?;
    let mut key_ids = BTreeMap::new();
    let mut key_id_map = BTreeMap::new();
    for key in key_items {
        let name = string_field(key, "name")?;
        let id = mapped_id(
            key,
            "potter_id",
            "shape_key",
            name,
            &mut key_id_map,
            mappings,
            &format!("ShapeKey:{mesh_name}"),
        )?;
        key_ids.insert(name.to_owned(), id);
    }
    let mut keys = Registry::new();
    let mut vertex_group_names = BTreeMap::new();
    for key in key_items {
        let name = string_field(key, "name")?;
        let id = key_ids
            .get(name)
            .cloned()
            .ok_or_else(|| invalid_intermediate("shape-key ID mapping is missing"))?;
        let relative_key = key
            .get("relative_key")
            .and_then(Value::as_str)
            .filter(|relative| *relative != "Basis")
            .map(|relative| {
                key_ids.get(relative).cloned().ok_or_else(|| {
                    invalid_intermediate("shape key references an unknown relative key")
                })
            })
            .transpose()?;
        let vertex_group = key.get("vertex_group").and_then(Value::as_str);
        if let Some(vertex_group) = vertex_group.filter(|value| !value.is_empty()) {
            vertex_group_names.insert(id.clone(), vertex_group.to_owned());
        }
        keys.insert(
            id.clone(),
            ShapeKey {
                id,
                name: name.to_owned(),
                value: number_field(key, "value", 0.0),
                mute: key.get("mute").and_then(Value::as_bool).unwrap_or(false),
                slider_min: number_field(key, "slider_min", 0.0),
                slider_max: number_field(key, "slider_max", 1.0),
                relative_key,
                vertex_group: None,
                frame: number_field(key, "frame", 0.0),
                positions: indexed_vertex_positions(key.get("positions"), mesh)?,
            },
        );
    }
    Ok((
        Some(ShapeKeyData {
            basis: indexed_vertex_positions(raw.get("basis"), mesh)?,
            keys,
            absolute: raw
                .get("absolute")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            evaluation_time: number_field(raw, "evaluation_time", 0.0),
            action: None,
            action_slot: raw
                .get("action_slot")
                .and_then(Value::as_str)
                .map(str::to_owned),
            muted_action_curves: raw
                .get("muted_action_curves")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
        }),
        vertex_group_names,
    ))
}

fn indexed_vertex_positions(raw: Option<&Value>, mesh: &Mesh) -> Result<BTreeMap<u32, [f64; 3]>> {
    let Some(values) = raw.and_then(Value::as_array) else {
        return Ok(mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co.to_array()))
            .collect());
    };
    let mut positions = BTreeMap::new();
    for value in values {
        let index = value
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| invalid_intermediate("shape-key vertex index is invalid"))?;
        let vertex_id = mesh
            .vertices
            .get(index)
            .map(|vertex| vertex.id)
            .ok_or_else(|| invalid_intermediate("shape key references a missing mesh vertex"))?;
        positions.insert(vertex_id, required_vec3(value, "co")?);
    }
    Ok(positions)
}

fn checked_element_id(index: usize) -> Result<u32> {
    u32::try_from(index)
        .ok()
        .and_then(|index| index.checked_add(1))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "mesh has too many elements for persistent IDs",
            )
        })
}

fn required_vec3(value: &Value, key: &str) -> Result<[f64; 3]> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .filter(|values| values.len() == 3)
        .ok_or_else(|| {
            invalid_intermediate(&format!("mesh vertex `{key}` must contain three numbers"))
        })?;
    let mut result = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        let number = value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                invalid_intermediate(&format!("mesh vertex `{key}` contains an invalid number"))
            })?;
        result[index] = number;
    }
    Ok(result)
}

fn imported_vertex_id(value: &Value) -> Result<u32> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .and_then(|index| index.checked_add(1))
        .ok_or_else(|| invalid_intermediate("mesh vertex index is out of range"))
}

fn next_counter(length: usize) -> Result<u32> {
    u32::try_from(length)
        .ok()
        .and_then(|number| number.checked_add(1))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "mesh has too many elements for persistent IDs",
            )
        })
}

fn potter_fcurve_path(path: &str, mappings: &BTreeMap<String, String>) -> String {
    let normalized = match path {
        "location" => "transform.translation",
        "rotation_euler" => "transform.rotation_euler",
        "rotation_quaternion" => "transform.rotation_quaternion",
        "scale" => "transform.scale",
        "lens" | "data.lens" => "camera.lens_mm",
        "energy" | "data.energy" => "light.energy",
        _ => path,
    };
    if normalized != path {
        return normalized.to_owned();
    }
    let Some(start) = path
        .find("pose.bones[")
        .map(|index| index + "pose.bones[".len())
    else {
        return path.to_owned();
    };
    let Some(offset) = path[start..].find(']') else {
        return path.to_owned();
    };
    let end = start + offset;
    let bone_name = path[start..end].trim_matches(['"', '\'']);
    let suffix = format!(":{bone_name}");
    let bone_ids = mappings
        .iter()
        .filter(|(key, _)| key.starts_with("Bone:") && key.ends_with(&suffix))
        .map(|(_, id)| id)
        .collect::<BTreeSet<_>>();
    if bone_ids.len() != 1 {
        return path.to_owned();
    }
    let Some(bone_id) = bone_ids.first() else {
        return path.to_owned();
    };
    let quote = path[start..].chars().next().unwrap_or('"');
    format!(
        "{}{}{}{}{}",
        &path[..start],
        quote,
        bone_id,
        quote,
        &path[end..]
    )
}

fn fcurve_handle(value: &Value, field: &str) -> Result<Option<[f64; 2]>> {
    let Some(value) = value.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let handle = value
        .as_array()
        .filter(|handle| handle.len() == 2)
        .ok_or_else(|| {
            invalid_intermediate(&format!("F-Curve `{field}` must have two coordinates"))
        })?;
    let x = handle[0]
        .as_f64()
        .filter(|coordinate| coordinate.is_finite())
        .ok_or_else(|| {
            invalid_intermediate(&format!("F-Curve `{field}` x coordinate is invalid"))
        })?;
    let y = handle[1]
        .as_f64()
        .filter(|coordinate| coordinate.is_finite())
        .ok_or_else(|| {
            invalid_intermediate(&format!("F-Curve `{field}` y coordinate is invalid"))
        })?;
    Ok(Some([x, y]))
}

fn convert_fcurve(value: &Value, mappings: &BTreeMap<String, String>) -> Result<FCurve> {
    let path = potter_fcurve_path(string_field(value, "path")?, mappings);
    let blender_index = integer_field(value, "index", 0);
    let index =
        if path == "transform.rotation_quaternion" || path.ends_with(".rotation_quaternion") {
            match blender_index {
                0 => 3,
                1 => 0,
                2 => 1,
                3 => 2,
                index => index,
            }
        } else {
            blender_index
        }
        .try_into()
        .map_err(|_| invalid_intermediate("F-Curve array index is invalid"))?;
    let keyframes = value
        .get("keyframes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|item| {
            let interpolation = match item
                .get("interpolation")
                .and_then(Value::as_str)
                .unwrap_or("LINEAR")
            {
                "CONSTANT" => Interpolation::Constant,
                "BEZIER" => Interpolation::Bezier,
                _ => Interpolation::Linear,
            };
            Ok(Keyframe {
                frame: number_field(item, "frame", 0.0),
                value: number_field(item, "value", 0.0),
                interpolation,
                handle_left: fcurve_handle(item, "handle_left")?,
                handle_right: fcurve_handle(item, "handle_right")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let extrapolation = match value
        .get("extrapolation")
        .and_then(Value::as_str)
        .unwrap_or("CONSTANT")
    {
        "LINEAR" => Extrapolation::Linear,
        _ => Extrapolation::Constant,
    };
    Ok(FCurve {
        path,
        index,
        keyframes,
        extrapolation,
    })
}

fn canonicalize_blender_id_references(
    value: &mut Value,
    object_ids: &BTreeMap<String, Id>,
    movie_clip_ids: &BTreeMap<String, Id>,
    action_ids: &BTreeMap<String, Id>,
) -> Result<()> {
    if let Some(object) = value.as_object_mut() {
        let id_type = object.get("id_type").and_then(Value::as_str);
        let name = object.get("name").and_then(Value::as_str);
        if let (Some(id_type), Some(name)) = (id_type, name) {
            let id = match id_type {
                "Object" => object_ids.get(name),
                "MovieClip" => movie_clip_ids.get(name),
                "Action" => action_ids.get(name),
                _ => None,
            };
            if matches!(id_type, "Object" | "MovieClip" | "Action") {
                let id = id.ok_or_else(|| {
                    invalid_intermediate(&format!(
                        "modifier or constraint references unknown {id_type} `{name}`"
                    ))
                })?;
                *value = json!(id.as_str());
                return Ok(());
            }
        }
        for child in object.values_mut() {
            canonicalize_blender_id_references(child, object_ids, movie_clip_ids, action_ids)?;
        }
    } else if let Some(array) = value.as_array_mut() {
        for child in array {
            canonicalize_blender_id_references(child, object_ids, movie_clip_ids, action_ids)?;
        }
    }
    Ok(())
}

fn parse_mesh_cache_params(
    properties: &Map<String, Value>,
    resource_ids: &BTreeMap<String, String>,
) -> Result<Map<String, Value>> {
    let mut params = properties.clone();
    let filepath = properties
        .get("cache_filepath")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty());
    if let Some(filepath) = filepath {
        let resource = resource_ids.get(filepath).ok_or_else(|| {
            invalid_intermediate("Mesh Cache filepath has no registered resource")
        })?;
        params.insert("resource".to_owned(), json!(resource));
    }
    params.remove("cache_filepath");
    params.remove("filepath");
    Ok(params)
}

fn parse_mesh_sequence_cache_params(
    properties: &Map<String, Value>,
    resource_ids: &BTreeMap<String, String>,
) -> Result<Map<String, Value>> {
    let path = properties
        .get("cache_filepath")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| invalid_intermediate("Mesh Sequence Cache has no cache filepath"))?;
    let resource = resource_ids
        .get(path)
        .ok_or_else(|| invalid_intermediate("Mesh Sequence Cache filepath has no resource"))?;
    let cache_settings = properties
        .get("cache_file_settings")
        .and_then(Value::as_object);
    let object_path = properties
        .get("object_path")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_intermediate("Mesh Sequence Cache has no object path"))?;
    let mut params = properties.clone();
    params.remove("cache_file");
    params.remove("cache_filepath");
    params.remove("cache_file_settings");
    params.insert("resource".to_owned(), json!(resource));
    params.insert("object_path".to_owned(), json!(object_path));
    insert_cache_settings_params(&mut params, cache_settings);
    Ok(params)
}

fn insert_cache_settings_params(
    params: &mut Map<String, Value>,
    settings: Option<&Map<String, Value>>,
) {
    params.insert(
        "frame_offset".to_owned(),
        json!(
            settings
                .and_then(|settings| settings.get("frame_offset"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
        ),
    );
    params.insert(
        "scale".to_owned(),
        json!(
            settings
                .and_then(|settings| settings.get("scale"))
                .and_then(Value::as_f64)
                .unwrap_or(1.0)
        ),
    );
    params.insert(
        "override_frame".to_owned(),
        if settings
            .and_then(|settings| settings.get("override_frame"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            settings
                .and_then(|settings| settings.get("frame"))
                .cloned()
                .unwrap_or(Value::Null)
        } else {
            Value::Null
        },
    );
}

fn blender_constraint_type(value: &str) -> Option<ConstraintType> {
    Some(match value {
        "COPY_LOCATION" => ConstraintType::CopyLocation,
        "COPY_ROTATION" => ConstraintType::CopyRotation,
        "COPY_SCALE" => ConstraintType::CopyScale,
        "TRACK_TO" => ConstraintType::TrackTo,
        "DAMPED_TRACK" => ConstraintType::DampedTrack,
        "LOCKED_TRACK" => ConstraintType::LockedTrack,
        "STRETCH_TO" => ConstraintType::StretchTo,
        "TRANSFORM" => ConstraintType::Transformation,
        "MAINTAIN_VOLUME" => ConstraintType::MaintainVolume,
        "FLOOR" => ConstraintType::Floor,
        "PIVOT" => ConstraintType::Pivot,
        "SHRINKWRAP" => ConstraintType::Shrinkwrap,
        "SPLINE_IK" => ConstraintType::SplineIk,
        "LIMIT_LOCATION" => ConstraintType::LimitLocation,
        "LIMIT_ROTATION" => ConstraintType::LimitRotation,
        "LIMIT_SCALE" => ConstraintType::LimitScale,
        "CHILD_OF" => ConstraintType::ChildOf,
        "ACTION" => ConstraintType::Action,
        "ARMATURE" => ConstraintType::Armature,
        "CAMERA_SOLVER" => ConstraintType::CameraSolver,
        "CLAMP_TO" => ConstraintType::ClampTo,
        "COPY_TRANSFORMS" => ConstraintType::CopyTransforms,
        "FOLLOW_PATH" => ConstraintType::FollowPath,
        "FOLLOW_TRACK" => ConstraintType::FollowTrack,
        "GEOMETRY_ATTRIBUTE" => ConstraintType::GeometryAttribute,
        "LIMIT_DISTANCE" => ConstraintType::LimitDistance,
        "OBJECT_SOLVER" => ConstraintType::ObjectSolver,
        "TRANSFORM_CACHE" => ConstraintType::TransformCache,
        "IK" => ConstraintType::Ik,
        _ => return None,
    })
}

fn blender_constraint_is_importable(item: &Value) -> bool {
    let Some(kind) = item
        .get("type")
        .and_then(Value::as_str)
        .and_then(blender_constraint_type)
    else {
        return false;
    };
    if kind != ConstraintType::Action {
        return true;
    }
    item.get("properties")
        .and_then(Value::as_object)
        .is_some_and(|properties| {
            ["action", "target"].into_iter().all(|key| {
                properties
                    .get(key)
                    .and_then(|value| value.get("name"))
                    .and_then(Value::as_str)
                    .is_some_and(|name| !name.is_empty())
            })
        })
}
fn supported_blender_constraint_type(kind: ConstraintType) -> bool {
    matches!(
        kind,
        ConstraintType::CopyLocation
            | ConstraintType::CopyRotation
            | ConstraintType::CopyScale
            | ConstraintType::TrackTo
            | ConstraintType::DampedTrack
            | ConstraintType::LockedTrack
            | ConstraintType::StretchTo
            | ConstraintType::Transformation
            | ConstraintType::MaintainVolume
            | ConstraintType::Floor
            | ConstraintType::Pivot
            | ConstraintType::Shrinkwrap
            | ConstraintType::SplineIk
            | ConstraintType::LimitLocation
            | ConstraintType::LimitRotation
            | ConstraintType::LimitScale
            | ConstraintType::ChildOf
            | ConstraintType::Action
            | ConstraintType::Armature
            | ConstraintType::CameraSolver
            | ConstraintType::ClampTo
            | ConstraintType::CopyTransforms
            | ConstraintType::FollowPath
            | ConstraintType::FollowTrack
            | ConstraintType::GeometryAttribute
            | ConstraintType::LimitDistance
            | ConstraintType::ObjectSolver
            | ConstraintType::TransformCache
            | ConstraintType::Ik
    )
}

fn parse_transform_cache_params(
    item: &Value,
    properties: &Map<String, Value>,
    resource_ids: &BTreeMap<String, String>,
) -> Result<Map<String, Value>> {
    let path = item
        .get("cache_filepath")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| invalid_intermediate("Transform Cache constraint has no cache filepath"))?;
    let resource_identity = item
        .get("cache_file_name")
        .and_then(Value::as_str)
        .map(|identity| format!("{path}\0{identity}"));
    let resource = resource_identity
        .as_ref()
        .and_then(|key| resource_ids.get(key))
        .or_else(|| resource_ids.get(path))
        .ok_or_else(|| invalid_intermediate("Transform Cache filepath has no resource"))?;
    let object_path = properties
        .get("object_path")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_intermediate("Transform Cache constraint has no object path"))?;
    let settings = item.get("cache_file_settings").and_then(Value::as_object);
    let mut params = properties.clone();
    params.remove("cache_file");
    params.insert("resource".to_owned(), json!(resource));
    params.insert("object_path".to_owned(), json!(object_path));
    insert_cache_settings_params(&mut params, settings);
    Ok(params)
}

fn parse_constraints(
    value: &Value,
    owner: &Id,
    object_ids: &BTreeMap<String, Id>,
    movie_clip_ids: &BTreeMap<String, Id>,
    action_ids: &BTreeMap<String, Id>,
    resource_ids: &BTreeMap<String, String>,
    mappings: &mut BTreeMap<String, String>,
    owner_bone: Option<&Id>,
) -> Result<Vec<Constraint>> {
    let mapping_kind = owner_bone.map_or_else(
        || format!("Constraint:{owner}"),
        |bone_id| format!("Constraint:{owner}:{bone_id}"),
    );
    let id_prefix = owner_bone.map_or_else(
        || "constraint".to_owned(),
        |bone_id| format!("constraint_{}", bone_id.as_str()),
    );
    let mut by_name = BTreeMap::new();
    let mut constraints = Vec::new();
    for item in optional_array(value, "constraints")? {
        let name = item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("Constraint");
        let Some(constraint_type) = item
            .get("type")
            .and_then(Value::as_str)
            .and_then(blender_constraint_type)
        else {
            continue;
        };
        if !blender_constraint_is_importable(item) {
            continue;
        }
        let id = mapped_id(
            item,
            "potter_id",
            &id_prefix,
            name,
            &mut by_name,
            mappings,
            &mapping_kind,
        )?;
        let properties = item
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let mut params = if constraint_type == ConstraintType::TransformCache {
            parse_transform_cache_params(item, &properties, resource_ids)?
        } else {
            properties.clone()
        };
        for parameter in params.values_mut() {
            canonicalize_blender_id_references(parameter, object_ids, movie_clip_ids, action_ids)?;
        }
        let target_name = properties
            .get("target")
            .and_then(|target| target.get("name"))
            .and_then(Value::as_str);
        let target = target_name
            .map(|name| {
                object_ids.get(name).cloned().ok_or_else(|| {
                    invalid_intermediate("constraint references an unknown target object")
                })
            })
            .transpose()?;
        let armature_target = (constraint_type == ConstraintType::Armature)
            .then(|| {
                params
                    .get("targets")
                    .and_then(Value::as_array)
                    .and_then(|targets| targets.first())
                    .and_then(|target| target.get("target"))
                    .and_then(Value::as_str)
                    .map(|target| Id::new(target.to_owned()))
            })
            .flatten()
            .transpose()?;
        let target = target.or(armature_target);
        params.remove("target");
        if let Some(pole_target) = params.remove("pole_target") {
            match pole_target {
                Value::Null => {}
                Value::String(id) => {
                    params.insert("pole_target".to_owned(), Value::String(id));
                }
                _ => {
                    return Err(invalid_intermediate("IK pole target is invalid"));
                }
            }
        }
        let subtarget_name = if constraint_type == ConstraintType::Armature {
            params
                .get("targets")
                .and_then(Value::as_array)
                .and_then(|targets| targets.first())
                .and_then(|target| target.get("subtarget"))
                .and_then(Value::as_str)
        } else {
            params.get("subtarget").and_then(Value::as_str)
        }
        .filter(|name| !name.is_empty());
        let subtarget = subtarget_name.and_then(|name| {
            mappings
                .iter()
                .find(|(key, _)| key.starts_with("Bone:") && key.ends_with(&format!(":{name}")))
                .and_then(|(_, value)| Id::new(value.clone()).ok())
        });
        let influence = number_field(&Value::Object(properties.clone()), "influence", 1.0);
        let enabled = !properties
            .get("mute")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        params.remove("influence");
        params.remove("mute");
        params.remove("name");
        params.remove("type");
        params.remove("rna_type");
        let inverse_frame = if constraint_type == ConstraintType::ObjectSolver {
            item.get("inverse_frame")
                .map(|frame| {
                    frame
                        .as_f64()
                        .filter(|frame| frame.is_finite())
                        .ok_or_else(|| {
                            invalid_intermediate("Object Solver inverse frame is invalid")
                        })
                })
                .transpose()?
        } else {
            None
        };
        constraints.push(Constraint {
            id,
            constraint_type,
            name: name.to_owned(),
            target,
            subtarget,
            owner_bone: owner_bone.cloned(),
            influence,
            enabled,
            params,
            inverse_matrix: (constraint_type == ConstraintType::ObjectSolver)
                .then(|| matrix_column_major(item.get("inverse_matrix")))
                .flatten(),
            inverse_frame,
        });
    }
    Ok(constraints)
}

fn parse_drivers(
    value: &Value,
    owner: &Id,
    object_ids: &BTreeMap<String, Id>,
    mappings: &mut BTreeMap<String, String>,
) -> Result<Vec<Driver>> {
    let mut by_name = BTreeMap::new();
    optional_array(value, "drivers")?
        .iter()
        .map(|item| {
            let curve = item
                .get("curve")
                .ok_or_else(|| invalid_intermediate("driver has no F-Curve"))?;
            let parsed_curve = convert_fcurve(curve, mappings)?;
            let name = format!("{}[{}]", parsed_curve.path, parsed_curve.index);
            let id = mapped_id(
                item,
                "potter_id",
                "driver",
                &name,
                &mut by_name,
                mappings,
                &format!("Driver:{owner}"),
            )?;
            let driver = item
                .get("driver")
                .ok_or_else(|| invalid_intermediate("driver has no driver settings"))?;
            let driver_type = match driver.get("type").and_then(Value::as_str).unwrap_or("") {
                "SCRIPTED" => DriverType::ScriptedExpression,
                "SUM" => DriverType::Sum,
                "MIN" => DriverType::Min,
                "MAX" => DriverType::Max,
                _ => DriverType::Average,
            };
            let mut variables = Vec::new();
            for variable in optional_array(driver, "variables")? {
                let variable_type = match variable.get("type").and_then(Value::as_str).unwrap_or("")
                {
                    "TRANSFORMS" => DriverVariableType::Transforms,
                    _ => DriverVariableType::SingleProp,
                };
                let target = variable
                    .get("targets")
                    .and_then(Value::as_array)
                    .and_then(|targets| targets.first())
                    .ok_or_else(|| invalid_intermediate("driver variable has no target"))?;
                let target_name = target
                    .get("id")
                    .and_then(|id| id.get("name"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid_intermediate("driver target has no object ID"))?;
                let target_id = object_ids
                    .get(target_name)
                    .cloned()
                    .ok_or_else(|| invalid_intermediate("driver target object was not imported"))?;
                variables.push(DriverVariable {
                    name: variable
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("var")
                        .to_owned(),
                    variable_type,
                    target: target_id,
                    path: target
                        .get("data_path")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    index: 0,
                    target_2: None,
                    transform_space: None,
                });
            }
            Ok(Driver {
                id,
                path: parsed_curve.path,
                index: parsed_curve.index,
                driver_type,
                variables,
                expression: driver
                    .get("expression")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            })
        })
        .collect()
}

fn parse_nla_tracks(
    value: &Value,
    owner: &Id,
    action_by_name: &BTreeMap<String, Id>,
    mappings: &mut BTreeMap<String, String>,
) -> Result<Vec<NlaTrack>> {
    let mut track_ids = BTreeMap::new();
    let mut strip_ids = BTreeMap::new();
    optional_array(value, "nla_tracks")?
        .iter()
        .map(|track| {
            let name = track
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("NLA Track");
            let id = mapped_id(
                track,
                "potter_id",
                "nla_track",
                name,
                &mut track_ids,
                mappings,
                &format!("NlaTrack:{owner}"),
            )?;
            let strips = optional_array(track, "strips")?
                .iter()
                .map(|strip| {
                    let strip_name = strip
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("NLA Strip");
                    let strip_id = mapped_id(
                        strip,
                        "potter_id",
                        "nla_strip",
                        strip_name,
                        &mut strip_ids,
                        mappings,
                        &format!("NlaStrip:{owner}"),
                    )?;
                    let action_name = strip
                        .get("action")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid_intermediate("NLA strip has no action"))?;
                    let action = action_by_name.get(action_name).cloned().ok_or_else(|| {
                        invalid_intermediate("NLA strip references an unknown action")
                    })?;
                    let properties = strip.get("properties").unwrap_or(&Value::Null);
                    let blend_type = match properties
                        .get("blend_type")
                        .and_then(Value::as_str)
                        .unwrap_or("REPLACE")
                    {
                        "ADD" => NlaBlendType::Add,
                        "COMBINE" => NlaBlendType::Combine,
                        _ => NlaBlendType::Replace,
                    };
                    let extrapolation = match properties
                        .get("extrapolation")
                        .and_then(Value::as_str)
                        .unwrap_or("HOLD")
                    {
                        "HOLD_FORWARD" => NlaExtrapolation::HoldForward,
                        "NOTHING" => NlaExtrapolation::Nothing,
                        _ => NlaExtrapolation::Hold,
                    };
                    Ok(NlaStrip {
                        id: strip_id,
                        action,
                        frame_start: number_field(properties, "frame_start", 1.0),
                        frame_end: number_field(properties, "frame_end", 1.0),
                        action_frame_start: number_field(properties, "action_frame_start", 1.0),
                        action_frame_end: number_field(properties, "action_frame_end", 1.0),
                        scale: number_field(properties, "scale", 1.0),
                        repeat: number_field(properties, "repeat", 1.0),
                        blend_type,
                        influence: number_field(properties, "influence", 1.0),
                        extrapolation,
                        blend_in: number_field(properties, "blend_in", 0.0),
                        blend_out: number_field(properties, "blend_out", 0.0),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(NlaTrack {
                id,
                name: name.to_owned(),
                mute: track.get("mute").and_then(Value::as_bool).unwrap_or(false),
                solo: track.get("solo").and_then(Value::as_bool).unwrap_or(false),
                strips,
            })
        })
        .collect()
}

fn blender_rotation(item: &Value, mode: &str) -> [f64; 4] {
    let quat = if mode == "QUATERNION" {
        let raw = item.get("rotation_quaternion").and_then(Value::as_array);
        raw.and_then(|values| {
            Some(DQuat::from_xyzw(
                values.first()?.as_f64()?,
                values.get(1)?.as_f64()?,
                values.get(2)?.as_f64()?,
                values.get(3)?.as_f64()?,
            ))
        })
    } else if mode == "AXIS_ANGLE" {
        let raw = item.get("rotation_axis_angle").and_then(Value::as_array);
        raw.and_then(|values| {
            Some(DQuat::from_axis_angle(
                glam::DVec3::new(
                    values[1].as_f64()?,
                    values[2].as_f64()?,
                    values[3].as_f64()?,
                ),
                values[0].as_f64()?,
            ))
        })
    } else {
        let euler = vec3_field(item, "rotation_euler", [0.0; 3]);
        // Blender stores angles by axis and applies them in the selected extrinsic order.
        let (order, angles) = match mode {
            "XZY" => (EulerRot::XZYEx, [euler[0], euler[2], euler[1]]),
            "YXZ" => (EulerRot::YXZEx, [euler[1], euler[0], euler[2]]),
            "YZX" => (EulerRot::YZXEx, [euler[1], euler[2], euler[0]]),
            "ZXY" => (EulerRot::ZXYEx, [euler[2], euler[0], euler[1]]),
            "ZYX" => (EulerRot::ZYXEx, [euler[2], euler[1], euler[0]]),
            _ => (EulerRot::XYZEx, euler),
        };
        Some(DQuat::from_euler(order, angles[0], angles[1], angles[2]))
    };
    quat.map_or([0.0, 0.0, 0.0, 1.0], |rotation| {
        [rotation.x, rotation.y, rotation.z, rotation.w]
    })
}

fn root_matrix_override(item: &Value) -> Option<(Transform, [f64; 16])> {
    if item.get("parent").is_some_and(|parent| !parent.is_null()) {
        return None;
    }
    let raw = item
        .get("custom_properties")?
        .get("potter.root_matrix_json")?
        .as_str()?;
    let metadata: Value = serde_json::from_str(raw).ok()?;
    if !matrix_rows_equal(item.get("matrix_basis")?, metadata.get("baked_basis")?) {
        return None;
    }
    let transform = serde_json::from_value(metadata.get("transform")?.clone()).ok()?;
    let parent_inverse = serde_json::from_value(metadata.get("parent_inverse")?.clone()).ok()?;
    Some((transform, parent_inverse))
}

fn matrix_rows_equal(left: &Value, right: &Value) -> bool {
    let (Some(left), Some(right)) = (left.as_array(), right.as_array()) else {
        return false;
    };
    if left.len() != 4 || right.len() != 4 {
        return false;
    }
    for (left_row, right_row) in left.iter().zip(right) {
        let (Some(left_row), Some(right_row)) = (left_row.as_array(), right_row.as_array()) else {
            return false;
        };
        if left_row.len() != 4 || right_row.len() != 4 {
            return false;
        }
        for (left_value, right_value) in left_row.iter().zip(right_row) {
            let (Some(left_value), Some(right_value)) = (left_value.as_f64(), right_value.as_f64())
            else {
                return false;
            };
            if !left_value.is_finite()
                || !right_value.is_finite()
                || (left_value - right_value).abs() > 1.0e-4
            {
                return false;
            }
        }
    }
    true
}

fn root_matrix_requires_approximation(node: &Node) -> bool {
    if node.parent.is_some() {
        return false;
    }
    let Some(parent_inverse) = node.parent_inverse else {
        return false;
    };
    let rotation = node.transform.rotation;
    let local = DMat4::from_scale_rotation_translation(
        DVec3::from_array(node.transform.scale),
        DQuat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
        DVec3::from_array(node.transform.translation),
    );
    let matrix = DMat4::from_cols_array(&parent_inverse) * local;
    let axes = [
        matrix.x_axis.truncate(),
        matrix.y_axis.truncate(),
        matrix.z_axis.truncate(),
    ];
    for (index, left) in axes.iter().enumerate() {
        for right in axes.iter().skip(index + 1) {
            let magnitude = left.length() * right.length();
            if magnitude > 0.0 && left.dot(*right).abs() > magnitude * 1.0e-8 {
                return true;
            }
        }
    }
    false
}

fn object_properties(item: &Value) -> Map<String, Value> {
    let mut props = item
        .get("custom_properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    props.insert("blender_object_metadata".to_owned(), json!({
        "parent_type": item.get("parent_type"), "parent_bone": item.get("parent_bone"),
        "delta_location": item.get("delta_location"), "delta_rotation_euler": item.get("delta_rotation_euler"),
        "delta_rotation_quaternion": item.get("delta_rotation_quaternion"), "delta_scale": item.get("delta_scale"),
        "matrix_basis": item.get("matrix_basis"), "matrix_local": item.get("matrix_local"),
        "matrix_world": item.get("matrix_world"), "rna_properties": item.get("rna_properties"),
    }));
    props
}

fn import_losses(raw: &Value) -> Vec<Loss> {
    let mut losses = Vec::new();
    for object in raw
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("UNKNOWN");
        if !matches!(
            kind,
            "MESH"
                | "EMPTY"
                | "CAMERA"
                | "LIGHT"
                | "ARMATURE"
                | "GREASE_PENCIL"
                | "GREASE_PENCIL_V3"
                | "CURVE"
                | "SURFACE"
                | "FONT"
                | "VOLUME"
        ) {
            losses.push(loss(
                format!("blender.object.{}", kind.to_lowercase()),
                object.get("name").and_then(Value::as_str),
                "object type is preserved in compatibility data but not editable in the potter graph",
            ));
        }
        let constraints = object
            .get("constraints")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .chain(
                object
                    .get("pose")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .flat_map(|pose| {
                        pose.get("constraints")
                            .and_then(Value::as_array)
                            .into_iter()
                            .flatten()
                    }),
            );
        for constraint in constraints {
            let constraint_type = constraint
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("UNKNOWN");
            if !blender_constraint_is_importable(constraint) {
                losses.push(loss(
                    format!("blender.constraint.{}", constraint_type.to_lowercase()),
                    constraint.get("name").and_then(Value::as_str),
                    "constraint is preserved in compatibility data but is not editable/evaluated by potter",
                ));
            }
        }
        for modifier in object
            .get("modifiers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let kind = modifier
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("UNKNOWN");
            if !known_modifier(kind) {
                losses.push(loss(format!("blender.modifier.{}", kind.to_lowercase()), modifier.get("name").and_then(Value::as_str), "modifier properties are preserved, but this modifier is not editable/evaluated by potter"));
            }
        }
    }
    for action in raw
        .get("actions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let action_name = action.get("name").and_then(Value::as_str);
        if action
            .get("slot_count")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            > 1
        {
            losses.push(loss(
                "blender.animation.action_slot",
                action_name,
                "action slots are flattened into a single F-Curve list",
            ));
        }
        for curve in action
            .get("fcurves")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if curve
                .get("modifiers")
                .and_then(Value::as_array)
                .is_some_and(|items| !items.is_empty())
            {
                losses.push(loss(
                    "blender.animation.fcurve_modifier",
                    action_name,
                    "F-Curve modifiers are retained in compatibility data but not editable",
                ));
            }
            if !matches!(
                curve.get("extrapolation").and_then(Value::as_str),
                None | Some("CONSTANT" | "LINEAR")
            ) {
                losses.push(loss(
                    "blender.animation.fcurve_extrapolation",
                    action_name,
                    "this F-Curve extrapolation mode is not representable",
                ));
            }
            for key in curve
                .get("keyframes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if !matches!(
                    key.get("interpolation").and_then(Value::as_str),
                    None | Some("CONSTANT" | "LINEAR" | "BEZIER")
                ) {
                    losses.push(loss(
                        "blender.animation.keyframe_interpolation",
                        action_name,
                        "this keyframe interpolation mode is not representable",
                    ));
                }
                if !matches!(
                    key.get("handle_left_type").and_then(Value::as_str),
                    None | Some("AUTO" | "AUTO_CLAMPED")
                ) || !matches!(
                    key.get("handle_right_type").and_then(Value::as_str),
                    None | Some("AUTO" | "AUTO_CLAMPED")
                ) {
                    losses.push(loss(
                        "blender.animation.keyframe_handles",
                        action_name,
                        "custom Bézier handle modes are retained only in compatibility data",
                    ));
                    break;
                }
            }
        }
    }
    for scene in raw
        .get("scenes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(world) = scene.get("world").filter(|world| !world.is_null()) {
            let has_unmapped_nodes = world
                .get("nodes")
                .and_then(|tree| tree.get("nodes"))
                .and_then(Value::as_array)
                .is_some_and(|nodes| {
                    nodes.iter().any(|node| {
                        !matches!(
                            node.get("type").and_then(Value::as_str),
                            Some("ShaderNodeBackground" | "ShaderNodeOutputWorld")
                        )
                    })
                });
            if has_unmapped_nodes {
                losses.push(loss(
                    "blender.world.node_graph",
                    scene.get("name").and_then(Value::as_str),
                    "world nodes beyond background color/strength are retained only in compatibility data",
                ));
            }
        }
        let native_engine = scene
            .get("render")
            .and_then(|render| render.get("engine_native"))
            .and_then(Value::as_str);
        if native_engine.is_some_and(|engine| {
            !matches!(engine, "CYCLES" | "BLENDER_EEVEE" | "BLENDER_EEVEE_NEXT")
        }) {
            losses.push(loss(
                "blender.render.engine",
                scene.get("name").and_then(Value::as_str),
                "this render engine is retained in compatibility data but not re-created",
            ));
        }
    }
    for data in raw
        .get("other_datablocks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let kind = data
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if kind == "shape_keys" {
            // Key containers are imported per-mesh as editable shape keys; the
            // container itself stays in the compatibility payload.
            continue;
        }
        if kind == "particles" {
            // ParticleSettings are represented by their particle-system or fluid settings.
            continue;
        }
        losses.push(loss(
            format!("blender.datablock.{kind}"),
            data.get("name").and_then(Value::as_str),
            "data-block type is retained in compatibility data but is not part of the editable graph",
        ));
    }
    if raw
        .get("unused_datablocks")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items.iter().any(|item| {
                !matches!(
                    item.get("type").and_then(Value::as_str),
                    Some(
                        "meshes"
                            | "materials"
                            | "cameras"
                            | "lights"
                            | "actions"
                            | "volumes"
                            | "worlds"
                    )
                )
            })
        })
    {
        losses.push(loss(
            "blender.orphan_datablock",
            None,
            "orphan IDs outside the editable mesh/material/camera/light/action/world registries remain preserve-only",
        ));
    }
    losses
}

fn has_verbatim_native_movie_clip(doc: &SceneDoc, clip: &MovieClip) -> bool {
    let original_blend = doc.compatibility.values().any(|entry| {
        entry.get("format").and_then(Value::as_str) == Some("blend")
            && entry
                .get("blobs")
                .and_then(Value::as_array)
                .is_some_and(|blobs| {
                    blobs.iter().any(|blob| {
                        blob.get("name").and_then(Value::as_str) == Some("original.blend")
                    })
                })
    });
    if !original_blend {
        return false;
    }
    let Some(intermediate) = doc
        .compatibility
        .get("blender_adapter")
        .and_then(|adapter| adapter.get("intermediate"))
    else {
        return false;
    };
    let Some(source_clip) = intermediate
        .get("movie_clips")
        .and_then(Value::as_array)
        .and_then(|clips| {
            clips.iter().find(|source| {
                source.get("name").and_then(Value::as_str) == Some(clip.name.as_str())
            })
        })
    else {
        return false;
    };
    let Some(source_tracking) = source_clip.get("tracking").and_then(Value::as_object) else {
        return false;
    };
    let Ok(current_tracking) = serde_json::to_value(&clip.tracking) else {
        return false;
    };
    let Some(current_tracking) = current_tracking.as_object() else {
        return false;
    };
    source_tracking
        .iter()
        .filter(|(key, _)| key.as_str() != "camera")
        .all(|(key, value)| current_tracking.get(key) == Some(value))
        && current_tracking
            .iter()
            .filter(|(key, _)| key.as_str() != "camera")
            .all(|(key, value)| source_tracking.get(key) == Some(value))
}

pub(crate) fn export_losses(doc: &SceneDoc) -> Vec<Loss> {
    let mut losses = Vec::new();
    for (clip_id, clip) in &doc.movie_clips {
        let reconstruction = &clip.tracking.reconstruction;
        let has_reconstruction = !reconstruction.cameras.is_empty()
            || !reconstruction.points.is_empty()
            || reconstruction.is_valid
            || reconstruction.average_error != 0.0
            || clip.tracking.objects.iter().any(|object| {
                !object.reconstruction.is_empty()
                    || object.reconstruction_is_valid
                    || object.reconstruction_average_error != 0.0
            });
        if has_reconstruction && !has_verbatim_native_movie_clip(doc, clip) {
            losses.push(loss(
                "blender.movieclip.reconstruction",
                Some(clip_id.as_str()),
                "Blender 5.2.2 reconstruction cameras, bundles, validity, and errors are read-only RNA; without unchanged native MovieClip data they cannot be restored, so tracking constraints may evaluate differently",
            ));
        }
    }
    for (id, node) in &doc.nodes {
        if !matches!(
            node.kind.as_str(),
            "mesh"
                | "empty"
                | "camera"
                | "light"
                | "armature"
                | "grease_pencil"
                | "curve"
                | "surface"
                | "text"
                | "group"
                | "collection_instance"
                | "volume"
        ) {
            losses.push(loss(
                format!("blender.object.{}", node.kind),
                Some(id.as_str()),
                "object kind cannot be represented as a Blender object",
            ));
        }
        for constraint in &node.constraints {
            if !supported_blender_constraint_type(constraint.constraint_type) {
                let type_name = match constraint.constraint_type {
                    ConstraintType::Action => "action",
                    ConstraintType::CameraSolver => "camera_solver",
                    ConstraintType::FollowTrack => "follow_track",
                    ConstraintType::GeometryAttribute => "geometry_attribute",
                    ConstraintType::ObjectSolver => "object_solver",
                    ConstraintType::TransformCache => "transform_cache",
                    _ => "unsupported",
                };
                losses.push(loss(
                    format!("blender.constraint.{type_name}"),
                    Some(constraint.id.as_str()),
                    "constraint type is represented but not supported by the Blender adapter",
                ));
            }
        }
        for modifier in &node.modifiers {
            if !known_modifier(&modifier.modifier_type.to_ascii_uppercase()) {
                losses.push(loss(
                    format!("blender.modifier.{}", modifier.modifier_type),
                    Some(modifier.id.as_str()),
                    "modifier kind is not supported by the adapter",
                ));
            }
        }
    }
    for (scene_id, scene) in &doc.scenes {
        if scene
            .world
            .as_ref()
            .is_some_and(|world| !doc.worlds.contains_key(world))
        {
            losses.push(loss(
                "blender.world",
                Some(scene_id.as_str()),
                "the scene references a world missing from the world registry",
            ));
        }
        if !matches!(scene.render.engine.as_str(), "path" | "realtime") {
            losses.push(loss(
                "blender.render.engine",
                Some(scene_id.as_str()),
                "only path and realtime render engines are mapped by the adapter",
            ));
        }
    }
    for (data_id, data) in &doc.data_blocks {
        if !matches!(
            data.data_type.as_str(),
            "mesh"
                | "camera"
                | "light"
                | "armature"
                | "grease_pencil"
                | "curve"
                | "surface"
                | "text"
                | "volume"
        ) {
            losses.push(loss(
                format!("blender.datablock.{}", data.data_type),
                Some(data_id.as_str()),
                "data-block is preserved in compatibility JSON but is not reconstructed as a Blender data-block",
            ));
        }
        if let Some(mesh) = &data.mesh {
            for attribute in mesh.attributes.keys() {
                if !matches!(
                    attribute.as_str(),
                    "blender_uv_layers"
                        | "blender_edge_flags"
                        | "blender_polygon_smooth"
                        | "blender_metadata"
                        | "blender_attributes"
                        | "blender_skin_vertices"
                        | "skin_radii"
                        | "skin_roots"
                ) {
                    losses.push(loss(
                        "blender.mesh.attribute",
                        Some(data_id.as_str()),
                        "custom mesh attributes are retained in compatibility JSON but not recreated",
                    ));
                    break;
                }
            }
        }
        if let Some(descriptor) = &data.descriptor
            && !matches!(
                descriptor.primitive.as_str(),
                "box"
                    | "sphere"
                    | "uv_sphere"
                    | "cylinder"
                    | "plane"
                    | "cone"
                    | "torus"
                    | "icosphere"
                    | "circle"
                    | "grid"
            )
        {
            losses.push(loss(
                "blender.mesh.primitive",
                Some(data_id.as_str()),
                "primitive descriptor cannot be expanded by the Blender bridge",
            ));
        }
    }
    if let Some(intermediate) = doc
        .compatibility
        .get("blender_adapter")
        .and_then(|adapter| adapter.get("intermediate"))
    {
        losses.extend(import_losses(intermediate));
    }
    losses
}

fn original_blend_blob_exists(doc: &SceneDoc, project: &Project) -> bool {
    doc.compatibility.values().any(|entry| {
        entry.get("format").and_then(Value::as_str) == Some("blend")
            && entry
                .get("blobs")
                .and_then(Value::as_array)
                .is_some_and(|blobs| {
                    blobs.iter().any(|blob| {
                        blob.get("name").and_then(Value::as_str) == Some("original.blend")
                            && blob
                                .get("path")
                                .and_then(Value::as_str)
                                .is_some_and(|path| project.path().join(path).is_file())
                    })
                })
    })
}

fn raw_mesh_for_data<'a>(doc: &'a SceneDoc, data_id: &str) -> Option<&'a Value> {
    let adapter = doc.compatibility.get("blender_adapter")?;
    let intermediate = adapter.get("intermediate")?;
    let mappings = adapter.get("id_mappings")?;
    intermediate.get("meshes")?.as_array()?.iter().find(|mesh| {
        mesh.get("potter_id").and_then(Value::as_str) == Some(data_id)
            || mesh
                .get("name")
                .and_then(Value::as_str)
                .and_then(|name| mappings.get(format!("Mesh:{name}")))
                .and_then(Value::as_str)
                == Some(data_id)
    })
}

fn mesh_topology_matches_raw(mesh: &Mesh, raw: &Value) -> bool {
    let Some(vertices) = raw.get("vertices").and_then(Value::as_array) else {
        return false;
    };
    let Some(edges) = raw.get("edges").and_then(Value::as_array) else {
        return false;
    };
    let Some(polygons) = raw.get("polygons").and_then(Value::as_array) else {
        return false;
    };
    if vertices.len() != mesh.vertices.len()
        || edges.len() != mesh.edges.len()
        || polygons.len() != mesh.faces.len()
    {
        return false;
    }
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    for (edge, raw_edge) in mesh.edges.iter().zip(edges) {
        let Some(raw_vertices) = raw_edge
            .get("v")
            .and_then(Value::as_array)
            .filter(|vertices| vertices.len() == 2)
        else {
            return false;
        };
        let Some(first) = vertex_indices.get(&edge.vertices[0]) else {
            return false;
        };
        let Some(second) = vertex_indices.get(&edge.vertices[1]) else {
            return false;
        };
        let raw_first = raw_vertices[0]
            .as_u64()
            .and_then(|index| usize::try_from(index).ok());
        let raw_second = raw_vertices[1]
            .as_u64()
            .and_then(|index| usize::try_from(index).ok());
        if raw_first != Some(*first) || raw_second != Some(*second) {
            return false;
        }
    }
    for (face, raw_face) in mesh.faces.iter().zip(polygons) {
        let Some(raw_vertices) = raw_face.get("v").and_then(Value::as_array) else {
            return false;
        };
        if raw_vertices.len() != face.vertices.len()
            || face
                .vertices
                .iter()
                .zip(raw_vertices)
                .any(|(vertex, raw_vertex)| {
                    vertex_indices.get(vertex).copied()
                        != raw_vertex
                            .as_u64()
                            .and_then(|index| usize::try_from(index).ok())
                })
        {
            return false;
        }
    }
    true
}

fn native_bind_source_topology_matches(doc: &SceneDoc, node_id: &Id, modifier: &Modifier) -> bool {
    let Some(node) = doc.nodes.get(node_id) else {
        return false;
    };
    let Some(data_id) = node.data.as_ref() else {
        return false;
    };
    let Some(mesh) = doc
        .data_blocks
        .get(data_id)
        .and_then(|data| data.mesh.as_ref())
    else {
        return false;
    };
    let Some(raw) = raw_mesh_for_data(doc, data_id.as_str()) else {
        return false;
    };
    if !mesh_topology_matches_raw(mesh, raw) {
        return false;
    }
    let target_key = match modifier.modifier_type.as_str() {
        "surface_deform" => Some("target"),
        "mesh_deform" => Some("object"),
        _ => None,
    };
    let Some(target_key) = target_key else {
        return true;
    };
    let Some(target_id) = modifier.params.get(target_key).and_then(Value::as_str) else {
        return false;
    };
    let Some(target_node) = doc
        .nodes
        .iter()
        .find_map(|(id, node)| (id.as_str() == target_id).then_some(node))
    else {
        return false;
    };
    let Some(target_data_id) = target_node.data.as_ref() else {
        return false;
    };
    let Some(target_mesh) = doc
        .data_blocks
        .get(target_data_id)
        .and_then(|data| data.mesh.as_ref())
    else {
        return false;
    };
    raw_mesh_for_data(doc, target_data_id.as_str())
        .is_some_and(|raw| mesh_topology_matches_raw(target_mesh, raw))
}

fn native_bind_export_losses(doc: &SceneDoc, project: &Project) -> Vec<Loss> {
    let original_blend_exists = original_blend_blob_exists(doc, project);
    let mut losses = Vec::new();
    for (node_id, node) in &doc.nodes {
        for modifier in &node.modifiers {
            let native_binding = modifier
                .binding_data
                .as_ref()
                .and_then(Value::as_object)
                .is_some_and(|data| {
                    data.get("format").and_then(Value::as_str) == Some("blender_native_bind_v1")
                });
            if !native_binding {
                continue;
            }
            if !original_blend_exists
                || !native_bind_source_topology_matches(doc, node_id, modifier)
            {
                losses.push(loss(
                    "blender.modifier.native_bind",
                    Some(modifier.id.as_str()),
                    "the original native bind cannot be reused because its compatibility .blend is unavailable or the bound mesh topology changed; Blender rebinding may change the saved bind state",
                ));
            }
        }
    }
    losses
}
fn potter_modifier_type(blender_type: &str) -> Option<&'static str> {
    Some(match blender_type {
        "MIRROR" => "mirror",
        "ARRAY" => "array",
        "SUBSURF" | "SUBDIVISION" => "subdivision",
        "MULTIRES" => "multires",
        "SOLIDIFY" => "solidify",
        "TRIANGULATE" => "triangulate",
        "VOLUME_DISPLACE" => "volume_displace",
        "BEVEL" => "bevel",
        "DECIMATE" => "decimate",
        "WELD" => "weld",
        "DISPLACE" => "displace",
        "SMOOTH" => "smooth",
        "NODES" => "nodes",
        "ARMATURE" => "armature",
        "LATTICE" => "lattice",
        "VOLUME_TO_MESH" => "volume_to_mesh",
        "MESH_TO_VOLUME" => "mesh_to_volume",
        "BOOLEAN" => "boolean",
        "SHRINKWRAP" => "shrinkwrap",
        "CAST" => "cast",
        "CURVE" => "curve",
        "HOOK" => "hook",
        "LAPLACIANSMOOTH" => "laplacian_smooth",
        "LAPLACIANDEFORM" | "LAPLACIAN_DEFORM" => "laplacian_deform",
        "CORRECTIVE_SMOOTH" => "corrective_smooth",
        "WAVE" => "wave",
        "WARP" => "warp",
        "SIMPLE_DEFORM" => "simple_deform",
        "SCREW" => "screw",
        "SKIN" => "skin",
        "WIREFRAME" => "wireframe",
        "EDGE_SPLIT" => "edge_split",
        "BUILD" => "build",
        "MASK" => "mask",
        "WEIGHTED_NORMAL" => "weighted_normal",
        "NORMAL_EDIT" => "normal_edit",
        "UV_PROJECT" => "uv_project",
        "UV_WARP" => "uv_warp",
        "VERTEX_WEIGHT_EDIT" => "vertex_weight_edit",
        "VERTEX_WEIGHT_MIX" => "vertex_weight_mix",
        "VERTEX_WEIGHT_PROXIMITY" => "vertex_weight_proximity",
        "SURFACE_DEFORM" => "surface_deform",
        "MESH_DEFORM" => "mesh_deform",
        "DATA_TRANSFER" => "data_transfer",
        "OCEAN" => "ocean",
        "PARTICLE_INSTANCE" => "particle_instance",
        "EXPLODE" => "explode",
        "FLUID" => "fluid",
        "CLOTH" => "cloth",
        "SOFT_BODY" => "soft_body",
        "COLLISION" => "collision",
        "DYNAMIC_PAINT" => "dynamic_paint",
        "PARTICLE_SYSTEM" => "particle_system",
        "REMESH" => "remesh",
        "MESH_SEQUENCE_CACHE" => "mesh_sequence_cache",
        "MESH_CACHE" => "mesh_cache",
        _ => return None,
    })
}

fn known_modifier(kind: &str) -> bool {
    potter_modifier_type(kind).is_some() || matches!(kind, "SUBDIVISION" | "LAPLACIAN_SMOOTH")
}
fn attach_native_target_vertex_orders(doc: &mut SceneDoc) {
    let mut binds = Vec::new();
    for (owner_id, node) in &doc.nodes {
        for modifier in &node.modifiers {
            let Some(binding) = modifier
                .binding_data
                .as_ref()
                .and_then(Value::as_object)
                .filter(|binding| {
                    binding.get("format").and_then(Value::as_str) == Some("blender_native_bind_v1")
                        && binding.get("type").and_then(Value::as_str) == Some("surface_deform")
                })
            else {
                continue;
            };
            let Some(positions) = binding
                .get("target_evaluated_vertices")
                .and_then(Value::as_array)
                .cloned()
            else {
                continue;
            };
            let Some(target_text) = modifier.params.get("target").and_then(Value::as_str) else {
                continue;
            };
            let Ok(target_id) = Id::new(target_text.to_owned()) else {
                continue;
            };
            binds.push((owner_id.clone(), modifier.id.clone(), target_id, positions));
        }
    }
    let targets = binds
        .iter()
        .map(|(_, _, target_id, _)| target_id.clone())
        .collect::<BTreeSet<_>>();
    for target_id in targets {
        let selected = BTreeSet::from([target_id.clone()]);
        let snapshot = crate::eval::Snapshot::evaluate_nodes_with_cache(
            doc,
            &crate::eval::EvaluationContext::default(),
            None,
            &selected,
        )
        .ok();
        let target_mesh = snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.meshes.get(&target_id));
        for (owner_id, modifier_id, _bind_target_id, blender_positions) in
            binds.iter().filter(|bind| bind.2 == target_id)
        {
            let order =
                target_mesh.and_then(|mesh| native_target_vertex_order(blender_positions, mesh));
            let Some(binding) = doc
                .nodes
                .get_mut(owner_id)
                .and_then(|node| {
                    node.modifiers
                        .iter_mut()
                        .find(|modifier| modifier.id == *modifier_id)
                })
                .and_then(|modifier| modifier.binding_data.as_mut())
                .and_then(Value::as_object_mut)
            else {
                continue;
            };
            binding.remove("target_evaluated_vertices");
            if let Some(order) = order {
                binding.insert("target_vertex_order".to_owned(), json!(order));
                binding.remove("target_vertex_order_error");
            } else {
                binding.insert(
                    "target_vertex_order_error".to_owned(),
                    json!("the evaluated target vertex order could not be matched"),
                );
            }
        }
    }
}

fn native_target_vertex_order(blender_positions: &[Value], mesh: &Mesh) -> Option<Vec<usize>> {
    if blender_positions.len() != mesh.vertices.len() {
        return None;
    }
    let blender_positions = blender_positions
        .iter()
        .map(value_position)
        .collect::<Option<Vec<_>>>()?;
    let actual_positions = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let scale = blender_positions
        .iter()
        .chain(&actual_positions)
        .map(|position| position.abs().max_element())
        .fold(1.0_f64, f64::max);
    let tolerance = scale * 1.0e-4;
    let tolerance_squared = tolerance * tolerance;
    let mut buckets = HashMap::<[i64; 3], Vec<usize>>::new();
    for (index, position) in actual_positions.iter().enumerate() {
        buckets
            .entry(target_position_bucket(*position, tolerance))
            .or_default()
            .push(index);
    }
    let mut used = vec![false; actual_positions.len()];
    let mut order = Vec::with_capacity(blender_positions.len());
    for position in blender_positions {
        let center = target_position_bucket(position, tolerance);
        let mut nearest = None;
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let key = [center[0] + x, center[1] + y, center[2] + z];
                    for candidate in buckets.get(&key).into_iter().flatten() {
                        if used[*candidate] {
                            continue;
                        }
                        let distance_squared =
                            position.distance_squared(actual_positions[*candidate]);
                        if distance_squared <= tolerance_squared
                            && nearest.is_none_or(|(_, best)| distance_squared < best)
                        {
                            nearest = Some((*candidate, distance_squared));
                        }
                    }
                }
            }
        }
        let (index, _) = nearest?;
        used[index] = true;
        order.push(index);
    }
    Some(order)
}

fn value_position(value: &Value) -> Option<DVec3> {
    let coordinates = value.as_array()?;
    if coordinates.len() != 3 {
        return None;
    }
    Some(DVec3::new(
        coordinates[0].as_f64().filter(|value| value.is_finite())?,
        coordinates[1].as_f64().filter(|value| value.is_finite())?,
        coordinates[2].as_f64().filter(|value| value.is_finite())?,
    ))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "coordinate buckets only need integer cell indices within a bounded mesh extent"
)]
fn target_position_bucket(position: DVec3, tolerance: f64) -> [i64; 3] {
    [
        (position.x / tolerance).floor() as i64,
        (position.y / tolerance).floor() as i64,
        (position.z / tolerance).floor() as i64,
    ]
}

fn validate_native_modifier_bindings(doc: &mut SceneDoc) -> Result<()> {
    for (node_id, node) in &mut doc.nodes {
        for modifier in &mut node.modifiers {
            let is_bound = modifier
                .params
                .remove("is_bound")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let is_bind = modifier
                .params
                .remove("is_bind")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let should_bind = is_bound || is_bind;
            if should_bind
                && matches!(
                    modifier.modifier_type.as_str(),
                    "surface_deform" | "mesh_deform" | "laplacian_deform"
                )
                && modifier.binding_data.is_none()
            {
                return Err(PotError::with_details(
                    ErrorCode::ImportFailed,
                    format!(
                        "bound {} modifier `{}` on `{}` has no native bind payload",
                        modifier.modifier_type, modifier.name, node.name
                    ),
                    json!({
                        "node_id": node_id,
                        "modifier_id": modifier.id,
                        "modifier": modifier.name,
                        "type": modifier.modifier_type
                    }),
                ));
            }
        }
    }
    Ok(())
}

fn loss(feature: impl Into<String>, data_id: Option<&str>, reason: &str) -> Loss {
    Loss { feature_id: feature.into(), data_id: data_id.map(str::to_owned), reason: reason.to_owned(),
           suggestion: Some("retain the compatibility payload, or use --allow-lossy only when the loss is acceptable".to_owned()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn modifier_type_mapping_covers_the_potter_schema() {
        let mappings = [
            ("mirror", "MIRROR"),
            ("array", "ARRAY"),
            ("subdivision", "SUBSURF"),
            ("multires", "MULTIRES"),
            ("solidify", "SOLIDIFY"),
            ("triangulate", "TRIANGULATE"),
            ("bevel", "BEVEL"),
            ("decimate", "DECIMATE"),
            ("weld", "WELD"),
            ("displace", "DISPLACE"),
            ("smooth", "SMOOTH"),
            ("nodes", "NODES"),
            ("armature", "ARMATURE"),
            ("lattice", "LATTICE"),
            ("volume_to_mesh", "VOLUME_TO_MESH"),
            ("mesh_to_volume", "MESH_TO_VOLUME"),
            ("boolean", "BOOLEAN"),
            ("shrinkwrap", "SHRINKWRAP"),
            ("cast", "CAST"),
            ("curve", "CURVE"),
            ("hook", "HOOK"),
            ("laplacian_smooth", "LAPLACIANSMOOTH"),
            ("laplacian_deform", "LAPLACIANDEFORM"),
            ("corrective_smooth", "CORRECTIVE_SMOOTH"),
            ("wave", "WAVE"),
            ("warp", "WARP"),
            ("simple_deform", "SIMPLE_DEFORM"),
            ("screw", "SCREW"),
            ("skin", "SKIN"),
            ("wireframe", "WIREFRAME"),
            ("edge_split", "EDGE_SPLIT"),
            ("build", "BUILD"),
            ("mask", "MASK"),
            ("weighted_normal", "WEIGHTED_NORMAL"),
            ("normal_edit", "NORMAL_EDIT"),
            ("uv_project", "UV_PROJECT"),
            ("uv_warp", "UV_WARP"),
            ("vertex_weight_edit", "VERTEX_WEIGHT_EDIT"),
            ("vertex_weight_mix", "VERTEX_WEIGHT_MIX"),
            ("vertex_weight_proximity", "VERTEX_WEIGHT_PROXIMITY"),
            ("surface_deform", "SURFACE_DEFORM"),
            ("mesh_deform", "MESH_DEFORM"),
            ("data_transfer", "DATA_TRANSFER"),
            ("ocean", "OCEAN"),
            ("particle_instance", "PARTICLE_INSTANCE"),
            ("explode", "EXPLODE"),
            ("fluid", "FLUID"),
            ("cloth", "CLOTH"),
            ("soft_body", "SOFT_BODY"),
            ("collision", "COLLISION"),
            ("dynamic_paint", "DYNAMIC_PAINT"),
            ("particle_system", "PARTICLE_SYSTEM"),
            ("remesh", "REMESH"),
            ("mesh_sequence_cache", "MESH_SEQUENCE_CACHE"),
        ];
        for (potter, blender) in mappings {
            assert_eq!(potter_modifier_type(blender), Some(potter), "{blender}");
            assert!(known_modifier(blender), "{blender}");
            assert!(known_modifier(&potter.to_ascii_uppercase()), "{potter}");
        }
    }

    #[test]
    fn constraint_type_mapping_covers_the_potter_schema() {
        let mappings = [
            ("COPY_LOCATION", ConstraintType::CopyLocation),
            ("COPY_ROTATION", ConstraintType::CopyRotation),
            ("COPY_SCALE", ConstraintType::CopyScale),
            ("TRACK_TO", ConstraintType::TrackTo),
            ("DAMPED_TRACK", ConstraintType::DampedTrack),
            ("LOCKED_TRACK", ConstraintType::LockedTrack),
            ("STRETCH_TO", ConstraintType::StretchTo),
            ("TRANSFORM", ConstraintType::Transformation),
            ("MAINTAIN_VOLUME", ConstraintType::MaintainVolume),
            ("FLOOR", ConstraintType::Floor),
            ("PIVOT", ConstraintType::Pivot),
            ("SHRINKWRAP", ConstraintType::Shrinkwrap),
            ("SPLINE_IK", ConstraintType::SplineIk),
            ("LIMIT_LOCATION", ConstraintType::LimitLocation),
            ("LIMIT_ROTATION", ConstraintType::LimitRotation),
            ("LIMIT_SCALE", ConstraintType::LimitScale),
            ("CHILD_OF", ConstraintType::ChildOf),
            ("ACTION", ConstraintType::Action),
            ("ARMATURE", ConstraintType::Armature),
            ("CAMERA_SOLVER", ConstraintType::CameraSolver),
            ("CLAMP_TO", ConstraintType::ClampTo),
            ("COPY_TRANSFORMS", ConstraintType::CopyTransforms),
            ("FOLLOW_PATH", ConstraintType::FollowPath),
            ("FOLLOW_TRACK", ConstraintType::FollowTrack),
            ("GEOMETRY_ATTRIBUTE", ConstraintType::GeometryAttribute),
            ("LIMIT_DISTANCE", ConstraintType::LimitDistance),
            ("OBJECT_SOLVER", ConstraintType::ObjectSolver),
            ("TRANSFORM_CACHE", ConstraintType::TransformCache),
            ("IK", ConstraintType::Ik),
        ];
        for (blender, expected) in mappings {
            assert_eq!(
                blender_constraint_type(blender),
                Some(expected),
                "{blender}"
            );
            assert!(supported_blender_constraint_type(expected), "{blender}");
        }
    }

    #[test]
    fn ids_are_sanitized_unique_and_type_prefixed() {
        let mut used = BTreeSet::new();
        assert_eq!(unique_id("My Cube", "node", &mut used), "node_my_cube");
        assert_eq!(unique_id("My Cube", "node", &mut used), "node_my_cube_2");
        assert_eq!(unique_id("___", "mesh", &mut used), "mesh_unnamed");
    }

    #[test]
    fn blender_version_parser_reads_numeric_triplets() {
        assert_eq!(parse_version_line("Blender 5.2.2"), Some((5, 2, 2)));
        assert_eq!(parse_version_line("Blender 5.3.0 Alpha"), Some((5, 3, 0)));
        assert_eq!(parse_version_line("not blender"), None);
    }

    #[test]
    fn blender_animation_paths_map_to_potter_model_paths() {
        let mappings = BTreeMap::from([("Bone:Armature:Root".to_owned(), "bone_root".to_owned())]);
        for (blender, potter) in [
            ("location", "transform.translation"),
            ("rotation_euler", "transform.rotation_euler"),
            ("rotation_quaternion", "transform.rotation_quaternion"),
            ("scale", "transform.scale"),
            ("data.lens", "camera.lens_mm"),
            ("data.energy", "light.energy"),
            (
                "pose.bones[\"Root\"].location",
                "pose.bones[\"bone_root\"].location",
            ),
        ] {
            assert_eq!(potter_fcurve_path(blender, &mappings), potter);
        }
    }

    #[test]
    fn blender_quaternion_components_convert_to_potter_xyzw() -> Result<()> {
        assert_eq!(
            blender_rotation(
                &json!({"rotation_quaternion": [1.0, 0.0, 0.0, 0.0]}),
                "QUATERNION",
            ),
            [1.0, 0.0, 0.0, 0.0],
        );
        for (blender_index, potter_index) in [(0_i64, 3_u32), (1, 0), (2, 1), (3, 2)] {
            let curve = convert_fcurve(
                &json!({"path": "rotation_quaternion", "index": blender_index}),
                &BTreeMap::new(),
            )?;
            assert_eq!(curve.path, "transform.rotation_quaternion");
            assert_eq!(curve.index, potter_index);
        }
        Ok(())
    }
    #[test]
    fn every_evaluated_modifier_maps_to_a_blender_modifier_type() {
        for (blender, potter) in [
            ("MIRROR", "mirror"),
            ("ARRAY", "array"),
            ("SUBSURF", "subdivision"),
            ("SOLIDIFY", "solidify"),
            ("TRIANGULATE", "triangulate"),
            ("BEVEL", "bevel"),
            ("DECIMATE", "decimate"),
            ("WELD", "weld"),
            ("DISPLACE", "displace"),
            ("MESH_SEQUENCE_CACHE", "mesh_sequence_cache"),
            ("NODES", "nodes"),
        ] {
            assert_eq!(potter_modifier_type(blender), Some(potter));
            assert!(known_modifier(blender));
        }
    }

    #[test]
    fn blend_pose_constraints_map_owner_ids_and_report_unsupported_types() -> Result<()> {
        let owner = Id::new("arm")?;
        let owner_bone = Id::new("tip")?;
        let object_ids = BTreeMap::from([("Goal".to_owned(), Id::new("goal")?)]);
        let mut mappings = BTreeMap::from([("Bone:Rig:Tip".to_owned(), "tip".to_owned())]);
        let constraints = parse_constraints(
            &json!({"constraints":[
                {"name":"Editable Copy","type":"COPY_TRANSFORMS",
                 "properties":{"target":{"name":"Goal"},"influence":0.5}},
                {"name":"Unsupported","type":"ALIEN_CONSTRAINT","properties":{}}
            ]}),
            &owner,
            &object_ids,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &mut mappings,
            Some(&owner_bone),
        )?;
        assert_eq!(constraints.len(), 1);
        assert_eq!(
            constraints[0].constraint_type,
            ConstraintType::CopyTransforms
        );
        assert_eq!(constraints[0].owner_bone.as_ref(), Some(&owner_bone));
        assert_eq!(constraints[0].target.as_ref(), Some(&Id::new("goal")?));
        assert_eq!(constraints[0].influence, 0.5);

        let losses = import_losses(&json!({
            "objects":[{
                "name":"Rig","type":"ARMATURE",
                "constraints":[{"name":"Object Mystery","type":"ALIEN_CONSTRAINT"}],
                "pose":[{"bone":"Tip","constraints":[
                    {"name":"Pose Mystery","type":"ALIEN_POSE"}
                ]}]
            }]
        }));
        assert_eq!(losses.len(), 2);
        assert!(losses.iter().any(|loss| {
            loss.feature_id == "blender.constraint.alien_constraint"
                && loss.data_id.as_deref() == Some("Object Mystery")
        }));
        assert!(losses.iter().any(|loss| {
            loss.feature_id == "blender.constraint.alien_pose"
                && loss.data_id.as_deref() == Some("Pose Mystery")
        }));
        Ok(())
    }

    #[test]
    fn intermediate_rejects_missing_required_graph_arrays() {
        let error = validate_intermediate(
            &json!({"bridge_version": 1, "blender_version": "5.2.2", "scenes": []}),
        );
        assert!(matches!(
            error,
            Err(PotError {
                code: ErrorCode::ImportFailed,
                ..
            })
        ));
    }
    #[test]
    fn blend_header_rejects_writer_versions_newer_than_profile() -> Result<()> {
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let supported = directory.path().join("supported.blend");
        let newer = directory.path().join("newer.blend");
        std::fs::write(&supported, b"BLENDER-v502").map_err(|error| PotError::io(&error))?;
        std::fs::write(&newer, b"BLENDER-v503").map_err(|error| PotError::io(&error))?;
        check_file_version(&supported)?;
        assert!(matches!(
            check_file_version(&newer),
            Err(PotError {
                code: ErrorCode::BlenderVersionUnsupported,
                ..
            })
        ));
        Ok(())
    }
    #[test]
    fn import_projection_preserves_shared_mesh_parent_material_and_animation() -> Result<()> {
        let raw: Value = serde_json::from_str(r#"{
            "bridge_version": 1, "blender_version": "5.2.2", "file_version": [5, 2, 2],
            "active_scene": "Scene",
            "scenes": [{
                "name": "Scene", "potter_id": "scene_test", "root_collection": "Collection",
                "frame_current": 12.5, "frame_start": 1, "frame_end": 50, "fps": 24, "fps_base": 1.0,
                "camera": null, "world": {"name": "World", "potter_id": "world_test",
                    "color": [0.1,0.2,0.3], "strength": 0.6,
                    "nodes": {"nodes": [{"name":"Background","type":"ShaderNodeBackground"}, {"name":"World Output","type":"ShaderNodeOutputWorld"}]},
                    "custom_properties": {}},
                "unit": {"system": "metric", "scale_length": 1.0},
                "render": {"resolution_x": 640, "resolution_y": 360, "resolution_percentage": 100,
                           "samples": 12, "seed": 5, "film_transparent": false, "engine": "path",
                           "engine_native": "CYCLES"},
                "view_layers": [{"name": "View Layer", "excluded_collections": []}]
            }],
            "collections": [{
                "name": "Collection", "potter_id": "collection_root", "children": [],
                "objects": ["Child", "Parent"]
            }],
            "objects": [
                {"name": "Child", "potter_id": "node_child", "type": "MESH", "data_name": "Shared",
                 "parent": "Parent", "parent_type": "OBJECT", "parent_bone": "",
                 "matrix_parent_inverse": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                 "location": [1,2,3], "rotation_mode": "XYZ", "rotation_euler": [0,0,0],
                 "rotation_quaternion": [1,0,0,0], "rotation_axis_angle": [0,0,0,1],
                 "scale": [1,1,1], "delta_location": [0,0,0], "delta_rotation_euler": [0,0,0],
                 "delta_rotation_quaternion": [1,0,0,0], "delta_scale": [1,1,1],
                 "matrix_basis": [], "matrix_local": [], "matrix_world": [],
                 "hide_viewport": true, "hide_render": false, "hide_select": false,
                 "materials": ["Red"], "action": "Move",
                 "custom_properties": {"potter.tags": ["animated"]}, "rna_properties": {},
                 "modifiers": [{"name": "Bevel", "type": "BEVEL", "enabled": false,
                                "properties": {"width": 0.125}, "custom_properties": {}}],
                 "data_name_unused": null},
                {"name": "Parent", "potter_id": "node_parent", "type": "MESH", "data_name": "Shared",
                 "parent": null, "parent_type": "OBJECT", "parent_bone": "",
                 "matrix_parent_inverse": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                 "location": [0,0,0], "rotation_mode": "QUATERNION", "rotation_euler": [0,0,0],
                 "rotation_quaternion": [0,0,0,1], "rotation_axis_angle": [0,0,0,1],
                 "scale": [1,1,1], "delta_location": [0,0,0], "delta_rotation_euler": [0,0,0],
                 "delta_rotation_quaternion": [1,0,0,0], "delta_scale": [1,1,1],
                 "matrix_basis": [], "matrix_local": [], "matrix_world": [],
                 "hide_viewport": false, "hide_render": false, "hide_select": false,
                 "materials": ["Red"], "action": null, "custom_properties": {}, "rna_properties": {}, "modifiers": []}
            ],
            "meshes": [{
                "name": "Shared", "potter_id": "mesh_shared", "vertices": [
                    {"co": [0,0,0]}, {"co": [1,0,0]}, {"co": [0,1,0]}
                ], "edges": [{"v": [0,1], "sharp": false, "seam": false},
                             {"v": [1,2], "sharp": false, "seam": false},
                             {"v": [2,0], "sharp": false, "seam": false}],
                "polygons": [{"v": [0,1,2], "material_index": 0, "smooth": false}],
                "uv_layers": [], "attributes": [], "custom_properties": {},
                "fake_user": false, "users": 2, "rna_properties": {}
            }],
            "materials": [{
                "name": "Red", "potter_id": "material_red", "base_color": [0.8,0.1,0.2,1],
                "metallic": 0.25, "roughness": 0.4, "double_sided": false,
                "nodes": {"nodes": [{"name":"Principled BSDF","type":"ShaderNodeBsdfPrincipled"}]},
                "custom_properties": {}, "fake_user": false, "users": 2
            }],
            "cameras": [], "lights": [],
            "actions": [{
                "name": "Move", "potter_id": "action_move",
                "fcurves": [{"path": "location", "index": 0,
                             "keyframes": [{"frame": 1, "value": 0, "interpolation": "LINEAR"},
                                           {"frame": 12, "value": 2, "interpolation": "BEZIER"}]}]
            }],
            "texts": [], "node_groups": [], "other_datablocks": [], "unused_datablocks": []
        }"#).map_err(PotError::internal_json)?;
        validate_intermediate(&raw)?;
        let imported = build_imported_graph(
            &raw,
            Path::new("sample.blend"),
            b"ORIGINAL".to_vec(),
            &BTreeMap::new(),
        )?;
        assert_eq!(imported.doc.active_scene.as_str(), "scene_test");
        let scene = imported
            .doc
            .scenes
            .get(&Id::new("scene_test")?)
            .ok_or_else(|| invalid_intermediate("synthetic scene was not imported"))?;
        assert_eq!(scene.world.as_ref().map(Id::as_str), Some("world_test"));
        assert_eq!(scene.render.resolution_x, 640);
        assert_eq!(scene.render.samples, 12);
        let world = imported
            .doc
            .worlds
            .get(&Id::new("world_test")?)
            .ok_or_else(|| invalid_intermediate("synthetic world was not imported"))?;
        for (actual, expected) in world.color.into_iter().zip([0.1, 0.2, 0.3]) {
            assert!(approximately(actual, expected));
        }
        assert!(approximately(world.strength, 0.6));
        assert_eq!(
            test_node_ref(&imported.doc, "node_child")?
                .parent
                .as_ref()
                .map(Id::as_str),
            Some("node_parent")
        );
        assert_eq!(
            test_node_ref(&imported.doc, "node_child")?
                .data
                .as_ref()
                .map(Id::as_str),
            Some("mesh_shared")
        );
        assert_eq!(
            test_node_ref(&imported.doc, "node_parent")?
                .data
                .as_ref()
                .map(Id::as_str),
            Some("mesh_shared")
        );
        assert_eq!(
            test_node_ref(&imported.doc, "node_child")?.materials[0].as_str(),
            "material_red"
        );
        assert!(!test_node_ref(&imported.doc, "node_child")?.modifiers[0].enabled);
        assert_eq!(
            test_node_ref(&imported.doc, "node_child")?
                .action
                .as_ref()
                .map(Id::as_str),
            Some("action_move")
        );
        assert_eq!(
            test_action_ref(&imported.doc, "action_move")?.fcurves[0]
                .keyframes
                .len(),
            2
        );
        assert!(!test_node_ref(&imported.doc, "node_child")?.visible);
        assert_eq!(imported.compat_blobs[0].1, b"ORIGINAL");
        assert_eq!(imported.id_mappings["Object:Child"], "node_child");
        Ok(())
    }

    #[test]
    fn advanced_blender_projection_maps_rig_animation_physics_nodes_and_grease_pencil() -> Result<()>
    {
        let raw = json!({
            "bridge_version":1,"blender_version":"5.2.2","file_version":[5,2,2],"active_scene":"Scene",
            "scenes":[{
                "name":"Scene","potter_id":"scene_main","root_collection":"Collection",
                "frame_current":1.0,"frame_start":1,"frame_end":250,"fps":24,"fps_base":1.0,
                "camera":null,"world":null,"unit":{"system":"METRIC","scale_length":1.0},
                "render":{},"view_layers":[{"name":"View Layer","excluded_collections":[]}],
                "markers":[{"name":"Marker","frame":12.0}],
                "rigid_body_world":{"enabled":true,"gravity":[0.0,0.0,-9.81],
                    "substeps":4,"solver_iterations":10,"frame_start":1,"frame_end":250,"seed":3}
            }],
            "collections":[{"name":"Collection","potter_id":"collection_root","children":[],
                "objects":["Rig","Skinned","Cube","Pencil"]}],
            "objects":[
                {"name":"Rig","potter_id":"node_rig","type":"ARMATURE","data_name":"Armature",
                    "parent":null,"parent_type":"OBJECT","parent_bone":"",
                    "location":[0,0,0],"rotation_mode":"XYZ","rotation_euler":[0,0,0],
                    "rotation_quaternion":[1,0,0,0],"scale":[1,1,1],"materials":[],
                    "modifiers":[],"constraints":[],"drivers":[],"nla_tracks":[],
                    "pose":[{"bone":"Root","location":[0,0,0],"rotation_quaternion":[1,0,0,0],
                        "scale":[1,1,1],"constraints":[]}],"custom_properties":{}},
                {"name":"Skinned","potter_id":"node_skin","type":"MESH","data_name":"Mesh",
                    "parent":"Rig","parent_type":"BONE","parent_bone":"Root",
                    "location":[0,0,0],"rotation_mode":"XYZ","rotation_euler":[0,0,0],
                    "rotation_quaternion":[1,0,0,0],"scale":[1,1,1],"materials":[],
                    "vertex_groups":[{"name":"Weights","potter_id":"group_weights",
                        "weights":[{"vertex_index":0,"weight":1.0}]}],
                    "modifiers":[],"constraints":[{"name":"IK","type":"IK","properties":{"target":{"id_type":"OBJECT","name":"Rig"},"subtarget":"Tip","influence":1.0},"custom_properties":{}}],
                    "drivers":[{"curve":{"path":"key_blocks[\"Smile\"].value","index":0,"keyframes":[],"extrapolation":"CONSTANT"},
                        "driver":{"type":"SCRIPTED","expression":"amount","variables":[{"name":"amount","type":"SINGLE_PROP","targets":[{"id_type":"OBJECT","id":{"name":"Cube"},"data_path":"location.x"}]}]}}],
                    "nla_tracks":[{"name":"WalkTrack","potter_id":"track_walk","mute":false,"solo":false,
                        "strips":[{"name":"WalkStrip","potter_id":"strip_walk","action":"Move",
                            "properties":{"frame_start":1.0,"frame_end":12.0,"action_frame_start":1.0,
                                "action_frame_end":12.0,"scale":1.0,"repeat":1.0,"blend_type":"REPLACE",
                                "influence":1.0,"extrapolation":"HOLD","blend_in":0.0,"blend_out":0.0},
                            "custom_properties":{}}]}],
                    "action":"Move","custom_properties":{}},
                {"name":"Cube","potter_id":"node_cube","type":"MESH","data_name":"Mesh",
                    "parent":null,"parent_type":"OBJECT","parent_bone":"",
                    "location":[0,0,0],"rotation_mode":"XYZ","rotation_euler":[0,0,0],
                    "rotation_quaternion":[1,0,0,0],"scale":[1,1,1],"materials":[],
                    "modifiers":[{"name":"GeometryNodes","potter_id":"mod_geo","type":"NODES",
                        "enabled":true,"properties":{"node_group":"Geometry","inputs":{}},
                        "custom_properties":{}}],
                    "constraints":[],"drivers":[],"nla_tracks":[],"action":null,
                    "rigid_body":{"type":"ACTIVE","mass":2.0,"friction":0.5,"restitution":0.1,
                        "shape":"BOX","linear_damping":0.04,"angular_damping":0.1,
                        "initial_velocity":[0,0,0]},"force_field":null,"custom_properties":{}},
                {"name":"Pencil","potter_id":"node_pencil","type":"GREASE_PENCIL","data_name":"PencilData",
                    "parent":null,"parent_type":"OBJECT","parent_bone":"",
                    "location":[0,0,0],"rotation_mode":"XYZ","rotation_euler":[0,0,0],
                    "rotation_quaternion":[1,0,0,0],"scale":[1,1,1],"materials":[],
                    "modifiers":[],"constraints":[],"drivers":[],"nla_tracks":[],"action":null,
                    "custom_properties":{}}
            ],
            "meshes":[{"name":"Mesh","potter_id":"mesh_shared",
                "vertices":[{"co":[0,0,0]},{"co":[1,0,0]},{"co":[0,1,0]}],
                "edges":[{"v":[0,1]},{"v":[1,2]},{"v":[2,0]}],
                "polygons":[{"v":[0,1,2],"material_index":0,"smooth":false}],
                "uv_layers":[],"attributes":[],
                "shape_keys":{"basis":[{"index":0,"co":[0,0,0]},{"index":1,"co":[1,0,0]},{"index":2,"co":[0,1,0]}],
                    "keys":[{"name":"Smile","potter_id":"shape_smile","value":0.25,
                        "slider_min":0.0,"slider_max":1.0,"relative_key":"Basis",
                        "vertex_group":"Weights","positions":[{"index":1,"co":[1,0,0.25]}]}]},
                "custom_properties":{},"fake_user":false,"users":2}],
            "armatures":[{"name":"Armature","potter_id":"armature_rig","bones":[
                {"name":"Root","potter_id":"bone_root","parent":null,"head":[0,0,0],"tail":[0,0,1],
                    "roll":0.0,"deform":true,"inherit_rotation":true,"use_connect":false},
                {"name":"Tip","potter_id":"bone_tip","parent":"Root","head":[0,0,1],"tail":[0,0,2],
                    "roll":0.0,"deform":true,"inherit_rotation":true,"use_connect":true}],
                "custom_properties":{},"fake_user":false}],
            "grease_pencils":[{"name":"PencilData","potter_id":"gp_data","layers":[
                {"name":"Ink","potter_id":"gp_layer","opacity":1.0,"visible":true,"frames":[
                    {"frame":1.0,"strokes":[{"name":"Stroke","potter_id":"gp_stroke",
                        "points":[{"position":[0,0,0],"pressure":1.0,"radius":0.05,"opacity":1.0,"time":0.0},
                                  {"position":[1,0,0],"pressure":0.75,"radius":0.04,"opacity":0.9,"time":0.1}],
                        "cyclic":false,"fill":null,"material":null}]}]}]}],
            "materials":[],"cameras":[],"lights":[],
            "actions":[{"name":"Move","potter_id":"action_move","slot_count":1,
                "slots":[{"name":"ObjectSlot","potter_id":"slot_move","node":"Skinned"}],
                "fcurves":[{"path":"location","index":0,"extrapolation":"CONSTANT",
                    "keyframes":[{"frame":1,"value":0,"interpolation":"LINEAR"}]}]}],
            "node_groups":[{"name":"Geometry","potter_id":"group_geo","type":"GeometryNodeTree",
                "tree":{"type":"GeometryNodeTree",
                    "interface":{"inputs":[],"outputs":[{"id":"Geometry","name":"Geometry",
                        "socket_type":"geometry","default":null}]},
                    "nodes":[{"name":"Transform","type":"GeometryNodeTransform","location":[0,0],
                        "inputs":{"Scale":[2,2,2]},"properties":{}},
                             {"name":"Output","type":"NodeGroupOutput","location":[300,0],
                        "inputs":{},"properties":{}}],
                    "links":[{"from_node":"Transform","from_socket":"Geometry",
                        "to_node":"Output","to_socket":"Geometry"}]},
                "custom_properties":{},"fake_user":false}],
            "texts":[],"other_datablocks":[],"unused_datablocks":[]
        });
        validate_intermediate(&raw)?;
        let imported = build_imported_graph(
            &raw,
            Path::new("advanced.blend"),
            b"ORIGINAL".to_vec(),
            &BTreeMap::new(),
        )?;
        assert!(
            imported.losses.is_empty(),
            "unexpected losses: {:?}",
            imported.losses
        );
        let scene = imported
            .doc
            .scenes
            .get(&Id::new("scene_main")?)
            .ok_or_else(|| invalid_intermediate("advanced scene was not imported"))?;
        assert_eq!(scene.markers.len(), 1);
        assert_eq!(
            scene.rigid_body_world.as_ref().map(|world| world.seed),
            Some(3)
        );
        let rig = test_node_ref(&imported.doc, "node_rig")?;
        let armature_id = rig
            .data
            .as_ref()
            .ok_or_else(|| invalid_intermediate("armature reference missing"))?;
        assert_eq!(
            imported
                .doc
                .data_blocks
                .get(armature_id)
                .and_then(|data| data.armature.as_ref())
                .map(|data| data.bones.len()),
            Some(2)
        );
        let skin = test_node_ref(&imported.doc, "node_skin")?;
        assert_eq!(skin.parent_type, ParentType::Bone);
        assert_eq!(rig.pose.len(), 1);
        assert_eq!(skin.nla_tracks.len(), 1);
        assert_eq!(skin.drivers.len(), 1);
        assert_eq!(skin.constraints.len(), 1);
        let mesh_id = skin
            .data
            .as_ref()
            .ok_or_else(|| invalid_intermediate("skin mesh reference missing"))?;
        let mesh_data = imported
            .doc
            .data_blocks
            .get(mesh_id)
            .ok_or_else(|| invalid_intermediate("skin data missing"))?;
        assert_eq!(mesh_data.vertex_groups.len(), 1);
        assert_eq!(
            mesh_data.shape_keys.as_ref().map(|shape| shape.keys.len()),
            Some(1)
        );
        assert_eq!(
            mesh_data
                .vertex_weights
                .get(&1)
                .map(std::collections::BTreeMap::len),
            Some(1)
        );
        let cube = test_node_ref(&imported.doc, "node_cube")?;
        assert!(cube.rigid_body.is_some());
        assert_eq!(cube.modifiers[0].modifier_type, "nodes");
        assert_eq!(cube.modifiers[0].params["node_group"], "group_geo");
        let pencil = test_node_ref(&imported.doc, "node_pencil")?;
        let pencil_data = imported
            .doc
            .data_blocks
            .get(
                pencil
                    .data
                    .as_ref()
                    .ok_or_else(|| invalid_intermediate("Grease Pencil data reference missing"))?,
            )
            .ok_or_else(|| invalid_intermediate("Grease Pencil data missing"))?;
        assert_eq!(
            pencil_data
                .grease_pencil
                .as_ref()
                .and_then(|data| data.layers.first())
                .and_then(|layer| layer.frames.first())
                .map(|frame| frame.strokes.len()),
            Some(1)
        );
        let action = imported
            .doc
            .actions
            .get(&Id::new("action_move")?)
            .ok_or_else(|| invalid_intermediate("action was not imported"))?;
        assert_eq!(action.slots.len(), 1);
        assert_eq!(imported.doc.node_groups.len(), 1);
        Ok(())
    }

    fn test_node_ref<'a>(doc: &'a SceneDoc, id: &str) -> Result<&'a Node> {
        doc.nodes
            .get(&Id::new(id)?)
            .ok_or_else(|| invalid_intermediate("test node was not imported"))
    }
    fn approximately(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() <= 1.0e-6
    }
    fn approximately_value(value: &Value, expected: f64) -> bool {
        value
            .as_f64()
            .is_some_and(|actual| approximately(actual, expected))
    }

    fn test_data_ref<'a>(doc: &'a SceneDoc, id: &str) -> Result<&'a DataBlock> {
        doc.data_blocks
            .get(&Id::new(id)?)
            .ok_or_else(|| invalid_intermediate("test data-block was not imported"))
    }

    fn test_action_ref<'a>(doc: &'a SceneDoc, id: &str) -> Result<&'a Action> {
        doc.actions
            .get(&Id::new(id)?)
            .ok_or_else(|| invalid_intermediate("test action was not imported"))
    }
    fn optional_blender() -> Option<PathBuf> {
        match resolve_blender(None) {
            Ok(blender) => Some(blender),
            Err(error) => {
                eprintln!("skipping Blender integration test: {error}");
                None
            }
        }
    }

    fn test_node(
        name: &str,
        data: &str,
        parent: Option<&str>,
        action: Option<&str>,
    ) -> Result<Node> {
        Ok(Node {
            name: name.to_owned(),
            kind: "mesh".to_owned(),
            primitive: None,
            tags: Vec::new(),
            parent: parent.map(Id::new).transpose()?,
            parent_inverse: None,
            transform: Transform::default(),
            data: Some(Id::new(data)?),
            materials: vec![Id::new("mat_red")?],
            modifiers: Vec::new(),
            visible: true,
            render_visible: true,
            selectable: true,
            action: action.map(Id::new).transpose()?,
            nla_tracks: Vec::new(),
            properties: Map::new(),
            rigid_body: None,
            force_field: None,
            ..Node::default()
        })
    }

    fn roundtrip_scene() -> Result<SceneDoc> {
        let mut doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        doc.worlds.insert(
            Id::new("world_studio")?,
            World {
                color: [0.15, 0.25, 0.35],
                strength: 0.75,
                ..World::default()
            },
        );
        let scene = doc
            .scenes
            .get_mut(&Id::new("scene_main")?)
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "default scene is missing"))?;
        scene.world = Some(Id::new("world_studio")?);
        scene.render.resolution_x = 640;
        scene.render.resolution_y = 360;
        scene.render.samples = 8;
        scene.render.seed = 7;
        let box_mesh = crate::geom::primitive("box", &json!({"size": 2.0}))
            .map_err(|error| PotError::invalid_argument(error.to_string()))?;
        let sphere_mesh = crate::geom::primitive(
            "sphere",
            &json!({"segments": 12, "ring_count": 8, "radius": 0.75}),
        )
        .map_err(|error| PotError::invalid_argument(error.to_string()))?;
        doc.data_blocks.insert(
            Id::new("mesh_box")?,
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(box_mesh),
                camera: None,
                light: None,
                grease_pencil: None,
                ..DataBlock::default()
            },
        );
        doc.data_blocks.insert(
            Id::new("mesh_sphere")?,
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(sphere_mesh),
                camera: None,
                light: None,
                grease_pencil: None,
                ..DataBlock::default()
            },
        );
        doc.materials.insert(
            Id::new("mat_red")?,
            Material {
                name: "Red".to_owned(),
                base_color: [0.8, 0.1, 0.2, 1.0],
                metallic: 0.25,
                roughness: 0.4,
                emission_color: [0.0; 3],
                emission_strength: 0.0,
                transmission: 0.0,
                ior: 1.45,
                double_sided: false,
                ..Material::default()
            },
        );
        doc.actions.insert(
            Id::new("act_move")?,
            Action {
                name: "Move".to_owned(),
                fcurves: vec![FCurve {
                    path: "transform.translation".to_owned(),
                    index: 0,
                    keyframes: vec![
                        Keyframe {
                            frame: 1.0,
                            value: 0.0,
                            interpolation: Interpolation::Linear,
                            ..Keyframe::default()
                        },
                        Keyframe {
                            frame: 12.0,
                            value: 2.0,
                            interpolation: Interpolation::Bezier,
                            ..Keyframe::default()
                        },
                    ],
                    extrapolation: Extrapolation::Constant,
                }],
                slots: Vec::new(),
            },
        );
        doc.nodes.insert(
            Id::new("box_parent")?,
            test_node("Box Parent", "mesh_box", None, None)?,
        );
        let mut child = test_node(
            "Box Child",
            "mesh_box",
            Some("box_parent"),
            Some("act_move"),
        )?;
        child.transform.translation = [1.0, 0.0, 0.0];
        doc.nodes.insert(Id::new("box_child")?, child);
        doc.nodes.insert(
            Id::new("sphere")?,
            test_node("Sphere", "mesh_sphere", None, None)?,
        );
        doc.collections
            .get_mut(&Id::new("collection_root")?)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "default root collection is missing",
                )
            })?
            .objects = vec![
            Id::new("box_parent")?,
            Id::new("box_child")?,
            Id::new("sphere")?,
        ];
        Ok(doc)
    }

    fn shader_roundtrip_scene() -> Result<SceneDoc> {
        let initial = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let applied = crate::ops::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [
                    {
                        "op": "material.create", "id": "simple", "name": "Simple PBR",
                        "base_color": [0.25, 0.5, 0.75, 1.0], "metallic": 0.5,
                        "roughness": 0.25, "emission_color": [0.125, 0.25, 0.375],
                        "emission_strength": 0.5, "transmission": 0.25, "ior": 1.5
                    },
                    {
                        "op": "material.create", "id": "inline", "name": "Inline PBR",
                        "base_color": [0.125, 0.25, 0.5, 1.0], "metallic": 0.75,
                        "roughness": 0.375, "emission_color": [0.25, 0.125, 0.5],
                        "emission_strength": 0.25, "transmission": 0.5, "ior": 1.25,
                        "graph": {
                            "name": "Inline shader",
                            "kind": "shader",
                            "nodes": {
                                "surface": {
                                    "name": "Principled",
                                    "type": "ShaderNodeBsdfPrincipled",
                                    "inputs": {
                                        "Base Color": [0.125, 0.25, 0.5, 1.0],
                                        "Metallic": 0.75,
                                        "Roughness": 0.375,
                                        "Emission Color": [0.25, 0.125, 0.5, 1.0],
                                        "Emission Strength": 0.25,
                                        "Transmission Weight": 0.5,
                                        "IOR": 1.25
                                    }
                                },
                                "output": {"name": "Output", "type": "OutputMaterial"}
                            },
                            "links": [{
                                "from_node": "surface", "from_socket": "BSDF",
                                "to_node": "output", "to_socket": "Surface"
                            }]
                        }
                    },
                    {
                        "op": "node.create", "id": "simple_body", "kind": "box",
                        "params": {"size": 1.0}, "material": "simple"
                    },
                    {
                        "op": "node.create", "id": "inline_body", "kind": "box",
                        "params": {"size": 1.0}, "material": "inline"
                    }
                ]
            }),
        )?;
        Ok(applied.doc)
    }

    #[test]
    fn blender_roundtrip_preserves_simple_and_inline_principled_materials() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let project = Project::init(directory.path().join("project"))?;
        let doc = shader_roundtrip_scene()?;
        let output = directory.path().join("shader-roundtrip.blend");
        let context = crate::eval::EvaluationContext::default();
        export_blend(
            &doc,
            &project,
            &output,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        )?;
        let imported = import_blend(&output, Some(&blender))?;
        assert!(
            imported.losses.is_empty(),
            "shader round-trip reported losses: {:?}",
            imported.losses
        );
        for (
            id,
            expected_color,
            metallic,
            roughness,
            emission,
            emission_strength,
            transmission,
            ior,
        ) in [
            (
                "simple",
                [0.25, 0.5, 0.75, 1.0],
                0.5,
                0.25,
                [0.125, 0.25, 0.375],
                0.5,
                0.25,
                1.5,
            ),
            (
                "inline",
                [0.125, 0.25, 0.5, 1.0],
                0.75,
                0.375,
                [0.25, 0.125, 0.5],
                0.25,
                0.5,
                1.25,
            ),
        ] {
            let id = Id::new(id)?;
            let expected = doc
                .materials
                .get(&id)
                .ok_or_else(|| invalid_intermediate("source shader material is missing"))?;
            let actual = imported
                .doc
                .materials
                .get(&id)
                .ok_or_else(|| invalid_intermediate("round-trip shader material is missing"))?;
            assert_eq!(actual.base_color, expected_color);
            assert_eq!(actual.base_color, expected.base_color);
            assert_eq!(actual.metallic, metallic);
            assert_eq!(actual.metallic, expected.metallic);
            assert_eq!(actual.roughness, roughness);
            assert_eq!(actual.roughness, expected.roughness);
            assert_eq!(actual.emission_color, emission);
            assert_eq!(actual.emission_color, expected.emission_color);
            assert_eq!(actual.emission_strength, emission_strength);
            assert_eq!(actual.emission_strength, expected.emission_strength);
            assert_eq!(actual.transmission, transmission);
            assert_eq!(actual.transmission, expected.transmission);
            assert_eq!(actual.ior, ior);
            assert_eq!(actual.ior, expected.ior);
            assert_eq!(actual.node_tree, expected.node_tree);
            let graph_id = actual
                .node_tree
                .as_ref()
                .ok_or_else(|| invalid_intermediate("round-trip material graph is missing"))?;
            let graph = imported
                .doc
                .node_groups
                .get(graph_id)
                .ok_or_else(|| invalid_intermediate("round-trip shader graph is missing"))?;
            assert_eq!(graph.kind, GraphKind::Shader);
            assert!(
                graph
                    .nodes
                    .values()
                    .any(|node| node.node_type == "OutputMaterial")
            );
        }
        Ok(())
    }

    #[test]
    fn blender_roundtrip_keeps_ids_shared_mesh_material_parent_and_keyframes() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let project = Project::init(directory.path().join("project"))?;
        let doc = roundtrip_scene()?;
        let output = directory.path().join("roundtrip.blend");
        let context = crate::eval::EvaluationContext::default();
        let options = ExportOptions {
            allow_lossy: false,
            pack: false,
            blender: Some(&blender),
            context: &context,
        };
        export_blend(&doc, &project, &output, &options)?;
        let imported = import_blend(&output, Some(&blender))?;
        assert_eq!(
            test_node_ref(&imported.doc, "box_parent")?.name,
            "Box Parent"
        );
        assert_eq!(test_node_ref(&imported.doc, "box_child")?.name, "Box Child");
        assert_eq!(test_node_ref(&imported.doc, "sphere")?.name, "Sphere");
        assert_eq!(
            test_node_ref(&imported.doc, "box_child")?
                .parent
                .as_ref()
                .map(Id::as_str),
            Some("box_parent")
        );
        assert_eq!(
            test_node_ref(&imported.doc, "box_parent")?.data,
            test_node_ref(&imported.doc, "box_child")?.data
        );
        assert_eq!(
            test_node_ref(&imported.doc, "box_child")?.materials[0].as_str(),
            "mat_red"
        );
        assert_eq!(
            test_action_ref(&imported.doc, "act_move")?.fcurves[0]
                .keyframes
                .len(),
            2
        );
        let imported_scene = imported
            .doc
            .scenes
            .get(&Id::new("scene_main")?)
            .ok_or_else(|| invalid_intermediate("round-trip scene was not imported"))?;
        assert_eq!(
            imported_scene.world.as_ref().map(Id::as_str),
            Some("world_studio")
        );
        assert_eq!(imported_scene.render.resolution_x, 640);
        assert_eq!(imported_scene.render.resolution_y, 360);
        assert_eq!(imported_scene.render.samples, 8);
        assert_eq!(imported_scene.render.seed, 7);
        let world = imported
            .doc
            .worlds
            .get(&Id::new("world_studio")?)
            .ok_or_else(|| invalid_intermediate("round-trip world was not imported"))?;
        for (actual, expected) in world.color.into_iter().zip([0.15, 0.25, 0.35]) {
            assert!(approximately(actual, expected));
        }
        assert!(approximately(world.strength, 0.75));
        let original = test_data_ref(&doc, "mesh_sphere")?
            .mesh
            .as_ref()
            .and_then(Mesh::bounds)
            .ok_or_else(|| invalid_intermediate("source sphere bounds are missing"))?;
        let imported_mesh = test_node_ref(&imported.doc, "sphere")?
            .data
            .as_ref()
            .and_then(|data_id| imported.doc.data_blocks.get(data_id))
            .and_then(|data| data.mesh.as_ref())
            .and_then(Mesh::bounds)
            .ok_or_else(|| invalid_intermediate("imported sphere bounds are missing"))?;
        assert!((original.min - imported_mesh.min).abs().max_element() <= 1.0e-6);
        assert!((original.max - imported_mesh.max).abs().max_element() <= 1.0e-6);
        Ok(())
    }

    #[test]
    fn blender_table_fixture_import_export_preserves_supported_state() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let script_path = directory.path().join("make_table_fixture.py");
        let blend_path = directory.path().join("table-fixture.blend");
        fs::write(
            &script_path,
            r#"
import bpy, sys
path = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
bpy.ops.mesh.primitive_cube_add()
top = bpy.context.object
top.name = "Tabletop"
top.data.name = "TabletopMesh"
top.scale = (2.0, 2.0, 0.25)
top.data.materials.clear()
material = bpy.data.materials.new("Walnut brown")
material.use_nodes = True
principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
principled.inputs["Base Color"].default_value = (0.25, 0.125, 0.0625, 1.0)
principled.inputs["Metallic"].default_value = 0.125
principled.inputs["Roughness"].default_value = 0.375
top.data.materials.append(material)
bevel = top.modifiers.new("Bevel", "BEVEL")
bevel.width = 0.05
top.location.x = 0.0
top.keyframe_insert(data_path="location", frame=1)
top.location.x = 1.0
top.keyframe_insert(data_path="location", frame=24)

leg_mesh = bpy.data.meshes.new("LegMesh")
leg_mesh.from_pydata(
    [(0,0,0), (0.25,0,0), (0.25,0.25,0), (0,0.25,0),
     (0,0,1), (0.25,0,1), (0.25,0.25,1), (0,0.25,1)],
    [],
    [(0,1,2,3), (4,7,6,5), (0,4,5,1), (1,5,6,2), (2,6,7,3), (3,7,4,0)])
leg_mesh.materials.append(material)
for name, location in (
    ("Leg_BL", (-1.5, -1.5, -1.25)), ("Leg_BR", (1.25, -1.5, -1.25)),
    ("Leg_FL", (-1.5, 1.25, -1.25)), ("Leg_FR", (1.25, 1.25, -1.25))):
    leg = bpy.data.objects.new(name, leg_mesh)
    scene.collection.objects.link(leg)
    leg.location = location

hidden = bpy.data.objects.new("Hidden_Helper", None)
scene.collection.objects.link(hidden)
hidden.hide_viewport = True
hidden.hide_render = True

orphan = bpy.data.meshes.new("Unused_FakeUser_Mesh")
orphan.from_pydata([(0,0,0), (1,0,0), (0,1,0)], [], [(0,1,2)])
orphan.use_fake_user = True
bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
"#,
        )
        .map_err(|error| PotError::io(&error))?;
        let generated = Command::new(&blender)
            .arg("--background")
            .arg("--factory-startup")
            .arg("--disable-depsgraph-on-file-load")
            .arg("--disable-autoexec")
            .arg("--python-exit-code")
            .arg("3")
            .arg("--python")
            .arg(&script_path)
            .arg("--")
            .arg(&blend_path)
            .output()
            .map_err(|error| PotError::io(&error))?;
        if !generated.status.success() {
            return Err(PotError::with_details(
                ErrorCode::ImportFailed,
                "Blender tabletop fixture creation failed",
                child_details(&generated),
            ));
        }
        let imported = import_blend(&blend_path, Some(&blender))?;
        assert!(
            imported.losses.is_empty(),
            "table fixture import reported losses: {:?}",
            imported.losses
        );
        let hidden_id = imported.id_mappings["Object:Hidden_Helper"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("hidden helper ID mapping is missing"))?;
        let hidden = test_node_ref(&imported.doc, hidden_id)?;
        assert!(!hidden.visible);
        assert!(!hidden.render_visible);
        let tabletop_id = imported.id_mappings["Object:Tabletop"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("tabletop ID mapping is missing"))?;
        let tabletop = test_node_ref(&imported.doc, tabletop_id)?;
        let action_id = tabletop
            .action
            .as_ref()
            .ok_or_else(|| invalid_intermediate("tabletop action was not imported"))?;
        assert_eq!(
            test_action_ref(&imported.doc, action_id.as_str())?.fcurves[0].path,
            "transform.translation"
        );
        let leg_ids = ["Leg_BL", "Leg_BR", "Leg_FL", "Leg_FR"]
            .map(|name| imported.id_mappings[&format!("Object:{name}")].as_str());
        let first_leg_id =
            leg_ids[0].ok_or_else(|| invalid_intermediate("first leg ID mapping is missing"))?;
        let first_leg = test_node_ref(&imported.doc, first_leg_id)?;
        let shared_mesh_id = first_leg
            .data
            .as_ref()
            .ok_or_else(|| invalid_intermediate("shared leg mesh reference is missing"))?;
        assert_eq!(first_leg.materials.len(), 1);
        for leg_id in leg_ids.into_iter().skip(1) {
            let leg_id = leg_id.ok_or_else(|| invalid_intermediate("leg ID mapping is missing"))?;
            let leg = test_node_ref(&imported.doc, leg_id)?;
            assert_eq!(leg.data.as_ref(), Some(shared_mesh_id));
            assert_eq!(leg.materials.len(), 1);
        }
        let orphan_id = imported.id_mappings["Mesh:Unused_FakeUser_Mesh"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("fake-user mesh ID mapping is missing"))?;
        assert_eq!(
            test_data_ref(&imported.doc, orphan_id)?
                .mesh
                .as_ref()
                .and_then(|mesh| mesh.attributes.get("blender_metadata"))
                .and_then(|metadata| metadata.get("fake_user"))
                .and_then(Value::as_bool),
            Some(true)
        );

        let project = Project::init(directory.path().join("project"))?;
        let output = directory.path().join("table-roundtrip.blend");
        let context = crate::eval::EvaluationContext::default();
        export_blend(
            &imported.doc,
            &project,
            &output,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        )?;
        let roundtrip = import_blend(&output, Some(&blender))?;
        assert!(
            roundtrip.losses.is_empty(),
            "table fixture round-trip reported losses: {:?}",
            roundtrip.losses
        );
        for leg_name in ["Leg_BL", "Leg_BR", "Leg_FL", "Leg_FR"] {
            let leg_id = roundtrip.id_mappings[&format!("Object:{leg_name}")]
                .as_str()
                .ok_or_else(|| invalid_intermediate("round-trip leg ID mapping is missing"))?;
            let leg = test_node_ref(&roundtrip.doc, leg_id)?;
            assert_eq!(leg.materials.len(), 1);
            assert_eq!(
                leg.data.as_ref().map(Id::as_str),
                Some(shared_mesh_id.as_str())
            );
        }
        let hidden_id = roundtrip.id_mappings["Object:Hidden_Helper"]
            .as_str()
            .ok_or_else(|| {
                invalid_intermediate("round-trip hidden helper ID mapping is missing")
            })?;
        assert!(!test_node_ref(&roundtrip.doc, hidden_id)?.visible);
        let orphan_id = roundtrip.id_mappings["Mesh:Unused_FakeUser_Mesh"]
            .as_str()
            .ok_or_else(|| {
                invalid_intermediate("round-trip fake-user mesh ID mapping is missing")
            })?;
        assert_eq!(
            test_data_ref(&roundtrip.doc, orphan_id)?
                .mesh
                .as_ref()
                .and_then(|mesh| mesh.attributes.get("blender_metadata"))
                .and_then(|metadata| metadata.get("fake_user"))
                .and_then(Value::as_bool),
            Some(true)
        );
        Ok(())
    }
    #[test]
    fn blender_default_startup_scene_imports_without_losses() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let script_path = directory.path().join("save_default.py");
        let blend_path = directory.path().join("default.blend");
        fs::write(
            &script_path,
            r#"
import bpy, sys
path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
"#,
        )
        .map_err(|error| PotError::io(&error))?;
        let generated = Command::new(&blender)
            .arg("--background")
            .arg("--factory-startup")
            .arg("--disable-depsgraph-on-file-load")
            .arg("--disable-autoexec")
            .arg("--python-exit-code")
            .arg("3")
            .arg("--python")
            .arg(&script_path)
            .arg("--")
            .arg(&blend_path)
            .output()
            .map_err(|error| PotError::io(&error))?;
        if !generated.status.success() {
            return Err(PotError::with_details(
                ErrorCode::ImportFailed,
                "Blender default scene creation failed",
                child_details(&generated),
            ));
        }
        let imported = import_blend(&blend_path, Some(&blender))?;
        assert!(
            imported.losses.is_empty(),
            "default Blender scene import reported losses: {:?}",
            imported.losses
        );
        Ok(())
    }

    #[test]
    fn blender_import_keeps_modifier_hidden_custom_property_and_orphan_mesh() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let script_path = directory.path().join("make_fixture.py");
        let blend_path = directory.path().join("fixture.blend");
        fs::write(
            &script_path,
            r#"
import bpy, sys
path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.object
obj.name = "FixtureCube"
obj.hide_viewport = True
obj.hide_render = True
obj["fixture.note"] = "kept"
parent = bpy.data.objects.new("FixtureParent", None)
bpy.context.scene.collection.objects.link(parent)
obj.parent = parent
mesh_attribute = obj.data.attributes.new(name="custom_weight", type="FLOAT", domain="POINT")
for index, item in enumerate(mesh_attribute.data):
    item.value = float(index) * 0.25
material = bpy.data.materials.new("FixtureMaterial")
material.use_nodes = True
shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
shader.inputs["Base Color"].default_value = (0.8, 0.15, 0.25, 1.0)
shader.inputs["Metallic"].default_value = 0.2
shader.inputs["Roughness"].default_value = 0.35
shader.inputs["Emission Color"].default_value = (0.3, 0.2, 0.1, 1.0)
shader.inputs["Emission Strength"].default_value = 0.6
shader.inputs["Transmission Weight"].default_value = 0.35
shader.inputs["IOR"].default_value = 1.38
obj.data.materials.append(material)
mirror = obj.modifiers.new(name="FixtureMirror", type="MIRROR")
mirror.use_axis = (True, False, False)
mirror.use_mirror_merge = True
mirror.merge_threshold = 0.002
array = obj.modifiers.new(name="FixtureArray", type="ARRAY")
array.count = 3
array.relative_offset_displace = (0.0, 0.0, 0.0)
array.constant_offset_displace = (1.0, 0.0, 0.0)
subdivision = obj.modifiers.new(name="FixtureSubdivision", type="SUBSURF")
subdivision.levels = 2
solidify = obj.modifiers.new(name="FixtureSolidify", type="SOLIDIFY")
solidify.thickness = 0.2
obj.modifiers.new(name="FixtureTriangulate", type="TRIANGULATE")
bevel = obj.modifiers.new(name="FixtureBevel", type="BEVEL")
bevel.width = 0.125
decimate = obj.modifiers.new(name="FixtureDecimate", type="DECIMATE")
decimate.ratio = 0.75
weld = obj.modifiers.new(name="FixtureWeld", type="WELD")
weld.merge_threshold = 0.003
displace = obj.modifiers.new(name="FixtureDisplace", type="DISPLACE")
displace.strength = 0.4
smooth = obj.modifiers.new(name="FixtureSmooth", type="SMOOTH")
smooth.factor = 0.25
smooth.iterations = 2
remesh = obj.modifiers.new(name="FixtureRemesh", type="REMESH")
remesh.mode = "VOXEL"
remesh.voxel_size = 0.1
obj.location.x = 0.0
obj.keyframe_insert(data_path="location", frame=1)
obj.location.x = 2.0
obj.keyframe_insert(data_path="location", frame=12)
text = bpy.data.texts.new("FixtureText")
text.write("No-op text payload\n")
text["fixture.tag"] = "retained"
text.use_fake_user = True
orphan = bpy.data.meshes.new("OrphanMesh")
orphan.from_pydata([(0,0,0), (1,0,0), (0,1,0)], [], [(0,1,2)])
orphan.use_fake_user = True
bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
"#,
        )
        .map_err(|error| PotError::io(&error))?;
        let output = Command::new(&blender)
            .arg("--background")
            .arg("--factory-startup")
            .arg("--disable-autoexec")
            .arg("--python-exit-code")
            .arg("3")
            .arg("--python")
            .arg(&script_path)
            .arg("--")
            .arg(&blend_path)
            .output()
            .map_err(|error| PotError::io(&error))?;
        if !output.status.success() {
            return Err(PotError::with_details(
                ErrorCode::ImportFailed,
                "Blender fixture creation failed",
                child_details(&output),
            ));
        }
        let imported = import_blend(&blend_path, Some(&blender))?;
        let cube_id = imported.id_mappings["Object:FixtureCube"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("fixture object ID mapping is missing"))?;
        let cube = imported
            .doc
            .nodes
            .get(&Id::new(cube_id)?)
            .ok_or_else(|| invalid_intermediate("fixture cube was not imported"))?;
        assert!(!cube.visible);
        assert!(!cube.render_visible);
        assert_eq!(
            cube.properties.get("fixture.note").and_then(Value::as_str),
            Some("kept")
        );
        let parent_id = imported.id_mappings["Object:FixtureParent"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("fixture parent ID mapping is missing"))?;
        assert_eq!(cube.kind, "mesh");
        assert_eq!(cube.parent.as_ref().map(Id::as_str), Some(parent_id));
        let parent = test_node_ref(&imported.doc, parent_id)?;
        assert_eq!(parent.name, "FixtureParent");
        assert_eq!(parent.kind, "empty");
        let modifier_types = cube
            .modifiers
            .iter()
            .map(|modifier| modifier.modifier_type.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            modifier_types,
            vec![
                "mirror",
                "array",
                "subdivision",
                "solidify",
                "triangulate",
                "bevel",
                "decimate",
                "weld",
                "displace",
                "smooth",
                "remesh",
            ]
        );
        assert_eq!(
            cube.modifiers[0].params["use_axis"],
            json!([true, false, false])
        );
        assert_eq!(cube.modifiers[1].params["count"], 3);
        assert_eq!(cube.modifiers[2].params["levels"], 2);
        assert!(approximately_value(
            &cube.modifiers[3].params["thickness"],
            0.2
        ));
        assert_eq!(cube.modifiers[5].params["width"], 0.125);
        assert_eq!(cube.modifiers[6].params["ratio"], 0.75);
        assert!(approximately_value(
            &cube.modifiers[7].params["merge_threshold"],
            0.003
        ));
        assert!(approximately_value(
            &cube.modifiers[8].params["strength"],
            0.4
        ));
        assert_eq!(cube.modifiers[9].params["factor"], 0.25);
        assert_eq!(cube.modifiers[10].params["mode"], "VOXEL");
        assert!(
            cube.modifiers[0].params["use_mirror_merge"]
                .as_bool()
                .unwrap_or(false)
        );
        assert!(approximately_value(
            &cube.modifiers[0].params["merge_threshold"],
            0.002
        ));
        assert_eq!(
            cube.modifiers[1].params["relative_offset_displace"],
            json!([0.0, 0.0, 0.0])
        );
        assert_eq!(
            cube.modifiers[1].params["constant_offset_displace"],
            json!([1.0, 0.0, 0.0])
        );
        assert_eq!(cube.modifiers[9].params["iterations"], 2);
        assert!(approximately_value(
            &cube.modifiers[10].params["voxel_size"],
            0.1
        ));
        let material_id = cube
            .materials
            .first()
            .ok_or_else(|| invalid_intermediate("fixture material slot was not imported"))?;
        let material = imported
            .doc
            .materials
            .get(material_id)
            .ok_or_else(|| invalid_intermediate("fixture material was not imported"))?;
        for (actual, expected) in material.base_color.into_iter().zip([0.8, 0.15, 0.25, 1.0]) {
            assert!(approximately(actual, expected));
        }
        let action_id = cube
            .action
            .as_ref()
            .ok_or_else(|| invalid_intermediate("fixture action was not imported"))?;
        assert_eq!(
            imported
                .doc
                .actions
                .get(action_id)
                .ok_or_else(|| invalid_intermediate("fixture action was missing"))?
                .fcurves
                .iter()
                .find(|curve| curve.path == "transform.translation")
                .map(|curve| curve.keyframes.len()),
            Some(2)
        );
        let mesh_id = cube
            .data
            .as_ref()
            .ok_or_else(|| invalid_intermediate("fixture mesh reference was missing"))?;
        let mesh = imported
            .doc
            .data_blocks
            .get(mesh_id)
            .and_then(|data| data.mesh.as_ref())
            .ok_or_else(|| invalid_intermediate("fixture mesh data was missing"))?;
        let attributes = mesh
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_intermediate("custom mesh attributes were not imported"))?;
        assert!(attributes.iter().any(
            |attribute| attribute.get("name").and_then(Value::as_str) == Some("custom_weight")
        ));
        let custom_attribute = attributes
            .iter()
            .find(|attribute| {
                attribute.get("name").and_then(Value::as_str) == Some("custom_weight")
            })
            .ok_or_else(|| invalid_intermediate("custom float attribute is missing"))?;
        assert_eq!(
            custom_attribute.get("domain").and_then(Value::as_str),
            Some("POINT")
        );
        assert_eq!(
            custom_attribute.get("data_type").and_then(Value::as_str),
            Some("FLOAT")
        );
        assert_eq!(
            custom_attribute
                .get("values")
                .and_then(Value::as_array)
                .and_then(|values| values.get(1))
                .and_then(Value::as_f64),
            Some(0.25)
        );
        let texts = imported
            .doc
            .compatibility
            .get("blender_adapter")
            .and_then(|adapter| adapter.get("intermediate"))
            .and_then(|source| source.get("texts"))
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_intermediate("Text data-blocks were not preserved"))?;
        assert!(texts.iter().any(|text| {
            text.get("name").and_then(Value::as_str) == Some("FixtureText")
                && text.get("body").and_then(Value::as_str) == Some("No-op text payload\n")
        }));
        assert!(
            cube.modifiers.iter().any(
                |modifier| modifier.name == "FixtureBevel" && modifier.modifier_type == "bevel"
            )
        );
        let orphan_id = imported.id_mappings["Mesh:OrphanMesh"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("orphan mesh ID mapping is missing"))?;
        let orphan = imported
            .doc
            .data_blocks
            .get(&Id::new(orphan_id)?)
            .ok_or_else(|| invalid_intermediate("orphan mesh was not imported"))?;
        assert_eq!(orphan.data_type, "mesh");
        assert_eq!(
            orphan
                .mesh
                .as_ref()
                .and_then(|mesh| mesh.attributes.get("blender_metadata"))
                .and_then(|metadata| metadata.get("fake_user"))
                .and_then(Value::as_bool),
            Some(true)
        );
        let project = Project::init(directory.path().join("project"))?;
        let context = crate::eval::EvaluationContext::default();
        let output_path = directory.path().join("returned.blend");
        let options = ExportOptions {
            allow_lossy: true,
            pack: false,
            blender: Some(&blender),
            context: &context,
        };
        export_blend(&imported.doc, &project, &output_path, &options)?;
        let roundtrip = import_blend(&output_path, Some(&blender))?;
        let roundtrip_cube_id = roundtrip.id_mappings["Object:FixtureCube"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("round-trip cube ID mapping is missing"))?;
        let roundtrip_cube = test_node_ref(&roundtrip.doc, roundtrip_cube_id)?;
        assert_eq!(roundtrip_cube.name, "FixtureCube");
        assert_eq!(roundtrip_cube.kind, "mesh");
        assert_eq!(
            roundtrip_cube.parent.as_ref().map(Id::as_str),
            Some(parent_id)
        );
        assert!(!roundtrip_cube.visible);
        assert!(!roundtrip_cube.render_visible);
        assert_eq!(
            roundtrip_cube
                .modifiers
                .iter()
                .map(|modifier| modifier.modifier_type.as_str())
                .collect::<Vec<_>>(),
            modifier_types
        );
        assert_eq!(
            roundtrip_cube
                .modifiers
                .iter()
                .map(|modifier| modifier.id.as_str())
                .collect::<Vec<_>>(),
            cube.modifiers
                .iter()
                .map(|modifier| modifier.id.as_str())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            roundtrip_cube.modifiers[0].params["use_axis"],
            json!([true, false, false])
        );
        assert_eq!(roundtrip_cube.modifiers[1].params["count"], 3);
        assert_eq!(roundtrip_cube.modifiers[5].params["width"], 0.125);
        assert_eq!(roundtrip_cube.materials, cube.materials);
        assert_eq!(roundtrip_cube.action, cube.action);
        let roundtrip_mesh = roundtrip_cube
            .data
            .as_ref()
            .and_then(|data_id| roundtrip.doc.data_blocks.get(data_id))
            .and_then(|data| data.mesh.as_ref())
            .ok_or_else(|| invalid_intermediate("round-trip cube mesh is missing"))?;
        let roundtrip_attributes = roundtrip_mesh
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_intermediate("round-trip custom mesh attributes are missing"))?;
        let roundtrip_attribute = roundtrip_attributes
            .iter()
            .find(|attribute| {
                attribute.get("name").and_then(Value::as_str) == Some("custom_weight")
            })
            .ok_or_else(|| invalid_intermediate("round-trip custom float attribute is missing"))?;
        assert_eq!(
            roundtrip_attribute.get("data_type").and_then(Value::as_str),
            Some("FLOAT")
        );
        assert_eq!(
            roundtrip_attribute
                .get("values")
                .and_then(Value::as_array)
                .and_then(|values| values.get(1))
                .and_then(Value::as_f64),
            Some(0.25)
        );
        let roundtrip_texts = roundtrip
            .doc
            .compatibility
            .get("blender_adapter")
            .and_then(|adapter| adapter.get("intermediate"))
            .and_then(|source| source.get("texts"))
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_intermediate("round-trip Text data-block is missing"))?;
        assert!(roundtrip_texts.iter().any(|text| {
            text.get("name").and_then(Value::as_str) == Some("FixtureText")
                && text.get("body").and_then(Value::as_str) == Some("No-op text payload\n")
        }));
        assert_eq!(
            test_action_ref(
                &roundtrip.doc,
                roundtrip_cube
                    .action
                    .as_ref()
                    .map(Id::as_str)
                    .ok_or_else(|| invalid_intermediate(
                        "round-trip action reference is missing"
                    ))?
            )?
            .fcurves
            .iter()
            .find(|curve| curve.path == "transform.translation")
            .map(|curve| curve.keyframes.len()),
            Some(2)
        );
        let roundtrip_material = roundtrip
            .doc
            .materials
            .get(&roundtrip_cube.materials[0])
            .ok_or_else(|| invalid_intermediate("round-trip material is missing"))?;
        for (actual, expected) in roundtrip_material
            .base_color
            .into_iter()
            .zip([0.8, 0.15, 0.25, 1.0])
        {
            assert!(approximately(actual, expected));
        }
        let roundtrip_orphan_id = roundtrip.id_mappings["Mesh:OrphanMesh"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("round-trip orphan ID mapping is missing"))?;
        let roundtrip_orphan = test_data_ref(&roundtrip.doc, roundtrip_orphan_id)?;
        assert_eq!(
            roundtrip_orphan
                .mesh
                .as_ref()
                .and_then(|mesh| mesh.attributes.get("blender_metadata"))
                .and_then(|metadata| metadata.get("fake_user"))
                .and_then(Value::as_bool),
            Some(true)
        );
        Ok(())
    }

    fn run_fixture_script(blender: &Path, script: &str, blend_path: &Path) -> Result<()> {
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let script_path = directory.path().join("fixture.py");
        fs::write(&script_path, script).map_err(|error| PotError::io(&error))?;
        let output = Command::new(blender)
            .arg("--background")
            .arg("--factory-startup")
            .arg("--disable-autoexec")
            .arg("--python-exit-code")
            .arg("3")
            .arg("--python")
            .arg(&script_path)
            .arg("--")
            .arg(blend_path)
            .output()
            .map_err(|error| PotError::io(&error))?;
        if !output.status.success() {
            return Err(PotError::with_details(
                ErrorCode::ImportFailed,
                "Blender fixture creation failed",
                child_details(&output),
            ));
        }
        Ok(())
    }

    fn primitive_scene() -> Result<SceneDoc> {
        let mut doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let mut nodes = Vec::new();
        for (index, (id, kind, params, translation)) in [
            ("prim_box", "box", json!({"size": 2.0}), [0.0, 0.0, 1.0]),
            (
                "prim_cylinder",
                "cylinder",
                json!({"vertices": 24, "radius": 0.5, "depth": 1.5}),
                [1.5, 0.0, 0.0],
            ),
            (
                "prim_sphere",
                "uv_sphere",
                json!({"segments": 16, "ring_count": 8, "radius": 0.75}),
                [-1.5, 0.0, 0.0],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let mesh = crate::geom::primitive(kind, &params)
                .map_err(|error| PotError::invalid_argument(error.to_string()))?;
            let data_id = Id::new(format!("mesh_{id}"))?;
            doc.data_blocks.insert(
                data_id.clone(),
                DataBlock {
                    data_type: "mesh".to_owned(),
                    descriptor: Some(crate::model::PrimitiveDescriptor {
                        primitive: kind.to_owned(),
                        params: params.as_object().cloned().unwrap_or_default(),
                    }),
                    mesh: Some(mesh),
                    camera: None,
                    light: None,
                    ..DataBlock::default()
                },
            );
            let mut node = test_node(&format!("Primitive {index}"), data_id.as_str(), None, None)?;
            node.transform.translation = translation;
            if kind == "box" {
                node.transform.scale = [1.0, 0.5, 0.25];
            }
            doc.nodes.insert(Id::new(id)?, node.clone());
            nodes.push(id.to_owned());
        }
        let root = doc
            .collections
            .get_mut(&Id::new("collection_root")?)
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "root collection missing"))?;
        root.objects = nodes
            .into_iter()
            .map(Id::new)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(doc)
    }

    #[test]
    fn blender_export_expands_primitives_and_restores_descriptors() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let project = Project::init(directory.path().join("project"))?;
        let doc = primitive_scene()?;
        let output = directory.path().join("primitives.blend");
        let context = crate::eval::EvaluationContext::default();
        export_blend(
            &doc,
            &project,
            &output,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        )?;
        let imported = import_blend(&output, Some(&blender))?;
        assert!(
            imported.losses.is_empty(),
            "primitive export reported losses: {:?}",
            imported.losses
        );
        for (node_id, kind, params) in [
            ("prim_box", "box", json!({"size": 2.0})),
            (
                "prim_cylinder",
                "cylinder",
                json!({"vertices": 24, "radius": 0.5, "depth": 1.5}),
            ),
            (
                "prim_sphere",
                "uv_sphere",
                json!({"segments": 16, "ring_count": 8, "radius": 0.75}),
            ),
        ] {
            let node = test_node_ref(&imported.doc, node_id)?;
            let data_id = node
                .data
                .as_ref()
                .ok_or_else(|| invalid_intermediate("primitive data reference missing"))?;
            let data = test_data_ref(&imported.doc, data_id.as_str())?;
            let descriptor = data
                .descriptor
                .as_ref()
                .ok_or_else(|| invalid_intermediate("primitive descriptor was not restored"))?;
            assert_eq!(descriptor.primitive, kind);
            assert_eq!(
                Value::from(descriptor.params.clone()),
                params
                    .as_object()
                    .cloned()
                    .map(Value::Object)
                    .unwrap_or_default()
            );
            let source_mesh = doc
                .data_blocks
                .get(&Id::new(format!("mesh_{node_id}"))?)
                .and_then(|data| data.mesh.as_ref())
                .ok_or_else(|| invalid_intermediate("source mesh is missing"))?;
            let imported_mesh = data
                .mesh
                .as_ref()
                .ok_or_else(|| invalid_intermediate("imported mesh is missing"))?;
            let source_bounds = source_mesh
                .bounds()
                .ok_or_else(|| invalid_intermediate("source mesh bounds are missing"))?;
            let imported_bounds = imported_mesh
                .bounds()
                .ok_or_else(|| invalid_intermediate("imported mesh bounds are missing"))?;
            assert!(
                (source_bounds.min - imported_bounds.min)
                    .abs()
                    .max_element()
                    <= 1.0e-6
                    && (source_bounds.max - imported_bounds.max)
                        .abs()
                        .max_element()
                        <= 1.0e-6,
                "primitive bounds diverged for {node_id}"
            );
        }
        Ok(())
    }

    #[test]
    fn blender_import_reads_rig_and_shape_key_ids_and_bone_roll() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let blend_path = directory.path().join("rig-fixture.blend");
        run_fixture_script(
            &blender,
            r#"
import bpy, sys
path = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
armature = bpy.data.armatures.new("Rig")
rig = bpy.data.objects.new("Rig", armature)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
root = armature.edit_bones.new("Root")
root.head = (0, 0, 0)
root.tail = (0, 0, 1)
root.roll = 0.7
tip = armature.edit_bones.new("Tip")
tip.head = (0, 0, 1)
tip.tail = (0, 0, 2)
tip.roll = -0.4
tip.parent = root
tip.use_connect = True
bpy.ops.object.mode_set(mode="OBJECT")
rod_mesh = bpy.data.meshes.new("RodMesh")
rod_mesh.from_pydata([(-0.1, -0.1, 0), (0.1, -0.1, 0), (0.1, 0.1, 0), (-0.1, 0.1, 0),
                      (-0.1, -0.1, 2), (0.1, -0.1, 2), (0.1, 0.1, 2), (-0.1, 0.1, 2)],
                     [], [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
                          (4, 5, 6, 7), (3, 2, 1, 0)])
rod = bpy.data.objects.new("Rod", rod_mesh)
scene.collection.objects.link(rod)
for index, vertex in enumerate(rod_mesh.vertices):
    group_name = "Root" if vertex.co.z < 1.0 else "Tip"
    group = rod.vertex_groups.get(group_name)
    if group is None:
        group = rod.vertex_groups.new(name=group_name)
    group.add([index], 1.0, "REPLACE")
armature_modifier = rod.modifiers.new("Armature", "ARMATURE")
armature_modifier.object = rig
pose = rig.pose.bones["Tip"]
ik = pose.constraints.new("IK")
ik.target = rig
ik.subtarget = "Root"
ik.chain_count = 2
rod.shape_key_add(name="Basis", from_mix=False)
shape = rod.shape_key_add(name="Bend", from_mix=False)
for point in shape.data:
    point.co.z += 0.25
bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
"#,
            &blend_path,
        )?;
        let imported = import_blend(&blend_path, Some(&blender))?;
        assert!(
            imported.losses.is_empty(),
            "rig fixture import reported losses: {:?}",
            imported.losses
        );
        let rod_id = imported.id_mappings["Object:Rod"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("rod ID mapping is missing"))?;
        let rod = test_node_ref(&imported.doc, rod_id)?;
        let mesh_id = rod
            .data
            .as_ref()
            .ok_or_else(|| invalid_intermediate("rod mesh reference is missing"))?;
        let mesh_data = test_data_ref(&imported.doc, mesh_id.as_str())?;
        let group_names = mesh_data
            .vertex_groups
            .iter()
            .map(|group| group.name.clone())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(group_names.len(), 2);
        let expected_weights = [("Root", 0_u64), ("Tip", 4)];
        for (group_name, vertex_index) in expected_weights {
            let group = mesh_data
                .vertex_groups
                .iter()
                .find(|group| group.name == group_name)
                .ok_or_else(|| invalid_intermediate("vertex group is missing"))?;
            let vertex_id = u32::try_from(vertex_index + 1)
                .map_err(|_| invalid_intermediate("vertex index does not fit a u32"))?;
            let weight = mesh_data
                .vertex_weights
                .get(&vertex_id)
                .and_then(|weights| weights.get(&group.id));
            assert!(
                weight.is_some_and(|weight| approximately(*weight, 1.0)),
                "missing weight for {group_name}"
            );
        }
        let shape_keys = mesh_data
            .shape_keys
            .as_ref()
            .ok_or_else(|| invalid_intermediate("shape keys were not imported"))?;
        assert_eq!(shape_keys.keys.len(), 1);
        let shape_key = shape_keys
            .keys
            .values()
            .next()
            .ok_or_else(|| invalid_intermediate("shape key is missing"))?;
        assert_eq!(shape_key.name, "Bend");
        let armature_modifier = rod
            .modifiers
            .iter()
            .find(|modifier| modifier.modifier_type == "armature")
            .ok_or_else(|| invalid_intermediate("armature modifier is missing"))?;
        let rig_node_id = armature_modifier
            .params
            .get("object")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_intermediate("armature modifier target is missing"))?;
        let rig_node = test_node_ref(&imported.doc, rig_node_id)?;
        let armature_id = rig_node
            .data
            .as_ref()
            .ok_or_else(|| invalid_intermediate("armature data reference is missing"))?;
        let armature = imported
            .doc
            .data_blocks
            .get(armature_id)
            .and_then(|data| data.armature.as_ref())
            .ok_or_else(|| invalid_intermediate("armature is missing"))?;
        assert_eq!(armature.bones.len(), 2);
        let root_bone = armature
            .bones
            .iter()
            .find(|(_, bone)| bone.name == "Root")
            .ok_or_else(|| invalid_intermediate("Root bone is missing"))?;
        assert!(approximately(root_bone.1.roll, 0.7));
        let tip_bone = armature
            .bones
            .values()
            .find(|bone| bone.name == "Tip")
            .ok_or_else(|| invalid_intermediate("Tip bone is missing"))?;
        assert!(approximately(tip_bone.roll, -0.4));
        let project = Project::init(directory.path().join("project"))?;
        let roundtrip = directory.path().join("rig-roundtrip.blend");
        let context = crate::eval::EvaluationContext::default();
        export_blend(
            &imported.doc,
            &project,
            &roundtrip,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        )?;
        let probe_path = directory.path().join("probe_rig.py");
        fs::write(
            &probe_path,
            r#"
import bpy, sys, json
path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.open_mainfile(filepath=path)
rig = bpy.data.objects.get("Rig")
rod = bpy.data.objects.get("Rod")
groups = sorted(group.name for group in rod.vertex_groups) if rod else []
shape_names = sorted(key.name for key in rod.data.shape_keys.key_blocks[1:]) if rod and rod.data.shape_keys else []
ik = None
if rig and rig.pose:
    bone = rig.pose.bones.get("Tip")
    if bone:
        ik = [(c.name, c.type, c.target.name if c.target else None, c.subtarget)
              for c in bone.constraints if c.type == "IK"]
weights = {}
if rod:
    for group in rod.vertex_groups:
        weights[group.name] = [round(v.groups[0].weight, 3) for v in rod.data.vertices
                               for a in v.groups if a.group == group.index]
print("RIGPROBE " + json.dumps({"groups": groups, "shapes": shape_names,
                          "bones": sorted(b.name for b in rig.data.bones) if rig and rig.data else [],
                          "ik": ik, "weights": weights}))
"#,
        )
        .map_err(|error| PotError::io(&error))?;
        let probe = Command::new(&blender)
            .arg("--background")
            .arg("--factory-startup")
            .arg("--disable-autoexec")
            .arg("--python-exit-code")
            .arg("3")
            .arg("--python")
            .arg(&probe_path)
            .arg("--")
            .arg(&roundtrip)
            .output()
            .map_err(|error| PotError::io(&error))?;
        if !probe.status.success() {
            return Err(PotError::with_details(
                ErrorCode::ExportFailed,
                "rig roundtrip probe failed",
                child_details(&probe),
            ));
        }
        let stdout = String::from_utf8_lossy(&probe.stdout);
        let line = stdout
            .lines()
            .find(|line| line.starts_with("RIGPROBE "))
            .ok_or_else(|| invalid_intermediate("rig probe output is missing"))?;
        let probe: Value = serde_json::from_str(line.trim_start_matches("RIGPROBE "))
            .map_err(PotError::internal_json)?;
        assert_eq!(
            probe.get("groups").and_then(Value::as_array).map(Vec::len),
            Some(2),
            "vertex groups were not restored: {probe}"
        );
        assert_eq!(
            probe.get("shapes").and_then(Value::as_array).map(Vec::len),
            Some(1),
            "shape keys were not restored: {probe}"
        );
        assert_eq!(
            probe.get("bones").and_then(Value::as_array).map(Vec::len),
            Some(2),
            "armature bones were not restored: {probe}"
        );
        let ik = probe
            .get("ik")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .ok_or_else(|| invalid_intermediate("pose IK constraint was not restored"))?;
        assert_eq!(ik[0], json!("IK"));
        assert_eq!(ik[1], json!("IK"));
        assert_eq!(ik[2], json!("Rig"));
        assert_eq!(ik[3], json!("Root"));
        let weights = probe
            .get("weights")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid_intermediate("vertex weights were not restored"))?;
        assert_eq!(weights.len(), 2);
        for (_, group_weights) in weights {
            assert!(group_weights.as_array().is_some_and(|values| {
                values
                    .iter()
                    .all(|value| approximately(value.as_f64().unwrap_or(0.0), 1.0))
            }));
        }
        Ok(())
    }

    fn rig_graph_scene() -> Result<SceneDoc> {
        let mut doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let mesh = crate::geom::primitive("box", &json!({"size": 0.5}))
            .map_err(|error| PotError::invalid_argument(error.to_string()))?;
        let first_vertex = mesh
            .vertices
            .first()
            .map(|vertex| vertex.id)
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "primitive mesh is empty"))?;
        doc.data_blocks.insert(
            Id::new("mesh_rod")?,
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                vertex_groups: vec![VertexGroup {
                    id: Id::new("group_root")?,
                    name: "Root".to_owned(),
                }],
                vertex_weights: {
                    let mut weights = BTreeMap::new();
                    weights.insert(
                        first_vertex,
                        BTreeMap::from([(Id::new("group_root")?, 1.0_f64)]),
                    );
                    weights
                },
                shape_keys: Some(ShapeKeyData {
                    keys: Registry::from([(
                        Id::new("key_bend")?,
                        ShapeKey {
                            id: Id::new("key_bend")?,
                            name: "Bend".to_owned(),
                            value: 0.5,
                            mute: false,
                            slider_min: 0.0,
                            slider_max: 1.0,
                            relative_key: None,
                            vertex_group: None,
                            frame: 0.0,
                            positions: BTreeMap::new(),
                        },
                    )]),
                    ..ShapeKeyData::default()
                }),
                mesh: Some(mesh),
                ..DataBlock::default()
            },
        );
        doc.data_blocks.insert(
            Id::new("armature_rig")?,
            DataBlock {
                data_type: "armature".to_owned(),
                armature: Some(ArmatureData {
                    bones: Registry::from([
                        (
                            Id::new("bone_root")?,
                            Bone {
                                name: "Root".to_owned(),
                                parent: None,
                                head: [0.0, 0.0, 0.0],
                                tail: [0.0, 0.0, 1.0],
                                roll: 0.7,
                                deform: true,
                                inherit_rotation: true,
                                use_connect: false,
                                custom_shape: None,
                                envelope_distance: 0.25,
                                envelope_weight: 1.0,
                                head_radius: 0.1,
                                tail_radius: 0.1,
                                bbone_settings: BTreeMap::new(),
                            },
                        ),
                        (
                            Id::new("bone_tip")?,
                            Bone {
                                name: "Tip".to_owned(),
                                parent: Some(Id::new("bone_root")?),
                                head: [0.0, 0.0, 1.0],
                                tail: [0.0, 0.0, 2.0],
                                roll: -0.4,
                                deform: true,
                                inherit_rotation: true,
                                use_connect: true,
                                custom_shape: None,
                                envelope_distance: 0.25,
                                envelope_weight: 1.0,
                                head_radius: 0.1,
                                tail_radius: 0.1,
                                bbone_settings: BTreeMap::new(),
                            },
                        ),
                    ]),
                    ..ArmatureData::default()
                }),
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            Id::new("rig")?,
            Node {
                name: "Rig".to_owned(),
                kind: "armature".to_owned(),
                data: Some(Id::new("armature_rig")?),
                ..Node::default()
            },
        );
        doc.nodes.insert(
            Id::new("rod")?,
            Node {
                name: "Rod".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(Id::new("mesh_rod")?),
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&Id::new("collection_root")?)
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "root collection missing"))?
            .objects = vec![Id::new("rig")?, Id::new("rod")?];
        Ok(doc)
    }

    #[test]
    fn blender_roundtrip_restores_rig_ids_without_id_properties() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let project = Project::init(directory.path().join("project"))?;
        let doc = rig_graph_scene()?;
        let output = directory.path().join("rig-ids.blend");
        let context = crate::eval::EvaluationContext::default();
        export_blend(
            &doc,
            &project,
            &output,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        )?;
        let imported = import_blend(&output, Some(&blender))?;
        assert!(
            imported.losses.is_empty(),
            "rig ID roundtrip reported losses: {:?}",
            imported.losses
        );
        let rod_id = imported.id_mappings["Object:Rod"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("rod ID mapping is missing"))?;
        let rod = test_node_ref(&imported.doc, rod_id)?;
        let mesh_id = rod
            .data
            .as_ref()
            .ok_or_else(|| invalid_intermediate("rod mesh reference is missing"))?;
        let mesh_data = test_data_ref(&imported.doc, mesh_id.as_str())?;
        let group_root = Id::new("group_root")?;
        assert_eq!(
            mesh_data
                .vertex_groups
                .first()
                .map(|group| group.id.as_str()),
            Some("group_root"),
            "vertex group ID was not restored"
        );
        assert!(
            mesh_data
                .vertex_weights
                .values()
                .any(|weights| weights.contains_key(&group_root)),
            "vertex weight group reference was not restored"
        );
        let shape_keys = mesh_data
            .shape_keys
            .as_ref()
            .ok_or_else(|| invalid_intermediate("shape keys were not restored"))?;
        assert_eq!(
            shape_keys.keys.keys().next().map(Id::as_str),
            Some("key_bend"),
            "shape key ID was not restored"
        );
        let rig_id = imported.id_mappings["Object:Rig"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("rig ID mapping is missing"))?;
        let rig = test_node_ref(&imported.doc, rig_id)?;
        let armature_id = rig
            .data
            .as_ref()
            .ok_or_else(|| invalid_intermediate("armature reference is missing"))?;
        let armature = imported
            .doc
            .data_blocks
            .get(armature_id)
            .and_then(|data| data.armature.as_ref())
            .ok_or_else(|| invalid_intermediate("armature was not restored"))?;
        let root_roll = armature
            .bones
            .get(&Id::new("bone_root")?)
            .map(|bone| bone.roll);
        assert!(
            root_roll.is_some_and(|roll| approximately(roll, 0.7)),
            "bone ID/roll were not restored: {root_roll:?}"
        );
        let tip_roll = armature
            .bones
            .get(&Id::new("bone_tip")?)
            .map(|bone| (bone.roll, bone.parent.clone()));
        assert!(
            tip_roll
                .as_ref()
                .is_some_and(|(roll, parent)| approximately(*roll, -0.4)
                    && parent
                        .as_ref()
                        .is_some_and(|parent| parent.as_str() == "bone_root")),
            "tip bone ID/roll/parent were not restored: {tip_roll:?}"
        );
        Ok(())
    }

    #[test]
    fn blender_export_writes_canonical_keyframes_over_compat_payload() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let blend_path = directory.path().join("animated-fixture.blend");
        run_fixture_script(
            &blender,
            r#"
import bpy, sys
path = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add()
cube = bpy.context.object
cube.name = "Cube"
cube.location.x = 0.0
cube.keyframe_insert(data_path="location", frame=1)
cube.location.x = 4.0
cube.keyframe_insert(data_path="location", frame=24)
bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
"#,
            &blend_path,
        )?;
        let imported = import_blend(&blend_path, Some(&blender))?;
        let cube_id = imported.id_mappings["Object:Cube"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("cube ID mapping is missing"))?;
        let cube = test_node_ref(&imported.doc, cube_id)?;
        let action_id = cube
            .action
            .clone()
            .ok_or_else(|| invalid_intermediate("cube action is missing"))?;
        let mut doc = imported.doc;
        let action = doc
            .actions
            .get_mut(&action_id)
            .ok_or_else(|| invalid_intermediate("action registry is missing"))?;
        for curve in &mut action.fcurves {
            for key in &mut curve.keyframes {
                key.value += 1.0;
            }
        }
        let project = Project::init(directory.path().join("project"))?;
        let output = directory.path().join("edited.blend");
        let context = crate::eval::EvaluationContext::default();
        export_blend(
            &doc,
            &project,
            &output,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        )?;
        let reimported = import_blend(&output, Some(&blender))?;
        let reimported_cube_id = reimported.id_mappings["Object:Cube"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("round-trip cube mapping is missing"))?;
        let reimported_cube = test_node_ref(&reimported.doc, reimported_cube_id)?;
        let reimported_action_id = reimported_cube
            .action
            .clone()
            .ok_or_else(|| invalid_intermediate("round-trip action is missing"))?;
        let reimported_action = test_action_ref(&reimported.doc, reimported_action_id.as_str())?;
        let shifted = reimported_action
            .fcurves
            .iter()
            .find(|curve| curve.path == "transform.translation")
            .ok_or_else(|| invalid_intermediate("translation fcurve is missing"))?;
        let values = shifted
            .keyframes
            .iter()
            .map(|key| key.value)
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 2);
        assert!(
            approximately(values[0], 1.0),
            "first keyframe not edited: {values:?}"
        );
        assert!(
            approximately(values[1], 5.0),
            "second keyframe not edited: {values:?}"
        );
        Ok(())
    }

    #[test]
    fn blender_export_keeps_static_transforms_on_animated_and_plain_objects() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let project = Project::init(directory.path().join("project"))?;
        let mut doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let mesh = crate::geom::primitive("box", &json!({"size": 1.0}))
            .map_err(|error| PotError::invalid_argument(error.to_string()))?;
        doc.data_blocks.insert(
            Id::new("mesh_common")?,
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(mesh),
                camera: None,
                light: None,
                ..DataBlock::default()
            },
        );
        doc.actions.insert(
            Id::new("act_lift")?,
            Action {
                name: "Lift".to_owned(),
                fcurves: vec![FCurve {
                    path: "transform.translation".to_owned(),
                    index: 2,
                    keyframes: vec![
                        Keyframe {
                            frame: 1.0,
                            value: 0.0,
                            interpolation: Interpolation::Linear,
                            ..Keyframe::default()
                        },
                        Keyframe {
                            frame: 24.0,
                            value: 1.0,
                            interpolation: Interpolation::Linear,
                            ..Keyframe::default()
                        },
                    ],
                    extrapolation: Extrapolation::Constant,
                }],
                slots: Vec::new(),
            },
        );
        let mut animated = test_node("Animated Box", "mesh_common", None, Some("act_lift"))?;
        animated.transform.translation = [-1.2, 0.0, 0.0];
        doc.nodes.insert(Id::new("animated")?, animated);
        let mut plain = test_node("Plain Sphere", "mesh_common", None, None)?;
        plain.transform.translation = [1.1, 0.0, 0.0];
        doc.nodes.insert(Id::new("plain")?, plain);
        let root = doc
            .collections
            .get_mut(&Id::new("collection_root")?)
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "root collection missing"))?;
        root.objects = vec![Id::new("animated")?, Id::new("plain")?];
        let output = directory.path().join("transforms.blend");
        let context = crate::eval::EvaluationContext::default();
        export_blend(
            &doc,
            &project,
            &output,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        )?;
        let script_path = directory.path().join("inspect.py");
        fs::write(
            &script_path,
            r#"
import bpy, sys, json
path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.open_mainfile(filepath=path)
result = {}
for name in ("Animated Box", "Plain Sphere"):
    obj = bpy.data.objects.get(name)
    result[name] = [round(value, 6) for value in obj.matrix_world.translation] if obj else None
print("TRANSFORMS " + json.dumps(result))
"#,
        )
        .map_err(|error| PotError::io(&error))?;
        let inspected = Command::new(&blender)
            .arg("--background")
            .arg("--factory-startup")
            .arg("--disable-autoexec")
            .arg("--python-exit-code")
            .arg("3")
            .arg("--python")
            .arg(&script_path)
            .arg("--")
            .arg(&output)
            .output()
            .map_err(|error| PotError::io(&error))?;
        let stdout = String::from_utf8_lossy(&inspected.stdout);
        let line = stdout
            .lines()
            .find(|line| line.starts_with("TRANSFORMS "))
            .ok_or_else(|| invalid_intermediate("transform inspection failed"))?;
        let transforms: Value = serde_json::from_str(line.trim_start_matches("TRANSFORMS "))
            .map_err(PotError::internal_json)?;
        let animated = transforms
            .get("Animated Box")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_intermediate("animated object transform is missing"))?;
        let plain = transforms
            .get("Plain Sphere")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_intermediate("plain object transform is missing"))?;
        for (axis, expected) in [-1.2_f64, 0.0, 0.0].into_iter().enumerate() {
            let actual = animated
                .get(axis)
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid_intermediate("animated object transform is missing"))?;
            assert!(
                approximately(actual, expected),
                "animated box transform mismatch at axis {axis}: {actual} != {expected}"
            );
        }
        for (axis, expected) in [1.1_f64, 0.0, 0.0].into_iter().enumerate() {
            let actual = plain
                .get(axis)
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid_intermediate("plain object transform is missing"))?;
            assert!(
                approximately(actual, expected),
                "plain object transform mismatch at axis {axis}: {actual} != {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn blender_export_prefers_graph_edits_over_compat_payload() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let blend_path = directory.path().join("overlay-fixture.blend");
        run_fixture_script(
            &blender,
            r#"
import bpy, sys
path = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add()
top = bpy.context.object
top.name = "Tabletop"
top.data.name = "TabletopMesh"
top.scale = (2.0, 2.0, 0.25)
material = bpy.data.materials.new("Walnut brown")
material.use_nodes = True
top.data.materials.append(material)
bevel = top.modifiers.new("Soft edges", "BEVEL")
bevel.width = 0.035
top.location.z = 1.0
top.keyframe_insert(data_path="location", frame=1)
top.location.z = 1.2
top.keyframe_insert(data_path="location", frame=24)
armature = bpy.data.armatures.new("RodArmatureData")
rig = bpy.data.objects.new("RodArmature", armature)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
root = armature.edit_bones.new("Root")
root.head = (0, 0, 0); root.tail = (0, 0, 1)
tip = armature.edit_bones.new("Tip")
tip.head = (0, 0, 1); tip.tail = (0, 0, 2); tip.parent = root; tip.use_connect = True
bpy.ops.object.mode_set(mode="OBJECT")
rod_mesh = bpy.data.meshes.new("RodMesh")
rod_mesh.from_pydata([(-0.1,-0.1,0),(0.1,-0.1,0),(0.1,0.1,0),(-0.1,0.1,0),
                      (-0.1,-0.1,2),(0.1,-0.1,2),(0.1,0.1,2),(-0.1,0.1,2)],
                     [], [(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7),(4,5,6,7),(3,2,1,0)])
rod = bpy.data.objects.new("RiggedRod", rod_mesh)
scene.collection.objects.link(rod)
for index, vertex in enumerate(rod_mesh.vertices):
    group_name = "RodRoot" if vertex.co.z < 1.0 else "RodTip"
    group = rod.vertex_groups.get(group_name) or rod.vertex_groups.new(name=group_name)
    group.add([index], 1.0, "REPLACE")
skin = rod.modifiers.new("RodSkin", "ARMATURE")
skin.object = rig
rod.shape_key_add(name="Basis", from_mix=False)
flex = rod.shape_key_add(name="Flex", from_mix=False)
flex.value = 0.5
bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
"#,
            &blend_path,
        )?;
        let imported = import_blend(&blend_path, Some(&blender))?;
        let doc = imported.doc;
        let tabletop = test_node_ref(&doc, "node_tabletop")?.clone();
        let bevel_id = tabletop
            .modifiers
            .iter()
            .find(|modifier| modifier.modifier_type == "bevel")
            .map(|modifier| modifier.id.clone())
            .ok_or_else(|| invalid_intermediate("bevel modifier is missing"))?;
        let material_id = tabletop
            .materials
            .first()
            .cloned()
            .ok_or_else(|| invalid_intermediate("tabletop material is missing"))?;
        let rod = test_node_ref(&doc, "node_riggedrod")?.clone();
        let shape_key_id = rod
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.shape_keys.as_ref())
            .and_then(|shape| shape.keys.keys().next().cloned())
            .ok_or_else(|| invalid_intermediate("rod shape key is missing"))?;
        let leg_id = imported.id_mappings["Object:RiggedRod"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| invalid_intermediate("rod ID mapping is missing"))?;
        let action_id = tabletop
            .action
            .clone()
            .ok_or_else(|| invalid_intermediate("tabletop action is missing"))?;
        let doc = crate::ops::apply_batch(
            &doc,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [
                    {"op": "modifier.update", "target": {"id": "node_tabletop"},
                     "id": bevel_id.as_str(), "set": {"params": {"width": 0.08}}},
                    {"op": "material.update", "target": {"id": material_id.as_str()},
                     "set": {"base_color": [0.9, 0.15, 0.1, 1.0]}},
                    {"op": "shape_key.update", "target": {"id": "node_riggedrod"},
                     "id": shape_key_id.as_str(), "set": {"value": 0.75}},
                    {"op": "constraint.create", "target": {"id": "node_tabletop"},
                     "id": "constraint_follow", "type": "copy_location",
                     "name": "Follow Leg", "constraint_target": leg_id.as_str(),
                     "influence": 1.0, "enabled": true, "params": {}},
                    {"op": "driver.create", "target": {"id": "node_tabletop"},
                     "id": "driver_follow", "path": "transform.translation",
                     "index": 0, "type": "scripted_expression",
                     "expression": "leg_x",
                     "variables": [{"name": "leg_x", "type": "single_prop",
                                    "target": leg_id.as_str(),
                                    "path": "transform.translation"}]},
                    {"op": "nla.track_create", "target": {"id": "node_tabletop"},
                     "id": "track_main", "name": "Main"},
                    {"op": "nla.strip_create", "target": {"id": "node_tabletop"},
                     "track": "track_main", "id": "strip_lift",
                     "action": action_id.as_str(), "frame_start": 1.0,
                     "frame_end": 11.0, "action_frame_start": 1.0,
                     "action_frame_end": 11.0, "scale": 1.0, "repeat": 1.0,
                     "blend_type": "replace", "influence": 1.0,
                     "extrapolation": "hold", "blend_in": 0.0, "blend_out": 0.0}
                ]
            }),
        )?
        .doc;
        let project = Project::init(directory.path().join("project"))?;
        let output = directory.path().join("overlay.blend");
        let context = crate::eval::EvaluationContext::default();
        export_blend(
            &doc,
            &project,
            &output,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        )?;
        let probe_path = directory.path().join("probe_overlay.py");
        fs::write(
            &probe_path,
            r#"
import bpy, sys, json
path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.open_mainfile(filepath=path)
tt = bpy.data.objects.get("Tabletop")
rod = bpy.data.objects.get("RiggedRod")
bevel = next((m for m in tt.modifiers if m.type == "BEVEL"), None)
mat = tt.data.materials[0] if tt.data.materials else None
base = None
if mat and mat.node_tree:
    principled = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled:
        base = [round(v, 3) for v in principled.inputs["Base Color"].default_value]
flex = None
if rod and rod.data.shape_keys:
    key = next((k for k in rod.data.shape_keys.key_blocks if k.name == "Flex"), None)
    flex = round(key.value, 3) if key else None
constraint = None
if tt.constraints:
    c = tt.constraints[0]
    constraint = [c.type, c.target.name if c.target else None, round(c.influence, 3)]
driver = None
if tt.animation_data:
    for curve in tt.animation_data.drivers:
        driver = [curve.driver.expression, curve.driver.type]
tracks = []
if tt.animation_data:
    for track in tt.animation_data.nla_tracks:
        for strip in track.strips:
            tracks.append([track.name, strip.action.name if strip.action else None,
                           strip.frame_start, strip.frame_end])
print("OVERLAY " + json.dumps({"bevel": round(bevel.width, 3) if bevel else None,
                               "base": base, "flex": flex, "constraint": constraint,
                               "driver": driver, "nla": tracks}))
"#,
        )
        .map_err(|error| PotError::io(&error))?;
        let probe = Command::new(&blender)
            .arg("--background")
            .arg("--factory-startup")
            .arg("--disable-autoexec")
            .arg("--python-exit-code")
            .arg("3")
            .arg("--python")
            .arg(&probe_path)
            .arg("--")
            .arg(&output)
            .output()
            .map_err(|error| PotError::io(&error))?;
        if !probe.status.success() {
            return Err(PotError::with_details(
                ErrorCode::ExportFailed,
                "overlay probe failed",
                child_details(&probe),
            ));
        }
        let stdout = String::from_utf8_lossy(&probe.stdout);
        let line = stdout
            .lines()
            .find(|line| line.starts_with("OVERLAY "))
            .ok_or_else(|| invalid_intermediate("overlay probe output is missing"))?;
        let probe: Value = serde_json::from_str(line.trim_start_matches("OVERLAY "))
            .map_err(PotError::internal_json)?;
        assert_eq!(
            probe["bevel"],
            json!(0.08),
            "bevel edit lost to compat payload"
        );
        assert_eq!(
            probe["base"],
            json!([0.9, 0.15, 0.1, 1.0]),
            "material edit lost"
        );
        assert_eq!(probe["flex"], json!(0.75), "shape key value edit lost");
        assert_eq!(probe["constraint"][0], json!("COPY_LOCATION"));
        assert_eq!(probe["constraint"][1], json!("RiggedRod"));
        assert_eq!(probe["driver"][0], json!("leg_x"));
        assert_eq!(probe["nla"][0][0], json!("Main"));
        assert!(probe["nla"][0][1].is_string(), "NLA strip lost its action");
        assert_eq!(probe["nla"][0][2], json!(1.0));
        assert_eq!(probe["nla"][0][3], json!(11.0));
        Ok(())
    }

    #[test]
    fn blender_export_bakes_root_parent_inverse_into_world_matrix() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let project = Project::init(directory.path().join("project"))?;
        let mut doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        let mesh = crate::geom::primitive("box", &json!({"size": 1.0}))
            .map_err(|error| PotError::invalid_argument(error.to_string()))?;
        doc.data_blocks.insert(
            Id::new("mesh_root")?,
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(mesh),
                ..DataBlock::default()
            },
        );
        let node_id = Id::new("root_shear")?;
        let node = Node {
            name: "Root Shear".to_owned(),
            kind: "mesh".to_owned(),
            data: Some(Id::new("mesh_root")?),
            transform: Transform {
                translation: [0.5, 0.0, 0.0],
                ..Transform::default()
            },
            parent_inverse: Some([
                1.0, 0.0, 0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0,
            ]),
            ..Node::default()
        };
        doc.nodes.insert(node_id.clone(), node);
        doc.collections
            .get_mut(&Id::new("collection_root")?)
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "root collection missing"))?
            .objects = vec![node_id];
        let output = directory.path().join("root-shear.blend");
        let context = crate::eval::EvaluationContext::default();
        let report = export_blend(
            &doc,
            &project,
            &output,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        )?;
        assert_eq!(report.conversions.len(), 1);
        assert_eq!(
            report.conversions[0]["feature_id"],
            json!("blend.root_matrix_shear")
        );
        let moved_output = directory.path().join("root-shear-moved.blend");
        let probe_path = directory.path().join("probe_root_shear.py");
        fs::write(
            &probe_path,
            r#"
import bpy, sys, json
args = sys.argv[sys.argv.index("--") + 1:]
path, moved_path = args
bpy.ops.wm.open_mainfile(filepath=path)
obj = bpy.data.objects.get("Root Shear")
matrix = [[round(obj.matrix_world[row][column], 6) for column in range(4)]
          for row in range(4)]
obj.location.x += 0.25
bpy.ops.wm.save_as_mainfile(filepath=moved_path, check_existing=False)
print("ROOT_MATRIX " + json.dumps(matrix))
"#,
        )
        .map_err(|error| PotError::io(&error))?;
        let probe = Command::new(&blender)
            .arg("--background")
            .arg("--factory-startup")
            .arg("--disable-autoexec")
            .arg("--python-exit-code")
            .arg("3")
            .arg("--python")
            .arg(&probe_path)
            .arg("--")
            .arg(&output)
            .arg(&moved_output)
            .output()
            .map_err(|error| PotError::io(&error))?;
        if !probe.status.success() {
            return Err(PotError::with_details(
                ErrorCode::ExportFailed,
                "root matrix probe failed",
                child_details(&probe),
            ));
        }
        let stdout = String::from_utf8_lossy(&probe.stdout);
        let line = stdout
            .lines()
            .find(|line| line.starts_with("ROOT_MATRIX "))
            .ok_or_else(|| invalid_intermediate("root matrix probe output is missing"))?;
        let matrix: Value = serde_json::from_str(line.trim_start_matches("ROOT_MATRIX "))
            .map_err(PotError::internal_json)?;
        let shear = matrix[0][1]
            .as_f64()
            .ok_or_else(|| invalid_intermediate("root matrix shear component is missing"))?;
        assert!(shear.abs() > 0.01, "root shear was dropped: {shear}");
        assert!(
            shear < 0.25,
            "Blender root display should report an approximation"
        );
        assert_eq!(
            matrix[0][3],
            json!(2.5),
            "root inverse x translation was dropped"
        );
        assert_eq!(matrix[1][3], json!(3.0));
        assert_eq!(matrix[2][3], json!(4.0));
        let imported = import_blend(&output, Some(&blender))?;
        let imported_id = imported.id_mappings["Object:Root Shear"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("root object ID mapping is missing"))?;
        let imported_node = test_node_ref(&imported.doc, imported_id)?;
        assert!(
            imported_node
                .properties
                .contains_key("potter.root_matrix_json"),
            "root matrix marker missing; properties: {:?}",
            imported_node.properties.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            imported_node.parent_inverse,
            Some([
                1.0, 0.0, 0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0,
            ]),
            "unchanged Blender basis must restore the exact Potter parent inverse"
        );
        assert_eq!(imported_node.transform.translation, [0.5, 0.0, 0.0]);
        let moved = import_blend(&moved_output, Some(&blender))?;
        let moved_id = moved.id_mappings["Object:Root Shear"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("moved root ID mapping is missing"))?;
        let moved_node = test_node_ref(&moved.doc, moved_id)?;
        assert_eq!(
            moved_node.parent_inverse,
            Some([
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
            "editing Blender's baked basis must stop restoring stale Potter matrices"
        );
        assert_ne!(
            moved_node.transform.translation,
            [0.5, 0.0, 0.0],
            "edited Blender location should be imported instead of stale metadata"
        );
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one Blender fixture exercises pose import, evaluation, update, and export"
    )]
    #[test]
    fn blender_pose_ik_constraints_import_evaluate_and_export_edits() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let source = directory.path().join("pose-ik.blend");
        run_fixture_script(
            &blender,
            r#"
import bpy, sys
path = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
armature = bpy.data.armatures.new("IKRigData")
rig = bpy.data.objects.new("IKRig", armature)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
root = armature.edit_bones.new("Root")
root.head = (0, 0, 0)
root.tail = (0, 1, 0)
tip = armature.edit_bones.new("Tip")
tip.head = (0, 1, 0)
tip.tail = (0, 2, 0)
tip.parent = root
bpy.ops.object.mode_set(mode="OBJECT")
goal = bpy.data.objects.new("IKGoal", None)
scene.collection.objects.link(goal)
goal.location = (1, 1, 0)
constraint = rig.pose.bones["Tip"].constraints.new("IK")
constraint.name = "Pose Reach"
constraint.target = goal
constraint.chain_count = 2
constraint.influence = 1.0
copy_constraint = rig.pose.bones["Root"].constraints.new("COPY_TRANSFORMS")
copy_constraint.name = "Pose Copy"
copy_constraint.target = goal
copy_constraint.influence = 0.0
bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
"#,
            &source,
        )?;
        let project_path = directory.path().join("project");
        Project::init(&project_path)?;
        let (_, import_result) = crate::commands::import::run(&crate::cli::ImportArgs {
            scene: project_path.clone(),
            file: source.clone(),
            format: crate::cli::ExchangeFormat::Blend,
            base_revision: 0,
            mode: crate::cli::ImportMode::Replace,
            asset_policy: crate::cli::AssetPolicy::Copy,
            allow_lossy: false,
            dry_run: false,
            blender: Some(blender.clone()),
        })?;
        assert_eq!(import_result["losses"], json!([]));
        let rig_id = import_result["id_mappings"]["Object:IKRig"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("IK armature object mapping is missing"))?;
        let goal_id = import_result["id_mappings"]["Object:IKGoal"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("IK target mapping is missing"))?;
        let project = Project::open(&project_path)?;
        let doc = project.doc();
        let rig_node = test_node_ref(doc, rig_id)?;
        let ik = rig_node
            .constraints
            .iter()
            .find(|constraint| constraint.name == "Pose Reach")
            .ok_or_else(|| invalid_intermediate("pose IK was not imported into the graph"))?;
        assert_eq!(ik.constraint_type, ConstraintType::Ik);
        let tip_id = ik
            .owner_bone
            .as_ref()
            .ok_or_else(|| invalid_intermediate("pose IK owner bone is missing"))?;
        assert_eq!(ik.target.as_ref().map(Id::as_str), Some(goal_id));
        assert_eq!(ik.params.get("chain_count"), Some(&json!(2)));
        let constraint_id = ik.id.to_string();
        let tip_id = tip_id.to_string();
        let armature_data_id = rig_node
            .data
            .as_ref()
            .ok_or_else(|| invalid_intermediate("IK armature data is missing"))?;
        let tip_bone = doc
            .data_blocks
            .get(armature_data_id)
            .and_then(|data| data.armature.as_ref())
            .and_then(|armature| armature.bones.get(&Id::new(tip_id.clone()).ok()?))
            .ok_or_else(|| invalid_intermediate("IK owner bone data is missing"))?;
        let bone_length = (glam::DVec3::from_array(tip_bone.tail)
            - glam::DVec3::from_array(tip_bone.head))
        .length();
        let (_, inspected) = crate::commands::inspect::run(crate::cli::InspectArgs {
            scene: project_path.clone(),
            id: Some(rig_id.to_owned()),
            tag: None,
            features: false,
            context: crate::cli::Context {
                scene_id: None,
                view_layer: None,
                frame: None,
            },
        })?;
        let rig_item = inspected["items"]
            .as_array()
            .and_then(|items| items.iter().find(|item| item["id"] == rig_id))
            .ok_or_else(|| invalid_intermediate("inspect omitted the imported armature"))?;
        let matrix_values = rig_item["evaluated_bones"][tip_id]
            .as_array()
            .ok_or_else(|| invalid_intermediate("inspect omitted the evaluated IK owner bone"))?;
        let mut matrix = [0.0; 16];
        for (target, value) in matrix.iter_mut().zip(matrix_values) {
            *target = value
                .as_f64()
                .ok_or_else(|| invalid_intermediate("evaluated bone matrix is not numeric"))?;
        }
        let end =
            glam::DMat4::from_cols_array(&matrix).transform_point3(glam::DVec3::Y * bone_length);
        let target = glam::DVec3::new(1.000_038_385_391_235_4, 1.000_020_980_834_961, 0.0);
        assert!(
            end.distance(target) <= 1.0e-6,
            "pose IK end bone did not reach its target: {end:?} vs {target:?}"
        );

        let batch_path = directory.path().join("pose-ik-update.json");
        fs::write(
            &batch_path,
            serde_json::to_vec(&json!({
                "schema_version": 1,
                "base_revision": 1,
                "operations": [{
                    "op": "constraint.update",
                    "target": {"id": rig_id},
                    "id": constraint_id,
                    "set": {"influence": 0.5, "params": {"chain_count": 1}}
                }]
            }))
            .map_err(PotError::internal_json)?,
        )
        .map_err(|error| PotError::io(&error))?;
        crate::commands::apply::run(crate::cli::ApplyArgs {
            scene: project_path.clone(),
            file: batch_path,
            preview: None,
            size: None,
            dry_run: false,
        })?;
        let output = directory.path().join("pose-ik-edited.blend");
        crate::commands::export::run(&crate::cli::ExportArgs {
            scene: project_path,
            format: crate::cli::ExchangeFormat::Blend,
            out: output.clone(),
            allow_lossy: false,
            pack: false,
            overwrite: false,
            blender: Some(blender.clone()),
            view: None,
            context: crate::cli::Context {
                scene_id: None,
                view_layer: None,
                frame: None,
            },
        })?;
        let probe_path = directory.path().join("probe_pose_ik.py");
        fs::write(
            &probe_path,
            r#"
import bpy, sys, json
path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.open_mainfile(filepath=path)
rig = bpy.data.objects.get("IKRig")
constraint = rig.pose.bones["Tip"].constraints.get("Pose Reach") if rig else None
copy_constraint = rig.pose.bones["Root"].constraints.get("Pose Copy") if rig else None
print("POSE_IK " + json.dumps({
    "type": constraint.type if constraint else None,
    "target": constraint.target.name if constraint and constraint.target else None,
    "chain_count": constraint.chain_count if constraint else None,
    "influence": round(constraint.influence, 3) if constraint else None,
    "copy_type": copy_constraint.type if copy_constraint else None,
    "copy_target": copy_constraint.target.name if copy_constraint and copy_constraint.target else None,
    "copy_influence": round(copy_constraint.influence, 3) if copy_constraint else None,
}))
"#,
        )
        .map_err(|error| PotError::io(&error))?;
        let probe = Command::new(&blender)
            .arg("--background")
            .arg("--factory-startup")
            .arg("--disable-autoexec")
            .arg("--python-exit-code")
            .arg("3")
            .arg("--python")
            .arg(&probe_path)
            .arg("--")
            .arg(&output)
            .output()
            .map_err(|error| PotError::io(&error))?;
        if !probe.status.success() {
            return Err(PotError::with_details(
                ErrorCode::ExportFailed,
                "pose IK probe failed",
                child_details(&probe),
            ));
        }
        let stdout = String::from_utf8_lossy(&probe.stdout);
        let line = stdout
            .lines()
            .find(|line| line.starts_with("POSE_IK "))
            .ok_or_else(|| invalid_intermediate("pose IK probe output is missing"))?;
        let pose_ik: Value = serde_json::from_str(line.trim_start_matches("POSE_IK "))
            .map_err(PotError::internal_json)?;
        assert_eq!(pose_ik["type"], json!("IK"));
        assert_eq!(pose_ik["target"], json!("IKGoal"));
        assert_eq!(pose_ik["chain_count"], json!(1));
        assert_eq!(pose_ik["influence"], json!(0.5));
        assert_eq!(pose_ik["copy_type"], json!("COPY_TRANSFORMS"));
        assert_eq!(pose_ik["copy_target"], json!("IKGoal"));
        assert_eq!(pose_ik["copy_influence"], json!(0.0));
        Ok(())
    }
    #[expect(
        clippy::too_many_lines,
        reason = "the Blender compatibility test checks loss reporting and round-trip preservation"
    )]
    #[test]
    fn blender_unsupported_pose_constraints_remain_compatibility_only_with_loss() -> Result<()> {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let source = directory.path().join("unsupported-pose-constraint.blend");
        run_fixture_script(
            &blender,
            r#"
import bpy, sys
path = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
armature = bpy.data.armatures.new("UnsupportedRigData")
rig = bpy.data.objects.new("UnsupportedRig", armature)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
root = armature.edit_bones.new("Root")
root.head = (0, 0, 0)
root.tail = (0, 1, 0)
tip = armature.edit_bones.new("Tip")
tip.head = (0, 1, 0)
tip.tail = (0, 2, 0)
tip.parent = root
bpy.ops.object.mode_set(mode="OBJECT")
goal = bpy.data.objects.new("Anchor", None)
scene.collection.objects.link(goal)
constraint = rig.pose.bones["Tip"].constraints.new("ACTION")
constraint.name = "Preserve Unsupported Action"
bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False)
"#,
            &source,
        )?;
        let imported = import_blend(&source, Some(&blender))?;
        assert!(imported.losses.iter().any(|loss| {
            loss.feature_id == "blender.constraint.action"
                && loss.data_id.as_deref() == Some("Preserve Unsupported Action")
        }));
        let rig_id = imported.id_mappings["Object:UnsupportedRig"]
            .as_str()
            .ok_or_else(|| invalid_intermediate("unsupported rig mapping is missing"))?;
        let rig_node = test_node_ref(&imported.doc, rig_id)?;
        assert!(
            rig_node
                .constraints
                .iter()
                .all(|constraint| constraint.name != "Preserve Unsupported Action"),
            "unsupported pose constraint unexpectedly entered the editable graph"
        );
        let project = Project::init(directory.path().join("project"))?;
        let output = directory
            .path()
            .join("unsupported-pose-constraint-export.blend");
        let report = export_blend(
            &imported.doc,
            &project,
            &output,
            &ExportOptions {
                allow_lossy: true,
                pack: false,
                blender: Some(&blender),
                context: &crate::eval::EvaluationContext::default(),
            },
        )?;
        assert!(report.losses.iter().any(|loss| {
            loss.feature_id == "blender.constraint.action"
                && loss.data_id.as_deref() == Some("Preserve Unsupported Action")
        }));
        let probe_path = directory
            .path()
            .join("probe_unsupported_pose_constraint.py");
        fs::write(
            &probe_path,
            r#"
import bpy, sys, json
path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.open_mainfile(filepath=path)
rig = bpy.data.objects.get("UnsupportedRig")
constraint = rig.pose.bones["Tip"].constraints.get("Preserve Unsupported Action") if rig else None
print("UNSUPPORTED_POSE " + json.dumps({
    "type": constraint.type if constraint else None,
    "name": constraint.name if constraint else None,
}))
"#,
        )
        .map_err(|error| PotError::io(&error))?;
        let probe = Command::new(&blender)
            .arg("--background")
            .arg("--factory-startup")
            .arg("--disable-autoexec")
            .arg("--python-exit-code")
            .arg("3")
            .arg("--python")
            .arg(&probe_path)
            .arg("--")
            .arg(&output)
            .output()
            .map_err(|error| PotError::io(&error))?;
        if !probe.status.success() {
            return Err(PotError::with_details(
                ErrorCode::ExportFailed,
                "unsupported pose constraint probe failed",
                child_details(&probe),
            ));
        }
        let stdout = String::from_utf8_lossy(&probe.stdout);
        let line = stdout
            .lines()
            .find(|line| line.starts_with("UNSUPPORTED_POSE "))
            .ok_or_else(|| {
                invalid_intermediate("unsupported pose constraint probe output is missing")
            })?;
        let constraint: Value = serde_json::from_str(line.trim_start_matches("UNSUPPORTED_POSE "))
            .map_err(PotError::internal_json)?;
        assert_eq!(constraint["type"], json!("ACTION"));
        assert_eq!(constraint["name"], json!("Preserve Unsupported Action"));
        Ok(())
    }
    #[expect(
        clippy::too_many_lines,
        reason = "the test covers strict loss handling and lossy MovieClip reopen behavior"
    )]
    #[test]
    fn blender_authored_movie_clip_solve_reports_loss_and_exports_writable_tracking() -> Result<()>
    {
        let Some(blender) = optional_blender() else {
            return Ok(());
        };
        let directory = tempfile::tempdir().map_err(|error| PotError::io(&error))?;
        let media_path = directory.path().join("authored-clip.jpg");
        image::RgbImage::from_pixel(160, 90, image::Rgb([32, 96, 160]))
            .save(&media_path)
            .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
        let media_bytes = fs::read(&media_path).map_err(|error| PotError::io(&error))?;
        let media_id = Id::new("clip_media".to_owned())?;
        let mut doc = SceneDoc::new(uuid::Uuid::new_v4().to_string());
        doc.resources.insert(
            media_id.clone(),
            json!({
                "kind": "image",
                "uri": media_path.to_string_lossy(),
                "filename": "authored-clip.jpg",
                "hash": crate::hash::sha256(&media_bytes),
            }),
        );
        let world_points = [
            [-2.0, -1.0, 4.0],
            [1.0, -1.0, 5.0],
            [2.0, 2.0, 6.0],
            [-1.0, 3.0, 7.0],
            [3.0, -2.0, 8.0],
            [-3.0, 2.0, 9.0],
        ];
        let observations = world_points
            .iter()
            .map(|world| {
                let denominator = world[2] + 1.0;
                json!({
                    "world": world,
                    "image": [
                        (2.0 * world[0] + world[2]) / denominator,
                        (3.0 * world[1] + world[2]) / denominator,
                    ],
                })
            })
            .collect::<Vec<_>>();
        let mut operations = vec![
            json!({
                "op": "tracking.clip_create",
                "id": "clip_authored",
                "name": "AuthoredClip",
                "source": media_id,
                "width": 160,
                "height": 90,
            }),
            json!({
                "op": "tracking.set_camera_intrinsics",
                "id": "clip_authored",
                "set": {
                    "focal_mm": 48.0,
                    "sensor_width_mm": 36.0,
                    "principal": [0.02, -0.03],
                    "k1": 0.1,
                    "k2": -0.02,
                    "k3": 0.003,
                },
            }),
        ];
        let marker_coordinates = [
            [0.2, 0.25],
            [0.28, 0.31],
            [0.36, 0.37],
            [0.44, 0.43],
            [0.52, 0.49],
            [0.6, 0.55],
        ];
        for (index, coordinate) in marker_coordinates.into_iter().enumerate() {
            operations.push(json!({
                "op": "tracking.track_add",
                "id": "clip_authored",
                "track": format!("track_{index}"),
                "name": format!("Track_{index}"),
                "frame": 1,
                "co": coordinate,
            }));
        }
        operations.push(json!({
            "op": "tracking.solve_camera",
            "id": "clip_authored",
            "frame": 1,
            "observations": observations,
        }));
        doc = crate::ops::apply_batch(
            &doc,
            &json!({"schema_version": 1, "base_revision": 0, "operations": operations}),
        )?
        .doc;
        let clip_id = Id::new("clip_authored".to_owned())?;
        let authored_clip = doc
            .movie_clips
            .get(&clip_id)
            .ok_or_else(|| invalid_intermediate("authored MovieClip disappeared"))?;
        assert_eq!(authored_clip.tracking.reconstruction.cameras.len(), 1);
        assert!(authored_clip.tracking.reconstruction.is_valid);
        let active_scene = doc
            .scenes
            .get(&doc.active_scene)
            .ok_or_else(|| invalid_intermediate("authored clip scene disappeared"))?;
        assert!(active_scene.active_clip.is_none());

        let project = Project::init(directory.path().join("project"))?;
        let context = crate::eval::EvaluationContext::default();
        let strict_output = directory.path().join("strict.blend");
        let Err(strict_error) = export_blend(
            &doc,
            &project,
            &strict_output,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        ) else {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "strict MovieClip reconstruction export unexpectedly succeeded",
            ));
        };
        assert_eq!(strict_error.code, ErrorCode::UnrepresentableFeature);
        assert_eq!(
            strict_error.details["losses"][0]["feature_id"],
            json!("blender.movieclip.reconstruction")
        );
        assert_eq!(
            strict_error.details["losses"][0]["data_id"],
            json!("clip_authored")
        );

        let output = directory.path().join("lossy.blend");
        let report = export_blend(
            &doc,
            &project,
            &output,
            &ExportOptions {
                allow_lossy: true,
                pack: false,
                blender: Some(&blender),
                context: &context,
            },
        )?;
        assert!(report.files.iter().any(|file| file == &output));
        assert!(report.losses.iter().any(|loss| {
            loss.feature_id == "blender.movieclip.reconstruction"
                && loss.data_id.as_deref() == Some("clip_authored")
                && loss.reason.contains("read-only RNA")
        }));
        let reopened = import_blend(&output, Some(&blender))?;
        let reopened_clip = reopened
            .doc
            .movie_clips
            .values()
            .find(|clip| clip.name == "AuthoredClip")
            .ok_or_else(|| {
                invalid_intermediate(&format!(
                    "authored MovieClip did not reopen; names={:?}",
                    reopened
                        .doc
                        .movie_clips
                        .values()
                        .map(|clip| &clip.name)
                        .collect::<Vec<_>>()
                ))
            })?;
        assert_eq!(reopened_clip.tracking.tracks.len(), 6);
        assert!(
            reopened_clip
                .tracking
                .tracks
                .iter()
                .all(|track| track.markers.len() == 1)
        );
        let marker = reopened_clip
            .tracking
            .tracks
            .iter()
            .find(|track| track.name == "Track_0")
            .and_then(|track| track.markers.first())
            .ok_or_else(|| invalid_intermediate("authored MovieClip marker did not reopen"))?;
        assert!((marker.co[0] - 0.2).abs() <= 1.0e-6 && (marker.co[1] - 0.25).abs() <= 1.0e-6);
        assert!(
            (reopened_clip.tracking.camera.focal_mm - 48.0).abs() <= 1.0e-6,
            "reopened focal length is {}",
            reopened_clip.tracking.camera.focal_mm
        );
        assert!(
            (reopened_clip.tracking.camera.principal[0] - 0.02).abs() <= 1.0e-6
                && (reopened_clip.tracking.camera.principal[1] + 0.03).abs() <= 1.0e-6,
            "reopened principal point is {:?}",
            reopened_clip.tracking.camera.principal
        );
        assert!((reopened_clip.tracking.camera.k1 - 0.1).abs() <= 1.0e-6);
        assert_eq!(reopened_clip.tracking.camera.units, "MILLIMETERS");
        assert_eq!(reopened_clip.tracking.camera.distortion_model, "POLYNOMIAL");
        assert!((reopened_clip.tracking.camera.sensor_width_mm - 36.0).abs() <= 1.0e-6);
        assert!((reopened_clip.tracking.camera.k2 + 0.02).abs() <= 1.0e-6);
        assert!((reopened_clip.tracking.camera.k3 - 0.003).abs() <= 1.0e-6);
        assert!(reopened_clip.tracking.reconstruction.cameras.is_empty());
        assert!(!reopened_clip.tracking.reconstruction.is_valid);
        Ok(())
    }
}
