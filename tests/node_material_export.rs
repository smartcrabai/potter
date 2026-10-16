//! Blender 5.2.2 parity for node-material export (PR10).
//!
//! `tests/fixtures/material_graphs/build_probe.py` builds one UV sphere per material graph
//! (P01-P45). Blender's glTF exporter and `pot export` both export the same `.blend`. Every
//! material of both GLBs is normalized to one record (glTF defaults filled in, default extension
//! values dropped, textures decoded to pixels), and the records are compared under `EXPECTATIONS`.
//! The table follows the PR10 research note, §2.2 (probe results) and §4 (conversion rules).

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::{Map, Value, json};
use tempfile::tempdir;

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Float tolerance for normalized glTF values.
const FLOAT_TOLERANCE: f64 = 1.0e-4;
/// Tolerance for decoded 8-bit texel channels (1/255).
const PIXEL_TOLERANCE: i64 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Potter's normalized record equals Blender's.
    Equal,
    /// Potter deliberately differs from Blender (PR10 D3). `ignore` removes the diverging fields
    /// from the comparison, and `expected` gives the values Potter must write there instead.
    Divergence,
    /// glTF cannot express part of the graph. Potter must report `losses`; `ignore` removes the
    /// fields the lost part affects, because Blender's value there is not a reference.
    Loss,
}

#[derive(Clone, Copy)]
struct Expectation {
    /// Probe name, identical to the Blender and Potter material name.
    name: &'static str,
    kind: Kind,
    /// Expected `material.*` feature ids, compared as a set.
    losses: &'static [&'static str],
    /// Texture slot (glTF field name) and the channels (`R`, `G`, `B`, `A`) compared for it.
    /// Slots not listed compare all four channels.
    channels: &'static [(&'static str, &'static str)],
    /// Record paths (`/`-separated) removed from both sides before comparing.
    ignore: &'static [&'static str],
    /// Record path and the JSON value Potter must hold there (Divergence rows).
    expected: &'static [(&'static str, &'static str)],
    /// Image whose alpha channel the base color texture of Potter's record must carry.
    alpha_source: Option<&'static str>,
}

const BASE: Expectation = Expectation {
    name: "",
    kind: Kind::Equal,
    losses: &[],
    channels: &[],
    ignore: &[],
    expected: &[],
    alpha_source: None,
};

/// One row per probe in `build_probe.py`. Every probe appears exactly once (checked by
/// `expectation_table_covers_every_probe_exactly_once`).
const EXPECTATIONS: &[Expectation] = &[
    Expectation {
        name: "P01_Default",
        ..BASE
    },
    Expectation {
        name: "P02_BaseColorTexMapping",
        ..BASE
    },
    Expectation {
        name: "P03_AlphaTextureDirect",
        ..BASE
    },
    Expectation {
        name: "P04_AlphaClipGreaterThan",
        ..BASE
    },
    Expectation {
        name: "P05_AlphaClipRound",
        ..BASE
    },
    Expectation {
        name: "P06_AlphaConstant",
        ..BASE
    },
    Expectation {
        name: "P07_NormalMapStrength",
        ..BASE
    },
    Expectation {
        name: "P08_EmissionPrincipled",
        ..BASE
    },
    Expectation {
        name: "P09_EmissionTexture",
        ..BASE
    },
    Expectation {
        name: "P10_EmissionShaderSurface",
        ..BASE
    },
    Expectation {
        name: "P11_BackgroundUnlit",
        ..BASE
    },
    // Divergence: Blender drops the alpha of Mix(Fac=Alpha, Transparent, X). Potter exports alpha = Fac.
    Expectation {
        name: "P12_MixTransparentAlpha",
        kind: Kind::Divergence,
        ignore: &["alphaMode", "baseColorTexture"],
        expected: &[("alphaMode", "\"BLEND\"")],
        alpha_source: Some("probe_rgba"),
        ..BASE
    },
    Expectation {
        name: "P13_ORMSeparateColor",
        channels: &[("metallicRoughnessTexture", "GB")],
        ..BASE
    },
    Expectation {
        name: "P14_ORMWithGltfOcclusionGroup",
        channels: &[
            ("metallicRoughnessTexture", "GB"),
            ("occlusionTexture", "R"),
        ],
        ..BASE
    },
    Expectation {
        name: "P15_VertexColorMultiply",
        kind: Kind::Loss,
        losses: &["material.shader_graph.vertex_color"],
        ..BASE
    },
    // Divergence: sheen color is tint x weight (Blender emits the tint only).
    Expectation {
        name: "P16_TransmissionCoatSheenSpecular",
        kind: Kind::Divergence,
        ignore: &["extensions/KHR_materials_sheen/sheenColorFactor"],
        expected: &[(
            "extensions/KHR_materials_sheen/sheenColorFactor",
            "[0.5,0.0,0.0]",
        )],
        ..BASE
    },
    Expectation {
        name: "P17_BumpNode",
        kind: Kind::Loss,
        losses: &["material.shader_graph.normal_map"],
        ignore: &["normalTexture"],
        ..BASE
    },
    Expectation {
        name: "P18_NoiseBaseColor",
        kind: Kind::Loss,
        losses: &["material.shader_graph.unsupported_node"],
        ignore: &["baseColorFactor"],
        ..BASE
    },
    Expectation {
        name: "P19_MetallicMathFactor",
        channels: &[("metallicRoughnessTexture", "B")],
        ..BASE
    },
    Expectation {
        name: "P20_SecondUVLayer",
        kind: Kind::Loss,
        losses: &["material.texture_uv_set"],
        ignore: &["baseColorTexture/texCoord"],
        ..BASE
    },
    Expectation {
        name: "P21_RoughnessTextureDirect",
        channels: &[("metallicRoughnessTexture", "GB")],
        ..BASE
    },
    Expectation {
        name: "P22_BaseColorConstantMultiply",
        ..BASE
    },
    Expectation {
        name: "P23_ClearcoatTexture",
        channels: &[("clearcoatTexture", "R")],
        ..BASE
    },
    Expectation {
        name: "P24_MixEmissionShader",
        kind: Kind::Loss,
        losses: &["material.shader_graph.unrepresentable_surface"],
        // Divergence (D3): Potter weights the emission by the 0.5 mix factor; Blender adds it at full weight.
        expected: &[("emissiveFactor", "[0.5,0.5,0.5]")],
        ignore: &["emissiveFactor", "emissiveStrength"],
        ..BASE
    },
    Expectation {
        name: "P25_DoubleSided",
        ..BASE
    },
    // Divergence: Background strength 2 x color 0.5 = 1.0 is folded into the color (Blender drops strength).
    Expectation {
        name: "P26_BackgroundStrength2",
        kind: Kind::Divergence,
        ignore: &["baseColorFactor"],
        expected: &[("baseColorFactor", "[1.0,1.0,1.0,1.0]")],
        ..BASE
    },
    Expectation {
        name: "P27_NormalMapObjectSpace",
        kind: Kind::Loss,
        losses: &["material.shader_graph.normal_map"],
        ignore: &["normalTexture"],
        ..BASE
    },
    Expectation {
        name: "P28_AlphaClipOneMinusLessThan",
        ..BASE
    },
    Expectation {
        name: "P29_MappingRotationXY",
        kind: Kind::Loss,
        losses: &["material.shader_graph.transform"],
        ..BASE
    },
    Expectation {
        name: "P30_MappingTextureType",
        ..BASE
    },
    Expectation {
        name: "P31_ClosestMirror",
        ..BASE
    },
    Expectation {
        name: "P32_TwoImagesMixAdd",
        kind: Kind::Loss,
        losses: &["material.shader_graph.unsupported_node"],
        ignore: &["baseColorTexture"],
        ..BASE
    },
    Expectation {
        name: "P33_MixAddConstantBaseColor",
        kind: Kind::Loss,
        losses: &["material.shader_graph.unsupported_node"],
        ..BASE
    },
    Expectation {
        name: "P34_DisplacementOutput",
        kind: Kind::Loss,
        losses: &["material.shader_graph.displacement"],
        ..BASE
    },
    // Subsurface and thin film have no glTF representation; anisotropy does.
    Expectation {
        name: "P35_AnisotropySubsurfaceThinFilm",
        losses: &["material.shader_graph.unsupported_input"],
        ..BASE
    },
    Expectation {
        name: "P36_TwoPrincipledMix",
        kind: Kind::Loss,
        losses: &["material.shader_graph.unrepresentable_surface"],
        ignore: &["baseColorFactor", "metallicFactor", "roughnessFactor"],
        ..BASE
    },
    Expectation {
        name: "P37_GlassOnly",
        kind: Kind::Loss,
        losses: &["material.shader_graph.unrepresentable_surface"],
        ignore: &["baseColorFactor", "metallicFactor", "roughnessFactor"],
        ..BASE
    },
    Expectation {
        name: "P38_DiffuseOnlyTexture",
        kind: Kind::Loss,
        losses: &["material.shader_graph.unrepresentable_surface"],
        ignore: &[
            "baseColorTexture",
            "baseColorFactor",
            "metallicFactor",
            "roughnessFactor",
        ],
        ..BASE
    },
    Expectation {
        name: "P39_NormalMapNoStrengthChange",
        ..BASE
    },
    Expectation {
        name: "P40_VertexColorBaseColorOnly",
        kind: Kind::Loss,
        losses: &["material.shader_graph.vertex_color"],
        ..BASE
    },
    Expectation {
        name: "P41_AddShaderEmission",
        ..BASE
    },
    Expectation {
        name: "P42_AlphaTextureTimesConstant",
        ..BASE
    },
    Expectation {
        name: "P43_AvatarTypical",
        ..BASE
    },
    Expectation {
        name: "P44_MixBackgroundPrincipled",
        kind: Kind::Loss,
        losses: &["material.shader_graph.unrepresentable_surface"],
        ignore: &[
            "unlit",
            "baseColorFactor",
            "metallicFactor",
            "roughnessFactor",
        ],
        ..BASE
    },
    Expectation {
        name: "P45_AvatarEmission",
        ..BASE
    },
];

// ---------------------------------------------------------------------------
// Table and probe-source checks (no Blender needed)
// ---------------------------------------------------------------------------

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/material_graphs")
        .join(name)
}

/// Probe names from the `add("P..", builder)` calls of `build_probe.py`.
fn probe_names(source: &str) -> Vec<&str> {
    source
        .split("add(\"")
        .skip(1)
        .filter_map(|rest| rest.split('"').next())
        .collect()
}

#[test]
fn expectation_table_covers_every_probe_exactly_once() -> TestResult {
    let source = fs::read_to_string(fixture("build_probe.py"))?;
    let probes = probe_names(&source);
    let probe_set: BTreeSet<&str> = probes.iter().copied().collect();
    assert_eq!(
        probe_set.len(),
        probes.len(),
        "build_probe.py adds a probe name twice"
    );
    let mut table = BTreeSet::new();
    for row in EXPECTATIONS {
        assert!(table.insert(row.name), "duplicate table row {}", row.name);
    }
    assert_eq!(
        table, probe_set,
        "table rows and build_probe.py probes differ"
    );
    Ok(())
}

#[test]
fn expectation_rows_are_well_formed() {
    for row in EXPECTATIONS {
        match row.kind {
            Kind::Equal => assert!(
                row.ignore.is_empty() && row.expected.is_empty() && row.alpha_source.is_none(),
                "{}: Equal rows compare every field",
                row.name
            ),
            Kind::Divergence => assert!(
                !row.expected.is_empty(),
                "{}: Divergence rows state Potter's value",
                row.name
            ),
            Kind::Loss => assert!(
                !row.losses.is_empty(),
                "{}: Loss rows name their feature ids",
                row.name
            ),
        }
        for (slot, letters) in row.channels {
            assert!(
                !letters.is_empty() && letters.chars().all(|letter| "RGBA".contains(letter)),
                "{}: bad channel list `{letters}` for {slot}",
                row.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Process helpers
// ---------------------------------------------------------------------------

fn require_blender() -> bool {
    env::var_os("POTTER_REQUIRE_BLENDER").is_some_and(|value| !value.is_empty())
}

/// Returns Blender 5.2.x, or `None` (skip) when it is missing. With `POTTER_REQUIRE_BLENDER` set,
/// a missing or wrong-version Blender fails the test.
fn probe_blender() -> TestResult<Option<PathBuf>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping node-material probes: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(&blender);
    command.arg("--version");
    let output = process::run_guarded(command)?;
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if version.starts_with("Blender 5.2") {
        return Ok(Some(blender));
    }
    assert!(
        !require_blender(),
        "node-material probes need Blender 5.2.x, found `{version}`"
    );
    eprintln!("skipping node-material probes: found `{version}`, need Blender 5.2.x");
    Ok(None)
}

fn run_blender(blender: &Path, script: &Path, environment: Vec<(&str, OsString)>) -> TestResult {
    let mut command = Command::new(blender);
    command
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(script)
        .envs(environment);
    let output = process::run_guarded(command)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "Blender failed on {}: stdout={stdout} stderr={stderr}",
        script.display()
    );
    Ok(())
}

/// Builds `materials.blend` and the probe images into `out`. `only` restricts the probes.
fn build_probe(blender: &Path, out: &Path, only: Option<&str>) -> TestResult<PathBuf> {
    fs::create_dir_all(out)?;
    let mut environment = vec![("POTTER_PROBE_OUT", out.as_os_str().to_owned())];
    if let Some(names) = only {
        environment.push(("POTTER_PROBE_ONLY", OsString::from(names)));
    }
    run_blender(blender, &fixture("build_probe.py"), environment)?;
    Ok(out.join("materials.blend"))
}

fn export_gltf(blender: &Path, blend: &Path, out: &Path) -> TestResult {
    run_blender(
        blender,
        &fixture("export_gltf.py"),
        vec![
            ("POTTER_PROBE_BLEND", blend.as_os_str().to_owned()),
            ("POTTER_PROBE_OUT_GLB", out.as_os_str().to_owned()),
        ],
    )
}

fn pot(arguments: Vec<OsString>) -> TestResult<Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(arguments).arg("--json");
    process::run_guarded(command)
}

fn pot_ok(arguments: Vec<OsString>) -> TestResult<Value> {
    let output = pot(arguments)?;
    assert!(
        output.status.success(),
        "pot command failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn new_project(root: &Path) -> TestResult<PathBuf> {
    let project = root.join("project");
    pot_ok(vec!["init".into(), project.as_os_str().to_owned()])?;
    Ok(project)
}

fn import_blend(project: &Path, blend: &Path, blender: &Path, allow_lossy: bool) -> TestResult {
    let mut arguments: Vec<OsString> = vec![
        "import".into(),
        project.as_os_str().to_owned(),
        "--file".into(),
        blend.as_os_str().to_owned(),
        "--format".into(),
        "blend".into(),
        "--mode".into(),
        "replace".into(),
        "--base-revision".into(),
        "0".into(),
        "--blender".into(),
        blender.as_os_str().to_owned(),
    ];
    if allow_lossy {
        arguments.push("--allow-lossy".into());
    }
    pot_ok(arguments)?;
    Ok(())
}

fn export_glb(project: &Path, out: &Path, allow_lossy: bool) -> TestResult<Output> {
    let mut arguments: Vec<OsString> = vec![
        "export".into(),
        project.as_os_str().to_owned(),
        "--format".into(),
        "glb".into(),
        "--out".into(),
        out.as_os_str().to_owned(),
        "--overwrite".into(),
    ];
    if allow_lossy {
        arguments.push("--allow-lossy".into());
    }
    pot(arguments)
}

/// The `result.losses` array of a failed `pot export` envelope.
fn reported_losses(output: &Output) -> TestResult<Vec<Value>> {
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    Ok(envelope["result"]["losses"]
        .as_array()
        .cloned()
        .unwrap_or_default())
}

/// Groups `material.*` loss feature ids by material name. `data_id` is a material id in
/// `scene.json`; a value that is already a name is used as is.
fn attribute_losses(
    project: &Path,
    losses: &[Value],
) -> TestResult<BTreeMap<String, BTreeSet<String>>> {
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let names: BTreeMap<&str, &str> = scene["materials"]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(id, material)| Some((id.as_str(), material["name"].as_str()?)))
        .collect();
    let mut attributed: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for loss in losses {
        let feature = loss["feature_id"].as_str().unwrap_or_default();
        if !feature.starts_with("material.") {
            continue;
        }
        let data = loss["data_id"].as_str().unwrap_or_default();
        let name = names.get(data).copied().unwrap_or(data);
        attributed
            .entry(name.to_owned())
            .or_default()
            .insert(feature.to_owned());
    }
    Ok(attributed)
}

// ---------------------------------------------------------------------------
// GLB reading and normalization
// ---------------------------------------------------------------------------

struct Glb {
    json: Value,
    binary: Vec<u8>,
}

fn read_u32(bytes: &[u8], at: usize) -> TestResult<u32> {
    let raw: [u8; 4] = bytes
        .get(at..at + 4)
        .ok_or("GLB is truncated")?
        .try_into()?;
    Ok(u32::from_le_bytes(raw))
}

fn parse_glb(bytes: &[u8]) -> TestResult<Glb> {
    if bytes.get(0..4) != Some(b"glTF".as_slice()) {
        return Err("output is not a GLB".into());
    }
    let total = usize::try_from(read_u32(bytes, 8)?)?;
    let mut offset = 12;
    let mut json: Option<Value> = None;
    let mut binary = Vec::new();
    while offset < total {
        let length = usize::try_from(read_u32(bytes, offset)?)?;
        let kind = read_u32(bytes, offset + 4)?;
        let body = bytes
            .get(offset + 8..offset + 8 + length)
            .ok_or("GLB chunk exceeds the file")?;
        match kind {
            0x4E4F_534A => json = Some(serde_json::from_slice(body)?),
            0x004E_4942 => binary = body.to_vec(),
            _ => {}
        }
        offset += 8 + length;
    }
    Ok(Glb {
        json: json.ok_or("GLB has no JSON chunk")?,
        binary,
    })
}

fn scalar(value: &Value, default: f64) -> Value {
    json!(value.as_f64().unwrap_or(default))
}

fn factor(value: &Value, default: &[f64]) -> Value {
    Value::Array(
        default
            .iter()
            .enumerate()
            .map(|(index, fallback)| json!(value[index].as_f64().unwrap_or(*fallback)))
            .collect(),
    )
}

/// `KHR_texture_transform` values, or null when they are the identity.
fn transform(value: &Value) -> Value {
    if !value.is_object() {
        return Value::Null;
    }
    let offset = factor(&value["offset"], &[0.0, 0.0]);
    let rotation = scalar(&value["rotation"], 0.0);
    let scale = factor(&value["scale"], &[1.0, 1.0]);
    if offset == json!([0.0, 0.0]) && rotation == json!(0.0) && scale == json!([1.0, 1.0]) {
        Value::Null
    } else {
        json!({ "offset": offset, "rotation": rotation, "scale": scale })
    }
}

/// Decoded RGBA8 pixels of the image a texture samples.
fn pixels(glb: &Glb, texture: &Value) -> TestResult<Value> {
    let Some(source) = texture["source"].as_u64() else {
        return Ok(Value::Null);
    };
    let image = &glb.json["images"][usize::try_from(source)?];
    let view_index = image["bufferView"]
        .as_u64()
        .ok_or("image is not embedded in a bufferView")?;
    let view = &glb.json["bufferViews"][usize::try_from(view_index)?];
    let start = usize::try_from(view["byteOffset"].as_u64().unwrap_or(0))?;
    let length = usize::try_from(
        view["byteLength"]
            .as_u64()
            .ok_or("bufferView has no byteLength")?,
    )?;
    let bytes = glb
        .binary
        .get(start..start + length)
        .ok_or("image bytes exceed the BIN chunk")?;
    let decoded = image::load_from_memory(bytes)?.to_rgba8();
    Ok(json!({
        "width": decoded.width(),
        "height": decoded.height(),
        "rgba": decoded.as_raw(),
    }))
}

/// A textureInfo with its sampler (glTF defaults filled in), transform and decoded pixels.
/// `scale` names the scalar on the info object that is compared with its default (normal scale,
/// occlusion strength). Null when the info is absent.
fn texture_info(glb: &Glb, info: &Value, scale: Option<(&str, f64)>) -> TestResult<Value> {
    let Some(index) = info["index"].as_u64() else {
        return Ok(Value::Null);
    };
    let texture = &glb.json["textures"][usize::try_from(index)?];
    let sampler = match texture["sampler"].as_u64() {
        Some(sampler) => glb.json["samplers"][usize::try_from(sampler)?].clone(),
        None => Value::Null,
    };
    let mut record = Map::new();
    record.insert(
        "texCoord".into(),
        json!(info["texCoord"].as_u64().unwrap_or(0)),
    );
    record.insert(
        "sampler".into(),
        json!({
            "magFilter": sampler["magFilter"].as_u64().unwrap_or(9729),
            "minFilter": sampler["minFilter"].as_u64().unwrap_or(9987),
            "wrapS": sampler["wrapS"].as_u64().unwrap_or(10497),
            "wrapT": sampler["wrapT"].as_u64().unwrap_or(10497),
        }),
    );
    record.insert(
        "transform".into(),
        transform(&info["extensions"]["KHR_texture_transform"]),
    );
    record.insert("pixels".into(), pixels(glb, texture)?);
    if let Some((key, default)) = scale {
        record.insert(key.into(), scalar(&info[key], default));
    }
    Ok(Value::Object(record))
}

/// The normalized record of one material. Defaults are filled in, so a glTF that omits a default
/// and one that writes it compare equal.
fn normalize_material(glb: &Glb, material: &Value) -> TestResult<Value> {
    let pbr = &material["pbrMetallicRoughness"];
    let alpha_mode = material["alphaMode"].as_str().unwrap_or("OPAQUE");
    let mut record = Map::new();
    record.insert("alphaMode".into(), json!(alpha_mode));
    record.insert(
        "alphaCutoff".into(),
        if alpha_mode == "MASK" {
            scalar(&material["alphaCutoff"], 0.5)
        } else {
            Value::Null
        },
    );
    record.insert(
        "doubleSided".into(),
        json!(material["doubleSided"].as_bool().unwrap_or(false)),
    );
    record.insert(
        "unlit".into(),
        json!(material["extensions"]["KHR_materials_unlit"].is_object()),
    );
    record.insert(
        "baseColorFactor".into(),
        factor(&pbr["baseColorFactor"], &[1.0, 1.0, 1.0, 1.0]),
    );
    record.insert("metallicFactor".into(), scalar(&pbr["metallicFactor"], 1.0));
    record.insert(
        "roughnessFactor".into(),
        scalar(&pbr["roughnessFactor"], 1.0),
    );
    record.insert(
        "baseColorTexture".into(),
        texture_info(glb, &pbr["baseColorTexture"], None)?,
    );
    record.insert(
        "metallicRoughnessTexture".into(),
        texture_info(glb, &pbr["metallicRoughnessTexture"], None)?,
    );
    record.insert(
        "normalTexture".into(),
        texture_info(glb, &material["normalTexture"], Some(("scale", 1.0)))?,
    );
    record.insert(
        "occlusionTexture".into(),
        texture_info(glb, &material["occlusionTexture"], Some(("strength", 1.0)))?,
    );
    record.insert(
        "emissiveFactor".into(),
        factor(&material["emissiveFactor"], &[0.0, 0.0, 0.0]),
    );
    record.insert(
        "emissiveTexture".into(),
        texture_info(glb, &material["emissiveTexture"], None)?,
    );
    record.insert(
        "emissiveStrength".into(),
        scalar(
            &material["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"],
            1.0,
        ),
    );
    let mut extensions = Map::new();
    if let Some(map) = material["extensions"].as_object() {
        for (name, value) in map {
            if let Some(normalized) = normalize_extension(glb, name, value)? {
                extensions.insert(name.clone(), normalized);
            }
        }
    }
    record.insert("extensions".into(), Value::Object(extensions));
    Ok(Value::Object(record))
}

/// `None` when the extension is absent in effect (every value at its default), so an extension
/// written with defaults compares equal to none. Unknown extensions are compared as written.
fn normalize_extension(glb: &Glb, name: &str, value: &Value) -> TestResult<Option<Value>> {
    let (record, default) = match name {
        "KHR_materials_clearcoat" => (clearcoat(glb, value)?, clearcoat(glb, &Value::Null)?),
        "KHR_materials_sheen" => (sheen(glb, value)?, sheen(glb, &Value::Null)?),
        "KHR_materials_specular" => (specular(glb, value)?, specular(glb, &Value::Null)?),
        "KHR_materials_anisotropy" => (anisotropy(glb, value)?, anisotropy(glb, &Value::Null)?),
        "KHR_materials_transmission" => {
            (transmission(glb, value)?, transmission(glb, &Value::Null)?)
        }
        "KHR_materials_ior" => (ior(value), ior(&Value::Null)),
        "KHR_materials_unlit" | "KHR_materials_emissive_strength" => return Ok(None),
        _ => return Ok(Some(value.clone())),
    };
    Ok((record != default).then_some(record))
}

fn clearcoat(glb: &Glb, value: &Value) -> TestResult<Value> {
    Ok(json!({
        "clearcoatFactor": scalar(&value["clearcoatFactor"], 0.0),
        "clearcoatTexture": texture_info(glb, &value["clearcoatTexture"], None)?,
        "clearcoatRoughnessFactor": scalar(&value["clearcoatRoughnessFactor"], 0.0),
        "clearcoatRoughnessTexture": texture_info(glb, &value["clearcoatRoughnessTexture"], None)?,
        "clearcoatNormalTexture": texture_info(glb, &value["clearcoatNormalTexture"], Some(("scale", 1.0)))?,
    }))
}

fn sheen(glb: &Glb, value: &Value) -> TestResult<Value> {
    Ok(json!({
        "sheenColorFactor": factor(&value["sheenColorFactor"], &[0.0, 0.0, 0.0]),
        "sheenColorTexture": texture_info(glb, &value["sheenColorTexture"], None)?,
        "sheenRoughnessFactor": scalar(&value["sheenRoughnessFactor"], 0.0),
        "sheenRoughnessTexture": texture_info(glb, &value["sheenRoughnessTexture"], None)?,
    }))
}

fn specular(glb: &Glb, value: &Value) -> TestResult<Value> {
    Ok(json!({
        "specularFactor": scalar(&value["specularFactor"], 1.0),
        "specularTexture": texture_info(glb, &value["specularTexture"], None)?,
        "specularColorFactor": factor(&value["specularColorFactor"], &[1.0, 1.0, 1.0]),
        "specularColorTexture": texture_info(glb, &value["specularColorTexture"], None)?,
    }))
}

fn anisotropy(glb: &Glb, value: &Value) -> TestResult<Value> {
    Ok(json!({
        "anisotropyStrength": scalar(&value["anisotropyStrength"], 0.0),
        "anisotropyRotation": scalar(&value["anisotropyRotation"], 0.0),
        "anisotropyTexture": texture_info(glb, &value["anisotropyTexture"], None)?,
    }))
}

fn transmission(glb: &Glb, value: &Value) -> TestResult<Value> {
    Ok(json!({
        "transmissionFactor": scalar(&value["transmissionFactor"], 0.0),
        "transmissionTexture": texture_info(glb, &value["transmissionTexture"], None)?,
    }))
}

fn ior(value: &Value) -> Value {
    json!({ "ior": scalar(&value["ior"], 1.5) })
}

/// Normalized records of every material in a GLB, keyed by material name.
fn records(glb: &Glb) -> TestResult<BTreeMap<String, Value>> {
    let materials = glb.json["materials"]
        .as_array()
        .ok_or("GLB has no materials array")?;
    let mut out = BTreeMap::new();
    for material in materials {
        let name = material["name"]
            .as_str()
            .ok_or("GLB material has no name")?;
        out.insert(name.to_owned(), normalize_material(glb, material)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Channels (`R`, `G`, `B`, `A`) of the texture slot named `slot` that are compared.
fn channel_mask(channels: &[(&str, &str)], slot: &str) -> [bool; 4] {
    let letters = channels
        .iter()
        .find(|(name, _)| *name == slot)
        .map_or("RGBA", |(_, letters)| *letters);
    let mut mask = [false; 4];
    for (index, letter) in "RGBA".chars().enumerate() {
        mask[index] = letters.contains(letter);
    }
    mask
}

/// Appends one message per difference between `expected` (Blender) and `actual` (Potter).
/// Numbers compare within `FLOAT_TOLERANCE`; a `pixels` object compares per channel within
/// `PIXEL_TOLERANCE`, over the channels `channels` selects for the slot.
fn diff(
    path: &str,
    channels: &[(&str, &str)],
    expected: &Value,
    actual: &Value,
    out: &mut Vec<String>,
) {
    match (expected, actual) {
        (Value::Object(left), Value::Object(right)) => {
            if left.contains_key("pixels") || right.contains_key("pixels") {
                let slot = path.rsplit('/').next().unwrap_or_default();
                pixel_diff(
                    path,
                    channel_mask(channels, slot),
                    left.get("pixels"),
                    right.get("pixels"),
                    out,
                );
            }
            let keys: BTreeSet<&str> = left
                .keys()
                .chain(right.keys())
                .map(String::as_str)
                .filter(|key| *key != "pixels")
                .collect();
            for key in keys {
                let child = format!("{path}/{key}");
                match (left.get(key), right.get(key)) {
                    (Some(l), Some(r)) => diff(&child, channels, l, r, out),
                    (None, None) => {}
                    _ => out.push(format!("{child}: present on one side only")),
                }
            }
        }
        (Value::Array(left), Value::Array(right)) => {
            if left.len() != right.len() {
                out.push(format!(
                    "{path}: length {} differs from {}",
                    left.len(),
                    right.len()
                ));
                return;
            }
            for (index, (l, r)) in left.iter().zip(right).enumerate() {
                diff(&format!("{path}[{index}]"), channels, l, r, out);
            }
        }
        (Value::Number(_), Value::Number(_)) => {
            let l = expected.as_f64().unwrap_or(f64::NAN);
            let r = actual.as_f64().unwrap_or(f64::NAN);
            if (l - r).abs() > FLOAT_TOLERANCE || l.is_nan() || r.is_nan() {
                out.push(format!("{path}: expected {l} but got {r}"));
            }
        }
        _ => {
            if expected != actual {
                out.push(format!("{path}: expected {expected} but got {actual}"));
            }
        }
    }
}

fn pixel_diff(
    path: &str,
    mask: [bool; 4],
    expected: Option<&Value>,
    actual: Option<&Value>,
    out: &mut Vec<String>,
) {
    fn present(value: Option<&Value>) -> Option<&Value> {
        value.filter(|pixels| !pixels.is_null())
    }
    match (present(expected), present(actual)) {
        (None, None) => {}
        (Some(left), Some(right)) => {
            if left["width"] != right["width"] || left["height"] != right["height"] {
                out.push(format!("{path}: texture size differs"));
                return;
            }
            let (Some(l), Some(r)) = (left["rgba"].as_array(), right["rgba"].as_array()) else {
                out.push(format!("{path}: texture pixels are missing"));
                return;
            };
            if l.len() != r.len() {
                out.push(format!("{path}: texel count differs"));
                return;
            }
            let mut bad = 0_usize;
            let mut first = None;
            for (index, (x, y)) in l.iter().zip(r).enumerate() {
                if !mask[index % 4] {
                    continue;
                }
                let x = x.as_i64().unwrap_or(-1);
                let y = y.as_i64().unwrap_or(-1);
                if (x - y).abs() > PIXEL_TOLERANCE {
                    bad += 1;
                    if first.is_none() {
                        first = Some(index);
                    }
                }
            }
            if let Some(first) = first {
                out.push(format!(
                    "{path}: {bad} channel values differ, first at byte {first}"
                ));
            }
        }
        _ => out.push(format!("{path}: texture present on one side only")),
    }
}

/// Checks that Potter's base color texture carries the alpha of `source` (a probe PNG).
fn alpha_mismatch(potter: &Value, source: &Path) -> TestResult<Option<String>> {
    let Some(actual) = potter["baseColorTexture"]["pixels"]["rgba"].as_array() else {
        return Ok(Some("base color texture has no pixels".to_owned()));
    };
    let decoded = image::load_from_memory(&fs::read(source)?)?.to_rgba8();
    let expected = decoded.as_raw();
    if actual.len() != expected.len() {
        return Ok(Some(format!(
            "base color texture has {} bytes, {} has {}",
            actual.len(),
            source.display(),
            expected.len()
        )));
    }
    let bad = actual
        .iter()
        .enumerate()
        .filter(|(index, value)| {
            index % 4 == 3
                && value
                    .as_i64()
                    .is_none_or(|v| (v - i64::from(expected[*index])).abs() > PIXEL_TOLERANCE)
        })
        .count();
    Ok((bad > 0).then(|| {
        format!(
            "base color alpha differs from {} in {bad} texels",
            source.display()
        )
    }))
}

/// Removes the value at a `/`-separated path (no-op when a segment is missing).
fn remove_path(value: &mut Value, path: &str) {
    let mut parts: Vec<&str> = path.split('/').collect();
    let Some(last) = parts.pop() else {
        return;
    };
    let mut current = value;
    for part in parts {
        let Some(next) = current.get_mut(part) else {
            return;
        };
        current = next;
    }
    if let Some(object) = current.as_object_mut() {
        let _removed = object.remove(last);
    }
}

fn check_row(
    row: &Expectation,
    probe: &Path,
    attributed: &BTreeMap<String, BTreeSet<String>>,
    blender: &BTreeMap<String, Value>,
    potter: &BTreeMap<String, Value>,
) -> Vec<String> {
    let name = row.name;
    let mut failures = Vec::new();
    let expected_losses: BTreeSet<String> = row
        .losses
        .iter()
        .map(|feature| (*feature).to_owned())
        .collect();
    let actual_losses = attributed.get(name).cloned().unwrap_or_default();
    if actual_losses != expected_losses {
        failures.push(format!(
            "{name}: losses {actual_losses:?}, expected {expected_losses:?}"
        ));
    }
    let (Some(blender_record), Some(potter_record)) = (blender.get(name), potter.get(name)) else {
        failures.push(format!("{name}: missing from one of the GLBs"));
        return failures;
    };
    let mut expected = blender_record.clone();
    let mut actual = potter_record.clone();
    for &path in row.ignore {
        remove_path(&mut expected, path);
        remove_path(&mut actual, path);
    }
    let mut differences = Vec::new();
    diff("", row.channels, &expected, &actual, &mut differences);
    failures.extend(
        differences
            .into_iter()
            .map(|difference| format!("{name}: {difference}")),
    );
    for &(path, literal) in row.expected {
        match serde_json::from_str::<Value>(literal) {
            Ok(wanted) => {
                let got = potter_record
                    .pointer(&format!("/{path}"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let mut mismatch = Vec::new();
                diff(path, &[], &wanted, &got, &mut mismatch);
                failures.extend(
                    mismatch
                        .into_iter()
                        .map(|difference| format!("{name}: divergence {difference}")),
                );
            }
            Err(error) => {
                failures.push(format!("{name}: bad expected literal `{literal}`: {error}"));
            }
        }
    }
    if let Some(source) = row.alpha_source {
        let source_path = probe.join("images").join(format!("{source}.png"));
        match alpha_mismatch(potter_record, &source_path) {
            Ok(Some(problem)) => failures.push(format!("{name}: {problem}")),
            Ok(None) => {}
            Err(error) => failures.push(format!("{name}: alpha check failed: {error}")),
        }
    }
    failures
}

// ---------------------------------------------------------------------------
// Blender-gated tests
// ---------------------------------------------------------------------------

#[test]
fn graph_materials_match_blender_probe_export() -> TestResult {
    let Some(blender) = probe_blender()? else {
        return Ok(());
    };
    let root = tempdir()?;
    let probe = root.path().join("probe");
    let blend = build_probe(&blender, &probe, None)?;
    let blender_glb = root.path().join("blender.glb");
    export_gltf(&blender, &blend, &blender_glb)?;

    let project = new_project(root.path())?;
    // The probe scene contains deliberate losses, so the import accepts them.
    import_blend(&project, &blend, &blender, true)?;

    let strict = export_glb(&project, &root.path().join("strict.glb"), false)?;
    assert!(
        !strict.status.success(),
        "the probe scene exported without --allow-lossy although it contains losses"
    );
    let attributed = attribute_losses(&project, &reported_losses(&strict)?)?;
    // Blender color layers are reported per mesh until COLOR_0 export exists: only P15 and P40 have one.
    let mut colors: Vec<String> = reported_losses(&strict)?
        .iter()
        .filter(|loss| loss["feature_id"] == "mesh.color_attribute")
        .filter_map(|loss| loss["data_id"].as_str().map(str::to_owned))
        .collect();
    colors.sort();
    assert_eq!(colors.len(), 2, "color layers reported: {colors:?}");
    assert!(
        colors
            .iter()
            .all(|id| id.contains("p15_") || id.contains("p40_")),
        "color layers reported on unexpected meshes: {colors:?}"
    );

    let potter_glb = root.path().join("potter.glb");
    let lossy = export_glb(&project, &potter_glb, true)?;
    assert!(
        lossy.status.success(),
        "pot export --allow-lossy failed: stdout={} stderr={}",
        String::from_utf8_lossy(&lossy.stdout),
        String::from_utf8_lossy(&lossy.stderr)
    );

    let blender_glb_parsed = parse_glb(&fs::read(&blender_glb)?)?;
    let potter_glb_parsed = parse_glb(&fs::read(&potter_glb)?)?;
    let required = |glb: &Glb| -> BTreeSet<String> {
        glb.json["extensionsRequired"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|value| value.as_str().map(str::to_owned))
            .collect()
    };
    assert_eq!(
        required(&potter_glb_parsed),
        required(&blender_glb_parsed),
        "extensionsRequired differs from Blender's export"
    );
    let blender_records = records(&parse_glb(&fs::read(&blender_glb)?)?)?;
    let potter_records = records(&parse_glb(&fs::read(&potter_glb)?)?)?;
    let table: BTreeSet<&str> = EXPECTATIONS.iter().map(|row| row.name).collect();
    let blender_names: BTreeSet<&str> = blender_records.keys().map(String::as_str).collect();
    let potter_names: BTreeSet<&str> = potter_records.keys().map(String::as_str).collect();

    let mut failures = Vec::new();
    if blender_names != table {
        failures.push(format!(
            "Blender materials {blender_names:?} differ from the table"
        ));
    }
    if potter_names != table {
        failures.push(format!(
            "Potter materials {potter_names:?} differ from the table"
        ));
    }
    for name in attributed.keys() {
        if !table.contains(name.as_str()) {
            failures.push(format!("losses attributed to unknown material {name}"));
        }
    }
    for row in EXPECTATIONS {
        failures.extend(check_row(
            row,
            &probe,
            &attributed,
            &blender_records,
            &potter_records,
        ));
    }
    assert!(
        failures.is_empty(),
        "{} node-material mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

/// Smoke test: the default Principled sphere (P01, no edits) exports as glTF without
/// `--allow-lossy`.
/// Blender 5.2 mesh attributes that multi-material, sharp, seamed and hidden meshes carry export
/// without `--allow-lossy`, with the same primitives as Blender's own export. The UV Warp plane is
/// the V-flip oracle: its UVs come from a modifier (Blender-layer data, V up).
#[test]
fn blender_mesh_attributes_export_without_allow_lossy() -> TestResult {
    let Some(blender) = probe_blender()? else {
        return Ok(());
    };
    let root = tempdir()?;
    let blend = root.path().join("attributes.blend");
    run_blender(
        &blender,
        &fixture("build_attributes.py"),
        vec![("POTTER_ATTRIBUTES_BLEND", blend.as_os_str().to_owned())],
    )?;
    let blender_glb = root.path().join("blender.glb");
    export_gltf(&blender, &blend, &blender_glb)?;

    let project = new_project(root.path())?;
    import_blend(&project, &blend, &blender, false)?;
    let potter_glb = root.path().join("potter.glb");
    let output = export_glb(&project, &potter_glb, false)?;
    assert!(
        output.status.success(),
        "mesh attributes need --allow-lossy: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let blender_parts = primitive_summary(&parse_glb(&fs::read(&blender_glb)?)?)?;
    let potter_parts = primitive_summary(&parse_glb(&fs::read(&potter_glb)?)?)?;
    assert_eq!(
        potter_parts
            .iter()
            .map(|(m, _, _)| m.as_str())
            .collect::<BTreeSet<_>>(),
        blender_parts
            .iter()
            .map(|(m, _, _)| m.as_str())
            .collect::<BTreeSet<_>>(),
        "material slots per primitive differ from Blender's export"
    );
    assert_eq!(
        potter_parts.iter().map(|(_, n, _)| *n).sum::<usize>(),
        blender_parts.iter().map(|(_, n, _)| *n).sum::<usize>(),
        "triangle counts differ from Blender's export"
    );
    let warp_blender = texcoords_for(&parse_glb(&fs::read(&blender_glb)?)?, "AttrWarp")?;
    let warp_potter = texcoords_for(&parse_glb(&fs::read(&potter_glb)?)?, "AttrWarp")?;
    assert!(
        !warp_blender.is_empty(),
        "Blender exported no UVs for the warp plane"
    );
    assert_eq!(
        warp_potter, warp_blender,
        "UV Warp plane TEXCOORD_0 differs from Blender's export"
    );
    Ok(())
}

/// (material name, triangle count, primitive index) for every primitive of a GLB.
fn primitive_summary(glb: &Glb) -> TestResult<Vec<(String, usize, usize)>> {
    let mut rows = Vec::new();
    for (mesh_index, mesh) in glb.json["meshes"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        for (index, primitive) in mesh["primitives"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            let material = primitive["material"]
                .as_u64()
                .and_then(|value| {
                    glb.json["materials"][usize::try_from(value).ok()?]["name"].as_str()
                })
                .unwrap_or_default()
                .to_owned();
            let indices = primitive["indices"].as_u64().ok_or("indices missing")?;
            let count = glb.json["accessors"][usize::try_from(indices)?]["count"]
                .as_u64()
                .ok_or("index count missing")?;
            rows.push((
                material,
                usize::try_from(count)? / 3,
                mesh_index * 100 + index,
            ));
        }
    }
    Ok(rows)
}

/// `TEXCOORD_0` values, sorted, of every primitive that uses `material`, rounded to 1e-4.
fn texcoords_for(glb: &Glb, material: &str) -> TestResult<Vec<[i64; 2]>> {
    let mut values = Vec::new();
    for mesh in glb.json["meshes"].as_array().into_iter().flatten() {
        for primitive in mesh["primitives"].as_array().into_iter().flatten() {
            let name = primitive["material"].as_u64().and_then(|value| {
                glb.json["materials"][usize::try_from(value).ok()?]["name"].as_str()
            });
            if name != Some(material) {
                continue;
            }
            let accessor = primitive["attributes"]["TEXCOORD_0"]
                .as_u64()
                .ok_or("TEXCOORD_0 missing")?;
            let accessor = &glb.json["accessors"][usize::try_from(accessor)?];
            let view = &glb.json["bufferViews"][usize::try_from(
                accessor["bufferView"]
                    .as_u64()
                    .ok_or("bufferView missing")?,
            )?];
            let start = usize::try_from(view["byteOffset"].as_u64().unwrap_or(0))?
                + usize::try_from(accessor["byteOffset"].as_u64().unwrap_or(0))?;
            let count = usize::try_from(accessor["count"].as_u64().ok_or("count missing")?)?;
            let binary = &glb.binary;
            for vertex in 0..count {
                let at = start + vertex * 8;
                let u = f32::from_le_bytes(binary[at..at + 4].try_into()?);
                let v = f32::from_le_bytes(binary[at + 4..at + 8].try_into()?);
                values.push([
                    (f64::from(u) * 1.0e4).round() as i64,
                    (f64::from(v) * 1.0e4).round() as i64,
                ]);
            }
        }
    }
    values.sort_unstable();
    Ok(values)
}

#[test]
fn default_principled_sphere_exports_without_allow_lossy() -> TestResult {
    let Some(blender) = probe_blender()? else {
        return Ok(());
    };
    let root = tempdir()?;
    let blend = build_probe(&blender, &root.path().join("probe"), Some("P01_Default"))?;
    let project = new_project(root.path())?;
    import_blend(&project, &blend, &blender, false)?;
    let glb_path = root.path().join("default.glb");
    let output = export_glb(&project, &glb_path, false)?;
    assert!(
        output.status.success(),
        "default sphere export needs --allow-lossy: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let names: Vec<String> = records(&parse_glb(&fs::read(&glb_path)?)?)?
        .into_keys()
        .collect();
    assert_eq!(names, ["P01_Default"]);
    Ok(())
}

/// Smoke test: a textured avatar-like material (P43: base texture, alpha clip, tangent normal map)
/// and an emissive one (P45: P43 plus Principled emission) export as glTF without
/// `--allow-lossy`.
#[test]
fn avatar_textured_material_exports_without_allow_lossy() -> TestResult {
    let Some(blender) = probe_blender()? else {
        return Ok(());
    };
    let root = tempdir()?;
    let blend = build_probe(
        &blender,
        &root.path().join("probe"),
        Some("P43_AvatarTypical,P45_AvatarEmission"),
    )?;
    let project = new_project(root.path())?;
    import_blend(&project, &blend, &blender, false)?;
    let glb_path = root.path().join("avatar.glb");
    let output = export_glb(&project, &glb_path, false)?;
    assert!(
        output.status.success(),
        "avatar export needs --allow-lossy: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let normalized = records(&parse_glb(&fs::read(&glb_path)?)?)?;

    let avatar = normalized
        .get("P43_AvatarTypical")
        .ok_or("P43_AvatarTypical is missing")?;
    assert_eq!(avatar["alphaMode"], "MASK");
    assert!(
        !avatar["baseColorTexture"].is_null(),
        "base color texture is missing"
    );
    assert!(
        !avatar["normalTexture"].is_null(),
        "normal texture is missing"
    );

    let emission = normalized
        .get("P45_AvatarEmission")
        .ok_or("P45_AvatarEmission is missing")?;
    assert_eq!(emission["alphaMode"], "MASK");
    let mut mismatch = Vec::new();
    diff(
        "/emissiveFactor",
        &[],
        &json!([1.0, 0.5, 0.2]),
        &emission["emissiveFactor"],
        &mut mismatch,
    );
    diff(
        "/emissiveStrength",
        &[],
        &json!(1.5),
        &emission["emissiveStrength"],
        &mut mismatch,
    );
    assert!(mismatch.is_empty(), "P45 emission: {mismatch:?}");
    Ok(())
}
