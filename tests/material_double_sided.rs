//! `double_sided` follows Blender 5.2.2: a new material is double-sided, every saved document
//! carries its value, and each exchange format that can express the flag keeps it. The Blender
//! reference values are checked by `material_double_sided_oracle.rs`.

use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    io::BufReader,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use potter_core::model::{Id, parse_scene};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn text(path: &Path) -> TestResult<&str> {
    path.to_str()
        .ok_or_else(|| format!("{} is not UTF-8", path.display()).into())
}

fn ok_json(output: &Output) -> TestResult<Value> {
    if !output.status.success() {
        return Err(format!("pot failed: {}", String::from_utf8_lossy(&output.stdout)).into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn init(scene: &Path) -> TestResult {
    ok_json(&pot().args(["init", text(scene)?, "--json"]).output()?)?;
    Ok(())
}

fn apply(scene: &Path, operations: &Value) -> TestResult {
    let file = scene.with_extension("operations.json");
    let batch = json!({"schema_version": 1, "base_revision": 0, "operations": operations});
    fs::write(&file, serde_json::to_vec(&batch)?)?;
    ok_json(
        &pot()
            .args(["apply", text(scene)?, "--file"])
            .arg(&file)
            .arg("--json")
            .output()?,
    )?;
    Ok(())
}

fn export_output(scene: &Path, format: &str, file: &Path, allow_lossy: bool) -> TestResult<Output> {
    let mut command = pot();
    command.args(["export", text(scene)?, "--format", format, "--out"]);
    command.arg(file);
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    Ok(command.arg("--json").output()?)
}

fn import_replace(scene: &Path, file: &Path, format: &str) -> TestResult {
    ok_json(
        &pot()
            .args(["import", text(scene)?, "--file"])
            .arg(file)
            .args([
                "--format",
                format,
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--json",
            ])
            .output()?,
    )?;
    Ok(())
}

fn scene_json(scene: &Path) -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(
        scene.join("scene.json"),
    )?)?)
}

/// `double_sided` of every material, keyed by material name.
fn double_sided_by_name(scene: &Path) -> TestResult<BTreeMap<String, bool>> {
    let document = scene_json(scene)?;
    document["materials"]
        .as_object()
        .ok_or("scene has no materials")?
        .values()
        .map(|material| {
            let name = material["name"].as_str().ok_or("material has no name")?;
            let value = material["double_sided"]
                .as_bool()
                .ok_or("material has no explicit double_sided")?;
            Ok((name.to_owned(), value))
        })
        .collect()
}

/// Two boxes: `Double` takes the default, `Single` is single-sided.
fn build_source(directory: &Path) -> TestResult<PathBuf> {
    let source = directory.join("source");
    init(&source)?;
    apply(
        &source,
        &json!([
            {"op": "material.create", "id": "double", "name": "Double"},
            {"op": "material.create", "id": "single", "name": "Single", "double_sided": false},
            {"op": "node.create", "id": "a", "kind": "box", "params": {"size": 1.0}, "material": "double"},
            {
                "op": "node.create",
                "id": "b",
                "kind": "box",
                "params": {"size": 1.0},
                "transform": {"translation": [2.0, 0.0, 0.0]},
                "material": "single"
            }
        ]),
    )?;
    Ok(source)
}

fn glb_json(bytes: &[u8]) -> TestResult<Value> {
    let length = u32::from_le_bytes(
        bytes
            .get(12..16)
            .ok_or("GLB has no JSON chunk")?
            .try_into()?,
    );
    let json = bytes
        .get(20..20 + length as usize)
        .ok_or("GLB JSON chunk is truncated")?;
    Ok(serde_json::from_slice(json)?)
}

#[test]
fn new_materials_are_double_sided_unless_the_operation_says_otherwise() -> TestResult {
    assert!(potter_core::model::Material::default().double_sided);
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    apply(
        &scene,
        &json!([
            {"op": "material.create", "id": "default", "name": "Default"},
            {"op": "material.create", "id": "single", "name": "Single", "double_sided": false},
            {"op": "material.create", "id": "double", "name": "Double", "double_sided": true}
        ]),
    )?;
    let expected = BTreeMap::from([
        ("Default".to_owned(), true),
        ("Single".to_owned(), false),
        ("Double".to_owned(), true),
    ]);
    assert_eq!(double_sided_by_name(&scene)?, expected);
    Ok(())
}

#[test]
fn glb_and_usda_and_fbx_round_trips_keep_double_sided() -> TestResult {
    let directory = tempdir()?;
    let source = build_source(directory.path())?;
    let expected = BTreeMap::from([("Double".to_owned(), true), ("Single".to_owned(), false)]);
    for (format, file) in [("glb", "box.glb"), ("usda", "box.usda"), ("fbx", "box.fbx")] {
        let exported = directory.path().join(file);
        let output = export_output(&source, format, &exported, false)?;
        ok_json(&output)?;
        let imported = directory.path().join(format!("imported-{format}"));
        init(&imported)?;
        import_replace(&imported, &exported, format)?;
        assert_eq!(double_sided_by_name(&imported)?, expected, "{format}");
    }
    Ok(())
}

#[test]
fn glb_writes_double_sided_for_every_material() -> TestResult {
    let directory = tempdir()?;
    let source = build_source(directory.path())?;
    let exported = directory.path().join("box.glb");
    ok_json(&export_output(&source, "glb", &exported, false)?)?;
    let document = glb_json(&fs::read(&exported)?)?;
    let materials = document["materials"]
        .as_array()
        .ok_or("GLB has no materials")?;
    for material in materials {
        let expected = material["name"] == "Double";
        assert_eq!(
            material["doubleSided"].as_bool(),
            Some(expected),
            "{}",
            material["name"]
        );
    }
    assert_eq!(materials.len(), 2);
    Ok(())
}

#[test]
fn obj_export_reports_single_sided_as_a_loss_and_imports_double_sided() -> TestResult {
    let directory = tempdir()?;
    let source = build_source(directory.path())?;
    let exported = directory.path().join("box.obj");
    let refused = export_output(&source, "obj", &exported, false)?;
    assert!(!refused.status.success(), "OBJ export must report the loss");
    assert!(
        String::from_utf8_lossy(&refused.stdout).contains("material.mtl.double_sided"),
        "{}",
        String::from_utf8_lossy(&refused.stdout)
    );
    ok_json(&export_output(&source, "obj", &exported, true)?)?;
    let imported = directory.path().join("imported-obj");
    init(&imported)?;
    import_replace(&imported, &exported, "obj")?;
    // MTL carries no two-sided flag, so Blender's OBJ importer makes every material double-sided.
    let expected = BTreeMap::from([("Double".to_owned(), true), ("Single".to_owned(), true)]);
    assert_eq!(double_sided_by_name(&imported)?, expected);
    Ok(())
}

#[test]
fn schema_one_documents_keep_their_explicit_double_sided() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/schema_v1_project");
    let bytes = fs::read(root.join("scene.json"))?;
    let raw: Value = serde_json::from_slice(&bytes)?;
    let parsed = parse_scene(&bytes, &root)?;
    let materials = raw["materials"]
        .as_object()
        .ok_or("fixture has no materials")?;
    assert!(!materials.is_empty());
    for (id, material) in materials {
        let expected = material["double_sided"]
            .as_bool()
            .ok_or("fixture material has no double_sided")?;
        assert_eq!(
            parsed.doc.materials[&Id::new(id)?].double_sided,
            expected,
            "{id}"
        );
    }
    Ok(())
}

/// Whether a centre pixel is a saturated red, the colour of the emissive test material.
fn is_red(pixel: [u8; 3]) -> bool {
    pixel[0] > 80 && pixel[0] > 2 * pixel[1] && pixel[0] > 2 * pixel[2]
}

fn png_centre(path: &Path) -> TestResult<[u8; 3]> {
    let mut decoder = png::Decoder::new(BufReader::new(fs::File::open(path)?));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let mut bytes = vec![0; reader.output_buffer_size().ok_or("PNG size is unknown")?];
    let info = reader.next_frame(&mut bytes)?;
    let (width, height) = (info.width as usize, info.height as usize);
    let channels = bytes.len() / (width * height);
    let offset = ((height / 2) * width + width / 2) * channels;
    Ok([bytes[offset], bytes[offset + 1], bytes[offset + 2]])
}

/// Previews one emissive plane in Beauty from the top and bottom views, and returns whether each
/// view shows its centre red. The plane's front face points up, so the bottom view sees its back.
fn plane_visible_from(double_sided: bool) -> TestResult<(bool, bool)> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let material = json!({
        "op": "material.create",
        "id": "red",
        "emission_color": [1.0, 0.0, 0.0],
        "emission_strength": 1.0,
        "double_sided": double_sided
    });
    apply(
        &scene,
        &json!([
            material,
            {"op": "node.create", "id": "plane", "kind": "plane", "params": {"size": 2.0}, "material": "red"}
        ]),
    )?;
    let out = directory.path().join("previews");
    ok_json(
        &pot()
            .arg("preview")
            .arg(&scene)
            .args([
                "--views",
                "top,bottom",
                "--size",
                "64",
                "--mode",
                "beauty",
                "--out",
            ])
            .arg(&out)
            .arg("--json")
            .output()?,
    )?;
    Ok((
        is_red(png_centre(&out.join("top.png"))?),
        is_red(png_centre(&out.join("bottom.png"))?),
    ))
}

#[test]
fn beauty_raster_culls_single_sided_back_faces_only() -> TestResult {
    assert_eq!(plane_visible_from(true)?, (true, true));
    let (top, bottom) = plane_visible_from(false)?;
    assert_ne!(
        top, bottom,
        "a single-sided plane must be visible from exactly one side"
    );
    Ok(())
}
