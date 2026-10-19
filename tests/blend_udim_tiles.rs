use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;
use potter_core::{
    exchange::blend::import_blend,
    image::{
        ImageData, ImageExtension, ImageInterpolation, load_image_data, sample_with_extension_alpha,
    },
};
use serde_json::{Value, json};
use tempfile::tempdir;

/// Two 1x2 UDIM tiles (1001 base, 1002) in a TILED image named "UDIM Image", rendered through a Cycles emission
/// material. The plane's U starts at a per-variant offset: "external" and "uvtile" read tile files from disk through the
/// <UDIM> and <UVTILE> tokens, "packed" reads only the tiles packed into its .blend, "negative" spans U in (-1.25, -0.25),
/// and "base_plane" covers tile 1001 alone.
const FIXTURE_SCRIPT: &str = r##"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
TILES = {
    1001: [(0.1, 0.2, 0.3, 1.0), (0.4, 0.5, 0.6, 1.0)],
    1002: [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)],
}
# "single_tile" is a TILED image whose only file is tile 1001, so Blender lists just that tile.
OFFSETS = {"external": 1.0, "uvtile": 1.0, "packed": 1.0, "negative": -1.25, "base_plane": 0.0, "single_tile": 0.5}


def write_tile(filename, pixels):
    os.makedirs(os.path.dirname(os.path.join(root, filename)), exist_ok=True)
    image = bpy.data.images.new("tile", width=1, height=2, alpha=True)
    image.colorspace_settings.name = "Non-Color"
    image.pixels.foreach_set([channel for pixel in pixels for channel in pixel])
    image.filepath_raw = os.path.join(root, filename)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def build(variant, expected):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if variant == "uvtile":
        template, base = "udim.<UVTILE>.png", "udim.u1_v1.png"
    elif variant == "single_tile":
        template, base = "single/udim.<UDIM>.png", "single/udim.1001.png"
    else:
        template, base = "udim.<UDIM>.png", "udim.1001.png"
    image = bpy.data.images.load(os.path.join(root, base), check_existing=False)
    image.source = "TILED"
    image.filepath = os.path.join(root, template)
    image.reload()
    image.name = "UDIM Image"
    image.colorspace_settings.name = "Non-Color"
    if variant == "packed":
        image.pack()

    mesh = bpy.data.meshes.new("Plane")
    mesh.from_pydata([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)], [], [(0, 1, 2, 3)])
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        uv_layer.data[loop.index].uv = (OFFSETS[variant] + co.x, co.y)
    obj = bpy.data.objects.new("Plane", mesh)
    bpy.context.scene.collection.objects.link(obj)

    material = bpy.data.materials.new("UdimMaterial")
    tree = material.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Closest"
    uv_map = tree.nodes.new("ShaderNodeUVMap")
    uv_map.uv_map = "UVMap"
    tree.links.new(uv_map.outputs["UV"], texture.inputs["Vector"])
    tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.append(material)

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera = bpy.data.objects.new("Camera", camera_data)
    camera.location = (0.5, 0.5, 5.0)
    scene = bpy.context.scene
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.max_bounces = 0
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.render.resolution_x = 4
    scene.render.resolution_y = 4
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    settings = scene.render.image_settings
    settings.file_format = "OPEN_EXR"
    settings.color_depth = "32"
    settings.color_mode = "RGBA"
    output_path = os.path.join(root, "render_%s.exr" % variant)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    rendered = bpy.data.images.load(output_path, check_existing=False)
    values = [0.0] * (rendered.size[0] * rendered.size[1] * 4)
    rendered.pixels.foreach_get(values)
    expected[variant] = {
        "width": rendered.size[0],
        "height": rendered.size[1],
        "pixels": [values[index:index + 4] for index in range(0, len(values), 4)],
    }
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "udim_%s.blend" % variant), relative_remap=True)


expected = {}
for number, pixels in TILES.items():
    write_tile("udim.%d.png" % number, pixels)
    offset = number - 1001
    write_tile("udim.u%d_v%d.png" % (offset % 10 + 1, offset // 10 + 1), pixels)
write_tile("single/udim.1001.png", TILES[1001])
for variant in ["external", "uvtile", "packed", "negative", "base_plane", "single_tile"]:
    build(variant, expected)
with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as handle:
    json.dump(expected, handle)
print("FIXTURE_OK")
"##;

/// Reopens an exported .blend and reports the UDIM image's source, tile numbers and sizes, and packed tiles.
const REPORT_SCRIPT: &str = r##"
import bpy
import json
import sys

args = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.open_mainfile(filepath=args[0])
image = bpy.data.images["UDIM Image"]
report = {
    "source": image.source,
    "tiles": [[tile.number, tile.size[0], tile.size[1]] for tile in image.tiles],
    "packed": sorted(item.tile_number for item in image.packed_files),
}
with open(args[1], "w", encoding="utf-8") as handle:
    json.dump(report, handle)
print("REPORT_OK")
"##;

/// Renders a saved .blend with its own camera and render settings, and writes the pixels as JSON in Blender's row order.
const RENDER_SCRIPT: &str = r##"
import bpy
import json
import sys

args = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.open_mainfile(filepath=args[0])
bpy.context.scene.render.filepath = args[1]
bpy.ops.render.render(write_still=True)
rendered = bpy.data.images.load(args[1], check_existing=False)
values = [0.0] * (rendered.size[0] * rendered.size[1] * 4)
rendered.pixels.foreach_get(values)
report = {
    "width": rendered.size[0],
    "height": rendered.size[1],
    "pixels": [values[index:index + 4] for index in range(0, len(values), 4)],
}
with open(args[2], "w", encoding="utf-8") as handle:
    json.dump(report, handle)
print("RENDER_OK")
"##;

fn run_ok(command: &mut Command, label: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(output)
}

fn pot_ok(args: &[&str], label: &str) -> Result<Value, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(args).arg("--json");
    let output = run_ok(&mut command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn blender_ok(blender: &Path, args: &[&str], label: &str) -> Result<Output, Box<dyn Error>> {
    let mut command = Command::new(blender);
    command.args(args);
    run_ok(&mut command, label)
}

fn path_str(path: &Path) -> Result<&str, Box<dyn Error>> {
    path.to_str()
        .ok_or_else(|| std::io::Error::other("path is not UTF-8").into())
}

fn missing(what: &str) -> std::io::Error {
    std::io::Error::other(format!("{what} is missing"))
}

fn write_assets(assets: &[(String, Vec<u8>)], store: &Path) -> Result<(), Box<dyn Error>> {
    for (digest, bytes) in assets {
        let hex = digest
            .strip_prefix("sha256:")
            .ok_or_else(|| std::io::Error::other("asset digest is not sha256"))?;
        let path = store.join("assets").join("sha256").join(hex).join("blob");
        fs::create_dir_all(
            path.parent()
                .ok_or_else(|| std::io::Error::other("blob has no parent"))?,
        )?;
        fs::write(path, bytes)?;
    }
    Ok(())
}

/// Renders a saved .blend in Blender and returns the pixels as JSON.
fn render(
    blender: &Path,
    fixture: &Path,
    blend: &Path,
    label: &str,
) -> Result<Value, Box<dyn Error>> {
    let script = fixture.join("render_udim.py");
    fs::write(&script, RENDER_SCRIPT)?;
    let rendered = fixture.join(format!("render_{label}.json"));
    blender_ok(
        blender,
        &[
            "--background",
            "--factory-startup",
            "--threads",
            "1",
            "--python",
            path_str(&script)?,
            "--",
            path_str(blend)?,
            path_str(&fixture.join(format!("render_{label}.exr")))?,
            path_str(&rendered)?,
        ],
        "UDIM render",
    )?;
    Ok(serde_json::from_slice(&fs::read(&rendered)?)?)
}

/// Compares Potter's Image Texture sample with Blender's render at every pixel. Each pixel's U is `offset` plus its
/// column, its V is its row, and both enter the evaluator's lookup: the routing UV keeps Blender's V, and the lookup
/// subtracts the truncated offset from both coordinates (`shader/evaluator.rs`), so the extension sees the local UV.
fn assert_render_matches(
    data: &ImageData,
    rendered: &Value,
    label: &str,
    offset: f64,
    extension: ImageExtension,
) -> Result<(), Box<dyn Error>> {
    let width = rendered["width"]
        .as_u64()
        .ok_or_else(|| missing("rendered width"))?;
    let height = rendered["height"]
        .as_u64()
        .ok_or_else(|| missing("rendered height"))?;
    let rows = rendered["pixels"]
        .as_array()
        .ok_or_else(|| missing("rendered pixels"))?;
    assert_eq!(rows.len() as u64, width * height, "{label}: pixel count");
    for (index, pixel) in rows.iter().enumerate() {
        let index = index as u64;
        let u = offset + ((index % width) as f64 + 0.5) / width as f64;
        let v = ((index / width) as f64 + 0.5) / height as f64;
        let got = sample_with_extension_alpha(
            data,
            [u, v],
            true,
            ImageInterpolation::Closest,
            extension,
            false,
        );
        for channel in 0..4 {
            let want = pixel[channel]
                .as_f64()
                .ok_or_else(|| missing("rendered channel"))?;
            assert!(
                (got[channel] - want).abs() <= 1e-4,
                "{label}: pixel {index} (u {u}, v {v}) channel {channel}: potter {} blender {want}",
                got[channel]
            );
        }
    }
    Ok(())
}

fn find_udim_image<'a>(
    graph: &'a potter_core::exchange::ImportedGraph,
) -> Result<&'a potter_core::model::Image, Box<dyn Error>> {
    graph
        .doc
        .images
        .values()
        .find(|image| image.name == "UDIM Image")
        .ok_or_else(|| missing("UDIM Image").into())
}

fn check_variant(
    blender: &Path,
    fixture: &Path,
    expected: &Value,
    variant: &str,
) -> Result<(), Box<dyn Error>> {
    let blend = fixture.join(format!("udim_{variant}.blend"));
    let graph = import_blend(&blend, Some(blender))?;
    let image = find_udim_image(&graph)?;
    assert_eq!(
        (image.width, image.height),
        (1, 2),
        "{variant}: base tile size"
    );
    assert_eq!(image.tiles.len(), 1, "{variant}: UDIM tile 1002 is missing");
    let tile = &image.tiles[0];
    assert_eq!(
        (tile.number, tile.width, tile.height),
        (1002, 1, 2),
        "{variant}: tile 1002 metadata"
    );
    assert!(
        tile.blob
            .as_deref()
            .is_some_and(|blob| blob.starts_with("sha256:")),
        "{variant}: tile blob hash"
    );

    let store = fixture.join(format!("store_{variant}"));
    write_assets(&graph.assets, &store)?;
    let data = load_image_data(image, &store, ImageInterpolation::Closest)?;
    assert!(
        data.tiles.contains_key(&1002),
        "{variant}: loaded tile 1002"
    );
    assert_render_matches(
        &data,
        &expected[variant],
        variant,
        1.0,
        ImageExtension::Repeat,
    )?;

    let project = fixture.join(format!("project_{variant}"));
    let out = fixture.join(format!("roundtrip_{variant}.blend"));
    let report = fixture.join(format!("report_{variant}.json"));
    let init = pot_ok(&["init", path_str(&project)?], "project initialization")?;
    assert_eq!(init["ok"], true);
    let imported = pot_ok(
        &[
            "import",
            path_str(&project)?,
            "--file",
            path_str(&blend)?,
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
            path_str(blender)?,
        ],
        "UDIM Blender import",
    )?;
    assert_eq!(imported["ok"], true);
    let exported = pot_ok(
        &[
            "export",
            path_str(&project)?,
            "--format",
            "blend",
            "--out",
            path_str(&out)?,
            "--blender",
            path_str(blender)?,
        ],
        "UDIM Blender export",
    )?;
    assert_eq!(exported["ok"], true);
    let script = fixture.join("report_udim.py");
    fs::write(&script, REPORT_SCRIPT)?;
    blender_ok(
        blender,
        &[
            "--background",
            "--factory-startup",
            "--python",
            path_str(&script)?,
            "--",
            path_str(&out)?,
            path_str(&report)?,
        ],
        "UDIM round-trip reopen",
    )?;
    let reopened: Value = serde_json::from_slice(&fs::read(&report)?)?;
    assert_eq!(
        reopened,
        json!({
            "source": "TILED",
            "tiles": [[1001, 1, 2], [1002, 1, 2]],
            "packed": [1001, 1002],
        }),
        "{variant}: round-trip tiles"
    );
    // PNG tiles export as their original bytes, so the reimported blobs are the same content hashes.
    let reimported = import_blend(&out, Some(blender))?;
    let reimage = find_udim_image(&reimported)?;
    assert_eq!(reimage.blob, image.blob, "{variant}: round-trip base bytes");
    assert_eq!(
        reimage.tiles, image.tiles,
        "{variant}: round-trip tile bytes"
    );
    Ok(())
}

/// Imports the negative-U variant. Blender routes U in (-1, 0) to tile 1001 with the local U kept negative, and the
/// default Repeat extension wraps it; U below -1 truncates to a column with no tile and renders as the missing colour.
fn check_negative(blender: &Path, fixture: &Path, expected: &Value) -> Result<(), Box<dyn Error>> {
    let graph = import_blend(&fixture.join("udim_negative.blend"), Some(blender))?;
    let image = find_udim_image(&graph)?;
    let store = fixture.join("store_negative");
    write_assets(&graph.assets, &store)?;
    let data = load_image_data(image, &store, ImageInterpolation::Closest)?;
    assert_render_matches(
        &data,
        &expected["negative"],
        "negative",
        -1.25,
        ImageExtension::Repeat,
    )
}

/// Imports the external variant after its tile-1002 file is removed. The listed tile keeps its number with the missing
/// state, is reported as a loss, and renders as Blender's missing colour.
fn check_missing_tile(blender: &Path, fixture: &Path) -> Result<(), Box<dyn Error>> {
    let blend = fixture.join("udim_external.blend");
    let graph = import_blend(&blend, Some(blender))?;
    let image = find_udim_image(&graph)?;
    assert_eq!(
        image.tiles.len(),
        1,
        "the listed missing tile 1002 was dropped"
    );
    let tile = &image.tiles[0];
    assert_eq!(
        (tile.number, tile.width, tile.height, &tile.blob),
        (1002, 0, 0, &None),
        "missing tile keeps its number with the missing state"
    );
    assert!(
        graph
            .losses
            .iter()
            .any(|loss| loss.feature_id == "blender.image.udim_tile_missing"
                && loss.data_id.as_deref() == Some("UDIM Image")),
        "missing tile 1002 was not reported: {:?}",
        graph.losses
    );

    let store = fixture.join("store_missing");
    write_assets(&graph.assets, &store)?;
    let data = load_image_data(image, &store, ImageInterpolation::Closest)?;
    assert!(
        !data.tiles.contains_key(&1002),
        "missing tile 1002 has pixels"
    );
    let rendered = render(blender, fixture, &blend, "missing_tile")?;
    assert_render_matches(
        &data,
        &rendered,
        "missing tile",
        1.0,
        ImageExtension::Repeat,
    )
}

/// Imports the base-plane variant after its base file is removed (tile 1002 is already gone). The image is kept with its
/// base in the missing state and its listed tile kept as missing, both reported as losses. Blender renders the plane's
/// tile-1001 lookups as the missing colour, and Potter's sampler matches at every pixel.
fn check_missing_base(blender: &Path, fixture: &Path) -> Result<(), Box<dyn Error>> {
    let blend = fixture.join("udim_base_plane.blend");
    let graph = import_blend(&blend, Some(blender))?;
    let image = find_udim_image(&graph)?;
    assert_eq!(
        (image.width, image.height),
        (0, 0),
        "a missing base has no size"
    );
    assert_eq!(image.blob, None, "a missing base has no blob");
    assert_eq!(image.tiles.len(), 1, "the listed tile 1002 was dropped");
    assert_eq!(
        image.tiles[0].blob, None,
        "the listed missing tile has a blob"
    );
    for number in [1001, 1002] {
        assert!(
            graph
                .losses
                .iter()
                .any(|loss| loss.feature_id == "blender.image.udim_tile_missing"
                    && loss.data_id.as_deref() == Some("UDIM Image")
                    && loss.reason.contains(&format!("tile {number} "))),
            "tile {number} was not reported: {:?}",
            graph.losses
        );
    }

    let store = fixture.join("store_base_plane");
    write_assets(&graph.assets, &store)?;
    let data = load_image_data(image, &store, ImageInterpolation::Closest)?;
    assert_eq!((data.width, data.height), (0, 0), "missing base has pixels");
    let rendered = render(blender, fixture, &blend, "base_plane")?;
    assert_render_matches(
        &data,
        &rendered,
        "missing base",
        0.0,
        ImageExtension::Repeat,
    )
}

/// Imports the single-tile variant: a TILED image whose only file is tile 1001. Blender keeps the TILED source, so a
/// lookup past tile 1001 is missing (magenta) while the base still samples, and Potter keeps the `tiled` flag and
/// routes the same way. The plane spans U from 0.5 to 1.5, so both cases appear in one render.
fn check_single_tile(
    blender: &Path,
    fixture: &Path,
    expected: &Value,
) -> Result<(), Box<dyn Error>> {
    let graph = import_blend(&fixture.join("udim_single_tile.blend"), Some(blender))?;
    let image = find_udim_image(&graph)?;
    assert!(image.tiled, "the TILED source was not kept");
    assert!(
        image.tiles.is_empty(),
        "tile 1001 is stored as a listed tile"
    );
    assert_eq!((image.width, image.height), (1, 2), "base tile size");
    let store = fixture.join("store_single_tile");
    write_assets(&graph.assets, &store)?;
    let data = load_image_data(image, &store, ImageInterpolation::Closest)?;
    assert_render_matches(
        &data,
        &expected["single_tile"],
        "single tile",
        0.5,
        ImageExtension::Repeat,
    )
}

/// Imports the single-tile blend after its only tile file is removed: a TILED image with no file. Blender lists tile
/// 1001 with size 0 and draws every lookup as the missing colour (probed: magenta for the whole plane), and Potter keeps
/// the TILED record with a zero-size base, which the sampler reads as the missing colour too.
fn check_tiled_without_files(blender: &Path, fixture: &Path) -> Result<(), Box<dyn Error>> {
    let blend = fixture.join("udim_single_tile.blend");
    let graph = import_blend(&blend, Some(blender))?;
    let image = find_udim_image(&graph)?;
    assert!(image.tiled, "the TILED source was not kept");
    assert!(image.tiles.is_empty(), "a tile with no file was imported");
    assert_eq!(
        (image.width, image.height),
        (0, 0),
        "a missing base has no size"
    );
    assert!(image.blob.is_none(), "a missing base has a blob");
    let store = fixture.join("store_tiled_without_files");
    write_assets(&graph.assets, &store)?;
    let data = load_image_data(image, &store, ImageInterpolation::Closest)?;
    let rendered = render(blender, fixture, &blend, "tiled_without_files")?;
    assert_render_matches(
        &data,
        &rendered,
        "tiled without files",
        0.5,
        ImageExtension::Repeat,
    )
}

/// Exports a UDIM image with no loadable tile through `pot`. The strict export refuses it, the lossy export writes a TILED
/// placeholder with its missing base and tiles reported as losses, and the placeholder reopens as the same TILED record
/// with a missing base.
fn check_missing_export(
    blender: &Path,
    fixture: &Path,
    blend_name: &str,
) -> Result<(), Box<dyn Error>> {
    let blend = fixture.join(blend_name);
    let stem = blend_name.trim_end_matches(".blend");
    let project = fixture.join(format!("project_export_{stem}"));
    let out = fixture.join(format!("export_{stem}.blend"));
    let init = pot_ok(&["init", path_str(&project)?], "project initialization")?;
    assert_eq!(init["ok"], true);
    pot_ok(
        &[
            "import",
            path_str(&project)?,
            "--file",
            path_str(&blend)?,
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
            path_str(blender)?,
            "--allow-lossy",
        ],
        "missing-image import",
    )?;
    let strict = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args([
            "export",
            path_str(&project)?,
            "--format",
            "blend",
            "--out",
            path_str(&out)?,
            "--blender",
            path_str(blender)?,
        ])
        .output()?;
    assert!(
        !strict.status.success(),
        "{stem}: strict export accepted a missing UDIM image"
    );
    let lossy = pot_ok(
        &[
            "export",
            path_str(&project)?,
            "--format",
            "blend",
            "--out",
            path_str(&out)?,
            "--blender",
            path_str(blender)?,
            "--allow-lossy",
        ],
        "missing-image lossy export",
    )?;
    assert_eq!(lossy["ok"], true, "{stem}: lossy export failed");
    let reopened = import_blend(&out, Some(blender))?;
    let image = find_udim_image(&reopened)?;
    assert!(image.tiled, "{stem}: the TILED source was not exported");
    assert_eq!(
        (image.width, image.height),
        (0, 0),
        "{stem}: missing base reopened with a size"
    );
    assert!(
        image.blob.is_none(),
        "{stem}: missing base reopened with a blob"
    );
    assert!(
        reopened
            .losses
            .iter()
            .any(|loss| loss.feature_id == "blender.image.udim_tile_missing"),
        "{stem}: the reopened missing image reports no loss"
    );
    Ok(())
}

#[test]
fn blender_udim_tiles_import_sample_and_round_trip() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender UDIM tile test: Blender is unavailable");
        return Ok(());
    };
    let root = tempdir()?;
    let fixture = fs::canonicalize(root.path())?;
    let script = fixture.join("make_udim.py");
    fs::write(&script, FIXTURE_SCRIPT)?;
    blender_ok(
        &blender,
        &[
            "--background",
            "--factory-startup",
            "--threads",
            "1",
            "--python",
            path_str(&script)?,
            "--",
            path_str(&fixture)?,
        ],
        "UDIM fixture generation",
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(fixture.join("expected.json"))?)?;

    check_variant(&blender, &fixture, &expected, "external")?;
    check_variant(&blender, &fixture, &expected, "uvtile")?;
    check_negative(&blender, &fixture, &expected)?;
    check_single_tile(&blender, &fixture, &expected)?;
    fs::remove_file(fixture.join("single/udim.1001.png"))?;
    check_tiled_without_files(&blender, &fixture)?;
    fs::remove_file(fixture.join("udim.1002.png"))?;
    check_missing_tile(&blender, &fixture)?;
    fs::remove_file(fixture.join("udim.1001.png"))?;
    check_missing_base(&blender, &fixture)?;
    check_missing_export(&blender, &fixture, "udim_single_tile.blend")?;
    check_missing_export(&blender, &fixture, "udim_base_plane.blend")?;
    // The packed variant must read only its packed tiles, so the remaining tile files are removed first.
    for name in ["udim.u1_v1.png", "udim.u2_v1.png"] {
        fs::remove_file(fixture.join(name))?;
    }
    check_variant(&blender, &fixture, &expected, "packed")?;
    Ok(())
}
