#![expect(
    clippy::unwrap_used,
    reason = "Blender integration fixtures use fixed valid data"
)]

use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/process.rs"]
mod process;

use blender_file::blender_executable;
use serde_json::{Value, json};
use tempfile::tempdir;

const FIXTURE_SCRIPT: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

mesh = bpy.data.meshes.new("Texture Carrier Mesh")
mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (0, 1, 0)], [], [(0, 1, 2)])
carrier = bpy.data.objects.new("Texture Carrier", mesh)
scene.collection.objects.link(carrier)

texture_specs = [
    ("Clouds", "CLOUDS", {"noise_scale": 0.37, "noise_depth": 3}),
    ("Marble", "MARBLE", {"noise_scale": 0.42, "noise_depth": 5,
                          "turbulence": 3.25, "marble_type": "SHARPER"}),
    ("Wood", "WOOD", {"noise_scale": 0.36, "turbulence": 2.75,
                      "wood_type": "RINGNOISE", "noise_basis_2": "TRI",
                      "noise_type": "HARD_NOISE", "noise_basis": "IMPROVED_PERLIN"}),
    ("Magic", "MAGIC", {"turbulence": 3.75, "noise_depth": 4}),
    ("Blend", "BLEND", {"progression": "DIAGONAL", "use_flip_axis": "VERTICAL"}),
    ("Stucci", "STUCCI", {"turbulence": 3.5, "noise_scale": 0.44,
                          "noise_type": "HARD_NOISE", "stucci_type": "WALL_OUT"}),
    ("Noise", "NOISE", {}),
    ("Distorted Noise", "DISTORTED_NOISE", {
        "distortion": 2.4, "noise_scale": 0.33, "noise_basis": "IMPROVED_PERLIN",
        "noise_distortion": "ORIGINAL_PERLIN"}),
    ("Voronoi", "VORONOI", {
        "weight_1": 0.7, "weight_2": 0.2, "weight_3": 0.3, "weight_4": 0.4,
        "minkovsky_exponent": 1.5, "distance_metric": "MANHATTAN",
        "color_mode": "POSITION_OUTLINE_INTENSITY", "noise_intensity": 0.8, "noise_scale": 0.27}),
    ("Musgrave", "MUSGRAVE", {
        "musgrave_type": "RIDGED_MULTIFRACTAL", "dimension_max": 1.2,
        "lacunarity": 2.3, "offset": 0.8, "gain": 1.1, "noise_intensity": 0.75,
        "noise_scale": 0.22, "noise_basis": "IMPROVED_PERLIN"}),
]
textures = []
for label, texture_type, values in texture_specs:
    texture = bpy.data.textures.new(label + " Legacy", type=texture_type)
    for key, value in values.items():
        setattr(texture, key, value)
    textures.append((label + " Displace", texture))

image = bpy.data.images.new("Legacy Image", width=2, height=2, alpha=True)
image.pixels = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
image_path = os.path.join(root, "texture.png")
image.filepath_raw = image_path
image.file_format = "PNG"
image.save()
bpy.data.images.remove(image)
image = bpy.data.images.load(image_path, check_existing=False)
image.name = "Legacy Image"
image_texture = bpy.data.textures.new("Image Legacy", type="IMAGE")
image_texture.image = image
image_texture.extension = "CLIP"
image_texture.use_color_ramp = True
image_texture.image_user.frame_offset = 3
for key, value in {
        "use_clamp": False, "saturation": 0.8,
        "factor_red": 0.7, "factor_green": 0.8, "factor_blue": 0.9,
        "intensity": 1.1,
}.items():
    if hasattr(image_texture, key):
        setattr(image_texture, key, value)
extra_ramp_element = image_texture.color_ramp.elements.new(0.55)
extra_ramp_element.color = (0.1, 0.8, 0.2, 1.0)
image_texture_filtered = bpy.data.textures.new("Image Legacy Filtered", type="IMAGE")
image_texture_filtered.image = image
image_texture_filtered.use_interpolation = False
image_texture_filtered.contrast = 1.25

fastpath_image = bpy.data.images.new("Fastpath Image", width=1, height=1, alpha=True)
fastpath_image.pixels = [0.25, 0.5, 0.75, 1.0]
fastpath_image_path = os.path.join(root, "fastpath.png")
fastpath_image.filepath_raw = fastpath_image_path
fastpath_image.file_format = "PNG"
fastpath_image.save()
bpy.data.images.remove(fastpath_image)
fastpath_image = bpy.data.images.load(fastpath_image_path, check_existing=False)
fastpath_image.name = "Fastpath Image"
fastpath_texture = bpy.data.textures.new("Fastpath Legacy", type="IMAGE")
fastpath_texture.image = fastpath_image

for name, texture in textures + [
        ("Image Displace", image_texture),
        ("Image Filtered Displace", image_texture_filtered),
        ("Fastpath Displace", fastpath_texture)]:
    modifier = carrier.modifiers.new(name, "DISPLACE")
    modifier.texture = texture
settings = {
    "CLOUDS": ("noise_scale", "noise_depth", "noise_basis", "noise_type", "cloud_type"),
    "MARBLE": ("noise_scale", "turbulence", "noise_depth", "noise_type", "marble_type",
               "noise_basis", "noise_basis_2"),
    "WOOD": ("noise_scale", "turbulence", "noise_basis", "noise_type", "wood_type",
             "noise_basis_2"),
    "MAGIC": ("turbulence", "noise_depth"),
    "BLEND": ("progression", "use_flip_axis"),
    "STUCCI": ("turbulence", "noise_basis", "noise_scale", "noise_type", "stucci_type"),
    "NOISE": ("noise_scale", "noise_basis", "noise_type"),
    "DISTORTED_NOISE": ("distortion", "noise_scale", "noise_basis", "noise_distortion"),
    "VORONOI": ("weight_1", "weight_2", "weight_3", "weight_4", "minkovsky_exponent",
                "distance_metric", "color_mode", "noise_intensity", "noise_scale"),
    "MUSGRAVE": ("musgrave_type", "dimension_max", "lacunarity", "offset", "gain",
                 "noise_intensity", "noise_scale", "noise_basis"),
    "IMAGE": ("extension", "use_interpolation", "use_color_ramp", "use_clamp",
              "contrast", "saturation", "factor_red", "factor_green",
              "factor_blue", "intensity"),
}
def texture_state(texture):
    fields = settings.get(texture.type, ())
    result = {key: getattr(texture, key) for key in fields if hasattr(texture, key)}
    if texture.type == "IMAGE":
        if texture.color_ramp is not None:
            ramp = texture.color_ramp
            result["color_ramp"] = {
                "color_mode": ramp.color_mode,
                "hue_interpolation": ramp.hue_interpolation,
                "interpolation": ramp.interpolation,
                "elements": [
                    {"position": float(element.position),
                     "color": [float(value) for value in element.color]}
                    for element in ramp.elements
                ],
            }
        result["image_user"] = {"frame_offset": texture.image_user.frame_offset}
    return result
def modifier_state():
    return {
        modifier.name: {
            "texture_name": modifier.texture.name_full,
            "texture_type": modifier.texture.type,
            "inline_texture": (
                modifier.texture.type != "IMAGE"
                or modifier.texture in (image_texture, image_texture_filtered)),
            "settings": texture_state(modifier.texture),
            "image_name": (modifier.texture.image.name_full
                           if modifier.texture.type == "IMAGE" and modifier.texture.image else None),
        }
        for modifier in carrier.modifiers
    }

expected = modifier_state()
with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as handle:
    json.dump(expected, handle, sort_keys=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "legacy-textures.blend"),
                            relative_remap=False)
"#;

const INSPECT_SCRIPT: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
carrier = bpy.data.objects["Texture Carrier"]
settings = {
    "CLOUDS": ("noise_scale", "noise_depth", "noise_basis", "noise_type", "cloud_type"),
    "MARBLE": ("noise_scale", "turbulence", "noise_depth", "noise_type", "marble_type",
               "noise_basis", "noise_basis_2"),
    "WOOD": ("noise_scale", "turbulence", "noise_depth", "noise_type", "wood_type",
             "noise_basis_2"),
    "MAGIC": ("turbulence", "noise_depth"),
    "BLEND": ("progression", "use_flip_axis"),
    "STUCCI": ("turbulence", "noise_basis", "noise_scale", "noise_type", "stucci_type"),
    "NOISE": ("noise_scale", "noise_basis", "noise_type"),
    "DISTORTED_NOISE": ("distortion", "noise_scale", "noise_basis", "noise_distortion"),
    "VORONOI": ("weight_1", "weight_2", "weight_3", "weight_4", "minkovsky_exponent",
                "distance_metric", "color_mode", "noise_intensity", "noise_scale"),
    "MUSGRAVE": ("musgrave_type", "dimension_max", "lacunarity", "offset", "gain",
                 "noise_intensity", "noise_scale", "noise_basis"),
    "IMAGE": ("extension", "use_interpolation", "use_color_ramp", "use_clamp",
              "contrast", "saturation", "factor_red", "factor_green",
              "factor_blue", "intensity"),
}
def texture_state(texture):
    fields = settings.get(texture.type, ())
    result = {key: getattr(texture, key) for key in fields if hasattr(texture, key)}
    if texture.type == "IMAGE":
        if texture.color_ramp is not None:
            ramp = texture.color_ramp
            result["color_ramp"] = {
                "color_mode": ramp.color_mode,
                "hue_interpolation": ramp.hue_interpolation,
                "interpolation": ramp.interpolation,
                "elements": [
                    {"position": float(element.position),
                     "color": [float(value) for value in element.color]}
                    for element in ramp.elements
                ],
            }
        result["image_user"] = {"frame_offset": texture.image_user.frame_offset}
    return result

actual = {}
for modifier in carrier.modifiers:
    texture = modifier.texture
    actual[modifier.name] = {
        "texture_name": texture.name_full,
        "texture_type": texture.type,
        "inline_texture": texture.get("potter.id") is not None,
        "settings": texture_state(texture),
        "image_name": (texture.image.name_full
                       if texture.type == "IMAGE" and texture.image else None),
    }
with open(os.path.join(root, "actual.json"), "w", encoding="utf-8") as handle:
    json.dump(actual, handle, sort_keys=True)
"#;

const NATIVE_NOISE_INSPECT_SCRIPT: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
modifier = bpy.data.objects["Texture Carrier"].modifiers["Clouds Displace"]
texture = modifier.texture
with open(os.path.join(root, "native_noise.json"), "w", encoding="utf-8") as handle:
    json.dump({"texture_name": texture.name_full if texture else None,
               "texture_type": texture.type if texture else None,
               "native_id_texture": next(
                   (item.name_full for item in bpy.data.textures
                    if item.get("potter.id") == "native_value_noise"), None)}, handle)
"#;

const DISPLACE_CASE_PRELUDE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

GRID_V = [0.25, 0.75]

def load_image(path, name):
    image = bpy.data.images.load(path, check_existing=False)
    image.name = name
    image.colorspace_settings.name = "Non-Color"
    return image

def grid_object(name, grid_u):
    coords = [(u, v) for v in GRID_V for u in grid_u]
    columns = len(grid_u)
    faces = [(j * columns + i, j * columns + i + 1,
              (j + 1) * columns + i + 1, (j + 1) * columns + i)
             for j in range(len(GRID_V) - 1) for i in range(columns - 1)]
    mesh = bpy.data.meshes.new(name + " Mesh")
    mesh.from_pydata([(u - 0.5, v - 0.5, 0.0) for u, v in coords], [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for index, loop in enumerate(mesh.loops):
        uv_layer.data[index].uv = coords[loop.vertex_index]
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj

def add_displace(obj, texture):
    modifier = obj.modifiers.new("Displace", "DISPLACE")
    modifier.texture = texture
    modifier.texture_coords = "UV"
    modifier.uv_layer = "UVMap"
    modifier.direction = "X"
    modifier.space = "LOCAL"
    modifier.strength = 1.0
    modifier.mid_level = 0.0

def evaluated_positions(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return [[float(component) for component in vertex.co] for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()

def write_case(subjects):
    expected = {obj.name: evaluated_positions(obj) for obj in subjects}
    with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as handle:
        json.dump(expected, handle, sort_keys=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "displace-case.blend"),
                                relative_remap=False)
"#;

const MULTILAYER_CASE_SCRIPT: &str = r#"
exr = load_image(os.path.join(root, "multilayer.exr"), "Multilayer Image")
import ctypes
# ImageUser.layer/pass/view are DNA shorts that RNA exposes read-only (rna_image.cc:877-890).
# Offsets: Blender 5.2.2 source/blender/makesdna/DNA_image_types.h, struct ImageUser
# (sizeof 40 on 64-bit): pass at 26, view at 34, layer at 36.
IMAGE_USER_DNA_OFFSETS = {"multilayer_pass": 26, "multilayer_view": 34, "multilayer_layer": 36}

def write_image_user_dna(image_user, key, value):
    address = image_user.as_pointer() + IMAGE_USER_DNA_OFFSETS[key]
    ctypes.c_short.from_address(address).value = value
    if getattr(image_user, key) != value:
        raise RuntimeError("%s read back %r after DNA write of %r"
                           % (key, getattr(image_user, key), value))
cases = [
    ("Multilayer Default", 0, 0, 0),
    ("Multilayer Layer 1 Pass 1 View 0", 1, 1, 0),
    ("Multilayer Layer 0 Pass 1 View 1", 0, 1, 1),
    ("Multilayer Layer 1 Pass 0 View 1", 1, 0, 1),
]
subjects = []
for name, layer, pass_index, view in cases:
    texture = bpy.data.textures.new(name + " Legacy", type="IMAGE")
    texture.image = exr
    texture.use_interpolation = False
    write_image_user_dna(texture.image_user, "multilayer_layer", layer)
    write_image_user_dna(texture.image_user, "multilayer_pass", pass_index)
    write_image_user_dna(texture.image_user, "multilayer_view", view)
    selected = (texture.image_user.multilayer_layer, texture.image_user.multilayer_pass,
                texture.image_user.multilayer_view)
    if selected != (layer, pass_index, view):
        raise RuntimeError("%s read back %r, expected %r"
                           % (name, selected, (layer, pass_index, view)))
    obj = grid_object(name, [0.2, 0.5, 0.8])
    add_displace(obj, texture)
    subjects.append(obj)
write_case(subjects)
"#;

const UDIM_CASE_SCRIPT: &str = r#"
udim = load_image(os.path.join(root, "udim.1001.png"), "UDIM Image")
udim.source = "TILED"
udim.filepath = os.path.join(root, "udim.<UDIM>.png")
udim.reload()
tile_numbers = sorted(tile.number for tile in udim.tiles)
if tile_numbers != [1001, 1002]:
    raise RuntimeError("UDIM tiles are %r" % (tile_numbers,))
if VARIANT == "packed":
    udim.pack()
plain = bpy.data.textures.new("UDIM Plain Legacy", type="IMAGE")
plain.image = udim
plain.use_interpolation = False
repeat_crop = bpy.data.textures.new("UDIM Repeat Crop Legacy", type="IMAGE")
repeat_crop.image = udim
repeat_crop.use_interpolation = False
repeat_crop.repeat_x = 2
repeat_crop.crop_min_x = 0.0
repeat_crop.crop_max_x = 1.5
subjects = []
for name, texture in [("UDIM Plain", plain), ("UDIM Repeat Crop", repeat_crop)]:
    obj = grid_object(name, [0.2, 0.6, 1.25, 1.6])
    add_displace(obj, texture)
    subjects.append(obj)
write_case(subjects)
if VARIANT == "packed":
    # Packed tiles must be the only source Potter can read, so the external PNGs are removed.
    for number in (1001, 1002):
        os.remove(os.path.join(root, "udim.%d.png" % number))
"#;

const PNG_CASE_SCRIPT: &str = r#"
image = load_image(os.path.join(root, "orientation.png"), "Orientation Image")
texture = bpy.data.textures.new("Orientation Legacy", type="IMAGE")
texture.image = image
texture.use_interpolation = False
obj = grid_object("PNG Orientation", [0.2, 0.5, 0.8])
add_displace(obj, texture)
write_case([obj])
"#;

fn run_ok(command: Command, operation: &str) -> Result<Output, Box<dyn Error>> {
    let output = process::run_guarded(command)?;
    assert!(
        output.status.success(),
        "{operation} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn run_blender_script(
    blender: &Path,
    script_path: &Path,
    script: &str,
    blend: Option<&Path>,
    args: &[&Path],
) -> Result<(), Box<dyn Error>> {
    fs::write(script_path, script)?;
    let mut command = Command::new(blender);
    command.args(["--background", "--disable-autoexec"]);
    if let Some(blend) = blend {
        command.arg(blend);
    } else {
        command.arg("--factory-startup");
    }
    command
        .args(["--python-exit-code", "1", "--python"])
        .arg(script_path)
        .arg("--");
    for arg in args {
        command.arg(arg);
    }
    run_ok(command, "Blender fixture script")?;
    Ok(())
}

fn import_blend(project: &Path, blend: &Path, blender: &Path) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command
        .args(["import"])
        .arg(project)
        .args(["--file"])
        .arg(blend)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(blender)
        .arg("--json");
    run_ok(command, "Blender import")?;
    Ok(())
}

fn assert_value_subset(actual: &Value, expected: &Value) {
    if let Some(expected) = expected.as_object() {
        for (key, value) in expected {
            assert_value_subset(actual.get(key).unwrap(), value);
        }
    } else {
        assert_eq!(actual, expected);
    }
}

fn assert_scene_textures(document: &Value, expected: &Value) {
    let resources = document["resources"].as_object().unwrap();
    let names = document["compatibility"]["blender_adapter"]["legacy_texture_names"]
        .as_object()
        .unwrap();
    let carrier = document["nodes"]
        .as_object()
        .unwrap()
        .values()
        .find(|node| node["name"] == "Texture Carrier")
        .unwrap();
    let mut shared_texture_ids = Vec::new();
    let mut shared_image_ids = Vec::new();
    for (modifier_name, state) in expected.as_object().unwrap() {
        let modifier = carrier["modifiers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|modifier| modifier["name"] == *modifier_name)
            .unwrap();
        let texture_id = modifier["params"]["texture"].as_str().unwrap();
        if state["texture_type"] == "IMAGE" && state["inline_texture"] == true {
            let texture = resources.get(texture_id).unwrap();
            assert_eq!(texture["storage"], "inline");
            assert_eq!(texture["kind"], "legacy_texture");
            assert_eq!(texture["texture_type"], "IMAGE");
            let image_id = texture["image"].as_str().unwrap();
            let image = document["images"]
                .as_object()
                .unwrap()
                .get(image_id)
                .unwrap();
            assert_eq!(image["name"], state["image_name"]);
            assert_value_subset(&texture["settings"], &state["settings"]);
            assert_eq!(names[texture_id], state["texture_name"]);
            assert!(texture.get("owner").is_none());
            assert!(texture.get("path").is_none());
            assert!(texture.get("uri").is_none());
            assert!(texture.get("hash").is_none());
            assert!(texture.get("packed_data").is_none());
            shared_texture_ids.push(texture_id);
            shared_image_ids.push(image_id);
        } else if state["texture_type"] == "IMAGE" {
            let image = document["images"]
                .as_object()
                .unwrap()
                .get(texture_id)
                .unwrap();
            assert_eq!(image["name"], state["image_name"]);
            assert!(!resources.contains_key(texture_id));
        } else {
            let texture = resources.get(texture_id).unwrap();
            assert_eq!(texture["storage"], "inline");
            assert_eq!(texture["kind"], "legacy_texture");
            assert_eq!(texture["texture_type"], state["texture_type"]);
            assert_eq!(texture["settings"], state["settings"]);
            assert_eq!(names[texture_id], state["texture_name"]);
            assert!(texture.get("owner").is_none());
            assert!(texture.get("path").is_none());
            assert!(texture.get("uri").is_none());
            assert!(texture.get("hash").is_none());
            assert!(texture.get("packed_data").is_none());
        }
    }
    assert_eq!(shared_texture_ids.len(), 2);
    assert_ne!(shared_texture_ids[0], shared_texture_ids[1]);
    assert_eq!(shared_image_ids[0], shared_image_ids[1]);
    let raw = &document["compatibility"]["blender_adapter"]["intermediate"];
    assert!(
        !raw["other_datablocks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|block| { block["type"] == "textures" })
    );
}

const MULTILAYER_LAYER_NAMES: [&str; 2] = ["LayerA", "LayerB"];
const MULTILAYER_PASS_NAMES: [&str; 2] = ["Combined", "Diffuse"];
const MULTILAYER_VIEW_NAMES: [&str; 2] = ["left", "right"];

/// Alpha-1 top and bottom rows for multilayer index `index = layer * 4 + pass * 2 + view`.
fn multilayer_pixels(index: usize) -> ([f32; 4], [f32; 4]) {
    let base = (index as f32 + 1.0) / 10.0;
    (
        [base, 0.9 - base, 0.2 + 0.5 * base, 1.0],
        [0.9 - base, base + 0.05, 0.6 - 0.4 * base, 1.0],
    )
}

/// Writes a 1x2 multilayer EXR with 2 layers x 2 passes x 2 views, all alpha 1.
/// Row 0 is the top row (`position.1 == 0`), matching the builder in `geom/legacy_texture.rs`.
fn write_multilayer_exr(path: &Path) -> Result<(), Box<dyn Error>> {
    use exr::prelude::{
        Encoding, Image as ExrImage, ImageAttributes, IntegerBounds, Layer, LayerAttributes,
        SpecificChannels, Vec2, WritableImage,
    };
    let size = Vec2(1_usize, 2_usize);
    let layer = |index: usize| {
        let layer_name = MULTILAYER_LAYER_NAMES[index / 4];
        let pass_name = MULTILAYER_PASS_NAMES[(index / 2) % 2];
        let view_name = MULTILAYER_VIEW_NAMES[index % 2];
        let (top, bottom) = multilayer_pixels(index);
        let mut attributes =
            LayerAttributes::named(format!("{layer_name}.{pass_name}-{view_name}").as_str());
        attributes.view_name = Some(view_name.into());
        Layer::new(
            size,
            attributes,
            Encoding::FAST_LOSSLESS,
            SpecificChannels::rgba(move |position: Vec2<usize>| {
                let rgba = if position.1 == 0 { top } else { bottom };
                (rgba[0], rgba[1], rgba[2], rgba[3])
            }),
        )
    };
    ExrImage::empty(ImageAttributes::new(IntegerBounds::from_dimensions(size)))
        .with_layer(layer(0))
        .with_layer(layer(1))
        .with_layer(layer(2))
        .with_layer(layer(3))
        .with_layer(layer(4))
        .with_layer(layer(5))
        .with_layer(layer(6))
        .with_layer(layer(7))
        .write()
        .to_file(path)?;
    Ok(())
}

/// Writes an 8-bit RGBA PNG with one pixel per row; `rows[0]` is the file's top row.
fn write_png(path: &Path, width: u32, rows: &[[u8; 4]]) -> Result<(), Box<dyn Error>> {
    let pixels: Vec<u8> = rows.iter().flatten().copied().collect();
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, rows.len() as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&pixels)?;
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn node_id_named<'a>(
    doc: &'a potter_core::model::SceneDoc,
    name: &str,
) -> Result<&'a potter_core::model::Id, Box<dyn Error>> {
    doc.nodes
        .iter()
        .find(|(_, node)| node.name == name)
        .map(|(id, _)| id)
        .ok_or_else(|| format!("imported Blender object {name} is missing").into())
}

/// Runs one Blender displacement fixture, imports its `.blend` into a fresh project, and checks
/// Potter's evaluated vertices against Blender's own evaluated vertices. Returns Blender's JSON.
fn run_displace_case(
    blender: &Path,
    root: &Path,
    case_name: &str,
    case_script: &str,
) -> Result<(Value, std::path::PathBuf), Box<dyn Error>> {
    let script = [DISPLACE_CASE_PRELUDE, case_script].concat();
    run_blender_script(
        blender,
        &root.join(format!("{case_name}.py")),
        &script,
        None,
        &[root],
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("expected.json"))?)?;
    let project = root.join(format!("{case_name}-project"));
    let mut init = Command::new(env!("CARGO_BIN_EXE_pot"));
    init.args(["init"]).arg(&project).arg("--json");
    run_ok(init, "displacement scene initialization")?;
    import_blend(&project, &root.join("displace-case.blend"), blender)?;
    Ok((expected, project))
}

fn assert_potter_matches_blender_positions(
    project: &Path,
    expected: &Value,
) -> Result<(), Box<dyn Error>> {
    let opened = potter_core::store::Project::open(project)?;
    let doc = opened.doc();
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        doc,
        &potter_core::eval::EvaluationContext::default(),
        Some(opened.path()),
        Some(opened.cache_directory()),
    )?;
    for (name, expected_positions) in expected
        .as_object()
        .ok_or("Blender positions are not an object")?
    {
        let id = node_id_named(doc, name)?;
        let mesh = snapshot
            .meshes
            .get(id)
            .ok_or_else(|| format!("Potter evaluated mesh {name} is missing"))?;
        let actual: Vec<[f64; 3]> = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co.to_array())
            .collect();
        let expected_positions = expected_positions
            .as_array()
            .ok_or("Blender vertex list is not an array")?;
        assert_eq!(
            actual.len(),
            expected_positions.len(),
            "{name}: vertex count"
        );
        for (index, (actual, expected)) in actual.iter().zip(expected_positions).enumerate() {
            for axis in 0..3 {
                let expected_value = expected[axis]
                    .as_f64()
                    .ok_or("Blender coordinate is not a number")?;
                assert!(
                    (actual[axis] - expected_value).abs() <= 1.0e-5,
                    "{name} vertex {index} axis {axis}: Potter {} vs Blender {expected_value}",
                    actual[axis]
                );
            }
        }
    }
    Ok(())
}

fn max_position_difference(left: &Value, right: &Value) -> f64 {
    left.as_array()
        .unwrap()
        .iter()
        .zip(right.as_array().unwrap())
        .flat_map(|(a, b)| {
            (0..3).map(move |axis| (a[axis].as_f64().unwrap() - b[axis].as_f64().unwrap()).abs())
        })
        .fold(0.0, f64::max)
}

fn assert_subjects_differ(expected: &Value) {
    let subjects: Vec<(&String, &Value)> = expected.as_object().unwrap().iter().collect();
    for (index, (left_name, left)) in subjects.iter().enumerate() {
        for (right_name, right) in &subjects[index + 1..] {
            assert!(
                max_position_difference(left, right) > 1.0e-5,
                "{left_name} and {right_name} evaluate identically in Blender"
            );
        }
    }
}

/// The imported UDIM image must carry tile 1002 as an `ImageTile` whose blob is content-addressed.
fn assert_imported_udim_tile(project: &Path) -> Result<(), Box<dyn Error>> {
    let opened = potter_core::store::Project::open(project)?;
    let image = opened
        .doc()
        .images
        .values()
        .find(|image| image.name == "UDIM Image")
        .ok_or("imported UDIM image is missing")?;
    let tile = image
        .tiles
        .iter()
        .find(|tile| tile.number == 1002)
        .ok_or("imported UDIM image has no tile 1002")?;
    assert_eq!((tile.width, tile.height), (1, 2), "tile 1002 dimensions");
    assert!(
        tile.blob.starts_with("sha256:"),
        "tile 1002 blob is not content-addressed"
    );
    Ok(())
}

#[test]
fn blender_legacy_image_user_multilayer_selection_matches_potter_displacement()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!(
            "skipping Blender legacy ImageUser multilayer displacement: Blender is unavailable"
        );
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    write_multilayer_exr(&root.join("multilayer.exr"))?;
    let (expected, project) =
        run_displace_case(&blender, root, "multilayer", MULTILAYER_CASE_SCRIPT)?;
    assert_potter_matches_blender_positions(&project, &expected)?;
    assert_subjects_differ(&expected);
    Ok(())
}

#[test]
fn blender_legacy_udim_tiles_repeat_crop_match_potter_displacement() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender legacy UDIM displacement: Blender is unavailable");
        return Ok(());
    };
    for variant in ["external", "packed"] {
        let directory = tempdir()?;
        let root = directory.path();
        write_png(
            &root.join("udim.1001.png"),
            1,
            &[[40, 200, 120, 255], [220, 60, 30, 255]],
        )?;
        write_png(
            &root.join("udim.1002.png"),
            1,
            &[[250, 250, 10, 255], [10, 30, 240, 255]],
        )?;
        let script = format!("VARIANT = {variant:?}\n{UDIM_CASE_SCRIPT}");
        let (expected, project) = run_displace_case(&blender, root, "udim", &script)?;
        assert_imported_udim_tile(&project)?;
        assert_potter_matches_blender_positions(&project, &expected)?;
        assert_subjects_differ(&expected);
    }
    Ok(())
}

#[test]
fn blender_legacy_png_row_orientation_matches_potter_displacement() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender legacy PNG orientation displacement: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    write_png(
        &root.join("orientation.png"),
        1,
        &[[200, 100, 50, 255], [30, 220, 90, 255]],
    )?;
    let (expected, project) = run_displace_case(&blender, root, "png", PNG_CASE_SCRIPT)?;
    assert_potter_matches_blender_positions(&project, &expected)?;
    // Grid rows are v=0.25 (vertices 0..3) and v=0.75 (vertices 3..6); each column's
    // rest x is identical across rows, so a differing x means the two PNG rows sample differently.
    let positions = expected["PNG Orientation"].as_array().unwrap();
    for column in 0..3 {
        let lower = positions[column][0].as_f64().unwrap();
        let upper = positions[column + 3][0].as_f64().unwrap();
        assert!(
            (lower - upper).abs() > 1.0e-5,
            "PNG rows sample identically in Blender at column {column}"
        );
    }
    Ok(())
}

#[test]
fn blender_legacy_texture_round_trip_and_native_value_noise_loss() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender legacy Texture round-trip: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        &root.join("make_fixture.py"),
        FIXTURE_SCRIPT,
        None,
        &[root],
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("expected.json"))?)?;
    assert_ne!(
        expected["Image Displace"]["settings"],
        expected["Image Filtered Displace"]["settings"],
    );
    let project = root.join("project");
    let mut init = Command::new(env!("CARGO_BIN_EXE_pot"));
    init.args(["init"]).arg(&project).arg("--json");
    run_ok(init, "scene initialization")?;
    let source = root.join("legacy-textures.blend");
    import_blend(&project, &source, &blender)?;
    let source_document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    assert_scene_textures(&source_document, &expected);

    let exported = root.join("legacy-textures-roundtrip.blend");
    let mut export = Command::new(env!("CARGO_BIN_EXE_pot"));
    export
        .args(["export"])
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&exported)
        .args(["--blender"])
        .arg(&blender)
        .arg("--json");
    run_ok(export, "Blender export")?;
    run_blender_script(
        &blender,
        &root.join("inspect_fixture.py"),
        INSPECT_SCRIPT,
        Some(&exported),
        &[root],
    )?;
    let actual: Value = serde_json::from_slice(&fs::read(root.join("actual.json"))?)?;
    for (modifier_name, state) in expected.as_object().unwrap() {
        let actual_state = &actual[modifier_name];
        assert_eq!(actual_state["texture_type"], state["texture_type"]);
        if state["texture_type"] == "IMAGE" {
            assert_eq!(actual_state["inline_texture"], state["inline_texture"]);
            assert_eq!(actual_state["image_name"], state["image_name"]);
            assert_value_subset(&actual_state["settings"], &state["settings"]);
            if state["inline_texture"] == true {
                assert_eq!(actual_state["texture_name"], state["texture_name"]);
            }
        } else {
            assert_eq!(actual_state, state);
        }
    }

    let second_project = root.join("round-trip-project");
    let mut init = Command::new(env!("CARGO_BIN_EXE_pot"));
    init.args(["init"]).arg(&second_project).arg("--json");
    run_ok(init, "round-trip scene initialization")?;
    import_blend(&second_project, &exported, &blender)?;
    let imported_document: Value =
        serde_json::from_slice(&fs::read(second_project.join("scene.json"))?)?;
    assert_scene_textures(&imported_document, &expected);
    let mut native_noise_document = imported_document.clone();
    let carrier_id = native_noise_document["nodes"]
        .as_object()
        .unwrap()
        .iter()
        .find(|(_, node)| node["name"] == "Texture Carrier")
        .map(|(id, _)| id.clone())
        .unwrap();
    let cloud_index = native_noise_document["nodes"][carrier_id.as_str()]["modifiers"]
        .as_array()
        .unwrap()
        .iter()
        .position(|modifier| modifier["name"] == "Clouds Displace")
        .unwrap();
    let cloud_modifier =
        &native_noise_document["nodes"][carrier_id.as_str()]["modifiers"][cloud_index];
    let cloud_modifier_id = cloud_modifier["id"].as_str().unwrap().to_owned();
    let old_texture_id = cloud_modifier["params"]["texture"]
        .as_str()
        .unwrap()
        .to_owned();
    native_noise_document["resources"]
        .as_object_mut()
        .unwrap()
        .remove(&old_texture_id);
    native_noise_document["resources"]["native_value_noise"] = json!({
        "storage": "inline",
        "kind": "legacy_texture",
        "texture_type": "NATIVE_VALUE_NOISE",
        "settings": {}
    });
    native_noise_document["nodes"][carrier_id.as_str()]["modifiers"][cloud_index]["params"]["texture"] =
        json!("native_value_noise");
    fs::write(
        second_project.join("scene.json"),
        serde_json::to_vec_pretty(&native_noise_document)?,
    )?;

    let strict_output = root.join("native-value-noise-refused.blend");
    let mut strict_export = Command::new(env!("CARGO_BIN_EXE_pot"));
    strict_export
        .args(["export"])
        .arg(&second_project)
        .args(["--format", "blend", "--out"])
        .arg(&strict_output)
        .args(["--blender"])
        .arg(&blender)
        .arg("--json");
    let rejected = process::run_guarded(strict_export)?;
    assert!(
        !strict_output.exists(),
        "refused export wrote a Blender file"
    );
    assert!(
        !rejected.status.success(),
        "native Value Noise export unexpectedly succeeded"
    );
    let rejection: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejection["error"]["code"], "UNREPRESENTABLE_FEATURE");
    let texture_loss = rejection["error"]["details"]["losses"]
        .as_array()
        .unwrap()
        .iter()
        .find(|loss| loss["feature_id"] == "blender.texture.native_value_noise")
        .unwrap();
    assert_eq!(
        texture_loss["data_id"].as_str(),
        Some(cloud_modifier_id.as_str())
    );
    assert!(
        texture_loss["reason"]
            .as_str()
            .unwrap()
            .contains("native_value_noise")
    );
    assert!(
        texture_loss["reason"]
            .as_str()
            .unwrap()
            .contains("has no Blender legacy Texture type")
    );
    assert!(
        texture_loss["suggestion"]
            .as_str()
            .unwrap()
            .contains("--allow-lossy")
    );

    let lossy_output = root.join("native-value-noise-lossy.blend");
    let mut lossy_export = Command::new(env!("CARGO_BIN_EXE_pot"));
    lossy_export
        .args(["export"])
        .arg(&second_project)
        .args(["--format", "blend", "--out"])
        .arg(&lossy_output)
        .args(["--allow-lossy", "--blender"])
        .arg(&blender)
        .arg("--json");
    let lossy = run_ok(lossy_export, "lossy native Value Noise export")?;
    let report: Value = serde_json::from_slice(&lossy.stdout)?;
    assert!(
        report["result"]["losses"]
            .as_array()
            .unwrap()
            .iter()
            .any(|loss| { loss["feature_id"] == "blender.texture.native_value_noise" })
    );
    run_blender_script(
        &blender,
        &root.join("inspect_native_noise.py"),
        NATIVE_NOISE_INSPECT_SCRIPT,
        Some(&lossy_output),
        &[root],
    )?;
    let native_state: Value = serde_json::from_slice(&fs::read(root.join("native_noise.json"))?)?;
    assert_eq!(native_state["texture_type"], Value::Null);
    assert_eq!(native_state["texture_name"], Value::Null);
    assert_eq!(native_state["native_id_texture"], Value::Null);
    Ok(())
}
