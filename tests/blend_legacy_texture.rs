#![expect(clippy::unwrap_used, reason = "Blender integration fixtures use fixed valid data")]

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
        "use_clamp": False, "brightness": 1.2, "saturation": 0.8,
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
              "brightness", "contrast", "saturation", "factor_red", "factor_green",
              "factor_blue", "intensity"),
}
def texture_state(texture):
    fields = settings.get(texture.type, ())
    result = {key: getattr(texture, key) for key in fields if hasattr(texture, key)}
    if texture.type == "IMAGE":
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
              "brightness", "contrast", "saturation", "factor_red", "factor_green",
              "factor_blue", "intensity"),
}
def texture_state(texture):
    fields = settings.get(texture.type, ())
    result = {key: getattr(texture, key) for key in fields if hasattr(texture, key)}
    if texture.type == "IMAGE":
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
        .args(["--format", "blend", "--mode", "replace", "--base-revision", "0", "--blender"])
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
            let image = document["images"].as_object().unwrap().get(image_id).unwrap();
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
            let image = document["images"].as_object().unwrap().get(texture_id).unwrap();
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
    assert!(!raw["other_datablocks"].as_array().unwrap().iter().any(|block| {
        block["type"] == "textures"
    }));
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
    let cloud_modifier = &native_noise_document["nodes"][carrier_id.as_str()]
        ["modifiers"][cloud_index];
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
    native_noise_document["nodes"][carrier_id.as_str()]["modifiers"][cloud_index]
        ["params"]["texture"] = json!("native_value_noise");
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
    assert!(!strict_output.exists(), "refused export wrote a Blender file");
    assert!(!rejected.status.success(), "native Value Noise export unexpectedly succeeded");
    let rejection: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejection["error"]["code"], "UNREPRESENTABLE_FEATURE");
    let texture_loss = rejection["error"]["details"]["losses"]
        .as_array()
        .unwrap()
        .iter()
        .find(|loss| loss["feature_id"] == "blender.texture.native_value_noise")
        .unwrap();
    assert_eq!(texture_loss["data_id"].as_str(), Some(cloud_modifier_id.as_str()));
    assert!(texture_loss["reason"]
        .as_str()
        .unwrap()
        .contains("native_value_noise"));
    assert!(texture_loss["reason"]
        .as_str()
        .unwrap()
        .contains("has no Blender legacy Texture type"));
    assert!(texture_loss["suggestion"]
        .as_str()
        .unwrap()
        .contains("--allow-lossy"));

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
    assert!(report["result"]["losses"].as_array().unwrap().iter().any(|loss| {
        loss["feature_id"] == "blender.texture.native_value_noise"
    }));
    run_blender_script(
        &blender,
        &root.join("inspect_native_noise.py"),
        NATIVE_NOISE_INSPECT_SCRIPT,
        Some(&lossy_output),
        &[root],
    )?;
    let native_state: Value =
        serde_json::from_slice(&fs::read(root.join("native_noise.json"))?)?;
    assert_eq!(native_state["texture_type"], Value::Null);
    assert_eq!(native_state["texture_name"], Value::Null);
    assert_eq!(native_state["native_id_texture"], Value::Null);
    Ok(())
}
