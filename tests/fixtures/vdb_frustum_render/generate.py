"""Build deterministic Cycles render captures for the frustum-map VDB regression.

From the repository root, with Blender 5.2.2:
    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --threads 1 \
        --disable-autoexec --python-exit-code 3 -P tests/fixtures/vdb_frustum_render/generate.py --
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parent
VDB = ROOT.parent / "openvdb_transform_types" / "frustum.vdb"
SIZE = 16
SAMPLES = 64
SEEDS = (0, 1, 2, 3)
ORTHO_SCALE = 0.28
EMISSION_STRENGTH = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_rgb(path: Path) -> list[list[float]]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = list(image.pixels)
        return [
            pixels[(row * width + column) * 4 : (row * width + column) * 4 + 3]
            for row in range(height - 1, -1, -1)
            for column in range(width)
        ]
    finally:
        bpy.data.images.remove(image)


def configure_scene() -> tuple[object, object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.seed = 0
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.volume_step_rate = 1.0
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"
    else:
        scene.cycles.filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.render.resolution_x = SIZE
    scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.exr_codec = "NONE"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    world = bpy.data.worlds.new("FrustumOracleWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.world = world

    bpy.ops.object.volume_import(filepath=str(VDB))
    volume = bpy.context.active_object
    volume.name = "FrustumDensity"
    volume.data.filepath = "//../openvdb_transform_types/frustum.vdb"
    material = bpy.data.materials.new("FrustumPrincipled")
    material.use_nodes = True
    material.cycles.volume_interpolation = "LINEAR"
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    principled = tree.nodes.new("ShaderNodeVolumePrincipled")
    principled.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    principled.inputs["Density"].default_value = 1.0
    principled.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs["Emission Strength"].default_value = EMISSION_STRENGTH
    tree.links.new(principled.outputs["Volume"], output.inputs["Volume"])
    volume.data.materials.append(material)

    camera_data = bpy.data.cameras.new("FrustumOracleCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ORTHO_SCALE
    camera = bpy.data.objects.new("FrustumOracleCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (-0.05, -0.05, 1.8)
    scene.camera = camera
    return scene, material


def render_exr(scene: object, path: Path, seed: int) -> list[list[float]]:
    scene.render.engine = "CYCLES"
    scene.cycles.seed = seed
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return image_rgb(path)


scene, material = configure_scene()
scene_path = ROOT / "scene.blend"
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(scene_path))

cycles = {}
for mode, interpolation in (("linear", "LINEAR"), ("cubic", "CUBIC")):
    material.cycles.volume_interpolation = interpolation
    captures = []
    for seed in SEEDS:
        path = ROOT / f"cycles_{mode}_{seed}.exr"
        pixels = render_exr(scene, path, seed)
        captures.append(pixels)
        print(f"CYCLES_CAPTURE {mode} seed={seed} path={path}")
    cycles[mode] = {
        "interpolation": interpolation,
        "seeds": list(SEEDS),
        "captures": [
            {
                "file": f"cycles_{mode}_{seed}.exr",
                "sha256": sha256(ROOT / f"cycles_{mode}_{seed}.exr"),
            }
            for seed in SEEDS
        ],
        "mean_rgb": [
            [sum(seed_pixels[index][channel] for seed_pixels in captures) / len(captures) for channel in range(3)]
            for index in range(SIZE * SIZE)
        ],
        "single_render_noise_rms": sum(
            sum(
                (seed_pixels[index][channel] - sum(sample[index][channel] for sample in captures) / len(captures)) ** 2
                for index in range(SIZE * SIZE)
                for channel in range(3)
            )
            / (SIZE * SIZE * 3 * len(captures))
            for seed_pixels in captures
        ) ** 0.5,
    }


manifest = {
    "blender_version": bpy.app.version_string,
    "blender_build_hash": bpy.app.build_hash.decode("ascii") if isinstance(bpy.app.build_hash, bytes) else bpy.app.build_hash,
    "generation_command": "/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --threads 1 --disable-autoexec --python-exit-code 3 -P tests/fixtures/vdb_frustum_render/generate.py --",
    "source_vdb": {
        "file": "../openvdb_transform_types/frustum.vdb",
        "sha256": sha256(VDB),
        "grid": "density",
        "map": "NonlinearFrustumMap",
        "frustum_taper": 0.5,
        "depth": 3.0,
        "voxel_size": 0.25,
    },
    "scene": {"file": "scene.blend", "sha256": sha256(scene_path)},
    "resolution": [SIZE, SIZE],
    "camera": {"projection": "ORTHO", "ortho_scale": ORTHO_SCALE, "location": [-0.05, -0.05, 1.8]},
    "material": {
        "name": "FrustumPrincipled",
        "shader": "ShaderNodeVolumePrincipled",
        "density": 1.0,
        "emission_color": [1.0, 1.0, 1.0],
        "emission_strength": EMISSION_STRENGTH,
    },
    "cycles": {
        "engine": "CYCLES",
        "device": "CPU",
        "samples": SAMPLES,
        "max_bounces": 0,
        "volume_bounces": 0,
        "denoising": False,
        "pixel_filter": "BOX",
        "filter_width": 1.0,
        "view_transform": "Raw",
        "row_order": "OpenEXR top-down",
        "noise": "single-render RMS over RGB pixels across the four native seeds; mean_rgb is the four-seed oracle",
        "cases": cycles,
    },
    "generator": {"file": "generate.py", "sha256": sha256(Path(__file__).resolve())},
}
(ROOT / "oracle.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("ORACLE_OK vdb_frustum_render")
