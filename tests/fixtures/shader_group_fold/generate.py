"""Generate a Blender 5.2.2 shader-group folding oracle for Cycles and EEVEE."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import bpy


def group_output(group, name="Group Output"):
    node = group.nodes.new("NodeGroupOutput")
    node.name = name
    node.is_active_output = True
    return node


def make_shader_graphs():
    inner = bpy.data.node_groups.new("Fold Inner", "ShaderNodeTree")
    weight = inner.interface.new_socket(name="Weight", in_out="INPUT", socket_type="NodeSocketFloat")
    weight.default_value = 0.0
    vector = inner.interface.new_socket(name="Vector", in_out="INPUT", socket_type="NodeSocketVector")
    vector.default_value = (1.0e-20, 0.0, 0.0)
    outside = inner.interface.new_socket(name="Outside", in_out="INPUT", socket_type="NodeSocketFloat")
    outside.default_value = 2.0
    result = inner.interface.new_socket(name="Result", in_out="OUTPUT", socket_type="NodeSocketFloat")
    result.default_value = 0.875
    inner_input = inner.nodes.new("NodeGroupInput")
    inner_output = group_output(inner)
    sign = inner.nodes.new("ShaderNodeMath")
    sign.operation = "SIGN"
    normalize = inner.nodes.new("ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    separate = inner.nodes.new("ShaderNodeSeparateXYZ")
    add_a = inner.nodes.new("ShaderNodeMath")
    add_a.operation = "ADD"
    add_b = inner.nodes.new("ShaderNodeMath")
    add_b.operation = "ADD"
    links = inner.links
    links.new(inner_input.outputs["Weight"], sign.inputs["Value"])
    links.new(inner_input.outputs["Vector"], normalize.inputs["Vector"])
    links.new(normalize.outputs["Vector"], separate.inputs["Vector"])
    links.new(sign.outputs["Value"], add_a.inputs["Value"])
    links.new(separate.outputs["X"], add_a.inputs["Value_001"])
    links.new(add_a.outputs["Value"], add_b.inputs["Value"])
    links.new(inner_input.outputs["Outside"], add_b.inputs["Value_001"])
    links.new(add_b.outputs["Value"], inner_output.inputs["Result"])

    outer = bpy.data.node_groups.new("Fold Outer", "ShaderNodeTree")
    outer_weight = outer.interface.new_socket(
        name="Weight", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    outer_weight.default_value = 0.0
    outer_vector = outer.interface.new_socket(
        name="Vector", in_out="INPUT", socket_type="NodeSocketVector"
    )
    outer_vector.default_value = (1.0e-20, 0.0, 0.0)
    outer_result = outer.interface.new_socket(
        name="Result", in_out="OUTPUT", socket_type="NodeSocketFloat"
    )
    outer_result.default_value = 0.875
    fallback = outer.interface.new_socket(
        name="Fallback", in_out="OUTPUT", socket_type="NodeSocketFloat"
    )
    fallback.default_value = 0.875
    outer_input = outer.nodes.new("NodeGroupInput")
    nested = outer.nodes.new("ShaderNodeGroup")
    nested.node_tree = inner
    outer_output = group_output(outer)
    outer_output.inputs["Fallback"].default_value = 0.25
    outer.links.new(outer_input.outputs["Weight"], nested.inputs["Weight"])
    outer.links.new(outer_input.outputs["Vector"], nested.inputs["Vector"])
    outer.links.new(nested.outputs["Result"], outer_output.inputs["Result"])
    return outer


def render_center(scene, engine):
    scene.render.engine = engine
    with tempfile.TemporaryDirectory(prefix="potter-shader-group-fold-") as directory:
        image_path = Path(directory) / "group_fold.exr"
        scene.render.filepath = str(image_path)
        scene.render.use_file_extension = False
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(image_path), check_existing=False)
        try:
            width, height = image.size
            pixel = (height // 2 * width + width // 2) * 4
            return [float(value) for value in image.pixels[pixel : pixel + 3]]
        finally:
            bpy.data.images.remove(image)


def main():
    if bpy.app.version[:3] != (5, 2, 2):
        raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")
    if "--" not in sys.argv or len(sys.argv) != sys.argv.index("--") + 2:
        raise SystemExit("expected one output directory after --")
    output_dir = Path(sys.argv[sys.argv.index("--") + 1])
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False

    camera_data = bpy.data.cameras.new("Fold Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.0
    camera = bpy.data.objects.new("Fold Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 4.0)
    scene.camera = camera

    bpy.ops.mesh.primitive_plane_add(size=2.0)
    obj = bpy.context.object
    obj.name = "Fold Probe"
    material = bpy.data.materials.new("Shader Group Fold Probe")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    group_node = tree.nodes.new("ShaderNodeGroup")
    group_node.node_tree = make_shader_graphs()
    group_node.inputs["Weight"].default_value = 0.0
    group_node.inputs["Vector"].default_value = (1.0e-20, 0.0, 0.0)
    add = tree.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.4, 0.6, 0.8, 1.0)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(group_node.outputs["Result"], add.inputs["Value"])
    tree.links.new(group_node.outputs["Fallback"], add.inputs["Value_001"])
    tree.links.new(add.outputs["Value"], emission.inputs["Strength"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.append(material)

    oracle = {
        "blender_version": bpy.app.version_string,
        "cycles": render_center(scene, "CYCLES"),
        "eevee": render_center(scene, "BLENDER_EEVEE"),
    }
    (output_dir / "oracle.json").write_text(json.dumps(oracle), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "shader_group_fold.blend"))


if __name__ == "__main__":
    main()
