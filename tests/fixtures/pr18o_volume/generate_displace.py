import bpy
import json
import os
import sys


def canonical_mesh(mesh):
    vertices = [tuple(float(component) for component in vertex.co) for vertex in mesh.vertices]
    ordered = sorted(enumerate(vertices), key=lambda item: item[1])
    remap = {old: new for new, (old, _) in enumerate(ordered)}
    sorted_vertices = [list(co) for _, co in ordered]
    polygons = []
    for polygon in mesh.polygons:
        face = [remap[index] for index in polygon.vertices]
        rotations = [face[offset:] + face[:offset] for offset in range(len(face))]
        polygons.append((min(rotations), bool(polygon.use_smooth)))
    polygons.sort()
    if sorted_vertices:
        bounds_min = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
        bounds_max = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    else:
        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]
    return {
        "vertex_count": len(sorted_vertices),
        "edge_count": len(mesh.edges),
        "polygon_count": len(polygons),
        "corner_count": sum(len(face) for face, _ in polygons),
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "vertices": sorted_vertices,
        "polygons": [face for face, _ in polygons],
        "smooth_flags": [smooth for _, smooth in polygons],
    }


def image_pixels(width, height):
    return [
        [x / (width - 1), y / (height - 1), 0.5, 1.0]
        for y in range(height)
        for x in range(width)
    ]


def run(output_dirs):
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
    outputs = []
    pixels = image_pixels(16, 16)
    for mapping in ("LOCAL", "GLOBAL", "OBJECT"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1.0, 0.0, 0.0))
        source = bpy.context.object

        image = bpy.data.images.new("pixels", width=16, height=16, alpha=True, float_buffer=True)
        image.pixels = [component for pixel in pixels for component in pixel]
        image.update()
        texture = bpy.data.textures.new("image_texture", "IMAGE")
        texture.image = image
        texture.extension = "REPEAT"
        texture.use_interpolation = True

        volume_data = bpy.data.volumes.new("volume_data")
        volume = bpy.data.objects.new("volume", volume_data)
        bpy.context.collection.objects.link(volume)
        volume.location = (2.0, 1.0, 0.0)
        mesh_to_volume = volume.modifiers.new("mesh_to_volume", "MESH_TO_VOLUME")
        mesh_to_volume.object = source
        mesh_to_volume.resolution_mode = "VOXEL_SIZE"
        mesh_to_volume.voxel_size = 0.25
        mesh_to_volume.interior_band_width = 0.25

        displace = volume.modifiers.new("volume_displace", "VOLUME_DISPLACE")
        displace.texture = texture
        displace.strength = 2.0
        displace.texture_sample_radius = 1.0
        displace.texture_map_mode = mapping
        displace.texture_mid_level = (0.5, 0.5, 0.5)
        map_object = None
        if mapping == "OBJECT":
            bpy.ops.object.empty_add(location=(0.3, -0.7, 0.0))
            map_object = bpy.context.object
            displace.texture_map_object = map_object

        carrier_data = bpy.data.meshes.new("carrier_data")
        carrier = bpy.data.objects.new("carrier", carrier_data)
        bpy.context.collection.objects.link(carrier)
        volume_to_mesh = carrier.modifiers.new("volume_to_mesh", "VOLUME_TO_MESH")
        volume_to_mesh.object = volume
        volume_to_mesh.grid_name = "density"
        volume_to_mesh.threshold = 0.1
        volume_to_mesh.adaptivity = 0.0
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = carrier.evaluated_get(depsgraph)
        result_mesh = evaluated.to_mesh()
        outputs.append({
            "texture_map_mode": mapping,
            "volume_location": [2.0, 1.0, 0.0],
            "source_location": [1.0, 0.0, 0.0],
            "texture_map_object_location": [0.3, -0.7, 0.0] if map_object else None,
            "image_colorspace": image.colorspace_settings.name,
            "image_extension": texture.extension,
            "image_interpolation": texture.use_interpolation,
            "strength": 2.0,
            "texture_sample_radius": 1.0,
            "texture_mid_level": [0.5, 0.5, 0.5],
            "voxel_size": 0.25,
            "interior_band_width": 0.25,
            "threshold": 0.1,
            "grid_name": "density",
            **canonical_mesh(result_mesh),
        })
        evaluated.to_mesh_clear()
    document = {
        "blender_version": bpy.app.version_string,
        "image_width": 16,
        "image_height": 16,
        "image_pixels": pixels,
        "cases": outputs,
    }
    for directory in output_dirs:
        with open(os.path.join(directory, "volume_displace_image.json"), "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")


if __name__ == "__main__":
    output = "/tmp/pr18o/volume_oracle"
    if "--" in sys.argv:
        output = sys.argv[sys.argv.index("--") + 1]
    run({output, "/tmp/pr18o/volume_oracle"})
