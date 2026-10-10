# SPDX-License-Identifier: GPL-3.0-or-later
"""JSON bridge for the pinned headless Blender adapter.

This script deliberately never executes scripts found in an input .blend. Unknown Blender
IDs are described in the intermediate payload and the Rust side also stores the original
file bytes as an immutable compatibility blob.
"""
import base64
import bpy
import json
import math
import os
import sys
import traceback
from mathutils import Euler, Matrix, Quaternion
DEFERRED_ID_ASSIGNMENTS = []
CACHE_FILES_BY_IDENTITY = {}
POTTER_TO_BLENDER_ANIMATION_PATH = {
    "transform.translation": "location",
    "transform.rotation_euler": "rotation_euler",
    "transform.rotation_quaternion": "rotation_quaternion",
    "transform.rotation": "rotation_quaternion",
    "transform.scale": "scale",
    "camera.lens_mm": "data.lens",
    "light.energy": "data.energy",
}
POTTER_TO_BLENDER_NODE_SOCKET_TYPE = {
    "geometry": "NodeSocketGeometry",
    "float": "NodeSocketFloat",
    "integer": "NodeSocketInt",
    "boolean": "NodeSocketBool",
    "vector": "NodeSocketVector",
    "color": "NodeSocketColor",
    "rotation": "NodeSocketRotation",
    "matrix": "NodeSocketMatrix",
    "string": "NodeSocketString",
    "menu": "NodeSocketMenu",
    "object": "NodeSocketObject",
    "collection": "NodeSocketCollection",
    "image": "NodeSocketImage",
    "material": "NodeSocketMaterial",
    "font": "NodeSocketFont",
    "sound": "NodeSocketSound",
}


def blender_animation_path(path, doc):
    path = POTTER_TO_BLENDER_ANIMATION_PATH.get(path, path)
    start = path.find("pose.bones[")
    if start < 0:
        return path
    start += len("pose.bones[")
    end = path.find("]", start)
    if end < 0:
        return path
    key = path[start:end].strip("\"'")
    for data in doc.get("data_blocks", {}).values():
        bones = data.get("armature", {}).get("bones", {})
        bone = bones.get(key)
        if bone is not None:
            quote = path[start:start + 1]
            return path[:start] + quote + bone.get("name", key) + quote + path[end:]
    return path


def blender_animation_index(path, index):
    if path == "rotation_quaternion" or path.endswith(".rotation_quaternion"):
        return (1, 2, 3, 0)[index] if 0 <= index < 4 else index
    return index

POTTER_TO_BLENDER_MODIFIER = {
    "mirror": "MIRROR", "array": "ARRAY", "subdivision": "SUBSURF",
    "multires": "MULTIRES", "solidify": "SOLIDIFY", "triangulate": "TRIANGULATE",
    "bevel": "BEVEL", "decimate": "DECIMATE", "weld": "WELD",
    "displace": "DISPLACE", "smooth": "SMOOTH", "nodes": "NODES",
    "armature": "ARMATURE", "lattice": "LATTICE", "volume_to_mesh": "VOLUME_TO_MESH",
    "volume_displace": "VOLUME_DISPLACE",
    "mesh_to_volume": "MESH_TO_VOLUME", "boolean": "BOOLEAN",
    "shrinkwrap": "SHRINKWRAP", "cast": "CAST", "curve": "CURVE",
    "hook": "HOOK", "laplacian_smooth": "LAPLACIANSMOOTH",
    "laplacian_deform": "LAPLACIANDEFORM",
    "corrective_smooth": "CORRECTIVE_SMOOTH", "wave": "WAVE", "warp": "WARP",
    "simple_deform": "SIMPLE_DEFORM", "screw": "SCREW", "skin": "SKIN",
    "wireframe": "WIREFRAME", "edge_split": "EDGE_SPLIT", "build": "BUILD",
    "mask": "MASK", "weighted_normal": "WEIGHTED_NORMAL",
    "normal_edit": "NORMAL_EDIT", "uv_project": "UV_PROJECT", "uv_warp": "UV_WARP",
    "vertex_weight_edit": "VERTEX_WEIGHT_EDIT",
    "vertex_weight_mix": "VERTEX_WEIGHT_MIX",
    "vertex_weight_proximity": "VERTEX_WEIGHT_PROXIMITY",
    "surface_deform": "SURFACE_DEFORM", "mesh_deform": "MESH_DEFORM",
    "data_transfer": "DATA_TRANSFER", "ocean": "OCEAN",
    "particle_instance": "PARTICLE_INSTANCE", "explode": "EXPLODE",
    "fluid": "FLUID", "cloth": "CLOTH", "soft_body": "SOFT_BODY",
    "collision": "COLLISION", "dynamic_paint": "DYNAMIC_PAINT",
    "particle_system": "PARTICLE_SYSTEM", "remesh": "REMESH",
    "mesh_sequence_cache": "MESH_SEQUENCE_CACHE",
    "mesh_cache": "MESH_CACHE",
}
BLENDER_TO_POTTER_MODIFIER = {native: name for name, native in POTTER_TO_BLENDER_MODIFIER.items()}
POTTER_TO_BLENDER_CONSTRAINT = {
    "copy_location": "COPY_LOCATION", "copy_rotation": "COPY_ROTATION",
    "copy_scale": "COPY_SCALE", "track_to": "TRACK_TO",
    "damped_track": "DAMPED_TRACK", "locked_track": "LOCKED_TRACK",
    "stretch_to": "STRETCH_TO", "transformation": "TRANSFORM",
    "maintain_volume": "MAINTAIN_VOLUME", "floor": "FLOOR", "pivot": "PIVOT",
    "shrinkwrap": "SHRINKWRAP", "spline_ik": "SPLINE_IK",
    "limit_location": "LIMIT_LOCATION", "limit_rotation": "LIMIT_ROTATION",
    "limit_scale": "LIMIT_SCALE", "child_of": "CHILD_OF", "action": "ACTION",
    "armature": "ARMATURE", "camera_solver": "CAMERA_SOLVER",
    "clamp_to": "CLAMP_TO", "copy_transforms": "COPY_TRANSFORMS",
    "follow_path": "FOLLOW_PATH", "follow_track": "FOLLOW_TRACK",
    "geometry_attribute": "GEOMETRY_ATTRIBUTE",
    "limit_distance": "LIMIT_DISTANCE", "object_solver": "OBJECT_SOLVER",
    "transform_cache": "TRANSFORM_CACHE", "ik": "IK",
}
BLENDER_TO_POTTER_CONSTRAINT = {
    native: name for name, native in POTTER_TO_BLENDER_CONSTRAINT.items()
}
NODE_TYPE_ALIASES = {
    "ShaderNodeTree": {
        "OutputMaterial": "ShaderNodeOutputMaterial",
    },
    "GeometryNodeTree": {
        "GroupInput": "NodeGroupInput",
        "GroupOutput": "NodeGroupOutput",
        "MeshCube": "GeometryNodeMeshCube",
        "MeshGrid": "GeometryNodeMeshGrid",
        "MeshUVSphere": "GeometryNodeMeshUVSphere",
        "MeshIcoSphere": "GeometryNodeMeshIcoSphere",
        "MeshCylinder": "GeometryNodeMeshCylinder",
        "MeshCone": "GeometryNodeMeshCone",
        "MeshCircle": "GeometryNodeMeshCircle",
        "MeshLine": "GeometryNodeMeshLine",
        "Transform": "GeometryNodeTransform",
        "SetPosition": "GeometryNodeSetPosition",
        "JoinGeometry": "GeometryNodeJoinGeometry",
        "InstanceOnPoints": "GeometryNodeInstanceOnPoints",
        "RealizeInstances": "GeometryNodeRealizeInstances",
        "DistributePointsOnFaces": "GeometryNodeDistributePointsOnFaces",
        "MeshToPoints": "GeometryNodeMeshToPoints",
        "SubdivisionSurface": "GeometryNodeSubdivisionSurface",
        "InputPosition": "GeometryNodeInputPosition",
        "InputNormal": "GeometryNodeInputNormal",
        "InputIndex": "GeometryNodeInputIndex",
        "StoreNamedAttribute": "GeometryNodeStoreNamedAttribute",
        "InputNamedAttribute": "GeometryNodeInputNamedAttribute",
        "CaptureAttribute": "GeometryNodeCaptureAttribute",
        "RepeatInput": "GeometryNodeRepeatInput",
        "RepeatOutput": "GeometryNodeRepeatOutput",
        "SimulationInput": "GeometryNodeSimulationInput",
        "SimulationOutput": "GeometryNodeSimulationOutput",
    },
    "CompositorNodeTree": {
        "RLayers": "CompositorNodeRLayers",
        "Composite": "CompositorNodeComposite",
        "Viewer": "CompositorNodeViewer",
        "OutputFile": "CompositorNodeOutputFile",
        "MixRGB": "CompositorNodeMixRGB",
        "AlphaOver": "CompositorNodeAlphaOver",
        "Blur": "CompositorNodeBlur",
        "Glare": "CompositorNodeGlare",
        "Denoise": "CompositorNodeDenoise",
        "Defocus": "CompositorNodeDefocus",
        "Exposure": "CompositorNodeExposure",
        "Gamma": "CompositorNodeGamma",
        "Invert": "CompositorNodeInvert",
        "BrightContrast": "CompositorNodeBrightContrast",
        "BrightnessContrast": "CompositorNodeBrightContrast",
        "ColorBalance": "CompositorNodeColorBalance",
        "Curves": "CompositorNodeCurves",
        "HueSat": "CompositorNodeHueSat",
        "Math": "CompositorNodeMath",
        "Scale": "CompositorNodeScale",
    },
}
BLENDER_NODE_TYPE_ALIASES = {
    tree_type: {blender_type: potter_type
                for potter_type, blender_type in aliases.items()
                if potter_type == "OutputMaterial"}
    for tree_type, aliases in NODE_TYPE_ALIASES.items()
}


def blender_node_type(potter_type, tree_type):
    return NODE_TYPE_ALIASES.get(tree_type, {}).get(potter_type, potter_type)


def potter_node_type(blender_type, tree_type):
    return BLENDER_NODE_TYPE_ALIASES.get(tree_type, {}).get(blender_type, blender_type)




def blender_modifier_type(potter_type):
    return POTTER_TO_BLENDER_MODIFIER.get(
        potter_type,
        POTTER_TO_BLENDER_MODIFIER.get(potter_type.lower(), potter_type.upper()),
    )


def potter_modifier_type(blender_type):
    return BLENDER_TO_POTTER_MODIFIER.get(blender_type, blender_type.lower())




def plain(value, depth=0):
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (set, frozenset)):
        return [plain(item, depth + 1) for item in sorted(value)]
    if isinstance(value, (bpy.types.ID,)):
        return {"id_type": value.bl_rna.identifier, "name": value.name_full}
    if isinstance(value, Matrix):
        return matrix_rows(value)
    if all(hasattr(value, axis) for axis in ("r", "g", "b")):
        return [float(value.r), float(value.g), float(value.b)]
    if hasattr(value, "w") and hasattr(value, "to_euler"):
        return [value.x, value.y, value.z, value.w]
    if hasattr(value, "to_list"):
        try:
            return [plain(v, depth + 1) for v in value.to_list()]
        except Exception:
            pass
    if hasattr(value, "to_tuple"):
        try:
            return [plain(v, depth + 1) for v in value.to_tuple()]
        except Exception:
            pass
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        try:
            return [plain(v, depth + 1) for v in value]
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): plain(v, depth + 1) for k, v in value.items()}
    try:
        return repr(value)
    except Exception:
        return "<unavailable>"


def matrix_rows(matrix):
    try:
        return [[float(matrix[row][column]) for column in range(4)]
                for row in range(4)]
    except Exception:
        return []


def custom_props(block):
    try:
        return {str(key): plain(block[key]) for key in block.keys()
                if key != "potter.object_solver_inverses_json"}
    except Exception:
        return {}

def _walk_rna_properties(block, omit=(), skip=(), skip_collections=False):
    try:
        properties = block.bl_rna.properties
    except Exception:
        return
    for prop in properties:
        key = prop.identifier
        if (key == "rna_type" or key in omit or key in skip
                or getattr(prop, "is_readonly", False)
                or (skip_collections and prop.type == "COLLECTION")):
            continue
        try:
            value = getattr(block, key)
        except Exception:
            continue
        yield prop, key, value


def rna_props(block, omit=()):
    result = {}
    for prop, key, val in _walk_rna_properties(block, omit=omit):
        if isinstance(val, bpy.types.ID):
            result[key] = {"id_type": val.bl_rna.identifier, "name": val.name_full}
        elif hasattr(val, "bl_rna"):
            result[key] = {"rna_type": val.bl_rna.identifier}
        else:
            result[key] = plain(val)
    return result


def _physics_rna_values(block, depth=0):
    if block is None or depth > 8:
        return {}
    result = {}
    for prop, key, value in _walk_rna_properties(
            block, skip=("name", "name_full", "type", "id_type"),
            skip_collections=True):
        if isinstance(value, bpy.types.ID):
            result[key] = plain(value)
        elif hasattr(value, "bl_rna"):
            if prop.type == "POINTER":
                result[key] = _physics_rna_values(value, depth + 1)
        else:
            serialized = plain(value)
            if serialized != "":
                result[key] = serialized
    return result




def _physics_point_cache(block):
    cache = getattr(block, "point_cache", None) if block is not None else None
    if cache is None:
        return None
    result = _physics_rna_values(cache)
    result["is_baked"] = bool(getattr(cache, "is_baked", False))
    return result


def _physics_modifier_settings(obj, modifier):
    result = {}
    if modifier.type == "DYNAMIC_PAINT":
        settings = getattr(modifier, "brush_settings", None)
        role = "brush" if settings is not None else "canvas"
        if settings is None:
            settings = getattr(modifier, "canvas_settings", None)
        result["role"] = role
        result.update(_physics_rna_values(settings))
        if role == "canvas" and settings is not None:
            surfaces = []
            for surface in getattr(settings, "canvas_surfaces", ()):
                surface_values = _physics_rna_values(surface)
                point_cache = _physics_point_cache(surface)
                if point_cache is not None:
                    surface_values["point_cache"] = point_cache
                surfaces.append(surface_values)
            result["canvas_surfaces"] = surfaces
    elif modifier.type == "PARTICLE_SYSTEM":
        particle_system = getattr(modifier, "particle_system", None)
        settings = getattr(particle_system, "settings", None) if particle_system else None
        result = _physics_rna_values(settings)
        if particle_system:
            result["seed"] = int(particle_system.seed)
            point_cache = _physics_point_cache(particle_system)
            if point_cache is not None:
                result["point_cache"] = point_cache
    elif modifier.type == "FLUID":
        fluid_type = getattr(modifier, "fluid_type", "")
        settings_name = {
            "DOMAIN": "domain_settings",
            "FLOW": "flow_settings",
            "EFFECTOR": "effector_settings",
        }.get(fluid_type)
        settings = getattr(modifier, settings_name, None) if settings_name else None
        result["role"] = fluid_type
        result.update(_physics_rna_values(settings))
    else:
        settings = getattr(modifier, "settings", None)
        result.update(_physics_rna_values(settings))
    if modifier.type == "CLOTH":
        collision_settings = getattr(modifier, "collision_settings", None)
        if collision_settings is not None:
            result["collision_settings"] = _physics_rna_values(collision_settings)
    point_cache = _physics_point_cache(modifier)
    if point_cache is not None:
        result["point_cache"] = point_cache
    return result
def modifier_params(modifier):
    omitted = (
        "type", "name", "execution_time", "persistent_uid", "show_viewport", "show_render",
    )
    if modifier.type == "PARTICLE_INSTANCE":
        omitted += ("particle_system",)
    if modifier.type in ("VERTEX_WEIGHT_EDIT", "VERTEX_WEIGHT_PROXIMITY"):
        omitted += ("map_curve",)
    params = rna_props(modifier, omitted)
    if modifier.type in ("VERTEX_WEIGHT_EDIT", "VERTEX_WEIGHT_PROXIMITY"):
        params["map_curve"] = _curve_mapping_points(modifier.map_curve)
    bound_property = "is_bound" if hasattr(modifier, "is_bound") else "is_bind"
    if hasattr(modifier, bound_property):
        params["is_bound"] = bool(getattr(modifier, bound_property))
    if modifier.type == "MESH_CACHE" and getattr(modifier, "filepath", ""):
        params["cache_filepath"] = bpy.path.abspath(modifier.filepath)
    if modifier.type == "UV_PROJECT":
        params["projectors"] = [
            {"object": plain(projector.object) if projector.object else None}
            for projector in modifier.projectors
        ]
    return params
def _curve_mapping_points(mapping):
    if mapping is None or not mapping.curves:
        return []
    return [
        [float(point.location[0]), float(point.location[1])]
        for point in mapping.curves[0].points
    ]


def _set_curve_mapping(modifier, values):
    mapping = getattr(modifier, "map_curve", None)
    if mapping is None or not mapping.curves or len(values) < 2:
        raise ValueError("modifier map_curve needs at least two control points")
    points = [(float(point[0]), float(point[1])) for point in values]
    curve = mapping.curves[0]
    for point in list(curve.points)[1:-1]:
        curve.points.remove(point)
    curve.points[0].location = points[0]
    curve.points[-1].location = points[-1]
    for x, y in points[1:-1]:
        curve.points.new(x, y)
    mapping.update()






def tree_dump(tree):
    if tree is None:
        return None
    nodes = []
    for node in tree.nodes:
        sockets = []
        for socket in node.inputs:
            if node.bl_idname == "GeometryNodeTransform" and socket.name in ("Mode", "Transform", "Rotation"):
                continue
            if not socket.name:
                continue
            try:
                default = plain(socket.default_value)
            except Exception:
                default = None
            socket_type = getattr(socket, "bl_socket_idname", getattr(socket, "bl_idname", socket.type))
            identifier = (socket.name
                          if node.bl_idname in ("NodeGroupInput", "NodeGroupOutput")
                          else socket.identifier)
            sockets.append({"name": socket.name, "identifier": identifier, "type": socket_type,
                            "enabled": socket.enabled, "default": default})
        custom_properties = custom_props(node)
        node_id = custom_properties.pop("potter.id", None)
        simple_material = custom_properties.pop("potter_simple_material", False)
        properties = rna_props(node, ("inputs", "outputs", "internal_links", "parent", "select", "location", "dimensions"))
        if node.bl_idname == "GeometryNodeMeshCube":
            properties = {}
        if simple_material:
            properties = {"potter_simple_material": True}
            sockets = []
        nodes.append({"name": node.name, "potter_id": node_id,
                      "type": potter_node_type(node.bl_idname, tree.bl_rna.identifier), "label": node.label,
                      "location": plain(node.location), "inputs": sockets,
                      "properties": properties, "custom_properties": custom_properties})
    links = [{
        "from_node": link.from_node.name,
        "from_socket": (link.from_socket.name
                        if link.from_node.bl_idname == "NodeGroupInput"
                        else link.from_socket.identifier),
        "to_node": link.to_node.name,
        "to_socket": (link.to_socket.name
                      if link.to_node.bl_idname == "NodeGroupOutput"
                      else link.to_socket.identifier),
    } for link in tree.links]
    result = {"name": tree.get("potter.graph_name", tree.name_full),
              "type": tree.bl_rna.identifier, "potter_id": tree.get("potter.id"),
              "nodes": nodes, "links": links}
    interface = getattr(tree, "interface", None)
    if interface is not None:
        sockets = {"inputs": [], "outputs": []}
        for item in getattr(interface, "items_tree", ()):
            if getattr(item, "item_type", None) != "SOCKET":
                continue
            direction = getattr(item, "in_out", "")
            if direction not in ("INPUT", "OUTPUT"):
                continue
            entry = {"id": item.identifier, "name": item.name,
                     "socket_type": getattr(item, "socket_type", None) or getattr(item, "bl_socket_idname", "")}
            try:
                entry["default"] = plain(item.default_value)
            except Exception:
                entry["default"] = None
            sockets["inputs" if direction == "INPUT" else "outputs"].append(entry)
        result["interface"] = sockets
    return result


def _id_map(block, property_name):
    try:
        value = json.loads(block.get(property_name, "{}"))
    except Exception:
        value = {}
    return value if isinstance(value, dict) else {}


def images_dump():
    result = []
    for image in bpy.data.images:
        if _startup_metadata_datablock("images", image):
            continue
        try:
            image_model = json.loads(image.get("potter.image_json", "null"))
        except Exception:
            image_model = None
        try:
            filepath = bpy.path.abspath(image.filepath, library=image.library)
        except TypeError:
            filepath = bpy.path.abspath(image.filepath)
        try:
            colorspace = image.colorspace_settings.name
        except Exception:
            colorspace = None
        result.append({"name": image.name_full, "potter_id": image.get("potter.id"),
                       "library": library_reference(image), "filepath": filepath,
                       "source": image.source, "colorspace": colorspace,
                       "potter_image": image_model})
    return result


def _resource_record(kind, owner, path, packed_file=None):
    resolved = bpy.path.abspath(path) if path else None
    packed_data = None
    if packed_file is not None:
        try:
            packed_data = base64.b64encode(packed_file.data).decode("ascii")
        except Exception:
            packed_data = None
    return {"kind": kind, "owner": owner, "path": resolved,
            "packed": packed_data is not None, "packed_data": packed_data}

def _library_resolved_path(library):
    # Library.filepath is relative to the file that links it (the parent
    # library, or the main file for top-level libraries), not to itself.
    return bpy.path.abspath(library.filepath, library=library.parent)


def library_reference(block):
    library = getattr(block, "library", None)
    if library is None:
        return None
    return {"name": library.name_full, "uri": library.filepath,
            "path": _library_resolved_path(library),
            "source_id": block.get("potter.id")}


def _cache_file_settings(cache_file):
    return {key: plain(getattr(cache_file, key))
            for key in ("is_sequence", "override_frame", "frame", "frame_offset",
                        "forward_axis", "up_axis", "scale", "velocity_name",
                        "velocity_unit")}


def _cache_file_resource(cache_file):
    record = _resource_record("alembic_cache", cache_file.name_full,
                              cache_file.filepath)
    record["cache_file"] = _cache_file_settings(cache_file)
    return record


def _object_solver_inverse_map(owner):
    encoded = owner.get("potter.object_solver_inverses_json")
    return json.loads(encoded) if encoded else {}

def _store_object_solver_inverse(owner, constraint_name, inverse_matrix,
                                 bone_name=None, inverse_frame=None):
    inverse_map = _object_solver_inverse_map(owner)
    entry = {"matrix": inverse_matrix}
    if inverse_frame is not None:
        entry["frame"] = float(inverse_frame)
    if bone_name is None:
        inverse_map.setdefault("objects", {})[constraint_name] = entry
    else:
        inverse_map.setdefault("bones", {}).setdefault(bone_name, {})[
            constraint_name] = entry
    owner["potter.object_solver_inverses_json"] = json.dumps(
        inverse_map, sort_keys=True)


def _constraint_dump(constraint, inverse_matrix=None):
    result = {"name": constraint.name, "type": constraint.type,
              "properties": rna_props(constraint),
              "custom_properties": custom_props(constraint)}
    if constraint.type == "ARMATURE":
        result["properties"]["targets"] = [
            {
                "target": ({"id_type": item.target.bl_rna.identifier,
                            "name": item.target.name_full} if item.target else None),
                "subtarget": item.subtarget,
                "weight": float(item.weight),
            }
            for item in constraint.targets
        ]
    if constraint.type == "TRANSFORM_CACHE":
        cache_file = getattr(constraint, "cache_file", None)
        result["cache_file_name"] = cache_file.name_full if cache_file else None
        result["cache_filepath"] = (
            bpy.path.abspath(cache_file.filepath) if cache_file and cache_file.filepath else None)
        result["cache_file_settings"] = (
            _cache_file_settings(cache_file) if cache_file else {})
    if constraint.type == "OBJECT_SOLVER" and inverse_matrix is not None:
        if isinstance(inverse_matrix, dict):
            result["inverse_matrix"] = inverse_matrix.get("matrix")
            if inverse_matrix.get("frame") is not None:
                result["inverse_frame"] = inverse_matrix["frame"]
        else:
            result["inverse_matrix"] = inverse_matrix
    return result

def _restore_armature_targets(constraint, properties, id_lookup, owner):
    for values in properties.pop("targets", []):
        target = constraint.targets.new()
        set_rna(target, values, id_lookup, strict=True, owner=owner)



def libraries_dump():
    return [{"name": library.name_full, "uri": library.filepath,
             "path": _library_resolved_path(library)}
            for library in bpy.data.libraries]


def volume_filepath(volume):
    filepath = bpy.path.abspath(volume.filepath) if volume.filepath else ""
    return "" if filepath and os.path.isdir(filepath) else filepath

def resources_dump():
    result = []
    for image in bpy.data.images:
        if _startup_metadata_datablock("images", image):
            continue
        try:
            filepath = bpy.path.abspath(image.filepath, library=image.library)
        except TypeError:
            filepath = bpy.path.abspath(image.filepath)
        record = _resource_record("image", image.name_full, filepath,
                                 getattr(image, "packed_file", None))
        if record["path"] or record["packed"]:
            result.append(record)
    for curve in bpy.data.curves:
        if getattr(getattr(curve, "bl_rna", None), "identifier", "") != "TextCurve":
            continue
        font = curve.font
        if font is not None and (font.filepath or getattr(font, "packed_file", None)):
            result.append(_resource_record("font", curve.name_full, font.filepath,
                                           getattr(font, "packed_file", None)))
    for volume in getattr(bpy.data, "volumes", ()):
        filepath = volume_filepath(volume)
        if filepath:
            result.append(_resource_record("volume", volume.name_full, filepath))
    for scene in bpy.data.scenes:
        editor = getattr(scene, "sequence_editor", None)
        sequences = (getattr(editor, "strips", None) or
                     getattr(editor, "sequences", ())) if editor else ()
        for strip in sequences:
            if getattr(strip, "type", None) == "SOUND":
                sound = getattr(strip, "sound", None)
                if sound is not None:
                    result.append(_resource_record("sound", strip.name, sound.filepath,
                                                   getattr(sound, "packed_file", None)))
            elif getattr(strip, "type", None) == "IMAGE":
                directory = getattr(strip, "directory", "")
                for element in getattr(strip, "elements", ()):
                    result.append(_resource_record(
                        "image_sequence", strip.name,
                        os.path.join(directory, element.filename)))
            elif hasattr(strip, "filepath"):
                kind = "movie" if getattr(strip, "type", None) == "MOVIE" else "image"
                result.append(_resource_record(kind, strip.name, strip.filepath,
                                               getattr(strip, "packed_file", None)))
    for cache_file in getattr(bpy.data, "cache_files", ()):
        if cache_file.filepath:
            result.append(_cache_file_resource(cache_file))
    for clip in getattr(bpy.data, "movieclips", ()):
        if clip.filepath:
            result.append(_resource_record("movie_clip", clip.name_full, clip.filepath))
    for obj in bpy.data.objects:
        for modifier in obj.modifiers:
            if modifier.type == "MESH_CACHE" and getattr(modifier, "filepath", ""):
                result.append(_resource_record(
                    "mesh_cache", "%s:%s" % (obj.name_full, modifier.name),
                    modifier.filepath))
    for library in bpy.data.libraries:
        record = _resource_record("library", library.name_full,
                                  _library_resolved_path(library))
        record["uri"] = library.filepath
        result.append(record)
    return result

def _tracking_object_pose_dump(tracking_object):
    reconstruction = getattr(tracking_object, "reconstruction", None)
    samples = getattr(reconstruction, "cameras", reconstruction or ())
    result = []
    try:
        for sample in samples:
            matrix = matrix_rows(getattr(sample, "matrix", None))
            if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                continue
            result.append({
                "frame": float(getattr(sample, "frame", 1.0)),
                "matrix": [matrix[row][column]
                           for column in range(4) for row in range(4)],
                "average_error": float(getattr(sample, "average_error", 0.0)),
            })
    except Exception:
        return []
    return result


def movie_clip_dump(clip):
    tracking = clip.tracking
    tracks = []
    objects = []
    points = []
    for tracking_object in tracking.objects:
        track_ids = []
        for track in tracking_object.tracks:
            track_id = "%s:%s" % (tracking_object.name, track.name)
            track_ids.append(track_id)
            markers = []
            for marker in track.markers:
                search_min = list(plain(marker.search_min))
                search_max = list(plain(marker.search_max))
                markers.append({
                    "frame": float(marker.frame),
                    "co": plain(marker.co),
                    "pattern_corners": plain(marker.pattern_corners),
                    "search_area": [
                        search_min,
                        [search_max[0], search_min[1]],
                        search_max,
                        [search_min[0], search_max[1]],
                    ],
                    "disabled": bool(marker.mute),
                })
            tracks.append({"id": track_id, "name": track.name, "markers": markers})
            if bool(getattr(track, "has_bundle", False)):
                points.append({"track": track_id, "co": plain(track.bundle)})
        object_reconstruction = tracking_object.reconstruction
        objects.append({
            "id": tracking_object.name,
            "name": tracking_object.name,
            "tracks": track_ids,
            "reconstruction": _tracking_object_pose_dump(tracking_object),
            "reconstruction_is_valid": bool(object_reconstruction.is_valid),
            "reconstruction_average_error": float(object_reconstruction.average_error),
            "scale": float(getattr(tracking_object, "scale", 1.0)),
        })
    camera = tracking.camera
    reconstruction_cameras = []
    reconstruction = tracking.reconstruction
    for sample in getattr(reconstruction, "cameras", ()):
        matrix = matrix_rows(getattr(sample, "matrix", None))
        if len(matrix) == 4 and all(len(row) == 4 for row in matrix):
            reconstruction_cameras.append({
                "frame": int(sample.frame),
                "matrix": [row[:4] for row in matrix[:3]],
                "average_error": float(sample.average_error),
                "matrix_is_camera_to_world": True,
            })
    principal = plain(getattr(camera, "principal_point", (0.5, 0.5)))
    if not isinstance(principal, list) or len(principal) < 2:
        principal = [0.5, 0.5]
    size = tuple(clip.size)
    fps = float(clip.fps)
    if not math.isfinite(fps) or fps <= 0.0:
        render = bpy.context.scene.render
        fps_base = float(render.fps_base)
        fps = float(render.fps) / fps_base if fps_base > 0.0 else float(render.fps)
    return {
        "name": clip.name_full,
        "potter_id": clip.get("potter.id"),
        "filepath": bpy.path.abspath(clip.filepath) if clip.filepath else "",
        "frame_start": int(clip.frame_start),
        "width": int(size[0]),
        "height": int(size[1]),
        "fps": fps,
        "tracking": {
            "tracks": tracks,
            "objects": objects,
            "camera": {
                "focal_mm": float(camera.focal_length),
                "sensor_width_mm": float(camera.sensor_width),
                "principal": principal[:2],
                "units": camera.units,
                "pixel_aspect": float(camera.pixel_aspect),
                "distortion_model": camera.distortion_model,
                "k1": float(camera.k1),
                "k2": float(camera.k2),
                "k3": float(camera.k3),
                "division_k1": float(camera.division_k1),
                "division_k2": float(camera.division_k2),
                "nuke_k1": float(camera.nuke_k1),
                "nuke_k2": float(camera.nuke_k2),
                "nuke_p1": float(camera.nuke_p1),
                "nuke_p2": float(camera.nuke_p2),
                "brown_k1": float(camera.brown_k1),
                "brown_k2": float(camera.brown_k2),
                "brown_k3": float(camera.brown_k3),
                "brown_k4": float(camera.brown_k4),
                "brown_p1": float(camera.brown_p1),
                "brown_p2": float(camera.brown_p2),
            },
            "reconstruction": {
                "cameras": reconstruction_cameras,
                "points": points,
                "is_valid": bool(reconstruction.is_valid),
                "average_error": float(reconstruction.average_error),
            },
            "plane_tracks": [],
        },
    }


def volume_dump(volume):
    try:
        volume.grids.load()
    except Exception:
        pass
    return {"name": volume.name_full, "potter_id": volume.get("potter.id"),
            "library": library_reference(volume),
            "filepath": volume_filepath(volume),
            "grids": [grid.name for grid in volume.grids],
            "custom_properties": custom_props(volume),
            "fake_user": bool(getattr(volume, "use_fake_user", False)),
            "users": int(volume.users)}


def _mesh_shape_keys(mesh):
    key_data = getattr(mesh, "shape_keys", None)
    blocks = getattr(key_data, "key_blocks", ()) if key_data else ()
    if not blocks:
        return {"absolute": False, "evaluation_time": 0.0, "basis": [], "keys": []}
    basis = blocks[0]
    basis_data = [{"index": index, "co": plain(point.co)} for index, point in enumerate(basis.data)]
    key_ids = _id_map(key_data, "potter.shape_key_ids_json")
    keys = []
    for key in list(blocks)[1:]:
        keys.append({"name": key.name, "potter_id": key_ids.get(key.name),
                     "value": float(key.value), "mute": bool(getattr(key, "mute", False)),
                     "slider_min": float(key.slider_min), "slider_max": float(key.slider_max),
                     "frame": float(getattr(key, "frame", 0.0)),
                     "relative_key": key.relative_key.name if key.relative_key else None,
                     "vertex_group": key.vertex_group,
                     "positions": [{"index": index, "co": plain(point.co)}
                                   for index, point in enumerate(key.data)]})
    animation_data = getattr(key_data, "animation_data", None)
    action = getattr(animation_data, "action", None) if animation_data else None
    action_slot = getattr(animation_data, "action_slot", None) if animation_data else None
    muted_action_curves = []
    if action is not None and action_slot is not None and hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                if strip.type != "KEYFRAME":
                    continue
                try:
                    channelbag = strip.channelbag(action_slot)
                except Exception:
                    channelbag = None
                if channelbag is not None:
                    muted_action_curves.extend(
                        curve.data_path for curve in channelbag.fcurves if curve.mute)
    elif action is not None and hasattr(action, "fcurves"):
        muted_action_curves.extend(
            curve.data_path for curve in action.fcurves if curve.mute)
    return {"absolute": not bool(getattr(key_data, "use_relative", True)),
            "evaluation_time": float(getattr(key_data, "eval_time", 0.0)),
            "action": action.name_full if action else None,
            "action_slot": (getattr(action_slot, "name_display",
                                    getattr(action_slot, "name", ""))
                            if action_slot else None),
            "muted_action_curves": sorted(set(muted_action_curves)),
            "basis": basis_data, "keys": keys}


def _armature_bone_rolls(armature):
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_mode = previous_active.mode if previous_active else "OBJECT"
    previous_selection = [obj for obj in view_layer.objects if obj.select_get()]
    probe_data = armature.copy()
    probe_object = bpy.data.objects.new("_potter_bone_roll_probe", probe_data)
    scene.collection.objects.link(probe_object)

    def set_mode(obj, mode):
        with bpy.context.temp_override(
            scene=scene, view_layer=view_layer, object=obj, active_object=obj,
            selected_objects=[obj], selected_editable_objects=[obj],
        ):
            bpy.ops.object.mode_set(mode=mode)

    try:
        if previous_active and previous_mode != "OBJECT":
            set_mode(previous_active, "OBJECT")
        for obj in previous_selection:
            obj.select_set(False)
        probe_object.select_set(True)
        view_layer.objects.active = probe_object
        set_mode(probe_object, "EDIT")
        rolls = {bone.name: float(bone.roll) for bone in probe_data.edit_bones}
        set_mode(probe_object, "OBJECT")
        return rolls
    finally:
        if probe_object.mode != "OBJECT":
            set_mode(probe_object, "OBJECT")
        probe_object.select_set(False)
        bpy.data.objects.remove(probe_object, do_unlink=True)
        if probe_data.users == 0:
            bpy.data.armatures.remove(probe_data)
        for obj in view_layer.objects:
            obj.select_set(obj in previous_selection)
        if previous_active and previous_active.name in view_layer.objects:
            view_layer.objects.active = previous_active
            if previous_mode != "OBJECT":
                set_mode(previous_active, previous_mode)


def _armature_bbone_settings(bone):
    custom_handle_fields = ("bbone_custom_handle_start", "bbone_custom_handle_end")
    settings = {}
    for prop in bone.bl_rna.properties:
        key = prop.identifier
        if not key.startswith("bbone_"):
            continue
        value = getattr(bone, key)
        if key in custom_handle_fields:
            value = value.name if value is not None else None
        settings[key] = plain(value)
    return settings


def _armature_dump(armature):
    bones = []
    bone_ids = _id_map(armature, "potter.bone_ids_json")
    bone_rolls = _armature_bone_rolls(armature)
    for bone in armature.bones:
        bones.append({"name": bone.name, "potter_id": bone_ids.get(bone.name),
                      "parent": bone.parent.name if bone.parent else None,
                      "head": plain(bone.head_local), "tail": plain(bone.tail_local),
                      "roll": bone_rolls[bone.name], "deform": bool(bone.use_deform),
                      "inherit_rotation": bool(bone.use_inherit_rotation),
                      "use_connect": bool(bone.use_connect),
                      "bbone_settings": _armature_bbone_settings(bone)})
    return {"name": armature.name_full, "potter_id": armature.get("potter.id"),
            "library": library_reference(armature),
            "bones": bones, "custom_properties": custom_props(armature),
            "fake_user": bool(armature.use_fake_user)}



def curve_dump(curve):
    if getattr(getattr(curve, "bl_rna", None), "identifier", "") == "TextCurve":
        font = curve.font
        font_path = bpy.path.abspath(font.filepath) if font and font.filepath else "builtin"
        return {
            "name": curve.name_full,
            "potter_id": curve.get("potter.id"),
            "library": library_reference(curve),
            "type": "FONT",
            "resolution_u": int(getattr(curve, "resolution_u", 12)),
            "bevel_depth": float(curve.bevel_depth),
            "bevel_resolution": int(curve.bevel_resolution),
            "extrude": float(curve.extrude),
            "fill_mode": str(curve.fill_mode),
            "splines": [],
            "text": {
                "body": curve.body,
                "font": font_path,
                "font_name": font.name_full if font else "Bfont",
                "size": float(curve.size),
                "align_x": curve.align_x.lower(),
                "align_y": {
                    "BASELINE": "baseline",
                    "TOP_BASELINE": "top",
                    "CENTER": "center",
                    "BOTTOM_BASELINE": "bottom",
                }.get(curve.align_y, curve.align_y.lower()),
                "extrude": float(curve.extrude),
                "bevel_depth": float(curve.bevel_depth),
                "character_spacing": float(getattr(curve, "space_character", 1.0)),
                "word_spacing": float(getattr(curve, "space_word", 1.0)),
                "line_spacing": float(getattr(curve, "space_line", 1.0)),
                "shear": float(getattr(curve, "shear", 0.0)),
                "offset_x": float(getattr(curve, "offset_x", 0.0)),
                "offset_y": float(getattr(curve, "offset_y", 0.0)),
                "small_caps_scale": float(getattr(curve, "small_caps_scale", 0.75)),
            },
            "custom_properties": custom_props(curve),
            "fake_user": bool(curve.use_fake_user),
            "users": int(curve.users),
        }
    splines = []
    for spline in curve.splines:
        points = []
        source_points = (spline.bezier_points if spline.type == "BEZIER" else spline.points)
        for point in source_points:
            if spline.type == "BEZIER":
                points.append({
                    "co": plain(point.co),
                    "handle_left": plain(point.handle_left),
                    "handle_right": plain(point.handle_right),
                    "handle_type": point.handle_left_type,
                    "weight": 1.0,
                    "radius": float(point.radius),
                    "tilt": float(point.tilt),
                })
            else:
                points.append({
                    "co": plain(point.co)[:3],
                    "handle_left": plain(point.co)[:3],
                    "handle_right": plain(point.co)[:3],
                    "handle_type": "AUTO",
                    "weight": float(point.co[3]),
                    "radius": float(getattr(point, "radius", 1.0)),
                    "tilt": float(getattr(point, "tilt", 0.0)),
                })
        splines.append({
            "type": spline.type.lower(),
            "points": points,
            "order": int(getattr(spline, "order_u", 3)),
            "cyclic": bool(getattr(spline, "use_cyclic_u", False)),
            "resolution": int(getattr(spline, "resolution_u", 12)),
            "use_endpoint": bool(getattr(spline, "use_endpoint_u", False)),
            "points_u": int(getattr(spline, "points_u", 0)),
            "points_v": int(getattr(spline, "points_v", 0)),
            "order_v": int(getattr(spline, "order_v", 3)),
            "cyclic_v": bool(getattr(spline, "use_cyclic_v", False)),
            "resolution_v": int(getattr(spline, "resolution_v", 12)),
            "use_endpoint_v": bool(getattr(spline, "use_endpoint_v", False)),
        })
    curve_type = getattr(curve, "type", "CURVE")
    eval_time_fcurves = []
    eval_time_action_name = None
    eval_time_action_slot_name = None
    if curve_type == "CURVE":
        animation_data = getattr(curve, "animation_data", None)
        action = getattr(animation_data, "action", None)
        action_slot = getattr(animation_data, "action_slot", None)
        eval_time_action_name = action.name_full if action is not None else None
        eval_time_action_slot_name = (
            getattr(action_slot, "name_display", getattr(action_slot, "name", None))
            if action_slot is not None else None)
        if action is not None and action_slot is not None and hasattr(action, "layers"):
            for layer in action.layers:
                for strip in layer.strips:
                    try:
                        channelbag = strip.channelbag(action_slot)
                    except Exception:
                        channelbag = None
                    if channelbag is not None:
                        eval_time_fcurves.extend(
                            dump_fcurve(fcurve) for fcurve in channelbag.fcurves
                            if fcurve.data_path == "eval_time")
        elif action is not None and hasattr(action, "fcurves"):
            eval_time_fcurves.extend(
                dump_fcurve(fcurve) for fcurve in action.fcurves
                if fcurve.data_path == "eval_time")
    result = {
        "eval_time_action_name": eval_time_action_name,
        "eval_time_action_slot_name": eval_time_action_slot_name,
        "name": curve.name_full,
        "potter_id": curve.get("potter.id"),
        "library": library_reference(curve),
        "type": curve_type,
        "resolution_u": int(getattr(curve, "resolution_u", 12)),
        "bevel_depth": float(getattr(curve, "bevel_depth", 0.0)),
        "bevel_resolution": int(getattr(curve, "bevel_resolution", 0)),
        "extrude": float(getattr(curve, "extrude", 0.0)),
        "fill_mode": str(getattr(curve, "fill_mode", "NONE")),
        "splines": splines,
        "custom_properties": custom_props(curve),
        "fake_user": bool(curve.use_fake_user),
        "users": int(curve.users),
    }
    if curve_type == "CURVE":
        result.update({
            "twist_mode": str(getattr(curve, "twist_mode", "MINIMUM")),
            "use_path": bool(curve.use_path),
            "path_duration": int(curve.path_duration),
            "eval_time": float(curve.eval_time),
            "eval_time_fcurves": eval_time_fcurves,
        })
    if curve_type == "SURFACE":
        result["surface"] = [{
            "points": [[{"co": point["co"], "weight": point["weight"]}
                        for point in spline["points"][row * spline["points_u"]:
                                                     (row + 1) * spline["points_u"]]]
                       for row in range(spline["points_v"])],
            "order_u": spline["order"],
            "order_v": spline["order_v"],
            "resolution": [spline["resolution"], spline["resolution_v"]],
            "cyclic_u": spline["cyclic"],
            "cyclic_v": spline["cyclic_v"],
            "use_endpoint_u": spline["use_endpoint"],
            "use_endpoint_v": spline["use_endpoint_v"],
        } for spline in splines]
    elif curve_type == "FONT":
        result["text"] = {
            "body": curve.body,
            "font": bpy.path.abspath(curve.font.filepath) if curve.font else "builtin",
            "size": float(curve.size),
            "align_x": curve.align_x.lower(),
            "align_y": curve.align_y.lower(),
            "extrude": float(curve.extrude),
            "bevel_depth": float(curve.bevel_depth),
        }
    return result

def _grease_attribute(drawing, name, color=False):
    groups = [getattr(drawing, "attributes", None)]
    if color:
        groups.append(getattr(drawing, "color_attributes", None))
    for group in groups:
        if group is not None:
            try:
                attr = group.get(name)
                if attr is not None:
                    return attr
            except Exception:
                pass
    return None


def _attribute_value(item):
    for field in ("vector", "color", "value"):
        if hasattr(item, field):
            return plain(getattr(item, field))
    raise ValueError("unsupported Grease Pencil drawing attribute value")


def _grease_pencil_dump(data):
    layers = []
    for layer in data.layers:
        frames = []
        for frame in layer.frames:
            drawing = frame.drawing
            offsets = getattr(drawing, "curve_offsets", None)
            if offsets is None:
                raise ValueError("Grease Pencil drawing has no curve offsets")
            offsets = [int(getattr(item, "value", item)) for item in offsets]
            strokes = list(getattr(drawing, "strokes", ()))
            if len(offsets) != len(strokes) + 1:
                raise ValueError("Grease Pencil curve offsets do not match strokes")
            point_count = offsets[-1] if offsets else 0
            attrs = {name: _grease_attribute(drawing, name, color=(name == "vertex_color"))
                     for name in ("position", "pressure", "radius", "opacity", "time",
                                  "cyclic", "fill", "material_index")}

            def values_for(name, attr, fallback_name):
                if attr is not None:
                    return [_attribute_value(item) for item in attr.data]
                values = []
                for stroke in strokes:
                    points = list(getattr(stroke, "points", ()))
                    for point in points:
                        if hasattr(point, fallback_name):
                            values.append(plain(getattr(point, fallback_name)))
                        else:
                            values = []
                            break
                    if not values and point_count:
                        break
                if values:
                    return values
                if point_count == 0:
                    return []
                raise ValueError("Grease Pencil drawing is missing required %s values" % name)

            point_values = {name: values_for(name, attrs[name], name)
                            for name in ("position", "pressure", "radius", "opacity", "time")}
            for name, values in point_values.items():
                if len(values) != point_count:
                    raise ValueError("Grease Pencil %s attribute count does not match point count" % name)
            cyclic_values = values_for("cyclic", attrs["cyclic"], "cyclic") if strokes else []
            material_values = values_for("material_index", attrs["material_index"], "material_index") if strokes else []
            fill_values = values_for("fill", attrs["fill"], "fill") if attrs["fill"] else [
                plain(stroke.fill) if hasattr(stroke, "fill") else None for stroke in strokes]
            drawing_strokes = []
            for stroke_index, stroke in enumerate(strokes):
                start, end = offsets[stroke_index], offsets[stroke_index + 1]
                material_index = int(material_values[stroke_index])
                material = data.materials[material_index] if 0 <= material_index < len(data.materials) else None
                fill = bool(fill_values[stroke_index]) if fill_values and fill_values[stroke_index] is not None else bool(
                    getattr(getattr(material, "grease_pencil", None), "show_fill", False))
                points = []
                for index in range(start, end):
                    points.append({"position": point_values["position"][index],
                                   "pressure": point_values["pressure"][index],
                                   "radius": point_values["radius"][index],
                                   "opacity": point_values["opacity"][index],
                                   "time": point_values["time"][index]})
                drawing_strokes.append({"points": points, "cyclic": bool(cyclic_values[stroke_index]),
                                        "fill": fill, "material": material.name_full if material else None})
            frames.append({"frame": int(frame.frame_number), "strokes": drawing_strokes})
        layers.append({"name": layer.name, "opacity": float(layer.opacity),
                       "visible": not bool(layer.hide), "frames": frames})
    return {"name": data.name_full, "potter_id": data.get("potter.id"),
            "library": library_reference(data), "layers": layers,
            "custom_properties": custom_props(data), "fake_user": bool(data.use_fake_user)}


def _node_group_interface_socket(socket):
    return {"id": socket.identifier, "name": socket.name,
            "socket_type": getattr(socket, "socket_type", None) or getattr(socket, "bl_socket_idname", "")}


def _modifier_node_inputs(modifier):
    group = getattr(modifier, "node_group", None)
    if group is None:
        return {}
    result = {}
    interface = getattr(group, "interface", None)
    for socket in getattr(interface, "items_tree", ()) if interface else ():
        if getattr(socket, "item_type", None) != "SOCKET" or getattr(socket, "in_out", None) != "INPUT":
            continue
        identifier = socket.identifier
        try:
            if identifier in modifier:
                result[identifier] = plain(modifier[identifier])
        except Exception:
            continue
    return result


def mesh_dump(mesh):
    vertices = [{"co": plain(v.co)} for v in mesh.vertices]
    edges = [{"v": list(edge.vertices), "sharp": bool(getattr(edge, "use_edge_sharp", False)),
              "seam": bool(getattr(edge, "use_seam", False))} for edge in mesh.edges]
    polygons = [{"v": list(poly.vertices), "material_index": int(poly.material_index),
                 "smooth": bool(poly.use_smooth)} for poly in mesh.polygons]
    uv_layers = []
    for layer in mesh.uv_layers:
        uv_layers.append({"name": layer.name, "values": [plain(item.uv) for item in layer.data]})
    attrs = []
    try:
        for attr in mesh.attributes:
            vals = []
            for item in attr.data:
                if hasattr(item, "vector"):
                    vals.append(plain(item.vector))
                elif hasattr(item, "color"):
                    vals.append(plain(item.color))
                elif hasattr(item, "value"):
                    vals.append(plain(item.value))
                else:
                    vals.append(None)
            attrs.append({"name": attr.name, "domain": attr.domain, "data_type": attr.data_type, "values": vals})
    except Exception:
        pass
    skin_vertices = []
    if len(mesh.skin_vertices):
        skin_vertices = [{"radius": plain(vertex.radius),
                          "root": bool(vertex.use_root),
                          "loose": bool(vertex.use_loose)}
                         for vertex in mesh.skin_vertices[0].data]
    return {"name": mesh.name_full, "potter_id": mesh.get("potter.id"),
            "library": library_reference(mesh), "vertices": vertices,
            "edges": edges, "polygons": polygons, "uv_layers": uv_layers, "attributes": attrs,
            "skin_vertices": skin_vertices,
            "shape_keys": _mesh_shape_keys(mesh),
            "descriptor": json.loads(mesh.get("potter.descriptor_json", "null")),
            "custom_properties": custom_props(mesh), "fake_user": bool(mesh.use_fake_user),
            "users": int(mesh.users), "rna_properties": rna_props(mesh, ("vertices", "edges", "polygons", "loops", "materials", "attributes", "uv_layers"))}




def _vertex_groups_dump(obj):
    result = []
    group_ids = _id_map(obj, "potter.vertex_group_ids_json")
    for group in obj.vertex_groups:
        weights = []
        for vertex in obj.data.vertices if obj.type == "MESH" else ():
            for assignment in vertex.groups:
                if assignment.group == group.index:
                    weights.append({"vertex_index": int(vertex.index),
                                    "weight": float(assignment.weight)})
                    break
        result.append({"name": group.name, "potter_id": group_ids.get(group.name),
                       "weights": weights})
    return result


def _pose_dump(obj):
    result = []
    if obj.type != "ARMATURE" or obj.pose is None:
        return result
    inverse_map = _object_solver_inverse_map(obj).get("bones", {})
    for bone in obj.pose.bones:
        constraints = inverse_map.get(bone.name, {})
        result.append({"bone": bone.name, "location": plain(bone.location),
                       "rotation_mode": bone.rotation_mode,
                       "rotation_quaternion": [
                           bone.rotation_quaternion.w,
                           bone.rotation_quaternion.x,
                           bone.rotation_quaternion.y,
                           bone.rotation_quaternion.z,
                       ],
                       "scale": plain(bone.scale),
                       "lock_ik": [bool(bone.lock_ik_x), bool(bone.lock_ik_y),
                                  bool(bone.lock_ik_z)],
                       "use_ik_limit": [bool(bone.use_ik_limit_x),
                                        bool(bone.use_ik_limit_y),
                                        bool(bone.use_ik_limit_z)],
                       "ik_min": [float(bone.ik_min_x), float(bone.ik_min_y),
                                  float(bone.ik_min_z)],
                       "ik_max": [float(bone.ik_max_x), float(bone.ik_max_y),
                                  float(bone.ik_max_z)],
                       "ik_stiffness": [float(bone.ik_stiffness_x),
                                        float(bone.ik_stiffness_y),
                                        float(bone.ik_stiffness_z)],
                       "ik_stretch": float(bone.ik_stretch),
                       "constraints": [
                           _constraint_dump(constraint, constraints.get(constraint.name))
                           for constraint in bone.constraints
                       ]})
    return result


def _rigid_body_dump(obj):
    body = getattr(obj, "rigid_body", None)
    if body is None:
        return None
    return {"type": body.type, "mass": float(body.mass),
            "friction": float(body.friction), "restitution": float(body.restitution),
            "shape": body.collision_shape, "linear_damping": float(body.linear_damping),
            "angular_damping": float(body.angular_damping),
            "initial_velocity": plain(body.linear_velocity) if hasattr(body, "linear_velocity") else None}


def _force_field_dump(obj):
    field = getattr(obj, "field", None)
    if field is None or getattr(field, "type", "NONE") == "NONE":
        return None
    return {"type": field.type, "strength": float(field.strength),
            "falloff": float(field.falloff_power)}


def _override_dump(obj):
    override = getattr(obj, "override_library", None)
    reference = getattr(override, "reference", None) if override else None
    if reference is None:
        return None
    path_map = {
        "location": ("transform.translation", "location"),
        "rotation_euler": ("transform.rotation", "rotation_quaternion"),
        "rotation_quaternion": ("transform.rotation", "rotation_quaternion"),
        "scale": ("transform.scale", "scale"),
        "hide_viewport": ("visible", "hide_viewport"),
        "hide_render": ("render_visible", "hide_render"),
        "hide_select": ("selectable", "hide_select"),
    }
    properties = []
    try:
        for property_override in override.properties:
            path = property_override.rna_path
            mapped = path_map.get(path)
            if mapped is None:
                continue
            value = getattr(obj, mapped[1])
            for operation in property_override.operations:
                name = operation.operation.lower()
                if name == "replace":
                    properties.append({"path": mapped[0], "operation": "replace",
                                       "value": plain(value)})
    except Exception:
        pass
    return {"reference": reference.name_full, "properties": properties}


def object_dump(obj):
    hide_viewport = bool(obj.hide_viewport)
    try:
        hide_set = bool(obj.hide_get())
    except Exception:
        hide_set = False
    hide_set_by_view_layer = {}
    for scene in bpy.data.scenes:
        for view_layer in scene.view_layers:
            if obj.name_full not in view_layer.objects:
                continue
            try:
                hide_set_by_view_layer.setdefault(scene.name_full, {})[view_layer.name] = bool(
                    obj.hide_get(view_layer=view_layer))
            except Exception:
                continue
    data = obj.data
    volume_bounds = None
    if obj.type == "VOLUME":
        try:
            corners = [plain(corner)[:3] for corner in obj.bound_box]
            if len(corners) == 8 and all(len(corner) == 3 for corner in corners):
                volume_bounds = [
                    [min(corner[axis] for corner in corners) for axis in range(3)],
                    [max(corner[axis] for corner in corners) for axis in range(3)],
                ]
        except Exception:
            volume_bounds = None
    materials = []
    for slot in obj.material_slots:
        materials.append(slot.material.name_full if slot.material else None)
    modifiers = []
    try:
        modifier_ids = json.loads(obj.get("potter.modifier_ids_json", "{}"))
    except Exception:
        modifier_ids = {}
    for modifier in obj.modifiers:
        properties = modifier_params(modifier)
        if modifier.type == "NODES":
            properties["node_group"] = modifier.node_group.name_full if modifier.node_group else None
            properties["inputs"] = _modifier_node_inputs(modifier)
        if modifier.type == "MESH_SEQUENCE_CACHE":
            cache_file = getattr(modifier, "cache_file", None)
            properties["cache_filepath"] = (
                bpy.path.abspath(cache_file.filepath)
                if cache_file and cache_file.filepath else None)
            properties["cache_file_settings"] = (
                rna_props(cache_file, ("filepath",)) if cache_file else {})
        native_target_mesh = None
        if modifier.type == "SURFACE_DEFORM" and modifier.is_bound and modifier.target:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated_target = modifier.target.evaluated_get(depsgraph)
            evaluated_mesh = evaluated_target.to_mesh()
            try:
                native_target_mesh = {
                    "vertices": [plain(vertex.co) for vertex in evaluated_mesh.vertices],
                }
            finally:
                evaluated_target.to_mesh_clear()
        modifiers.append({"name": modifier.name, "type": modifier.type,
                          "potter_id": modifier_ids.get(modifier.name),
                          "enabled": bool(modifier.show_viewport and modifier.show_render),
                          "properties": properties,
                          "native_target_mesh": native_target_mesh,
                          "physics_settings": _physics_modifier_settings(obj, modifier),
                          "custom_properties": custom_props(modifier)})
    inverse_map = _object_solver_inverse_map(obj).get("objects", {})
    constraints = [
        _constraint_dump(constraint, inverse_map.get(constraint.name))
        for constraint in obj.constraints
    ]
    action = None
    action_slot = None
    drivers = []
    nla_tracks = []
    if obj.animation_data:
        if obj.animation_data.action:
            action = obj.animation_data.action.name_full
        slot = getattr(obj.animation_data, "action_slot", None)
        if slot is not None:
            action_slot = {"identifier": getattr(slot, "identifier", ""),
                           "name": getattr(slot, "name_display", getattr(slot, "name", "")),
                           "target_type": getattr(slot, "target_id_type", "")}
        for curve in obj.animation_data.drivers:
            drivers.append({"curve": dump_fcurve(curve),
                            "driver": {"type": curve.driver.type, "expression": curve.driver.expression,
                                       "variables": [{"name": variable.name, "type": variable.type,
                                                      "targets": [{"id_type": target.id_type,
                                                                   "id": plain(target.id),
                                                                   "data_path": target.data_path}
                                                                  for target in variable.targets]}
                                                     for variable in curve.driver.variables]}})
        for track in obj.animation_data.nla_tracks:
            nla_tracks.append({"name": track.name, "mute": bool(track.mute), "solo": bool(track.is_solo),
                               "strips": [{"name": strip.name,
                                           "action": strip.action.name_full if strip.action else None,
                                           "action_slot": {"identifier": getattr(strip.action_slot, "identifier", ""),
                                                           "name": getattr(strip.action_slot, "name_display",
                                                                           getattr(strip.action_slot, "name", "")),
                                                           "target_type": getattr(strip.action_slot, "target_id_type", "")}
                                           if getattr(strip, "action_slot", None) else None,
                                           "properties": rna_props(strip),
                                           "custom_properties": custom_props(strip)} for strip in track.strips]})
    return {"name": obj.name_full, "potter_id": obj.get("potter.id"), "type": obj.type,
            "library": library_reference(obj), "override_library": _override_dump(obj),
            "data_name": data.name_full if data else None, "parent": obj.parent.name_full if obj.parent else None,
            "parent_type": obj.parent_type, "parent_bone": obj.parent_bone,
            "instance_collection": obj.instance_collection.name_full
            if getattr(obj, "instance_collection", None) else None,
            "matrix_parent_inverse": matrix_rows(obj.matrix_parent_inverse),
            "location": plain(obj.location),
            "rotation_mode": obj.rotation_mode,
            "rotation_euler": [float(obj.rotation_euler.x), float(obj.rotation_euler.y),
                               float(obj.rotation_euler.z)],
            "rotation_quaternion": plain(obj.rotation_quaternion), "rotation_axis_angle": plain(obj.rotation_axis_angle),
            "scale": plain(obj.scale), "delta_location": plain(obj.delta_location),
            "delta_rotation_euler": plain(obj.delta_rotation_euler), "delta_rotation_quaternion": plain(obj.delta_rotation_quaternion),
            "delta_scale": plain(obj.delta_scale), "matrix_basis": matrix_rows(obj.matrix_basis),
            "matrix_local": matrix_rows(obj.matrix_local), "matrix_world": matrix_rows(obj.matrix_world),
            "hide_viewport": hide_viewport, "hide_set": hide_set,
            "hide_set_by_view_layer": hide_set_by_view_layer,
            "hide_render": bool(obj.hide_render),
            "hide_select": bool(obj.hide_select), "materials": materials, "modifiers": modifiers,
            "constraints": constraints, "drivers": drivers, "nla_tracks": nla_tracks,
            "action": action, "action_slot": action_slot, "pose": _pose_dump(obj),
            "vertex_groups": _vertex_groups_dump(obj), "rigid_body": _rigid_body_dump(obj),
            "force_field": _force_field_dump(obj), "custom_properties": custom_props(obj),
            "volume_bounds": volume_bounds,
            "rna_properties": rna_props(obj, ("data", "parent", "children", "users_collection", "modifiers", "constraints", "animation_data", "material_slots"))}


def action_dump(action):
    slot_ids = _id_map(action, "potter.action_slot_ids_json")
    slots = []
    curves = []
    channelbags = []
    try:
        for slot in action.slots:
            slot_curves = []
            for layer in action.layers:
                for strip in layer.strips:
                    if strip.type != "KEYFRAME":
                        continue
                    try:
                        channelbag = strip.channelbag(slot)
                    except Exception:
                        channelbag = None
                    if channelbag is None:
                        continue
                    bag_curves = [dump_fcurve(curve) for curve in channelbag.fcurves]
                    slot_curves.extend(bag_curves)
                    channelbags.append({"slot": getattr(slot, "identifier", ""),
                                        "layer": layer.name,
                                        "strip": getattr(strip, "name", strip.bl_rna.identifier),
                                        "fcurves": bag_curves})
            slots.append({"identifier": getattr(slot, "identifier", ""),
                          "name": getattr(slot, "name_display", getattr(slot, "name", "")),
                          "target_type": getattr(slot, "target_id_type", ""),
                          "potter_id": slot_ids.get(getattr(slot, "identifier", "")),
                          "fcurves": slot_curves})
            curves.extend(slot_curves)
    except Exception as error:
        if getattr(action, "slots", None):
            raise ValueError("cannot read Blender Action slots: %s" % error) from error
    if not curves and hasattr(action, "fcurves"):
        try:
            curves = [dump_fcurve(curve) for curve in action.fcurves]
        except Exception as error:
            raise ValueError("cannot read Blender Action fcurves: %s" % error) from error
    return {"name": action.name_full, "potter_id": action.get("potter.id"),
            "library": library_reference(action),
            "slots": slots, "channelbags": channelbags, "slot_count": len(action.slots),
            "fcurves": curves, "custom_properties": custom_props(action),
            "rna_properties": rna_props(action, ("fcurves", "layers", "slots"))}


def dump_fcurve(curve):
    curve.update()
    keys = []
    for point in curve.keyframe_points:
        keys.append({"frame": float(point.co.x), "value": float(point.co.y),
                     "interpolation": point.interpolation,
                     "handle_left": plain(point.handle_left), "handle_right": plain(point.handle_right),
                     "handle_left_type": point.handle_left_type, "handle_right_type": point.handle_right_type})
    return {"path": curve.data_path, "index": int(curve.array_index), "keyframes": keys,
            "extrapolation": curve.extrapolation, "mute": bool(curve.mute),
            "modifiers": [{"type": m.type, "properties": rna_props(m)} for m in curve.modifiers]}


def collection_dump(collection):
    return {"name": collection.get("potter.name", collection.name_full),
            "potter_id": collection.get("potter.id"), "library": library_reference(collection),
            "children": [child.name_full for child in collection.children],
            "objects": [obj.name_full for obj in collection.objects],
            "custom_properties": custom_props(collection), "hide_viewport": bool(collection.hide_viewport),
            "hide_render": bool(collection.hide_render), "fake_user": bool(collection.use_fake_user)}

def collection_dumps():
    collections = list(bpy.data.collections)
    known = {collection.as_pointer() for collection in collections}
    for scene in bpy.data.scenes:
        root = scene.collection
        if root.as_pointer() not in known:
            collections.append(root)
            known.add(root.as_pointer())
    return [collection_dump(collection) for collection in collections]

def _rigid_body_world_dump(scene):
    world = getattr(scene, "rigidbody_world", None)
    if world is None:
        return None
    cache = world.point_cache
    return {"enabled": bool(world.enabled), "gravity": plain(scene.gravity),
            "substeps": int(world.substeps_per_frame),
            "solver_iterations": int(world.solver_iterations),
            "frame_start": int(cache.frame_start), "frame_end": int(cache.frame_end),
            # Blender has no solver seed; retain the contract value as scene metadata.
            "seed": scene.get("potter.rigid_body_seed")}


def _view_transform_name(value):
    token = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"agx": "ag_x", "falsecolor": "false_color"}
    return aliases.get(token, token)


def _color_management_dump(scene):
    view = scene.view_settings
    curve = []
    if bool(getattr(view, "use_curve_mapping", False)):
        mapping = getattr(view, "curve_mapping", None)
        if mapping is None or len(mapping.curves) < 4:
            raise ValueError("Blender color-management curve mapping is unavailable")
        curve = [plain(point.location)[:2] for point in mapping.curves[3].points]
    return {"display_device": scene.display_settings.display_device,
            "view_transform": _view_transform_name(view.view_transform),
            "look": potter_color_look(view.look), "exposure": float(view.exposure),
            "gamma": float(view.gamma), "curve": curve}


def _render_passes(scene):
    passes = set()
    for layer in scene.view_layers:
        for prop in layer.bl_rna.properties:
            name = prop.identifier
            if not name.startswith("use_pass_"):
                continue
            try:
                if bool(getattr(layer, name)):
                    passes.add(name[len("use_pass_"):].lower())
            except Exception:
                continue
    return sorted(passes)


def _scene_compositor(scene):
    tree = getattr(scene, "compositing_node_group", None)
    if tree is None:
        tree = getattr(scene, "node_tree", None)
    if tree is None:
        return None
    return {"id_type": tree.bl_rna.identifier, "name": tree.name_full,
            "potter_id": tree.get("potter.id")}


def _node_groups_dump():
    groups = list(bpy.data.node_groups)
    pointers = {group.as_pointer() for group in groups}
    for scene in bpy.data.scenes:
        tree = getattr(scene, "compositing_node_group", None)
        if tree is None:
            tree = getattr(scene, "node_tree", None)
        if tree is not None and tree.as_pointer() not in pointers:
            groups.append(tree)
            pointers.add(tree.as_pointer())
    return [{"name": group.name_full, "potter_id": group.get("potter.id"),
             "library": library_reference(group),
             "type": group.bl_idname, "tree": tree_dump(group),
             "custom_properties": custom_props(group),
             "fake_user": bool(getattr(group, "use_fake_user", False))}
            for group in groups]


def _sequencer_dump(scene):
    editor = getattr(scene, "sequence_editor", None)
    if editor is None:
        return {"channels": 32, "strips": []}
    sequences = getattr(editor, "strips", None) or getattr(editor, "sequences", ())
    result = []
    for strip in sequences:
        kind = strip.type
        source = None
        if hasattr(strip, "filepath"):
            source = bpy.path.abspath(strip.filepath)
        elif getattr(strip, "sound", None) is not None:
            source = bpy.path.abspath(strip.sound.filepath)
        elif getattr(strip, "scene", None) is not None:
            source = strip.scene.name_full
        source_type = kind
        if kind == "MOVIE":
            source_type = "movie"
        elif kind == "IMAGE":
            source_type = "image_sequence" if len(getattr(strip, "elements", ())) > 1 else "image"
        elif kind == "SOUND":
            source_type = "sound"
        elif kind == "SCENE":
            source_type = "scene"
        elif kind == "COLOR":
            source_type = "color"
        elif kind == "TEXT":
            source_type = "text"
        elif kind == "META":
            source_type = "meta"
        else:
            source_type = "transition" if kind in ("CROSS", "GAMMA_CROSS", "WIPE") else "effect"
        transition = {"CROSS": "cross", "GAMMA_CROSS": "gamma_cross",
                      "WIPE": "wipe"}.get(kind)
        effect = {
            "ADD": "add", "SUBTRACT": "subtract", "MULTIPLY": "multiply",
            "ALPHA_OVER": "alpha_over", "TRANSFORM": "transform", "SPEED": "speed",
            "GLOW": "glow", "GAUSSIAN_BLUR": "gaussian_blur",
        }.get(kind)
        color = list(plain(getattr(strip, "color", (0.0, 0.0, 0.0, 1.0))))
        if len(color) != 4:
            color = [0.0, 0.0, 0.0, 1.0]
        retiming = []
        for key in getattr(strip, "retiming_keys", ()):
            retiming.append({"frame": float(getattr(key, "timeline_frame", 0.0)),
                             "source_frame": float(getattr(key, "source_frame", 0.0))})
        result.append({
            "name": strip.name,
            "type": source_type,
            "channel": int(strip.channel),
            "frame_start": float(strip.frame_final_start),
            "frame_offset_start": float(getattr(strip, "frame_offset_start", 0.0)),
            "frame_offset_end": float(getattr(strip, "frame_offset_end", 0.0)),
            "length": float(getattr(strip, "frame_final_duration", 1.0)),
            "blend_type": str(getattr(strip, "blend_type", "REPLACE")).lower(),
            "opacity": float(getattr(strip, "blend_alpha", 1.0)),
            "mute": bool(getattr(strip, "mute", False)),
            "retiming_keys": retiming,
            "modifiers": [],
            "sound_volume": float(getattr(strip, "volume", 1.0)),
            "sound_pan": float(getattr(strip, "pan", 0.0)),
            "sound_pitch": float(getattr(strip, "pitch", 1.0)),
            "source": source,
            "color": color,
            "text": getattr(strip, "text", None),
            "transition": transition,
            "inputs": [getattr(strip, name).name_full for name in ("input_1", "input_2")
                       if getattr(strip, name, None) is not None],
            "effect": effect,
        })
    channels = getattr(editor, "channels", 32)
    if not isinstance(channels, int):
        channels = len(channels)
    return {"channels": int(channels), "strips": result}


def potter_audio_codec(value):
    codec = str(value).upper()
    return "wav" if codec in ("PCM", "WAV") else codec.lower()


def blender_audio_codec(value):
    codec = str(value).upper()
    return "PCM" if codec == "WAV" else codec


def potter_color_look(value):
    normalized = str(value).strip().casefold()
    return {
        "none": "none",
        "very high contrast": "very_high_contrast",
        "high contrast": "high_contrast",
        "medium high contrast": "medium_high_contrast",
        "medium contrast": "medium_contrast",
        "medium low contrast": "medium_low_contrast",
        "low contrast": "low_contrast",
        "very low contrast": "very_low_contrast",
    }.get(normalized, str(value))


def blender_color_look(value):
    return {
        "none": "None",
        "very_high_contrast": "Very High Contrast",
        "high_contrast": "High Contrast",
        "medium_high_contrast": "Medium High Contrast",
        "medium_contrast": "Medium Contrast",
        "medium_low_contrast": "Medium Low Contrast",
        "low_contrast": "Low Contrast",
        "very_low_contrast": "Very Low Contrast",
    }.get(str(value).strip().casefold(), value)


def dump_scene(scene):
    layers = []
    view_layer_ids = _id_map(scene, "potter.view_layer_ids_json")
    for view_layer in scene.view_layers:
        excluded = []
        def walk(layer_collection):
            if layer_collection.exclude:
                excluded.append(layer_collection.collection.name_full)
            for child in layer_collection.children:
                walk(child)
        walk(view_layer.layer_collection)
        layers.append({"name": view_layer.name, "potter_id": view_layer_ids.get(view_layer.name),
                       "excluded_collections": excluded})
    world = scene.world
    world_info = None
    if world:
        color = list(plain(world.color))[:3]
        background_color = None
        strength = 1.0
        if world.use_nodes and world.node_tree:
            for node in world.node_tree.nodes:
                if node.type == "BACKGROUND":
                    try:
                        strength = float(node.inputs.get("Strength").default_value)
                        background_color = list(plain(node.inputs.get("Color").default_value))[:3]
                    except Exception:
                        pass
                    break
        world_info = {"name": world.name_full, "potter_id": world.get("potter.id"),
                      "library": library_reference(world), "color": color,
                      "background_color": background_color, "strength": strength,
                      "nodes": tree_dump(world.node_tree),
                      "custom_properties": custom_props(world), "fake_user": bool(world.use_fake_user),
                      "users": int(world.users)}
    camera = scene.camera.name_full if scene.camera else None
    unit = scene.unit_settings
    render_engine = scene.render.engine
    samples = 64
    seed = 0
    if render_engine == "CYCLES":
        samples = int(scene.cycles.samples)
        seed = int(scene.cycles.seed)
    elif render_engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT") and hasattr(scene, "eevee"):
        samples = int(scene.eevee.taa_render_samples)
    return {"name": scene.name_full, "potter_id": scene.get("potter.id"),
            "frame_current": float(scene.frame_current) + float(getattr(scene, "frame_subframe", 0.0)),
            "frame_start": int(scene.frame_start), "frame_end": int(scene.frame_end), "fps": int(scene.render.fps),
            "fps_base": float(scene.render.fps_base), "camera": camera,
            "active_clip": (scene.active_clip.name_full
                            if getattr(scene, "active_clip", None) else None),
            "render": {
                "resolution_x": int(scene.render.resolution_x),
                "resolution_y": int(scene.render.resolution_y),
                "resolution_percentage": int(scene.render.resolution_percentage),
                "samples": samples,
                "seed": seed,
                "max_bounces": int(getattr(getattr(scene, "cycles", None), "max_bounces", 4)),
                "use_sequencer": bool(getattr(scene.render, "use_sequencer", False)),
                "audio_codec": potter_audio_codec(getattr(getattr(scene.render, "ffmpeg", None),
                                                          "audio_codec", "PCM")),
                "film_transparent": bool(scene.render.film_transparent),
                "engine": "path" if render_engine == "CYCLES" else "realtime",
                "engine_native": render_engine,
                "passes": _render_passes(scene),
            },
            "use_compositing": bool(getattr(scene.render, "use_compositing", False)),
            "compositor": _scene_compositor(scene),
            "color_management": _color_management_dump(scene),
            "view_layers": layers,
            "markers": [{"name": marker.name, "frame": int(marker.frame)} for marker in scene.timeline_markers],
            "rigid_body_world": _rigid_body_world_dump(scene),
            "sequencer": _sequencer_dump(scene),
            "root_collection": scene.get("potter.root_collection", scene.collection.get("potter.name", scene.collection.name_full)),
            "world": world_info, "custom_properties": custom_props(scene),
            "rna_properties": rna_props(scene, ("objects", "collection", "view_layers", "world", "camera", "render", "tool_settings", "unit_settings"))}


def _startup_metadata_datablock(collection_name, block):
    if collection_name in ("all_ids", "screens", "window_managers", "workspaces"):
        return True
    name = block.name_full
    return ((collection_name == "images" and name in ("Render Result", "Viewer Node"))
            or (collection_name == "linestyles" and name == "LineStyle")
            or (collection_name == "palettes" and name == "Palette"))


def other_datablocks_dump():
    known = {"actions", "armatures", "cache_files", "cameras", "collections", "curves", "fonts",
             "grease_pencils_v3", "images", "libraries", "lights", "materials", "meshes",
             "movieclips", "node_groups", "objects", "scenes", "texts", "volumes", "worlds"}
    result = []
    for collection_name in dir(bpy.data):
        if collection_name.startswith("_") or collection_name in known:
            continue
        try:
            collection = getattr(bpy.data, collection_name)
            if "bpy_prop_collection" not in str(type(collection)):
                continue
            for block in collection:
                if _startup_metadata_datablock(collection_name, block):
                    continue
                result.append({"type": collection_name, "name": block.name_full,
                               "potter_id": block.get("potter.id"),
                               "library": library_reference(block), "users": int(block.users),
                               "fake_user": bool(getattr(block, "use_fake_user", False)),
                               "custom_properties": custom_props(block), "rna_properties": rna_props(block)})
        except Exception:
            continue
    return result


def material_dump(material):
    base_color = plain(material.diffuse_color)
    metallic = float(getattr(material, "metallic", 0.0))
    roughness = float(getattr(material, "roughness", 0.8))
    emission_color = [0.0, 0.0, 0.0]
    emission_strength = 0.0
    transmission = 0.0
    ior = 1.45
    if material.use_nodes and material.node_tree:
        shader = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
        if shader:
            try:
                base_color = plain(shader.inputs["Base Color"].default_value)
                metallic = float(shader.inputs["Metallic"].default_value)
                roughness = float(shader.inputs["Roughness"].default_value)
                emission = shader.inputs.get("Emission Color")
                if emission is None:
                    emission = shader.inputs.get("Emission")
                if emission is not None:
                    emission_color = list(plain(emission.default_value))[:3]
                emission_power = shader.inputs.get("Emission Strength")
                if emission_power is not None:
                    emission_strength = float(emission_power.default_value)
                transmission_socket = shader.inputs.get("Transmission Weight")
                if transmission_socket is None:
                    transmission_socket = shader.inputs.get("Transmission")
                if transmission_socket is not None:
                    transmission = float(transmission_socket.default_value)
                ior_socket = shader.inputs.get("IOR")
                if ior_socket is not None:
                    ior = float(ior_socket.default_value)
            except Exception:
                pass
    return {"name": material.name_full, "potter_id": material.get("potter.id"),
            "library": library_reference(material),
            "base_color": base_color, "metallic": metallic, "roughness": roughness,
            "emission_color": emission_color, "emission_strength": emission_strength,
            "transmission": transmission, "ior": ior,
            "double_sided": not bool(getattr(material, "use_backface_culling", False)),
            "nodes": tree_dump(material.node_tree), "custom_properties": custom_props(material),
            "fake_user": bool(material.use_fake_user), "users": int(material.users)}


def dump_all():
    for volume in getattr(bpy.data, "volumes", ()):
        try:
            volume.grids.load()
        except Exception:
            pass
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    data = {"bridge_version": 1, "blender_version": "%d.%d.%d" % bpy.app.version,
            "file_version": list(getattr(bpy.app, "version_file", bpy.app.version)),
            "active_scene": bpy.context.scene.name_full,
            "scenes": [dump_scene(item) for item in bpy.data.scenes],
            "collections": collection_dumps(),
            "objects": [object_dump(item) for item in bpy.data.objects],
            "armatures": [_armature_dump(item) for item in bpy.data.armatures],
            "grease_pencil": [_grease_pencil_dump(item) for item in
                              getattr(bpy.data, "grease_pencils_v3", ())],
            "curves": [curve_dump(item) for item in bpy.data.curves],
            "volumes": [volume_dump(item) for item in getattr(bpy.data, "volumes", ())],
            "meshes": [mesh_dump(item) for item in bpy.data.meshes],
            "images": images_dump(),
            "movie_clips": [movie_clip_dump(item)
                            for item in getattr(bpy.data, "movieclips", ())],
            "materials": [material_dump(item) for item in bpy.data.materials],
            "cameras": [{"name": x.name_full, "potter_id": x.get("potter.id"),
                         "library": library_reference(x), "type": x.type,
                         "lens": float(x.lens), "sensor_width": float(x.sensor_width),
                         "sensor_height": float(x.sensor_height), "sensor_fit": x.sensor_fit,
                         "ortho_scale": float(x.ortho_scale),
                         "clip_start": float(x.clip_start), "clip_end": float(x.clip_end),
                         "shift_x": float(x.shift_x), "shift_y": float(x.shift_y), "custom_properties": custom_props(x),
                         "fake_user": bool(x.use_fake_user), "users": int(x.users)} for x in bpy.data.cameras],
            "lights": [{"name": x.name_full, "potter_id": x.get("potter.id"),
                        "library": library_reference(x), "type": x.type,
                        "color": plain(x.color), "energy": float(x.energy), "shadow_soft_size": float(x.shadow_soft_size),
                        "spot_size": float(getattr(x, "spot_size", math.pi / 4)),
                        "spot_blend": float(getattr(x, "spot_blend", 0.15)),
                        "shape": getattr(x, "shape", None), "custom_properties": custom_props(x),
                        "size": float(getattr(x, "size", 0.25)),
                        "size_y": float(getattr(x, "size_y", 0.25)),
                        "fake_user": bool(x.use_fake_user), "users": int(x.users)} for x in bpy.data.lights],
            "actions": [action_dump(x) for x in bpy.data.actions],
            "texts": [{"name": x.name_full, "potter_id": x.get("potter.id"),
                       "library": library_reference(x), "body": x.as_string(),
                       "custom_properties": custom_props(x), "fake_user": bool(x.use_fake_user)} for x in bpy.data.texts],
            "resources": resources_dump(),
            "libraries": libraries_dump(),
            "node_groups": _node_groups_dump(),
            "other_datablocks": other_datablocks_dump(), "unused_datablocks": []}
    for collection_name in ("meshes", "materials", "cameras", "lights", "actions",
                            "node_groups", "volumes", "worlds"):
        for block in getattr(bpy.data, collection_name):
            if getattr(block, "users", 0) == 0 or getattr(block, "use_fake_user", False):
                data["unused_datablocks"].append({"type": collection_name, "name": block.name_full,
                                                  "potter_id": block.get("potter.id"), "users": int(block.users),
                                                  "fake_user": bool(getattr(block, "use_fake_user", False))})
    for block in data["other_datablocks"]:
        if block["users"] == 0 or block["fake_user"]:
            data["unused_datablocks"].append(block)
    return data


def set_id(block, value):
    if block is not None and value:
        block["potter.id"] = value

def set_scene_frame(scene, frame):
    value = float(frame)
    whole_frame = math.floor(value)
    scene.frame_set(whole_frame, subframe=value - whole_frame)


def set_props(block, values):
    for key, value in values.items():
        if key == "potter.id":
            continue
        try:
            block[key] = value
        except Exception:
            pass


def _blender_id_pointer_type(block, key, current):
    if isinstance(current, bpy.types.ID):
        return current.bl_rna.identifier
    try:
        prop = block.bl_rna.properties[key]
        fixed_type = prop.fixed_type
        if prop.type == "POINTER":
            identifier = getattr(fixed_type, "identifier", None)
            if identifier is None:
                identifier = getattr(getattr(fixed_type, "bl_rna", None), "identifier", None)
            fixed_class = getattr(bpy.types, identifier, None)
            if (isinstance(fixed_class, type)
                    and issubclass(fixed_class, bpy.types.ID)):
                return identifier
    except Exception:
        return None
    return None


def set_rna(block, values, id_lookup=None, *, strict=False, owner=None):
    if block is None and not strict:
        return
    for key, value in values.items():
        if (strict and isinstance(value, dict) and "rna_type" in value
                and "id_type" not in value):
            continue
        if key in ("name", "type", "rna_type") or (strict and key == "action_slot"):
            continue
        if not hasattr(block, key):
            if strict:
                raise ValueError("%s cannot represent RNA property %s" % (owner, key))
            continue
        try:
            current = getattr(block, key)
            if strict and isinstance(value, dict) and "id_type" in value and "name" in value:
                candidate = _resolve_compat_value(value, id_lookup)
                _set_required(block, key, candidate, owner)
                continue
            pointer_type = _blender_id_pointer_type(block, key, current)
            if pointer_type is not None:
                candidate = None
                if isinstance(value, dict) and id_lookup is not None:
                    candidate = id_lookup.get(value.get("name"))
                    valid = (candidate is not None
                             and candidate.bl_rna.identifier == value.get("id_type")
                             and candidate.bl_rna.identifier == pointer_type)
                    if not valid:
                        if strict:
                            raise ValueError("%s references unavailable ID %s"
                                             % (owner, value.get("name")))
                        candidate = None
                elif isinstance(value, str) and id_lookup is not None:
                    candidate = id_lookup.get(value)
                    if candidate is not None and candidate.bl_rna.identifier != pointer_type:
                        if strict:
                            raise ValueError("%s references wrong ID type %s" % (owner, value))
                        candidate = None
                    elif candidate is None and strict:
                        DEFERRED_ID_ASSIGNMENTS.append((block, key, value, owner))
                elif value is None:
                    if strict:
                        _set_required(block, key, None, owner)
                    else:
                        setattr(block, key, None)
                elif strict:
                    raise ValueError("%s cannot restore ID property %s" % (owner, key))
                if candidate is not None:
                    if strict:
                        _set_required(block, key, candidate, owner)
                    else:
                        setattr(block, key, candidate)
                continue
            if strict:
                if isinstance(current, Matrix) and isinstance(value, (list, tuple)):
                    _set_required(block, key, Matrix(value), owner)
                    continue
                if hasattr(current, "bl_rna"):
                    if value not in (None, {"rna_type": current.bl_rna.identifier}):
                        raise ValueError("%s cannot restore nested RNA property %s"
                                         % (owner, key))
                    continue
                _set_required(block, key, value, owner)
                continue
            if isinstance(value, dict) and hasattr(current, "bl_rna"):
                set_rna(current, value, id_lookup)
                continue
            if hasattr(current, "bl_rna"):
                continue
            property_definition = block.bl_rna.properties.get(key)
            if (getattr(property_definition, "is_enum_flag", False)
                    and isinstance(value, (list, tuple))):
                value = set(value)
            if isinstance(current, (int, float, bool, str)) or hasattr(current, "__len__"):
                setattr(block, key, value)
        except Exception:
            if strict:
                raise
            continue

def _restore_modifier_id_references(modifier, params, id_lookup):
    for key, value in params.items():
        if not hasattr(modifier, key):
            continue
        pointer_type = _blender_id_pointer_type(modifier, key, getattr(modifier, key))
        if pointer_type not in ("Object", "Collection") or value is None:
            continue
        name = value.get("name") if isinstance(value, dict) else value
        candidate = id_lookup.get(name) if isinstance(name, str) else None
        if (candidate is None
                or candidate.bl_rna.identifier != pointer_type
                or (isinstance(value, dict)
                    and value.get("id_type") != pointer_type)):
            raise ValueError(
                "modifier %s references unavailable %s ID %s"
                % (modifier.name, pointer_type, name))
        setattr(modifier, key, candidate)


def _create_modifier(obj, name, modifier_type):
    if modifier_type != "PARTICLE_SYSTEM":
        return obj.modifiers.new(name, modifier_type)
    scene = bpy.context.scene
    temporarily_linked = obj.name not in scene.objects
    if temporarily_linked:
        scene.collection.objects.link(obj)
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    was_selected = obj.select_get()
    try:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(True)
        view_layer.objects.active = obj
        result = bpy.ops.object.particle_system_add()
        if "FINISHED" not in result or not obj.particle_systems:
            raise ValueError("Blender did not create a particle system")
        particle_system = obj.particle_systems[-1]
        modifier = next(
            (candidate for candidate in obj.modifiers
             if candidate.type == "PARTICLE_SYSTEM"
             and candidate.particle_system == particle_system),
            None)
        if modifier is None:
            raise ValueError("Blender did not create a Particle System modifier")
        modifier.name = name
        return modifier
    finally:
        obj.select_set(was_selected)
        view_layer.objects.active = previous_active
        if temporarily_linked:
            scene.collection.objects.unlink(obj)




def _ensure_dynamic_paint_role(modifier, obj, role):
    settings_name = "brush_settings" if role == "brush" else "canvas_settings"
    settings = getattr(modifier, settings_name, None)
    if settings is not None:
        return False
    scene = bpy.context.scene
    temporarily_linked = obj.name not in scene.objects
    if temporarily_linked:
        scene.collection.objects.link(obj)
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    was_selected = obj.select_get()
    try:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(True)
        view_layer.objects.active = obj
        result = bpy.ops.dpaint.type_toggle(type=role.upper())
        if "FINISHED" not in result:
            raise ValueError("Blender did not create Dynamic Paint %s settings" % role)
    finally:
        obj.select_set(was_selected)
        view_layer.objects.active = previous_active
    settings = getattr(modifier, settings_name, None)
    if settings is None:
        raise ValueError("Blender did not initialize Dynamic Paint %s settings" % role)
    return temporarily_linked


def _bake_point_cache(obj, cache):
    if cache is None:
        return
    scene = bpy.context.scene
    temporarily_linked = obj.name not in scene.objects
    if temporarily_linked:
        scene.collection.objects.link(obj)
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    was_selected = obj.select_get()
    previous_frame = scene.frame_current
    try:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        obj.select_set(True)
        view_layer.objects.active = obj
        scene.frame_set(int(cache.frame_start))
        with bpy.context.temp_override(object=obj, active_object=obj, point_cache=cache):
            result = bpy.ops.ptcache.bake(bake=True)
        if "FINISHED" not in result or not cache.is_baked:
            raise ValueError("Blender did not bake point cache for %s" % obj.name)
    finally:
        scene.frame_set(previous_frame)
        obj.select_set(was_selected)
        view_layer.objects.active = previous_active
        if temporarily_linked:
            scene.collection.objects.unlink(obj)


def set_modifier_params(modifier, item, id_lookup, doc, target, node_id, obj):
    params = item.get("params", {})
    kind = item.get("type")
    physics_property = {
        "cloth": "physics_cloth",
        "soft_body": "physics_soft_body",
        "collision": "physics_collision",
        "dynamic_paint": "physics_dynamic_paint",
        "fluid": "physics_fluid",
        "particle_system": "physics_particle_emitter",
    }.get(kind)
    node = doc.get("nodes", {}).get(node_id, {})
    node_properties = node.get("properties", {})
    settings_id = params.get("settings_id")
    if kind == "particle_system":
        settings = node_properties.get("physics_particle_systems", {}).get(settings_id, {})
    elif physics_property:
        settings = node_properties.get(physics_property, {})
    else:
        settings = {}
    dynamic_paint_temporary_link = False
    if kind == "dynamic_paint" and settings.get("role") in ("brush", "canvas"):
        dynamic_paint_temporary_link = _ensure_dynamic_paint_role(
            modifier, obj, settings["role"])
    rna_params = dict(params)
    for parameter in ("projectors", "resource", "is_bound", "cache_filepath", "map_curve"):
        rna_params.pop(parameter, None)
    set_rna(modifier, rna_params, id_lookup)
    _restore_modifier_id_references(modifier, rna_params, id_lookup)
    if kind in ("vertex_weight_edit", "vertex_weight_proximity") and "map_curve" in params:
        _set_curve_mapping(modifier, params["map_curve"])
    if kind == "mesh_cache" and params.get("resource"):
        filepath = _resource_path(doc, params["resource"])
        if not filepath or not os.path.isfile(filepath):
            raise ValueError("Mesh Cache resource is missing: %s" % params["resource"])
        modifier.filepath = _relative_blend_path(filepath, target)
    elif kind == "uv_project":
        projectors = params.get("projectors", [])
        if projectors:
            modifier.projector_count = len(projectors)
        for index, reference in enumerate(projectors):
            candidate = id_lookup.get(reference) if isinstance(reference, str) else None
            if candidate is None:
                raise ValueError("UV Project projector references missing object %s" % reference)
            modifier.projectors[index].object = candidate
    if physics_property:
        settings = dict(settings)
        settings.pop("role", None)
        canvas_surfaces = settings.pop("canvas_surfaces", [])
        point_cache = settings.pop("point_cache", None)
        collision_settings = settings.pop("collision_settings", None)
        particle_seed = settings.pop("seed", None) if kind == "particle_system" else None
        particle_system = None
        if kind in ("cloth", "soft_body", "collision"):
            settings_block = getattr(modifier, "settings", None)
        elif kind == "dynamic_paint":
            settings_block = (getattr(modifier, "brush_settings", None)
                              or getattr(modifier, "canvas_settings", None))
        elif kind == "fluid":
            settings_name = {
                "DOMAIN": "domain_settings",
                "FLOW": "flow_settings",
                "EFFECTOR": "effector_settings",
            }.get(getattr(modifier, "fluid_type", ""))
            settings_block = getattr(modifier, settings_name, None) if settings_name else None
        else:
            particle_system = getattr(modifier, "particle_system", None)
            settings_block = getattr(particle_system, "settings", None) if particle_system else None
        if settings_block is not None:
            set_rna(settings_block, settings, id_lookup)
        if particle_system is not None and particle_seed is not None:
            particle_system.seed = int(particle_seed)
        if collision_settings is not None:
            set_rna(getattr(modifier, "collision_settings", None),
                    collision_settings, id_lookup)
        if canvas_surfaces:
            canvas_settings = getattr(modifier, "canvas_settings", None)
            if canvas_settings is None:
                raise ValueError("Dynamic Paint canvas settings are missing")
            view_layer = bpy.context.view_layer
            previous_active = view_layer.objects.active
            was_selected = obj.select_get()
            try:
                obj.select_set(True)
                view_layer.objects.active = obj
                with bpy.context.temp_override(object=obj, active_object=obj):
                    while len(canvas_settings.canvas_surfaces) < len(canvas_surfaces):
                        result = bpy.ops.dpaint.surface_slot_add()
                        if "FINISHED" not in result:
                            raise ValueError("Blender did not add Dynamic Paint surface")
                for index, surface_values in enumerate(canvas_surfaces):
                    surface_values = dict(surface_values)
                    surface_cache = surface_values.pop("point_cache", None)
                    surface = canvas_settings.canvas_surfaces[index]
                    set_rna(surface, surface_values, id_lookup)
                    if surface_cache:
                        cache_block = getattr(surface, "point_cache", None)
                        set_rna(cache_block, surface_cache, id_lookup)
                        if surface_cache.get("is_baked"):
                            _bake_point_cache(obj, cache_block)
            finally:
                obj.select_set(was_selected)
                view_layer.objects.active = previous_active
        cache_block = None
        if kind == "particle_system":
            cache_block = getattr(particle_system, "point_cache", None)
        else:
            cache_block = getattr(modifier, "point_cache", None)
        if point_cache is not None:
            set_rna(cache_block, point_cache, id_lookup)
            if point_cache.get("is_baked"):
                _bake_point_cache(obj, cache_block)
    if dynamic_paint_temporary_link:
        obj.select_set(False)
        bpy.context.scene.collection.objects.unlink(obj)

def _bind_time_mesh_coordinates(doc, node_id):
    node = doc.get("nodes", {}).get(node_id, {})
    data_id = node.get("data")
    if data_id is None:
        return None
    mesh = compatibility_block(doc, "meshes", "Mesh", data_id)
    vertices = mesh.get("vertices")
    if not isinstance(vertices, list):
        return None
    coordinates = [vertex.get("co") for vertex in vertices]
    if any(not isinstance(coordinate, (list, tuple)) or len(coordinate) != 3
           for coordinate in coordinates):
        return None
    return coordinates


def _set_mesh_coordinates(obj, coordinates):
    for vertex, coordinate in zip(obj.data.vertices, coordinates):
        vertex.co = coordinate
    obj.data.update()

def _modifier_is_bound(modifier):
    return bool(getattr(modifier, "is_bound", getattr(modifier, "is_bind", False)))


def _same_mesh_topology(left, right):
    return (
        left is not None
        and right is not None
        and len(left.vertices) == len(right.vertices)
        and [tuple(edge.vertices) for edge in left.edges]
        == [tuple(edge.vertices) for edge in right.edges]
        and [tuple(polygon.vertices) for polygon in left.polygons]
        == [tuple(polygon.vertices) for polygon in right.polygons]
    )


def _source_modifier_object_copies(doc, mesh_map):
    source_path = doc.get("blender_original_blend_path")
    if not source_path or not os.path.isfile(source_path):
        return {}, set()
    source_names = set()
    for node_id, node in doc.get("nodes", {}).items():
        if not any(
            ((modifier.get("binding_data") or {}).get("format")
             == "blender_native_bind_v1"
             and modifier.get("type") in (
                 "surface_deform", "mesh_deform", "laplacian_deform"))
            or modifier.get("type") in ("boolean", "decimate", "solidify")
            for modifier in node.get("modifiers", [])
        ):
            continue
        source = compatibility_block(doc, "objects", "Object", node_id)
        source_names.add(source.get("name") or node.get("name"))
    if not source_names:
        return {}, set()

    source_collections = (
        "actions", "armatures", "cameras", "collections", "curves",
        "grease_pencils_v3", "images", "lights", "materials", "meshes",
        "node_groups", "objects", "shape_keys", "volumes", "worlds",
    )
    before = {
        name: {block.as_pointer() for block in getattr(bpy.data, name, ())}
        for name in source_collections
    }
    with bpy.data.libraries.load(source_path, link=False) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects if name in source_names]
    loaded_objects = [
        obj for obj in bpy.data.objects
        if obj.as_pointer() not in before["objects"]
    ]
    source_objects = {
        obj.name_full: obj for obj in data_to.objects if obj is not None
    }
    copies = {}
    preserved = set()
    for node_id, node in doc.get("nodes", {}).items():
        source = compatibility_block(doc, "objects", "Object", node_id)
        source_obj = source_objects.get(source.get("name") or node.get("name"))
        graph_mesh = mesh_map.get(node.get("data"))
        if source_obj is None or source_obj.type != "MESH":
            continue
        source_topology_matches = _same_mesh_topology(source_obj.data, graph_mesh)
        preserve_source_stack = any(
            item.get("type") in ("boolean", "decimate", "solidify")
            and (native := source_obj.modifiers.get(item.get("name"))) is not None
            and native.type == blender_modifier_type(item.get("type"))
            for item in node.get("modifiers", [])
        )
        native_names = set()
        for item in node.get("modifiers", []):
            binding_data = item.get("binding_data") or {}
            if not source_topology_matches:
                continue
            kind = item.get("type")
            if (binding_data.get("format") != "blender_native_bind_v1"
                    or kind not in ("surface_deform", "mesh_deform", "laplacian_deform")):
                continue
            native = source_obj.modifiers.get(item.get("name"))
            if (native is None or native.type != blender_modifier_type(kind)
                    or not _modifier_is_bound(native)):
                continue
            params = item.get("params", {})
            if kind in ("surface_deform", "mesh_deform"):
                target_id = params.get("target" if kind == "surface_deform" else "object")
                target_node = doc.get("nodes", {}).get(target_id, {})
                target_mesh = mesh_map.get(target_node.get("data"))
                source_target = getattr(
                    native, "target" if kind == "surface_deform" else "object", None)
                if (source_target is None or source_target.type != "MESH"
                        or not _same_mesh_topology(source_target.data, target_mesh)):
                    continue
            native_names.add(native.name)
        if not native_names and not preserve_source_stack:
            continue
        copied = source_obj.copy()
        copied.data = graph_mesh
        for constraint in list(copied.constraints):
            copied.constraints.remove(constraint)
        if copied.animation_data is not None:
            copied.animation_data_clear()
        copied.parent = None
        for key in list(copied.keys()):
            del copied[key]
        for group in list(copied.vertex_groups):
            copied.vertex_groups.remove(group)
        for modifier in list(copied.modifiers):
            if (modifier.type in (
                    "SURFACE_DEFORM", "MESH_DEFORM", "LAPLACIANDEFORM")
                    and modifier.name not in native_names):
                copied.modifiers.remove(modifier)
        copies[node_id] = copied
        preserved.update((node_id, name) for name in native_names)
    for source_obj in loaded_objects:
        bpy.data.objects.remove(source_obj, do_unlink=True)
    for collection_name in (
        "collections", "meshes", "curves", "armatures", "cameras", "lights",
        "grease_pencils_v3", "volumes", "materials", "images", "node_groups",
        "shape_keys", "actions", "worlds",
    ):
        collection = getattr(bpy.data, collection_name, None)
        if collection is None:
            continue
        for block in list(collection):
            if block.as_pointer() in before[collection_name]:
                continue
            if bool(getattr(block, "use_fake_user", False)):
                block.use_fake_user = False
            if block.users == 0:
                collection.remove(block)
    for node_id, copied in copies.items():
        copied.name = doc["nodes"][node_id]["name"]
    return copies, preserved


def _bind_blender_modifiers(doc, object_map, preserved_native_binds):
    operators = {
        "surface_deform": bpy.ops.object.surfacedeform_bind,
        "mesh_deform": bpy.ops.object.meshdeform_bind,
        "laplacian_deform": bpy.ops.object.laplaciandeform_bind,
    }
    view_layer = bpy.context.view_layer
    for node_id, node in doc.get("nodes", {}).items():
        obj = object_map.get(node_id)
        if obj is None:
            continue
        for item in node.get("modifiers", []):
            if (node_id, item.get("name")) in preserved_native_binds:
                continue
            operator = operators.get(item.get("type"))
            params = item.get("params", {})
            binding_data = item.get("binding_data")
            if operator is None or binding_data is None:
                continue
            modifier = obj.modifiers.get(item.get("name"))
            if modifier is None:
                raise ValueError("cannot bind missing modifier %s on %s" %
                                 (item.get("name"), obj.name))
            previous_active = view_layer.objects.active
            was_selected = obj.select_get()
            bind_objects = [(node_id, obj)]
            if item.get("type") in ("surface_deform", "mesh_deform"):
                target_key = "target" if item.get("type") == "surface_deform" else "object"
                target_node_id = params.get(target_key)
                target_obj = object_map.get(target_node_id)
                if target_obj is not None:
                    bind_objects.append((target_node_id, target_obj))
            saved_geometry = []
            try:
                if obj.mode != "OBJECT":
                    bpy.ops.object.mode_set(mode="OBJECT")
                obj.select_set(True)
                view_layer.objects.active = obj
                seen_meshes = set()
                for bind_node_id, bind_obj in bind_objects:
                    if bind_obj.type != "MESH":
                        continue
                    mesh_pointer = bind_obj.data.as_pointer()
                    if mesh_pointer in seen_meshes:
                        continue
                    seen_meshes.add(mesh_pointer)
                    coordinates = _bind_time_mesh_coordinates(doc, bind_node_id)
                    if coordinates is None:
                        continue
                    if len(coordinates) != len(bind_obj.data.vertices):
                        raise ValueError("bind-time mesh vertex count changed for %s" % bind_obj.name)
                    saved_geometry.append((bind_obj, [list(vertex.co) for vertex in bind_obj.data.vertices]))
                    _set_mesh_coordinates(bind_obj, coordinates)
                view_layer.update()
                result = operator(modifier=modifier.name)
                if "FINISHED" not in result:
                    raise ValueError("Blender did not bind modifier %s" % modifier.name)
            finally:
                for bind_obj, coordinates in reversed(saved_geometry):
                    _set_mesh_coordinates(bind_obj, coordinates)
                view_layer.update()
                obj.select_set(was_selected)
                view_layer.objects.active = previous_active
def mesh_create(name, raw, block_id):
    mesh = bpy.data.meshes.new(name)
    attributes = raw.get("attributes", {})
    vertex_items = raw.get("vertices", [])
    vertices = [item["co"] for item in vertex_items]
    vertex_indexes = {item["id"]: index for index, item in enumerate(vertex_items)}
    edges = [[vertex_indexes[value] for value in item["v"]] for item in raw.get("edges", [])]
    faces = [[vertex_indexes[value] for value in item["v"]] for item in raw.get("faces", [])]
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()
    edge_flags = attributes.get("blender_edge_flags", [])
    for index, edge in enumerate(mesh.edges):
        if index < len(edge_flags):
            source = edge_flags[index]
            try:
                if source.get("sharp"):
                    edge.use_edge_sharp = True
                if source.get("seam"):
                    edge.use_seam = True
            except Exception:
                pass
    polygon_smooth = attributes.get("blender_polygon_smooth", [])
    for index, polygon in enumerate(mesh.polygons):
        if index < len(raw.get("faces", [])):
            material_index = int(raw["faces"][index].get("material_index", 0))
            if material_index:
                polygon.material_index = material_index
        if index < len(polygon_smooth) and polygon_smooth[index]:
            polygon.use_smooth = True
    uv_layers = attributes.get("blender_uv_layers", raw.get("uv_layers", []))
    for layer in uv_layers:
        uv = mesh.uv_layers.new(name=layer["name"])
        for index, item in enumerate(layer.get("values", [])):
            if index < len(uv.data):
                uv.data[index].uv = item
    for source in attributes.get("blender_attributes", []):
        name = source.get("name")
        if not name or name == "position":
            continue
        try:
            attribute = mesh.attributes.get(name)
            if attribute is None:
                attribute = mesh.attributes.new(
                    name=name,
                    type=source["data_type"],
                    domain=source["domain"],
                )
            for index, value in enumerate(source.get("values", [])):
                if index >= len(attribute.data) or value is None:
                    continue
                data = attribute.data[index]
                if hasattr(data, "vector"):
                    data.vector = value
                elif hasattr(data, "color"):
                    data.color = value
                elif hasattr(data, "value"):
                    data.value = value
        except Exception:
            continue
    metadata = attributes.get("blender_metadata", {})
    set_props(mesh, metadata.get("custom_properties", {}))
    if metadata.get("fake_user"):
        mesh.use_fake_user = True
    set_id(mesh, block_id)
    return mesh


def _restore_skin_vertices(obj, vertices):
    if not vertices:
        return
    if not len(obj.data.skin_vertices):
        view_layer = bpy.context.view_layer
        previous_active = view_layer.objects.active
        previous_selected = list(bpy.context.selected_objects)
        try:
            for selected in previous_selected:
                selected.select_set(False)
            obj.select_set(True)
            view_layer.objects.active = obj
            bpy.ops.mesh.customdata_skin_add()
        finally:
            obj.select_set(False)
            for selected in previous_selected:
                selected.select_set(True)
            view_layer.objects.active = previous_active
    layer = obj.data.skin_vertices[0]
    for index, source in enumerate(vertices):
        if index >= len(layer.data):
            break
        target = layer.data[index]
        target.radius = source.get("radius", (0.25, 0.25))
        target.use_root = bool(source.get("root", False))
        target.use_loose = bool(source.get("loose", False))

def compatibility_block(doc, collection_name, entity_kind, identifier):
    compatibility = doc.get("compatibility", {}).get("blender_adapter", {})
    intermediate = compatibility.get("intermediate", {})
    mappings = compatibility.get("id_mappings", {})
    if collection_name == "worlds":
        candidates = [scene.get("world") for scene in intermediate.get("scenes", [])
                      if scene.get("world") is not None]
    else:
        candidates = intermediate.get(collection_name, [])
    for item in candidates:
        name = item.get("name")
        mapped = item.get("potter_id") or mappings.get(entity_kind + ":" + name)
        if mapped == identifier:
            return item
    return {}


def compatibility_name(doc, entity_kind, identifier, fallback):
    compatibility = doc.get("compatibility", {}).get("blender_adapter", {})
    for key, value in compatibility.get("id_mappings", {}).items():
        if key.startswith(entity_kind + ":") and value == identifier:
            return key.split(":", 1)[1]
    return fallback




def _resolve_compat_value(value, id_lookup):
    if isinstance(value, dict):
        if "id_type" in value and "name" in value:
            candidate = id_lookup.get(value["name"])
            if candidate is None or candidate.bl_rna.identifier != value["id_type"]:
                raise ValueError(
                    "compatibility data references unavailable ID %s (available: %s)"
                    % (value["name"], sorted(id_lookup)[:40]))
            return candidate
        return {key: _resolve_compat_value(item, id_lookup) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_compat_value(item, id_lookup) for item in value]
    return value
def _set_required(block, key, value, owner):
    if not hasattr(block, key):
        raise ValueError("%s cannot represent %s" % (owner, key))
    try:
        setattr(block, key, value)
    except Exception as error:
        raise ValueError("%s cannot restore %s" % (owner, key)) from error



def _resolve_deferred_id_assignments(id_lookup):
    for block, key, reference, owner in DEFERRED_ID_ASSIGNMENTS:
        candidate = id_lookup.get(reference)
        if candidate is None:
            raise ValueError("%s references unavailable ID %s" % (owner, reference))
        _set_required(block, key, candidate, owner)
    DEFERRED_ID_ASSIGNMENTS.clear()

def _restore_node_tree(tree, raw, id_lookup, owner):
    if raw.get("name"):
        tree["potter.graph_name"] = raw["name"]
    interface = getattr(tree, "interface", None)
    socket_map = {}
    if interface is not None:
        for direction, entries in (("INPUT", raw.get("interface", {}).get("inputs", [])),
                                   ("OUTPUT", raw.get("interface", {}).get("outputs", []))):
            for entry in entries:
                socket_type = entry.get("socket_type")
                if not socket_type:
                    raise ValueError("%s has an interface socket without a socket type" % owner)
                try:
                    socket = interface.new_socket(name=entry["name"], in_out=direction,
                                                  socket_type=socket_type)
                    socket_map[entry.get("id", "")] = socket.identifier
                    if direction == "INPUT" and entry.get("default") is not None and hasattr(socket, "default_value"):
                        socket.default_value = _resolve_compat_value(entry["default"], id_lookup)
                except Exception as error:
                    raise ValueError("%s cannot restore interface socket %s" %
                                     (owner, entry.get("name", ""))) from error
    tree.nodes.clear()
    nodes = {}
    for source in raw.get("nodes", []):
        try:
            node = tree.nodes.new(blender_node_type(source["type"], tree.bl_rna.identifier))
        except Exception as error:
            raise ValueError("%s cannot recreate node %s (%s)" %
                             (owner, source.get("name"), source.get("type"))) from error
        node.name = source.get("name", node.name)
        node.label = source.get("label", "")
        if source.get("location") is not None:
            node.location = source["location"]
        node_properties = dict(source.get("properties", {}))
        simple_material = node_properties.pop("potter_simple_material", False) is True
        node_color = node_properties.get("color")
        if node_color is not None and (not isinstance(node_color, list) or len(node_color) != 3):
            # Node color is display-only; discard malformed legacy presentation metadata.
            node_properties.pop("color", None)
        set_rna(node, node_properties, id_lookup, strict=True,
                owner="%s node %s" % (owner, node.name))
        set_id(node, source.get("potter_id"))
        custom_properties = dict(source.get("custom_properties", {}))
        if simple_material:
            custom_properties["potter_simple_material"] = True
        set_props(node, custom_properties)
        for socket_source in source.get("inputs", []):
            socket = next((value for value in node.inputs
                           if value.identifier == socket_source.get("identifier")), None)
            if socket is None:
                socket = next((value for value in node.inputs
                               if value.name == socket_source.get("name")), None)
            if socket is None or socket_source.get("default") is None or not hasattr(socket, "default_value"):
                continue
            try:
                socket.default_value = _resolve_compat_value(socket_source["default"], id_lookup)
            except Exception as error:
                raise ValueError("%s cannot restore input %s on node %s" %
                                 (owner, socket_source.get("name"), node.name)) from error
        nodes[node.name] = node
    for source in raw.get("links", []):
        from_node = nodes.get(source.get("from_node"))
        to_node = nodes.get(source.get("to_node"))
        if from_node is None or to_node is None:
            raise ValueError("%s has a link to an unavailable node" % owner)
        from_socket = next((value for value in from_node.outputs
                            if value.identifier == source.get("from_socket")), None)
        if from_socket is None:
            from_socket = next((value for value in from_node.outputs
                                if value.name == source.get("from_socket")), None)
        to_socket = next((value for value in to_node.inputs
                          if value.identifier == source.get("to_socket")), None)
        if to_socket is None:
            to_socket = next((value for value in to_node.inputs
                              if value.name == source.get("to_socket")), None)
        if from_socket is None or to_socket is None:
            raise ValueError("%s has an unavailable node socket" % owner)
        tree.links.new(from_socket, to_socket)
    return socket_map


def _model_group_tree(group):
    tree_type = {"geometry": "GeometryNodeTree", "shader": "ShaderNodeTree",
                 "compositor": "CompositorNodeTree"}.get(group.get("kind"), "")
    node_names = {identifier: node.get("name", identifier)
                  for identifier, node in group.get("nodes", {}).items()}
    nodes = []
    for identifier, node in group.get("nodes", {}).items():
        properties = dict(node.get("properties", {}))
        custom_properties = properties.pop("_blender_custom_properties", {})
        nodes.append({
            "name": node_names[identifier],
            "potter_id": identifier,
            "type": blender_node_type(node.get("type", "Node"), tree_type),
            "location": node.get("location", [0.0, 0.0]),
            "properties": properties,
            "custom_properties": custom_properties,
            "inputs": [{"identifier": name, "name": name, "default": value}
                       for name, value in node.get("inputs", {}).items()],
        })
    interface = group.get("interface", {})
    links = []
    for link in group.get("links", []):
        from_node = node_names.get(link.get("from_node"))
        to_node = node_names.get(link.get("to_node"))
        if from_node is None or to_node is None:
            raise ValueError("node group link references an unavailable node")
        links.append({
            "from_node": from_node,
            "from_socket": link.get("from_socket", ""),
            "to_node": to_node,
            "to_socket": link.get("to_socket", ""),
        })
    return {
        "name": group.get("name"),
        "type": tree_type,
        "interface": {
            direction: [{"id": socket.get("id"), "name": socket.get("name"),
                         "socket_type": POTTER_TO_BLENDER_NODE_SOCKET_TYPE.get(
                             socket.get("socket_type"), socket.get("socket_type")),
                         "default": socket.get("default")}
                        for socket in interface.get(direction, [])]
            for direction in ("inputs", "outputs")
        },
        "nodes": nodes,
        "links": links,
    }


def _create_object_override(reference, scene):
    view_layer = scene.view_layers[0]
    previous_active = view_layer.objects.active
    previous_selection = {obj.as_pointer() for obj in view_layer.objects if obj.select_get()}
    reference_pointer = reference.as_pointer()
    added_to_scene = not any(
        obj.as_pointer() == reference_pointer for obj in scene.collection.objects)
    if added_to_scene:
        scene.collection.objects.link(reference)
    try:
        area = next((area for area in bpy.context.screen.areas
                     if area.type == "VIEW_3D"), None)
        if area is None:
            raise ValueError("Blender has no 3D View context for library overrides")
        region = next((region for region in area.regions if region.type == "WINDOW"), None)
        if region is None:
            raise ValueError("Blender 3D View has no window region")
        with bpy.context.temp_override(
            window=bpy.context.window,
            screen=bpy.context.screen,
            scene=scene,
            view_layer=view_layer,
            area=area,
            region=region,
            object=reference,
            active_object=reference,
            selected_objects=[reference],
        ):
            reference_pointer = reference.as_pointer()
            for obj in view_layer.objects:
                obj.select_set(obj.as_pointer() == reference_pointer)
            view_layer.objects.active = reference
            reference.select_set(True, view_layer=view_layer)
            result = bpy.ops.object.make_override_library(collection=0)
        if "FINISHED" not in result:
            raise ValueError("Blender did not create a library override")
        override = next((
            obj for obj in bpy.data.objects
            if obj.override_library
            and obj.override_library.reference == reference
        ), None)
        if override is None:
            raise ValueError("Blender did not return the library override object")

        return override
    finally:
        for obj in view_layer.objects:
            obj.select_set(obj.as_pointer() in previous_selection)
        if previous_active is None or previous_active.name in view_layer.objects:
            view_layer.objects.active = previous_active
        if added_to_scene and any(
            obj.as_pointer() == reference_pointer for obj in scene.collection.objects
        ):
            scene.collection.objects.unlink(reference)

def _linked_source_name(doc, registry, identifier, name):
    marker = doc.get("compatibility", {}).get("linked_ids", {}).get(
        "%s:%s" % (registry, identifier), {})
    library_name = marker.get("library_name")
    suffix = " [%s]" % library_name if library_name else ""
    return name[:-len(suffix)] if suffix and name.endswith(suffix) else name



def _load_blend_libraries(doc, target):
    linked = {
        "nodes": {}, "collections": {}, "data_blocks": {}, "materials": {},
        "actions": {}, "node_groups": {}, "worlds": {}, "resources": {},
    }
    for library_id, library in doc.get("libraries", {}).items():
        reference = library.get("resource") or library.get("resolved_path")
        filepath = _resource_path(doc, reference)
        if not filepath or not os.path.isfile(filepath):
            raise ValueError("Blender library is missing: %s" % library.get("uri", library_id))
        blend_path = os.path.realpath(filepath)
        requests = {}
        for registry, identifiers in library.get("linked_ids", {}).items():
            for identifier in identifiers:
                entry = doc.get(registry, {}).get(identifier)
                if entry is None:
                    raise ValueError("linked %s item %s is missing" % (registry, identifier))
                if registry == "resources":
                    collection_name = "images"
                    name = entry.get("owner") or entry.get("filename")
                    if not name:
                        continue
                    name = _linked_source_name(doc, registry, identifier, name)
                    requests.setdefault((registry, collection_name), []).append(
                        (identifier, name))
                    continue
                if registry == "data_blocks":
                    data_type = entry.get("type")
                    target_collection = {
                        "mesh": ("meshes", "Mesh"),
                        "camera": ("cameras", "Camera"),
                        "light": ("lights", "Light"),
                        "armature": ("armatures", "Armature"),
                        "grease_pencil": ("grease_pencils_v3", "GreasePencil"),
                        "curve": ("curves", "Curve"),
                        "surface": ("curves", "Curve"),
                        "text": ("curves", "Curve"),
                        "volume": ("volumes", "Volume"),
                    }.get(data_type)
                    if target_collection is None:
                        continue
                    collection_name, entity_kind = target_collection
                    name = compatibility_name(doc, entity_kind, identifier, identifier)
                else:
                    collection_name = {
                        "nodes": "objects", "collections": "collections",
                        "materials": "materials", "actions": "actions",
                        "node_groups": "node_groups", "worlds": "worlds",
                    }.get(registry)
                    if collection_name is None:
                        continue
                    name = compatibility_name(
                        doc, {
                            "nodes": "Object", "collections": "Collection",
                            "materials": "Material", "actions": "Action",
                            "node_groups": "NodeGroup", "worlds": "World",
                        }[registry],
                        identifier,
                        entry.get("name", identifier))
                name = _linked_source_name(doc, registry, identifier, name)
                requests.setdefault((registry, collection_name), []).append(
                    (identifier, name))
        if not requests:
            continue
        with bpy.data.libraries.load(blend_path, link=True, relative=False) as (data_from, data_to):
            for (registry, collection_name), items in requests.items():
                if not hasattr(data_from, collection_name):
                    raise ValueError("Blender library has no %s IDs" % collection_name)
                available = set(getattr(data_from, collection_name))
                missing = [name for _, name in items if name not in available]
                if missing:
                    raise ValueError("Blender library is missing linked IDs: %s" % missing)
                setattr(data_to, collection_name, [name for _, name in items])
        for (registry, collection_name), items in requests.items():
            loaded = getattr(data_to, collection_name)
            for (identifier, name), block in zip(items, loaded):
                if block is None:
                    raise ValueError("Blender did not link %s %s" % (collection_name, name))
                linked[registry][identifier] = block
    for library in bpy.data.libraries:
        if getattr(library, "library", None) is not None:
            continue
        current_path = os.path.realpath(bpy.path.abspath(library.filepath))
        for library_record in doc.get("libraries", {}).values():
            reference = library_record.get("resource") or library_record.get("resolved_path")
            filepath = _resource_path(doc, reference)
            if filepath and current_path == os.path.realpath(filepath):
                library.filepath = _relative_blend_path(filepath, target)
                break
    return linked


def _create_geometry_groups(doc, linked_group_map=None):
    adapter = doc.get("compatibility", {}).get("blender_adapter", {})
    intermediate = adapter.get("intermediate", {})
    mappings = adapter.get("id_mappings", {})
    group_map = dict(linked_group_map or {})
    group_sources = {}
    direct_tree_ids = {material.get("node_tree")
                       for material in doc.get("materials", {}).values()
                       if material.get("node_tree")}
    direct_tree_ids.update(world.get("node_tree")
                           for world in doc.get("worlds", {}).values()
                           if world.get("node_tree"))
    def add_group(name, identifier, tree_type, tree_raw, properties=None, fake_user=False):
        if not tree_type or identifier in group_map:
            return
        tree = bpy.data.node_groups.new(name, tree_type)
        set_id(tree, identifier)
        set_props(tree, (properties or {}).get("custom_properties", {}))
        tree.use_fake_user = bool(fake_user or (properties or {}).get("fake_user", False))
        source = dict(properties or {})
        source["name"] = name
        source["potter_id"] = identifier
        source["tree"] = tree_raw
        group_map[name] = tree
        if identifier:
            group_map[identifier] = tree
        group_sources[name] = source
        if identifier:
            group_sources[identifier] = source

    for source in intermediate.get("node_groups", []):
        tree_raw = source.get("tree") or {}
        tree_type = tree_raw.get("type") or source.get("type")
        name = source.get("name", "Node Group")
        identifier = (source.get("potter_id") or mappings.get("NodeGroup:" + name)
                      or mappings.get("NodeTree:" + name))
        if identifier not in direct_tree_ids:
            add_group(name, identifier, tree_type, tree_raw, source, source.get("fake_user", False))

    for identifier, group in doc.get("node_groups", {}).items():
        name = compatibility_name(doc, "NodeGroup", identifier,
                                  group.get("name") or identifier)
        tree_type = {"geometry": "GeometryNodeTree", "shader": "ShaderNodeTree",
                     "compositor": "CompositorNodeTree"}.get(group.get("kind"), "")
        if identifier not in direct_tree_ids:
            add_group(name, identifier, tree_type, _model_group_tree(group))
    return group_map, group_sources


def _populate_geometry_groups(group_map, group_sources, id_lookup):
    socket_maps = {}
    id_lookup.update({tree.name_full: tree for tree in group_map.values()})
    populated = set()
    for name, source in group_sources.items():
        tree = group_map.get(name)
        if tree is None:
            continue
        pointer = tree.as_pointer()
        if pointer not in populated:
            socket_map = _restore_node_tree(tree, source.get("tree") or {}, id_lookup,
                                            "Node group %s" % tree.name)
            populated.add(pointer)
        else:
            socket_map = socket_maps.get(tree.name_full, {})
        socket_maps[name] = socket_map
        socket_maps[tree.name_full] = socket_map
    return socket_maps




def _restore_shape_keys(obj, mesh_source):
    if not mesh_source:
        return
    shape_keys = mesh_source.get("shape_keys") or {}
    basis = shape_keys.get("basis", [])
    if not shape_keys.get("keys") and not basis:
        return
    key_blocks = {}
    key_ids = {}
    absolute = bool(shape_keys.get("absolute", False))
    key_items = list(shape_keys.get("keys", []))
    if absolute:
        key_items.sort(key=lambda item: float(item.get("frame", 0.0)))
    try:
        basis_key = obj.shape_key_add(name="Basis", from_mix=False)
        key_blocks[basis_key.name] = basis_key
        for point in basis:
            index = int(point["index"])
            if index >= len(basis_key.data):
                raise ValueError("shape-key basis vertex index is out of range")
            basis_key.data[index].co = point["co"]
        for item in key_items:
            key = obj.shape_key_add(name=item["name"], from_mix=False)
            key_blocks[key.name] = key
            if item.get("potter_id"):
                key_ids[key.name] = item["potter_id"]
            for point in item.get("positions", []):
                index = int(point["index"])
                if index >= len(key.data):
                    raise ValueError("shape-key vertex index is out of range")
                key.data[index].co = point["co"]
            key.value = float(item.get("value", 0.0))
            key.mute = bool(item.get("mute", False))
            key.slider_min = float(item.get("slider_min", 0.0))
            key.slider_max = float(item.get("slider_max", 1.0))
            key.vertex_group = item.get("vertex_group", "")
            if absolute:
                try:
                    key.frame = float(item.get("frame", 0.0))
                except (AttributeError, TypeError):
                    pass
        key_data = obj.data.shape_keys
        if key_ids:
            key_data["potter.shape_key_ids_json"] = json.dumps(key_ids, sort_keys=True)
        key_data.use_relative = not absolute
        key_data.eval_time = float(shape_keys.get("evaluation_time", 0.0))
        if not absolute:
            for item in key_items:
                key = key_blocks[item["name"]]
                relative = key_blocks.get(item.get("relative_key") or "Basis")
                if relative is None:
                    raise ValueError("shape key references missing relative key %s" %
                                     item.get("relative_key"))
                key.relative_key = relative
    except Exception as error:
        raise ValueError("cannot restore shape keys on %s" % obj.name) from error


def _restore_vertex_groups(obj, groups):
    group_ids = {}
    for source in groups:
        group = obj.vertex_groups.new(name=source["name"])
        if source.get("potter_id"):
            group_ids[group.name] = source["potter_id"]
        for item in source.get("weights", []):
            index = int(item["vertex_index"])
            if index < 0 or index >= len(obj.data.vertices):
                raise ValueError("vertex group %s references an out-of-range vertex" % group.name)
            try:
                group.add([index], float(item["weight"]), "REPLACE")
            except Exception as error:
                raise ValueError("cannot restore weight for vertex group %s" % group.name) from error
    if group_ids:
        obj["potter.vertex_group_ids_json"] = json.dumps(group_ids, sort_keys=True)


def _restore_grease_pencil(data, raw, materials):
    for layer_raw in raw.get("layers", []):
        layer = data.layers.new(layer_raw["name"], set_active=True)
        layer.opacity = float(layer_raw.get("opacity", 1.0))
        layer.hide = not bool(layer_raw.get("visible", True))
        for frame_raw in layer_raw.get("frames", []):
            frame = layer.frames.new(int(frame_raw["frame"]))
            strokes = frame_raw.get("strokes", [])
            sizes = [len(stroke.get("points", [])) for stroke in strokes]
            if any(size < 1 for size in sizes):
                raise ValueError("Grease Pencil strokes must contain at least one point")
            drawing = frame.drawing
            if sizes:
                drawing.add_strokes(sizes)
            offsets = list(getattr(drawing, "curve_offsets", ()))
            offsets = [int(getattr(value, "value", value)) for value in offsets]
            if len(offsets) != len(strokes) + 1:
                raise ValueError("cannot restore Grease Pencil stroke offsets")

            def ensure_attribute(name, data_type, domain):
                attributes = getattr(drawing, "attributes", None)
                if attributes is None:
                    raise ValueError("Grease Pencil drawing does not support attributes")
                attribute = attributes.get(name)
                if attribute is None:
                    attribute = attributes.new(name=name, type=data_type, domain=domain)
                return attribute

            attr_specs = (("position", "FLOAT_VECTOR", "POINT", "position"),
                          ("pressure", "FLOAT", "POINT", "pressure"),
                          ("radius", "FLOAT", "POINT", "radius"),
                          ("opacity", "FLOAT", "POINT", "opacity"),
                          ("time", "FLOAT", "POINT", "time"),
                          ("cyclic", "BOOLEAN", "CURVE", "cyclic"),
                          ("fill", "BOOLEAN", "CURVE", "fill"),
                          ("material_index", "INT", "CURVE", "material_index"))
            attrs = {name: ensure_attribute(name, data_type, domain)
                     for name, data_type, domain, _ in attr_specs if strokes}
            for stroke_index, stroke in enumerate(strokes):
                material = materials.get(stroke.get("material"))
                if stroke.get("material") is not None and material is None:
                    raise ValueError("Grease Pencil stroke references missing material %s" % stroke["material"])
                if material is not None and material.name not in [slot.name for slot in data.materials]:
                    data.materials.append(material)
                material_index = next((index for index, slot in enumerate(data.materials)
                                       if material is not None and slot == material), 0)
                start, end = offsets[stroke_index], offsets[stroke_index + 1]
                for local_index, point in enumerate(stroke.get("points", [])):
                    index = start + local_index
                    for name, value in (("position", point["position"]),
                                        ("pressure", point["pressure"]), ("radius", point["radius"]),
                                        ("opacity", point["opacity"]), ("time", point["time"])):
                        item = attrs[name].data[index]
                        if hasattr(item, "vector"):
                            item.vector = value
                        elif hasattr(item, "value"):
                            item.value = value
                        else:
                            raise ValueError("Grease Pencil %s attribute is not writable" % name)
                for name, value in (("cyclic", bool(stroke.get("cyclic", False))),
                                    ("fill", bool(stroke.get("fill", False))),
                                    ("material_index", int(material_index))):
                    item = attrs[name].data[stroke_index]
                    if not hasattr(item, "value"):
                        raise ValueError("Grease Pencil %s attribute is not writable" % name)
                    item.value = value




def _restore_armature(obj, source, scene):
    bones = source.get("bones", [])
    if not bones:
        return
    context = bpy.context
    # mode_set ignores scene/view_layer overrides in background builds, so the
    # object must be active and selected in the real view layer.
    if context.window and scene is not context.scene:
        context.window.scene = scene
    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    previous_selection = [item for item in view_layer.objects if item.select_get()]
    if obj.name in view_layer.objects:
        for item in view_layer.objects:
            item.select_set(item is obj)
        view_layer.objects.active = obj
    with context.temp_override(object=obj, active_object=obj,
                               selected_objects=[obj],
                               selected_editable_objects=[obj]):
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            if getattr(obj.data, "is_editmode", False) is not True:
                raise RuntimeError("armature %s did not enter edit mode" % obj.name)
            edit_bones = {}
            bone_ids = {}
            for item in bones:
                bone = obj.data.edit_bones.new(item["name"])
                bone.head = item["head"]
                bone.tail = item["tail"]
                bone.roll = float(item.get("roll", 0.0))
                bone.use_deform = bool(item.get("deform", True))
                bone.use_inherit_rotation = bool(item.get("inherit_rotation", True))
                bone.use_connect = bool(item.get("use_connect", False))
                if item.get("potter_id"):
                    bone_ids[bone.name] = item["potter_id"]
                edit_bones[item["name"]] = bone
            for item in bones:
                parent = item.get("parent")
                if parent:
                    if parent not in edit_bones:
                        raise ValueError("armature bone references missing parent %s" % parent)
                    edit_bones[item["name"]].parent = edit_bones[parent]
            for item in bones:
                settings = item.get("bbone_settings", {})
                if settings is None:
                    settings = {}
                if not isinstance(settings, dict):
                    raise ValueError("armature bone B-Bone settings must be an object")
                bone = edit_bones[item["name"]]
                owner = "armature bone %s" % item["name"]
                for key, value in settings.items():
                    if key in ("bbone_custom_handle_start", "bbone_custom_handle_end"):
                        continue
                    _set_required(bone, key, value, owner)
                for key in ("bbone_custom_handle_start", "bbone_custom_handle_end"):
                    if key not in settings:
                        continue
                    handle_name = settings[key]
                    if handle_name is None:
                        handle = None
                    elif isinstance(handle_name, str):
                        handle = edit_bones.get(handle_name)
                        if handle is None:
                            raise ValueError(
                                "%s references missing B-Bone handle %s"
                                % (owner, handle_name))
                    else:
                        raise ValueError("%s has an invalid B-Bone handle name" % owner)
                    _set_required(bone, key, handle, owner)
            bpy.ops.object.mode_set(mode="OBJECT")
            if bone_ids:
                obj.data["potter.bone_ids_json"] = json.dumps(bone_ids, sort_keys=True)
        except Exception as error:
            if getattr(obj.data, "is_editmode", False):
                bpy.ops.object.mode_set(mode="OBJECT")
            raise ValueError("cannot restore armature bones on %s" % obj.name) from error
        finally:
            for item in view_layer.objects:
                item.select_set(item in previous_selection)
            if previous_active is not None and previous_active.name in view_layer.objects:
                view_layer.objects.active = previous_active

def _constraint_is_graph_editable(source):
    kind = source.get("type")
    if kind not in BLENDER_TO_POTTER_CONSTRAINT:
        return False
    if kind != "ACTION":
        return True
    properties = source.get("properties") or {}
    return all(
        isinstance(properties.get(key), dict) and properties[key].get("name")
        for key in ("action", "target")
    )


def _canonical_pose(doc, node, source_pose):
    """Overlay canonical pose values and pose constraints onto compat details."""
    model_pose = _pose_from_model(doc, node)
    if not model_pose:
        return source_pose
    by_bone = {item.get("bone"): item for item in source_pose or []}
    result = []
    seen = set()
    for model_item in model_pose:
        bone = model_item.get("bone")
        seen.add(bone)
        source = by_bone.get(bone) or {}
        merged = dict(source)
        merged["bone"] = bone
        merged.update({key: model_item[key] for key in
                       ("location", "rotation_quaternion", "scale", "lock_ik",
                        "use_ik_limit", "ik_min", "ik_max", "ik_stiffness",
                        "ik_stretch")})
        source_constraints = {item.get("name"): item
                              for item in source.get("constraints", [])}
        constraints = []
        for constraint in model_item.get("constraints", []):
            compat = source_constraints.get(constraint.get("name"), {})
            merged_constraint = dict(compat)
            merged_constraint.update(constraint)
            merged_constraint.pop("owner_bone", None)
            merged_constraint["custom_properties"] = compat.get("custom_properties", {})
            constraints.append(merged_constraint)
        constraints.extend(
            constraint for constraint in source.get("constraints", [])
            if not _constraint_is_graph_editable(constraint)
        )
        merged["constraints"] = constraints
        result.append(merged)
    for bone, source in by_bone.items():
        if bone not in seen:
            result.append(source)
    return result


def _canonical_constraints(doc, node, source_constraints):
    """Rebuild object constraints from the canonical potter graph.

    Compatibility payload only contributes custom properties for matching
    constraints; graph deletions stay deleted.
    """
    model = [item for item in _constraints_from_model(doc, node)
             if item.get("owner_bone") is None]
    compat_by_name = {item.get("name"): item for item in source_constraints or []}
    result = []
    for constraint in model:
        source = compat_by_name.get(constraint.get("name"), {})
        merged = dict(constraint)
        merged["custom_properties"] = source.get("custom_properties", {})
        result.append(merged)
    result.extend(
        constraint for constraint in source_constraints or []
        if not _constraint_is_graph_editable(constraint)
    )
    return result




def _canonical_drivers(doc, node, source_drivers):
    """Rebuild drivers from the canonical potter graph, keeping compatibility
    keyframe handles for unchanged values."""
    model = _drivers_from_model(doc, node)
    by_key = {}
    for source in source_drivers or []:
        curve = source.get("curve") or {}
        by_key[(curve.get("path"), int(curve.get("index", 0)))] = source
    result = []
    for driver in model:
        curve = driver.get("curve") or {}
        source = by_key.get((curve.get("path"), int(curve.get("index", 0))), {})
        merged_curves = _canonical_fcurves([dict(curve)], [source.get("curve") or {}], doc)
        result.append({
            "curve": merged_curves[0] if merged_curves else dict(curve),
            "driver": driver.get("driver"),
        })
    return result


def _canonical_nla(doc, node, source_tracks):
    """Rebuild NLA tracks from the canonical potter graph; compatibility
    payload only fills properties the graph does not model."""
    model = _nla_from_model(doc, node)
    compat_by_track = {item.get("name"): item for item in source_tracks or []}
    result = []
    for track in model:
        source_track = compat_by_track.get(track.get("name"), {})
        compat_strips = {item.get("name"): item
                         for item in source_track.get("strips", [])}
        strips = []
        for strip in track.get("strips", []):
            source_strip = compat_strips.get(strip.get("name"), {})
            merged_properties = dict(source_strip.get("properties", {}))
            merged_properties.update(strip.get("properties", {}))
            merged = dict(source_strip)
            merged.update(strip)
            merged["properties"] = merged_properties
            if source_strip.get("custom_properties") and not strip.get("custom_properties"):
                merged["custom_properties"] = source_strip["custom_properties"]
            strips.append(merged)
        merged_track = dict(source_track)
        merged_track.update(track)
        merged_track["strips"] = strips
        result.append(merged_track)
    return result


def _is_simple_principled_tree(tree_source):
    nodes = (tree_source or {}).get("nodes", []) or []
    types = [item.get("type") for item in nodes]
    links = (tree_source or {}).get("links", []) or []
    has_principled = "ShaderNodeBsdfPrincipled" in types
    has_output = any(kind in ("OutputMaterial", "ShaderNodeOutputMaterial")
                     for kind in types)
    return (has_principled and has_output and links
            and all(link.get("from_socket") == "BSDF"
                    and link.get("to_socket") == "Surface" for link in links))


def _set_object_solver_inverse(owner, constraint, owner_type, pose_bone=None,
                               inverse_frame=None):
    scene = bpy.context.scene
    previous_frame = scene.frame_current
    previous_subframe = getattr(scene, "frame_subframe", 0.0)
    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = list(bpy.context.selected_objects)
    previous_mode = owner.mode
    try:
        if inverse_frame is not None:
            frame = float(inverse_frame)
            base_frame = math.floor(frame)
            scene.frame_set(int(base_frame), subframe=frame - base_frame)
        if owner.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for selected in previous_selected:
            selected.select_set(False)
        owner.select_set(True)
        view_layer.objects.active = owner
        if owner_type == "BONE":
            owner.data.bones.active = pose_bone.bone
            bpy.ops.object.mode_set(mode="POSE")
        with bpy.context.temp_override(
                object=owner, active_object=owner, selected_objects=[owner],
                selected_editable_objects=[owner], scene=scene, view_layer=view_layer):
            result = bpy.ops.constraint.objectsolver_set_inverse(
                constraint=constraint.name, owner=owner_type)
        if "FINISHED" not in result:
            raise ValueError("Blender did not set the Object Solver inverse")
    finally:
        if owner.mode != previous_mode:
            bpy.ops.object.mode_set(mode=previous_mode)
        owner.select_set(False)
        for selected in previous_selected:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        view_layer.objects.active = previous_active
        if inverse_frame is not None:
            scene.frame_set(previous_frame, subframe=previous_subframe)


def _restore_pose(obj, pose, id_lookup, doc, target):
    for item in pose or []:
        bone = obj.pose.bones.get(item.get("bone"))
        if bone is None:
            raise ValueError("pose references missing armature bone %s" % item.get("bone"))
        bone.rotation_mode = item.get("rotation_mode", "QUATERNION")
        rotation = item.get("rotation_quaternion", (1.0, 0.0, 0.0, 0.0))
        if bone.rotation_mode == "QUATERNION":
            bone.rotation_quaternion = rotation
        elif bone.rotation_mode == "AXIS_ANGLE":
            angle, axis = Quaternion((rotation[0], rotation[1], rotation[2], rotation[3])).to_axis_angle()
            bone.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
        else:
            bone.rotation_euler = Quaternion(
                (rotation[0], rotation[1], rotation[2], rotation[3])).to_euler(
                    bone.rotation_mode)
        bone.scale = item.get("scale", (1.0, 1.0, 1.0))
        bone.lock_ik_x, bone.lock_ik_y, bone.lock_ik_z = item.get(
            "lock_ik", (False, False, False))
        bone.use_ik_limit_x, bone.use_ik_limit_y, bone.use_ik_limit_z = item.get(
            "use_ik_limit", (False, False, False))
        bone.ik_min_x, bone.ik_min_y, bone.ik_min_z = item.get(
            "ik_min", (-math.pi, -math.pi, -math.pi))
        bone.ik_max_x, bone.ik_max_y, bone.ik_max_z = item.get(
            "ik_max", (math.pi, math.pi, math.pi))
        bone.ik_stiffness_x, bone.ik_stiffness_y, bone.ik_stiffness_z = item.get(
            "ik_stiffness", (0.0, 0.0, 0.0))
        bone.ik_stretch = item.get("ik_stretch", 0.0)
        for source in item.get("constraints", []):
            constraint = bone.constraints.new(source["type"])
            constraint.name = source.get("name", constraint.name)
            properties = dict(source.get("properties", {}))
            inverse_matrix = properties.pop("_potter_inverse_matrix", None)
            inverse_frame = properties.pop("_potter_inverse_frame", None)
            if source["type"] == "TRANSFORM_CACHE":
                resource_id = properties.pop("resource")
                object_path = properties.pop("object_path")
                cache_file = _cache_file_for_resource(
                    doc, resource_id, target, properties)
                constraint.cache_file = cache_file
                constraint.object_path = object_path
                for key in ("frame_offset", "scale", "override_frame"):
                    properties.pop(key, None)
            if source["type"] == "ARMATURE":
                _restore_armature_targets(
                    constraint, properties, id_lookup,
                    "pose constraint %s target" % constraint.name)
            set_rna(constraint, properties, id_lookup, strict=True,
                    owner="pose constraint %s" % constraint.name)
            set_props(constraint, source.get("custom_properties", {}))
            if inverse_matrix is not None:
                _set_object_solver_inverse(
                    obj, constraint, "BONE", bone, inverse_frame)
                _store_object_solver_inverse(
                    obj, constraint.name, inverse_matrix, bone.name, inverse_frame)

def _armature_from_model(doc, data_id, source):
    armature = doc.get("data_blocks", {}).get(data_id, {}).get("armature", {})
    bones = armature.get("bones", {})
    model_bones = {}
    for identifier, bone in bones.items():
        parent = bones.get(bone.get("parent"), {})
        model_bones[bone.get("name", identifier)] = {
            "name": bone.get("name", identifier),
            "potter_id": identifier,
            "parent": parent.get("name"),
            "head": bone.get("head", [0.0, 0.0, 0.0]),
            "tail": bone.get("tail", [0.0, 0.0, 1.0]),
            "roll": bone.get("roll", 0.0),
            "deform": bone.get("deform", True),
            "inherit_rotation": bone.get("inherit_rotation", True),
            "use_connect": bone.get("use_connect", False),
            "bbone_settings": bone.get("bbone_settings", {}),
        }
    result = dict(source)
    if source.get("bones"):
        merged_bones = []
        for source_bone in source["bones"]:
            model_bone = model_bones.get(source_bone.get("name"))
            if model_bone:
                merged_bones.append(dict(source_bone, **model_bone))
            else:
                merged_bones.append(source_bone)
        source_names = {bone.get("name") for bone in source["bones"]}
        merged_bones.extend(bone for name, bone in model_bones.items()
                            if name not in source_names)
        result["bones"] = merged_bones
        return result
    result["bones"] = list(model_bones.values())
    return result


def _pose_from_model(doc, node):
    data_id = node.get("data")
    armature = doc.get("data_blocks", {}).get(data_id, {}).get("armature", {})
    bones = armature.get("bones", {})
    pose_by_bone = node.get("pose", {})
    owner_bones = {
        constraint.get("owner_bone")
        for constraint in node.get("constraints", [])
        if constraint.get("owner_bone")
    }
    bone_ids = sorted(set(bones) | set(pose_by_bone) | owner_bones)
    model_constraints = _constraints_from_model(doc, node)
    result = []
    for bone_id in bone_ids:
        bone = bones.get(bone_id)
        if bone is None:
            raise ValueError("pose references missing armature bone %s" % bone_id)
        pose = pose_by_bone.get(bone_id, {})
        rotation = pose.get("rotation", [0.0, 0.0, 0.0, 1.0])
        constraints = []
        for constraint in model_constraints:
            if constraint.get("owner_bone") == bone_id:
                pose_constraint = dict(constraint)
                pose_constraint.pop("owner_bone", None)
                constraints.append(pose_constraint)
        result.append({
            "bone": bone.get("name", bone_id),
            "location": pose.get("translation", [0.0, 0.0, 0.0]),
            "rotation_quaternion": [rotation[3], rotation[0], rotation[1], rotation[2]],
            "scale": pose.get("scale", [1.0, 1.0, 1.0]),
            "lock_ik": pose.get("lock_ik", [False, False, False]),
            "use_ik_limit": pose.get("use_ik_limit", [False, False, False]),
            "ik_min": pose.get("ik_min", [-math.pi, -math.pi, -math.pi]),
            "ik_max": pose.get("ik_max", [math.pi, math.pi, math.pi]),
            "ik_stiffness": pose.get("ik_stiffness", [0.0, 0.0, 0.0]),
            "ik_stretch": pose.get("ik_stretch", 0.0),
            "constraints": constraints,
        })
    return result


def _constraints_from_model(doc, node):
    result = []
    for source in node.get("constraints", []):
        kind = POTTER_TO_BLENDER_CONSTRAINT.get(source.get("type"))
        if kind is None:
            continue
        properties = dict(source.get("params", {}))
        target_id = source.get("target")
        target = doc.get("nodes", {}).get(target_id)
        if target_id and target is None:
            raise ValueError("constraint references missing target %s" % target_id)
        if kind == "ARMATURE":
            for armature_target in properties.get("targets", []):
                armature_target_ref = armature_target.get("target")
                if isinstance(armature_target_ref, dict):
                    target_name = armature_target_ref.get("name")
                    armature_target_node = next(
                        (candidate for candidate in doc.get("nodes", {}).values()
                         if candidate.get("name") == target_name),
                        None)
                elif isinstance(armature_target_ref, str):
                    armature_target_node = doc.get("nodes", {}).get(armature_target_ref)
                else:
                    armature_target_node = None
                if armature_target_ref and armature_target_node is None:
                    raise ValueError(
                        "armature constraint references missing target %s" %
                        armature_target_ref)
                armature_target["target"] = (
                    {"id_type": "Object", "name": armature_target_node["name"]}
                    if armature_target_node else None)
                subtarget_id = armature_target.get("subtarget")
                if subtarget_id and armature_target_node:
                    armature_id = armature_target_node.get("data")
                    bones = doc.get("data_blocks", {}).get(armature_id, {}).get(
                        "armature", {}).get("bones", {})
                    subtarget = bones.get(subtarget_id)
                    if subtarget is None:
                        subtarget = next(
                            (bone for bone in bones.values()
                             if bone.get("name") == subtarget_id),
                            None)
                    if subtarget is None:
                        raise ValueError(
                            "armature constraint references missing target bone %s" %
                            subtarget_id)
                    armature_target["subtarget"] = subtarget.get("name", "")
        elif target is not None:
            properties["target"] = {"id_type": "Object", "name": target["name"]}
        inverse_matrix = source.get("inverse_matrix")
        inverse_frame = source.get("inverse_frame")
        if kind == "OBJECT_SOLVER" and inverse_matrix is not None:
            properties["_potter_inverse_matrix"] = inverse_matrix
            if inverse_frame is not None:
                properties["_potter_inverse_frame"] = inverse_frame
        subtarget_id = source.get("subtarget")
        if subtarget_id and kind != "ARMATURE":
            armature_id = target.get("data") if target else None
            bones = doc.get("data_blocks", {}).get(armature_id, {}).get(
                "armature", {}).get("bones", {})
            subtarget = bones.get(subtarget_id)
            if subtarget is None:
                raise ValueError("constraint references missing subtarget bone %s" % subtarget_id)
            properties["subtarget"] = subtarget.get("name", "")
        pole_target_id = properties.get("pole_target")
        if pole_target_id:
            pole_target = doc.get("nodes", {}).get(pole_target_id)
            if pole_target is None:
                raise ValueError("IK pole target references missing object %s" % pole_target_id)
            properties["pole_target"] = {
                "id_type": "Object", "name": pole_target["name"]}
        properties["influence"] = source.get("influence", 1.0)
        properties["mute"] = not source.get("enabled", True)
        result.append({"name": source.get("name", "Constraint"), "type": kind,
                       "properties": properties,
                       "custom_properties": {},
                       "owner_bone": source.get("owner_bone")})
    return result


def _drivers_from_model(doc, node):
    types = {"average": "AVERAGE", "sum": "SUM", "min": "MIN", "max": "MAX",
             "scripted_expression": "SCRIPTED"}
    variable_types = {"single_prop": "SINGLE_PROP", "transforms": "TRANSFORMS"}
    result = []
    for source in node.get("drivers", []):
        variables = []
        for item in source.get("variables", []):
            target_node = doc.get("nodes", {}).get(item.get("target"))
            if target_node is None:
                raise ValueError("driver references missing target object %s" % item.get("target"))
            variables.append({
                "name": item.get("name", "var"),
                "type": variable_types.get(item.get("type"), "SINGLE_PROP"),
                "targets": [{"id_type": "OBJECT",
                             "id": {"id_type": "Object", "name": target_node["name"]},
                             "data_path": item.get("path", "")}],
            })
        driver_type = types.get(source.get("type"), "SCRIPTED")
        result.append({
            "curve": {"path": source.get("path", ""),
                      "index": int(source.get("index", 0)),
                      "keyframes": [], "extrapolation": "CONSTANT"},
            "driver": {"type": driver_type,
                       "expression": source.get("expression") or "0",
                       "variables": variables},
        })
    return result


def _vertex_groups_from_model(doc, data_id):
    data = doc.get("data_blocks", {}).get(data_id, {})
    mesh = data.get("mesh") or {}
    vertex_indexes = {str(vertex["id"]): index
                      for index, vertex in enumerate(mesh.get("vertices", []))}
    weights = data.get("vertex_weights", {})
    result = []
    for group in data.get("vertex_groups", []):
        group_id = group.get("id")
        group_weights = []
        for vertex_id, index in vertex_indexes.items():
            amount = weights.get(vertex_id, {}).get(group_id)
            if amount is not None:
                group_weights.append({"vertex_index": index, "weight": amount})
        result.append({"name": group.get("name", "Vertex Group"),
                       "potter_id": group_id, "weights": group_weights})
    return result


def _shape_keys_from_model(doc, data_id):
    data = doc.get("data_blocks", {}).get(data_id, {})
    shape = data.get("shape_keys")
    if not shape:
        return None
    mesh = data.get("mesh") or {}
    vertex_indexes = {str(vertex["id"]): index
                      for index, vertex in enumerate(mesh.get("vertices", []))}
    groups = {group.get("id"): group.get("name")
              for group in data.get("vertex_groups", [])}
    keys = shape.get("keys", {})
    result_keys = []
    for identifier, key in keys.items():
        relative = keys.get(key.get("relative_key"), {})
        positions = [{"index": vertex_indexes[vertex_id], "co": position}
                     for vertex_id, position in key.get("positions", {}).items()
                     if vertex_id in vertex_indexes]
        result_keys.append({
            "name": key.get("name", identifier), "potter_id": identifier,
            "value": key.get("value", 0.0), "mute": bool(key.get("mute", False)),
            "slider_min": key.get("slider_min", 0.0), "slider_max": key.get("slider_max", 1.0),
            "frame": key.get("frame", 0.0),
            "relative_key": relative.get("name", "Basis"),
            "vertex_group": groups.get(key.get("vertex_group"), ""),
            "positions": positions,
        })
    if shape.get("absolute"):
        result_keys.sort(key=lambda key: float(key.get("frame", 0.0)))
    basis = [{"index": vertex_indexes[vertex_id], "co": position}
             for vertex_id, position in shape.get("basis", {}).items()
             if vertex_id in vertex_indexes]
    action = doc.get("actions", {}).get(shape.get("action"))
    return {"absolute": bool(shape.get("absolute", False)),
            "evaluation_time": shape.get("evaluation_time", 0.0),
            "action": action.get("name") if action else None,
            "action_slot": shape.get("action_slot"),
            "muted_action_curves": sorted(shape.get("muted_action_curves", [])),
            "basis": basis, "keys": result_keys}




def _resource_record_for_reference(doc, reference):
    if not reference:
        return None
    for resource_id, resource in doc.get("resources", {}).items():
        if (resource_id == reference or resource.get("uri") == reference
                or resource.get("original_path") == reference):
            return resource
    return None


def _resource_path(doc, reference):
    resource = _resource_record_for_reference(doc, reference)
    path = (resource.get("export_path") if resource else None)
    if not path and resource:
        path = resource.get("uri")
    if not path:
        path = reference
    if not path:
        return None
    path = path[7:] if path.startswith("file://") else path
    return os.path.abspath(path) if not path.startswith("//") else bpy.path.abspath(path)


def _relative_blend_path(path, target):
    relative = os.path.relpath(os.path.realpath(path),
                               os.path.dirname(os.path.realpath(target)))
    return "//" + relative.replace(os.sep, "/")


def _cache_file_for_resource(doc, reference, target, params=None):
    filepath = _resource_path(doc, reference)
    if not filepath or not os.path.isfile(filepath):
        raise ValueError("Alembic cache resource is missing: %s" % reference)
    params = params or {}
    resource = _resource_record_for_reference(doc, reference)
    identity = (
        str(resource.get("owner", reference))
        if resource and resource.get("kind") == "alembic_cache"
        else None
    )
    cache_key = (os.path.realpath(filepath), identity)
    cached = CACHE_FILES_BY_IDENTITY.get(cache_key)
    if cached is not None:
        return cached
    cache_file = _cache_file_from_alembic(filepath, identity)
    cache_file.filepath = _relative_blend_path(filepath, target)
    if resource:
        if identity is not None:
            cache_file.name = identity
        set_rna(cache_file, resource.get("cache_file", {}), {}, strict=True,
                owner="Alembic CacheFile")
    params = params or {}
    if "frame_offset" in params:
        cache_file.frame_offset = float(params["frame_offset"])
    if "scale" in params:
        cache_file.scale = float(params["scale"])
    override_frame = params.get("override_frame")
    cache_file.override_frame = override_frame is not None
    if override_frame is not None:
        cache_file.frame = float(override_frame)
    CACHE_FILES_BY_IDENTITY[cache_key] = cache_file
    return cache_file
def _movie_clip_camera_values(clip, camera_source):
    property_names = {
        "sensor_width_mm": "sensor_width",
        "principal": "principal_point",
    }
    camera = clip.tracking.camera
    for key in (
            "units", "distortion_model", "sensor_width_mm", "pixel_aspect", "principal"):
        if key not in camera_source:
            continue
        try:
            setattr(camera, property_names.get(key, key), camera_source[key])
        except Exception as error:
            raise ValueError("cannot restore MovieClip camera property %s" % key) from error
    if "focal_mm" in camera_source:
        try:
            focal = float(camera_source["focal_mm"])
            if camera.units == "MILLIMETERS":
                width = float(clip.size[0])
                if width <= 0.0 or camera.sensor_width <= 0.0:
                    raise ValueError("MovieClip dimensions and sensor width must be positive")
                focal_pixels = focal * width / float(camera.sensor_width)
            else:
                focal_pixels = focal
            camera.focal_length_pixels = focal_pixels
        except Exception as error:
            raise ValueError("cannot restore MovieClip camera property focal_mm") from error
    for key, value in camera_source.items():
        if key in (
                "focal_mm", "sensor_width_mm", "principal", "units",
                "distortion_model", "pixel_aspect"):
            continue
        try:
            setattr(camera, key, value)
        except Exception as error:
            raise ValueError("cannot restore MovieClip camera property %s" % key) from error




def _restore_movie_clip_tracking(clip, tracking):
    clip["potter.tracking_json"] = json.dumps(tracking, sort_keys=True)
    track_ids = {
        track_source.get("id"): track_source
        for track_source in tracking.get("tracks", [])
    }
    object_sources = list(tracking.get("objects", []))
    assigned_tracks = {
        track_id
        for object_source in object_sources
        for track_id in object_source.get("tracks", [])
    }
    unassigned_tracks = [
        track_id for track_id in track_ids if track_id not in assigned_tracks]
    if unassigned_tracks:
        camera_source = next(
            (source for source in object_sources
             if source.get("name") == "Camera" or source.get("id") == "Camera"),
            None)
        if camera_source is None:
            camera_source = {"id": "Camera", "name": "Camera", "tracks": []}
            object_sources.append(camera_source)
        camera_source["tracks"] = (
            list(camera_source.get("tracks", [])) + unassigned_tracks)
    for object_source in object_sources:
        object_name = object_source.get("name") or object_source.get("id")
        if not object_name:
            continue
        tracking_object = clip.tracking.objects.get(object_name)
        if tracking_object is None:
            tracking_object = clip.tracking.objects.new(object_name)
        tracking_object.scale = float(object_source.get("scale", 1.0))
        object_frames = []
        for track_id in object_source.get("tracks", []):
            source = track_ids.get(track_id)
            if source is None:
                raise ValueError("MovieClip tracking object references missing track %s" % track_id)
            markers = source.get("markers", [])
            first_frame = int(round(float(markers[0].get("frame", 1)))) if markers else 1
            track = tracking_object.tracks.new(
                name=source.get("name", track_id), frame=first_frame)
            for marker_source in markers:
                frame = int(round(float(marker_source.get("frame", first_frame))))
                object_frames.append(frame)
                coordinate = marker_source.get("co", (0.0, 0.0))
                marker = track.markers.find_frame(frame)
                if marker is None:
                    marker = track.markers.insert_frame(frame, co=coordinate)
                else:
                    marker.co = coordinate
                marker.mute = bool(marker_source.get("disabled", False))
                pattern = marker_source.get("pattern_corners")
                if pattern is not None:
                    marker.pattern_corners = pattern
                search = marker_source.get("search_area")
                if isinstance(search, (list, tuple)) and len(search) >= 3:
                    marker.search_min = search[0]
                    marker.search_max = search[2]
        if object_frames:
            tracking_object.keyframe_a = min(object_frames)
            tracking_object.keyframe_b = max(object_frames)

def _same_tracking_except_camera(left, right):
    return (
        all(left.get(key) == value for key, value in right.items() if key != "camera")
        and all(right.get(key) == value for key, value in left.items() if key != "camera")
    )


def _native_movie_clips(doc):
    source_path = doc.get("blender_original_blend_path")
    if not source_path or not os.path.isfile(source_path):
        return {}
    source_intermediate = (
        doc.get("compatibility", {}).get("blender_adapter", {}).get(
            "intermediate", {}))
    source_clips = {
        item.get("name"): item.get("tracking", {})
        for item in source_intermediate.get("movie_clips", [])
    }
    requested = [
        item.get("name")
        for item in doc.get("movie_clips", {}).values()
        if item.get("name") in source_clips
        and _same_tracking_except_camera(
            source_clips[item.get("name")], item.get("tracking", {}))
    ]
    with bpy.data.libraries.load(source_path, link=False) as (source, destination):
        available = set(source.movieclips)
        destination.movieclips = [
            name for name in requested if name and name in available]
    return {
        clip.name: clip for clip in destination.movieclips if clip is not None
    }


def _movie_clips_from_model(doc, target):
    result = {}
    native_clips = _native_movie_clips(doc)
    for clip_id, item in doc.get("movie_clips", {}).items():
        source = item.get("source")
        filepath = _resource_path(doc, source)
        if not filepath or not os.path.isfile(filepath):
            resource = _resource_record_for_reference(doc, source)
            raise ValueError(
                "MovieClip source resource is missing: reference=%r path=%r resource=%r"
                % (source, filepath, resource))
        name = item.get("name")
        tracking = item.get("tracking", {})
        clip = native_clips.pop(name, None)
        if clip is None:
            clip = bpy.data.movieclips.load(filepath, check_existing=False)
            _restore_movie_clip_tracking(clip, tracking)
        else:
            clip["potter.tracking_json"] = json.dumps(tracking, sort_keys=True)
        clip.name = name or clip.name
        clip.filepath = os.path.abspath(filepath)
        set_id(clip, clip_id)
        clip.use_fake_user = True
        clip.frame_start = int(item.get("frame_start", 1))
        _movie_clip_camera_values(clip, tracking.get("camera", {}))
        result[clip_id] = clip
    return result




def _cache_file_from_alembic(filepath, identity):
    wanted = os.path.realpath(filepath)
    existing = next((cache for cache in bpy.data.cache_files
                     if (identity is None or cache.name_full == identity)
                     and os.path.realpath(bpy.path.abspath(cache.filepath)) == wanted), None)
    if existing is not None:
        return existing
    collection_names = (
        "objects", "collections", "actions", "armatures", "cameras", "curves",
        "grease_pencils_v3", "lights", "materials", "meshes", "node_groups",
        "volumes", "worlds", "cache_files",
    )
    before = {
        name: {block.as_pointer() for block in getattr(bpy.data, name, ())}
        for name in collection_names
    }
    bpy.ops.wm.alembic_import(
        filepath=filepath, set_frame_range=False, always_add_cache_reader=True,
        as_background_job=False)
    cache_file = next((
        cache for cache in bpy.data.cache_files
        if cache.as_pointer() not in before["cache_files"]
        and os.path.realpath(bpy.path.abspath(cache.filepath)) == wanted
    ), None)
    if cache_file is None:
        raise ValueError("Blender did not create a CacheFile for %s" % filepath)
    if identity is not None:
        cache_file.name = identity
    for obj in list(bpy.data.objects):
        if obj.as_pointer() not in before["objects"]:
            bpy.data.objects.remove(obj, do_unlink=True)
    new_collections = [
        collection for collection in bpy.data.collections
        if collection.as_pointer() not in before["collections"]]
    for collection in new_collections:
        for scene in bpy.data.scenes:
            if collection in scene.collection.children:
                scene.collection.children.unlink(collection)
        for parent in bpy.data.collections:
            if collection in parent.children:
                parent.children.unlink(collection)
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    for name in collection_names:
        if name in ("objects", "collections", "cache_files"):
            continue
        collection = getattr(bpy.data, name, None)
        if collection is None:
            continue
        for block in list(collection):
            if block.as_pointer() not in before[name] and block.users == 0:
                collection.remove(block)
    return cache_file




def _curve_data_create(name, data_id, item, source, doc, target, pack):
    curve_data = item.get("curve") or {}
    resolution_u = curve_data.get("resolution_u", source.get("resolution_u", 12))
    if item.get("type") == "text":
        raw = item.get("text") or {}
        curve = bpy.data.curves.new(name, "FONT")
        curve.body = raw.get("body", "")
        curve.size = float(raw.get("size", 1.0))
        curve.align_x = raw.get("align_x", "left").upper()
        curve.align_y = {
            "baseline": "BASELINE",
            "top": "TOP_BASELINE",
            "center": "CENTER",
            "bottom": "BOTTOM_BASELINE",
        }.get(raw.get("align_y", "baseline"), raw.get("align_y", "baseline").upper())
        curve.extrude = float(raw.get("extrude", 0.0))
        curve.bevel_depth = float(raw.get("bevel_depth", 0.0))
        font_path = raw.get("font", "builtin")
        if font_path not in ("builtin", "", None):
            resolved_font = _resource_path(doc, font_path)
            if not resolved_font or not os.path.isfile(resolved_font):
                raise ValueError("external text font is missing: %s" % font_path)
            font = bpy.data.fonts.load(resolved_font, check_existing=True)
            font.filepath = _relative_blend_path(resolved_font, target)
            if raw.get("font_name"):
                font.name = raw["font_name"]
            if pack and hasattr(font, "pack"):
                font.pack()
            curve.font = font
        for field, attribute in (
            ("character_spacing", "space_character"),
            ("word_spacing", "space_word"),
            ("line_spacing", "space_line"),
            ("shear", "shear"),
            ("offset_x", "offset_x"),
            ("offset_y", "offset_y"),
            ("small_caps_scale", "small_caps_scale"),
        ):
            if hasattr(curve, attribute) and field in raw:
                setattr(curve, attribute, float(raw[field]))
    elif item.get("type") == "surface":
        raw = item.get("surface") or {}
        curve = bpy.data.curves.new(name, "SURFACE")
        rows = raw.get("points", [])
        if rows:
            count_u = len(rows[0])
            if count_u == 0 or any(len(row) != count_u for row in rows):
                raise ValueError("surface control grid is ragged")
            spline = curve.splines.new("NURBS")
            spline.points.add(count_u * len(rows) - 1)
            for v, row in enumerate(rows):
                for u, point in enumerate(row):
                    spline.points[v * count_u + u].co = tuple(point.get("co", [0, 0, 0])) + (
                        float(point.get("weight", 1.0)),)
            spline.points_u = count_u
            spline.points_v = len(rows)
            spline.order_u = int(raw.get("order_u", 3))
            spline.order_v = int(raw.get("order_v", 3))
            spline.resolution_u = int(raw.get("resolution", [12, 12])[0])
            spline.resolution_v = int(raw.get("resolution", [12, 12])[1])
            spline.use_cyclic_u = bool(raw.get("cyclic_u", False))
            spline.use_cyclic_v = bool(raw.get("cyclic_v", False))
            spline.use_endpoint_u = bool(raw.get("use_endpoint_u", True))
            spline.use_endpoint_v = bool(raw.get("use_endpoint_v", True))
    else:
        raw = item.get("curve") or {}
        curve = bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "2D" if raw.get("dimensions") == "two_d" else "3D"
        curve.twist_mode = str(raw.get("twist_mode", "MINIMUM")).upper()
        curve.bevel_depth = float(raw.get("bevel_depth", 0.0))
        curve.bevel_resolution = int(raw.get("bevel_resolution", 0))
        curve.extrude = float(raw.get("extrude", 0.0))
        fill_mode = raw.get("fill_mode", "none").upper()
        curve.fill_mode = {"NONE": "FULL", "BOTH": "FULL"}.get(fill_mode, fill_mode)
        curve.use_path = bool(raw.get("use_path", False))
        curve.path_duration = int(raw.get("path_duration", 100))
        curve.eval_time = float(raw.get("eval_time", 0.0))
        for source_spline in raw.get("splines", []):
            spline_type = source_spline.get("type", "poly").upper()
            spline = curve.splines.new(spline_type)
            points = source_spline.get("points", [])
            if not points:
                curve.splines.remove(spline)
                continue
            collection = spline.bezier_points if spline_type == "BEZIER" else spline.points
            collection.add(len(points) - 1)
            for index, point in enumerate(points):
                if spline_type == "BEZIER":
                    target = collection[index]
                    target.co = point["co"]
                    target.handle_left = point.get("handle_left", point["co"])
                    target.handle_right = point.get("handle_right", point["co"])
                    handle_type = point.get("handle_type", "auto").upper()
                    target.handle_left_type = handle_type
                    target.handle_right_type = handle_type
                else:
                    collection[index].co = tuple(point["co"]) + (float(point.get("weight", 1.0)),)
                    target = collection[index]
                target.radius = float(point.get("radius", 1.0))
                target.tilt = float(point.get("tilt", 0.0))
            spline.use_cyclic_u = bool(source_spline.get("cyclic", False))
            spline.resolution_u = int(source_spline.get("resolution", 12))
            if spline_type == "NURBS":
                spline.order_u = int(source_spline.get("order", 3))
                spline.use_endpoint_u = bool(source_spline.get("use_endpoint", False))
        eval_time_fcurves = raw.get("eval_time_fcurves", [])
        if eval_time_fcurves:
            action_name = source.get("eval_time_action_name") or (
                name + " Path Evaluation")
            action = bpy.data.actions.new(action_name)
            slot_name = source.get("eval_time_action_slot_name") or name
            slot = action.slots.new("CURVE", slot_name)
            animation_data = curve.animation_data_create()
            animation_data.action = action
            animation_data.action_slot = slot
            layer = action.layers.new("Potter")
            strip = layer.strips.new(type="KEYFRAME")
            channelbag = strip.channelbag(slot, ensure=True)
            _restore_action_curves(
                channelbag, eval_time_fcurves, "Curve %s evaluation time" % name, doc)
    curve.resolution_u = int(resolution_u)
    set_id(curve, data_id)
    set_props(curve, source.get("custom_properties", {}))
    if source.get("fake_user"):
        curve.use_fake_user = True
    return curve


def _restore_constraints(obj, sources, id_lookup, doc, target):
    for source in sources or []:
        try:
            constraint = obj.constraints.new(source["type"])
            constraint.name = source.get("name", constraint.name)
            properties = dict(source.get("properties", {}))
            inverse_matrix = properties.pop("_potter_inverse_matrix", None)
            inverse_frame = properties.pop("_potter_inverse_frame", None)
            if source["type"] == "TRANSFORM_CACHE":
                resource_id = properties.pop("resource")
                object_path = properties.pop("object_path")
                cache_file = _cache_file_for_resource(
                    doc, resource_id, target, properties)
                constraint.cache_file = cache_file
                constraint.object_path = object_path
                for key in ("frame_offset", "scale", "override_frame"):
                    properties.pop(key, None)
            if source["type"] == "ARMATURE":
                _restore_armature_targets(
                    constraint, properties, id_lookup,
                    "constraint %s target" % constraint.name)
            set_rna(constraint, properties, id_lookup, strict=True,
                    owner="constraint %s on %s" % (constraint.name, obj.name))
            set_props(constraint, source.get("custom_properties", {}))
            if inverse_matrix is not None:
                _set_object_solver_inverse(
                    obj, constraint, "OBJECT", inverse_frame=inverse_frame)
                _store_object_solver_inverse(
                    obj, constraint.name, inverse_matrix,
                    inverse_frame=inverse_frame)
        except Exception as error:
            raise ValueError("cannot restore constraint %s on %s" %
                             (source.get("name", ""), obj.name)) from error


def _restore_drivers(obj, sources, id_lookup, doc):
    if not sources:
        return
    animation_data = obj.animation_data_create()
    for source in sources:
        curve_source = source.get("curve", {})
        driver_source = source.get("driver", {})
        try:
            curve = obj.driver_add(blender_animation_path(curve_source["path"], doc),
                                   int(curve_source.get("index", 0)))
            curve.extrapolation = curve_source.get("extrapolation", "CONSTANT").upper()
            curve.mute = bool(curve_source.get("mute", False))
            for key in curve_source.get("keyframes", []):
                point = curve.keyframe_points.insert(float(key["frame"]), float(key["value"]))
                point.interpolation = key.get("interpolation", "LINEAR").upper()
                point.handle_left_type = key.get("handle_left_type", "AUTO").upper()
                point.handle_right_type = key.get("handle_right_type", "AUTO").upper()
            driver = curve.driver
            driver.type = driver_source.get("type", "SCRIPTED")
            driver.expression = driver_source.get("expression", "0")
            for variable_source in driver_source.get("variables", []):
                variable = driver.variables.new()
                variable.name = variable_source.get("name", variable.name)
                variable.type = variable_source.get("type", "SINGLE_PROP")
                targets = variable_source.get("targets", [])
                for target, target_source in zip(variable.targets, targets):
                    target.id_type = target_source.get("id_type", "OBJECT")
                    reference = target_source.get("id")
                    target.id = _resolve_compat_value(reference, id_lookup)
                    target.data_path = target_source.get("data_path", "")
                    for key in ("bone_target", "transform_type", "transform_space", "rotation_mode"):
                        if key in target_source:
                            _set_required(target, key, target_source[key],
                                          "driver target on %s" % obj.name)
            for modifier_source in curve_source.get("modifiers", []):
                modifier = curve.modifiers.new(modifier_source["type"])
                set_rna(modifier, modifier_source.get("properties", {}),
                        id_lookup, strict=True, owner="driver FCurve modifier")
            curve.update()
        except Exception as error:
            raise ValueError("cannot restore driver %s on %s" %
                             (curve_source.get("path", ""), obj.name)) from error

def _restore_rigid_body(obj, source, scene):
    if not source:
        return
    if obj.rigid_body is None:
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0],
                                       object=obj, active_object=obj,
                                       selected_objects=[obj], selected_editable_objects=[obj]):
            try:
                bpy.ops.rigidbody.object_add()
            except Exception as error:
                raise ValueError("cannot add rigid body to %s" % obj.name) from error
    body = obj.rigid_body
    for key, attr in (("type", "type"), ("mass", "mass"), ("friction", "friction"),
                      ("restitution", "restitution"), ("shape", "collision_shape"),
                      ("linear_damping", "linear_damping"), ("angular_damping", "angular_damping")):
        if key in source:
            value = source[key]
            if key == "type":
                value = {"active": "ACTIVE", "passive": "PASSIVE"}.get(str(value).lower(), value)
            elif key == "shape":
                value = str(value).upper()
            _set_required(body, attr, value, "rigid body on %s" % obj.name)
    initial_velocity = source.get("initial_velocity")
    if initial_velocity is not None:
        if not hasattr(body, "linear_velocity"):
            raise ValueError("Blender cannot restore rigid-body initial velocity on %s" % obj.name)
        _set_required(body, "linear_velocity", initial_velocity, "rigid body on %s" % obj.name)


def _restore_force_field(obj, source):
    if not source:
        return
    field = getattr(obj, "field", None)
    if field is None:
        raise ValueError("Blender object %s cannot host a force field" % obj.name)
    field_type = str(source["type"]).upper()
    _set_required(field, "type", field_type, "force field on %s" % obj.name)
    _set_required(field, "strength", float(source["strength"]), "force field on %s" % obj.name)
    _set_required(field, "falloff_power", float(source["falloff"]), "force field on %s" % obj.name)


def _restore_rigid_body_world(scene, source):
    if source is None:
        return
    if source.get("seed") is not None:
        scene["potter.rigid_body_seed"] = int(source["seed"])
    if scene.rigidbody_world is None:
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            try:
                bpy.ops.rigidbody.world_add()
            except Exception as error:
                raise ValueError("cannot create rigid-body world for scene %s" % scene.name) from error
    world = scene.rigidbody_world
    world.enabled = bool(source.get("enabled", True))
    scene.gravity = source.get("gravity", (0.0, 0.0, -9.81))
    world.substeps_per_frame = int(source.get("substeps", 10))
    world.solver_iterations = int(source.get("solver_iterations", 10))
    world.point_cache.frame_start = int(source.get("frame_start", scene.frame_start))
    world.point_cache.frame_end = int(source.get("frame_end", scene.frame_end))

def _restore_sequencer(scene, sequencer, scene_map, doc, target):
    strips = sequencer.get("strips", []) if sequencer else []
    if not strips:
        return
    editor = scene.sequence_editor_create()
    sequences = getattr(editor, "strips", None)
    if sequences is None:
        sequences = editor.sequences
    pending = list(strips)
    created = {}
    strip_types = {
        "image": "IMAGE", "image_sequence": "IMAGE", "movie": "MOVIE",
        "sound": "SOUND", "scene": "SCENE", "color": "COLOR", "text": "TEXT",
        "meta": "META",
    }
    transition_types = {"cross": "CROSS", "gamma_cross": "GAMMA_CROSS", "wipe": "WIPE"}
    effect_types = {
        "add": "ADD", "subtract": "SUBTRACT", "multiply": "MULTIPLY",
        "alpha_over": "ALPHA_OVER", "transform": "TRANSFORM", "speed": "SPEED",
        "glow": "GLOW", "gaussian_blur": "GAUSSIAN_BLUR",
    }
    while pending:
        progressed = False
        for source in list(pending):
            inputs = [created.get(identifier) for identifier in source.get("inputs", [])]
            if any(value is None for value in inputs):
                continue
            name = source.get("name", "Strip")
            channel = int(source.get("channel", 1))
            frame_start = int(source.get("frame_start", 1))
            kind = source.get("type", "image")
            frame_end = frame_start + int(source.get("length", 1.0))
            media_path = source.get("source")
            relative_media_path = None
            if kind in ("image", "image_sequence", "movie", "sound"):
                media_path = _resource_path(doc, media_path)
                if not media_path or not os.path.isfile(media_path):
                    raise ValueError("sequencer media source is missing: %s" % source.get("source"))
                relative_media_path = _relative_blend_path(media_path, target)
            try:
                if kind == "transition":
                    native_type = transition_types.get(source.get("transition"))
                    if native_type is None:
                        raise ValueError("unsupported sequencer transition")
                    strip = sequences.new_effect(name, native_type, channel, frame_start,
                                                 frame_end, *(inputs + [None, None])[:2])
                elif kind == "effect":
                    native_type = effect_types.get(source.get("effect"))
                    if native_type is None:
                        raise ValueError("unsupported sequencer effect")
                    strip = sequences.new_effect(name, native_type, channel, frame_start,
                                                 frame_end, *(inputs + [None, None])[:2])
                elif kind == "image":
                    strip = sequences.new_image(name, media_path, channel, frame_start)
                elif kind == "image_sequence":
                    strip = sequences.new_image(name, media_path, channel, frame_start)
                elif kind == "movie":
                    strip = sequences.new_movie(name, media_path, channel, frame_start)
                elif kind == "sound":
                    strip = sequences.new_sound(name, media_path, channel, frame_start)
                elif kind == "scene":
                    target_scene = scene_map.get(source.get("source"))
                    if target_scene is None:
                        raise ValueError("sequencer Scene strip references a missing scene")
                    strip = sequences.new_scene(name, target_scene, channel, frame_start)
                elif kind == "color":
                    strip = sequences.new_effect(name, "COLOR", channel, frame_start, frame_end)
                elif kind == "text":
                    strip = sequences.new_effect(name, "TEXT", channel, frame_start, frame_end)
                elif kind == "meta":
                    strip = sequences.new_meta(name, channel, frame_start, frame_end)
                else:
                    raise ValueError("unsupported sequencer strip type %s" % kind)
                strip.name = name
                if relative_media_path:
                    if kind == "sound" and getattr(strip, "sound", None):
                        strip.sound.filepath = relative_media_path
                    elif hasattr(strip, "filepath"):
                        strip.filepath = relative_media_path
                strip.frame_offset_start = float(source.get("frame_offset_start", 0.0))
                strip.frame_offset_end = float(source.get("frame_offset_end", 0.0))
                strip.blend_type = source.get("blend_type", "replace").upper()
                strip.blend_alpha = float(source.get("opacity", 1.0))
                strip.mute = bool(source.get("mute", False))
                if hasattr(strip, "volume"):
                    strip.volume = float(source.get("sound_volume", 1.0))
                if hasattr(strip, "pan"):
                    strip.pan = float(source.get("sound_pan", 0.0))
                if hasattr(strip, "pitch"):
                    strip.pitch = float(source.get("sound_pitch", 1.0))
                if hasattr(strip, "color"):
                    strip.color = source.get("color", (0.0, 0.0, 0.0, 1.0))
                if hasattr(strip, "text"):
                    strip.text = source.get("text") or ""
                created[source.get("id")] = strip
                pending.remove(source)
                progressed = True
            except Exception as error:
                raise ValueError("cannot restore sequencer strip %s" % name) from error
        if not progressed:
            raise ValueError("sequencer strips contain an unresolved input cycle or reference")


def _canonical_fcurves(model_curves, source_curves, doc, include_unmatched=True):
    sources = {}
    for source in source_curves:
        key = (source.get("path"), int(source.get("index", 0)))
        sources.setdefault(key, []).append(source)
    result = []
    for model in model_curves:
        path = model.get("path", "")
        index = int(model.get("index", 0))
        key = (blender_animation_path(path, doc), index)
        candidates = sources.get(key, [])
        source = candidates.pop(0) if candidates else None
        if source is None and not include_unmatched:
            continue
        curve = dict(source or {})
        curve["path"] = path
        curve["index"] = index
        curve["extrapolation"] = str(model.get("extrapolation", "constant")).upper()
        old_keys = {float(item.get("frame", 0.0)): item
                    for item in (source or {}).get("keyframes", [])}
        keyframes = []
        for item in model.get("keyframes", []):
            frame = float(item["frame"])
            value = float(item["value"])
            key = {"frame": frame, "value": value,
                   "interpolation": str(item.get("interpolation", "linear")).upper()}
            old = old_keys.get(frame)
            if old is not None and float(old.get("value", 0.0)) == value:
                for field in ("handle_left", "handle_right",
                              "handle_left_type", "handle_right_type"):
                    if field in old:
                        key[field] = old[field]
            keyframes.append(key)
        curve["keyframes"] = keyframes
        result.append(curve)
    return result


def _model_action_slot(doc, action_id, model_slot, flat_curves):
    node = doc.get("nodes", {}).get(model_slot.get("node"), {})
    data = doc.get("data_blocks", {}).get(node.get("data"), {})
    shape_keys = data.get("shape_keys") or {}
    is_key_slot = shape_keys.get("action") == action_id
    return {
        "identifier": model_slot.get("id", ""),
        "potter_id": model_slot.get("id"),
        "name": (shape_keys.get("action_slot") or node.get("name", "Action")
                 if is_key_slot else node.get("name", "Action")),
        "target_type": "KEY" if is_key_slot else "OBJECT",
        "fcurves": flat_curves,
    }


def _canonical_action_data(doc, item, source, action_id):
    model_curves = item.get("fcurves", [])
    flat_curves = _canonical_fcurves(
        model_curves, source.get("fcurves", []), doc)
    slot_sources = [dict(slot) for slot in source.get("slots", [])]
    for model_slot in item.get("slots", []):
        descriptor = _model_action_slot(doc, action_id, model_slot, flat_curves)
        existing = next((
            slot for slot in slot_sources
            if slot.get("potter_id") == model_slot.get("id")
            or (slot.get("name") == descriptor["name"]
                and slot.get("target_type", descriptor["target_type"])
                == descriptor["target_type"])
        ), None)
        if existing is None:
            slot_sources.append(descriptor)
        elif not existing.get("potter_id"):
            existing["potter_id"] = model_slot.get("id")
    if not slot_sources and flat_curves:
        slot_sources = [{"identifier": "", "name": item["name"],
                         "target_type": "OBJECT", "fcurves": flat_curves}]
    elif len(slot_sources) == 1:
        slot_sources = [dict(slot_sources[0], fcurves=flat_curves)]
    else:
        slot_sources = [
            dict(slot, fcurves=_canonical_fcurves(
                model_curves, slot.get("fcurves", []), doc, include_unmatched=False))
            for slot in slot_sources
        ]
    return slot_sources, flat_curves


def _restore_action_curves(channelbag, curves, owner, doc):
    for source in curves:
        try:
            path = blender_animation_path(source["path"], doc)
            array_index = blender_animation_index(path, int(source["index"]))
            curve = channelbag.fcurves.new(
                data_path=path,
                index=array_index)
            curve.extrapolation = source.get("extrapolation", "CONSTANT").upper()
            curve.mute = bool(source.get("mute", False))
            for key in source.get("keyframes", []):
                point = curve.keyframe_points.insert(float(key["frame"]), float(key["value"]))
                point.interpolation = key.get("interpolation", "LINEAR").upper()
                point.handle_left_type = key.get("handle_left_type", "AUTO").upper()
                point.handle_right_type = key.get("handle_right_type", "AUTO").upper()
                if point.handle_left_type in ("FREE", "ALIGNED"):
                    point.handle_left = key.get("handle_left", point.handle_left)
                if point.handle_right_type in ("FREE", "ALIGNED"):
                    point.handle_right = key.get("handle_right", point.handle_right)
            for modifier_source in source.get("modifiers", []):
                modifier = curve.modifiers.new(modifier_source["type"])
                set_rna(modifier, modifier_source.get("properties", {}),
                        {}, strict=True, owner="%s FCurve modifier" % owner)
            curve.update()
        except Exception as error:
            raise ValueError("%s cannot restore FCurve %s" %
                             (owner, source.get("path", ""))) from error

def _slot_matches(slots, reference):
    if not reference:
        return None
    for slot in slots:
        if reference.get("identifier") and slot.identifier == reference["identifier"]:
            return slot
    for slot in slots:
        slot_name = getattr(slot, "name_display", getattr(slot, "name", ""))
        if reference.get("name") == slot_name and (
                not reference.get("target_type") or
                reference["target_type"] == getattr(slot, "target_id_type", "")):
            return slot
    return None


def _restore_nla(obj, tracks, action_by_name, action_slots_by_name, id_lookup):
    if not tracks:
        return
    animation_data = obj.animation_data_create()
    for source_track in tracks:
        track = animation_data.nla_tracks.new()
        track.name = source_track.get("name", track.name)
        track.mute = bool(source_track.get("mute", False))
        track.is_solo = bool(source_track.get("solo", False))
        for source_strip in source_track.get("strips", []):
            action = action_by_name.get(source_strip.get("action"))
            if action is None:
                raise ValueError("NLA strip references missing action %s" % source_strip.get("action"))
            strip_properties = source_strip.get("properties", {})
            frame_start = int(strip_properties.get("frame_start", 1))
            strip = track.strips.new(source_strip.get("name", action.name), frame_start, action)
            set_rna(strip, strip_properties, id_lookup, strict=True,
                    owner="NLA strip %s" % strip.name)
            set_props(strip, source_strip.get("custom_properties", {}))
            slot_reference = source_strip.get("action_slot")
            action_slots = action_slots_by_name.get(action.name_full, [])
            slot = _slot_matches(action_slots, slot_reference)
            if slot_reference and slot is None:
                raise ValueError("NLA strip %s references missing Action slot" % strip.name)
            if slot is not None and hasattr(strip, "action_slot"):
                strip.action_slot = slot

def _nla_from_model(doc, node):
    actions = doc.get("actions", {})
    tracks = []
    for source_track in node.get("nla_tracks", []):
        track = {"name": source_track.get("name", "NLA Track"),
                 "mute": source_track.get("mute", False),
                 "solo": source_track.get("solo", False), "strips": []}
        for source_strip in source_track.get("strips", []):
            action = actions.get(source_strip.get("action"))
            if action is None:
                raise ValueError("NLA strip references missing action %s" %
                                 source_strip.get("action"))
            track["strips"].append({
                "name": action.get("name", "Action"),
                "action": action.get("name"),
                "properties": {
                    "frame_start": source_strip.get("frame_start", 1.0),
                    "frame_end": source_strip.get("frame_end", 1.0),
                    "action_frame_start": source_strip.get("action_frame_start", 1.0),
                    "action_frame_end": source_strip.get("action_frame_end", 1.0),
                    "scale": source_strip.get("scale", 1.0),
                    "repeat": source_strip.get("repeat", 1.0),
                    "blend_type": source_strip.get("blend_type", "replace").upper(),
                    "influence": source_strip.get("influence", 1.0),
                    "extrapolation": source_strip.get("extrapolation", "hold").upper(),
                    "blend_in": source_strip.get("blend_in", 0.0),
                    "blend_out": source_strip.get("blend_out", 0.0),
                }})
        tracks.append(track)
    return tracks

def _set_principled_values(material, item):
    shader = next((node for node in material.node_tree.nodes
                   if node.type == "BSDF_PRINCIPLED"), None)
    if shader is None:
        return
    shader.inputs["Base Color"].default_value = item["base_color"]
    shader.inputs["Metallic"].default_value = item["metallic"]
    alpha = shader.inputs.get("Alpha")
    if alpha is not None:
        alpha.default_value = float(item["base_color"][3])
    shader.inputs["Roughness"].default_value = item["roughness"]
    emission = shader.inputs.get("Emission Color")
    if emission is None:
        emission = shader.inputs.get("Emission")
    if emission is not None:
        color = item.get("emission_color", [0.0, 0.0, 0.0])
        emission.default_value = (color[0], color[1], color[2], 1.0)
    emission_power = shader.inputs.get("Emission Strength")
    if emission_power is not None:
        emission_power.default_value = item.get("emission_strength", 0.0)
    transmission_socket = shader.inputs.get("Transmission Weight")
    if transmission_socket is None:
        transmission_socket = shader.inputs.get("Transmission")
    if transmission_socket is not None:
        transmission_socket.default_value = item.get("transmission", 0.0)
    ior_socket = shader.inputs.get("IOR")
    if ior_socket is not None:
        ior_socket.default_value = item.get("ior", 1.45)


def _text_from_string(name, body):
    text = bpy.data.texts.new(name)
    text.from_string(body)
    return text


def export_doc(doc, target, pack, scene_hash, context):
    CACHE_FILES_BY_IDENTITY.clear()
    DEFERRED_ID_ASSIGNMENTS.clear()
    # Factory startup is already clean, but remove all scene objects to make this robust
    # when invoked by Blender builds that ignore the startup flag.
    for scene in bpy.data.scenes:
        scene.camera = None
        scene.world = None
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene_map = {}
    for scene in list(bpy.data.scenes):
        for child in list(scene.collection.children):
            scene.collection.children.unlink(child)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    for collection_name in (
        "actions", "armatures", "cameras", "curves", "grease_pencils_v3",
        "images", "lights", "materials", "meshes", "node_groups", "volumes", "worlds",
    ):
        collection = getattr(bpy.data, collection_name, None)
        if collection is None:
            continue
        for block in list(collection):
            if block.users == 0 and not bool(getattr(block, "use_fake_user", False)):
                collection.remove(block)
    base_scene = bpy.context.scene
    scenes_raw = doc.get("scenes", {})
    collections_raw = doc.get("collections", {})
    if not scenes_raw:
        raise ValueError("scene document has no scenes")
    for index, (scene_id, item) in enumerate(scenes_raw.items()):
        scene = base_scene if index == 0 else bpy.data.scenes.new(item["name"])
        scene.name = item["name"]
        set_id(scene, scene_id)
        scene["potter.scene_id"] = doc["scene_id"]
        scene["potter.revision"] = int(doc.get("revision", 0))
        scene["potter.scene_hash"] = scene_hash
        source = compatibility_block(doc, "scenes", "Scene", scene_id)
        set_props(scene, source.get("custom_properties", {}))
        scene["potter.root_collection"] = collections_raw[item["root_collection"]]["name"]
        set_scene_frame(scene, item["frame_current"])
        scene.frame_start = item["frame_start"]
        scene.frame_end = item["frame_end"]
        for marker in source.get("markers", []):
            scene.timeline_markers.new(marker["name"], frame=int(marker["frame"]))
        scene.render.fps = item["fps"]
        scene.render.fps_base = item["fps_base"]
        render = item.get("render", {})
        scene.render.resolution_x = int(render.get("resolution_x", 1920))
        scene.render.resolution_y = int(render.get("resolution_y", 1080))
        scene.render.resolution_percentage = int(render.get("resolution_percentage", 100))
        scene.render.film_transparent = bool(render.get("film_transparent", False))
        scene.render.use_sequencer = bool(render.get("use_sequencer", False))
        available_engines = {
            item.identifier
            for item in scene.render.bl_rna.properties["engine"].enum_items
        }
        engine = "CYCLES" if render.get("engine", "path") == "path" else next(
            (
                candidate
                for candidate in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT")
                if candidate in available_engines
            ),
            None,
        )
        if engine is None:
            raise ValueError("Blender build has no supported real-time render engine")
        if hasattr(scene.render, "ffmpeg"):
            scene.render.ffmpeg.audio_codec = blender_audio_codec(render.get("audio_codec", "wav"))
        if engine == "CYCLES" and hasattr(scene, "cycles"):
            scene.cycles.max_bounces = int(render.get("max_bounces", 4))
        scene.render.engine = engine
        if engine == "CYCLES" and hasattr(scene, "cycles"):
            scene.cycles.samples = int(render.get("samples", 64))
            scene.cycles.seed = int(render.get("seed", 0))
        elif hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = int(render.get("samples", 64))
        unit = item.get("unit", {})
        try:
            scene.unit_settings.system = unit.get("system", "metric").upper()
            scene.unit_settings.scale_length = unit.get("scale_length", 1.0)
        except Exception:
            pass
        scene_map[scene_id] = scene
    linked_maps = _load_blend_libraries(doc, target)
    collection_map = dict(linked_maps["collections"])
    for scene_id, item in scenes_raw.items():
        root_id = item["root_collection"]
        root_item = collections_raw.get(root_id)
        if root_item is None:
            raise ValueError("scene references an unknown root collection")
        root = scene_map[scene_id].collection
        root["potter.name"] = root_item["name"]
        set_id(root, root_id)
        source = compatibility_block(doc, "collections", "Collection", root_id)
        set_props(root, source.get("custom_properties", {}))
        root.hide_viewport = bool(source.get("hide_viewport", False))
        root.hide_render = bool(source.get("hide_render", False))
        if source.get("fake_user"):
            root.use_fake_user = True
        collection_map[root_id] = root
    for collection_id, item in collections_raw.items():
        if collection_id in collection_map:
            continue
        collection = bpy.data.collections.new(item["name"])
        set_id(collection, collection_id)
        source = compatibility_block(doc, "collections", "Collection", collection_id)
        set_props(collection, source.get("custom_properties", {}))
        collection.hide_viewport = bool(source.get("hide_viewport", False))
        collection.hide_render = bool(source.get("hide_render", False))
        if source.get("fake_user"):
            collection.use_fake_user = True
        collection_map[collection_id] = collection
    for collection_id, item in collections_raw.items():
        parent_col = collection_map[collection_id]
        if parent_col.library is not None:
            continue
        for child_id in item.get("children", []):
            child = collection_map.get(child_id)
            if child and child != parent_col and child.name not in parent_col.children:
                parent_col.children.link(child)
    # Remove the factory-startup Collection after unlinking it from the base scene.
    for collection in list(bpy.data.collections):
        if collection not in collection_map.values() and collection.users == 0:
            bpy.data.collections.remove(collection)
    for scene_id, item in scenes_raw.items():
        scene = scene_map[scene_id]
        view_layer_ids = {}
        for index, (layer_id, layer_item) in enumerate(item.get("view_layers", {}).items()):
            if index == 0:
                view_layer = scene.view_layers[0]
                view_layer.name = layer_item["name"]
            else:
                view_layer = scene.view_layers.new(name=layer_item["name"])
            view_layer_ids[view_layer.name] = layer_id
            excluded_ids = set(layer_item.get("excluded_collections", []))
            def apply_exclusion(layer_collection):
                if layer_collection.collection.get("potter.id") in excluded_ids:
                    layer_collection.exclude = True
                for child in layer_collection.children:
                    apply_exclusion(child)
            apply_exclusion(view_layer.layer_collection)
        scene["potter.view_layer_ids_json"] = json.dumps(view_layer_ids, sort_keys=True)
    for image_id, item in doc.get("images", {}).items():
        image = bpy.data.images.new(item["name"], width=int(item["width"]),
                                    height=int(item["height"]), alpha=True)
        set_id(image, image_id)
        image["potter.image_json"] = json.dumps(item, sort_keys=True)
        image.use_fake_user = True
    data_raw = doc.get("data_blocks", {})
    linked_data_blocks = linked_maps["data_blocks"]
    mesh_map = {identifier: block for identifier, block in linked_data_blocks.items()
                if data_raw[identifier].get("type") == "mesh"}
    armature_map = {identifier: block for identifier, block in linked_data_blocks.items()
                    if data_raw[identifier].get("type") == "armature"}
    grease_pencil_map = {
        identifier: block for identifier, block in linked_data_blocks.items()
        if data_raw[identifier].get("type") == "grease_pencil"}
    grease_pencil_sources = {}
    camera_map = {identifier: block for identifier, block in linked_data_blocks.items()
                  if data_raw[identifier].get("type") == "camera"}
    light_map = {identifier: block for identifier, block in linked_data_blocks.items()
                 if data_raw[identifier].get("type") == "light"}
    curve_map = {identifier: block for identifier, block in linked_data_blocks.items()
                 if data_raw[identifier].get("type") in ("curve", "surface", "text")}
    volume_map = {identifier: block for identifier, block in linked_data_blocks.items()
                  if data_raw[identifier].get("type") == "volume"}
    for data_id, item in data_raw.items():
        if data_id in linked_data_blocks:
            continue
        if item.get("type") == "mesh" and item.get("mesh"):
            mesh_map[data_id] = mesh_create(
                compatibility_name(doc, "Mesh", data_id, data_id), item["mesh"], data_id)
            if item.get("descriptor"):
                mesh_map[data_id]["potter.descriptor_json"] = json.dumps(
                    item["descriptor"], sort_keys=True)
        elif item.get("type") == "mesh" and item.get("descriptor"):
            import bmesh
            bm = bmesh.new()
            kind = item["descriptor"].get("primitive", "box")
            params = item["descriptor"].get("params", {})
            operators = {
                "box": bpy.ops.mesh.primitive_cube_add,
                "plane": bpy.ops.mesh.primitive_plane_add,
                "uv_sphere": bpy.ops.mesh.primitive_uv_sphere_add,
                "sphere": bpy.ops.mesh.primitive_uv_sphere_add,
                "cylinder": bpy.ops.mesh.primitive_cylinder_add,
                "cone": bpy.ops.mesh.primitive_cone_add,
                "torus": bpy.ops.mesh.primitive_torus_add,
                "icosphere": bpy.ops.mesh.primitive_ico_sphere_add,
                "circle": bpy.ops.mesh.primitive_circle_add,
                "grid": bpy.ops.mesh.primitive_grid_add,
            }
            operator = operators.get(kind)
            if operator is None:
                raise ValueError("cannot expand primitive descriptor %s" % kind)
            operator(**params)
            generated_object = bpy.context.object
            bm.from_mesh(generated_object.data)
            bpy.data.objects.remove(generated_object, do_unlink=True)
            mesh = bpy.data.meshes.new(compatibility_name(doc, "Mesh", data_id, data_id))
            bm.to_mesh(mesh)
            bm.free()
            set_id(mesh, data_id)
            mesh["potter.descriptor_json"] = json.dumps(item["descriptor"], sort_keys=True)
            mesh_map[data_id] = mesh
        elif item.get("type") in ("curve", "surface", "text"):
            source = compatibility_block(doc, "curves", "Curve", data_id)
            name = compatibility_name(doc, "Curve", data_id, source.get("name", data_id))
            curve_map[data_id] = _curve_data_create(
                name, data_id, item, source, doc, target, pack)
        elif item.get("type") == "volume":
            source = compatibility_block(doc, "volumes", "Volume", data_id)
            name = compatibility_name(doc, "Volume", data_id, source.get("name", data_id))
            volume = bpy.data.volumes.new(name)
            volume_data = item.get("volume") or {}
            volume_source = volume_data.get("source") or {}
            if volume_source.get("kind") == "file":
                metadata = volume_source.get("metadata") or {}
                reference = metadata.get("content_ref") or source.get("filepath")
                filepath = _resource_path(doc, reference)
                if not filepath:
                    raise ValueError("volume %s has no resolvable source file" % name)
                volume.filepath = _relative_blend_path(filepath, target)
            elif volume_source.get("kind") == "generated" and not volume_data.get("grids"):
                volume.filepath = ""
            else:
                raise ValueError("generated volume data cannot be exported without a mesh-to-volume modifier")
            set_id(volume, data_id)
            set_props(volume, source.get("custom_properties", {}))
            if source.get("fake_user"):
                volume.use_fake_user = True
            volume_map[data_id] = volume
        elif item.get("type") == "armature":
            source = compatibility_block(doc, "armatures", "Armature", data_id)
            armature = bpy.data.armatures.new(
                compatibility_name(doc, "Armature", data_id, source.get("name", data_id)))
            set_id(armature, data_id)
            set_props(armature, source.get("custom_properties", {}))
            if source.get("fake_user"):
                armature.use_fake_user = True
            armature_map[data_id] = armature
        elif item.get("type") in ("grease_pencil", "grease-pencil", "grease_pencils_v3"):
            source = compatibility_block(doc, "grease_pencil", "GreasePencil", data_id)
            collection = getattr(bpy.data, "grease_pencils_v3", None)
            if collection is None:
                raise ValueError("Blender does not support Grease Pencil v3 data")
            grease = collection.new(compatibility_name(doc, "GreasePencil", data_id,
                                                        source.get("name", data_id)))
            set_id(grease, data_id)
            set_props(grease, source.get("custom_properties", {}))
            if source.get("fake_user"):
                grease.use_fake_user = True
            grease_pencil_map[data_id] = grease
            grease_pencil_sources[data_id] = source
        elif item.get("type") == "camera":
            raw = item.get("camera") or {}
            source = compatibility_block(doc, "cameras", "Camera", data_id)
            camera = bpy.data.cameras.new(compatibility_name(doc, "Camera", data_id, data_id))
            camera.lens = raw.get("lens_mm", 50.0)
            camera.sensor_width = raw.get("sensor_width_mm", 36.0)
            camera.sensor_height = raw.get("sensor_height_mm", 24.0)
            camera.sensor_fit = raw.get("sensor_fit", "AUTO")
            camera.ortho_scale = raw.get("ortho_scale", 6.0)
            camera.clip_start = raw.get("clip_start", 0.1)
            camera.clip_end = raw.get("clip_end", 1000.0)
            camera.shift_x, camera.shift_y = raw.get("shift", [0.0, 0.0])
            if raw.get("projection") == "orthographic":
                camera.type = "ORTHO"
            set_id(camera, data_id)
            set_props(camera, source.get("custom_properties", {}))
            if source.get("fake_user"):
                camera.use_fake_user = True
            camera_map[data_id] = camera
        elif item.get("type") == "light":
            raw = item.get("light") or {}
            source = compatibility_block(doc, "lights", "Light", data_id)
            light = bpy.data.lights.new(compatibility_name(doc, "Light", data_id, data_id),
                                        raw.get("light_type", "point").upper())
            light.color = raw.get("color", [1.0, 1.0, 1.0])
            light.energy = raw.get("energy", 1000.0)
            light.shadow_soft_size = raw.get("radius", 0.1)
            if raw.get("light_type") == "area":
                light.shape = raw.get("area_shape", "SQUARE")
                light.size = float(raw.get("area_size", 0.25))
                light.size_y = float(raw.get("area_size_y", light.size))
            if raw.get("light_type") == "spot":
                light.spot_size = raw.get("spot_size", math.pi / 4)
                light.spot_blend = raw.get("spot_blend", 0.15)
            set_id(light, data_id)
            set_props(light, source.get("custom_properties", {}))
            if source.get("fake_user"):
                light.use_fake_user = True
            light_map[data_id] = light
    world_map = dict(linked_maps["worlds"])
    for world_id, item in doc.get("worlds", {}).items():
        if world_id in world_map:
            continue
        world = bpy.data.worlds.new(compatibility_name(doc, "World", world_id, world_id))
        world.color = item["color"]
        world.use_nodes = True
        source = compatibility_block(doc, "worlds", "World", world_id)
        set_id(world.node_tree, item.get("node_tree"))
        background = next((node for node in world.node_tree.nodes if node.type == "BACKGROUND"), None)
        if background:
            color = item.get("background_color") or item["color"]
            background.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
            background.inputs["Strength"].default_value = item["strength"]
        set_id(world, world_id)
        set_props(world, source.get("custom_properties", {}))
        if source.get("fake_user"):
            world.use_fake_user = True
        world_map[world_id] = world
    material_map = dict(linked_maps["materials"])
    for material_id, item in doc.get("materials", {}).items():
        if material_id in material_map:
            continue
        mat = bpy.data.materials.new(item["name"])
        mat.diffuse_color = item["base_color"]
        mat.use_nodes = True
        set_id(mat.node_tree, item.get("node_tree"))
        _set_principled_values(mat, item)
        mat.use_backface_culling = not bool(item.get("double_sided", False))
        set_id(mat, material_id)
        set_props(mat, source.get("custom_properties", {}))
        if source.get("fake_user"):
            mat.use_fake_user = True
        material_map[material_id] = mat
    materials_by_name = {material.name_full: material for material in material_map.values()}
    for data_id, source in grease_pencil_sources.items():
        _restore_grease_pencil(grease_pencil_map[data_id], source, materials_by_name)
    group_map, group_sources = _create_geometry_groups(
        doc, linked_maps["node_groups"])
    shape_key_meshes = set()
    native_bind_copies, preserved_native_binds = _source_modifier_object_copies(
        doc, mesh_map)
    object_map = dict(linked_maps["nodes"])
    for node_id, item in doc.get("nodes", {}).items():
        if node_id in linked_maps["nodes"]:
            continue
        source_object = compatibility_block(doc, "objects", "Object", node_id)
        kind = item.get("kind", "empty")
        native_type = source_object.get("type")
        obj_data = None
        data_id = item.get("data")
        if kind == "mesh" or native_type == "MESH":
            obj_data = mesh_map.get(data_id)
        elif kind == "camera" or native_type == "CAMERA":
            obj_data = camera_map.get(data_id)
        elif kind == "light" or native_type == "LIGHT":
            obj_data = light_map.get(data_id)
        elif kind == "armature" or native_type == "ARMATURE":
            obj_data = armature_map.get(data_id)
            if obj_data is None:
                raise ValueError("armature object references missing armature data %s" % data_id)
        elif kind in ("grease_pencil", "grease-pencil") or native_type == "GREASEPENCIL":
            obj_data = grease_pencil_map.get(data_id)
            if obj_data is None:
                raise ValueError("Grease Pencil object references missing data %s" % data_id)
        elif kind in ("curve", "surface", "text") or native_type in ("CURVE", "SURFACE", "FONT"):
            obj_data = curve_map.get(data_id)
            if obj_data is None:
                raise ValueError("curve object references missing curve data %s" % data_id)
        elif kind == "volume" or native_type == "VOLUME":
            obj_data = volume_map.get(data_id)
            if obj_data is None:
                raise ValueError("volume object references missing volume data %s" % data_id)
        override = item.get("properties", {}).get("library_override")
        if override:
            reference_id = override.get("reference_id")
            reference = object_map.get(reference_id)
            if reference is None:
                raise ValueError("library override references missing linked object %s"
                                 % reference_id)
            override_scene = scene_map.get(doc.get("active_scene"), bpy.context.scene)
            obj = _create_object_override(reference, override_scene)
            obj.name = item["name"]
        elif node_id in native_bind_copies:
            obj = native_bind_copies[node_id]
            obj.data = obj_data
        else:
            obj = bpy.data.objects.new(item["name"], obj_data)
        instance_name = source_object.get("instance_collection")
        instance_id = item.get("properties", {}).get("instance_collection")
        instance = collection_map.get(instance_id) if instance_id else None
        if instance is None and instance_name:
            instance = next((candidate for candidate in collection_map.values()
                             if candidate.name_full == instance_name), None)
        if instance_id or instance_name:
            if instance is None:
                raise ValueError("collection instance references missing collection %s"
                                 % (instance_id or instance_name))
            obj.instance_type = "COLLECTION"
            obj.instance_collection = instance
        set_id(obj, node_id)
        object_map[node_id] = obj
        try:
            transform = item["transform"]
            rotation_mode = transform.get("rotation_mode", "XYZ")
            obj.rotation_mode = rotation_mode.upper()
            obj.location = transform["translation"]
            x, y, z, w = transform["rotation"]
            quaternion = Quaternion((w, x, y, z))
            source_rotation_mode = source_object.get("rotation_mode", "XYZ").upper()
            source_rotation_values = source_object.get("rotation_quaternion")
            source_euler_values = source_object.get("rotation_euler")
            source_axis_angle = source_object.get("rotation_axis_angle")
            if source_rotation_mode == "QUATERNION":
                source_quaternion = (
                    Quaternion(source_rotation_values)
                    if isinstance(source_rotation_values, (list, tuple))
                    and len(source_rotation_values) == 4
                    else None
                )
            elif source_rotation_mode == "AXIS_ANGLE":
                source_quaternion = (
                    Quaternion(
                        source_axis_angle[1:4], source_axis_angle[0])
                    if isinstance(source_axis_angle, (list, tuple))
                    and len(source_axis_angle) == 4
                    else None
                )
            else:
                source_quaternion = (
                    Euler(source_euler_values, source_rotation_mode).to_quaternion()
                    if isinstance(source_euler_values, (list, tuple))
                    and len(source_euler_values) == 3
                    else None
                )
            source_rotation_unchanged = (
                source_quaternion is not None
                and source_rotation_mode == obj.rotation_mode
                and abs(abs(source_quaternion.dot(quaternion)) - 1.0) <= 1.0e-14
            )
            if source_rotation_unchanged and obj.rotation_mode == "QUATERNION":
                obj.rotation_quaternion = source_rotation_values
            elif source_rotation_unchanged and obj.rotation_mode == "AXIS_ANGLE":
                source_axis_angle = source_object.get("rotation_axis_angle")
                if isinstance(source_axis_angle, (list, tuple)) and len(source_axis_angle) == 4:
                    obj.rotation_axis_angle = source_axis_angle
                else:
                    angle, axis = quaternion.to_axis_angle()
                    obj.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
            elif source_rotation_unchanged:
                source_euler = source_object.get("rotation_euler")
                if isinstance(source_euler, (list, tuple)) and len(source_euler) == 3:
                    obj.rotation_euler = source_euler
                else:
                    obj.rotation_euler = quaternion.to_euler(obj.rotation_mode)
            elif obj.rotation_mode == "QUATERNION":
                obj.rotation_quaternion = quaternion
            elif obj.rotation_mode == "AXIS_ANGLE":
                angle, axis = quaternion.to_axis_angle()
                obj.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
            else:
                obj.rotation_euler = quaternion.to_euler(obj.rotation_mode)
            obj.scale = transform["scale"]
        except Exception:
            pass
        metadata = item.get("properties", {}).get("blender_object_metadata") or {}
        delta_location = metadata.get("delta_location")
        if isinstance(delta_location, (list, tuple)) and len(delta_location) == 3:
            obj.delta_location = [float(value) for value in delta_location]
        delta_scale = metadata.get("delta_scale")
        if isinstance(delta_scale, (list, tuple)) and len(delta_scale) == 3:
            obj.delta_scale = [float(value) for value in delta_scale]
        delta_rotation_euler = metadata.get("delta_rotation_euler")
        if isinstance(delta_rotation_euler, (list, tuple)) and len(delta_rotation_euler) == 3:
            obj.delta_rotation_euler = [float(value) for value in delta_rotation_euler]
        delta_rotation_quaternion = metadata.get("delta_rotation_quaternion")
        if (isinstance(delta_rotation_quaternion, (list, tuple))
                and len(delta_rotation_quaternion) == 4):
            obj.delta_rotation_quaternion = [float(value) for value in delta_rotation_quaternion]
        obj.hide_viewport = not item.get("visible", True)
        obj.hide_render = not item.get("render_visible", True)
        obj.hide_select = not item.get("selectable", True)
        set_props(obj, item.get("properties", {}))
        visibility_rna = {
            key: value for key, value in (metadata.get("rna_properties") or {}).items()
            if key.startswith("visible_")
            or (key.startswith("hide_")
                and key not in ("hide_viewport", "hide_render", "hide_select"))
        }
        set_rna(obj, visibility_rna)
        if item.get("tags"):
            obj["potter.tags"] = item["tags"]
        if item.get("modifiers"):
            obj["potter.modifier_ids_json"] = json.dumps(
                {modifier["name"]: modifier["id"] for modifier in item["modifiers"]},
                sort_keys=True,
            )
        for material_id in item.get("materials", []):
            material = material_map.get(material_id)
            if material and obj.type in ("MESH", "CURVE", "SURFACE", "FONT", "GREASEPENCIL"):
                if obj.data.materials.get(material.name_full) is None:
                    obj.data.materials.append(material)
        if obj.type == "MESH":
            groups = _vertex_groups_from_model(doc, data_id)
            if not groups:
                groups = source_object.get("vertex_groups", [])
            _restore_vertex_groups(obj, groups)
            mesh_source = compatibility_block(doc, "meshes", "Mesh", data_id)
            if data_id not in shape_key_meshes:
                model_shape_keys = _shape_keys_from_model(doc, data_id)
                if model_shape_keys is not None:
                    mesh_source = dict(mesh_source)
                    mesh_source["shape_keys"] = model_shape_keys
                elif not mesh_source.get("shape_keys"):
                    mesh_source = dict(mesh_source)
                    mesh_source["shape_keys"] = {}
                _restore_shape_keys(obj, mesh_source)
                shape_key_meshes.add(data_id)
    object_lookup = {obj.name_full: obj for obj in object_map.values()}
    id_lookup = {}
    for collection in (mesh_map, camera_map, light_map, armature_map, grease_pencil_map,
                       world_map, material_map, collection_map):
        id_lookup.update({block.name_full: block for block in collection.values()})
    # Objects override same-named data blocks: compatibility references with
    # id_type "Object" must resolve to the Object, not identically-named data.
    id_lookup.update(object_lookup)
    id_lookup.update(object_map)
    id_lookup.update(collection_map)
    movie_clip_map = _movie_clips_from_model(doc, target)
    for clip_id, clip in movie_clip_map.items():
        id_lookup[clip_id] = clip
        id_lookup[clip.name_full] = clip
    for scene_id, item in scenes_raw.items():
        clip_id = item.get("active_clip")
        if clip_id is not None:
            clip = movie_clip_map.get(clip_id)
            if clip is None:
                raise ValueError("scene references missing MovieClip %s" % clip_id)
            scene_map[scene_id].active_clip = clip
    id_lookup.update({tree.name_full: tree for tree in group_map.values()})
    id_lookup.update({image.name_full: image for image in linked_maps["resources"].values()})
    socket_maps = _populate_geometry_groups(group_map, group_sources, id_lookup)
    for material_id, item in doc.get("materials", {}).items():
        material = material_map.get(material_id)
        if (material is None or material.node_tree is None
                or material.library is not None):
            continue
        source = compatibility_block(doc, "materials", "Material", material_id)
        graph_id = item.get("node_tree")
        graph = doc.get("node_groups", {}).get(graph_id)
        node_group_source = group_sources.get(graph_id, {})
        tree_source = _model_group_tree(graph) if graph else None
        if tree_source is None:
            tree_source = node_group_source.get("tree")
        if tree_source is None:
            tree_source = source.get("nodes")
        if tree_source:
            _restore_node_tree(material.node_tree, tree_source, id_lookup,
                               "Material shader tree %s" % material.name)
            if (_is_simple_principled_tree(tree_source)
                    or any(node.get("properties", {}).get("potter_simple_material") is True
                           for node in tree_source.get("nodes", []))):
                _set_principled_values(material, item)
    for world_id, item in doc.get("worlds", {}).items():
        world = world_map.get(world_id)
        if world is None or world.node_tree is None:
            continue
        if world.library is not None:
            continue
        source = compatibility_block(doc, "worlds", "World", world_id)
        graph_id = item.get("node_tree")
        graph = doc.get("node_groups", {}).get(graph_id)
        tree_source = _model_group_tree(graph) if graph else None
        if tree_source is None:
            tree_source = source.get("nodes")
        if tree_source:
            _restore_node_tree(world.node_tree, tree_source, id_lookup,
                               "World shader tree %s" % world.name)
    for scene_id, item in scenes_raw.items():
        scene = scene_map[scene_id]
        compositor_id = item.get("compositor")
        tree = group_map.get(compositor_id)
        if tree is None:
            source_scene = compatibility_block(doc, "scenes", "Scene", scene_id)
            compositor_name = (source_scene.get("compositor") or {}).get("name")
            tree = group_map.get(compositor_name)
        if tree is not None and hasattr(scene, "compositing_node_group"):
            scene.compositing_node_group = tree
        render = item.get("render", {})
        scene.render.use_compositing = bool(item.get("use_compositing", False))
        scene.render.use_sequencer = bool(render.get("use_sequencer", False))
        audio_codec = getattr(getattr(scene.render, "ffmpeg", None), "audio_codec", None)
        if audio_codec is not None:
            scene.render.ffmpeg.audio_codec = blender_audio_codec(render.get("audio_codec", "wav"))
        color = item.get("color_management", {})
        scene.display_settings.display_device = color.get("display_device", "sRGB")
        view = scene.view_settings
        transform = color.get("view_transform", "standard")
        view.view_transform = {"ag_x": "AgX", "false_color": "False Color"}.get(
            transform, transform.title())
        view.look = blender_color_look(color.get("look", "none"))
        view.exposure = float(color.get("exposure", 0.0))
        view.gamma = float(color.get("gamma", 1.0))
        _restore_sequencer(scene, item.get("sequencer"), scene_map, doc, target)
    for node_id, item in doc.get("nodes", {}).items():
        if (node_id in linked_maps["nodes"]
                or item.get("properties", {}).get("library_override")):
            continue
        obj = object_map[node_id]
        source_object = compatibility_block(doc, "objects", "Object", node_id)
        source_modifiers = {modifier.get("name"): modifier
                            for modifier in source_object.get("modifiers", [])}
        for modifier_item in item.get("modifiers", []):
            try:
                params = modifier_item.get("params", {})
                modifier_type = blender_modifier_type(modifier_item["type"])
                modifier = obj.modifiers.get(modifier_item["name"])
                if modifier is not None and modifier.type != modifier_type:
                    obj.modifiers.remove(modifier)
                    modifier = None
                if modifier is None:
                    modifier = _create_modifier(
                        obj, modifier_item["name"], modifier_type)
                modifier.show_viewport = modifier_item.get("enabled", True)
                modifier.show_render = modifier_item.get("enabled", True)
                set_modifier_params(
                    modifier, modifier_item, id_lookup, doc, target, node_id, obj)
                if modifier_item.get("type") == "mesh_sequence_cache":
                    cache_file = _cache_file_for_resource(
                        doc, params["resource"], target, params)
                    modifier.cache_file = cache_file
                    modifier.object_path = params["object_path"]
                source_modifier = source_modifiers.get(modifier_item.get("name"), {})
                source_properties = source_modifier.get("properties", {})
                if modifier.type == "NODES":
                    model_params = modifier_item.get("params", {})
                    group_name = model_params.get("node_group")
                    if group_name is None:
                        group_name = source_properties.get("node_group")
                    if group_name is not None:
                        group = group_map.get(group_name)
                        if group is None:
                            group_name = compatibility_name(
                                doc, "NodeGroup", group_name, group_name)
                            group = group_map.get(group_name)
                        if group is None:
                            raise ValueError("Geometry Nodes modifier references missing group %s" % group_name)
                        modifier.node_group = group
                    inputs = model_params.get("inputs") or source_properties.get("inputs", {})
                    socket_map = socket_maps.get(group_name, {}) if group_name else {}
                    for socket_id, value in inputs.items():
                        target_id = socket_map.get(socket_id, socket_id)
                        try:
                            modifier[target_id] = _resolve_compat_value(value, id_lookup)
                        except Exception as error:
                            raise ValueError("cannot restore Geometry Nodes modifier input %s" % socket_id) from error
                set_props(modifier, source_modifier.get("custom_properties", {}))
            except Exception as error:
                raise RuntimeError(
                    "cannot reconstruct modifier %s (%s)" % (modifier_item.get("name"), modifier_item.get("type"))
                ) from error
        desired_modifier_names = {
            modifier.get("name") for modifier in item.get("modifiers", [])
        }
        for modifier in list(obj.modifiers):
            if modifier.name not in desired_modifier_names:
                obj.modifiers.remove(modifier)
    for node_id, item in doc.get("nodes", {}).items():
        obj = object_map[node_id]
        is_linked = node_id in linked_maps["nodes"]
        root_parent_inverse = None
        if not is_linked:
            parent = object_map.get(item.get("parent"))
            if parent:
                obj.parent = parent
                if item.get("parent_type") == "bone":
                    bone_name = compatibility_block(
                        doc, "objects", "Object", node_id).get("parent_bone")
                    if bone_name is None:
                        parent_node = doc.get("nodes", {}).get(item.get("parent"), {})
                        armature_id = parent_node.get("data")
                        armature = doc.get("data_blocks", {}).get(
                            armature_id, {}).get("armature", {})
                        bone_id = item.get("parent_bone")
                        bone_name = armature.get("bones", {}).get(bone_id, {}).get("name")
                    if not bone_name:
                        raise ValueError("bone-parented object %s has no parent bone" % obj.name)
                    obj.parent_type = "BONE"
                    obj.parent_bone = bone_name
                if item.get("parent_inverse"):
                    values = item["parent_inverse"]
                    rows = [[values[column * 4 + row] for column in range(4)]
                            for row in range(4)]
                    obj.matrix_parent_inverse = Matrix(rows)
            elif item.get("parent_inverse"):
                values = item["parent_inverse"]
                rows = [[values[column * 4 + row] for column in range(4)]
                        for row in range(4)]
                root_parent_inverse = Matrix(rows)
        memberships = 0
        for collection_id, collection_item in collections_raw.items():
            if node_id not in collection_item.get("objects", []):
                continue
            collection = collection_map[collection_id]
            if collection.library is not None:
                memberships += 1
                continue
            if obj.name not in collection.objects:
                collection.objects.link(obj)
            memberships += 1
        if memberships == 0 and not is_linked:
            active_scene = scenes_raw.get(doc.get("active_scene")) or next(iter(scenes_raw.values()))
            collection = collection_map[active_scene["root_collection"]]
            if obj.name not in collection.objects:
                collection.objects.link(obj)
        if root_parent_inverse is not None and root_parent_inverse != Matrix.Identity(4):
            combined = root_parent_inverse @ obj.matrix_basis.copy()
            # Without a parent Blender ignores matrix_parent_inverse. Bake its
            # offset into the linked root's basis and world matrix instead.
            obj.matrix_basis = combined
            obj.matrix_world = combined
            obj["potter.root_matrix_json"] = json.dumps({
                "parent_inverse": values,
                "transform": item["transform"],
                "baked_basis": matrix_rows(obj.matrix_basis),
            }, sort_keys=True)
        if obj.type == "MESH":
            mesh_source = compatibility_block(doc, "meshes", "Mesh", item.get("data"))
            _restore_skin_vertices(obj, mesh_source.get("skin_vertices", []))
    _bind_blender_modifiers(doc, object_map, preserved_native_binds)
    for scene_id, scene_item in scenes_raw.items():
        source_scene = compatibility_block(doc, "scenes", "Scene", scene_id)
        rigid_world = scene_item.get("rigid_body_world", source_scene.get("rigid_body_world"))
        _restore_rigid_body_world(scene_map[scene_id], rigid_world)
    for node_id, item in doc.get("nodes", {}).items():
        if (node_id in linked_maps["nodes"]
                or item.get("properties", {}).get("library_override")):
            continue
        obj = object_map[node_id]
        source_object = compatibility_block(doc, "objects", "Object", node_id)
        data_id = item.get("data")
        scene = next((candidate for candidate in scene_map.values()
                      if obj.name_full in candidate.objects), bpy.context.scene)
        if obj.type == "ARMATURE":
            armature_source = compatibility_block(doc, "armatures", "Armature", data_id)
            _restore_armature(obj, _armature_from_model(doc, data_id, armature_source), scene)
            _restore_pose(obj, _canonical_pose(doc, item, source_object.get("pose")),
                          id_lookup, doc, target)
        _restore_constraints(
            obj, _canonical_constraints(doc, item, source_object.get("constraints")),
            id_lookup, doc, target)
        _restore_drivers(
            obj, _canonical_drivers(doc, item, source_object.get("drivers")), id_lookup, doc)
        rigid_body = source_object.get("rigid_body", item.get("rigid_body"))
        force_field = source_object.get("force_field", item.get("force_field"))
        _restore_rigid_body(obj, rigid_body, scene)
        _restore_force_field(obj, force_field)
    action_map = {}
    action_by_name = {}
    action_slots_by_name = {}
    for action_id, action in linked_maps["actions"].items():
        slot_ids = _id_map(action, "potter.action_slot_ids_json")
        slots = list(action.slots)
        slot_sources = [
            {"identifier": getattr(slot, "identifier", ""),
             "name": getattr(slot, "name_display", getattr(slot, "name", "")),
             "potter_id": slot_ids.get(getattr(slot, "identifier", ""))}
            for slot in slots]
        action_map[action_id] = (action, slots, slot_sources)
        action_by_name[action.name_full] = action
        action_slots_by_name[action.name_full] = slots
        id_lookup[action.name_full] = action
        id_lookup[action_id] = action
    for action_id, item in doc.get("actions", {}).items():
        if action_id in linked_maps["actions"]:
            continue
        action = bpy.data.actions.new(item["name"])
        set_id(action, action_id)
        source = compatibility_block(doc, "actions", "Action", action_id)
        set_props(action, source.get("custom_properties", {}))
        if source.get("fake_user"):
            action.use_fake_user = True
        slot_sources, flat_curves = _canonical_action_data(
            doc, item, source, action_id)
        slot_ids = {}
        slots = []
        source_slot_refs = []
        for slot_source in slot_sources:
            target_type = slot_source.get("target_type") or "OBJECT"
            slot_name = slot_source.get("name") or item["name"]
            try:
                slot = action.slots.new(target_type, slot_name)
            except Exception as error:
                raise ValueError("cannot create Action slot %s (%s)" %
                                 (slot_name, target_type)) from error
            slots.append(slot)
            source_slot_refs.append(slot_source)
            slot_id = slot_source.get("potter_id")
            if slot_id:
                slot_ids[getattr(slot, "identifier", "")] = slot_id
        if slot_ids:
            action["potter.action_slot_ids_json"] = json.dumps(slot_ids, sort_keys=True)
        action_slots_by_name[action.name_full] = slots
        if slots:
            layer = action.layers.new("Potter")
            strip = layer.strips.new(type="KEYFRAME")
            for slot, slot_source in zip(slots, source_slot_refs):
                curves = slot_source.get("fcurves", [])
                if not curves and len(slots) == 1:
                    curves = flat_curves
                if not curves:
                    continue
                channelbag = strip.channelbag(slot, ensure=True)
                _restore_action_curves(channelbag, curves, "Action %s slot %s" %
                                       (action.name, slot.name_display), doc)
        action_map[action_id] = (action, slots, source_slot_refs)
        action_by_name[action.name_full] = action
        id_lookup[action.name_full] = action
        id_lookup[action_id] = action
    _resolve_deferred_id_assignments(id_lookup)
    for data_id, data in doc.get("data_blocks", {}).items():
        shape = data.get("shape_keys") or {}
        action_id = shape.get("action")
        if not action_id or data_id in linked_data_blocks:
            continue
        mesh = mesh_map.get(data_id)
        action_data = action_map.get(action_id)
        if mesh is None or mesh.shape_keys is None or action_data is None:
            raise ValueError("shape-key animation references missing mesh or action")
        action, slots, _ = action_data
        animation_data = mesh.shape_keys.animation_data_create()
        animation_data.action = action
        slot_name = shape.get("action_slot")
        slot = _slot_matches(slots, {"name": slot_name, "target_type": "KEY"}) if slot_name else None
        if slot is None and not slot_name:
            slot = next((slot for slot in slots
                         if getattr(slot, "target_id_type", "") == "KEY"), None)
        if slot is None and slots:
            raise ValueError("shape-key animation references missing Key Action slot")
        if slot is not None:
            animation_data.action_slot = slot
    for node_id, node in doc.get("nodes", {}).items():
        if (node_id in linked_maps["nodes"]
                or node.get("properties", {}).get("library_override")):
            continue
        obj = object_map.get(node_id)
        if obj is None:
            continue
        source_object = compatibility_block(doc, "objects", "Object", node_id)
        action_id = node.get("action")
        action_data = action_map.get(action_id)
        action_name = source_object.get("action")
        if action_data is None and action_name:
            action = action_by_name.get(action_name)
            if action is not None:
                action_data = next((entry for entry in action_map.values() if entry[0] == action), None)
        if action_data:
            action, slots, source_slot_refs = action_data
            obj.animation_data_create()
            obj.animation_data.action = action
            reference = node.get("action_slot") or source_object.get("action_slot")
            slot = None
            if reference:
                slot = _slot_matches(slots, reference)
                if slot is None:
                    raise ValueError("object %s references missing Action slot" % obj.name)
            elif slots:
                slot = slots[0]
            if slot is not None:
                obj.animation_data.action_slot = slot
        tracks = _canonical_nla(doc, node, source_object.get("nla_tracks"))
        _restore_nla(obj, tracks, action_by_name, action_slots_by_name, id_lookup)
        hide_set_states = source_object.get("hide_set_by_view_layer") or {}
        restored_hide_set = False
        for scene in scene_map.values():
            scene_states = hide_set_states.get(scene.name_full, {})
            for view_layer in scene.view_layers:
                if obj.name_full not in view_layer.objects:
                    continue
                if view_layer.name in scene_states:
                    obj.hide_set(bool(scene_states[view_layer.name]), view_layer=view_layer)
                    restored_hide_set = True
        if not restored_hide_set and "hide_set" in source_object:
            scene = next((candidate for candidate in scene_map.values()
                          if obj.name_full in candidate.objects), None)
            if (scene and scene.view_layers
                    and obj.name_full in scene.view_layers[0].objects):
                obj.hide_set(bool(source_object["hide_set"]),
                             view_layer=scene.view_layers[0])
    for scene_id, item in scenes_raw.items():
        scene = scene_map[scene_id]
        if item.get("camera") in object_map:
            scene.camera = object_map[item["camera"]]
        if item.get("world") in world_map:
            scene.world = world_map[item["world"]]
    context_scene_id = context.get("scene_id") or doc.get("active_scene")
    active_scene = scene_map.get(context_scene_id)
    if active_scene:
        if context.get("frame") is not None:
            set_scene_frame(active_scene, context["frame"])
        if bpy.context.window:
            bpy.context.window.scene = active_scene
            layers = scenes_raw[context_scene_id].get("view_layers", {})
            selected_layer = layers.get(context.get("view_layer"))
            if selected_layer:
                bpy.context.window.view_layer = next(
                    (layer for layer in active_scene.view_layers if layer.name == selected_layer["name"]),
                    active_scene.view_layers[0])
    compatibility = doc.get("compatibility", {})
    if compatibility:
        _text_from_string("potter_compatibility.json",
                          json.dumps(compatibility, ensure_ascii=False, sort_keys=True))
    adapter = compatibility.get("blender_adapter", {})
    intermediate = adapter.get("intermediate", {})
    mappings = adapter.get("id_mappings", {})
    for item in intermediate.get("texts", []):
        text = _text_from_string(item["name"], item.get("body", ""))
        set_id(text, item.get("potter_id") or mappings.get("Text:" + item["name"]))
        set_props(text, item.get("custom_properties", {}))
        if item.get("fake_user"):
            text.use_fake_user = True
    for item in doc.get("compatibility", {}).get("blender_texts", []):
        text = _text_from_string(item.get("name", "Imported Text"), item.get("body", ""))
        set_id(text, item.get("potter_id"))
        set_props(text, item.get("custom_properties", {}))
    output_dir = os.path.dirname(os.path.abspath(target))
    os.makedirs(output_dir, exist_ok=True)
    if pack:
        bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=target, check_existing=False,
                                relative_remap=False)
    if bpy.data.movieclips:
        for clip in bpy.data.movieclips:
            if clip.filepath:
                clip.filepath = _relative_blend_path(
                    bpy.path.abspath(clip.filepath), target)
        bpy.ops.wm.save_as_mainfile(filepath=target, check_existing=False,
                                    relative_remap=False)


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) != 3:
        raise ValueError("expected mode, input path and output path")
    mode, input_path, output_path = args
    if mode == "import":
        bpy.ops.wm.open_mainfile(filepath=input_path, load_ui=False, use_scripts=False)
        result = dump_all()
        with open(output_path, "w", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    elif mode == "export":
        with open(input_path, "r", encoding="utf-8") as source:
            payload = json.load(source)
        export_doc(payload["doc"], output_path, bool(payload.get("pack", False)),
                   payload["scene_hash"], payload.get("context", {}))
    else:
        raise ValueError("unknown bridge mode: " + mode)


try:
    main()
except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.exit(3)
