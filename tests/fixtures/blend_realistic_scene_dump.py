import bpy
import json
import struct
import sys
from pathlib import Path
from mathutils import Color, Vector

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = Path(args[-1]) if args else Path(__file__).resolve().parent
DUMP = OUT / ('roundtrip_dump.json' if '--reopen' in args else 'original_dump.json')
scene = bpy.context.scene


def rna_value(value):
    if value is None:
        return None
    if hasattr(value, 'name'):
        return value.name
    if isinstance(value, (Color, Vector)):
        return list(value)
    if isinstance(value, (tuple, list)):
        return list(value)
    return value


def matrix_bits(matrix):
    return [[struct.pack('<f', float(matrix[row][column])).hex()
             for column in range(4)] for row in range(4)]


def mesh_attribute_value(item):
    for field in ('vector', 'color', 'value', 'value_int2', 'value_float2'):
        if hasattr(item, field):
            value = getattr(item, field)
            if isinstance(value, (Color, Vector, tuple, list)):
                return list(value)
            if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                return list(value)
            return value
    return None


def mesh_snapshot(mesh):
    attributes = [{
        'name': attribute.name,
        'domain': attribute.domain,
        'data_type': attribute.data_type,
        'values': [mesh_attribute_value(item) for item in attribute.data],
    } for attribute in mesh.attributes]
    attributes.sort(key=lambda item: item['name'])
    return {
        'vertices': [list(vertex.co) for vertex in mesh.vertices],
        'vertex_normals': [list(vertex.normal) for vertex in mesh.vertices],
        'edges': [{'vertices': list(edge.vertices),
                   'sharp': bool(getattr(edge, 'use_edge_sharp', False)),
                   'seam': bool(getattr(edge, 'use_seam', False))}
                  for edge in mesh.edges],
        'polygons': [{
            'vertices': list(polygon.vertices),
            'loop_start': int(polygon.loop_start),
            'loop_total': int(polygon.loop_total),
            'material_index': int(polygon.material_index),
            'smooth': bool(polygon.use_smooth),
            'normal': list(polygon.normal),
        } for polygon in mesh.polygons],
        'loops': [{'vertex_index': int(loop.vertex_index),
                   'edge_index': int(loop.edge_index),
                   'normal': list(loop.normal)} for loop in mesh.loops],
        'corner_normals': [list(normal.vector) for normal in mesh.corner_normals],
        'normals_domain': str(mesh.normals_domain),
        'has_custom_normals': bool(mesh.has_custom_normals),
        'attributes': attributes,
    }


def modifier_prefixes(obj):
    modifiers = list(obj.modifiers)
    enabled = [modifier.show_viewport for modifier in modifiers]
    prefixes = {}
    try:
        for prefix in range(1, len(modifiers) + 1):
            for index, modifier in enumerate(modifiers):
                modifier.show_viewport = index < prefix
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            try:
                prefixes[str(prefix)] = mesh_snapshot(mesh)
            finally:
                evaluated.to_mesh_clear()
    finally:
        for modifier, value in zip(modifiers, enabled):
            modifier.show_viewport = value
        bpy.context.view_layer.update()
    return prefixes


def modifier_info(mod):
    keys = {
        'BEVEL': ('width', 'segments'),
        'ARRAY': ('count', 'use_relative_offset', 'relative_offset_displace', 'end_cap'),
        'SUBSURF': ('subdivision_type', 'levels', 'render_levels'),
        'WEIGHTED_NORMAL': ('keep_sharp',),
        'BOOLEAN': ('operation', 'solver', 'object'),
        'DECIMATE': ('decimate_type', 'ratio'),
        'SOLIDIFY': ('thickness',),
        'REMESH': ('mode', 'voxel_size', 'use_smooth_shade'),
        'ARMATURE': ('object',),
        'SURFACE_DEFORM': ('target', 'is_bound'),
    }.get(mod.type, ())
    return {
        'name': mod.name,
        'type': mod.type,
        **{key: rna_value(getattr(mod, key)) for key in keys},
    }


def constraint_info(constraint):
    keys = {
        'IK': ('target', 'pole_target', 'chain_count', 'pole_angle'),
        'FOLLOW_PATH': ('target', 'use_curve_follow', 'forward_axis', 'up_axis'),
        'COPY_ROTATION': ('target', 'influence'),
        'TRACK_TO': ('target', 'track_axis', 'up_axis'),
    }.get(constraint.type, ('influence',))
    return {
        'name': constraint.name,
        'type': constraint.type,
        **{key: rna_value(getattr(constraint, key)) for key in keys},
    }


def action_info(owner):
    animation = getattr(owner, 'animation_data', None)
    action = animation.action if animation else None
    if action is None:
        return None
    slot = animation.action_slot
    return {
        'name': action.name,
        'frame_range': list(action.frame_range),
        'slot': slot.identifier if slot else None,
        'slots': [
            {'identifier': item.identifier, 'target_id_type': item.target_id_type}
            for item in action.slots
        ],
    }


objects_meta = {}
view_layer = scene.view_layers[0]
for obj in scene.objects:
    data = obj.data
    meta = {
        'type': obj.type,
        'modifiers': [modifier_info(mod) for mod in obj.modifiers],
        'constraints': [constraint_info(constraint) for constraint in obj.constraints],
        'materials': [material.name if material else None for material in data.materials]
        if data and hasattr(data, 'materials') else [],
        'hide_set': bool(obj.hide_get(view_layer=view_layer)),
        'hide_viewport': bool(obj.hide_viewport),
        'hide_render': bool(obj.hide_render),
        'hide_select': bool(obj.hide_select),
        'visible_camera': bool(obj.visible_camera),
        'actions': {
            'object': action_info(obj),
            'data': action_info(data) if data else None,
            'shape_keys': action_info(data.shape_keys)
            if data and getattr(data, 'shape_keys', None) else None,
        },
    }
    if obj.type == 'LIGHT':
        meta['light'] = {
            key: rna_value(getattr(data, key))
            for key in ('type', 'shape', 'size', 'size_y', 'energy', 'color')
            if hasattr(data, key)
        }
    if obj.type == 'CURVE':
        meta['curve'] = {
            key: rna_value(getattr(data, key))
            for key in ('resolution_u', 'dimensions', 'bevel_depth', 'bevel_resolution',
                        'use_path', 'path_duration', 'eval_time')
        }
    if obj.type == 'CAMERA':
        meta['camera'] = {'lens': float(data.lens), 'use_dof': bool(data.dof.use_dof)}
    objects_meta[obj.name] = meta

result = {
    'scene': {
        'frame_start': scene.frame_start,
        'frame_current': scene.frame_current,
        'frame_end': scene.frame_end,
        'resolution_x': scene.render.resolution_x,
        'fps': scene.render.fps,
        'resolution_y': scene.render.resolution_y,
        'resolution_percentage': scene.render.resolution_percentage,
        'engine': scene.render.engine,
        'camera': scene.camera.name if scene.camera else None,
        'world_color': list(scene.world.color) if scene.world else None,
    },
    'objects': objects_meta,
    'frames': {},
    'raw_meshes': {},
}
for name in ('Housing_Boolean_Decimate_Solidify', 'Housing_Bore_Cutter',
             'EdgeOrderRegression'):
    obj = bpy.data.objects[name]
    result['raw_meshes'][name] = {
        'matrix_world_bits': matrix_bits(obj.matrix_world),
        'mesh': mesh_snapshot(obj.data),
        'prefixes': modifier_prefixes(obj)
        if name == 'Housing_Boolean_Decimate_Solidify' else {},
    }
for frame in (1, 5, 10):
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame_data = {}
    for obj in scene.objects:
        ev = obj.evaluated_get(depsgraph)
        matrix = [[float(ev.matrix_world[row][column]) for column in range(4)] for row in range(4)]
        entry = {'type': obj.type, 'matrix_world': matrix, 'matrix_world_bits': matrix_bits(ev.matrix_world)}
        if obj.type in ('MESH', 'CURVE'):
            mesh = ev.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
            entry['vertices_world'] = [list(ev.matrix_world @ vertex.co) for vertex in mesh.vertices]
            entry['faces'] = len(mesh.polygons)
            ev.to_mesh_clear()
        frame_data[obj.name] = entry
    result['frames'][str(frame)] = frame_data
DUMP.write_text(json.dumps(result, separators=(',', ':')))
