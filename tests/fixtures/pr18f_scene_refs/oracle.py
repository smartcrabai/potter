import bpy, json, math
from mathutils import Matrix, Vector

# Cube corners at +-0.5 (Potter `box` size 1.0). Keys are corner indices; weights depend on position.
CORNERS = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
GROUP_WEIGHT = lambda p: 1.0 if p[0] > 0 else (0.5 if p[1] > 0 else 0.0)
R = math.radians


def fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(o):
    bpy.context.scene.collection.objects.link(o)
    return o


def empty(name, loc, rot_deg, scale):
    o = bpy.data.objects.new(name, None)
    o.empty_display_size = 0.5
    link(o)
    o.location = loc
    o.rotation_mode = 'XYZ'
    o.rotation_euler = [R(a) for a in rot_deg]
    o.scale = scale
    return o


def mesh_obj(name, loc, rot_deg, scale):
    m = bpy.data.meshes.new(name + 'M')
    m.from_pydata(CORNERS, [], [])
    m.update()
    o = link(bpy.data.objects.new(name, m))
    o.location = loc
    o.rotation_mode = 'XYZ'
    o.rotation_euler = [R(a) for a in rot_deg]
    o.scale = scale
    return o


def add_group(subj, name):
    g = subj.vertex_groups.new(name=name)
    for i, p in enumerate(CORNERS):
        w = GROUP_WEIGHT(p)
        if w > 0:
            g.add([i], w, 'REPLACE')
    return g


def colmajor(m):
    return [m[r][c] for c in range(4) for r in range(4)]


def evaluated(obj):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    out = [[round(v, 9) for v in vert.co] for vert in me.vertices]
    ev.to_mesh_clear()
    return out


results = {}

# ---- Hook: object target, subject transformed, non-identity matrix_inverse, vertex-group weights
fresh()
subj = mesh_obj('Subj', (1, 2, 3), (0, 0, 30), (2, 1, 0.5))
hk = empty('Hk', (0.5, -1, 2), (20, 0, 0), (1.5, 0.7, 1))
add_group(subj, 'H')
mod = subj.modifiers.new('H', 'HOOK')
mod.object = hk
mod.vertex_group = 'H'
mod.falloff_type = 'NONE'
mod.strength = 1.0
M = Matrix.Translation((0.2, -0.3, 0.1)) @ Matrix.Rotation(R(20), 4, 'Z')
mod.matrix_inverse = M
results['hook_object'] = {'matrix_inverse': colmajor(M), 'out': evaluated(subj)}

# ---- Cast: control object location, sphere
fresh()
subj = mesh_obj('Subj', (1, 2, 3), (0, 0, 30), (2, 1, 0.5))
ct = empty('Ct', (0.5, 0.5, 0.5), (0, 0, 40), (1.3, 0.8, 1))
mod = subj.modifiers.new('C', 'CAST')
mod.cast_type = 'SPHERE'
mod.factor = 0.5
mod.radius = 0.0
mod.size = 2.0
mod.use_radius_as_size = False
mod.object = ct
mod.use_transform = False
results['cast_object'] = evaluated(subj)

# ---- Simple Deform TAPER with origin object
fresh()
subj = mesh_obj('Subj', (1, 2, 3), (0, 0, 30), (2, 1, 0.5))
og = empty('Og', (0.5, -0.5, 1), (11.5, 17.2, 22.9), (1.2, 0.9, 1.1))
mod = subj.modifiers.new('S', 'SIMPLE_DEFORM')
mod.deform_method = 'TAPER'
mod.deform_axis = 'Z'
mod.factor = 0.5
mod.origin = og
results['simple_origin'] = evaluated(subj)

# ---- Warp: object_from / object_to, falloff NONE
fresh()
subj = mesh_obj('Subj', (1, 2, 3), (0, 0, 30), (2, 1, 0.5))
wf = empty('Wf', (1, 0, 0), (0, 0, 20), (1, 1.5, 1))
wt = empty('Wt', (0, 2, 1), (40, 0, 0), (0.8, 1, 1.2))
mod = subj.modifiers.new('W', 'WARP')
mod.object_from = wf
mod.object_to = wt
mod.falloff_type = 'NONE'
mod.strength = 1.0
results['warp_objects'] = evaluated(subj)

# ---- Wave: start_position_object at frame 10
fresh()
subj = mesh_obj('Subj', (1, 2, 3), (0, 0, 30), (2, 1, 0.5))
ws = empty('Ws', (1, -1, 0), (0, 0, 0), (1, 1, 1))
bpy.context.scene.frame_set(10)
mod = subj.modifiers.new('V', 'WAVE')
mod.start_position_object = ws
results['wave_start_object'] = evaluated(subj)

# ---- Lattice: object transforms, strength, vertex group, invert, negative strength
def lattice_case(strength, group_on, invert):
    fresh()
    subj = mesh_obj('Subj', (-0.2, 0.1, 0.1), (0, 0, 25), (0.26, 0.24, 0.26))
    lat_obj = bpy.data.objects.new('Lat', bpy.data.lattices.new('LatD'))
    link(lat_obj)
    lat_obj.location = (0.2, -0.1, 0.3)
    lat_obj.rotation_mode = 'XYZ'
    lat_obj.rotation_euler = [R(10), R(15), R(-5)]
    lat_obj.scale = (1.2, 0.9, 1.1)
    lat = lat_obj.data
    lat.interpolation_type_u = 'KEY_BSPLINE'
    lat.interpolation_type_v = 'KEY_BSPLINE'
    lat.interpolation_type_w = 'KEY_BSPLINE'
    for i, p in enumerate(lat.points):
        base = Vector(p.co_deform)
        off = Vector((0.15 * math.sin(i + 1), 0.1 * math.cos(2 * i), -0.12 * (i % 2)))
        p.co_deform = base + off
    if group_on:
        add_group(subj, 'G')
    mod = subj.modifiers.new('L', 'LATTICE')
    mod.object = lat_obj
    mod.strength = strength
    if group_on:
        mod.vertex_group = 'G'
        mod.invert_vertex_group = invert
    bpy.context.view_layer.update()
    lm = lat_obj.matrix_world.inverted() @ subj.matrix_world
    local = [[round(v, 6) for v in (lm @ Vector(v)).to_tuple()] for v in CORNERS]
    return {
        'out': evaluated(subj),
        'lattice_points': [[round(v, 9) for v in p.co_deform] for p in lat.points],
        'lattice_local_of_corners': local,
    }


results['lattice_strength_group'] = lattice_case(0.6, True, False)
results['lattice_strength_group_invert'] = lattice_case(0.6, True, True)
results['lattice_negative_no_group'] = lattice_case(-0.5, False, False)

print('ORACLE_JSON_BEGIN')
print(json.dumps(results))
print('ORACLE_JSON_END')
