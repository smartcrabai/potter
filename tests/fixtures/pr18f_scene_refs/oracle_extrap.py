import bpy, json, math
from mathutils import Vector

# Subject: 8 box corners. Lattice: default 2x2x2 grid with deformed points (co_deform).
CORNERS = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
R = math.radians


def fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(o):
    bpy.context.scene.collection.objects.link(o)
    return o


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


def run_case(interp, subject_loc, subject_scale, subject_rot_deg):
    fresh()
    m = bpy.data.meshes.new('SubjM')
    m.from_pydata(CORNERS, [], [])
    m.update()
    subj = link(bpy.data.objects.new('Subj', m))
    subj.location = subject_loc
    subj.rotation_mode = 'XYZ'
    subj.rotation_euler = [R(a) for a in subject_rot_deg]
    subj.scale = subject_scale

    lat_obj = link(bpy.data.objects.new('Lat', bpy.data.lattices.new('LatD')))
    lat_obj.location = (0.2, -0.1, 0.3)
    lat_obj.rotation_mode = 'XYZ'
    lat_obj.rotation_euler = [R(10), R(15), R(-5)]
    lat_obj.scale = (1.2, 0.9, 1.1)
    lat = lat_obj.data
    lat.interpolation_type_u = interp
    lat.interpolation_type_v = interp
    lat.interpolation_type_w = interp
    for i, p in enumerate(lat.points):
        base = Vector(p.co_deform)
        off = Vector((0.15 * math.sin(i + 1), 0.1 * math.cos(2 * i), -0.12 * (i % 2)))
        p.co_deform = base + off

    mod = subj.modifiers.new('L', 'LATTICE')
    mod.object = lat_obj
    mod.strength = 1.0
    bpy.context.view_layer.update()
    lm = lat_obj.matrix_world.inverted() @ subj.matrix_world
    # Lattice cell coordinate: rest grid point i sits at (-1 + 2i/(n-1)) in lattice local units.
    # The cage spans [-1,1] only for default domain; report local coords for the reader.
    local = [[round(v, 6) for v in (lm @ Vector(v)).to_tuple()] for v in CORNERS]
    return {
        'out': evaluated(subj),
        'lattice_local_of_corners': local,
        'lattice_points': [[round(v, 9) for v in p.co_deform] for p in lat.points],
        'lattice_matrix': colmajor(lat_obj.matrix_world),
        'subject_matrix': colmajor(subj.matrix_world),
    }


results = {}
# Subject scale/offset chosen so some corners lie inside, some in the extrapolated cells, and some
# beyond the two-cell clamp in lattice-local space. Lattice domain here is the Blender default +-1.
for name, interp in [('linear', 'KEY_LINEAR'), ('bspline', 'KEY_BSPLINE'),
                     ('cardinal', 'KEY_CARDINAL'), ('catmull_rom', 'KEY_CATMULL_ROM')]:
    results[name + '_near'] = run_case(interp, (0.1, 0.0, 0.0), (0.9, 0.9, 0.9), (0, 0, 0))
    results[name + '_far'] = run_case(interp, (0.5, -0.3, 0.2), (2.4, 2.1, 2.2), (12, -8, 20))
    results[name + '_extreme'] = run_case(interp, (-0.9, 0.6, -0.4), (4.6, 4.2, 3.8), (5, 25, -15))

print('ORACLE_JSON_BEGIN')
print(json.dumps(results))
print('ORACLE_JSON_END')
