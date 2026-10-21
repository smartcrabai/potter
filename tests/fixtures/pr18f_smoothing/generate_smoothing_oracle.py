# Blender 5.2.2 oracle for laplacian_smooth and corrective_smooth (PR18-F).
# Run: /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python oracle.py
import bpy, json, math, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_oracle.json")
R, S = 4, 6  # rings, segments

def r4(v):
    return round(v, 4)

def flat_positions():
    pts = []
    for k in range(R):
        for j in range(S):
            ang = 2.0 * math.pi * j / S + 0.13 * k
            rad = 1.0 + 0.25 * (((j * 3 + k * 5) % 5) / 4.0)
            x = rad * math.cos(ang)
            y = rad * math.sin(ang) * (1.0 + 0.1 * k)
            z = 0.9 * k + 0.12 * (((j * 7 + k * 3) % 4))
            pts.append((r4(x), r4(y), r4(z)))
    return pts

def bent_positions(flat):
    out = []
    for x, y, z in flat:
        out.append((r4(1.3 * x + 0.15 * z * z), r4(0.8 * y + 0.05 * x), r4(z + 0.1 * x * z)))
    return out

def faces(closed):
    f = []
    k_range = R if closed else R - 1
    for k in range(k_range):
        for j in range(S):
            a = k * S + j
            b = k * S + (j + 1) % S
            c = ((k + 1) % R) * S + (j + 1) % S
            d = ((k + 1) % R) * S + j
            f.append([a, b, c, d])
    return f

def groups():
    g = []
    for i in range(R * S):
        if i % 4 == 3:
            continue
        g.append((i, ((i * 7) % 5) / 4.0))
    return g

def make_obj(name, verts, fs, grp=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], fs)
    me.update(calc_edges=True)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    if grp:
        vg = obj.vertex_groups.new(name="G")
        for i, w in groups():
            vg.add([i], w, 'REPLACE')
    return obj

def evaluate(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    out = [[v.co.x, v.co.y, v.co.z] for v in me.vertices]
    ev.to_mesh_clear()
    return out

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def clean():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)

flat = flat_positions()
bent = bent_positions(flat)
result = {
    "blender": bpy.app.version_string,
    "rings": R, "segments": S,
    "flat": flat, "bent": bent,
    "groups": groups(),
    "faces_open": faces(False), "faces_closed": faces(True),
    "cases": {},
}

LAP_CASES = {
    "lap_a_plain_iter3": dict(iterations=3, lambda_factor=0.5, lambda_border=0.2, use_normalized=False, use_volume_preserve=False),
    "lap_b_norm_vol_iter2": dict(iterations=2, lambda_factor=0.5, lambda_border=0.2, use_normalized=True, use_volume_preserve=True),
    "lap_c_negative_axes": dict(iterations=2, lambda_factor=-0.4, lambda_border=0.3, use_normalized=True, use_volume_preserve=False, use_x=True, use_y=False, use_z=True),
    "lap_d_neg_border_noz": dict(iterations=2, lambda_factor=0.6, lambda_border=-0.5, use_normalized=False, use_volume_preserve=True, use_z=False),
    "lap_e_group": dict(iterations=2, lambda_factor=0.5, lambda_border=0.2, use_normalized=True, use_volume_preserve=False, vertex_group="G", invert_vertex_group=False),
    "lap_f_group_inv_plain": dict(iterations=2, lambda_factor=0.5, lambda_border=0.2, use_normalized=False, use_volume_preserve=True, vertex_group="G", invert_vertex_group=True),
    "lap_g_missing_group": dict(iterations=2, lambda_factor=0.5, lambda_border=0.2, use_normalized=True, use_volume_preserve=True, vertex_group="nope", invert_vertex_group=True),
    "lap_h_zero_lambda_border": dict(iterations=2, lambda_factor=0.5, lambda_border=0.0, use_normalized=False, use_volume_preserve=False),
    "lap_i_bent_norm": dict(iterations=2, lambda_factor=0.7, lambda_border=0.4, use_normalized=True, use_volume_preserve=True, use_x=False, use_y=True, use_z=True),
}

for name, params in LAP_CASES.items():
    clean()
    obj = make_obj(name, bent, faces(False), grp=True)
    m = obj.modifiers.new("L", 'LAPLACIANSMOOTH')
    for k, v in params.items():
        setattr(m, k, v)
    result["cases"][name] = {"params": params, "out": evaluate(obj)}

clean()
obj = make_obj("torus", bent, faces(True), grp=False)
m = obj.modifiers.new("L", 'LAPLACIANSMOOTH')
m.iterations = 2; m.lambda_factor = 0.5; m.use_normalized = False; m.use_volume_preserve = True
result["cases"]["lap_t_closed_torus"] = {"params": dict(iterations=2, lambda_factor=0.5, use_normalized=False, use_volume_preserve=True), "out": evaluate(obj)}

COR_CASES = {
    "cor_a_simple_bind": dict(factor=0.5, iterations=5, scale=1.0, smooth_type='SIMPLE'),
    "cor_b_length_bind": dict(factor=0.5, iterations=3, scale=1.2, smooth_type='LENGTH_WEIGHTED'),
    "cor_c_pin_bind": dict(factor=0.5, iterations=4, scale=1.0, smooth_type='SIMPLE', use_pin_boundary=True),
    "cor_d_group_bind": dict(factor=0.5, iterations=2, scale=1.0, smooth_type='SIMPLE', vertex_group="G", invert_vertex_group=False),
    "cor_e_group_inv_length_pin": dict(factor=0.4, iterations=3, scale=-0.5, smooth_type='LENGTH_WEIGHTED', vertex_group="G", invert_vertex_group=True, use_pin_boundary=True),
    "cor_f_only_smooth": dict(factor=0.5, iterations=3, scale=1.0, smooth_type='SIMPLE', use_only_smooth=True),
    "cor_g_only_smooth_length_group": dict(factor=0.3, iterations=2, scale=1.0, smooth_type='LENGTH_WEIGHTED', use_only_smooth=True, vertex_group="G", invert_vertex_group=True),
}

for name, params in COR_CASES.items():
    clean()
    obj = make_obj(name, flat, faces(False), grp=True)
    m = obj.modifiers.new("C", 'CORRECTIVE_SMOOTH')
    m.rest_source = 'BIND'
    for k, v in params.items():
        setattr(m, k, v)
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.object.correctivesmooth_bind(modifier="C")
    evaluate(obj)  # stores bind coords from the flat (rest) input
    bound = m.is_bind
    me = obj.data
    me.vertices.foreach_set("co", [c for p in bent for c in p])
    me.update()
    result["cases"][name] = {"params": params, "is_bind": bound, "out": evaluate(obj)}

# ORCO: rest = object data (bent), input = bent -> identity when scale 1.
for name, params in {
    "cor_h_orco_scale1": dict(factor=0.5, iterations=3, scale=1.0, rest_source='ORCO'),
    "cor_i_orco_scale05": dict(factor=0.5, iterations=3, scale=0.5, rest_source='ORCO', smooth_type='LENGTH_WEIGHTED'),
}.items():
    clean()
    obj = make_obj(name, bent, faces(False), grp=False)
    m = obj.modifiers.new("C", 'CORRECTIVE_SMOOTH')
    for k, v in params.items():
        setattr(m, k, v)
    result["cases"][name] = {"params": params, "out": evaluate(obj)}

# Chained: laplacian (lap_b) then corrective ORCO with scale 0.5; ORCO rest = object data (bent).
clean()
obj = make_obj("chain", bent, faces(False), grp=False)
l = obj.modifiers.new("L", 'LAPLACIANSMOOTH')
for k, v in LAP_CASES["lap_b_norm_vol_iter2"].items():
    if k == "vertex_group": continue
    setattr(l, k, v)
c = obj.modifiers.new("C", 'CORRECTIVE_SMOOTH')
c.factor = 0.5; c.iterations = 3; c.scale = 0.5; c.rest_source = 'ORCO'
result["cases"]["chain_lap_b_then_cor_orco"] = {"out": evaluate(obj)}

# Chained: corrective BIND then laplacian (lap_a).
clean()
obj = make_obj("chain2", flat, faces(False), grp=False)
c = obj.modifiers.new("C", 'CORRECTIVE_SMOOTH')
c.factor = 0.5; c.iterations = 4; c.scale = 1.0; c.rest_source = 'BIND'
with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
    bpy.ops.object.correctivesmooth_bind(modifier="C")
evaluate(obj)
obj.data.vertices.foreach_set("co", [cc for p in bent for cc in p]); obj.data.update()
l = obj.modifiers.new("L", 'LAPLACIANSMOOTH')
for k, v in LAP_CASES["lap_a_plain_iter3"].items():
    setattr(l, k, v)
result["cases"]["chain_cor_bind_then_lap_a"] = {"out": evaluate(obj)}

with open(OUT, "w") as f:
    json.dump(result, f, indent=1)
print("WROTE", OUT, len(result["cases"]))
