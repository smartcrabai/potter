import bpy, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
from mathutils import Matrix
def mlist(m):
    return [m[r][c] for c in range(4) for r in range(4)]
W = {0:1.0, 1:0.25, 2:0.6, 3:0.0, 4:1.0, 5:0.5, 6:0.8, 7:0.3}
def finish(subj, extra):
    bpy.context.view_layer.update()
    out = {"positions": evaluate(subj), "subject": mlist(subj.matrix_world), "verts": [list(p) for p in P]}
    out.update(extra)
    return out

# ---------- cast ----------
def cast_case(cast_type, axes=(True,True,True), factor=0.5, radius=0.0, size=0.0, urad=True, obj=None, use_transform=False, weights=None, invert=False, obj_transform=((-0.4,0.6,0.3),(0.4,-0.1,0.7),(1.3,0.7,1.1))):
    def build():
        subj = subject()
        if weights is not None:
            set_groups(subj, 'G', weights)
        m = subj.modifiers.new('C', 'CAST')
        m.cast_type = cast_type
        m.use_x, m.use_y, m.use_z = axes
        m.factor = factor
        m.radius = radius
        m.size = size
        m.use_radius_as_size = urad
        ex = None
        if obj:
            loc, rot, scl = obj_transform
            ex = empty('Ctrl', loc=loc, rot=rot, scale=scl)
            m.object = ex
            m.use_transform = use_transform
        if weights is not None:
            m.vertex_group = 'G'
            m.invert_vertex_group = invert
        extra = {"cast_type": cast_type, "axes": list(axes), "factor": factor, "radius": radius, "size": size,
                 "urad": urad, "obj": obj, "use_transform": use_transform, "invert": invert, "weights": weights is not None}
        if ex is not None:
            bpy.context.view_layer.update()
            extra["M"] = mlist(subj.matrix_world.inverted() @ ex.matrix_world)
        return finish(subj, extra)
    return build
cases = []
for ct in ['SPHERE','CYLINDER','CUBOID']:
    cases.append((f"cast_{ct}_auto", cast_case(ct)))
    cases.append((f"cast_{ct}_radius", cast_case(ct, radius=1.6, factor=0.7, urad=True)))
    cases.append((f"cast_{ct}_size", cast_case(ct, size=1.25, factor=0.4, urad=False)))
    cases.append((f"cast_{ct}_radsize_false", cast_case(ct, radius=1.6, size=0.9, factor=0.6, urad=False)))
    cases.append((f"cast_{ct}_axes_xy", cast_case(ct, axes=(True,True,False), factor=0.8, size=1.0, urad=False)))
    cases.append((f"cast_{ct}_axes_z", cast_case(ct, axes=(False,False,True), factor=0.8, radius=2.0)))
    cases.append((f"cast_{ct}_obj", cast_case(ct, obj=True, factor=0.6, radius=2.5, use_transform=False)))
    cases.append((f"cast_{ct}_obj_tr", cast_case(ct, obj=True, factor=0.6, size=1.1, urad=False, use_transform=True)))
    cases.append((f"cast_{ct}_obj_tr_radius", cast_case(ct, obj=True, factor=0.5, radius=1.8, use_transform=True, axes=(True,False,True))))
    cases.append((f"cast_{ct}_vg", cast_case(ct, weights=W, factor=0.9, size=1.0, urad=False, obj=True, use_transform=True)))
    cases.append((f"cast_{ct}_vginv", cast_case(ct, weights=W, invert=True, factor=0.9, radius=1.5, obj=True, use_transform=False)))
cases.append(("cast_sphere_factor_one", cast_case('SPHERE', factor=1.0, size=2.0, urad=False)))
run_cases(cases, out_path())
