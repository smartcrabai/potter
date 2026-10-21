import bpy, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
from mathutils import Matrix
def mlist(m):
    return [m[r][c] for c in range(4) for r in range(4)]
W = {0:1.0, 1:0.25, 2:0.6, 3:0.0, 4:1.0, 5:0.5, 6:0.8, 7:0.3}
def base_subject(weights=None):
    subj = subject()
    if weights is not None:
        set_groups(subj, 'G', weights)
    return subj
def finish(subj, extra):
    bpy.context.view_layer.update()
    out = {"positions": evaluate(subj), "subject": mlist(subj.matrix_world), "verts": [list(p) for p in P]}
    out.update(extra)
    return out

# ---------- simple deform ----------
def sd_case(method, axis, angle=0.9, factor=0.6, limits=(0.0,1.0), locks=(False,False,False), origin=False, weights=None, invert=False):
    def build():
        subj = base_subject(weights)
        m = subj.modifiers.new('SD', 'SIMPLE_DEFORM')
        m.deform_method = method
        m.deform_axis = axis
        if method in ('TWIST','BEND'):
            m.angle = angle
        else:
            m.factor = factor
        m.limits = limits
        m.lock_x, m.lock_y, m.lock_z = locks
        ex = None
        if origin:
            ex = empty('Origin', loc=(-0.4, 0.6, 0.3), rot=(0.4, -0.1, 0.7), scale=(1.3, 0.7, 1.1))
            m.origin = ex
        if weights is not None:
            m.vertex_group = 'G'
            m.invert_vertex_group = invert
        extra = {"method": method, "axis": axis, "angle": angle, "factor": factor, "limits": list(limits),
                 "locks": list(locks), "origin": origin, "invert": invert, "weights": weights is not None}
        if origin:
            bpy.context.view_layer.update()
            extra["M"] = mlist(subj.matrix_world.inverted() @ ex.matrix_world)
        return finish(subj, extra)
    return build
cases = []
for method in ['TWIST','BEND','TAPER','STRETCH']:
    for axis in ['X','Y','Z']:
        cases.append((f"sd_{method}_{axis}", sd_case(method, axis)))
        cases.append((f"sd_{method}_{axis}_lim", sd_case(method, axis, limits=(0.2,0.7), factor=1.1, angle=1.3)))
        cases.append((f"sd_{method}_{axis}_origin", sd_case(method, axis, origin=True, factor=0.8, angle=1.1, limits=(0.15,0.85))))
        cases.append((f"sd_{method}_{axis}_lock", sd_case(method, axis, locks=(True,True,True), origin=True, factor=0.7, angle=0.8)))
        cases.append((f"sd_{method}_{axis}_vg", sd_case(method, axis, weights=W, factor=0.9, angle=1.0, origin=True)))
        cases.append((f"sd_{method}_{axis}_vginv", sd_case(method, axis, weights=W, invert=True, factor=0.9, angle=1.0, origin=True, limits=(0.1,0.9))))
cases.append(("sd_alias_probe", None))
cases = [c for c in cases if c[1] is not None]
# alias probe only in simple (no mesh needed)
run_cases(cases, out_path())
