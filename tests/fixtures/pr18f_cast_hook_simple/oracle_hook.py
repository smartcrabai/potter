import bpy, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
from mathutils import Matrix
def mlist(m):
    return [m[r][c] for c in range(4) for r in range(4)]
W = {0:1.0, 1:0.25, 2:0.6, 3:0.0, 4:1.0, 5:0.5, 6:0.8, 7:0.3}
PI_MAT = Matrix.Translation((0.35, -0.2, 0.5)) @ Matrix.Rotation(0.6, 4, 'Z') @ Matrix.Scale(1.4, 4, (1.0, 0.5, 1.0)) if False else None
def pinv_matrix():
    from mathutils import Euler
    return (Matrix.Translation((0.35, -0.2, 0.5)) @ Euler((0.3, 0.4, -0.2), 'XYZ').to_matrix().to_4x4() @ Matrix.Diagonal((1.2, 0.8, 1.0, 1.0)))
def finish(subj, extra):
    bpy.context.view_layer.update()
    out = {"positions": evaluate(subj), "subject": mlist(subj.matrix_world), "verts": [list(p) for p in P]}
    out.update(extra)
    return out
def hook_case(falloff='SMOOTH', radius=1.5, strength=0.8, center=(0.2,-0.3,0.4), uniform=False, indices='ALL', weights=None, invert=False, curve_points=None, use_pinv=True, obj_transform=((-0.4,0.6,0.3),(0.4,-0.1,0.7),(1.3,0.7,1.1))):
    def build():
        subj = subject()
        if weights is not None:
            set_groups(subj, 'G', weights)
        loc, rot, scl = obj_transform
        ex = empty('Hk', loc=loc, rot=rot, scale=scl)
        m = subj.modifiers.new('H', 'HOOK')
        m.object = ex
        m.falloff_type = falloff
        m.falloff_radius = radius
        m.strength = strength
        m.center = center
        m.use_falloff_uniform = uniform
        pinv = pinv_matrix() if use_pinv else Matrix.Identity(4)
        m.matrix_inverse = pinv
        if curve_points is not None:
            cm = m.falloff_curve.curves[0]
            pts = cm.points
            # reset to exactly the given points
            while len(pts) > 2:
                pts.remove(pts[-1])
            pts[0].location = curve_points[0]
            pts[1].location = curve_points[-1]
            for p in curve_points[1:-1]:
                pts.new(p[0], p[1])
            m.falloff_curve.update()
        sel = indices
        if sel == 'ALL':
            sel = None if weights is not None else list(range(len(P)))
        if sel is not None:
            m.vertex_indices_set(sel)
        if weights is not None:
            m.vertex_group = 'G'
            m.invert_vertex_group = invert
        bpy.context.view_layer.update()
        extra = {"falloff": falloff, "radius": radius, "strength": strength, "center": list(center), "uniform": uniform,
                 "indices": sel, "weights": weights is not None, "invert": invert, "curve": curve_points,
                 "parentinv": mlist(pinv), "M": mlist(subj.matrix_world.inverted() @ ex.matrix_world)}
        return finish(subj, extra)
    return build
cases = []
for ft in ['NONE','CURVE','SMOOTH','SPHERE','ROOT','INVERSE_SQUARE','SHARP','LINEAR','CONSTANT']:
    cases.append((f"hook_{ft}", hook_case(falloff=ft)))
cases.append(("hook_noop_no_indices_no_group", hook_case(falloff='SMOOTH', indices=None)))
cases.append(("hook_SMOOTH_radius0", hook_case(falloff='SMOOTH', radius=0.0)))
cases.append(("hook_NONE_radius", hook_case(falloff='NONE', radius=1.5)))
cases.append(("hook_SMOOTH_nopinv", hook_case(falloff='SMOOTH', use_pinv=False)))
cases.append(("hook_SMOOTH_uniform", hook_case(falloff='SMOOTH', uniform=True)))
cases.append(("hook_LINEAR_uniform_big", hook_case(falloff='LINEAR', uniform=True, radius=2.6, strength=1.0)))
cases.append(("hook_ROOT_uniform", hook_case(falloff='ROOT', uniform=True, radius=2.2)))
cases.append(("hook_indices", hook_case(falloff='SMOOTH', indices=[0,2,5,7], radius=0.0)))
cases.append(("hook_indices_falloff", hook_case(falloff='SPHERE', indices=[0,2,5,7], radius=2.0)))
cases.append(("hook_vg", hook_case(falloff='LINEAR', weights=W, radius=2.0)))
cases.append(("hook_vg_inv", hook_case(falloff='LINEAR', weights=W, invert=True, radius=2.0)))
cases.append(("hook_indices_vg", hook_case(falloff='CONSTANT', indices=[0,1,2,3,4,5,6,7], weights=W, radius=0.0)))
cases.append(("hook_indices_vg_inv", hook_case(falloff='NONE', indices=[1,2,3,4,5,6,7], weights=W, invert=True, radius=0.0)))
cases.append(("hook_curve_default", hook_case(falloff='CURVE', curve_points=[(0,0),(1,1)], radius=2.0)))
cases.append(("hook_curve_3pt", hook_case(falloff='CURVE', curve_points=[(0,0),(0.3,0.7),(1,1)], radius=2.0)))
cases.append(("hook_curve_3pt_b", hook_case(falloff='CURVE', curve_points=[(0,0),(0.5,0.1),(1,1)], radius=2.0)))
cases.append(("hook_curve_4pt", hook_case(falloff='CURVE', curve_points=[(0,0.2),(0.25,0.8),(0.6,0.3),(1,0.9)], radius=2.0)))
run_cases(cases, out_path())
