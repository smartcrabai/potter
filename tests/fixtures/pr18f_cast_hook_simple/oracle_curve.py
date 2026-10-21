import bpy, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
from mathutils import Matrix
BENT = [(0,0,0),(0,0,2.0),(1.5,0.0,2.8),(2.0,1.2,3.5)]
def matrix_list(m):
    return [m[r][c] for c in range(4) for r in range(4)]  # column-major
def curve_case(points, axis, dims='3D', curve_loc=(0.3,-0.4,0.2), curve_rot=(0.2,0.5,-0.3), curve_scale=(1,1,1), weights=None, invert=False, verts=P):
    def build():
        cu = bpy.data.curves.new('Path', 'CURVE')
        cu.dimensions = dims
        sp = cu.splines.new('POLY')
        sp.points.add(len(points) - 1)
        for p, q in zip(sp.points, points):
            p.co = (q[0], q[1], q[2], 1.0)
        cobj = bpy.data.objects.new('PathObj', cu)
        bpy.context.scene.collection.objects.link(cobj)
        cobj.location = curve_loc
        cobj.rotation_euler = Euler(curve_rot, 'XYZ')
        cobj.scale = curve_scale
        subj = subject()
        if weights is not None:
            set_groups(subj, 'G', weights)
        m = subj.modifiers.new('CV', 'CURVE')
        m.object = cobj
        m.deform_axis = axis
        if weights is not None:
            m.vertex_group = 'G'
            m.invert_vertex_group = invert
        bpy.context.view_layer.update()
        M = subj.matrix_world.inverted() @ cobj.matrix_world
        pos = evaluate(subj)
        return {"positions": pos, "M": matrix_list(M), "path": [list(p) for p in points],
                "verts": [list(v) for v in verts], "subject": matrix_list(subj.matrix_world), "curve": matrix_list(cobj.matrix_world)}
    return build
W = {0:1.0, 1:0.25, 2:0.6, 3:0.0, 4:1.0, 5:0.5, 6:0.8, 7:0.3}
cases = []
for ax in ['POS_X','POS_Y','POS_Z','NEG_X','NEG_Y','NEG_Z']:
    cases.append((f"bent3d_{ax}", curve_case(BENT, ax)))
cases.append(("bent3d_POS_X_scaled", curve_case(BENT, 'POS_X', curve_scale=(1.2,0.8,1.0), curve_rot=(0.0,0.0,0.4))))
cases.append(("bent3d_NEG_Y_vgroup", curve_case(BENT, 'NEG_Y', weights=W)))
cases.append(("bent3d_POS_Z_vgroup_inv", curve_case(BENT, 'POS_Z', weights=W, invert=True)))
cases.append(("bent2d_POS_X", curve_case([(0,0,0),(0,2.0,0),(1.5,2.8,0),(2.0,3.5,1.0)], 'POS_X', dims='2D')))
cases.append(("bent3d_POS_X_planar", curve_case([(0,0,0),(2.0,0,0),(2.8,1.5,0),(3.5,2.0,0)], 'POS_X')))
cases.append(("line3d_POS_Z_default_loc", curve_case([(0,0,0),(0,0,3.0)], 'POS_Z', curve_loc=(0,0,0), curve_rot=(0,0,0))))
run_cases(cases, out_path())
