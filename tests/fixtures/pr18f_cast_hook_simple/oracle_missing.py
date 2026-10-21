import bpy, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
from mathutils import Matrix, Euler
W = {0:1.0, 1:0.25, 2:0.6, 3:0.0, 4:1.0, 5:0.5, 6:0.8, 7:0.3}
def mlist(m): return [m[r][c] for c in range(4) for r in range(4)]
def pinv_matrix():
    return (Matrix.Translation((0.35, -0.2, 0.5)) @ Euler((0.3, 0.4, -0.2), 'XYZ').to_matrix().to_4x4() @ Matrix.Diagonal((1.2, 0.8, 1.0, 1.0)))
def weighted():
    s = subject(); set_groups(s, 'G', W); return s
def cast_missing():
    s = weighted(); m = s.modifiers.new('C','CAST'); m.cast_type='SPHERE'; m.factor=0.5
    m.vertex_group = 'missing'; m.invert_vertex_group = True
    return {"positions": evaluate(s)}
def sd_missing():
    s = weighted(); ex = empty('O', loc=(-0.4,0.6,0.3), rot=(0.4,-0.1,0.7), scale=(1.3,0.7,1.1))
    m = s.modifiers.new('SD','SIMPLE_DEFORM'); m.deform_method='TWIST'; m.deform_axis='Z'; m.angle=0.9; m.origin=ex
    m.vertex_group='missing'; m.invert_vertex_group=True
    return {"positions": evaluate(s), "M": mlist(s.matrix_world.inverted() @ ex.matrix_world)}
def hook_missing(indices=True):
    def build():
        s = weighted(); ex = empty('H', loc=(-0.4,0.6,0.3), rot=(0.4,-0.1,0.7), scale=(1.3,0.7,1.1))
        m = s.modifiers.new('H','HOOK'); m.object=ex; m.falloff_type='SMOOTH'; m.falloff_radius=1.5; m.strength=0.8
        m.center=(0.2,-0.3,0.4); m.matrix_inverse=pinv_matrix()
        if indices: m.vertex_indices_set([0,1,2,3,4,5,6,7])
        m.vertex_group='missing'; m.invert_vertex_group=True
        return {"positions": evaluate(s), "M": mlist(s.matrix_world.inverted() @ ex.matrix_world), "parentinv": mlist(pinv_matrix())}
    return build
def curve_missing():
    cu = bpy.data.curves.new('Path','CURVE'); cu.dimensions='3D'
    sp = cu.splines.new('POLY'); pts=[(0,0,0),(0,0,2.0),(1.5,0,2.8),(2.0,1.2,3.5)]
    sp.points.add(len(pts)-1)
    for p,q in zip(sp.points, pts): p.co=(q[0],q[1],q[2],1.0)
    co = bpy.data.objects.new('PathObj', cu); bpy.context.scene.collection.objects.link(co)
    co.location=(0.3,-0.4,0.2); co.rotation_euler=Euler((0.2,0.5,-0.3),'XYZ')
    s = weighted(); m = s.modifiers.new('CV','CURVE'); m.object=co; m.deform_axis='POS_X'
    m.vertex_group='missing'; m.invert_vertex_group=True
    bpy.context.view_layer.update()
    return {"positions": evaluate(s), "M": mlist(s.matrix_world.inverted() @ co.matrix_world), "path": [list(p) for p in pts]}
cases = [("cast_missing", cast_missing), ("sd_missing", sd_missing), ("hook_missing_indices", hook_missing(True)),
         ("hook_missing_no_indices", hook_missing(False)), ("curve_missing", curve_missing)]
run_cases(cases, out_path())
