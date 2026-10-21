import bpy, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
from mathutils import Matrix
def mlist(m):
    return [m[r][c] for c in range(4) for r in range(4)]
W = {0:1.0, 1:0.25, 2:0.6, 3:0.0, 4:1.0, 5:0.5, 6:0.8, 7:0.3}
def pinv_matrix():
    from mathutils import Euler
    return (Matrix.Translation((0.35, -0.2, 0.5)) @ Euler((0.3, 0.4, -0.2), 'XYZ').to_matrix().to_4x4() @ Matrix.Diagonal((1.2, 0.8, 1.0, 1.0)))
def chain_case():
    subj = subject()
    e1 = empty('Origin', loc=(-0.4, 0.6, 0.3), rot=(0.4, -0.1, 0.7), scale=(1.3, 0.7, 1.1))
    e2 = empty('Ctrl', loc=(0.2, -0.5, 0.9), rot=(-0.3, 0.2, 0.1), scale=(0.9, 1.2, 1.0))
    e3 = empty('Hk', loc=(-0.1, 0.4, -0.6), rot=(0.2, 0.5, -0.4), scale=(1.1, 0.8, 1.2))
    set_groups(subj, 'G', W)
    sd = subj.modifiers.new('SD', 'SIMPLE_DEFORM')
    sd.deform_method = 'TWIST'; sd.deform_axis = 'Z'; sd.angle = 1.2
    sd.origin = e1; sd.limits = (0.1, 0.9)
    c = subj.modifiers.new('C', 'CAST')
    c.cast_type = 'SPHERE'; c.factor = 0.6; c.size = 1.4; c.use_radius_as_size = False
    c.object = e2; c.use_transform = True; c.use_x = True; c.use_y = True; c.use_z = False
    h = subj.modifiers.new('H', 'HOOK')
    h.object = e3; h.falloff_type = 'SMOOTH'; h.falloff_radius = 2.5; h.strength = 0.9
    h.center = (0.1, 0.2, -0.3)
    h.matrix_inverse = pinv_matrix()
    h.vertex_indices_set([1, 3, 4, 6])
    bpy.context.view_layer.update()
    M_sd = subj.matrix_world.inverted() @ e1.matrix_world
    M_c = subj.matrix_world.inverted() @ e2.matrix_world
    M_h = subj.matrix_world.inverted() @ e3.matrix_world
    return {"positions": evaluate(subj), "verts": [list(p) for p in P],
            "M_sd": mlist(M_sd), "M_cast": mlist(M_c), "M_hook": mlist(M_h), "parentinv": mlist(pinv_matrix()),
            "weights": {str(k): v for k, v in W.items()}}
run_cases([("chain_sd_cast_hook", chain_case)], out_path())
