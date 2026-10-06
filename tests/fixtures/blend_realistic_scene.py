import bpy
import json
import math
from mathutils import Vector
from pathlib import Path

OUT = Path(__file__).resolve().parent
BLEND = OUT / 'stacked.blend'
DUMP = OUT / 'original_dump.json'

# Start from a deliberately clean scene, retaining no template assets.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.cameras, bpy.data.lights, bpy.data.armatures, bpy.data.materials):
    for block in list(datablocks):
        if block.users == 0:
            datablocks.remove(block)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 10
scene.render.fps = 24
scene.render.resolution_x = 1100
scene.render.resolution_y = 760
scene.render.resolution_percentage = 100
scene.render.engine = 'BLENDER_EEVEE'
scene.world.color = (0.055, 0.065, 0.085)

# Saturated but physically plausible studio materials make both the original and
# round-tripped scene easy to read in solid/beauty previews.
def material(name, color, metallic=0.0, roughness=0.42):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1.0)
    m.node_tree.nodes.clear()
    bsdf = m.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    output = m.node_tree.nodes.new('ShaderNodeOutputMaterial')
    output.location = (280, 0)
    m.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    return m

steel = material('Titanium | cool brushed steel', (0.22, 0.34, 0.48), 0.72, 0.27)
copper = material('Copper | machined housing', (0.72, 0.25, 0.085), 0.58, 0.3)
ceramic = material('Ceramic | graphite blue', (0.055, 0.15, 0.24), 0.24, 0.24)
bright = material('Signal | amber', (0.95, 0.48, 0.075), 0.34, 0.28)

# Custom closed annular gear profile, extruded with shallow machined teeth.
def make_gear_mesh(name, teeth=12, root=0.66, tip=0.86, bore=0.2, half_depth=0.13):
    # Four evenly spaced profile samples per tooth: root, tooth shoulder,
    # tooth shoulder, root. The profile remains a clean, watertight solid.
    outer_xy = []
    for t in range(teeth):
        center = 2.0 * math.pi * t / teeth
        for offset, radius in ((-0.43, root), (-0.24, tip), (0.24, tip), (0.43, root)):
            a = center + offset * (2.0 * math.pi / teeth)
            outer_xy.append((radius * math.cos(a), radius * math.sin(a)))
    n = len(outer_xy)
    verts = []
    for z in (-half_depth, half_depth):
        verts.extend((x, y, z) for x, y in outer_xy)
        verts.extend((bore * math.cos(2.0 * math.pi * i / n), bore * math.sin(2.0 * math.pi * i / n), z) for i in range(n))
    faces = []
    # Top annulus and reversed bottom annulus.
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
        faces.append((2*n + i, 3*n + i, 3*n + j, 2*n + j))
        # Outer and inner cylinder walls.
        faces.append((i, 2*n + i, 2*n + j, j))
        faces.append((n + i, n + j, 3*n + j, 3*n + i))
    mesh = bpy.data.meshes.new(name + 'Mesh')
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(steel)
    mesh.update()
    return mesh

gear_mesh = make_gear_mesh('DriveGearProfile')
gear = bpy.data.objects.new('Gear_Array_Bevel_Subdivision_Normal', gear_mesh)
scene.collection.objects.link(gear)
gear.location = (-5.0, 0.0, 1.0)
bevel = gear.modifiers.new('01 | two-pass edge break', 'BEVEL')
bevel.width = 0.028
bevel.segments = 2
array = gear.modifiers.new('02 | three-stage train with end cap', 'ARRAY')
array.count = 3
array.use_relative_offset = True
array.relative_offset_displace = (1.55, 0.0, 0.0)
# The cap object is deliberately excluded from display/render; the Array
# modifier still consumes its mesh as the end cap reference.
cap = bpy.data.objects.new('Gear_Array_EndCap_Source', make_gear_mesh('EndCapProfile', teeth=12, root=0.66, tip=0.86, bore=0.2, half_depth=0.13))
scene.collection.objects.link(cap)
cap.location = (-12.0, -8.0, -5.0)
cap.hide_render = True
cap.hide_set(True)
array.end_cap = cap
subd = gear.modifiers.new('03 | finishing subdivision', 'SUBSURF')
subd.subdivision_type = 'CATMULL_CLARK'
subd.levels = 1
subd.render_levels = 1
wn = gear.modifiers.new('04 | weighted face normals', 'WEIGHTED_NORMAL')
wn.keep_sharp = True

# A machined block with an exact cylindrical bore, intentionally retaining
# downstream topology-changing modifiers for stack-interaction coverage.
bpy.ops.mesh.primitive_cube_add(size=1.8, location=(3.25, 0.0, 1.0))
box = bpy.context.object
box.name = 'Housing_Boolean_Decimate_Solidify'
box.data.name = 'HousingBlockMesh'
box.data.materials.append(copper)
cutter_mesh = None
bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.39, depth=2.8, location=(3.25, 0.0, 1.0), rotation=(math.radians(16), math.radians(8), math.radians(17)))
cutter = bpy.context.object
cutter.name = 'Housing_Bore_Cutter'
cutter_mesh = cutter.data
cutter.hide_render = True
cutter.hide_set(True)
boolean = box.modifiers.new('01 | exact through-bore', 'BOOLEAN')
boolean.operation = 'DIFFERENCE'
boolean.solver = 'EXACT'
boolean.object = cutter
dec = box.modifiers.new('02 | collapse reduction', 'DECIMATE')
dec.decimate_type = 'COLLAPSE'
dec.ratio = 0.6
solid = box.modifiers.new('03 | shell finish', 'SOLIDIFY')
solid.thickness = 0.055

# Preserve an explicit Blender edge order that differs from Potter's
# face-boundary insertion order for the same polygon.
edge_order_mesh = bpy.data.meshes.new('ExplicitEdgeOrderMesh')
edge_order_mesh.from_pydata(
    [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
    [(2, 3), (0, 3), (1, 2), (0, 1)],
    [(0, 1, 2, 3)],
)
edge_order_mesh.update()
edge_order_obj = bpy.data.objects.new('EdgeOrderRegression', edge_order_mesh)
scene.collection.objects.link(edge_order_obj)
edge_order_obj.location = (30, 0, 0)

# Closed sphere remeshed in voxel mode.
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.88, location=(7.1, 0.0, 1.0))
remesh_obj = bpy.context.object
remesh_obj.name = 'Sphere_Voxel_Remesh'
remesh_obj.data.name = 'SphereSourceMesh'
remesh_obj.data.materials.append(ceramic)
for p in remesh_obj.data.polygons:
    p.use_smooth = True
remesh = remesh_obj.modifiers.new('01 | watertight voxel remesh', 'REMESH')
remesh.mode = 'VOXEL'
remesh.voxel_size = 0.12
remesh.use_smooth_shade = True

# A 3-bone articulated rig and a weighted cylindrical skin.
rig_base = Vector((-5.0, 4.0, 0.0))
arm_data = bpy.data.armatures.new('ThreeLinkRigData')
arm = bpy.data.objects.new('Rig_ThreeBones_IK', arm_data)
scene.collection.objects.link(arm)
arm.location = rig_base
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
prev = None
for idx in range(3):
    bone = arm_data.edit_bones.new('Link_%02d' % (idx + 1))
    bone.head = (0.0, 0.0, float(idx))
    bone.tail = (0.0, 0.0, float(idx + 1))
    if prev:
        bone.parent = prev
        bone.use_connect = True
    prev = bone
bpy.ops.object.mode_set(mode='OBJECT')

bpy.ops.object.empty_add(type='SPHERE', location=(-4.0, 4.0, 2.45))
ik_target = bpy.context.object
ik_target.name = 'IK_Target_Animated'
ik_target.empty_display_size = 0.18
ik_target.keyframe_insert(data_path='location', frame=1)
ik_target.location.x = -3.55
ik_target.location.y = 4.1
ik_target.location.z = 2.2
ik_target.keyframe_insert(data_path='location', frame=10)

bpy.ops.object.empty_add(type='CIRCLE', location=(-4.0, 5.3, 1.3))
pole = bpy.context.object
pole.name = 'IK_Pole'
pole.empty_display_size = 0.2
ik = arm.pose.bones['Link_03'].constraints.new('IK')
ik.name = 'Two-link IK with pole'
ik.target = ik_target
ik.pole_target = pole
ik.chain_count = 2
ik.pole_angle = math.radians(90)

bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.23, depth=3.0, location=tuple(rig_base + Vector((0.0, 0.0, 1.5))))
skin = bpy.context.object
skin.name = 'Rigged_Cylinder_Skin'
skin.data.name = 'RiggedCylinderMesh'
skin.data.materials.append(bright)
arm_mod = skin.modifiers.new('01 | three-link armature skin', 'ARMATURE')
arm_mod.object = arm
groups = [skin.vertex_groups.new(name='Link_%02d' % (i + 1)) for i in range(3)]
for v in skin.data.vertices:
    z = v.co.z + 1.5
    seg = max(0.0, min(2.999, z))
    lo = int(seg)
    frac = seg - lo
    if lo < 2 and frac > 1e-6:
        groups[lo].add([v.index], 1.0 - frac, 'REPLACE')
        groups[lo + 1].add([v.index], frac, 'REPLACE')
    else:
        groups[min(lo, 2)].add([v.index], 1.0, 'REPLACE')

# Animated path and a child marker carrying a half-strength Copy Rotation.
curve_data = bpy.data.curves.new('AnimatedGuidePathData', 'CURVE')
curve_data.dimensions = '3D'
curve_data.resolution_u = 24
curve_data.bevel_depth = 0.035
curve_data.bevel_resolution = 3
spline = curve_data.splines.new('BEZIER')
spline.bezier_points.add(3)
for bp, co in zip(spline.bezier_points, ((-1.0, 4.0, 0.8), (0.4, 4.0, 1.0), (1.8, 4.3, 0.8), (3.4, 4.0, 1.0))):
    bp.co = co
    bp.handle_left_type = 'AUTO'
    bp.handle_right_type = 'AUTO'
curve_data.use_path = True
curve_data.path_duration = 10
curve_data.eval_time = 0.0
path = bpy.data.objects.new('Animated_Follow_Path', curve_data)
scene.collection.objects.link(path)
path.data.materials.append(bright)
curve_data.eval_time = 0.0
curve_data.keyframe_insert(data_path='eval_time', frame=1)
curve_data.eval_time = 10.0
curve_data.keyframe_insert(data_path='eval_time', frame=10)

bpy.ops.object.empty_add(type='CUBE', location=(0.0, 0.0, 0.0))
follower = bpy.context.object
follower.name = 'Path_Follower_Empty'
follower.empty_display_size = 0.22
follow = follower.constraints.new('FOLLOW_PATH')
follow.name = 'Follow animated guide path'
follow.target = path
follow.use_curve_follow = True
follow.forward_axis = 'FORWARD_X'
follow.up_axis = 'UP_Z'
# A compact marker shows the path follower and half-influence rotation constraint.
bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=0.24, location=(0.0, 0.0, 0.0))
marker = bpy.context.object
marker.name = 'Path_Follower_CopyRotation_Marker'
marker.data.name = 'PathMarkerMesh'
marker.data.materials.append(bright)
marker.parent = follower
marker.location = (0.0, 0.0, 0.0)
copy_rot = marker.constraints.new('COPY_ROTATION')
copy_rot.name = 'Half-strength path rotation'
copy_rot.target = follower
copy_rot.influence = 0.5

# Surface Deform is bound between matching grids; an animated shape key raises
# the target beneath the plane during the ten-frame shot.
target_verts = []
target_faces = []
target_cells = 12
for j in range(target_cells + 1):
    y = -1.4 + 2.8 * j / target_cells
    for i in range(target_cells + 1):
        x = -1.4 + 2.8 * i / target_cells
        target_verts.append((x, y, 0.18 - 0.06 * (x*x + y*y)))
for j in range(target_cells):
    for i in range(target_cells):
        a = j * (target_cells + 1) + i
        target_faces.append((a, a + 1, a + target_cells + 2, a + target_cells + 1))
target_mesh = bpy.data.meshes.new('SurfaceTargetMesh')
target_mesh.from_pydata(target_verts, [], target_faces)
target_mesh.materials.append(steel)
target_mesh.update()
surface_target = bpy.data.objects.new('SurfaceDeform_Target', target_mesh)
scene.collection.objects.link(surface_target)
surface_target.location = (5.1, 4.0, 1.18)
basis = surface_target.shape_key_add(name='Basis')
raised = surface_target.shape_key_add(name='RaisedSurface')
for vertex in raised.data:
    x, y = vertex.co.x, vertex.co.y
    vertex.co.z += 0.15 * math.exp(-0.65 * (x*x + y*y))
shape_keys = surface_target.data.shape_keys
raised.value = 0.0
shape_keys.keyframe_insert(data_path='key_blocks["RaisedSurface"].value', frame=1)
raised.value = 1.0
shape_keys.keyframe_insert(data_path='key_blocks["RaisedSurface"].value', frame=10)

# The smaller dense plane grid is bound to the curved target surface.
verts = []
faces = []
rows = cols = 9
for j in range(rows + 1):
    for i in range(cols + 1):
        x = -1.0 + 2.0 * i / cols
        y = -1.0 + 2.0 * j / rows
        verts.append((x, y, 0.04))
for j in range(rows):
    for i in range(cols):
        a = j * (cols + 1) + i
        faces.append((a, a + 1, a + cols + 2, a + cols + 1))
plane_mesh = bpy.data.meshes.new('SurfaceDeformGridMesh')
plane_mesh.from_pydata(verts, [], faces)
plane_mesh.materials.append(ceramic)
plane_mesh.update()
plane = bpy.data.objects.new('Plane_SurfaceDeform_Bound', plane_mesh)
scene.collection.objects.link(plane)
plane.location = (5.1, 4.0, 1.32)
for p in plane.data.polygons:
    p.use_smooth = True
sd = plane.modifiers.new('01 | bound surface deformation', 'SURFACE_DEFORM')
sd.target = surface_target
bpy.ops.object.select_all(action='DESELECT')
plane.select_set(True)
bpy.context.view_layer.objects.active = plane
bpy.context.view_layer.update()
bpy.ops.object.surfacedeform_bind(modifier=sd.name)
if not sd.is_bound:
    raise RuntimeError('Surface Deform modifier did not bind')

# Camera tracks a scene target. A three-point lighting rig keeps the materials
# legible if the importer/exporter preserves lights and camera data.
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(1.0, 2.1, 0.9))
camera_target = bpy.context.object
camera_target.name = 'Camera_Aim'
camera_target.empty_display_size = 0.2
cam_data = bpy.data.cameras.new('InspectionCameraData')
cam = bpy.data.objects.new('Inspection_Camera_TrackTo', cam_data)
scene.collection.objects.link(cam)
cam.location = (11.5, -17.5, 13.0)
cam.data.lens = 48
cam.data.dof.use_dof = False
track = cam.constraints.new('TRACK_TO')
track.name = 'Track camera to assembly'
track.target = camera_target
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'
scene.camera = cam

def area_light(name, loc, energy, size, color, target=(1.0, 2.0, 0.6)):
    data = bpy.data.lights.new(name + 'Data', 'AREA')
    data.energy = energy
    data.shape = 'DISK'
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = loc
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    return obj

area_light('Key_Softbox', (0.0, -7.0, 13.0), 1800, 9.0, (0.80, 0.88, 1.0))
area_light('Warm_Rim', (7.0, 8.0, 10.0), 1300, 7.0, (1.0, 0.58, 0.31))
area_light('Cool_Fill', (-8.0, 1.0, 6.0), 900, 6.0, (0.34, 0.58, 1.0))

# Force a deterministic view-layer update and write the source artifact.
scene.frame_set(1)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(BLEND))

# Capture source or reopened evaluated state using the shared dump helper.
exec(Path(__file__).with_name('blend_realistic_scene_dump.py').read_text())
