#![recursion_limit = "512"]
#![expect(
    clippy::unwrap_used,
    reason = "Blender adapter fixtures use fixed valid paths and values"
)]

use glam::DVec3;
use serde_json::{Value, json};
use std::process::Command;
use std::{collections::HashMap, error::Error, fs, path::Path};

use tempfile::tempdir;

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/blender_script_guarded.rs"]
mod blender_script_guarded;
#[path = "common/pot_json_guarded_escaped.rs"]
mod pot_json_guarded_escaped;
#[path = "common/process.rs"]
mod process;
#[path = "common/process_escaped.rs"]
mod process_escaped;

use blender_checked::blender_executable as blender;
use blender_script_guarded::run_blender_script;
use pot_json_guarded_escaped::pot_json_escaped_newlines as pot_json;
use process_escaped::run_guarded_escaped_newlines as run_guarded;

const BLENDER_FIXTURE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_set(1)
mesh = bpy.data.meshes.new("AdapterMesh")
mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
mesh.update()
owner = bpy.data.objects.new("AdapterOwner", mesh)
scene.collection.objects.link(owner)
target = bpy.data.objects.new("AdapterTarget", None)
scene.collection.objects.link(target)
target.location = (2.0, 0.25, -0.5)
owner.location = (1.0, 0.0, 0.0)
modifier = owner.modifiers.new("NonDefaultMirror", "MIRROR")
modifier.use_axis = (False, True, False)
modifier.use_clip = False
modifier.use_mirror_merge = False
modifier.merge_threshold = 0.003
constraint = owner.constraints.new("COPY_LOCATION")
constraint.name = "NonDefaultCopyLocation"
constraint.target = target
constraint.use_x = True
constraint.use_y = False
constraint.use_z = False
constraint.influence = 0.5
clip_image = bpy.data.images.new("TrackingFrame", width=64, height=64)
clip_image.filepath_raw = os.path.join(root, "tracking.png")
clip_image.file_format = "PNG"
clip_image.save()
clip = bpy.data.movieclips.load(clip_image.filepath_raw)
track = clip.tracking.tracks.new(name="TrackA")
marker = track.markers[0]
marker.co = (0.25, 0.5)
marker.pattern_corners = ((0.2, 0.45), (0.3, 0.45), (0.3, 0.55), (0.2, 0.55))
scene.active_clip = clip
follow = owner.constraints.new("FOLLOW_TRACK")
follow.name = "MutedFollowTrack"
follow.clip = clip
follow.track = "TrackA"
follow.mute = True
follow.influence = 0.6
solver = owner.constraints.new("OBJECT_SOLVER")
solver.name = "MutedObjectSolver"
solver.clip = clip
solver.object = "Camera"
solver.mute = True
solver.influence = 0.7

def sample():
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = owner.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    positions = sorted([[float(v.co.x), float(v.co.y), float(v.co.z)] for v in mesh.vertices])
    matrix = [float(evaluated.matrix_world[row][column])
              for column in range(4) for row in range(4)]
    evaluated.to_mesh_clear()
    return {"positions": positions, "matrix": matrix}

with open(os.path.join(root, "blender_before.json"), "w", encoding="utf-8") as handle:
    json.dump(sample(), handle)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"))
"#;

const BLENDER_REOPEN: &str = r#"
import bpy
import json
import sys

output = sys.argv[sys.argv.index("--") + 1]
owner = bpy.data.objects["AdapterOwner"]
modifier = next(m for m in owner.modifiers if m.name == "NonDefaultMirror")
constraint = next(c for c in owner.constraints if c.name == "NonDefaultCopyLocation")
scene = bpy.context.scene
clip = scene.active_clip
camera_object = clip.tracking.objects.get("Camera") if clip else None
track = camera_object.tracks.get("TrackA") if camera_object else None
bpy.context.scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = owner.evaluated_get(depsgraph)
mesh = evaluated.to_mesh()
positions = sorted([[float(v.co.x), float(v.co.y), float(v.co.z)] for v in mesh.vertices])
matrix = [float(evaluated.matrix_world[row][column])
          for column in range(4) for row in range(4)]
evaluated.to_mesh_clear()
with open(output, "w", encoding="utf-8") as handle:
    json.dump({
        "modifier_type": modifier.type,
        "use_axis": list(modifier.use_axis),
        "use_mirror_merge": modifier.use_mirror_merge,
        "merge_threshold": modifier.merge_threshold,
        "constraint_type": constraint.type,
        "target": constraint.target.name,
        "use_x": constraint.use_x,
        "use_y": constraint.use_y,
        "use_z": constraint.use_z,
        "influence": constraint.influence,
        "active_clip": clip.name if clip else None,
        "tracking_marker": list(track.markers.find_frame(1).co) if track else None,
        "positions": positions,
        "matrix": matrix,
    }, handle)
"#;

fn assert_vec3_cloud(actual: &[Value], expected: &[Value]) {
    fn points(values: &[Value]) -> Vec<[f64; 3]> {
        let mut points = values
            .iter()
            .map(|point| {
                [
                    point[0].as_f64().unwrap(),
                    point[1].as_f64().unwrap(),
                    point[2].as_f64().unwrap(),
                ]
            })
            .collect::<Vec<_>>();
        points.sort_by(|left, right| {
            left[0]
                .total_cmp(&right[0])
                .then(left[1].total_cmp(&right[1]))
                .then(left[2].total_cmp(&right[2]))
        });
        points
    }
    let actual = points(actual);
    let expected = points(expected);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        for axis in 0..3 {
            let delta = actual[axis] - expected[axis];
            assert!(
                delta.abs() <= 1.0e-5,
                "vertex axis {axis} differs by {delta}"
            );
        }
    }
}

fn assert_matrices(actual: &[Value], expected: &[Value]) {
    for (actual, expected) in actual.iter().zip(expected) {
        let delta = actual.as_f64().unwrap() - expected.as_f64().unwrap();
        assert!(delta.abs() <= 2.0e-4, "matrix component differs by {delta}");
    }
}
fn assert_matrix_parity(actual: &[Value], expected: &[Value], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: matrix size");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let delta = actual.as_f64().unwrap() - expected.as_f64().unwrap();
        assert!(
            delta.abs() <= 1.0e-6,
            "{context}: matrix component {index} differs by {delta}; actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn blender_modifier_and_constraint_import_inspect_export_round_trip() -> Result<(), Box<dyn Error>>
{
    let Some(blender) = blender() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_fixture.py",
        BLENDER_FIXTURE,
        root,
        &[],
        "Blender fixture failed",
        run_guarded,
    )?;
    let before: Value = serde_json::from_slice(&fs::read(root.join("blender_before.json"))?)?;
    let project = root.join("project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert!(
        imported["result"]["losses"].as_array().unwrap().is_empty(),
        "{imported}"
    );
    let mappings = imported["result"]["id_mappings"].as_object().unwrap();
    let owner_id = mappings["Object:AdapterOwner"].as_str().unwrap();
    let target_id = mappings["Object:AdapterTarget"].as_str().unwrap();
    let inspected = pot_json(&["inspect", project.to_str().unwrap(), "--id", owner_id])?;
    let owner = inspected["result"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["id"] == owner_id)
        .unwrap();
    let modifier = &owner["modifiers"][0];
    assert_eq!(modifier["type"], "mirror");
    assert_eq!(modifier["params"]["use_axis"], json!([false, true, false]));
    assert_eq!(modifier["params"]["use_mirror_merge"], false);
    assert!((modifier["params"]["merge_threshold"].as_f64().unwrap() - 0.003).abs() < 1.0e-6);
    let constraint = &owner["constraints"][0];
    assert_eq!(constraint["type"], "copy_location");
    assert_eq!(constraint["target"], target_id);
    assert_eq!(constraint["influence"], 0.5);
    assert_eq!(constraint["params"]["use_x"], true);
    assert_eq!(constraint["params"]["use_y"], false);

    assert_vec3_cloud(
        owner["evaluated_geometry"]["positions"].as_array().unwrap(),
        before["positions"].as_array().unwrap(),
    );
    assert_matrices(
        owner["transform"]["world"]["matrix"].as_array().unwrap(),
        before["matrix"].as_array().unwrap(),
    );

    let exported = root.join("roundtrip.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let reopened_path = root.join("blender_after.json");
    let script_path = root.join("reopen.py");
    fs::write(&script_path, BLENDER_REOPEN)?;
    let mut command = Command::new(&blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&script_path)
        .arg("--")
        .arg(&reopened_path);
    let output = run_guarded(command)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "Blender reopen failed: stdout={stdout} stderr={stderr}",
    );
    let after: Value = serde_json::from_slice(&fs::read(reopened_path)?)?;
    assert_eq!(after["modifier_type"], "MIRROR");
    assert_eq!(after["use_axis"], json!([false, true, false]));
    assert_eq!(after["use_mirror_merge"], false);
    assert!((after["merge_threshold"].as_f64().unwrap() - 0.003).abs() < 1.0e-6);
    assert_eq!(after["constraint_type"], "COPY_LOCATION");
    assert_eq!(after["target"], "AdapterTarget");
    assert_eq!(after["use_x"], true);
    assert_eq!(after["use_y"], false);
    assert_eq!(after["influence"], 0.5);
    assert_vec3_cloud(
        after["positions"].as_array().unwrap(),
        before["positions"].as_array().unwrap(),
    );
    assert_matrices(
        after["matrix"].as_array().unwrap(),
        before["matrix"].as_array().unwrap(),
    );
    Ok(())
}

const MODIFIER_FIXTURE: &str = r#"
import bpy
import json
import os
import sys
import math

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
cases = json.load(open(os.path.join(root, "modifier_cases.json"), encoding="utf-8"))
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

cube_vertices = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]
cube_faces = [
    (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
]

def mesh_object(name, vertices=cube_vertices, edges=(), faces=cube_faces):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj

def empty_object(name, location):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    obj.location = location
    return obj

case_types = {case["type"] for case in cases}
if "MESH_TO_VOLUME" in case_types:
    mesh_object("MeshToVolumeOperand")
if "BOOLEAN" in case_types:
    boolean_operand = mesh_object("BooleanOperand")
    # Coplanar contacts made Blender's FLOAT solver return an empty mesh for this fixture.
    boolean_operand.location = (0.55, 0.23, 0.19)
    boolean_collection = bpy.data.collections.new("BooleanOperands")
    scene.collection.children.link(boolean_collection)
    boolean_collection.objects.link(boolean_operand)
if "SHRINKWRAP" in case_types:
    shrink_target = mesh_object(
        "ShrinkTarget",
        [(-3, -3, 0.4), (3, -3, 0.4), (3, 3, 0.4), (-3, 3, 0.4)],
        faces=[(0, 1, 2, 3)])
if "CURVE" in case_types:
    curve_data = bpy.data.curves.new("CurveTargetData", "CURVE")
    curve_data.dimensions = "3D"
    curve_spline = curve_data.splines.new("POLY")
    curve_spline.points.add(1)
    curve_spline.points[0].co = (0, 0, 0, 1)
    curve_spline.points[1].co = (3, 0.5, 0, 1)
    curve_target = bpy.data.objects.new("CurveTarget", curve_data)
    scene.collection.objects.link(curve_target)
if "ARRAY" in case_types:
    mesh_object("ArrayStartCap")
    mesh_object("ArrayEndCap")
if "HOOK" in case_types:
    hook_target = empty_object("HookTarget", (0.3, 0.1, 0.2))
if "WARP" in case_types:
    warp_from = empty_object("WarpFrom", (-0.5, 0.0, 0.0))
    warp_to = empty_object("WarpTo", (0.8, 0.4, 0.0))
if "ARMATURE" in case_types:
    armature_data = bpy.data.armatures.new("ArmatureTargetData")
    armature_target = bpy.data.objects.new("ArmatureTarget", armature_data)
    scene.collection.objects.link(armature_target)
    bpy.context.view_layer.objects.active = armature_target
    armature_target.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bone = armature_data.edit_bones.new("Bone")
    bone.head = (0, 0, -1)
    bone.tail = (0, 0, 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    armature_target.pose.bones["Bone"].location = (0.0, 0.25, 0.0)
    armature_target.select_set(False)
if "NODES" in case_types:
    group = bpy.data.node_groups.new("TransformGroup", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    mesh_cube = group.nodes.new("GeometryNodeMeshCube")
    mesh_cube.inputs["Size"].default_value = (1.25, 1.5, 1.75)
    transform = group.nodes.new("GeometryNodeTransform")
    transform.inputs["Translation"].default_value = (0.3, -0.15, 0.2)
    group_output = group.nodes.new("NodeGroupOutput")
    group.links.new(mesh_cube.outputs["Mesh"], transform.inputs["Geometry"])
    group.links.new(transform.outputs["Geometry"], group_output.inputs["Geometry"])

def set_group(obj, name, weights):
    group = obj.vertex_groups.new(name=name)
    for index, weight in enumerate(weights):
        if weight:
            group.add([index], float(weight), "REPLACE")

def resolve(value):
    if isinstance(value, str) and value.startswith("@object:"):
        return bpy.data.objects[value[len("@object:") :]]
    if isinstance(value, str) and value.startswith("@collection:"):
        return bpy.data.collections[value[len("@collection:") :]]
    if isinstance(value, str) and value.startswith("@group:"):
        return bpy.data.node_groups[value[len("@group:") :]]
    return value

def sample(obj):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    if obj.type == "VOLUME":
        corners = [tuple(float(value) for value in point[:3]) for point in evaluated.bound_box]
        grids = list(evaluated.data.grids)
        voxel_size = float(grids[0].matrix_object[0][0]) if grids else 0.0
        dimensions = [
            int(round((max(point[axis] for point in corners)
                       - min(point[axis] for point in corners)) / voxel_size))
            for axis in range(3)
        ] if voxel_size > 0.0 else []
        density_samples = []
        modifier = next((item for item in obj.modifiers if item.type == "MESH_TO_VOLUME"), None)
        if modifier and grids and modifier.object:
            import openvdb
            import numpy as np
            operand = modifier.object
            operand_mesh = operand.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
            operand_mesh.calc_loop_triangles()
            operand_to_volume = evaluated.matrix_world.inverted() @ operand.matrix_world
            points = np.asarray([tuple(operand_to_volume @ vertex.co)
                                 for vertex in operand_mesh.vertices], dtype=np.float32)
            triangles = np.asarray([tuple(triangle.vertices)
                                    for triangle in operand_mesh.loop_triangles], dtype=np.uint32)
            level_set = openvdb.FloatGrid.createLevelSetFromPolygons(
                points, triangles,
                transform=openvdb.createLinearTransform(voxel_size),
                exBandWidth=1.0,
                inBandWidth=max(1.0, float(modifier.interior_band_width) / voxel_size))
            accessor = level_set.getAccessor()
            cutoff = max(voxel_size, float(modifier.interior_band_width))
            for point in ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)):
                coordinate = tuple(int(round(value / voxel_size)) for value in point)
                signed_distance = float(accessor.getValue(coordinate))
                density_samples.append(
                    max(0.0, min(1.0, -signed_distance / cutoff)) * float(modifier.density))
            operand.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
        return {
            "volume_grid_names": [grid.name for grid in grids],
            "volume_grid_types": [grid.data_type for grid in grids],
            "volume_grid_channels": [int(grid.channels) for grid in grids],
            "volume_grid_voxel_size": voxel_size,
            "volume_grid_dimensions": dimensions,
            "volume_density_samples": density_samples,
            "volume_bounds": [
                [min(point[axis] for point in corners) for axis in range(3)],
                [max(point[axis] for point in corners) for axis in range(3)],
            ],
        }
    mesh = evaluated.to_mesh()
    geometry = {
        "vertex_count": len(mesh.vertices),
        "edge_count": len(mesh.edges),
        "face_count": len(mesh.polygons),
        "positions": [[float(v.co.x), float(v.co.y), float(v.co.z)] for v in mesh.vertices],
        "faces": [list(polygon.vertices) for polygon in mesh.polygons],
        "uv_layers": [
            {"name": layer.name,
             "values": [[float(item.uv.x), float(item.uv.y)] for item in layer.data]}
            for layer in mesh.uv_layers
        ],
        "matrix": [float(evaluated.matrix_world[row][column])
                   for column in range(4) for row in range(4)],
    }
    evaluated.to_mesh_clear()
    return geometry

owners = {}
for case in cases:
    if case.get("unsupported"):
        continue
    name = case["name"]
    if case["type"] == "VOLUME_TO_MESH":
        owner = mesh_object(name)
    elif case["type"] == "MESH_TO_VOLUME":
        volume = bpy.data.volumes.new(name + "Volume")
        owner = bpy.data.objects.new(name, volume)
        scene.collection.objects.link(owner)
    elif case["type"] == "ARRAY":
        owner = mesh_object(name)
        if case["name"] == "CaseArrayFitCurve":
            curve_data = bpy.data.curves.new("ArrayFitCurveData", "CURVE")
            curve_data.dimensions = "3D"
            curve_data.use_path = True
            curve_spline = curve_data.splines.new("POLY")
            curve_spline.points.add(1)
            curve_spline.points[0].co = (0, 0, 0, 1)
            curve_spline.points[1].co = (2.5, 0, 0, 1)
            fit_curve = bpy.data.objects.new("ArrayFitCurve", curve_data)
            scene.collection.objects.link(fit_curve)
            fit_curve.scale = (1.5, 1.5, 1.5)
        offset_object = bpy.data.objects.new("ArrayOffset", None)
        scene.collection.objects.link(offset_object)
        offset_object.location = (0.5, 0.0, 0.0)
        offset_object.rotation_euler = (0.0, 0.0, 0.35)
    elif case["type"] == "SOLIDIFY":
        owner = mesh_object(name, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                            faces=[(0, 1, 2, 3)])
    elif case["type"] == "SCREW":
        owner = mesh_object(name, [(1, 0, -1), (1, 0, 1)], [(0, 1)], [])
    elif case["type"] == "SKIN":
        if case["name"].startswith("CaseSkinBranch"):
            owner = mesh_object(
                name,
                [(0, 0, 0), (1.2, 0, 0), (0, 1.2, 0), (0, 0, 1.2)],
                [(0, 1), (0, 2), (0, 3)],
                [])
        else:
            owner = mesh_object(name, [(-0.5, 0, -1), (0.5, 0, 1)], [(0, 1)], [])
    elif case["type"] == "CURVE":
        curve_vertices = [
            (0, -1, -1), (3, -1, -1), (3, 1, -1), (0, 1, -1),
            (0, -1, 1), (3, -1, 1), (3, 1, 1), (0, 1, 1),
        ]
        owner = mesh_object(name, curve_vertices)
    elif case["type"] == "DECIMATE" and case.get("shape") == "grid":
        grid_size = case.get("grid_size", 9)
        grid_vertices = []
        grid_faces = []
        for y_index in range(grid_size):
            y = -1.0 + 2.0 * y_index / (grid_size - 1)
            for x_index in range(grid_size):
                x = -1.0 + 2.0 * x_index / (grid_size - 1)
                z = (0.0 if case.get("flat") else
                     0.12 * math.cos(x * 2.0) * math.sin(y * 2.0))
                grid_vertices.append((x, y, z))
        for y_index in range(grid_size - 1):
            for x_index in range(grid_size - 1):
                first = y_index * grid_size + x_index
                grid_faces.append((first, first + 1, first + grid_size + 1, first + grid_size))
        owner = mesh_object(name, grid_vertices, faces=grid_faces)
    elif case["type"] == "MASK":
        mask_vertices = cube_vertices + [
            (x + 3, y, z) for x, y, z in cube_vertices]
        mask_faces = cube_faces + [
            tuple(index + 8 for index in face) for face in cube_faces]
        owner = mesh_object(name, mask_vertices, faces=mask_faces)
    else:
        owner = mesh_object(name)
    if case["type"] in ("ARRAY", "DISPLACE"):
        owner.location = (1.3, -0.4, 0.7)
        owner.rotation_mode = "XYZ"
        owner.rotation_euler = (0.0, 0.0, math.pi / 2.0)
        owner.scale = (1.2, 0.8, 1.1)
    if case["name"] == "CaseArrayUV":
        uv_layer = owner.data.uv_layers.new(name="ArrayUV")
        for index, item in enumerate(uv_layer.data):
            item.uv = (float(index) * 0.125, float(index) * -0.25)
    if case["type"] == "DECIMATE" and case.get("shape") == "grid":
        grid_size = case.get("grid_size", 9)
        delimit = case["props"].get("delimit", [])
        if "MATERIAL" in delimit:
            owner.data.materials.append(bpy.data.materials.new("DecimateA"))
            owner.data.materials.append(bpy.data.materials.new("DecimateB"))
            for polygon in owner.data.polygons:
                polygon.material_index = polygon.index % 2
        if "SEAM" in delimit or "SHARP" in delimit:
            center = grid_size // 2
            for edge in owner.data.edges:
                first, second = edge.vertices
                if first % grid_size == center and second % grid_size == center:
                    if "SEAM" in delimit:
                        edge.use_seam = True
                    if "SHARP" in delimit:
                        edge.use_edge_sharp = True
        if "UV" in delimit:
            uv_layer = owner.data.uv_layers.new(name="DecimateUV")
            for polygon in owner.data.polygons:
                offset = 4.0 if sum(owner.data.vertices[index].co.x
                                    for index in polygon.vertices) >= 0.0 else 0.0
                for loop_index in polygon.loop_indices:
                    vertex = owner.data.vertices[owner.data.loops[loop_index].vertex_index]
                    uv_layer.data[loop_index].uv = (vertex.co.x + 1.0 + offset,
                                                    vertex.co.y + 1.0)
    owners[name] = owner

    if case["type"] == "MASK":
        set_group(owner, case["group"], [0.25 if i < 8 else 0.8
                                         for i in range(len(owner.data.vertices))])
    elif case["type"] in ("HOOK", "LATTICE", "DISPLACE", "SMOOTH", "VERTEX_WEIGHT_EDIT"):
        set_group(owner, case["group"], [0.25 if i == 0 else 0.8
                                         for i in range(len(owner.data.vertices))])
    elif case["type"] == "DECIMATE" and case.get("group"):
        set_group(owner, case["group"],
                  [0.0 if vertex.co.x < 0.0 else 1.0 for vertex in owner.data.vertices])
    if case["type"] == "ARMATURE":
        set_group(owner, "ArmatureWeights", [1.0] * len(owner.data.vertices))

    modifier = owner.modifiers.new("Modifier_" + case["type"], case["type"])
    for key, value in case["props"].items():
        if key == "map_curve":
            curve = modifier.map_curve.curves[0]
            for point in list(curve.points)[1:-1]:
                curve.points.remove(point)
            curve.points[0].location = value[0]
            curve.points[-1].location = value[-1]
            for x, y in value[1:-1]:
                curve.points.new(x, y)
            modifier.map_curve.update()
        else:
            setattr(modifier, key, set(value) if key == "delimit" else resolve(value))
    if case["type"] == "HOOK":
        modifier.vertex_indices_set(list(range(len(owner.data.vertices))))
    if case["type"] == "SKIN":
        skin_data = owner.data.skin_vertices[0].data
        if case["name"].startswith("CaseSkinBranch"):
            skin_data[0].radius = (0.3, 0.2)
            skin_data[0].use_root = True
            skin_data[1].radius = (0.25, 0.35)
            skin_data[2].radius = (0.2, 0.3)
            skin_data[3].radius = (0.35, 0.25)
        else:
            skin_data[0].radius = (0.3, 0.2)
            skin_data[0].use_root = True
            skin_data[1].radius = (0.25, 0.35)

scene.frame_set(5)
bpy.context.view_layer.update()
before = {name: sample(obj) for name, obj in owners.items()}
with open(os.path.join(root, "modifier_before.json"), "w", encoding="utf-8") as handle:
    json.dump(before, handle)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "modifier_source.blend"))
"#;

const TRANSFORM_FIXTURE: &str = r#"
import bpy
import json
import math
import os
import sys
from mathutils import Quaternion

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

parent = bpy.data.objects.new("A_TransformParent", None)
scene.collection.objects.link(parent)
parent.location = (-1.1, 0.6, 0.4)
parent.rotation_euler = (0.25, -0.4, 0.3)
parent.scale = (0.8, 1.3, 1.1)

def make_mesh(name):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(
        [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj

cases = [
    ("EulerRoot", "XYZ", False, False),
    ("EulerChild", "XYZ", True, True),
    ("EulerXZY", "XZY", False, True),
    ("EulerYXZ", "YXZ", True, False),
    ("EulerYZX", "YZX", False, False),
    ("EulerZXY", "ZXY", True, True),
    ("EulerZYX", "ZYX", False, True),
    ("QuaternionRoot", "QUATERNION", False, True),
    ("QuaternionChild", "QUATERNION", True, False),
    ("AxisAngleRoot", "AXIS_ANGLE", False, False),
    ("AxisAngleChild", "AXIS_ANGLE", True, True),
]
objects = {}
for name, mode, use_parent, use_modifier in cases:
    obj = make_mesh(name)
    obj.location = (0.35, -0.7, 1.2)
    obj.scale = (1.4, 0.65, 1.1)
    obj.rotation_mode = mode
    if mode in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
        obj.rotation_euler = (0.37, -0.21, math.pi / 2.0)
    elif mode == "QUATERNION":
        obj.rotation_quaternion = Quaternion((1.0, 2.0, -1.0), 0.91)
    else:
        obj.rotation_axis_angle = (1.17, 1.0 / math.sqrt(5.0), 2.0 / math.sqrt(5.0), 0.0)
    if use_parent:
        obj.parent = parent
    if use_modifier:
        obj.modifiers.new("TransformProbe", "SUBSURF")
    objects[name] = obj

bpy.context.view_layer.update()
def matrix_values(matrix):
    return [float(matrix[row][column])
            for column in range(4) for row in range(4)]

expected = {
    name: {
        "local": matrix_values(obj.matrix_basis),
        "world": matrix_values(obj.matrix_world),
    }
    for name, obj in objects.items()
}
expected["A_TransformParent"] = {
    "local": matrix_values(parent.matrix_basis),
    "world": matrix_values(parent.matrix_world),
}
with open(os.path.join(root, "transform_expected.json"), "w", encoding="utf-8") as handle:
    json.dump(expected, handle)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "transform_source.blend"))
"#;

const MODIFIER_REOPEN: &str = r#"
import bpy
import json
import sys

output = sys.argv[sys.argv.index("--") + 1]
cases_path = sys.argv[sys.argv.index("--") + 2]
cases = json.load(open(cases_path, encoding="utf-8"))
bpy.context.scene.frame_set(5)
result = {}
for case in cases:
    if case.get("unsupported"):
        continue
    owner = bpy.data.objects[case["name"]]
    modifier = next(m for m in owner.modifiers if m.name == "Modifier_" + case["type"])
    props = {}
    for key in case["props"]:
        if key == "map_curve":
            curve = modifier.map_curve.curves[0]
            props[key] = [
                [float(point.location[0]), float(point.location[1])]
                for point in curve.points
            ]
            continue
        value = getattr(modifier, key)
        if isinstance(value, bpy.types.ID):
            props[key] = value.name_full
        elif hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            props[key] = list(value)
        else:
            props[key] = value
    bpy.context.view_layer.update()
    evaluated = owner.evaluated_get(bpy.context.evaluated_depsgraph_get())
    if owner.type == "VOLUME":
        corners = [tuple(float(value) for value in point[:3]) for point in evaluated.bound_box]
        grids = list(evaluated.data.grids)
        voxel_size = float(grids[0].matrix_object[0][0]) if grids else 0.0
        dimensions = [
            int(round((max(point[axis] for point in corners)
                       - min(point[axis] for point in corners)) / voxel_size))
            for axis in range(3)
        ] if voxel_size > 0.0 else []
        density_samples = []
        if modifier.type == "MESH_TO_VOLUME" and grids and modifier.object:
            import openvdb
            import numpy as np
            operand = modifier.object
            operand_mesh = operand.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
            operand_mesh.calc_loop_triangles()
            operand_to_volume = evaluated.matrix_world.inverted() @ operand.matrix_world
            points = np.asarray([tuple(operand_to_volume @ vertex.co)
                                 for vertex in operand_mesh.vertices], dtype=np.float32)
            triangles = np.asarray([tuple(triangle.vertices)
                                    for triangle in operand_mesh.loop_triangles], dtype=np.uint32)
            level_set = openvdb.FloatGrid.createLevelSetFromPolygons(
                points, triangles,
                transform=openvdb.createLinearTransform(voxel_size),
                exBandWidth=1.0,
                inBandWidth=max(1.0, float(modifier.interior_band_width) / voxel_size))
            accessor = level_set.getAccessor()
            cutoff = max(voxel_size, float(modifier.interior_band_width))
            for point in ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)):
                coordinate = tuple(int(round(value / voxel_size)) for value in point)
                signed_distance = float(accessor.getValue(coordinate))
                density_samples.append(
                    max(0.0, min(1.0, -signed_distance / cutoff)) * float(modifier.density))
            operand.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
        geometry = {
            "volume_grid_names": [grid.name for grid in grids],
            "volume_grid_types": [grid.data_type for grid in grids],
            "volume_grid_channels": [int(grid.channels) for grid in grids],
            "volume_grid_voxel_size": voxel_size,
            "volume_grid_dimensions": dimensions,
            "volume_density_samples": density_samples,
            "volume_bounds": [
                [min(point[axis] for point in corners) for axis in range(3)],
                [max(point[axis] for point in corners) for axis in range(3)],
            ],
        }
    else:
        mesh = evaluated.to_mesh()
        geometry = {
            "vertex_count": len(mesh.vertices),
            "edge_count": len(mesh.edges),
            "face_count": len(mesh.polygons),
            "positions": [[float(v.co.x), float(v.co.y), float(v.co.z)] for v in mesh.vertices],
            "faces": [list(polygon.vertices) for polygon in mesh.polygons],
            "uv_layers": [
                {"name": layer.name,
                 "values": [[float(item.uv.x), float(item.uv.y)] for item in layer.data]}
                for layer in mesh.uv_layers
            ],
        }
        evaluated.to_mesh_clear()
    result[case["name"]] = {"type": modifier.type, "props": props, **geometry}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"#;

#[expect(
    clippy::approx_constant,
    reason = "Blender fixture uses the exact 3.14 threshold rather than pi"
)]
fn modifier_cases() -> Value {
    json!([
        {"name":"CaseMirror","type":"MIRROR","props":{"use_axis":[false,true,false],"use_mirror_merge":false,"merge_threshold":0.003},"expected":{"use_axis":[false,true,false],"use_mirror_merge":false,"merge_threshold":0.003}},
        {"name":"CaseArray","type":"ARRAY","props":{"count":3,"fit_type":"FIT_LENGTH","fit_length":4.3,"use_relative_offset":true,"relative_offset_displace":[0.25,1.0,0.0],"use_constant_offset":true,"constant_offset_displace":[-1.0,0.0,0.0],"use_object_offset":true,"offset_object":"@object:ArrayOffset","use_merge_vertices":true,"merge_threshold":0.01},"expected":{"count":3,"fit_type":"FIT_LENGTH","fit_length":4.3,"relative_offset_displace":[0.25,1.0,0.0],"constant_offset_displace":[-1.0,0.0,0.0],"use_relative_offset":true,"use_constant_offset":true,"use_object_offset":true,"offset_object":"@object:ArrayOffset","use_merge_vertices":true,"merge_threshold":0.01}},
        {"name":"CaseArrayCaps","type":"ARRAY","props":{"count":3,"fit_type":"FIXED_COUNT","use_relative_offset":true,"relative_offset_displace":[1.0,0.0,0.0],"start_cap":"@object:ArrayStartCap","end_cap":"@object:ArrayEndCap","use_merge_vertices":true,"use_merge_vertices_cap":true,"merge_threshold":0.001},"expected":{"fit_type":"FIXED_COUNT","relative_offset_displace":[1.0,0.0,0.0],"start_cap":"@object:ArrayStartCap","end_cap":"@object:ArrayEndCap","use_merge_vertices":true,"use_merge_vertices_cap":true,"merge_threshold":0.001}},
        {"name":"CaseArrayFitCurve","type":"ARRAY","props":{"count":4,"fit_type":"FIT_CURVE","curve":"@object:ArrayFitCurve","use_relative_offset":true,"relative_offset_displace":[1.0,0.0,0.0]},"expected":{"count":4,"fit_type":"FIT_CURVE","curve":"@object:ArrayFitCurve","relative_offset_displace":[1.0,0.0,0.0]}},
        {"name":"CaseArrayUV","type":"ARRAY","props":{"count":3,"fit_type":"FIXED_COUNT","use_relative_offset":true,"relative_offset_displace":[1.0,0.0,0.0],"offset_u":0.25,"offset_v":-0.5},"expected":{"fit_type":"FIXED_COUNT","relative_offset_displace":[1.0,0.0,0.0],"offset_u":0.25,"offset_v":-0.5}},
        {"name":"CaseSubdivision","type":"SUBSURF","props":{"levels":2},"expected":{"levels":2}},
        {"name":"CaseMultires","type":"MULTIRES","props":{"levels":0,"quality":5},"expected":{"levels":0,"quality":5}},
        {"name":"CaseSolidify","type":"SOLIDIFY","props":{"thickness":0.15,"offset":0.25,"use_even_offset":true,"use_quality_normals":true,"use_rim":true,"use_rim_only":true,"use_flip_normals":true},"expected":{"thickness":0.15,"offset":0.25,"use_even_offset":true,"use_quality_normals":true,"use_rim":true,"use_rim_only":true,"use_flip_normals":true}},
        {"name":"CaseTriangulate","type":"TRIANGULATE","props":{"quad_method":"FIXED"},"expected":{"quad_method":"FIXED"}},
        {"name":"CaseBevel","type":"BEVEL","props":{"width":0.1,"segments":2},"expected":{"width":0.1,"segments":2}},
        {"name":"CaseBevelVertices","type":"BEVEL","props":{"width":0.1,"segments":2,"affect":"VERTICES"},"expected":{"width":0.1,"segments":2,"affect":"VERTICES"}},
        {"name":"CaseBevelWidth","type":"BEVEL","props":{"width":0.1,"segments":2,"offset_type":"WIDTH","limit_method":"ANGLE","angle_limit":0.5,"profile":0.5,"use_clamp_overlap":false,"loop_slide":true,"harden_normals":true},"expected":{"width":0.1,"segments":2,"offset_type":"WIDTH","limit_method":"ANGLE","angle_limit":0.5,"profile":0.5,"use_clamp_overlap":false,"loop_slide":true,"harden_normals":true}},
        {"name":"CaseDecimate","type":"DECIMATE","props":{"ratio":0.65},"expected":{"ratio":0.65}},
        {"name":"CaseDecimateUnsubdiv","type":"DECIMATE","shape":"grid","props":{"decimate_type":"UNSUBDIV","iterations":1},"expected":{"decimate_type":"UNSUBDIV","iterations":1}},
        {"name":"CaseDecimateUnsubdivIterations","type":"DECIMATE","shape":"grid","props":{"decimate_type":"UNSUBDIV","iterations":2},"expected":{"decimate_type":"UNSUBDIV","iterations":2}},
        {"name":"CaseDecimateDissolve","type":"DECIMATE","shape":"grid","props":{"decimate_type":"DISSOLVE","angle_limit":0.5},"expected":{"decimate_type":"DISSOLVE","angle_limit":0.5}},
        {"name":"CaseDecimateDissolveNormal","type":"DECIMATE","shape":"grid","flat":true,"props":{"decimate_type":"DISSOLVE","angle_limit":3.14,"delimit":["NORMAL"]},"expected":{"decimate_type":"DISSOLVE","angle_limit":3.14,"delimit":["NORMAL"]}},
        {"name":"CaseDecimateDissolveMaterial","type":"DECIMATE","shape":"grid","props":{"decimate_type":"DISSOLVE","angle_limit":3.14,"delimit":["MATERIAL"]},"expected":{"decimate_type":"DISSOLVE","angle_limit":3.14,"delimit":["MATERIAL"]}},
        {"name":"CaseDecimateDissolveSeam","type":"DECIMATE","shape":"grid","props":{"decimate_type":"DISSOLVE","angle_limit":3.14,"delimit":["SEAM"]},"expected":{"decimate_type":"DISSOLVE","angle_limit":3.14,"delimit":["SEAM"]}},
        {"name":"CaseDecimateDissolveSharp","type":"DECIMATE","shape":"grid","props":{"decimate_type":"DISSOLVE","angle_limit":3.14,"delimit":["SHARP"]},"expected":{"decimate_type":"DISSOLVE","angle_limit":3.14,"delimit":["SHARP"]}},
        {"name":"CaseDecimateDissolveUV","type":"DECIMATE","shape":"grid","props":{"decimate_type":"DISSOLVE","angle_limit":3.14,"delimit":["UV"]},"expected":{"decimate_type":"DISSOLVE","angle_limit":3.14,"delimit":["UV"]}},
        {"name":"CaseDecimateDissolveBoundaries","type":"DECIMATE","shape":"grid","props":{"decimate_type":"DISSOLVE","angle_limit":0.5,"use_dissolve_boundaries":true},"expected":{"decimate_type":"DISSOLVE","angle_limit":0.5,"use_dissolve_boundaries":true}},
        {"name":"CaseDecimateSymmetry","type":"DECIMATE","shape":"grid","props":{"ratio":0.7,"use_symmetry":true,"symmetry_axis":"X"},"expected":{"ratio":0.7,"use_symmetry":true,"symmetry_axis":"X"}},
        {"name":"CaseDecimateTriangulate","type":"DECIMATE","shape":"grid","props":{"ratio":0.8,"use_collapse_triangulate":true},"expected":{"ratio":0.8,"use_collapse_triangulate":true}},
        {"name":"CaseDecimateVertexGroup","type":"DECIMATE","shape":"grid","group":"DecimateWeights","props":{"ratio":0.7,"vertex_group":"DecimateWeights","invert_vertex_group":false,"vertex_group_factor":1.0},"expected":{"ratio":0.7,"vertex_group":"DecimateWeights","invert_vertex_group":false,"vertex_group_factor":1.0}},
        {"name":"CaseDecimateVertexGroupInvert","type":"DECIMATE","shape":"grid","group":"DecimateWeights","unsupported_feature_id":"modifier.decimate.collapse_vertex_group_invert","props":{"ratio":0.7,"vertex_group":"DecimateWeights","invert_vertex_group":true,"vertex_group_factor":1.0},"expected":{"ratio":0.7,"vertex_group":"DecimateWeights","invert_vertex_group":true,"vertex_group_factor":1.0}},
        {"name":"CaseDecimateVertexGroupInvertHalf","type":"DECIMATE","shape":"grid","group":"DecimateWeights","unsupported_feature_id":"modifier.decimate.collapse_vertex_group_invert","props":{"ratio":0.5,"vertex_group":"DecimateWeights","invert_vertex_group":true,"vertex_group_factor":1.0},"expected":{"ratio":0.5,"vertex_group":"DecimateWeights","invert_vertex_group":true,"vertex_group_factor":1.0}},
        {"name":"CaseDecimateVertexGroupInvertNearOne","type":"DECIMATE","shape":"grid","group":"DecimateWeights","unsupported_feature_id":"modifier.decimate.collapse_vertex_group_invert","props":{"ratio":0.999,"vertex_group":"DecimateWeights","invert_vertex_group":true,"vertex_group_factor":1.0},"expected":{"ratio":0.999,"vertex_group":"DecimateWeights","invert_vertex_group":true,"vertex_group_factor":1.0}},
        {"name":"CaseDecimateVertexGroupInvertRatioOne","type":"DECIMATE","shape":"grid","group":"DecimateWeights","props":{"ratio":1.0,"vertex_group":"DecimateWeights","invert_vertex_group":true,"vertex_group_factor":1.0},"expected":{"ratio":1.0,"vertex_group":"DecimateWeights","invert_vertex_group":true,"vertex_group_factor":1.0}},
        {"name":"CaseDecimateVertexGroupFactor","type":"DECIMATE","shape":"grid","group":"DecimateWeights","props":{"ratio":0.7,"vertex_group":"DecimateWeights","invert_vertex_group":false,"vertex_group_factor":0.5},"expected":{"ratio":0.7,"vertex_group":"DecimateWeights","invert_vertex_group":false,"vertex_group_factor":0.5}},
        {"name":"CaseDecimateFirstCollapse","type":"DECIMATE","shape":"grid","props":{"ratio":0.999},"expected":{"ratio":0.999}},
        {"name":"CaseDecimateOpenGridHalf","type":"DECIMATE","shape":"grid","props":{"ratio":0.5},"expected":{"ratio":0.5}},
        {"name":"CaseWeld","type":"WELD","props":{"merge_threshold":0.025},"expected":{"merge_threshold":0.025}},
        {"name":"CaseDisplace","type":"DISPLACE","group":"DisplaceWeights","props":{"strength":0.4,"mid_level":0.25,"direction":"X","space":"GLOBAL","vertex_group":"DisplaceWeights"},"expected":{"strength":0.4,"mid_level":0.25,"direction":"X","space":"GLOBAL","vertex_group":"DisplaceWeights"}},
        {"name":"CaseSmooth","type":"SMOOTH","group":"SmoothWeights","props":{"factor":0.7,"iterations":3,"use_x":true,"use_y":false,"use_z":true,"vertex_group":"SmoothWeights"},"expected":{"factor":0.7,"iterations":3,"use_x":true,"use_y":false,"use_z":true,"vertex_group":"SmoothWeights"}},
        {"name":"CaseShrinkwrap","type":"SHRINKWRAP","props":{"wrap_method":"NEAREST_SURFACEPOINT","offset":0.15,"target":"@object:ShrinkTarget"},"expected":{"wrap_method":"NEAREST_SURFACEPOINT","offset":0.15,"target":"@object:ShrinkTarget"}},
        {"name":"CaseCast","type":"CAST","props":{"cast_type":"CYLINDER","factor":0.6,"radius":0.8,"size":1.25,"use_radius_as_size":false},"expected":{"cast_type":"CYLINDER","factor":0.6,"radius":0.8,"size":1.25,"use_radius_as_size":false}},
        {"name":"CaseCurve","type":"CURVE","props":{"deform_axis":"POS_X","object":"@object:CurveTarget"},"expected":{"deform_axis":"POS_X","object":"@object:CurveTarget"}},
        {"name":"CaseHook","type":"HOOK","group":"HookWeights","props":{"vertex_group":"HookWeights","object":"@object:HookTarget","strength":0.5},"expected":{"vertex_group":"HookWeights","object":"@object:HookTarget","strength":0.5}},
        {"name":"CaseLaplacianSmooth","type":"LAPLACIANSMOOTH","props":{"lambda_factor":0.3,"lambda_border":0.2,"iterations":2,"use_normalized":true,"use_volume_preserve":true,"use_x":true,"use_y":false,"use_z":true},"expected":{"lambda_factor":0.3,"lambda_border":0.2,"iterations":2,"use_normalized":true,"use_volume_preserve":true,"use_x":true,"use_y":false,"use_z":true}},
        {"name":"CaseCorrectiveSmooth","type":"CORRECTIVE_SMOOTH","props":{"factor":0.65,"iterations":3,"scale":0.8},"expected":{"factor":0.65,"iterations":3,"scale":0.8}},
        {"name":"CaseWave","type":"WAVE","props":{"height":0.25,"width":0.75,"speed":0.4},"expected":{"height":0.25,"width":0.75,"speed":0.4}},
        {"name":"CaseWarp","type":"WARP","props":{"object_from":"@object:WarpFrom","object_to":"@object:WarpTo","strength":0.5,"falloff_type":"SMOOTH","falloff_radius":2.0,"use_volume_preserve":true},"expected":{"object_from":"@object:WarpFrom","object_to":"@object:WarpTo","strength":0.5,"falloff_type":"SMOOTH","falloff_radius":2.0,"use_volume_preserve":true}},
        {"name":"CaseSimpleDeform","type":"SIMPLE_DEFORM","props":{"deform_method":"TWIST","angle":0.5,"deform_axis":"Z"},"expected":{"deform_method":"TWIST","angle":0.5,"deform_axis":"Z"}},
        {"name":"CaseScrew","type":"SCREW","props":{"steps":8,"angle":2.2,"screw_offset":0.3,"axis":"X"},"expected":{"steps":8,"angle":2.2,"screw_offset":0.3,"axis":"X"}},
        {"name":"CaseSkin","type":"SKIN","props":{"branch_smoothing":0.35,"use_smooth_shade":false,"use_x_symmetry":true,"use_y_symmetry":false,"use_z_symmetry":false},"expected":{"branch_smoothing":0.35,"use_smooth_shade":false,"use_x_symmetry":true,"use_y_symmetry":false,"use_z_symmetry":false}},
        {"name":"CaseSkinBranch","type":"SKIN","props":{"branch_smoothing":0.35,"use_smooth_shade":false,"use_x_symmetry":false,"use_y_symmetry":false,"use_z_symmetry":false},"expected":{"branch_smoothing":0.35,"use_smooth_shade":false,"use_x_symmetry":false,"use_y_symmetry":false,"use_z_symmetry":false}},
        {"name":"CaseSkinBranchSymmetry","type":"SKIN","props":{"branch_smoothing":0.35,"use_smooth_shade":false,"use_x_symmetry":true,"use_y_symmetry":false,"use_z_symmetry":false},"expected":{"branch_smoothing":0.35,"use_smooth_shade":false,"use_x_symmetry":true,"use_y_symmetry":false,"use_z_symmetry":false}},
        {"name":"CaseWireframe","type":"WIREFRAME","props":{"thickness":0.12,"offset":0.2,"use_boundary":true,"use_replace":true,"use_even_offset":true,"use_relative_offset":true,"use_crease":true,"crease_weight":0.65,"material_offset":1},"expected":{"thickness":0.12,"offset":0.2,"use_boundary":true,"use_replace":true,"use_even_offset":true,"use_relative_offset":true,"use_crease":true,"crease_weight":0.65,"material_offset":1}},
        {"name":"CaseEdgeSplit","type":"EDGE_SPLIT","props":{"split_angle":0.6,"use_edge_sharp":false},"expected":{"split_angle":0.6,"use_edge_sharp":false}},
        {"name":"CaseBuild","type":"BUILD","props":{"frame_start":2.0,"frame_duration":20.0,"use_random_order":true,"seed":17,"use_reverse":true},"expected":{"frame_start":2.0,"frame_duration":20.0,"use_random_order":true,"seed":17,"use_reverse":true}},
        {"name":"CaseMask","type":"MASK","group":"MaskWeights","props":{"mode":"VERTEX_GROUP","vertex_group":"MaskWeights","threshold":0.3,"invert_vertex_group":true},"expected":{"mode":"VERTEX_GROUP","vertex_group":"MaskWeights","threshold":0.3,"invert_vertex_group":true}},
        {"name":"CaseVertexWeightEdit","type":"VERTEX_WEIGHT_EDIT","group":"CurveWeights","props":{"vertex_group":"CurveWeights","falloff_type":"CURVE","map_curve":[[0.0,0.0],[0.5,0.2],[1.0,1.0]],"use_add":false,"use_remove":false,"default_weight":0.0,"normalize":false},"expected":{"vertex_group":"CurveWeights","falloff_type":"CURVE","map_curve":[[0.0,0.0],[0.5,0.2],[1.0,1.0]],"use_add":false,"use_remove":false,"default_weight":0.0,"normalize":false}},
        {"name":"CaseLattice","type":"LATTICE","group":"LatticeWeights","props":{"object":null,"vertex_group":"LatticeWeights","show_viewport":false},"expected_enabled":false,"expected":{"object":null,"vertex_group":"LatticeWeights"}},
        {"name":"CaseArmature","type":"ARMATURE","props":{"object":"@object:ArmatureTarget","use_deform_preserve_volume":true},"expected":{"object":"@object:ArmatureTarget","use_deform_preserve_volume":true}},
        {"name":"CaseNodes","type":"NODES","props":{"node_group":"@group:TransformGroup"},"expected":{"node_group":"@group:TransformGroup"}},
        {"name":"VolumeOwner","type":"VOLUME_TO_MESH","props":{"threshold":0.2,"adaptivity":0.1,"show_viewport":false},"expected_enabled":false,"expected":{"threshold":0.2,"adaptivity":0.1}},
        {"name":"MeshToVolumeOwner","type":"MESH_TO_VOLUME","props":{"object":"@object:MeshToVolumeOperand","resolution_mode":"VOXEL_SIZE","voxel_size":0.25,"interior_band_width":0.2,"density":1.0},"expected":{"object":"@object:MeshToVolumeOperand","resolution_mode":"VOXEL_SIZE","voxel_size":0.25,"interior_band_width":0.2,"density":1.0}},
        {"name":"MeshToVolumeOwnerAmount","type":"MESH_TO_VOLUME","props":{"object":"@object:MeshToVolumeOperand","resolution_mode":"VOXEL_AMOUNT","voxel_amount":16,"interior_band_width":0.35,"density":0.65},"expected":{"object":"@object:MeshToVolumeOperand","resolution_mode":"VOXEL_AMOUNT","voxel_amount":16,"interior_band_width":0.35,"density":0.65}}
    ])
}
fn boolean_cases() -> Vec<Value> {
    vec![
        json!({"name":"CaseBooleanDifferenceObjectExact","type":"BOOLEAN","props":{"operation":"DIFFERENCE","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"EXACT","use_self":false,"use_hole_tolerant":false},"expected":{"operation":"DIFFERENCE","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"EXACT","use_self":false,"use_hole_tolerant":false}}),
        json!({"name":"CaseBooleanUnionObjectExact","type":"BOOLEAN","props":{"operation":"UNION","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"EXACT","use_self":false,"use_hole_tolerant":true},"expected":{"operation":"UNION","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"EXACT","use_self":false,"use_hole_tolerant":true}}),
        json!({"name":"CaseBooleanIntersectObjectExact","type":"BOOLEAN","props":{"operation":"INTERSECT","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"EXACT","use_self":true,"use_hole_tolerant":true},"expected":{"operation":"INTERSECT","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"EXACT","use_self":true,"use_hole_tolerant":true}}),
        json!({"name":"CaseBooleanDifferenceObjectFloat","type":"BOOLEAN","props":{"operation":"DIFFERENCE","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"FLOAT","use_self":true,"use_hole_tolerant":true},"expected":{"operation":"DIFFERENCE","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"FLOAT","use_self":true,"use_hole_tolerant":true}}),
        json!({"name":"CaseBooleanUnionObjectFloat","type":"BOOLEAN","props":{"operation":"UNION","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"FLOAT","use_self":true,"use_hole_tolerant":false},"expected":{"operation":"UNION","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"FLOAT","use_self":true,"use_hole_tolerant":false}}),
        json!({"name":"CaseBooleanIntersectObjectFloat","type":"BOOLEAN","props":{"operation":"INTERSECT","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"FLOAT","use_self":false,"use_hole_tolerant":false},"expected":{"operation":"INTERSECT","operand_type":"OBJECT","object":"@object:BooleanOperand","solver":"FLOAT","use_self":false,"use_hole_tolerant":false}}),
        json!({"name":"CaseBooleanDifferenceCollectionExact","type":"BOOLEAN","props":{"operation":"DIFFERENCE","operand_type":"COLLECTION","collection":"@collection:BooleanOperands","solver":"EXACT","use_self":false,"use_hole_tolerant":true},"expected":{"operation":"DIFFERENCE","operand_type":"COLLECTION","collection":"@collection:BooleanOperands","solver":"EXACT","use_self":false,"use_hole_tolerant":true}}),
        json!({"name":"CaseBooleanUnionCollectionExact","type":"BOOLEAN","props":{"operation":"UNION","operand_type":"COLLECTION","collection":"@collection:BooleanOperands","solver":"EXACT","use_self":false,"use_hole_tolerant":true},"expected":{"operation":"UNION","operand_type":"COLLECTION","collection":"@collection:BooleanOperands","solver":"EXACT","use_self":false,"use_hole_tolerant":true}}),
        json!({"name":"CaseBooleanIntersectCollectionExact","type":"BOOLEAN","props":{"operation":"INTERSECT","operand_type":"COLLECTION","collection":"@collection:BooleanOperands","solver":"EXACT","use_self":true,"use_hole_tolerant":true},"expected":{"operation":"INTERSECT","operand_type":"COLLECTION","collection":"@collection:BooleanOperands","solver":"EXACT","use_self":true,"use_hole_tolerant":true}}),
        json!({"name":"CaseBooleanDifferenceCollectionFloat","type":"BOOLEAN","props":{"operation":"DIFFERENCE","operand_type":"COLLECTION","collection":"@collection:BooleanOperands","solver":"FLOAT","use_self":true,"use_hole_tolerant":false},"expected":{"operation":"DIFFERENCE","operand_type":"COLLECTION","collection":"@collection:BooleanOperands","solver":"FLOAT","use_self":true,"use_hole_tolerant":false}}),
        json!({"name":"CaseBooleanUnionCollectionFloat","type":"BOOLEAN","props":{"operation":"UNION","operand_type":"COLLECTION","collection":"@collection:BooleanOperands","solver":"FLOAT","use_self":true,"use_hole_tolerant":false},"expected":{"operation":"UNION","operand_type":"COLLECTION","collection":"@collection:BooleanOperands","solver":"FLOAT","use_self":true,"use_hole_tolerant":false}}),
    ]
}

fn expected_modifier_value(value: &Value, mappings: &serde_json::Map<String, Value>) -> Value {
    if let Some(reference) = value.as_str() {
        if let Some(name) = reference.strip_prefix("@object:") {
            return mappings[&format!("Object:{name}")].clone();
        }
        if let Some(name) = reference.strip_prefix("@collection:") {
            return mappings[&format!("Collection:{name}")].clone();
        }
        if let Some(name) = reference.strip_prefix("@group:") {
            return mappings[&format!("NodeGroup:{name}")].clone();
        }
    }
    value.clone()
}

fn assert_json_value(actual: &Value, expected: &Value, context: &str) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let delta = actual.as_f64().unwrap() - expected.as_f64().unwrap();
            assert!(delta.abs() <= 1.0e-6, "{context}: value delta is {delta}");
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{context}: array length");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_json_value(actual, expected, &format!("{context}[{index}]"));
            }
        }
        _ => assert_eq!(actual, expected, "{context}"),
    }
}

fn modifier_position_error(actual: &[Value], expected: &[Value]) -> Option<f64> {
    fn points(values: &[Value]) -> Vec<[f64; 3]> {
        values
            .iter()
            .map(|point| {
                [
                    point[0].as_f64().unwrap(),
                    point[1].as_f64().unwrap(),
                    point[2].as_f64().unwrap(),
                ]
            })
            .collect()
    }
    if actual.len() != expected.len() {
        return None;
    }
    let mut unmatched = points(expected);
    let mut maximum_error = 0.0_f64;
    for point in points(actual) {
        let mut nearest_index = 0;
        let mut nearest_distance = f64::INFINITY;
        for (index, candidate) in unmatched.iter().enumerate() {
            let distance = (0..3)
                .map(|axis| (point[axis] - candidate[axis]).powi(2))
                .sum::<f64>();
            if distance < nearest_distance {
                nearest_distance = distance;
                nearest_index = index;
            }
        }
        let nearest = unmatched.swap_remove(nearest_index);
        let error = (0..3)
            .map(|axis| (point[axis] - nearest[axis]).abs())
            .fold(0.0, f64::max);
        maximum_error = maximum_error.max(error);
    }
    Some(maximum_error)
}
fn modifier_vertex_set_gap(actual: &[Value], expected: &[Value]) -> Option<f64> {
    fn points(values: &[Value]) -> Vec<[f64; 3]> {
        values
            .iter()
            .map(|point| {
                [
                    point[0].as_f64().unwrap(),
                    point[1].as_f64().unwrap(),
                    point[2].as_f64().unwrap(),
                ]
            })
            .collect()
    }
    fn directed_gap(source: &[[f64; 3]], target: &[[f64; 3]]) -> Option<f64> {
        if source.is_empty() || target.is_empty() {
            return None;
        }
        let mut maximum = 0.0_f64;
        for point in source {
            let nearest = target.iter().min_by(|first, second| {
                let first_distance = (0..3)
                    .map(|axis| (point[axis] - first[axis]).powi(2))
                    .sum::<f64>();
                let second_distance = (0..3)
                    .map(|axis| (point[axis] - second[axis]).powi(2))
                    .sum::<f64>();
                first_distance.total_cmp(&second_distance)
            })?;
            let error = (0..3)
                .map(|axis| (point[axis] - nearest[axis]).abs())
                .fold(0.0, f64::max);
            maximum = maximum.max(error);
        }
        Some(maximum)
    }
    let actual = points(actual);
    let expected = points(expected);
    Some(directed_gap(&actual, &expected)?.max(directed_gap(&expected, &actual)?))
}

fn assert_matching_topology(actual: &Value, expected: &Value, context: &str) {
    for key in ["vertex_count", "edge_count", "face_count"] {
        assert_eq!(actual[key], expected[key], "{context} topology {key}");
    }
}
fn geometry_triangles(geometry: &Value) -> Vec<[DVec3; 3]> {
    let positions = geometry["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|point| {
            DVec3::new(
                point[0].as_f64().unwrap(),
                point[1].as_f64().unwrap(),
                point[2].as_f64().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let mut triangles = Vec::new();
    for face in geometry["faces"].as_array().unwrap() {
        let indices = face
            .as_array()
            .unwrap()
            .iter()
            .map(|index| usize::try_from(index.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert!(indices.len() >= 3, "evaluated mesh contains a short face");
        triangulate_face(&indices, &positions, &mut triangles);
    }
    triangles
}

fn triangulate_face(indices: &[usize], points: &[DVec3], triangles: &mut Vec<[DVec3; 3]>) {
    let mut normal = DVec3::ZERO;
    for index in 0..indices.len() {
        normal += points[indices[index]].cross(points[indices[(index + 1) % indices.len()]]);
    }
    let dominant_axis = if normal.x.abs() >= normal.y.abs() && normal.x.abs() >= normal.z.abs() {
        0
    } else if normal.y.abs() >= normal.z.abs() {
        1
    } else {
        2
    };
    let project = |point: DVec3| match dominant_axis {
        0 => [point.y, point.z],
        1 => [point.x, point.z],
        _ => [point.x, point.y],
    };
    let coordinates = indices
        .iter()
        .map(|index| project(points[*index]))
        .collect::<Vec<_>>();
    let signed_area = coordinates
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let next = coordinates[(index + 1) % coordinates.len()];
            point[0] * next[1] - point[1] * next[0]
        })
        .sum::<f64>();
    let winding = signed_area.signum();
    assert_ne!(winding, 0.0, "evaluated mesh contains a zero-area face");
    let mut ring = (0..indices.len()).collect::<Vec<_>>();
    while ring.len() > 3 {
        let mut removed = false;
        for ring_index in 0..ring.len() {
            let previous = ring[(ring_index + ring.len() - 1) % ring.len()];
            let current = ring[ring_index];
            let next = ring[(ring_index + 1) % ring.len()];
            let turn = orient_2d(
                coordinates[previous],
                coordinates[current],
                coordinates[next],
            ) * winding;
            if turn.abs() <= 1.0e-14 {
                ring.remove(ring_index);
                removed = true;
                break;
            }
            if turn < 0.0 {
                continue;
            }
            let contains_vertex = ring.iter().copied().any(|candidate| {
                candidate != previous
                    && candidate != current
                    && candidate != next
                    && point_in_triangle_2d(
                        coordinates[candidate],
                        coordinates[previous],
                        coordinates[current],
                        coordinates[next],
                        winding,
                    )
            });
            if !contains_vertex {
                triangles.push([
                    points[indices[previous]],
                    points[indices[current]],
                    points[indices[next]],
                ]);
                ring.remove(ring_index);
                removed = true;
                break;
            }
        }
        assert!(removed, "evaluated face could not be triangulated");
    }
    triangles.push([
        points[indices[ring[0]]],
        points[indices[ring[1]]],
        points[indices[ring[2]]],
    ]);
}

fn orient_2d(first: [f64; 2], second: [f64; 2], third: [f64; 2]) -> f64 {
    (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])
}

fn point_in_triangle_2d(
    point: [f64; 2],
    first: [f64; 2],
    second: [f64; 2],
    third: [f64; 2],
    winding: f64,
) -> bool {
    orient_2d(first, second, point) * winding >= -1.0e-14
        && orient_2d(second, third, point) * winding >= -1.0e-14
        && orient_2d(third, first, point) * winding >= -1.0e-14
}

fn geometry_volume(geometry: &Value) -> f64 {
    geometry_triangles(geometry)
        .iter()
        .map(|triangle| triangle[0].dot(triangle[1].cross(triangle[2])) / 6.0)
        .sum::<f64>()
        .abs()
}

fn geometry_topology(geometry: &Value) -> (bool, usize) {
    let faces = geometry["faces"].as_array().unwrap();
    let mut edge_faces = HashMap::<(usize, usize), Vec<usize>>::new();
    for (face_index, face) in faces.iter().enumerate() {
        let vertices = face
            .as_array()
            .unwrap()
            .iter()
            .map(|index| usize::try_from(index.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert!(vertices.len() >= 3, "evaluated mesh contains a short face");
        for index in 0..vertices.len() {
            let first = vertices[index];
            let second = vertices[(index + 1) % vertices.len()];
            let edge = (first.min(second), first.max(second));
            edge_faces.entry(edge).or_default().push(face_index);
        }
    }
    let watertight = !faces.is_empty() && edge_faces.values().all(|owners| owners.len() == 2);
    let mut adjacency = vec![Vec::new(); faces.len()];
    for owners in edge_faces.values().filter(|owners| owners.len() == 2) {
        adjacency[owners[0]].push(owners[1]);
        adjacency[owners[1]].push(owners[0]);
    }
    let mut visited = vec![false; faces.len()];
    let mut components = 0;
    for start in 0..faces.len() {
        if visited[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(face) = stack.pop() {
            for neighbor in &adjacency[face] {
                if !visited[*neighbor] {
                    visited[*neighbor] = true;
                    stack.push(*neighbor);
                }
            }
        }
    }
    (watertight, components)
}

fn point_triangle_distance_squared(point: DVec3, triangle: &[DVec3; 3]) -> f64 {
    fn segment_distance_squared(point: DVec3, first: DVec3, second: DVec3) -> f64 {
        let edge = second - first;
        let length_squared = edge.length_squared();
        let parameter = if length_squared > f64::MIN_POSITIVE {
            ((point - first).dot(edge) / length_squared).clamp(0.0, 1.0)
        } else {
            0.0
        };
        point.distance_squared(first + edge * parameter)
    }
    let [a, b, c] = *triangle;
    let ab = b - a;
    let ac = c - a;
    let normal = ab.cross(ac);
    let normal_squared = normal.length_squared();
    if normal_squared > f64::MIN_POSITIVE {
        let projection = point - normal * (normal.dot(point - a) / normal_squared);
        let projected = projection - a;
        let d00 = ab.dot(ab);
        let d01 = ab.dot(ac);
        let d11 = ac.dot(ac);
        let d20 = projected.dot(ab);
        let d21 = projected.dot(ac);
        let denominator = d00 * d11 - d01 * d01;
        if denominator > f64::MIN_POSITIVE {
            let u = (d11 * d20 - d01 * d21) / denominator;
            let v = (d00 * d21 - d01 * d20) / denominator;
            if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                return point.distance_squared(projection);
            }
        }
    }
    segment_distance_squared(point, a, b)
        .min(segment_distance_squared(point, b, c))
        .min(segment_distance_squared(point, c, a))
}

fn directed_surface_distance(source: &[[DVec3; 3]], target: &[[DVec3; 3]]) -> f64 {
    const DIVISIONS: usize = 16;
    let mut maximum = 0.0_f64;
    for triangle in source {
        for first_weight in 0..=DIVISIONS {
            for second_weight in 0..=DIVISIONS - first_weight {
                let first_weight = first_weight as f64 / DIVISIONS as f64;
                let second_weight = second_weight as f64 / DIVISIONS as f64;
                let point = triangle[0] * (1.0 - first_weight - second_weight)
                    + triangle[1] * first_weight
                    + triangle[2] * second_weight;
                let nearest_squared = target
                    .iter()
                    .map(|candidate| point_triangle_distance_squared(point, candidate))
                    .fold(f64::INFINITY, f64::min);
                maximum = maximum.max(nearest_squared.sqrt());
            }
        }
    }
    maximum
}

fn assert_boolean_geometry(actual: &Value, expected: &Value, context: &str) -> (f64, f64) {
    // Blender's exact BMesh solver and Potter's BSP retain different redundant boundary
    // vertices (the original UNION fixture had 34 Potter edges versus Blender's 28).
    // Compare the resulting closed surfaces instead of requiring solver-specific topology.
    let (actual_watertight, actual_components) = geometry_topology(actual);
    let (expected_watertight, expected_components) = geometry_topology(expected);
    assert!(
        actual_watertight,
        "{context}: Potter result is not watertight"
    );
    assert!(
        expected_watertight,
        "{context}: Blender result is not watertight"
    );
    assert_eq!(
        actual_components, expected_components,
        "{context}: connected-component count"
    );
    let expected_volume = geometry_volume(expected);
    let relative_volume_error = (geometry_volume(actual) - expected_volume).abs() / expected_volume;
    assert!(
        relative_volume_error <= 1.0e-6,
        "{context}: relative enclosed-volume error {relative_volume_error}"
    );
    let actual_triangles = geometry_triangles(actual);
    let expected_triangles = geometry_triangles(expected);
    let surface_error = directed_surface_distance(&actual_triangles, &expected_triangles).max(
        directed_surface_distance(&expected_triangles, &actual_triangles),
    );
    assert!(
        surface_error <= 1.0e-5,
        "{context}: two-way surface distance {surface_error}"
    );
    (surface_error, relative_volume_error)
}
fn assert_skin_branch_round_trip(
    case: &Value,
    blender: &Path,
    project: &Path,
    root: &Path,
    owner_id: &str,
    mappings: &serde_json::Map<String, Value>,
    before_geometry: &Value,
) -> Result<(), Box<dyn Error>> {
    let scene: potter_core::model::SceneDoc =
        serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let owner_key = potter_core::model::Id::new(owner_id.to_owned())?;
    let modifier = scene.nodes[&owner_key].modifiers.first().unwrap();
    assert_eq!(modifier.modifier_type, "skin", "{}", case["name"]);
    for (key, value) in case["expected"].as_object().unwrap() {
        assert_json_value(
            &modifier.params[key],
            &expected_modifier_value(value, mappings),
            &format!("{}.params.{key}", case["name"]),
        );
    }

    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command
        .args(["inspect"])
        .arg(project)
        .args(["--id", owner_id, "--json"]);
    let output = run_guarded(command)?;
    assert!(
        !output.status.success(),
        "Skin branch inspection must reject unsupported evaluated geometry"
    );
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(response["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        response["error"]["details"]["feature_id"],
        "modifier.skin.branch_hull"
    );

    let feature_catalog = potter_core::catalog::feature_catalog();
    let branch_feature = feature_catalog["features"]
        .as_array()
        .unwrap()
        .iter()
        .find(|feature| feature["feature_id"] == "modifier.skin.branch_hull")
        .unwrap();
    assert_eq!(branch_feature["status"], "not_supported");
    assert_eq!(branch_feature["capabilities"]["evaluate"], false);

    let exported = root.join("modifier_roundtrip.blend");
    let export_result = pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    assert_eq!(export_result["result"]["format"], "blend");
    assert!(export_result["result"]["context"]["evaluation_hash"].is_null());

    let reopened_path = root.join("modifier_after.json");
    let script_path = root.join("reopen_modifier.py");
    fs::write(&script_path, MODIFIER_REOPEN)?;
    let mut command = Command::new(blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&script_path)
        .arg("--")
        .arg(&reopened_path)
        .arg(root.join("modifier_cases.json"));
    let output = run_guarded(command)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "Blender branch Skin reopen failed: stdout={stdout} stderr={stderr}",
    );
    let after: Value = serde_json::from_slice(&fs::read(&reopened_path)?)?;
    let reopened = &after[case["name"].as_str().unwrap()];
    assert_eq!(reopened["type"], "SKIN");
    for (key, value) in case["props"].as_object().unwrap() {
        assert_json_value(
            &reopened["props"][key],
            value,
            &format!("{}.exported.{key}", case["name"]),
        );
    }
    assert_matching_topology(reopened, before_geometry, case["name"].as_str().unwrap());
    assert_eq!(reopened["faces"], before_geometry["faces"]);
    let position_error = modifier_position_error(
        reopened["positions"].as_array().unwrap(),
        before_geometry["positions"].as_array().unwrap(),
    )
    .unwrap();
    assert!(
        position_error <= 1.0e-5,
        "{}: Blender-evaluated branch geometry after export differs by {position_error}",
        case["name"]
    );
    println!(
        "{}: Potter inspection reports UnsupportedFeature; .blend modifier settings and Blender-evaluated geometry round-trip",
        case["name"]
    );
    Ok(())
}

fn modifier_round_trip(case_name: &str) -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender() else {
        return Ok(());
    };
    let all_cases = modifier_cases();
    let boolean_cases = boolean_cases();
    let case = all_cases
        .as_array()
        .unwrap()
        .iter()
        .chain(boolean_cases.iter())
        .find(|case| case["name"] == case_name)
        .unwrap()
        .clone();
    let cases = json!([case]);
    let case = &cases[0];
    let kind = case["type"].as_str().unwrap();
    let directory = tempdir()?;
    let root = directory.path();
    fs::write(
        root.join("modifier_cases.json"),
        serde_json::to_vec(&cases)?,
    )?;
    run_blender_script(
        &blender,
        "make_modifier.py",
        MODIFIER_FIXTURE,
        root,
        &[],
        "Blender fixture failed",
        run_guarded,
    )?;
    if case["unsupported"] == true {
        println!("{kind}: unavailable in Blender 5.2.2");
        return Ok(());
    }
    let before: Value = serde_json::from_slice(&fs::read(root.join("modifier_before.json"))?)?;
    let project = root.join("modifier_project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("modifier_source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let mappings = imported["result"]["id_mappings"].as_object().unwrap();
    let owner_id = mappings[&format!("Object:{case_name}")].as_str().unwrap();
    let losses = imported["result"]["losses"].as_array().unwrap();
    assert!(
        !losses.iter().any(|loss| {
            loss["feature_id"] == format!("blender.modifier.{}", kind.to_ascii_lowercase())
        }),
        "{case_name}: import reported modifier loss: {losses:?}"
    );
    if kind == "SKIN" && case_name.starts_with("CaseSkinBranch") {
        assert_skin_branch_round_trip(
            case,
            &blender,
            &project,
            root,
            owner_id,
            mappings,
            &before[case_name],
        )?;
        return Ok(());
    }

    if let Some(feature_id) = case["unsupported_feature_id"].as_str() {
        let scene: potter_core::model::SceneDoc =
            serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
        let owner_key = potter_core::model::Id::new(owner_id.to_owned())?;
        let modifier = scene.nodes[&owner_key].modifiers.first().unwrap();
        assert_eq!(modifier.modifier_type, "decimate", "{case_name}");
        for (key, value) in case["expected"].as_object().unwrap() {
            assert_json_value(
                &modifier.params[key],
                &expected_modifier_value(value, mappings),
                &format!("{case_name}.params.{key}"),
            );
        }

        let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
        command
            .args(["inspect"])
            .arg(project)
            .args(["--id", owner_id, "--json"]);
        let output = run_guarded(command)?;
        assert!(
            !output.status.success(),
            "{case_name}: unsupported COLLAPSE option must reject evaluation"
        );
        let response: Value = serde_json::from_slice(&output.stdout)?;
        assert_eq!(
            response["error"]["code"], "UNSUPPORTED_FEATURE",
            "{case_name}"
        );
        assert_eq!(
            response["error"]["details"]["feature_id"], feature_id,
            "{case_name}"
        );
        let catalog = potter_core::catalog::feature_catalog();
        let feature = catalog["features"]
            .as_array()
            .unwrap()
            .iter()
            .find(|feature| feature["feature_id"] == feature_id)
            .unwrap();
        assert_eq!(feature["status"], "not_supported", "{case_name}");
        assert_eq!(feature["capabilities"]["evaluate"], false, "{case_name}");
        assert!(
            feature["reason"]
                .as_str()
                .is_some_and(|reason| !reason.is_empty()),
            "{case_name}: catalog row must explain the parity limitation"
        );
        return Ok(());
    }

    let inspected = pot_json(&["inspect", project.to_str().unwrap(), "--id", owner_id])?;
    let owner = inspected["result"]["items"]
        .as_array()
        .unwrap()
        .first()
        .unwrap();
    let modifier = owner["modifiers"].as_array().unwrap().first().unwrap();
    let expected_type = match kind {
        "LAPLACIANSMOOTH" => "laplacian_smooth".to_owned(),
        "SUBSURF" => "subdivision".to_owned(),
        _ => kind.to_ascii_lowercase(),
    };
    assert_eq!(modifier["type"], expected_type, "{case_name}");
    if let Some(expected_enabled) = case["expected_enabled"].as_bool() {
        assert_eq!(modifier["enabled"], expected_enabled, "{case_name}.enabled");
    }
    for (key, value) in case["expected"].as_object().unwrap() {
        assert_json_value(
            &modifier["params"][key],
            &expected_modifier_value(value, mappings),
            &format!("{case_name}.params.{key}"),
        );
    }
    let before_geometry = &before[case_name];
    let imported_geometry = &owner["evaluated_geometry"];
    if kind == "DECIMATE" {
        let position_error = (imported_geometry["vertex_count"] == before_geometry["vertex_count"])
            .then(|| {
                modifier_position_error(
                    imported_geometry["positions"].as_array().unwrap(),
                    before_geometry["positions"].as_array().unwrap(),
                )
            })
            .flatten();
        let vertex_set_gap = modifier_vertex_set_gap(
            imported_geometry["positions"].as_array().unwrap(),
            before_geometry["positions"].as_array().unwrap(),
        );
        let position_error = position_error.map_or_else(
            || "n/a (vertex counts differ)".to_owned(),
            |error| format!("{error:.6e}"),
        );
        let vertex_set_gap = vertex_set_gap.map_or_else(
            || "n/a (an evaluated mesh is empty)".to_owned(),
            |error| format!("{error:.6e}"),
        );
        println!(
            "{case_name}: Blender topology [{}, {}, {}], Potter topology [{}, {}, {}], max per-vertex error={position_error}, bidirectional vertex-set gap={vertex_set_gap}",
            before_geometry["vertex_count"],
            before_geometry["edge_count"],
            before_geometry["face_count"],
            imported_geometry["vertex_count"],
            imported_geometry["edge_count"],
            imported_geometry["face_count"],
        );
    }
    if matches!(kind, "ARRAY" | "DISPLACE") {
        assert_matrix_parity(
            owner["transform"]["world"]["matrix"].as_array().unwrap(),
            before_geometry["matrix"].as_array().unwrap(),
            &format!("{case_name} world transform"),
        );
    }
    let (import_error, import_volume_error) = if kind == "MESH_TO_VOLUME" {
        assert_eq!(owner["kind"], "volume", "{case_name}: modifier host type");
        assert!(
            imported_geometry.is_null(),
            "{case_name}: Volume has no mesh output"
        );
        assert_eq!(before_geometry["volume_grid_names"], json!(["density"]));
        assert_eq!(before_geometry["volume_grid_types"], json!(["FLOAT"]));
        let scene: potter_core::model::SceneDoc =
            serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
        let owner_key = potter_core::model::Id::new(owner_id.to_owned())?;
        let data_id = scene.nodes.get(&owner_key).unwrap().data.as_ref().unwrap();
        let snapshot = potter_core::eval::Snapshot::evaluate(
            &scene,
            &potter_core::eval::EvaluationContext::default(),
        )?;
        let volume = snapshot.volume_data.get(data_id).unwrap();
        let grid = volume.grids.first().unwrap();
        assert_json_value(
            &json!(grid.voxel_size),
            &before_geometry["volume_grid_voxel_size"],
            &format!("{case_name}.volume_grid_voxel_size"),
        );
        assert_eq!(
            json!(grid.dims),
            before_geometry["volume_grid_dimensions"],
            "{case_name}: grid dimensions"
        );
        let bounds = snapshot
            .nodes
            .get(&owner_key)
            .and_then(|node| node.bounds.as_ref())
            .unwrap();
        assert_json_value(
            &json!([bounds.min.to_array(), bounds.max.to_array()]),
            &before_geometry["volume_bounds"],
            &format!("{case_name}.volume_bounds"),
        );
        let samples = [
            potter_core::geom::volume::sample_density_checked(DVec3::ZERO, volume)?,
            potter_core::geom::volume::sample_density_checked(DVec3::new(1.5, 0.0, 0.0), volume)?,
        ];
        for (index, (actual, expected)) in samples
            .iter()
            .zip(
                before_geometry["volume_density_samples"]
                    .as_array()
                    .unwrap(),
            )
            .enumerate()
        {
            let expected = expected.as_f64().unwrap();
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "{case_name}: density sample {index} differs: Potter {actual}, Blender {expected}"
            );
        }
        (0.0, 0.0)
    } else if kind == "BOOLEAN" {
        assert_boolean_geometry(imported_geometry, before_geometry, case_name)
    } else if kind == "DECIMATE" {
        assert_matching_topology(imported_geometry, before_geometry, case_name);
        let before_positions = before_geometry["positions"].as_array().unwrap();
        let imported_positions = imported_geometry["positions"].as_array().unwrap();
        let error = modifier_position_error(imported_positions, before_positions)
            .ok_or_else(|| format!("{case_name}: imported vertex count differs from Blender"))?;
        assert!(
            error <= 1.0e-5,
            "{case_name}: imported evaluation max vertex error {error}"
        );
        (error, 0.0)
    } else {
        assert_matching_topology(imported_geometry, before_geometry, case_name);
        let before_positions = before_geometry["positions"].as_array().unwrap();
        let imported_positions = imported_geometry["positions"].as_array().unwrap();
        let error = modifier_position_error(imported_positions, before_positions)
            .ok_or_else(|| format!("{case_name}: imported vertex count differs from Blender"))?;
        assert!(
            error
                <= (if matches!(kind, "ARRAY" | "SKIN" | "WIREFRAME") {
                    1.0e-5
                } else {
                    1.0e-4
                }),
            "{case_name}: imported evaluation max vertex error {error}"
        );
        (error, 0.0)
    };

    let exported = root.join("modifier_roundtrip.blend");
    pot_json(&[
        "export",
        project.to_str().unwrap(),
        "--format",
        "blend",
        "--out",
        exported.to_str().unwrap(),
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let reopened_path = root.join("modifier_after.json");
    let script_path = root.join("reopen_modifier.py");
    fs::write(&script_path, MODIFIER_REOPEN)?;
    let mut command = Command::new(&blender);
    command
        .args(["--background"])
        .arg(&exported)
        .arg("--python")
        .arg(&script_path)
        .arg("--")
        .arg(&reopened_path)
        .arg(root.join("modifier_cases.json"));
    let output = run_guarded(command)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && !stdout.contains("Traceback") && !stderr.contains("Traceback"),
        "Blender modifier reopen failed: stdout={stdout} stderr={stderr}",
    );
    let after: Value = serde_json::from_slice(&fs::read(&reopened_path)?)?;
    let reopened = &after[case_name];
    assert_eq!(reopened["type"], kind, "{case_name}");
    let exported_params = case.get("exported").unwrap_or(&case["props"]);
    for (key, value) in exported_params.as_object().unwrap() {
        let expected = if let Some(name) = value
            .as_str()
            .and_then(|text| text.strip_prefix("@object:"))
        {
            json!(name)
        } else if let Some(name) = value
            .as_str()
            .and_then(|text| text.strip_prefix("@collection:"))
        {
            json!(name)
        } else if let Some(name) = value.as_str().and_then(|text| text.strip_prefix("@group:")) {
            json!(name)
        } else {
            value.clone()
        };
        assert_json_value(
            &reopened["props"][key],
            &expected,
            &format!("{case_name}.exported.{key}"),
        );
    }
    if case_name == "CaseArrayUV" {
        assert_json_value(
            &reopened["uv_layers"],
            &before_geometry["uv_layers"],
            "CaseArrayUV UV offsets",
        );
    }
    let (export_error, export_volume_error) = if kind == "MESH_TO_VOLUME" {
        assert_eq!(
            reopened["volume_grid_names"],
            before_geometry["volume_grid_names"]
        );
        assert_eq!(
            reopened["volume_grid_types"],
            before_geometry["volume_grid_types"]
        );
        assert_eq!(
            reopened["volume_grid_channels"],
            before_geometry["volume_grid_channels"]
        );
        assert_json_value(
            &reopened["volume_grid_voxel_size"],
            &before_geometry["volume_grid_voxel_size"],
            &format!("{case_name}.volume_grid_voxel_size"),
        );
        assert_eq!(
            reopened["volume_grid_dimensions"], before_geometry["volume_grid_dimensions"],
            "{case_name}: exported grid dimensions"
        );
        assert_json_value(
            &reopened["volume_density_samples"],
            &before_geometry["volume_density_samples"],
            &format!("{case_name}.volume_density_samples"),
        );
        assert_json_value(
            &reopened["volume_bounds"],
            &before_geometry["volume_bounds"],
            &format!("{case_name}.volume_bounds"),
        );
        (0.0, 0.0)
    } else if kind == "BOOLEAN" {
        assert_boolean_geometry(reopened, before_geometry, case_name)
    } else if kind == "DECIMATE" {
        let topology_matches = ["vertex_count", "edge_count", "face_count"]
            .into_iter()
            .all(|key| reopened[key] == before_geometry[key]);
        let before_positions = before_geometry["positions"].as_array().unwrap();
        let reopened_positions = reopened["positions"].as_array().unwrap();
        let error = modifier_position_error(reopened_positions, before_positions);
        println!(
            "{case_name}: reopened Blender topology [{}, {}, {}], expected [{}, {}, {}], max per-vertex error={error:?}",
            reopened["vertex_count"],
            reopened["edge_count"],
            reopened["face_count"],
            before_geometry["vertex_count"],
            before_geometry["edge_count"],
            before_geometry["face_count"],
        );
        assert!(
            topology_matches,
            "{case_name}: exported Blender topology mismatch"
        );
        let error = error.ok_or_else(|| format!("{case_name}: exported vertex counts differ"))?;
        assert!(
            error <= 1.0e-5,
            "{case_name}: exported evaluation max vertex error {error}"
        );
        (error, 0.0)
    } else {
        assert_matching_topology(reopened, before_geometry, case_name);
        let reopened_positions = reopened["positions"].as_array().unwrap();
        let before_positions = before_geometry["positions"].as_array().unwrap();
        let error = modifier_position_error(reopened_positions, before_positions)
            .ok_or_else(|| format!("{case_name}: exported vertex count differs from Blender"))?;
        assert!(
            error
                <= (if matches!(kind, "ARRAY" | "SKIN" | "WIREFRAME") {
                    1.0e-5
                } else {
                    1.0e-4
                }),
            "{case_name}: exported evaluation max vertex error {error}"
        );
        (error, 0.0)
    };
    if kind == "DECIMATE" {
        println!(
            "{case_name}: DECIMATE modifier properties round-trip; evaluated geometry metrics reported above"
        );
    } else if kind == "MESH_TO_VOLUME" {
        println!("{kind}: Volume owner / mesh operand / grid and bounds round trip ok");
    } else if kind == "BOOLEAN" {
        println!(
            "{kind}: import ok / params ok / max surface errors import={import_error:.6e}, export={export_error:.6e}; relative volume errors import={import_volume_error:.6e}, export={export_volume_error:.6e} / export ok"
        );
    } else {
        println!(
            "{kind}: import ok / params ok / eval max errors import={import_error:.6e}, export={export_error:.6e} / export ok"
        );
    }
    Ok(())
}

macro_rules! modifier_test {
    ($test_name:ident, $case_name:literal) => {
        #[test]
        fn $test_name() -> Result<(), Box<dyn Error>> {
            modifier_round_trip($case_name)
        }
    };
}

modifier_test!(blender_mirror_round_trip, "CaseMirror");
modifier_test!(blender_array_round_trip, "CaseArray");
modifier_test!(blender_array_caps_round_trip, "CaseArrayCaps");
modifier_test!(blender_array_fit_curve_round_trip, "CaseArrayFitCurve");
modifier_test!(blender_array_uv_offsets_round_trip, "CaseArrayUV");
modifier_test!(blender_subdivision_round_trip, "CaseSubdivision");
modifier_test!(blender_multires_round_trip, "CaseMultires");
modifier_test!(blender_solidify_round_trip, "CaseSolidify");
modifier_test!(blender_triangulate_round_trip, "CaseTriangulate");
modifier_test!(blender_bevel_round_trip, "CaseBevel");
modifier_test!(blender_bevel_vertices_round_trip, "CaseBevelVertices");
modifier_test!(blender_bevel_width_round_trip, "CaseBevelWidth");
modifier_test!(blender_decimate_round_trip, "CaseDecimate");
modifier_test!(blender_decimate_unsubdiv_round_trip, "CaseDecimateUnsubdiv");
modifier_test!(
    blender_decimate_unsubdiv_iterations_round_trip,
    "CaseDecimateUnsubdivIterations"
);
modifier_test!(blender_decimate_dissolve_round_trip, "CaseDecimateDissolve");
modifier_test!(
    blender_decimate_dissolve_normal_round_trip,
    "CaseDecimateDissolveNormal"
);
modifier_test!(
    blender_decimate_dissolve_material_round_trip,
    "CaseDecimateDissolveMaterial"
);
modifier_test!(
    blender_decimate_dissolve_seam_round_trip,
    "CaseDecimateDissolveSeam"
);
modifier_test!(
    blender_decimate_dissolve_sharp_round_trip,
    "CaseDecimateDissolveSharp"
);
modifier_test!(
    blender_decimate_dissolve_uv_round_trip,
    "CaseDecimateDissolveUV"
);
modifier_test!(
    blender_decimate_dissolve_boundaries_round_trip,
    "CaseDecimateDissolveBoundaries"
);
modifier_test!(blender_decimate_symmetry_round_trip, "CaseDecimateSymmetry");
modifier_test!(
    blender_decimate_triangulate_round_trip,
    "CaseDecimateTriangulate"
);
modifier_test!(
    blender_decimate_vertex_group_round_trip,
    "CaseDecimateVertexGroup"
);
modifier_test!(
    blender_decimate_vertex_group_invert_round_trip,
    "CaseDecimateVertexGroupInvert"
);
modifier_test!(
    blender_decimate_vertex_group_invert_half_round_trip,
    "CaseDecimateVertexGroupInvertHalf"
);
modifier_test!(
    blender_decimate_vertex_group_invert_near_one_round_trip,
    "CaseDecimateVertexGroupInvertNearOne"
);
modifier_test!(
    blender_decimate_vertex_group_invert_ratio_one_round_trip,
    "CaseDecimateVertexGroupInvertRatioOne"
);
modifier_test!(
    blender_decimate_vertex_group_factor_round_trip,
    "CaseDecimateVertexGroupFactor"
);
modifier_test!(
    blender_decimate_first_collapse_round_trip,
    "CaseDecimateFirstCollapse"
);
modifier_test!(
    blender_decimate_open_grid_half_round_trip,
    "CaseDecimateOpenGridHalf"
);
modifier_test!(blender_weld_round_trip, "CaseWeld");
modifier_test!(blender_displace_round_trip, "CaseDisplace");
modifier_test!(blender_smooth_round_trip, "CaseSmooth");
// Blender 5.2 does not support COLLECTION + FLOAT + INTERSECT.
#[test]
fn blender_boolean_round_trip() -> Result<(), Box<dyn Error>> {
    for case in boolean_cases() {
        modifier_round_trip(case["name"].as_str().unwrap())?;
    }
    Ok(())
}
modifier_test!(blender_shrinkwrap_round_trip, "CaseShrinkwrap");
modifier_test!(blender_cast_round_trip, "CaseCast");
modifier_test!(blender_curve_round_trip, "CaseCurve");
modifier_test!(blender_hook_round_trip, "CaseHook");
modifier_test!(blender_laplacian_smooth_round_trip, "CaseLaplacianSmooth");
modifier_test!(blender_corrective_smooth_round_trip, "CaseCorrectiveSmooth");
modifier_test!(blender_wave_round_trip, "CaseWave");
modifier_test!(blender_warp_round_trip, "CaseWarp");
modifier_test!(blender_simple_deform_round_trip, "CaseSimpleDeform");
modifier_test!(blender_screw_round_trip, "CaseScrew");
modifier_test!(blender_skin_round_trip, "CaseSkin");
modifier_test!(blender_skin_branch_round_trip, "CaseSkinBranch");
modifier_test!(
    blender_skin_branch_symmetry_round_trip,
    "CaseSkinBranchSymmetry"
);
modifier_test!(blender_wireframe_round_trip, "CaseWireframe");
modifier_test!(blender_edge_split_round_trip, "CaseEdgeSplit");
modifier_test!(blender_build_round_trip, "CaseBuild");
modifier_test!(blender_mask_round_trip, "CaseMask");
modifier_test!(
    blender_vertex_weight_edit_curve_mapping_round_trip,
    "CaseVertexWeightEdit"
);
modifier_test!(blender_lattice_round_trip, "CaseLattice");
modifier_test!(blender_armature_round_trip, "CaseArmature");
modifier_test!(blender_nodes_round_trip, "CaseNodes");
modifier_test!(blender_volume_to_mesh_round_trip, "VolumeOwner");
modifier_test!(blender_mesh_to_volume_round_trip, "MeshToVolumeOwner");
modifier_test!(
    blender_mesh_to_volume_voxel_amount_round_trip,
    "MeshToVolumeOwnerAmount"
);

#[test]
fn blender_transform_import_parity_across_rotation_modes() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender() else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_transform_fixture.py",
        TRANSFORM_FIXTURE,
        root,
        &[],
        "Blender fixture failed",
        run_guarded,
    )?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("transform_expected.json"))?)?;
    let project = root.join("transform_project");
    pot_json(&["init", project.to_str().unwrap()])?;
    let imported = pot_json(&[
        "import",
        project.to_str().unwrap(),
        "--file",
        root.join("transform_source.blend").to_str().unwrap(),
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender.to_str().unwrap(),
    ])?;
    let mappings = imported["result"]["id_mappings"].as_object().unwrap();
    for (name, matrices) in expected.as_object().unwrap() {
        let id = mappings[&format!("Object:{name}")].as_str().unwrap();
        let inspected = pot_json(&["inspect", project.to_str().unwrap(), "--id", id])?;
        let node = inspected["result"]["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["id"] == id)
            .unwrap();
        assert_matrix_parity(
            node["transform"]["evaluated"]["matrix"].as_array().unwrap(),
            matrices["local"].as_array().unwrap(),
            &format!("{name} local"),
        );
        assert_matrix_parity(
            node["transform"]["world"]["matrix"].as_array().unwrap(),
            matrices["world"].as_array().unwrap(),
            &format!("{name} world"),
        );
    }
    Ok(())
}
