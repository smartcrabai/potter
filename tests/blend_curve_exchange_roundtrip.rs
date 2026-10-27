//! Blender 5.2.2 round-trip coverage for curve exchange edits.
//!
//! When a taper Data-Block has no node, `blend_bridge.py` creates a marked `_Potter Taper Proxy`
//! object (`taper_objects_by_data`) and excludes marked proxies from the imported `objects` dump
//! (`object_dump`). This test checks the Blender-side proxy and that it never becomes a Potter node.

use std::{env, error::Error, path::PathBuf, process::Command};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/blender_script.rs"]
mod blender_script;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use blender_script::run_blender_script;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    exchange::{
        ExportOptions,
        blend::{export_blend, import_blend},
    },
    model::{
        CurveData, CurveDimensions, CurveFillMode, CurveFillRule, CurveFillSolver,
        CurveTaperRadiusMode, Id, SceneDoc,
    },
    ops,
    store::Project,
};
use serde_json::json;
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";
const TAPER_NODE_ID: &str = "taper_profile";
const TAPER_DATA_ID: &str = "taper_profile_data";
const HOST_NODE_ID: &str = "tapered_host";
const HOST_DATA_ID: &str = "tapered_host_data";
const HOST_NAME: &str = "Tapered Host";
const FILLED_NODE_ID: &str = "filled_profile";
const FILLED_NAME: &str = "Filled Profile";
const SPLINE_RESOLUTIONS: [u32; 2] = [3, 9];

const BLENDER_READBACK: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"curve exchange round-trip requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.open_mainfile(filepath=str(root / "curves.blend"))
host = bpy.data.objects.get("Tapered Host")
assert host is not None and host.type == "CURVE", "tapered host was not exported"
assert host.data.taper_object is not None, "taper object reference was not exported"
proxy = host.data.taper_object
assert bpy.context.scene.collection.objects.get(proxy.name) is proxy, "taper proxy is not linked to the scene"
assert proxy.get("_potter.taper_proxy") is True, proxy.name
assert proxy.name.startswith("_Potter Taper Proxy "), proxy.name
assert host.data.taper_radius_mode == "MULTIPLY", host.data.taper_radius_mode
assert host.data.use_map_taper is True
filled = bpy.data.objects.get("Filled Profile")
assert filled is not None and filled.type == "CURVE", "filled curve was not exported"
assert filled.data.dimensions == "2D", filled.data.dimensions
assert filled.data.fill_mode == "FRONT", filled.data.fill_mode
assert filled.data.resolution_u == 6, filled.data.resolution_u
assert filled.data.fill_solver == "SWEEP_LINE", filled.data.fill_solver
assert filled.data.fill_rule == "NONZERO", filled.data.fill_rule
resolutions = [spline.resolution_u for spline in filled.data.splines]
assert resolutions == [3, 9], resolutions
"#;

const BLENDER_NONCURVE_TAPER: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"non-curve taper test requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

mesh_data = bpy.data.meshes.new("Mesh Taper Data")
mesh_data.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
mesh_taper = bpy.data.objects.new("Mesh Taper", mesh_data)
scene.collection.objects.link(mesh_taper)

def make_curve(name):
    data = bpy.data.curves.new(name + " Data", "CURVE")
    data.dimensions = "3D"
    data.bevel_depth = 0.2
    data.bevel_resolution = 2
    data.resolution_u = 1
    spline = data.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
    spline.points[1].co = (2.0, 0.0, 0.0, 1.0)
    spline.resolution_u = 1
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    return obj

host = make_curve("Noncurve Taper")
host.data.taper_object = mesh_taper
baseline = make_curve("No Taper Baseline")
bpy.context.view_layer.update()

def sample(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return (
            [tuple(float(value) for value in vertex.co) for vertex in mesh.vertices],
            [tuple(int(index) for index in polygon.vertices) for polygon in mesh.polygons],
        )
    finally:
        evaluated.to_mesh_clear()

assert sample(host) == sample(baseline), "Blender non-curve taper must use factor 1.0"
bpy.ops.wm.save_as_mainfile(filepath=str(root / "noncurve_taper.blend"))
"#;

const BLENDER_MIXED_CURVES_SOURCE: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"mixed Curves round-trip requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
legacy = bpy.data.curves.new("Mixed Catmull Bezier Data", "CURVE")
legacy.dimensions = "3D"
legacy.twist_mode = "Z_UP"
curve_points = (
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
    ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (2.0, 1.0, 2.0)),
)
for curve_index, coordinates in enumerate(curve_points):
    spline = legacy.splines.new("BEZIER")
    spline.bezier_points.add(len(coordinates) - 1)
    spline.resolution_u = (5, 7)[curve_index]
    spline.use_cyclic_u = (False, True)[curve_index]
    for point, coordinate in zip(spline.bezier_points, coordinates):
        point.co = coordinate

poly = legacy.splines.new("POLY")
poly.points.add(2)
poly.resolution_u = 6
for point, coordinate in zip(
    poly.points,
    ((0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (2.0, 1.0, 2.0)),
):
    point.co = (*coordinate, 1.0)
obj = bpy.data.objects.new("Mixed Catmull Bezier", legacy)
scene.collection.objects.link(obj)
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.convert(target="CURVES")
obj = bpy.context.view_layer.objects.active

tree = bpy.data.node_groups.new("Make first spline Catmull", "GeometryNodeTree")
tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
input_node = tree.nodes.new("NodeGroupInput")
output_node = tree.nodes.new("NodeGroupOutput")
set_type = tree.nodes.new("GeometryNodeCurveSplineType")
set_type.spline_type = "CATMULL_ROM"
index = tree.nodes.new("GeometryNodeInputIndex")
compare = tree.nodes.new("FunctionNodeCompare")
compare.data_type = "INT"
compare.operation = "EQUAL"
compare.inputs["B"].default_value = 0
tree.links.new(input_node.outputs["Geometry"], set_type.inputs["Curve"])
tree.links.new(index.outputs["Index"], compare.inputs["A"])
tree.links.new(compare.outputs["Result"], set_type.inputs["Selection"])
set_nurbs = tree.nodes.new("GeometryNodeCurveSplineType")
set_nurbs.spline_type = "NURBS"
compare_nurbs = tree.nodes.new("FunctionNodeCompare")
compare_nurbs.data_type = "INT"
compare_nurbs.operation = "EQUAL"
compare_nurbs.inputs["B"].default_value = 2
tree.links.new(index.outputs["Index"], compare_nurbs.inputs["A"])
tree.links.new(compare_nurbs.outputs["Result"], set_nurbs.inputs["Selection"])
tree.links.new(set_type.outputs["Curve"], set_nurbs.inputs["Curve"])
tree.links.new(set_nurbs.outputs["Curve"], output_node.inputs["Geometry"])
modifier = obj.modifiers.new("Set first spline to Catmull", "NODES")
modifier.node_group = tree
bpy.context.view_layer.update()
bpy.ops.object.modifier_apply(modifier=modifier.name)
data = obj.data
assert [item.value for item in data.attributes["curve_type"].data] == [0, 2, 3]

def set_attribute(name, data_type, domain, values):
    attribute = data.attributes.get(name)
    if attribute is None:
        attribute = data.attributes.new(name=name, type=data_type, domain=domain)
    assert (attribute.data_type, attribute.domain) == (data_type, domain), name
    assert len(attribute.data) == len(values), name
    for item, value in zip(attribute.data, values):
        if hasattr(item, "vector"):
            item.vector = value
        else:
            item.value = value

set_attribute("cyclic", "BOOLEAN", "CURVE", [False, True, False])
set_attribute("resolution", "INT", "CURVE", [5, 7, 6])
set_attribute("nurbs_order", "INT8", "CURVE", [4, 4, 3])
set_attribute("knots_mode", "INT8", "CURVE", [0, 0, 1])
set_attribute("normal_mode", "INT8", "CURVE", [2, 1, 1])
set_attribute("radius", "FLOAT", "POINT", [
    0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 0.25, 0.5, 0.75,
])
set_attribute("tilt", "FLOAT", "POINT", [
    -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 0.25, 0.5, 0.75,
])
set_attribute("nurbs_weight", "FLOAT", "POINT", [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.5, 2.0,
])
set_attribute("handle_left", "FLOAT_VECTOR", "POINT", [
    (-0.25, 0.0, 0.0), (0.0, -0.25, 0.0), (0.25, 0.0, 0.0),
    (0.0, 0.0, 0.75), (0.75, 0.75, 0.75), (1.75, 0.75, 1.75),
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
])
set_attribute("handle_right", "FLOAT_VECTOR", "POINT", [
    (0.25, 0.0, 0.0), (1.0, 0.25, 0.0), (2.25, 1.0, 0.0),
    (0.0, 0.0, 1.25), (1.25, 1.0, 1.0), (2.25, 1.0, 2.25),
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
])
set_attribute("handle_type_left", "INT8", "POINT", [0, 1, 2, 3, 1, 0, 0, 0, 0])
set_attribute("handle_type_right", "INT8", "POINT", [1, 2, 3, 0, 2, 1, 0, 0, 0])
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(root / "mixed_curves.blend"))
"#;

const BLENDER_MIXED_CURVES_READBACK: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"mixed Curves round-trip requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.open_mainfile(filepath=str(root / "potter_mixed_curves.blend"))
obj = bpy.data.objects.get("Mixed Catmull Bezier")
assert obj is not None and obj.type == "CURVES", "Potter did not restore the native Curves object"
data = obj.data
assert [curve.points_length for curve in data.curves] == [3, 3, 3]
assert [
    [tuple(float(value) for value in point.position) for point in curve.points]
    for curve in data.curves
] == [
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0)],
    [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (2.0, 1.0, 2.0)],
    [(0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (2.0, 1.0, 2.0)],
]

def attribute_values(name):
    attribute = data.attributes.get(name)
    assert attribute is not None, f"Potter omitted Curves attribute {name!r}"
    values = []
    for item in attribute.data:
        if hasattr(item, "vector"):
            values.append(tuple(float(value) for value in item.vector))
        else:
            values.append(item.value)
    return attribute.domain, attribute.data_type, values

expected = {
    "curve_type": ("CURVE", "INT8", [0, 2, 3]),
    "cyclic": ("CURVE", "BOOLEAN", [False, True, False]),
    "resolution": ("CURVE", "INT", [5, 7, 6]),
    "nurbs_order": ("CURVE", "INT8", [4, 4, 3]),
    "knots_mode": ("CURVE", "INT8", [0, 0, 1]),
    "normal_mode": ("CURVE", "INT8", [2, 1, 1]),
    "nurbs_weight": ("POINT", "FLOAT", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.5, 2.0]),
    "radius": ("POINT", "FLOAT", [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 0.25, 0.5, 0.75]),
    "tilt": ("POINT", "FLOAT", [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 0.25, 0.5, 0.75]),
    "handle_left": ("POINT", "FLOAT_VECTOR", [
        (-0.25, 0.0, 0.0), (0.0, -0.25, 0.0), (0.25, 0.0, 0.0),
        (0.0, 0.0, 0.75), (0.75, 0.75, 0.75), (1.75, 0.75, 1.75),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
    ]),
    "handle_right": ("POINT", "FLOAT_VECTOR", [
        (0.25, 0.0, 0.0), (1.0, 0.25, 0.0), (2.25, 1.0, 0.0),
        (0.0, 0.0, 1.25), (1.25, 1.0, 1.0), (2.25, 1.0, 2.25),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
    ]),
    "handle_type_left": ("POINT", "INT8", [0, 1, 2, 3, 1, 0, 0, 0, 0]),
    "handle_type_right": ("POINT", "INT8", [1, 2, 3, 0, 2, 1, 0, 0, 0]),
}
for name, expected_value in expected.items():
    assert attribute_values(name) == expected_value, (name, attribute_values(name))
"#;

const BLENDER_CUSTOM_KNOTS_SOURCE: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"custom knot round-trip requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
data = bpy.data.hair_curves.new("Custom Knot Curves Data")
data.add_curves([3])
for point, position in zip(data.points, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0))):
    point.position = position

def set_attribute(name, data_type, domain, values):
    attribute = data.attributes.get(name)
    if attribute is None:
        attribute = data.attributes.new(name=name, type=data_type, domain=domain)
    assert len(attribute.data) == len(values), name
    for item, value in zip(attribute.data, values):
        item.value = value

set_attribute("curve_type", "INT8", "CURVE", [3])
set_attribute("cyclic", "BOOLEAN", "CURVE", [False])
set_attribute("resolution", "INT", "CURVE", [4])
set_attribute("nurbs_order", "INT8", "CURVE", [3])
set_attribute("knots_mode", "INT8", "CURVE", [4])
set_attribute("normal_mode", "INT8", "CURVE", [0])
set_attribute("nurbs_weight", "FLOAT", "POINT", [1.0, 0.5, 1.5])
set_attribute("radius", "FLOAT", "POINT", [1.0, 0.75, 0.5])
set_attribute("tilt", "FLOAT", "POINT", [0.0, 0.25, 0.5])
data.use_fake_user = True
obj = bpy.data.objects.new("Custom Knot Curves", data)
obj.use_fake_user = True
bpy.ops.wm.save_as_mainfile(filepath=str(root / "custom_knots.blend"))
"#;

const BLENDER_CUSTOM_KNOTS_READBACK: &str = r#"
import bpy
import sys
from pathlib import Path

root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.open_mainfile(filepath=str(root / "potter_custom_knots.blend"))
obj = bpy.data.objects.get("Custom Knot Curves")
assert obj is not None and obj.type == "CURVES"
data = obj.data
assert [curve.points_length for curve in data.curves] == [3]
assert [
    [tuple(float(value) for value in point.position) for point in curve.points]
    for curve in data.curves
] == [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0)]]
assert [item.value for item in data.attributes["curve_type"].data] == [3]
assert [item.value for item in data.attributes["knots_mode"].data] == [0]
assert [item.value for item in data.attributes["nurbs_order"].data] == [3]
assert [item.value for item in data.attributes["nurbs_weight"].data] == [1.0, 0.5, 1.5]
"#;

const BLENDER_LEGACY_CUSTOM_KNOTS_SOURCE: &str = r#"
import bpy
import ctypes
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"legacy custom knots require Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
data = bpy.data.curves.new("Legacy Custom Knot Curve Data", "CURVE")
spline = data.splines.new("NURBS")
spline.points.add(2)
for point, coordinate, weight in zip(
    spline.points,
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0)),
    (1.0, 0.5, 1.5),
):
    point.co = (*coordinate, weight)
spline.order_u = 3
spline.use_endpoint_u = True
knot_flag = ctypes.c_uint16.from_address(
    spline.as_pointer() + 2 * ctypes.sizeof(ctypes.c_void_p) + 28)
knot_flag.value |= 1 << 3
assert knot_flag.value & (1 << 3)
data.use_fake_user = True
obj = bpy.data.objects.new("Legacy Custom Knot Curve", data)
obj.use_fake_user = True
bpy.ops.wm.save_as_mainfile(filepath=str(root / "legacy_custom_knots.blend"))
"#;

const BLENDER_LEGACY_CUSTOM_KNOTS_READBACK: &str = r#"
import bpy
import ctypes
import sys
from pathlib import Path

root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.open_mainfile(filepath=str(root / "potter_legacy_custom_knots.blend"))
obj = bpy.data.objects.get("Legacy Custom Knot Curve")
assert obj is not None and obj.type == "CURVE"
spline = obj.data.splines[0]
knot_flag = ctypes.c_uint16.from_address(
    spline.as_pointer() + 2 * ctypes.sizeof(ctypes.c_void_p) + 28)
assert not knot_flag.value & (1 << 3), "lossy export retained the custom-knot flag"
assert spline.use_endpoint_u
assert [tuple(float(value) for value in point.co[:3]) for point in spline.points] == [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0)
]
"#;

fn require_blender() -> bool {
    env::var_os("POTTER_REQUIRE_BLENDER").is_some_and(|value| !value.is_empty())
}

fn blender_522() -> TestResult<Option<PathBuf>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping curve exchange round-trip: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(&blender);
    command.arg("--version");
    let output = process::run_guarded(command)?;
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if version.starts_with("Blender 5.2.2") {
        return Ok(Some(blender));
    }
    assert!(
        !require_blender(),
        "POTTER_REQUIRE_BLENDER is set but curve exchange round-trip requires Blender 5.2.2, found `{version}`"
    );
    eprintln!("skipping curve exchange round-trip: found `{version}`, need Blender 5.2.2");
    Ok(None)
}

fn curve_doc() -> TestResult<SceneDoc> {
    let doc = SceneDoc::new(SCENE_ID.to_owned());
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": [
            {
                "op": "curve.create",
                "id": TAPER_NODE_ID,
                "data_id": TAPER_DATA_ID,
                "name": "Taper Profile",
                "dimensions": "two_d",
                "splines": [{
                    "type": "poly",
                    "points": [
                        {"co": [0.0, 0.2, 0.0]},
                        {"co": [0.5, 0.55, 0.0]},
                        {"co": [1.0, 1.0, 0.0]}
                    ],
                    "order": 2,
                    "resolution": 4,
                    "use_endpoint": true
                }]
            },
            {
                "op": "curve.create",
                "id": HOST_NODE_ID,
                "data_id": HOST_DATA_ID,
                "name": HOST_NAME,
                "dimensions": "three_d",
                "bevel_depth": 0.15,
                "bevel_resolution": 2,
                "taper": TAPER_DATA_ID,
                "taper_radius_mode": "multiply",
                "use_map_taper": true,
                "splines": [{
                    "type": "poly",
                    "points": [
                        {"co": [0.0, 0.0, 0.0]},
                        {"co": [1.0, 0.0, 0.0]},
                        {"co": [3.0, 1.0, 0.0]}
                    ],
                    "order": 2,
                    "resolution": 5,
                    "use_endpoint": true
                }]
            },
            {
                "op": "curve.create",
                "id": FILLED_NODE_ID,
                "name": FILLED_NAME,
                "dimensions": "two_d",
                "resolution_u": 6,
                "fill_mode": "front",
                "fill_solver": "sweep_line",
                "fill_rule": "nonzero",
                "splines": [
                    {
                        "type": "poly",
                        "points": [
                            {"co": [0.0, 0.0, 0.0]},
                            {"co": [3.0, 0.0, 0.0]},
                            {"co": [3.0, 3.0, 0.0]},
                            {"co": [0.0, 3.0, 0.0]}
                        ],
                        "order": 2,
                        "cyclic": true,
                        "resolution": SPLINE_RESOLUTIONS[0],
                        "use_endpoint": true
                    },
                    {
                        "type": "poly",
                        "points": [
                            {"co": [5.0, 0.0, 0.0]},
                            {"co": [7.0, 0.0, 0.0]},
                            {"co": [7.0, 2.0, 0.0]},
                            {"co": [5.0, 2.0, 0.0]}
                        ],
                        "order": 2,
                        "cyclic": true,
                        "resolution": SPLINE_RESOLUTIONS[1],
                        "use_endpoint": true
                    }
                ]
            }
        ]
    });
    let mut doc = ops::apply_batch(&doc, &batch)?.doc;
    let taper_node_id = Id::new(TAPER_NODE_ID)?;
    doc.nodes
        .remove(&taper_node_id)
        .ok_or("taper proxy source node was not created")?;
    let root_collection = doc.scenes[&doc.active_scene].root_collection.clone();
    doc.collections
        .get_mut(&root_collection)
        .ok_or("scene root collection is missing")?
        .objects
        .retain(|id| id != &taper_node_id);
    Ok(doc)
}

fn curve_for_node<'a>(doc: &'a SceneDoc, name: &str) -> TestResult<&'a CurveData> {
    let node = doc
        .nodes
        .values()
        .find(|node| node.name == name)
        .ok_or_else(|| format!("curve node `{name}` is missing"))?;
    let data_id = node
        .data
        .as_ref()
        .ok_or_else(|| format!("curve node `{name}` has no data-block"))?;
    doc.data_blocks
        .get(data_id)
        .and_then(|data| data.curve.as_ref())
        .ok_or_else(|| format!("curve node `{name}` has no curve data").into())
}

#[test]
fn curve_exchange_edits_round_trip_without_importing_taper_proxy_node() -> TestResult {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let project = Project::init(root.join("project"))?;
    let doc = curve_doc()?;

    let host_curve = curve_for_node(&doc, HOST_NAME)?;
    let taper_id = host_curve
        .taper
        .as_ref()
        .ok_or("tapered curve has no taper reference")?;
    assert_eq!(taper_id.as_str(), TAPER_DATA_ID);
    assert!(
        doc.data_blocks
            .get(taper_id)
            .and_then(|data| data.curve.as_ref())
            .is_some()
    );
    assert!(
        !doc.nodes
            .values()
            .any(|node| node.data.as_ref() == Some(taper_id))
    );

    let blend = root.join("curves.blend");
    let context = EvaluationContext::default();
    export_blend(
        &doc,
        &project,
        &blend,
        &ExportOptions {
            allow_lossy: false,
            pack: false,
            blender: Some(blender.as_path()),
            context: &context,
        },
    )?;
    run_blender_script(
        &blender,
        "read_curve_exchange.py",
        BLENDER_READBACK,
        root,
        "Blender curve exchange readback failed",
    )?;

    let imported = import_blend(&blend, Some(blender.as_path()))?;
    let imported_host = curve_for_node(&imported.doc, HOST_NAME)?;
    let imported_taper_id = imported_host
        .taper
        .as_ref()
        .ok_or("imported tapered curve has no taper reference")?;
    assert_eq!(imported_taper_id.as_str(), TAPER_DATA_ID);
    assert!(
        imported
            .doc
            .data_blocks
            .get(imported_taper_id)
            .and_then(|data| data.curve.as_ref())
            .is_some(),
        "imported taper reference does not resolve to curve data"
    );
    assert_eq!(imported_host.dimensions, CurveDimensions::ThreeD);
    assert_eq!(
        imported_host.taper_radius_mode,
        CurveTaperRadiusMode::Multiply
    );
    assert!(imported_host.use_map_taper);

    let imported_fill = curve_for_node(&imported.doc, FILLED_NAME)?;
    assert_eq!(imported_fill.dimensions, CurveDimensions::TwoD);
    assert_eq!(imported_fill.resolution_u, 6);
    assert_eq!(imported_fill.fill_mode, CurveFillMode::Front);
    assert_eq!(imported_fill.fill_solver, CurveFillSolver::SweepLine);
    assert_eq!(imported_fill.fill_rule, CurveFillRule::Nonzero);
    assert_eq!(
        imported_fill
            .splines
            .iter()
            .map(|spline| spline.resolution)
            .collect::<Vec<_>>(),
        SPLINE_RESOLUTIONS
    );

    assert_eq!(
        imported.doc.nodes.len(),
        2,
        "bridge-only taper proxy became a Potter node"
    );
    assert!(!imported.doc.nodes.values().any(|node| {
        node.name.starts_with("_Potter Taper Proxy")
            || node.data.as_ref() == Some(imported_taper_id)
    }));
    Ok(())
}

#[test]
fn non_curve_taper_blend_import_uses_blenders_unit_factor() -> TestResult {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "build_noncurve_taper.py",
        BLENDER_NONCURVE_TAPER,
        root,
        "Blender non-curve taper oracle generation failed",
    )?;
    let imported = import_blend(&root.join("noncurve_taper.blend"), Some(blender.as_path()))?;
    assert!(
        curve_for_node(&imported.doc, "Noncurve Taper")?
            .taper
            .is_none()
    );

    let node_id = |name: &str| -> TestResult<Id> {
        imported
            .doc
            .nodes
            .iter()
            .find(|(_, node)| node.name == name)
            .map(|(id, _)| id.clone())
            .ok_or_else(|| format!("curve node `{name}` is missing").into())
    };
    let host_id = node_id("Noncurve Taper")?;
    let baseline_id = node_id("No Taper Baseline")?;
    let snapshot = Snapshot::evaluate(&imported.doc, &EvaluationContext::default())?;
    let host = &snapshot.meshes[&host_id];
    let baseline = &snapshot.meshes[&baseline_id];
    assert_eq!(
        host.vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>(),
        baseline
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>()
    );
    assert_eq!(host.faces, baseline.faces);
    Ok(())
}

#[test]
fn native_catmull_rom_bezier_and_nurbs_curves_round_trip_through_potter() -> TestResult {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "build_mixed_curves.py",
        BLENDER_MIXED_CURVES_SOURCE,
        root,
        "Blender mixed Curves source generation failed",
    )?;

    let project = root.join("project");
    let project_arg = project.to_str().ok_or("project path is not UTF-8")?;
    let source = root.join("mixed_curves.blend");
    let source_arg = source.to_str().ok_or("source blend path is not UTF-8")?;
    let mut initialize = Command::new(env!("CARGO_BIN_EXE_pot"));
    initialize.args(["init", project_arg, "--json"]);
    let initialized = process::run_guarded(initialize)?;
    assert!(
        initialized.status.success(),
        "pot init failed: {}",
        String::from_utf8_lossy(&initialized.stderr)
    );

    let mut import = Command::new(env!("CARGO_BIN_EXE_pot"));
    import.args([
        "import",
        project_arg,
        "--file",
        source_arg,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
    ]);
    import.arg(&blender).arg("--json");
    let imported = process::run_guarded(import)?;
    assert!(
        imported.status.success(),
        "pot import failed: {}",
        String::from_utf8_lossy(&imported.stderr)
    );

    let exported = root.join("potter_mixed_curves.blend");
    let exported_arg = exported.to_str().ok_or("export blend path is not UTF-8")?;
    let mut export = Command::new(env!("CARGO_BIN_EXE_pot"));
    export.args([
        "export",
        project_arg,
        "--format",
        "blend",
        "--out",
        exported_arg,
        "--blender",
    ]);
    export.arg(&blender).arg("--json");
    let exported_result = process::run_guarded(export)?;
    assert!(
        exported_result.status.success(),
        "pot export failed: {}",
        String::from_utf8_lossy(&exported_result.stderr)
    );
    run_blender_script(
        &blender,
        "read_mixed_curves.py",
        BLENDER_MIXED_CURVES_READBACK,
        root,
        "Blender mixed Curves readback failed",
    )?;
    Ok(())
}

#[test]
fn custom_knot_nurbs_fallback_export_reports_loss_and_writes_standard_knots() -> TestResult {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "build_custom_knots.py",
        BLENDER_CUSTOM_KNOTS_SOURCE,
        root,
        "Blender custom-knot Curves source generation failed",
    )?;

    let project = root.join("custom_knots_project");
    let project_arg = project
        .to_str()
        .ok_or("custom-knot project path is not UTF-8")?;
    let mut initialize = Command::new(env!("CARGO_BIN_EXE_pot"));
    initialize.args(["init", project_arg, "--json"]);
    let initialized = process::run_guarded(initialize)?;
    assert!(initialized.status.success());
    let source_arg = root
        .join("custom_knots.blend")
        .to_str()
        .ok_or("custom-knot source path is not UTF-8")?
        .to_owned();
    let mut import = Command::new(env!("CARGO_BIN_EXE_pot"));
    import
        .args([
            "import",
            project_arg,
            "--file",
            &source_arg,
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--allow-lossy",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let imported = process::run_guarded(import)?;
    assert!(
        imported.status.success(),
        "custom-knot import failed: {}",
        String::from_utf8_lossy(&imported.stderr)
    );
    let import_report: serde_json::Value = serde_json::from_slice(&imported.stdout)?;
    assert!(
        import_report["result"]["losses"]
            .as_array()
            .is_some_and(|losses| losses
                .iter()
                .any(|loss| loss["feature_id"] == "hair_curves.nurbs.custom_knots")),
        "custom-knot import loss was not reported"
    );

    let strict_output = root.join("strict_custom_knots.blend");
    let strict_output_arg = strict_output
        .to_str()
        .ok_or("strict custom-knot output path is not UTF-8")?;
    let mut strict_export = Command::new(env!("CARGO_BIN_EXE_pot"));
    strict_export
        .args([
            "export",
            project_arg,
            "--format",
            "blend",
            "--out",
            strict_output_arg,
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let rejected = process::run_guarded(strict_export)?;
    assert!(!rejected.status.success());
    let rejected: serde_json::Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejected["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert!(
        rejected["error"]["details"]["losses"]
            .as_array()
            .is_some_and(|losses| losses
                .iter()
                .any(|loss| loss["feature_id"] == "hair_curves.nurbs.custom_knots")),
        "strict export did not report the custom-knot loss"
    );

    let output = root.join("potter_custom_knots.blend");
    let output_arg = output
        .to_str()
        .ok_or("custom-knot output path is not UTF-8")?;
    let mut export = Command::new(env!("CARGO_BIN_EXE_pot"));
    export
        .args([
            "export",
            project_arg,
            "--format",
            "blend",
            "--out",
            output_arg,
            "--allow-lossy",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let exported = process::run_guarded(export)?;
    assert!(
        exported.status.success(),
        "lossy custom-knot export failed: {}",
        String::from_utf8_lossy(&exported.stderr)
    );
    let export_report: serde_json::Value = serde_json::from_slice(&exported.stdout)?;
    assert!(
        export_report["result"]["losses"]
            .as_array()
            .is_some_and(|losses| losses
                .iter()
                .any(|loss| loss["feature_id"] == "hair_curves.nurbs.custom_knots")),
        "lossy export did not report the custom-knot loss"
    );
    run_blender_script(
        &blender,
        "read_custom_knots.py",
        BLENDER_CUSTOM_KNOTS_READBACK,
        root,
        "Blender custom-knot Curves readback failed",
    )?;
    Ok(())
}

#[test]
fn legacy_curve_custom_knots_report_loss_and_export_as_standard_knots() -> TestResult {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "build_legacy_custom_knots.py",
        BLENDER_LEGACY_CUSTOM_KNOTS_SOURCE,
        root,
        "Blender legacy custom-knot source generation failed",
    )?;

    let project = root.join("legacy_custom_knots_project");
    let project_arg = project
        .to_str()
        .ok_or("legacy custom-knot project path is not UTF-8")?;
    let source_arg = root
        .join("legacy_custom_knots.blend")
        .to_str()
        .ok_or("legacy custom-knot source path is not UTF-8")?
        .to_owned();
    let mut initialize = Command::new(env!("CARGO_BIN_EXE_pot"));
    initialize.args(["init", project_arg, "--json"]);
    let initialized = process::run_guarded(initialize)?;
    assert!(initialized.status.success());

    let mut import = Command::new(env!("CARGO_BIN_EXE_pot"));
    import
        .args([
            "import",
            project_arg,
            "--file",
            &source_arg,
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--allow-lossy",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let imported = process::run_guarded(import)?;
    assert!(
        imported.status.success(),
        "legacy custom-knot import failed: {}",
        String::from_utf8_lossy(&imported.stderr)
    );
    let import_report: serde_json::Value = serde_json::from_slice(&imported.stdout)?;
    assert!(
        import_report["result"]["losses"]
            .as_array()
            .is_some_and(|losses| losses
                .iter()
                .any(|loss| loss["feature_id"] == "blender.curve.nurbs.custom_knots")),
        "legacy custom-knot import loss was not reported"
    );

    let strict_output = root.join("strict_legacy_custom_knots.blend");
    let strict_output_arg = strict_output
        .to_str()
        .ok_or("strict legacy custom-knot output path is not UTF-8")?;
    let mut strict_export = Command::new(env!("CARGO_BIN_EXE_pot"));
    strict_export
        .args([
            "export",
            project_arg,
            "--format",
            "blend",
            "--out",
            strict_output_arg,
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let rejected = process::run_guarded(strict_export)?;
    assert!(!rejected.status.success());
    let rejected: serde_json::Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejected["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert!(
        rejected["error"]["details"]["losses"]
            .as_array()
            .is_some_and(|losses| losses
                .iter()
                .any(|loss| loss["feature_id"] == "blender.curve.nurbs.custom_knots")),
        "strict legacy custom-knot export did not report the loss"
    );

    let output = root.join("potter_legacy_custom_knots.blend");
    let output_arg = output
        .to_str()
        .ok_or("lossy legacy custom-knot output path is not UTF-8")?;
    let mut export = Command::new(env!("CARGO_BIN_EXE_pot"));
    export
        .args([
            "export",
            project_arg,
            "--format",
            "blend",
            "--out",
            output_arg,
            "--allow-lossy",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let exported = process::run_guarded(export)?;
    assert!(
        exported.status.success(),
        "lossy legacy custom-knot export failed: {}",
        String::from_utf8_lossy(&exported.stderr)
    );
    let export_report: serde_json::Value = serde_json::from_slice(&exported.stdout)?;
    assert!(
        export_report["result"]["losses"]
            .as_array()
            .is_some_and(|losses| losses
                .iter()
                .any(|loss| loss["feature_id"] == "blender.curve.nurbs.custom_knots")),
        "lossy legacy Curve export did not report the custom-knot loss"
    );
    run_blender_script(
        &blender,
        "read_legacy_custom_knots.py",
        BLENDER_LEGACY_CUSTOM_KNOTS_READBACK,
        root,
        "Blender lossy legacy custom-knot readback failed",
    )?;
    Ok(())
}
