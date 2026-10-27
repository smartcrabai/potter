import bpy
import json
import sys
from pathlib import Path


ROOT = Path(sys.argv[sys.argv.index("--") + 1])
REQUIRED_ATTRIBUTES = (
    "radius",
    "tilt",
    "handle_left",
    "handle_right",
    "handle_type_left",
    "handle_type_right",
    "nurbs_weight",
    "curve_type",
    "cyclic",
    "resolution",
    "nurbs_order",
    "knots_mode",
    "normal_mode",
)

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"curve.convert oracle requires Blender 5.2.2, found {bpy.app.version_string}")

bpy.ops.wm.read_factory_settings(use_empty=True)

def poly_object(name, coordinates, dimensions="3D", cyclic=False):
    data = bpy.data.curves.new(f"{name} Data", "CURVE")
    data.dimensions = dimensions
    spline = data.splines.new("POLY")
    spline.points.add(len(coordinates) - 1)
    for point, coordinate in zip(spline.points, coordinates):
        point.co = (*coordinate, 1.0)
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    return obj


def convert_object(source):
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    source.select_set(True)
    bpy.context.view_layer.objects.active = source
    bpy.ops.object.convert(target="CURVES")
    converted = bpy.context.view_layer.objects.active
    if converted is None:
        raise RuntimeError(f"Blender conversion lost active object {source.name!r}")
    return converted


def capture_curves(converted, require_all_attributes=False):
    curves_data = converted.data
    attributes = {}
    for name in REQUIRED_ATTRIBUTES:
        attribute = curves_data.attributes.get(name)
        if attribute is None:
            if require_all_attributes:
                raise RuntimeError(f"Blender 5.2.2 conversion omitted required attribute {name!r}")
            continue
        values = []
        for item in attribute.data:
            if hasattr(item, "vector"):
                value = list(item.vector)
            elif isinstance(item.value, bytes):
                value = item.value.decode("utf-8")
            else:
                value = item.value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = float(value) if attribute.data_type.startswith("FLOAT") else int(value)
            values.append(value)
        attributes[name] = {
            "domain": attribute.domain,
            "data_type": attribute.data_type,
            "values": values,
        }
    return {
        "object_name": converted.name,
        "materials": [material.name if material else None for material in curves_data.materials],
        "curves": [
            {
                "points": [list(point.position) for point in curve.points],
                "radius": float(curves_data.points[curve.first_point_index].radius)
                if curve.points_length
                else 0.0,
            }
            for curve in curves_data.curves
        ],
        "attributes": attributes,
    }
scene = bpy.context.scene

material_names = ("Curve Convert Material A", "Curve Convert Material B")
materials = [bpy.data.materials.new(name) for name in material_names]
data = bpy.data.curves.new("Mixed Legacy Curve Data", "CURVE")
data.dimensions = "3D"
data.resolution_u = 7
data.twist_mode = "Z_UP"
data.materials.clear()
for material in materials:
    data.materials.append(material)

bezier = data.splines.new("BEZIER")
bezier.bezier_points.add(2)
bezier_points = (
    ((-4.0, 0.5, 1.0), (-4.6, 0.1, 0.8), (-3.7, 0.2, 1.2), 0.35, -0.4, "FREE", "VECTOR"),
    ((-2.1, 1.7, -0.5), (-2.8, 1.2, -0.8), (-1.8, 2.2, -0.1), 1.25, 0.65, "ALIGNED", "FREE"),
    ((-0.3, 0.4, 1.4), (-0.9, -0.1, 1.0), (0.4, 0.8, 1.7), 2.1, 1.15, "VECTOR", "ALIGNED"),
)
for point, (co, left, right, radius, tilt, left_type, right_type) in zip(bezier.bezier_points, bezier_points):
    point.co = co
    point.handle_left = left
    point.handle_right = right
    point.handle_left_type = left_type
    point.handle_right_type = right_type
    point.radius = radius
    point.tilt = tilt
bezier.resolution_u = 5
bezier.use_cyclic_u = False

poly = data.splines.new("POLY")
poly.points.add(2)
poly_points = (
    ((1.0, -2.0, 0.25), 2.4, 1.0, 0.15),
    ((2.4, -1.1, 1.75), 0.6, 1.4, -0.35),
    ((3.8, -2.3, 0.65), 3.1, 0.7, 0.85),
)
for point, (co, weight, radius, tilt) in zip(poly.points, poly_points):
    point.co = (*co, weight)
    point.radius = radius
    point.tilt = tilt
poly.resolution_u = 3
poly.use_cyclic_u = True

nurbs = data.splines.new("NURBS")
nurbs.points.add(3)
nurbs_points = (
    ((4.7, 1.0, -1.0), 1.0, 0.55, -0.7),
    ((5.8, 2.2, 0.5), 1.5, 0.9, 0.25),
    ((7.2, 1.3, 1.8), 2.25, 1.2, 0.95),
    ((8.6, 2.7, -0.2), 0.8, 1.6, -1.1),
)
for point, (co, weight, radius, tilt) in zip(nurbs.points, nurbs_points):
    point.co = (*co, weight)
    point.radius = radius
    point.tilt = tilt
nurbs.resolution_u = 6
nurbs.order_u = 3
nurbs.use_endpoint_u = True
nurbs.use_bezier_u = False
nurbs.use_cyclic_u = False

obj = bpy.data.objects.new("Mixed Legacy Curve", data)
scene.collection.objects.link(obj)
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

poly_only = poly_object(
    "Poly-only Legacy Curve",
    ((-1.0, 0.0, 0.5), (0.5, 1.25, -0.25), (2.0, 0.0, 1.0)),
    cyclic=True,
)
poly_only.data.splines[0].resolution_u = 4

bezier_only_data = bpy.data.curves.new("Bezier-only Legacy Curve Data", "CURVE")
bezier_only_data.dimensions = "3D"
bezier_only = bezier_only_data.splines.new("BEZIER")
bezier_only.bezier_points.add(2)
for point, co in zip(
    bezier_only.bezier_points,
    ((-2.0, 0.0, 0.0), (0.0, 1.5, 0.5), (2.0, 0.0, 1.0)),
):
    point.co = co
    point.handle_left_type = "FREE"
    point.handle_right_type = "FREE"
    point.handle_left = (co[0] - 0.5, co[1] - 0.5, co[2])
    point.handle_right = (co[0] + 0.5, co[1] + 0.5, co[2])
bezier_only.resolution_u = 5
bezier_only_obj = bpy.data.objects.new("Bezier-only Legacy Curve", bezier_only_data)
scene.collection.objects.link(bezier_only_obj)

bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "source.blend"))

converted_payloads = {}
for source in (obj, poly_only, bezier_only_obj):
    converted = convert_object(source)
    if converted.type != "CURVES":
        raise RuntimeError(f"Blender did not convert {source.name!r} to Curves")
    converted_payloads[source.name] = capture_curves(
        converted,
        require_all_attributes=(source is obj),
    )

mixed = converted_payloads[obj.name]
assert len(mixed["curves"]) == 3
assert mixed["attributes"]["curve_type"]["values"] == [2, 1, 3]
assert mixed["attributes"]["nurbs_weight"]["values"][:6] == [1.0] * 6

bevel_profile = poly_object(
    "Bevel Profile",
    ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)),
    cyclic=True,
)
taper_profile = poly_object(
    "Taper Profile",
    ((0.0, 0.25, 0.0), (0.5, 0.6, 0.0), (1.0, 1.0, 0.0)),
)


def conversion_probe(name, settings, dimensions="3D", coordinates=None, cyclic=False):
    if coordinates is None:
        coordinates = ((0.0, 0.0, 0.0), (1.0, 0.5, 0.0), (2.0, 0.0, 0.5))
    source = poly_object(name, coordinates, dimensions=dimensions, cyclic=cyclic)
    for setting, value in settings.items():
        setattr(source.data, setting, value)
    return convert_object(source).type


conversion_probes = {
    "bevel_depth": conversion_probe("Probe Bevel Depth", {"bevel_depth": 0.2}),
    "extrude": conversion_probe("Probe Extrude", {"extrude": 0.2}),
    "bevel_object": conversion_probe(
        "Probe Bevel Object",
        {"bevel_mode": "OBJECT", "bevel_object": bevel_profile},
    ),
    "2d_fill": conversion_probe(
        "Probe 2D Fill",
        {"fill_mode": "BOTH"},
        dimensions="2D",
        coordinates=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)),
        cyclic=True,
    ),
    "taper": conversion_probe("Probe Taper", {"taper_object": taper_profile}),
    "use_path_path_duration": conversion_probe(
        "Probe Path", {"use_path": True, "path_duration": 37}
    ),
    "eval_time": conversion_probe("Probe Evaluation Time", {"eval_time": 1.75}),
}
assert conversion_probes == {
    "bevel_depth": "CURVE",
    "extrude": "CURVE",
    "bevel_object": "CURVE",
    "2d_fill": "CURVE",
    "taper": "CURVES",
    "use_path_path_duration": "CURVES",
    "eval_time": "CURVES",
}

expected = {
    "blender_version": bpy.app.version_string,
    **mixed,
    "attribute_names_by_object": {
        name: sorted(payload["attributes"])
        for name, payload in converted_payloads.items()
    },
    "conversion_probes": conversion_probes,
}
with (ROOT / "blender_expected.json").open("w", encoding="utf-8") as output:
    json.dump(expected, output, indent=2, sort_keys=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bevel_profile = poly_object(
    "Imported Bevel Profile",
    ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)),
    cyclic=True,
)
bevel_source = poly_object(
    "Bevel Object Source",
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.5, 0.0)),
)
bevel_source.data.bevel_object = bevel_profile
bevel_source.data.bevel_mode = "OBJECT"
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "bevel_object_source.blend"))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
extrude_source = poly_object(
    "Extrude Source",
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.5, 0.0)),
)
extrude_source.data.extrude = 0.2
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "extrude_source.blend"))

