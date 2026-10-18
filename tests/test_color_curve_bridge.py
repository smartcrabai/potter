import ast
from pathlib import Path
import unittest


BRIDGE_PATH = Path(__file__).parents[1] / "crates/potter-core/src/exchange/blend_bridge.py"


def bridge_helpers():
    tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
    names = {"_curve_extend_to_blender", "_restore_color_curve_clip"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise RuntimeError("color curve bridge helpers are missing")
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(BRIDGE_PATH), "exec"), namespace)
    return namespace


class FakeMapping:
    def __init__(self):
        self._clip_min_x = 0.0
        self._clip_min_y = 0.0
        self._clip_max_x = 1.0
        self._clip_max_y = 1.0

    @property
    def clip_min_x(self):
        return self._clip_min_x

    @clip_min_x.setter
    def clip_min_x(self, value):
        self._clip_min_x = max(-100.0, min(float(value), self._clip_max_x))

    @property
    def clip_min_y(self):
        return self._clip_min_y

    @clip_min_y.setter
    def clip_min_y(self, value):
        self._clip_min_y = max(-100.0, min(float(value), self._clip_max_y))

    @property
    def clip_max_x(self):
        return self._clip_max_x

    @clip_max_x.setter
    def clip_max_x(self, value):
        self._clip_max_x = max(self._clip_min_x, min(100.0, max(-100.0, float(value))))

    @property
    def clip_max_y(self):
        return self._clip_max_y

    @clip_max_y.setter
    def clip_max_y(self, value):
        self._clip_max_y = max(self._clip_min_y, min(100.0, max(-100.0, float(value))))


class ColorCurveBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = bridge_helpers()

    def test_serialized_extension_boolean_maps_to_blender_enum(self):
        convert = self.helpers["_curve_extend_to_blender"]
        self.assertEqual(convert({"extend_extrapolate": True}), "EXTRAPOLATED")
        self.assertEqual(convert({"extend_extrapolate": False}), "HORIZONTAL")

    def test_clip_rectangle_restores_minima_after_expanding_maxima(self):
        mapping = FakeMapping()
        self.helpers["_restore_color_curve_clip"](mapping, [2.0, 0.0, 3.0, 1.0])
        self.assertEqual(
            [mapping.clip_min_x, mapping.clip_min_y, mapping.clip_max_x, mapping.clip_max_y],
            [2.0, 0.0, 3.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
