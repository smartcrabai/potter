"""Checks that the Wire overlay is not view-transformed: the Standard and AgX captures of the same scene are
byte-identical (the Workbench overlay bypasses the view transform).

Run from this directory, with the Blender 5.2.2 executable as the BLENDER variable:
  BLENDER=/Applications/Blender.app/Contents/MacOS/Blender python3 check_view_transform.py
Exits non-zero when the two captures differ. The Rust tests do not repeat this check.
"""
import filecmp
import os
import subprocess
import sys
import tempfile

BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
HERE = os.path.dirname(os.path.abspath(__file__))
scene = os.path.join(HERE, "wire_scene.blend")
capture = os.path.join(HERE, "capture_wire.py")
if not os.path.exists(scene):
    subprocess.run([BLENDER, "--factory-startup", "--background", "--python",
                    os.path.join(HERE, "build_scene.py"), "--", scene], check=True)
with tempfile.TemporaryDirectory() as directory:
    outputs = {}
    for transform in ("Standard", "AgX"):
        png = os.path.join(directory, transform + ".png")
        json_out = os.path.join(directory, transform + ".json")
        subprocess.run(
            [BLENDER, "--factory-startup", "--window-geometry", "0", "0", "1000", "800", scene,
             "--python", capture, "--", png, json_out, transform, "None", "None", "factory"],
            check=True,
        )
        outputs[transform] = png
    same = filecmp.cmp(outputs["Standard"], outputs["AgX"], shallow=False)
print("view transform: Standard and AgX captures", "identical" if same else "DIFFER")
sys.exit(0 if same else 1)
