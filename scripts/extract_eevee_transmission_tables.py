#!/usr/bin/env python3
"""Extract EEVEE's GGX glass lookup tables into a little-endian f32 binary asset.

Usage:
    python3 scripts/extract_eevee_transmission_tables.py <eevee_lut.cc> <output.f32le>

The input is `source/blender/draw/engines/eevee/eevee_lut.cc` at Blender v5.2.2:
https://raw.githubusercontent.com/blender/blender/v5.2.2/source/blender/draw/engines/eevee/eevee_lut.cc

Five arrays are extracted, in this order, as little-endian f32 values:
- `brdf_ggx[64][64][3]`, indexed `[y][x][channel]`: the split-sum scale and bias (channels 0 and 1; the metal bias
  channel 2 is dropped).
- `btdf_ggx[16][64][64][1]`, indexed `[layer][y][x]`: the transmission factor for an IOR above one.
- `bsdf_ggx[16][64][64][3]`, indexed `[layer][y][x][channel]`: scale, bias and transmission factor for an IOR below one.
- `ltc_mat_ggx[64][64][4]`, indexed `[y][x][channel]`: the inverse LTC matrix of the GGX lobe, entries
  `(m11, m13, m31, m33)` as `eevee::lut::ltc::unpack` reads them.
- `ltc_disk_integral[64][64][1]`, indexed `[y][x]`: the clipped sphere diffuse integral of a disk light.

`ltc_mag_ggx` is not extracted: Blender 5.2.2 declares it in `eevee_lut.hh` but no shader or pipeline reads it, since
the LTC matrices are normalised by construction (`eevee_pipeline.hh` `UtilityTexture` copies only `ltc_mat_ggx`).

Literals are copied as C float literals (the `f` suffix removed) and rounded to f32, which is how the C compiler
stores them. Each array's element count is checked against its declared size, and the output size is checked too.
"""

import re
import sys
from array import array
from pathlib import Path

SOURCE_PATH = "source/blender/draw/engines/eevee/eevee_lut.cc"
SCRIPT_PATH = "scripts/extract_eevee_transmission_tables.py"
BLENDER_VERSION = "v5.2.2"

TABLE_SIZE = 64
LAYER_COUNT = 16
LITERAL_RE = re.compile(r"-?\d+\.\d+")

# (C array name, declared dimensions, channels kept)
ARRAYS = [
    ("brdf_ggx", [TABLE_SIZE, TABLE_SIZE, 3], 2),
    ("btdf_ggx", [LAYER_COUNT, TABLE_SIZE, TABLE_SIZE, 1], 1),
    ("bsdf_ggx", [LAYER_COUNT, TABLE_SIZE, TABLE_SIZE, 3], 3),
    ("ltc_mat_ggx", [TABLE_SIZE, TABLE_SIZE, 4], 4),
    ("ltc_disk_integral", [TABLE_SIZE, TABLE_SIZE, 1], 1),
]


def fail(message):
    sys.exit(f"extract_eevee_transmission_tables: {message}")


def strip_c_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def extract_values(text, name, dims, kept):
    """Return the kept channels of `const float name[dims...] = {...}`, in storage order."""
    declared = "".join(f"[{size}]" for size in dims)
    pattern = re.compile(
        r"const\s+float\s+" + re.escape(name) + re.escape(declared) + r"\s*=\s*\{(.*?)\}\s*;",
        re.S,
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        fail(f"expected exactly one definition of {name}{declared}, found {len(matches)}")
    body = strip_c_comments(matches[0].group(1))
    literals = []
    for raw in body.split(","):
        token = raw.strip().strip("{}").strip()
        if not token:
            continue
        if token[-1] not in "fF":
            fail(f"{name}: literal {token!r} has no C float suffix")
        literal = token[:-1]
        if not LITERAL_RE.fullmatch(literal):
            fail(f"{name}: unsupported literal {token!r}")
        literals.append(float(literal))
    total = 1
    for size in dims:
        total *= size
    if len(literals) != total:
        fail(f"{name}: found {len(literals)} elements, expected {total}")
    channels = dims[-1]
    if kept == channels:
        return literals
    return [value for index, value in enumerate(literals) if index % channels < kept]


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    text = Path(argv[1]).read_text(encoding="utf-8")
    values = array("f")
    for name, dims, kept in ARRAYS:
        values.extend(extract_values(text, name, dims, kept))
    expected = (
        (TABLE_SIZE * TABLE_SIZE * 2)
        + (LAYER_COUNT * TABLE_SIZE * TABLE_SIZE)
        + (LAYER_COUNT * TABLE_SIZE * TABLE_SIZE * 3)
        + (TABLE_SIZE * TABLE_SIZE * 4)
        + (TABLE_SIZE * TABLE_SIZE)
    )
    if len(values) != expected:
        fail(f"wrote {len(values)} values, expected {expected}")
    if sys.byteorder == "big":
        values.byteswap()
    Path(argv[2]).write_bytes(values.tobytes())
    print(f"{SCRIPT_PATH} from {SOURCE_PATH} at {BLENDER_VERSION}: {len(values)} f32 values")


if __name__ == "__main__":
    main(sys.argv)
