"""Write the 4x4 sRGB texture fixtures used by the Blender sRGB sampling oracle.

Run from this directory: python3 make_srgb_fixture.py

The texture is 4x4 sRGB with R = 85x, G = 85y, B = 128 (8-bit values, all 255 alpha). Each glTF uses one
triangle and the texture as its base colour. The sampler varies by file: linear or closest filtering, with clamp
(CLAMP_TO_EDGE) or repeat wrapping. The glTF importers read the base colour as sRGB, so Blender and Potter decode the
same encoded bytes.
"""

import json
import struct
import zlib


def chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def texture_png(width=4, height=4):
    raw = b""
    for y in range(height):
        raw += b"\x00"
        for x in range(width):
            raw += bytes([x * 85, y * 85, 128, 255])
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def triangle_bytes():
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    texcoords = struct.pack("<6f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    return positions, texcoords


def build(name, mag, minf, wrap):
    png = texture_png()
    positions, texcoords = triangle_bytes()
    blob = bytearray()

    def add(data):
        offset = len(blob)
        blob.extend(data)
        while len(blob) % 4:
            blob.append(0)
        return offset

    p_off = add(positions)
    t_off = add(texcoords)
    i_off = add(struct.pack("<3H", 0, 1, 2))
    img_off = add(png)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"name": "srgb", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "baseColorTexture": {"index": 0, "texCoord": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": mag, "minFilter": minf, "wrapS": wrap, "wrapT": wrap}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": p_off, "byteLength": 36},
            {"buffer": 0, "byteOffset": t_off, "byteLength": 24},
            {"buffer": 0, "byteOffset": i_off, "byteLength": 6},
            {"buffer": 0, "byteOffset": img_off, "byteLength": len(png)},
        ],
        "buffers": [{"uri": f"{name}.bin", "byteLength": len(blob)}],
    }
    with open(f"{name}.bin", "wb") as handle:
        handle.write(bytes(blob))
    with open(f"{name}.gltf", "w") as handle:
        json.dump(gltf, handle, indent=1)


# Linear filtering is 9729/9987 and closest is 9728/9984, as Potter's exporter writes them. CLAMP_TO_EDGE is 33071 and
# REPEAT is 10497.
build("srgb_linear_clamp", 9729, 9987, 33071)
build("srgb_closest_clamp", 9728, 9984, 33071)
build("srgb_linear_repeat", 9729, 9987, 10497)
print("wrote srgb_linear_clamp, srgb_closest_clamp, srgb_linear_repeat")
