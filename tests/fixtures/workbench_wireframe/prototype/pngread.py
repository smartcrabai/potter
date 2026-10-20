"""Minimal PNG reader (8-bit RGB/RGBA, non-interlaced) for oracle checks without third-party packages."""
import struct
import zlib


def read_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos = 8
    width = height = depth = ctype = interlace = None
    idat = b""
    while pos < len(data):
        length, = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat += chunk
        elif kind == b"IEND":
            break
    assert depth == 8 and interlace == 0 and ctype in (2, 6), (depth, ctype, interlace)
    channels = 3 if ctype == 2 else 4
    raw = zlib.decompress(idat)
    stride = width * channels
    rows = []
    prev = bytearray(stride)
    offset = 0
    for _ in range(height):
        ftype = raw[offset]
        line = bytearray(raw[offset + 1:offset + 1 + stride])
        offset += 1 + stride
        for i in range(stride):
            left = line[i - channels] if i >= channels else 0
            up = prev[i]
            upleft = prev[i - channels] if i >= channels else 0
            if ftype == 1:
                line[i] = (line[i] + left) & 255
            elif ftype == 2:
                line[i] = (line[i] + up) & 255
            elif ftype == 3:
                line[i] = (line[i] + (left + up) // 2) & 255
            elif ftype == 4:
                p = left + up - upleft
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - upleft)
                pred = left if pa <= pb and pa <= pc else (up if pb <= pc else upleft)
                line[i] = (line[i] + pred) & 255
        rows.append(bytes(line))
        prev = line
    pixels = [tuple(row[x * channels:x * channels + 3]) for row in rows for x in range(width)]
    return width, height, pixels
