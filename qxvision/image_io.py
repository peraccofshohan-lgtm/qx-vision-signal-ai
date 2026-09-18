"""Small dependency-free image reader used by tests and training utilities.

Pillow is used when installed. The fallback supports PPM and non-interlaced
8-bit PNG, which keeps the core test suite runnable in a clean Python image.
JPEG/WEBP are rejected with a truthful error when no decoder is available.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import List, Tuple

from .domain import ImageFrame


def _png(path: Path) -> ImageFrame:
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos = 8
    width = height = bit_depth = color_type = None
    interlace = 0
    chunks: List[bytes] = []
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        kind = raw[pos + 4 : pos + 8]
        data = raw[pos + 8 : pos + 8 + length]
        pos += length + 12
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
            if bit_depth != 8 or compression != 0 or filtering != 0 or interlace != 0:
                raise ValueError("only non-interlaced 8-bit PNG is supported without Pillow")
        elif kind == b"IDAT":
            chunks.append(data)
        elif kind == b"IEND":
            break
    if width is None or height is None:
        raise ValueError("PNG has no header")
    channels = {2: 3, 6: 4, 0: 1, 4: 2}.get(color_type)
    if channels is None:
        raise ValueError("unsupported PNG color type")
    scan = zlib.decompress(b"".join(chunks))
    stride = width * channels
    rows: List[Tuple[Tuple[int, int, int], ...]] = []
    previous = bytearray(stride)
    at = 0
    for _ in range(height):
        filter_type = scan[at]
        at += 1
        current = bytearray(scan[at : at + stride])
        at += stride
        for i in range(stride):
            left = current[i - channels] if i >= channels else 0
            up = previous[i]
            up_left = previous[i - channels] if i >= channels else 0
            if filter_type == 1:
                current[i] = (current[i] + left) & 255
            elif filter_type == 2:
                current[i] = (current[i] + up) & 255
            elif filter_type == 3:
                current[i] = (current[i] + ((left + up) // 2)) & 255
            elif filter_type == 4:
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                pr = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                current[i] = (current[i] + pr) & 255
            elif filter_type != 0:
                raise ValueError("unsupported PNG filter")
        row = []
        for x in range(width):
            p = x * channels
            if channels >= 3:
                row.append((current[p], current[p + 1], current[p + 2]))
            else:
                row.append((current[p], current[p], current[p]))
        rows.append(tuple(row))
        previous = current
    return ImageFrame(width, height, tuple(rows), "png")


def _ppm(path: Path) -> ImageFrame:
    data = path.read_bytes()
    if not data.startswith((b"P6", b"P3")):
        raise ValueError("not a PPM")
    tokens: List[bytes] = []
    i = 0
    while len(tokens) < 4 and i < len(data):
        while i < len(data) and data[i] in b" \t\r\n":
            i += 1
        if i < len(data) and data[i] == 35:
            while i < len(data) and data[i] not in b"\r\n":
                i += 1
            continue
        j = i
        while j < len(data) and data[j] not in b" \t\r\n":
            j += 1
        tokens.append(data[i:j])
        i = j
    width, height, max_value = map(int, tokens[1:4])
    if max_value != 255:
        raise ValueError("PPM max value must be 255")
    while i < len(data) and data[i] in b" \t\r\n":
        i += 1
    values = list(data[i:]) if tokens[0] == b"P6" else [int(x) for x in data[i:].split()]
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            p = (y * width + x) * 3
            row.append((values[p], values[p + 1], values[p + 2]))
        rows.append(tuple(row))
    return ImageFrame(width, height, tuple(rows), "ppm")


def load_image(path: str | Path) -> ImageFrame:
    p = Path(path)
    try:
        from PIL import Image  # type: ignore
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            rows = tuple(tuple(rgb.getpixel((x, y)) for x in range(rgb.width)) for y in range(rgb.height))
            return ImageFrame(rgb.width, rgb.height, rows, p.suffix.lower().lstrip("."))
    except ImportError:
        pass
    if p.suffix.lower() == ".png":
        return _png(p)
    if p.suffix.lower() in (".ppm", ".pnm"):
        return _ppm(p)
    raise ValueError("image decoder unavailable for %s; install Pillow for JPEG/WEBP" % p.suffix)


def save_ppm(frame: ImageFrame, path: str | Path) -> None:
    p = Path(path)
    with p.open("wb") as out:
        out.write(f"P6\n{frame.width} {frame.height}\n255\n".encode())
        for row in frame.pixels:
            for r, g, b in row:
                out.write(bytes((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))))
