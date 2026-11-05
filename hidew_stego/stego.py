"""
Hide a secret in a PNG, and get it back.

Image layout::

    ┌────────────────────────── image pixels ──────────────────────────┐
    │ reserved: first 1,024 px │ payload: every other pixel, in a      │
    │ header, scanline order   │ password-derived shuffled order       │
    └──────────────────────────────────────────────────────────────────┘

    payload bits = length (u32) ‖ secret, as-is ‖ CRC-32 (u32)

The secret is not encrypted. The checksum lets the reader tell a correct
extraction from a wrong password or a modified image.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from . import lsb
from .container import HEADER_SIZE, RESERVED_PIXELS, Header, NotAHideWImage
from .keys import DEFAULT_MEMORY_KIB, DEFAULT_PARALLELISM, DEFAULT_TIME_COST, new_header, order_seed
from .permutation import PixelOrder

ImageLike = Union[str, Path, Image.Image]

_LENGTH_BITS = 32
_OVERHEAD = 8  # length prefix + CRC-32


class CapacityError(ValueError):
    """The secret does not fit in the cover image."""


class ExtractionError(ValueError):
    """Wrong password, or the image was modified after the secret was hidden."""


@dataclass
class _Canvas:
    """An image split into a flat, writable RGB array plus untouched alpha."""

    image: Image.Image
    rgb: np.ndarray
    width: int
    height: int

    @classmethod
    def load(cls, source: ImageLike) -> "_Canvas":
        image = source if isinstance(source, Image.Image) else Image.open(source)
        image = image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB")
        pixels = np.array(image)
        rgb = np.ascontiguousarray(pixels[..., :3]).reshape(-1)
        return cls(image, rgb, image.width, image.height)

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    def render(self) -> Image.Image:
        pixels = np.array(self.image)
        pixels[..., :3] = self.rgb.reshape(self.height, self.width, 3)
        return Image.fromarray(pixels, mode=self.image.mode)


def capacity(source: ImageLike) -> int:
    """Largest secret, in bytes, that fits in this image."""
    canvas = _Canvas.load(source)
    payload_bits = max(0, canvas.total_pixels - RESERVED_PIXELS) * lsb.CHANNELS
    return max(0, payload_bits // 8 - _OVERHEAD)


def read_header(source: ImageLike) -> Header:
    """Read the public header (no password needed)."""
    canvas = _Canvas.load(source)
    reserved = range(min(RESERVED_PIXELS, canvas.total_pixels))
    bits = lsb.read(canvas.rgb, reserved, min(HEADER_SIZE * 8, len(reserved) * lsb.CHANNELS))
    return Header.unpack(lsb.from_bits(bits))


def hide(
    cover: ImageLike,
    secret: bytes,
    password: str,
    *,
    memory_kib: int = DEFAULT_MEMORY_KIB,
    time_cost: int = DEFAULT_TIME_COST,
    parallelism: int = DEFAULT_PARALLELISM,
) -> Image.Image:
    """Hide ``secret`` in ``cover``, at positions only ``password`` can find.

    Returns the stego image. Save it as PNG — any lossy format destroys it.
    """
    canvas = _Canvas.load(cover)
    if canvas.total_pixels <= RESERVED_PIXELS:
        raise CapacityError("image is too small")
    available = capacity(canvas.image)
    if len(secret) > available:
        raise CapacityError(f"secret is {len(secret)} bytes; this image holds at most {available}")

    header = new_header(memory_kib, time_cost, parallelism)

    # 1) Header into the reserved region, in scanline order.
    header_bits = lsb.to_bits(header.pack())
    lsb.write(canvas.rgb, range(lsb.pixels_for(len(header_bits))), header_bits)

    # 2) Length ‖ secret ‖ checksum into password-derived positions.
    payload = struct.pack(">I", len(secret)) + secret + struct.pack(">I", zlib.crc32(secret))
    payload_bits = lsb.to_bits(payload)
    order = PixelOrder(canvas.total_pixels, RESERVED_PIXELS, order_seed(password, header))
    lsb.write(canvas.rgb, order.take(lsb.pixels_for(len(payload_bits))), payload_bits)

    return canvas.render()


def reveal(stego: ImageLike, password: str) -> bytes:
    """Recover the secret hidden by :func:`hide`.

    Raises :class:`NotAHideWImage` if there is no header, and
    :class:`ExtractionError` for a wrong password or a modified image.
    """
    canvas = _Canvas.load(stego)
    header = read_header(canvas.image)
    order = PixelOrder(canvas.total_pixels, RESERVED_PIXELS, order_seed(password, header))

    # The length comes first. A wrong password reads the wrong pixels, so an
    # impossible length is reported straight away.
    length_bits = lsb.read(canvas.rgb, order.take(lsb.pixels_for(_LENGTH_BITS)), _LENGTH_BITS)
    (length,) = struct.unpack(">I", lsb.from_bits(length_bits))
    if length > capacity(canvas.image):
        raise ExtractionError("wrong password, or the image has been modified")

    total_bits = (length + _OVERHEAD) * 8
    data = lsb.from_bits(lsb.read(canvas.rgb, order.take(lsb.pixels_for(total_bits)), total_bits))
    secret = data[4 : 4 + length]
    (checksum,) = struct.unpack(">I", data[4 + length :])
    if zlib.crc32(secret) != checksum:
        raise ExtractionError("wrong password, or the image has been modified")
    return secret


__all__ = [
    "CapacityError",
    "ExtractionError",
    "NotAHideWImage",
    "capacity",
    "hide",
    "read_header",
    "reveal",
]
