"""
The header that tells a reader how to open an image.

It is written into the *reserved region* — the least-significant bits of the
first ``RESERVED_PIXELS`` pixels, in plain scanline order — so it can be read
before any key exists. It carries no secrets: only the KDF parameters and the
salt needed to re-derive the pixel order from the password.

Layout (26 bytes, big-endian)::

    offset  size  field
    0       2     magic        b"HW"
    2       1     version      0x01
    3       1     kdf          0x01 = Argon2id
    4       4     memory_kib   Argon2 memory cost
    8       1     time_cost    Argon2 iterations
    9       1     parallelism  Argon2 lanes
    10      16    salt         random, per image
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"HW"
VERSION = 1
KDF_ARGON2ID = 1

#: Pixels at the start of the image kept for the header (3 bits each).
RESERVED_PIXELS = 1024

_LAYOUT = struct.Struct(">2sBBIBB16s")
HEADER_SIZE = _LAYOUT.size  # 26 bytes


class NotAHideWImage(ValueError):
    """The image carries no HideW header."""


@dataclass(frozen=True)
class Header:
    memory_kib: int
    time_cost: int
    parallelism: int
    salt: bytes
    version: int = VERSION
    kdf: int = KDF_ARGON2ID

    def pack(self) -> bytes:
        return _LAYOUT.pack(
            MAGIC, self.version, self.kdf, self.memory_kib, self.time_cost, self.parallelism, self.salt
        )

    @classmethod
    def unpack(cls, data: bytes) -> "Header":
        if len(data) < HEADER_SIZE:
            raise NotAHideWImage("image is too small to hold a header")
        magic, version, kdf, memory_kib, time_cost, parallelism, salt = _LAYOUT.unpack(data[:HEADER_SIZE])
        if magic != MAGIC:
            raise NotAHideWImage("no HideW header found in this image")
        if version != VERSION or kdf != KDF_ARGON2ID:
            raise NotAHideWImage(f"unsupported container (version {version}, kdf {kdf})")
        return cls(memory_kib, time_cost, parallelism, salt, version, kdf)
