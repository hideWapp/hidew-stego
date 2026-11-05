"""
Least-significant-bit embedding.

Each pixel offers three carrier bits — the lowest bit of its red, green and
blue channels. Changing that bit alters a channel value by at most 1 out of
255, far below what the eye can see. Alpha is never touched, so transparency
is preserved exactly.

Bit ``n`` of the message lands in pixel ``positions[n // 3]``, channel
``n % 3``. Bits are MSB-first within each byte.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

CHANNELS = 3  # R, G, B


def to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def from_bits(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def pixels_for(bit_count: int) -> int:
    """Pixels needed to carry ``bit_count`` bits."""
    return -(-bit_count // CHANNELS)


def _channel_indices(positions: Sequence[int], bit_count: int) -> np.ndarray:
    base = np.asarray(positions, dtype=np.int64) * CHANNELS
    return (base[:, None] + np.arange(CHANNELS)).reshape(-1)[:bit_count]


def write(rgb: np.ndarray, positions: Sequence[int], bits: np.ndarray) -> None:
    """Write ``bits`` into the LSBs of a flat RGB array, in place."""
    idx = _channel_indices(positions, len(bits))
    rgb[idx] = (rgb[idx] & 0xFE) | bits


def read(rgb: np.ndarray, positions: Sequence[int], bit_count: int) -> np.ndarray:
    """Read ``bit_count`` LSBs from a flat RGB array."""
    return rgb[_channel_indices(positions, bit_count)] & 1
