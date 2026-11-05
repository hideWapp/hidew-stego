"""
Key-derived pixel order.

Writing a payload into pixels 0, 1, 2… leaves a dense band of modified LSBs
that simple steganalysis spots at once. Instead, the payload is scattered
across the whole image in an order only the key holder can reproduce:

* An AES-256-CTR keystream, keyed from the password-derived seed, acts as a
  cryptographically strong pseudo-random generator.
* Its output drives a Fisher–Yates shuffle of the payload pixels.

The shuffle is run *lazily and sparsely*: only the swaps actually touched are
stored, so taking the first ``k`` positions of a 12-megapixel image costs
O(k) time and memory rather than shuffling every pixel up front. Because the
sequence is deterministic, the reader regenerates exactly the same order —
first to find the length prefix, then the rest.
"""

from __future__ import annotations

import hashlib
import struct

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_BATCH = 4096


class PixelOrder:
    """Positions in ``[start, total)`` in key-derived shuffled order."""

    def __init__(self, total: int, start: int, seed: bytes):
        if total <= start:
            raise ValueError("image has no pixels outside the reserved region")
        self._size = total - start
        self._start = start
        key = hashlib.sha256(b"key:" + seed).digest()
        iv = hashlib.sha256(b"iv:" + seed).digest()[:16]
        self._stream = Cipher(algorithms.AES(key), modes.CTR(iv)).encryptor()
        self._swapped: dict[int, int] = {}
        self._order: list[int] = []

    def __len__(self) -> int:
        return self._size

    def _random_u32(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f">{count}I", self._stream.update(bytes(4 * count)))

    def take(self, count: int) -> list[int]:
        """The first ``count`` positions of the order (extends on demand)."""
        if count > self._size:
            raise ValueError(f"requested {count} positions, only {self._size} available")
        while len(self._order) < count:
            batch = min(_BATCH, count - len(self._order))
            for r in self._random_u32(batch):
                # Fisher–Yates from the end: pick j in [0, i], swap a[i] and a[j];
                # a[i] is then final and is emitted next.
                i = self._size - 1 - len(self._order)
                j = r % (i + 1)
                a_i = self._swapped.get(i, i)
                a_j = self._swapped.get(j, j)
                self._swapped[j] = a_i
                self._swapped.pop(i, None)
                self._order.append(a_j + self._start)
        return self._order[:count]
