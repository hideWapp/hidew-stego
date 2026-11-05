"""
Measure what embedding does to an image — and what it does not.

* :func:`compare` — PSNR and how many channel values changed (never by more
  than 1).
* :func:`bit_plane` — renders one bit of every channel as black/white, the
  classic way to *see* LSB data.
* :func:`difference_map` — amplifies every changed value so the scattered,
  key-derived placement is visible.
* :func:`chi_square_lsb` — the Westfeld–Pfitzmann "pairs of values" test, the
  textbook detector for sequential LSB embedding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .stego import ImageLike


def _rgb(source: ImageLike) -> np.ndarray:
    image = source if isinstance(source, Image.Image) else Image.open(source)
    return np.asarray(image.convert("RGB"), dtype=np.int16)


@dataclass(frozen=True)
class Comparison:
    psnr_db: float
    changed_values: int
    total_values: int
    max_difference: int

    @property
    def changed_ratio(self) -> float:
        return self.changed_values / self.total_values


def compare(cover: ImageLike, stego: ImageLike) -> Comparison:
    a, b = _rgb(cover), _rgb(stego)
    if a.shape != b.shape:
        raise ValueError("images differ in size")
    diff = np.abs(a - b)
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
    return Comparison(psnr, int(np.count_nonzero(diff)), diff.size, int(diff.max()))


def bit_plane(source: ImageLike, bit: int = 0) -> Image.Image:
    """One bit of each channel, scaled to 0/255 — bit 0 is the LSB."""
    plane = ((_rgb(source) >> bit) & 1).astype(np.uint8) * 255
    return Image.fromarray(plane, mode="RGB")


def difference_map(cover: ImageLike, stego: ImageLike) -> Image.Image:
    """Changed channel values in green on black."""
    changed = np.any(_rgb(cover) != _rgb(stego), axis=-1)
    out = np.zeros((*changed.shape, 3), dtype=np.uint8)
    out[changed] = (1, 177, 129)
    return Image.fromarray(out, mode="RGB")


def _chi_square_sf(x: float, k: int) -> float:
    """Survival function of the chi-square distribution (Wilson–Hilferty)."""
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def chi_square_lsb(source: ImageLike, region: slice = slice(None)) -> float:
    """Probability that the region carries sequential LSB embedding.

    Embedding random bits equalises the counts of each value pair (2k, 2k+1).
    The test measures how equal they are: values near 1 mean "looks
    embedded", near 0 mean "looks natural". ``region`` selects pixels in
    scanline order.
    """
    values = _rgb(source).reshape(-1, 3)[region].reshape(-1)
    hist = np.bincount(values, minlength=256).astype(np.float64)
    even, odd = hist[0::2], hist[1::2]
    expected = (even + odd) / 2
    used = expected > 4  # the test needs reasonably populated bins
    stat = float(np.sum((even[used] - expected[used]) ** 2 / expected[used]))
    return _chi_square_sf(stat, int(np.count_nonzero(used)) - 1)
