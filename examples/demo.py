"""
End-to-end walkthrough of the HideW technique.

    python examples/demo.py                     # uses examples/sunset.png
    python examples/demo.py --cover my.png --out results/

1. Hide a secret message in a photo and get it back.
2. Show that a wrong password and a JPEG re-save both fail closed.
3. Measure the change (PSNR, max difference) and render LSB bit planes.
4. Steganalysis: the chi-square attack catches naive sequential LSB
   embedding, but not password-scattered embedding.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hidew_stego import ExtractionError, NotAHideWImage, capacity, hide, read_header, reveal  # noqa: E402
from hidew_stego import analysis, lsb  # noqa: E402
from hidew_stego.container import RESERVED_PIXELS  # noqa: E402

HERE = Path(__file__).resolve().parent
SECRET = b"Meet at the old lighthouse at 6am. Bring the blue notebook."
PASSWORD = "Sunset#Harbour42"


def step(title: str) -> None:
    print(f"\n\033[1;32m▍ {title}\033[0m")


def row(label: str, value: object) -> None:
    print(f"  {label:<26}{value}")


def naive_sequential(cover: Image.Image, data: bytes) -> Image.Image:
    """Textbook LSB: write the bits into consecutive pixels, in scanline order."""
    pixels = np.array(cover.convert("RGB"))
    flat = pixels.reshape(-1)
    bits = lsb.to_bits(data)
    start = RESERVED_PIXELS * lsb.CHANNELS
    flat[start : start + len(bits)] = (flat[start : start + len(bits)] & 0xFE) | bits
    return Image.fromarray(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cover", default=HERE / "sunset.png", type=Path)
    parser.add_argument("--out", default=HERE / "output", type=Path)
    parser.add_argument("--memory-mib", type=int, default=64, help="Argon2 memory cost")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cover = Image.open(args.cover).convert("RGB")
    options = {"memory_kib": args.memory_mib * 1024, "time_cost": 3}

    # 1 ------------------------------------------------------------------
    step("1. Hide a message in a photo")
    row("Cover", f"{args.cover.name} · {cover.width}×{cover.height}")
    row("Capacity", f"{capacity(cover):,} bytes")
    row("Secret", f"{len(SECRET)} bytes of text")
    started = time.perf_counter()
    stego = hide(cover, SECRET, PASSWORD, **options)
    row("Hidden in", f"{time.perf_counter() - started:.2f}s (Argon2id {args.memory_mib} MiB, t=3)")
    stego_path = args.out / "stego.png"
    stego.save(stego_path, format="PNG", optimize=True)
    header = read_header(stego_path)
    row("Saved", stego_path)
    row("Public header", f"Argon2id {header.memory_kib // 1024} MiB · salt {header.salt.hex()[:16]}…")

    started = time.perf_counter()
    recovered = reveal(stego_path, PASSWORD)
    assert recovered == SECRET
    row("Revealed", f"{recovered[:32].decode()}… ({time.perf_counter() - started:.2f}s) ✓")

    # 2 ------------------------------------------------------------------
    step("2. Only the right password finds it")
    try:
        reveal(stego_path, "Sunset#Harbour43")
    except ExtractionError as error:
        row("Wrong password", f"rejected — {error}")
    jpeg = io.BytesIO()
    Image.open(stego_path).save(jpeg, format="JPEG", quality=95)
    jpeg.seek(0)
    try:
        reveal(Image.open(jpeg), PASSWORD)
    except (ExtractionError, NotAHideWImage) as error:
        row("Re-saved as JPEG (q=95)", f"secret destroyed — {error}")

    # 3 ------------------------------------------------------------------
    step("3. Invisible to the eye")
    result = analysis.compare(cover, stego)
    row("PSNR", f"{result.psnr_db:.1f} dB  (>50 dB is visually identical)")
    row("Channel values changed", f"{result.changed_values:,} of {result.total_values:,} ({result.changed_ratio:.4%})")
    row("Largest change", f"±{result.max_difference} out of 255")
    cover.save(args.out / "cover.png")
    analysis.bit_plane(cover).save(args.out / "lsb-cover.png")
    analysis.bit_plane(stego).save(args.out / "lsb-stego.png")

    # 4 ------------------------------------------------------------------
    step("4. Steganalysis: sequential vs. HideW")
    load = os.urandom(capacity(cover) // 4)  # a quarter of capacity, so the effect is measurable
    sequential = naive_sequential(cover, load)
    scattered = hide(cover, load, PASSWORD, **options)
    region = slice(RESERVED_PIXELS, RESERVED_PIXELS + lsb.pixels_for(len(load) * 8))
    row("Payload", f"{len(load):,} random bytes")
    row("Chi-square, cover", f"{analysis.chi_square_lsb(cover, region):.3f}")
    row("Chi-square, sequential", f"{analysis.chi_square_lsb(sequential, region):.3f}  ← detected")
    row("Chi-square, HideW", f"{analysis.chi_square_lsb(scattered, region):.3f}")
    print("  (probability of embedding in the first pixels; ~1.0 means detected)")
    analysis.difference_map(cover, sequential).save(args.out / "difference-sequential.png")
    analysis.difference_map(cover, scattered).save(args.out / "difference-hidew.png")

    step(f"Done — images written to {args.out}")
    for name in sorted(p.name for p in args.out.glob("*.png")):
        print(f"  {name}")


if __name__ == "__main__":
    main()
