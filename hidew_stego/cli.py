"""Command-line interface: ``hidew-stego <command> …`` or ``python -m hidew_stego``."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from . import analysis
from .container import NotAHideWImage, RESERVED_PIXELS
from .keys import DEFAULT_MEMORY_KIB, DEFAULT_TIME_COST
from .stego import CapacityError, ExtractionError, capacity, hide, read_header, reveal


def _password(args: argparse.Namespace, confirm: bool = False) -> str:
    if args.password is not None:
        return args.password
    password = getpass.getpass("Password: ")
    if confirm and password != getpass.getpass("Confirm password: "):
        sys.exit("Passwords do not match.")
    return password


def _cmd_hide(args: argparse.Namespace) -> None:
    if args.file:
        secret = Path(args.file).read_bytes()
    elif args.message is not None:
        secret = args.message.encode("utf-8")
    else:
        secret = sys.stdin.buffer.read()

    stego = hide(
        args.cover,
        secret,
        _password(args, confirm=True),
        memory_kib=args.memory_mib * 1024,
        time_cost=args.time_cost,
    )
    out = Path(args.output)
    if out.suffix.lower() != ".png":
        sys.exit("Output must be a .png file — lossy formats destroy the hidden data.")
    stego.save(out, format="PNG", optimize=True)
    print(f"Hid {len(secret)} bytes in {out} ({len(secret) / capacity(stego):.2%} of capacity).")


def _cmd_reveal(args: argparse.Namespace) -> None:
    secret = reveal(args.image, _password(args))
    if args.output:
        Path(args.output).write_bytes(secret)
        print(f"Wrote {len(secret)} bytes to {args.output}.")
    else:
        sys.stdout.write(secret.decode("utf-8", errors="replace") + "\n")


def _cmd_inspect(args: argparse.Namespace) -> None:
    print(f"Capacity:     {capacity(args.image):,} bytes")
    try:
        header = read_header(args.image)
    except NotAHideWImage as error:
        print(f"Header:       none ({error})")
        return
    print("Header:       HideW container v%d" % header.version)
    print(f"  KDF:        Argon2id, {header.memory_kib // 1024} MiB, t={header.time_cost}, p={header.parallelism}")
    print(f"  Salt:       {header.salt.hex()}")


def _cmd_analyze(args: argparse.Namespace) -> None:
    result = analysis.compare(args.cover, args.stego)
    print(f"PSNR:            {result.psnr_db:.2f} dB")
    print(f"Changed values:  {result.changed_values:,} of {result.total_values:,} ({result.changed_ratio:.3%})")
    print(f"Max difference:  {result.max_difference} (out of 255)")
    payload = slice(RESERVED_PIXELS, None)
    print(f"Chi-square LSB:  cover {analysis.chi_square_lsb(args.cover, payload):.3f} · "
          f"stego {analysis.chi_square_lsb(args.stego, payload):.3f}  (1.0 = looks embedded)")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        analysis.bit_plane(args.cover).save(out / "lsb-cover.png")
        analysis.bit_plane(args.stego).save(out / "lsb-stego.png")
        analysis.difference_map(args.cover, args.stego).save(out / "difference.png")
        print(f"Wrote bit planes and difference map to {out}/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hidew-stego",
        description="Hide a secret inside a PNG image.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("hide", help="hide a secret in an image")
    p.add_argument("cover", help="cover image (PNG recommended)")
    p.add_argument("output", help="where to write the stego PNG")
    source = p.add_mutually_exclusive_group()
    source.add_argument("-m", "--message", help="text to hide")
    source.add_argument("-f", "--file", help="file to hide (default: stdin)")
    p.add_argument("-p", "--password", help="password (prompted if omitted)")
    p.add_argument("--memory-mib", type=int, default=DEFAULT_MEMORY_KIB // 1024, help="Argon2 memory cost")
    p.add_argument("--time-cost", type=int, default=DEFAULT_TIME_COST, help="Argon2 iterations")
    p.set_defaults(run=_cmd_hide)

    p = sub.add_parser("reveal", help="recover a hidden secret")
    p.add_argument("image")
    p.add_argument("-p", "--password", help="password (prompted if omitted)")
    p.add_argument("-o", "--output", help="write the secret to a file instead of stdout")
    p.set_defaults(run=_cmd_reveal)

    p = sub.add_parser("inspect", help="show capacity and the public header")
    p.add_argument("image")
    p.set_defaults(run=_cmd_inspect)

    p = sub.add_parser("analyze", help="measure the change between a cover and its stego image")
    p.add_argument("cover")
    p.add_argument("stego")
    p.add_argument("--out-dir", help="also write LSB bit planes and a difference map here")
    p.set_defaults(run=_cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.run(args)
    except (CapacityError, NotAHideWImage, ExtractionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
