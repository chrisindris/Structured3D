#!/usr/bin/env python3

"""Find unique image dimensions for selected panorama filestems.

By default, this scans:
  /scratch/indrisch/Structured3D/data_sandbox/Structured3D_panorama_00

For each filestem in:
  albedo, depth, normal, rgb_coldlight, rgb_warmlight, semantic

the script finds every FILESTEM.png in the directory hierarchy and reports
the unique image sizes found as (width, height).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image


DEFAULT_ROOT = Path("/scratch/indrisch/Structured3D/data_sandbox/Structured3D_panorama_00")
DEFAULT_FILESTEMS = (
    "albedo",
    "depth",
    "normal",
    "rgb_coldlight",
    "rgb_warmlight",
    "semantic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect unique dimensions for FILESTEM.png files in a directory tree."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--filestems",
        nargs="+",
        default=list(DEFAULT_FILESTEMS),
        help="Filestems to scan, matching files named FILESTEM.png.",
    )
    return parser.parse_args()


def collect_unique_sizes(root: Path, filestems: list[str]) -> tuple[dict[str, set[tuple[int, int]]], dict[str, int], list[Path]]:
    unique_sizes: dict[str, set[tuple[int, int]]] = defaultdict(set)
    file_counts: dict[str, int] = defaultdict(int)
    unreadable: list[Path] = []

    for stem in filestems:
        pattern = f"**/{stem}.png"
        for image_path in root.glob(pattern):
            file_counts[stem] += 1
            try:
                with Image.open(image_path) as image:
                    unique_sizes[stem].add(image.size)
            except Exception:
                unreadable.append(image_path)

    return unique_sizes, file_counts, unreadable


def print_report(
    root: Path,
    filestems: list[str],
    unique_sizes: dict[str, set[tuple[int, int]]],
    file_counts: dict[str, int],
    unreadable: list[Path],
) -> None:
    print(f"Scan root: {root}")
    print("\nUnique sizes by filestem (width, height):")

    for stem in filestems:
        sizes_sorted = sorted(unique_sizes.get(stem, set()))
        print(f"\n{stem}:")
        print(f"  files found: {file_counts.get(stem, 0)}")
        print(f"  unique sizes: {len(sizes_sorted)}")
        if sizes_sorted:
            for width, height in sizes_sorted:
                print(f"    ({width}, {height})")

    if unreadable:
        print(f"\nUnreadable files: {len(unreadable)}")
        for bad_path in unreadable[:20]:
            print(f"  {bad_path}")
        if len(unreadable) > 20:
            print(f"  ... and {len(unreadable) - 20} more")


def main() -> None:
    args = parse_args()

    if not args.root.exists():
        raise SystemExit(f"Root path does not exist: {args.root}")
    if not args.root.is_dir():
        raise SystemExit(f"Root path is not a directory: {args.root}")

    unique_sizes, file_counts, unreadable = collect_unique_sizes(args.root, args.filestems)
    print_report(args.root, args.filestems, unique_sizes, file_counts, unreadable)


if __name__ == "__main__":
    main()
