#!/usr/bin/env python3
"""
Sample tissue-containing PSR tiles for CDP clustering training.

This script creates a reusable tile pool from whole-slide images. The original
paper tile pool was a stored sample of tissue tiles; if regenerating a comparable
pool, use a fixed --seed and keep the emitted manifest.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdp_processor import get_otsu_threshold, get_tile, is_tissue  # noqa: E402
from wsi_reader import TiffReader  # noqa: E402


def slide_series(slide_path: Path) -> int:
    return 1 if slide_path.suffix.lower() == ".scn" else 0


def iter_slide_paths(patterns: list[str]) -> list[Path]:
    paths = set()
    for pattern in patterns:
        paths.update(Path(path) for path in glob(pattern))
    return sorted(paths)


def sample_slide_tiles(
    slide_path: Path,
    out_dir: Path,
    rng: random.Random,
    *,
    start_index: int,
    tile_size: int,
    sample_probability: float,
    tissue_fraction: float,
) -> list[dict[str, object]]:
    slide = TiffReader(slide_path, series=slide_series(slide_path))
    mask_threshold = get_otsu_threshold(slide)
    width, height = slide.level_dimensions[0]
    tiles_horizontal = width // tile_size
    tiles_vertical = height // tile_size

    rows = []
    tissue_tiles = 0
    sampled_tiles = 0

    for t_h in range(tiles_horizontal):
        for t_v in range(tiles_vertical):
            try:
                tile = get_tile(slide, 0, tile_size, t_h, t_v)
                tile = (tile * 255).astype(np.uint8)
            except Exception:
                continue

            if not is_tissue(tile, mask_threshold, tissue_fraction):
                continue

            tissue_tiles += 1
            if rng.random() > sample_probability:
                continue

            sampled_tiles += 1
            tile_id = f"tile_{start_index + len(rows) + 1:08d}"
            filename = f"{slide_path.stem}_{t_h}_{t_v}.png"
            out_path = out_dir / filename
            Image.fromarray(tile).save(out_path)
            rows.append(
                {
                    "tile_id": tile_id,
                    "tile_path": str(out_path),
                    "slide": slide_path.name,
                    "tile_x": t_h,
                    "tile_y": t_v,
                    "tile_size": tile_size,
                }
            )

    print(
        f"{slide_path.name}: sampled {sampled_tiles} of {tissue_tiles} "
        f"tissue tiles ({tiles_horizontal * tiles_vertical} total grid tiles)"
    )
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tile_id", "tile_path", "slide", "tile_x", "tile_y", "tile_size"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample tissue PSR tiles for CDP training.")
    parser.add_argument(
        "--input-glob",
        action="append",
        required=True,
        help="Slide glob. Repeat for multiple patterns, e.g. histology/data/*PSR*.ndpi.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--sample-probability", type=float, default=0.01)
    parser.add_argument("--tissue-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not 0 < args.sample_probability <= 1:
        raise SystemExit("--sample-probability must be in (0, 1]")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []

    slides = iter_slide_paths(args.input_glob)
    if not slides:
        raise SystemExit("No slides matched --input-glob")

    for slide_path in slides:
        rows.extend(
            sample_slide_tiles(
                slide_path,
                args.out_dir,
                rng,
                start_index=len(rows),
                tile_size=args.tile_size,
                sample_probability=args.sample_probability,
                tissue_fraction=args.tissue_fraction,
            )
        )

    manifest = args.manifest or args.out_dir / "tile_manifest.csv"
    write_manifest(manifest, rows)
    print(f"Wrote {len(rows)} sampled tiles")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main(parse_args())
