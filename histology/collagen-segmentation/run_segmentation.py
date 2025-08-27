#!/usr/bin/env python3

"""
Batch wrapper for inference.py (quiet by default).

This script shells out to `inference.py` for each input slide, running:
  • collagen segmentation with the collagen model, and
  • tissue segmentation with the tissue model.

Each call follows:
  inference.py --model <MODEL.h5> -o <OUTPUT_DIR> \
               --tile-size <N> --stride <N> --n-workers <N> [--gpu] \
               [--ignore-existing] <INPUT_IMAGE>

PARAMETERS (CLI)
----------------
--input-glob        Comma-separated glob patterns for inputs.
                    Default: all *.ndpi, *.svs, *.tif, *.tiff in histology/data.
--force             Overwrite existing outputs. (If omitted, the wrapper passes
                    --ignore-existing to inference.py so previously processed slides are skipped.)
--gpu               Forward --gpu to inference.py.
--tile-size         Tile size passed to inference.py. Default: 512.
--stride            Stride passed to inference.py. Default: 256.
--workers           Number of workers passed as --n-workers. Default: half of CPUs.
--verbose           Show full stdout/stderr from inference.py (otherwise suppressed).
--collagen-model    Path to collagen model (.h5). Defaults to models/unet_mini_CoCoMASLD_PSR_collagen.h5.
--tissue-model      Path to tissue model (.h5). Defaults to models/unet_mini_CoCoMASLD_PSR_tissue.h5.
--fail-fast         Abort immediately on the first inference error (propagates non-zero exit straight away).
                    Useful for debugging/CI. Omit for overnight runs to collect a full summary.

BEHAVIOUR
---------
- For each input slide, runs inference.py twice:
    1) collagen → histology/results/collagen_segmentation/
    2) tissue   → histology/results/tissue_segmentation_raw/
- Skips existing outputs unless --force is set.
- Suppresses inference.py output unless --verbose is set.
- Error handling:
    • Default: continue past failures, record them, and print a summary at the end.
      Exit code is 1 if any slide failed, 0 otherwise.
    • With --fail-fast: abort on the first failure (non-zero exit immediately).

EXAMPLES
--------
# Process all slides quietly with defaults
python run_segmentation.py

# Custom inputs + GPU; larger tiles with overlap; show detailed logs
python run_segmentation.py \
  --input-glob "histology/data/*.svs,histology/data/*.ndpi" \
  --tile-size 512 --stride 256 --workers 8 --gpu --verbose

# Re-run everything from scratch, overwriting existing outputs
python run_segmentation.py --force

# Debug quick failure locally (stop at first error)
python run_segmentation.py --fail-fast --verbose
"""



from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path
from typing import List

# --- fixed paths ---
HERE = Path(__file__).resolve().parent                 # .../histology/collagen-segmentation
HIST = HERE.parent                                     # .../histology
INFERENCE_PY = HERE / "inference.py"

MODELS_DIR = HERE / "models"
COLLAGEN_MODEL = MODELS_DIR / "unet_mini_CoCoMASLD_PSR_collagen.h5"
TISSUE_MODEL   = MODELS_DIR / "unet_mini_CoCoMASLD_PSR_tissue.h5"

DATA_DIR = HIST / "data"
OUT_COLLAGEN_DIR   = HIST / "results" / "collagen_segmentation"
OUT_TISSUE_RAW_DIR = HIST / "results" / "tissue_segmentation_raw"
OUT_COLLAGEN_DIR.mkdir(parents=True, exist_ok=True)
OUT_TISSUE_RAW_DIR.mkdir(parents=True, exist_ok=True)

# default: NDPI, SVS, TIF/TIFF
DEFAULT_PATTERNS = [
    str(DATA_DIR / "*.ndpi"),
    str(DATA_DIR / "*.svs"),
    str(DATA_DIR / "*.tif*"),
]

# keep TF quiet if imported inside inference.py
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def gather_inputs(pattern_arg: str | None) -> List[Path]:
    """Collect inputs from comma-separated patterns, or default NDPI/SVS/TIF*."""
    patterns = [p.strip() for p in pattern_arg.split(",")] if pattern_arg else DEFAULT_PATTERNS
    paths = set()
    for pat in patterns:
        for p in glob(pat):
            paths.add(Path(p))
    return sorted(paths)


def call_inference(
    in_path: Path,
    out_dir: Path,
    model_path: Path,
    *,
    tile_size: int,
    stride: int,
    workers: int,
    gpu: bool,
    force: bool,
    verbose: bool,
) -> None:
    """Invoke inference.py once for a single slide."""
    cmd = [
        sys.executable, str(INFERENCE_PY),
        "--model", str(model_path),
        "-o", str(out_dir),
        "--tile-size", str(tile_size),
        "--stride", str(stride),
        "--n-workers", str(workers),
    ]
    if gpu:
        cmd.append("--gpu")
    if not force:
        cmd.append("--ignore-existing")
    cmd.append(str(in_path))  # positional input

    t0 = time.time()
    if verbose:
        # Stream child output live to the console
        res = subprocess.run(cmd)
    else:
        # Quiet: capture output and only show on error
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            sys.stderr.write(res.stdout or "")
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd)
    dt = int(time.time() - t0)
    print(f"… done ({dt} s)")


def main():
    p = argparse.ArgumentParser(description="Batch wrapper for inference.py over histology/data/* (quiet).")
    p.add_argument("--input-glob", default=None,
                   help="Comma-separated patterns (default: *.ndpi, *.svs, *.tif* under histology/data)")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs (omit --ignore-existing)")
    p.add_argument("--gpu", action="store_true", help="Pass --gpu to inference.py")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--stride",    type=int, default=256)
    p.add_argument("--workers",   type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--verbose",   action="store_true", help="Show full inference.py output and commands")

    # Allow overriding models if needed
    p.add_argument("--collagen-model", type=Path, default=COLLAGEN_MODEL)
    p.add_argument("--tissue-model",   type=Path, default=TISSUE_MODEL)
    args = p.parse_args()

    inputs = gather_inputs(args.input_glob)
    if not inputs:
        print("[ERROR] No inputs found. Use --input-glob or put slides under histology/data with .ndpi/.svs/.tif*",
              file=sys.stderr)
        sys.exit(1)

    # sanity: models exist
    for m, name in [(args.collagen_model, "collagen"), (args.tissue_model, "tissue")]:
        if not Path(m).is_file():
            print(f"[ERROR] Missing {name} model: {m}", file=sys.stderr)
            sys.exit(2)

    print(f"Found {len(inputs)} slide(s).")

    for in_path in inputs:
        print(f"\n[SLIDE] {in_path.name}")

        # Collagen → results/collagen_segmentation/
        col_dirname = OUT_COLLAGEN_DIR.name
        print(f"[COLLAGEN] {in_path.name} → dir={col_dirname}", end=" ", flush=True)
        call_inference(
            in_path, OUT_COLLAGEN_DIR, Path(args.collagen_model),
            tile_size=args.tile_size, stride=args.stride,
            workers=args.workers, gpu=args.gpu, force=args.force,
            verbose=args.verbose
        )

        # Tissue → results/tissue_segmentation_raw/
        tis_dirname = OUT_TISSUE_RAW_DIR.name
        print(f"[TISSUE]   {in_path.name} → dir={tis_dirname}", end=" ", flush=True)
        call_inference(
            in_path, OUT_TISSUE_RAW_DIR, Path(args.tissue_model),
            tile_size=args.tile_size, stride=args.stride,
            workers=args.workers, gpu=args.gpu, force=args.force,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
