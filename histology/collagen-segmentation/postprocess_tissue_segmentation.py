#!/usr/bin/env python3
"""
Post-process tissue segmentation masks.

Fixed project layout:
  histology/
    results/
      tissue_segmentation_raw/      (input masks)
      tissue_segmentation_closed/   (output masks written here)

Behaviour:
  - Loads each mask, converts to 1-bit (no dithering), then:
      remove_small_holes -> closing(disk=3) -> remove_small_holes
  - Saves uint8 masks (0/255) with same filenames in .../tissue_segmentation_closed/.
  - Skips already processed files unless --force is given.
"""

import os
from pathlib import Path
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import morphology, img_as_ubyte
import argparse
import tifffile as tiff
import pyvips
import tempfile

Image.MAX_IMAGE_PIXELS = int(1e11)  # allow very large WSIs

tiffsave_kwargs = dict(
    tile=True,
    tile_width=256,
    tile_height=256,
    squash=False,
    pyramid=False,
    bigtiff=True,
    compression="deflate",
    properties=False,
)

def process_slide(path: Path, out_dir: Path, force: bool = False):
    """Process a single tissue segmentation map into a closed mask."""
    slide_name = Path(path).stem
    if slide_name.endswith("_raw"):
        slide_name = slide_name[:-4]  # drop the "_raw"

    out_path = out_dir / f"{slide_name}.tiff"

    if out_path.exists() and not force:
        print(f"[SKIP] {out_path.name} already exists")
        return

    slide = Image.open(path)
    # Explicitly disable dithering to avoid speckle.
    segmented = slide.convert("1", dither=Image.Dither.NONE)
    arr = np.asarray(segmented, dtype=bool)

    # Morphology: fill small holes, close, fill larger holes
    arr = morphology.remove_small_holes(arr, area_threshold=2**14)
    arr = morphology.closing(arr, morphology.disk(3))
    arr = morphology.remove_small_holes(arr, area_threshold=2**20)

    # Save to temporary tiled BigTIFF with tifffile
    arr_u8 = img_as_ubyte(arr)
    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
        tmp_name = tmp.name
        tiff.imwrite(tmp_name, arr_u8, bigtiff=True, tile=(256, 256), photometric="minisblack")

    # Re-open with pyvips and save as pyramidal compressed TIFF
    pyvips.Image.new_from_file(tmp_name).tiffsave(str(out_path), **tiffsave_kwargs)

    # cleanup
    os.remove(tmp_name)

def main(force: bool = False):
    hist_root = Path("histology").resolve().parent.parent
    input_dir = hist_root / "results" / "tissue_segmentation_raw"
    out_dir   = hist_root / "results" / "tissue_segmentation_closed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading raw tissue masks from:", input_dir)
    print("Writing closed masks to      :", out_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {input_dir}")

    files = sorted(glob(str(input_dir / "*.tif*")))
    if not files:
        raise FileNotFoundError(f"No input files matching *.tif* under {input_dir}")

    ok, skipped, failed = 0, 0, 0
    for f in tqdm(files):
        try:
            out_path = out_dir / Path(f).name
            if out_path.exists() and not force:
                print(f"[SKIP] {out_path.name} already exists")
                skipped += 1
                continue
            process_slide(Path(f), out_dir, force=force)
            print(f" [PROCESSING] {out_path.name}", flush=True)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] Failed on {f}: {e}")

    print(f"[DONE] Processed {ok} file(s). Skipped: {skipped}. Failed: {failed}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate closed tissue masks from raw segmentation maps.")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing closed masks")
    args = parser.parse_args()
    main(force=args.force)
