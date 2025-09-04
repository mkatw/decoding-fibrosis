#!/usr/bin/env python3
"""
Compute collagen and tissue area fractions from segmentation outputs.

Assumed project layout:
  histology/
    results/
      collagen_segmentation/        (collagen masks, TIFFs)
      tissue_segmentation_closed/   (tissue masks, TIFFs)
      cpa_results.csv               (CSV written here)

Behaviour:
  - Each mask is read separately (low memory).
  - Masks are greyscale; pixels >= 128 are 'positive'.
  - Computes per-slide fractions and CPA:
        CPA = collagen_fraction / tissue_fraction
  - Output is saved to histology/results/cpa_results.csv
"""


from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

Image.MAX_IMAGE_PIXELS = int(1e11)  # allow very large TIFFs



def positive_fraction_128(path: Path) -> float:
    """
    Fraction of pixels >= 128 in a greyscale segmentation mask.
    """
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))  # greyscale 0–255
    total = arr.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(arr >= 128)) / float(total)


def main():
    # Locate histology root
    hist_root = Path(__file__).resolve().parent
    while hist_root.name != "histology" and hist_root != hist_root.parent:
        hist_root = hist_root.parent
    if hist_root.name != "histology":
        raise FileNotFoundError("Could not locate 'histology' root directory.")

    # Fixed paths
    collagen_dir = hist_root / "results" / "collagen_segmentation"
    tissue_dir   = hist_root / "results" / "tissue_segmentation_closed"
    out_file     = hist_root / "results" / "cpa_results.csv"

    print("Reading collagen masks from:", collagen_dir)
    print("Reading tissue masks   from:", tissue_dir)
    print("Saving results to       :", out_file)

    collagen_files = sorted(glob(str(collagen_dir / "*.tif*")))
    tissue_files   = sorted(glob(str(tissue_dir   / "*.tif*")))

    if not collagen_files or not tissue_files:
        raise FileNotFoundError(
            "No input files found!\n"
            f"  collagen_files: {collagen_files}\n"
            f"  tissue_files:   {tissue_files}"
        )


    if len(collagen_files) != len(tissue_files):
        print("[WARNING] Different number of collagen and tissue files!")
        print(f"  collagen: {len(collagen_files)} files")
        print(f"  tissue:   {len(tissue_files)} files")
        print("  Pairing by sorted order. Align names if this is unintended.")

    results = []
    for coll_path, tis_path in tqdm(zip(collagen_files, tissue_files), total=len(collagen_files)):
        slide_id = Path(coll_path).stem
        coll_frac = positive_fraction_128(Path(coll_path))
        tis_frac  = positive_fraction_128(Path(tis_path))
        cpa = (coll_frac / tis_frac) if tis_frac > 0 else 0.0
        results.append({
            "slide_id": slide_id,
            "tissue_fraction":   tis_frac,
            "collagen_fraction": coll_frac,
            "CPA":               cpa,
        })

    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"[OK] Wrote {out_file}  ({len(df)} slides)")


if __name__ == "__main__":
    main()
