# Collagen & Tissue Segmentation

The code provides deep-learning–based segmentation of liver biopsy slides stained with Picrosirius Red (PSR).  
It outputs two kinds of masks:

- **Collagen segmentation** – collagen fibres segmented by a U-Net model.  
- **Tissue segmentation** – tissue regions cleaned with morphological post-processing.

These are then used for downstream quantification of collagen proportionate area (CPA) and are needed as input to the collagen deposition phenotype (CDP) prediction pipeline.

The segmentation networks were trained on PSR slides from the CoCoMASLD cohort. 

---

## Project layout

```
histology/
  data/                       # input slides (NDPI, SVS, TIFF)
  results/
    collagen_segmentation/     # model outputs: collagen masks
    tissue_segmentation_raw/   # model outputs: raw tissue masks
    tissue_segmentation_closed/# post-processed tissue masks
    area_quantification/       # CSV tables with CPA metrics
  collagen-segmentation/
    models/                    # pretrained U-Nets (.h5)
    inference.py               # core tile-wise inference
    run_segmentation.py        # wrapper: batch process all slides
    postprocess_tissue_segmentation.py  # morphological closing
    compute_cpa.py             # collagen/tissue area quantification
    wsi_reader.py              # helper for reading large slides
```

---

## Installation

1. Install `conda` or `mamba`.  
2. Create the environment:
   ```bash
   mamba env create -f env/environment.yml
   conda activate segmentation_tf
   ```
3. Sanity check:
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
   ```
   On a machine with an NVIDIA GPU + driver ≥450, you should see a `GPU:0` listed.  
   Otherwise, TensorFlow will fall back to CPU (slower but functional).

---

## Usage

### 1. Place input slides
Put your PSR-stained slides (NDPI, SVS, or TIFF) in `histology/data/`.

### 2. Run segmentation
Use the wrapper script to run both collagen and tissue models in one go:

```bash
cd histology/collagen-segmentation
python run_segmentation.py
```

Options:
- `--workers N` : number of CPU workers for I/O (default: half your cores).  
- `--stride`    : 512 for no overlap; 256 for 50% overlap (default: 256).  
- `--force`     : overwrite existing outputs.  
- `--verbose`   : show full inference logs.  
- `--fail-fast` : abort immediately on first error (default: continue and summarise errors at the end).  

Outputs go to:
- `results/collagen_segmentation/*.tiff`  
- `results/tissue_segmentation_raw/*.tiff`

### 3. Post-process tissue masks
```bash
python postprocess_tissue_segmentation.py
```
This fills holes and applies morphological closing.  
Outputs go to `results/tissue_segmentation_closed/`.

### 4. Quantify collagen areas
```bash
python compute_cpa.py
```
This writes `results/cpa_results.csv` with per-slide tissue area, collagen area, and collagen proportionate area (CPA).

---

## Quick demo

To try the pipeline without your own data, download the demo slide from Zenodo:  

👉 [Demo slide and outputs on Zenodo](https://doi.org/10.5281/zenodo.16967316)

1. Place the demo file (i.e. `example_PSR_slide.ndpi`) in `histology/data/`.  
2. Run:
   ```bash
   cd histology/collagen-segmentation
   python run_segmentation.py
   python postprocess_tissue_segmentation.py
   python compute_cpa.py
   ```
3. Check outputs under `histology/results/`.

---

## Environment

The pipeline was tested with:
- Python 3.10  
- TensorFlow 2.11 (GPU build with CUDA 11.8, cuDNN 8.4)  
- numpy (≥1.24), pandas, scikit-image, tifffile, opencv, openslide, pyvips  

See [`environment.yml`](./env/environment.yml) for the full specification.

Other platforms may require adjustments.

---

## Notes
- Large slides can take several minutes each to process, depending on GPU speed and number of tiles.  
- The models expect 512×512 tiles.  
- For reproducibility: if you re-train models, keep the same tile size and preprocessing.
