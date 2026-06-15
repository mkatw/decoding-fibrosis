# Collagen Segmentation Training

This folder contains the training entry point for the PSR collagen U-Net used by
the segmentation and CDP pipelines.

The script supports two public workflows:

- train a new U-Net from paired image/mask tiles
- fine-tune an existing `.h5` model with additional paired annotations

The architecture is fixed to the U-Net mini used in the paper.

## Data Layout

Each training dataset should contain RGB image tiles and binary collagen masks
with matching filename stems:

```text
collagen_training_data/
  images/
    tile_001.jpg
    tile_002.jpg
  masks/
    tile_001.png
    tile_002.png
```

Masks are read as grayscale images. Any pixel value above zero is treated as
collagen. The default tile size is 512 x 512 pixels, matching the inference
pipeline.

For the extended cohort retraining, the local source layout was:

```text
extended_cohort/collagen_segmentation_QuPath/
  original_training_set/
    images/
    masks/
  paired_for_training/
    images/
    masks/
```

## Environment

From `histology/collagen-segmentation`:

```bash
mamba env create -f env/environment.yml
conda activate segmentation_tf
```

## Train From Scratch

```bash
python training/train_collagen_unet.py \
  --data-root data/collagen_training/original_training_set \
  --out models/unet_mini_CoCoMASLD_PSR_collagen_retrained.h5 \
  --epochs 30 \
  --batch-size 8 \
  --augment
```

## Fine-Tune The Public Collagen Model

```bash
python training/train_collagen_unet.py \
  --data-root data/collagen_training/original_training_set \
  --data-root data/collagen_training/paired_for_training \
  --pretrained models/unet_mini_CoCoMASLD_PSR_collagen.h5 \
  --out models/unet_mini_CoCoMASLD_PSR_collagen_finetuned.h5 \
  --epochs 30 \
  --batch-size 8 \
  --augment
```

Repeat `--data-root` for each paired tile dataset to include in training.

## Outputs

For an output model path such as:

```text
models/unet_mini_CoCoMASLD_PSR_collagen_finetuned.h5
```

the script also writes:

- `*.pairs.csv`: image/mask pairs and train/validation split
- `*.training_log.csv`: epoch-wise metrics
- `*.history.json`: Keras training history
- `*.metrics.png`: loss and Dice curves, if matplotlib is installed

The `.pairs.csv` file is useful for recording the exact split used for a
published model.

## Notes

- The script loads tile arrays into memory. For very large public training sets,
  split the dataset or adapt the loader to stream batches from disk.
- The default validation split is random but seeded with `--seed`.
- Tiles are expected to be 512 x 512 pixels.
