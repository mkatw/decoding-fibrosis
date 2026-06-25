# CDP Clustering Training

This folder contains the public training workflow for the collagen deposition
phenotype (CDP) k-means classifiers.

The paper inference pipeline uses:

- a collagen U-Net to convert PSR tiles into collagen probability maps
- ImageNet-pretrained ResNet18 embeddings of those collagen maps
- a primary k-means model for CDP labels 0-4
- a second k-means model that splits the highest-collagen primary cluster into
  CDP labels 4-6

## Reproducibility Note

The original CDP clustering tile pool was a stored tissue-tile sample. The
random seed used to generate that exact pool was not retained. For exact
reproduction of the paper classifiers, use the released clustering tile or
feature artifact when available. The scripts here can regenerate a comparable
pool from PSR slides, but the sampled tiles may not be identical.

## Workflow

### 1. Sample Tissue Tiles

```bash
python histology/CDPs/training/extract_tissue_tiles.py \
  --input-glob "histology/data/*PSR*" \
  --out-dir histology/CDPs/training/data/tissue_tiles \
  --sample-probability 0.01 \
  --seed 42
```

This writes sampled 512 x 512 tissue tiles and a `tile_manifest.csv`.

### 2. Segment Tile Pool

```bash
python histology/CDPs/training/segment_tile_pool.py \
  --tile-dir histology/CDPs/training/data/tissue_tiles \
  --model histology/collagen-segmentation/models/unet_mini_CoCoMASLD_PSR_collagen.h5 \
  --out-dir histology/CDPs/training/data/collagen_tiles
```

By default, this writes 8-bit collagen probability maps, matching the inference
pipeline. Use `--binary` only for experiments.

### 3. Train CDP Classifiers

```bash
python histology/CDPs/training/train_cdp_clusters.py \
  --tile-dir histology/CDPs/training/data/collagen_tiles \
  --out-dir histology/CDPs/training/models \
  --n-clusters 5 \
  --subcluster-label 4 \
  --n-subclusters 3
```

The output files are compatible with `histology/CDPs/batch_process_clustering.py`:

```text
cdps_k5.joblib
cdps_k5.labels.json
cdp4_k3.joblib
cdp4_k3.labels.json
cdp_cluster_training_manifest.csv
```

To use newly trained classifiers for inference, copy the `.joblib` and
`.labels.json` files into `histology/CDPs/kmeans_classifiers/`.

## Notes

- Labels are ordered by mean collagen fraction so that higher labels correspond
  to higher collagen content.
- The sub-clustering step is trained within primary label 4 and maps sub-cluster
  labels to CDP labels 4, 5, and 6.
- `--device auto` uses CUDA when available and CPU otherwise.
- ResNet18 weights are loaded through `torchvision`; the first run may download
  ImageNet weights if they are not already cached.
