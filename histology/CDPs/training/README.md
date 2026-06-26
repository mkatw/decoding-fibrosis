# CDP Clustering Training

This folder contains the training workflow for the collagen deposition
phenotype (CDP) k-means classifiers.

The paper inference pipeline uses:

- a collagen U-Net to convert PSR tiles into collagen probability maps
- ImageNet-pretrained ResNet18 embeddings of those collagen maps
- a primary k-means model for CDP labels 0-4
- a second k-means model that splits the highest-collagen primary cluster into
  CDP labels 4-6

## Reproducibility Note

The random seed used to generate the original CDP clustering tile pool was not
retained. For exact reproduction of the paper classifiers, use the released
trained models. The scripts here can regenerate a comparable training pool,
but the sampled tiles may not be identical.

## Environment

Use the CDP inference environment from `histology/CDPs/environment.yml`. The
training and K-search scripts share the same core dependencies as the inference
pipeline, with `matplotlib` and `umap-learn` used for diagnostic plots.

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

### 3. Search Candidate Cluster Numbers

The original model-development workflow was iterative: fit clusters for a range
of K values, inspect summary metrics, and visually check example tiles from each
cluster.

```bash
python histology/CDPs/training/search_cdp_clusters.py \
  --tile-dir histology/CDPs/training/data/collagen_tiles \
  --out-dir histology/CDPs/training/models/k_search \
  --feature-cache histology/CDPs/training/models/cdp_training_features.npz \
  --k-min 2 \
  --k-max 10 \
  --algorithm kmeans
```

This writes:

```text
k_search_metrics.csv
k_search_cluster_stats.csv
k_search_metrics.png
k_search_cluster_sizes.png
k_search_mean_collagen_fraction.png
k02/random_examples.png
k02/centroid_examples.png
...
```

`random_examples.png` and `centroid_examples.png` are the main sanity-check
plots: each row is one collagen-sorted cluster, with representative tiles from
that cluster.

Useful options:

- `--algorithm minibatch_kmeans` for faster approximate k-means on large tile pools
- `--algorithm agglomerative_ward` for Ward agglomerative clustering experiments
- `--embedding-method pca` or `--embedding-method umap` to add per-K embedding
  and silhouette plots
- `--silhouette-sample-size 0` to compute silhouette diagnostics on all tiles
- `--copy-clustered-images` to copy every tile into per-cluster folders

### 4. Train CDP Classifiers

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
