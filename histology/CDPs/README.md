# Collagen Deposition Phenotypes (CDPs)

This folder contains the **paper version** of the CDP inference workflow.  
It runs collagen segmentation (U-Net, TensorFlow), feature embedding (ResNet18, PyTorch), and k-means clustering to assign **Collagen Deposition Phenotypes** tile-by-tile across PSR-stained slides.

---

## Workflow overview

1. **Input**: PSR whole-slide images in `histology/data/` (filenames containing `*PSR*`).
2. **Segmentation**: a pre-trained U-Net generates collagen probability maps. 
3. **Feature extraction**: each collagen tile is embedded using ResNet18.
4. **Clustering**: features are classified into k=7 CDPs using the supplied k-means models and label mappings.
5. **Output**:  
   - Prediction map (`.npy`) with CDP label per tile  
   - Overlay PNG with CDPs visualised on the slide  

All results are written to `../histology/results/`.

---

## Contents

- `batch_process_clustering.py` # CLI wrapper (entry point)
- `cdp_processor.py` # core logic (importable)
- `kmeans_classifiers/` # pre-trained k-means + label mapping
- `training/` # scripts for sampling tiles and retraining CDP k-means models

---

## Example usage

From the repository root:

```bash
python histology/CDPs/batch_process_clustering.py 
```

---

## Notes

The pipeline may be slow, particularly if no GPU is available.

While you could technically bypass segmentation and start from precomputed collagen maps, this is not recommended: the results may differ subtly from those reported in the paper, and only the full pipeline is guaranteed to reproduce them.

The classifiers were trained to handle embeddings from 512 by 512 px tiles extracted at level 0 of 40x WSIs. If your data was scanned at 20x you may need to upsample each tile before passing it to the feature extractor. I haven't tried it!

GPU is optional. Pipeline defaults to CPU if no CUDA is available.

## Training the clustering models

The k-means classifiers can be retrained from a pool of PSR tissue tiles. See
[`training/`](training) for the cleaned public workflow:

1. sample tissue-containing PSR tiles
2. segment the tile pool with the collagen U-Net
3. extract ResNet18 embeddings and fit the primary/sub-cluster k-means models

The original paper tile pool was stored, but the random seed used to generate it
was not retained. Use the released clustering tile/feature artifact for exact
paper reproduction when available; regenerated tile pools should be treated as
comparable rather than identical.
