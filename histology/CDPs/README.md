# Collagen Deposition Phenotypes (CDPs)

This folder contains the **paper version** of the CDP inference workflow.  
It runs collagen segmentation (U-Net, TensorFlow), feature embedding (ResNet18, PyTorch), and k-means clustering to assign **Collagen Deposition Phenotypes** tile-by-tile across PSR-stained slides.

---

## Workflow overview

1. **Input**: PSR whole-slide images in `histology/data/` (filenames containing `*PSR*`).
2. **Segmentation**: a pre-trained U-Net generates collagen probability maps. 
3. **Feature extraction**: each collagen tile is embedded using ResNet18.
4. **Clustering**: features are classified into **k=7 CDPs** using the supplied k-means models and label mappings.
5. **Output**:  
   - Prediction map (`.npy`) with CDP label per tile  
   - Overlay PNG with CDPs visualised on the slide  
   - Histogram PNG of CDP distribution

All results are written to `../histology/results/`.

---

## Contents

- `batch_process_clustering.py` # CLI wrapper (entry point)
- `cdp_processor.py` # core logic (importable)
- `kmeans_classifiers/` # pre-trained k-means + label mapping


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

This workflow supports k=7 clusters (paper version).

GPU is optional. Pipeline defaults to CPU if no CUDA is available.
