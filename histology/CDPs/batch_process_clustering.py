"""
batch_process_clustering.py
Thin CLI wrapper for batch CDP inference.
"""

import os
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from joblib import load
import json
from wsi_reader import TiffReader
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF info/warnings
import tensorflow as tf
import torch
from cdp_processor import ResNet, get_otsu_threshold, get_output_dimensions, get_predictions, plot_results


def main(args):
    # Project paths
    root = Path(__file__).resolve().parents[1]
    slides_root = root / "data"
    output_dir = root / "results" / "CDPs"
    output_dir.mkdir(exist_ok=True)
    classifiers_root = root / "CDPs" / "kmeans_classifiers"
    k = 7  # number of clusters

    # Load classifiers
    kmeans = load(classifiers_root / "cdps_k5.joblib")
    with open(classifiers_root / 'cdps_k5.labels.json') as f:
        label_mapping = {int(key): v for key, v in json.load(f).items()}
    #label_mapping = {-1: -1, 0: 1, 1: 4, 2: 2, 3: 0, 4: 3}
    sub_kmeans = load(classifiers_root / "cdp4_k3.joblib")
    with open(classifiers_root / 'cdp4_k3.labels.json') as f:
        sub_label_mapping = {int(key): v for key, v in json.load(f).items()}

    # Load ResNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet().to(device).eval()

    # Load U-Net
    preprocessing_model = tf.keras.models.load_model(root / "collagen-segmentation" / "models" / "unet_mini_CoCoMASLD_PSR_collagen.h5")

    # Process slides
    slides = sorted(slides_root.glob("*PSR*"))
    for slide_path in slides:
        case_name = Path(slide_path).stem
        file_name = output_dir / f"{case_name}_prediction.npy"

        if file_name.exists() and args.ignore_existing:
            print(f"{file_name} exists, skipping...")
            continue

        slide = TiffReader(slide_path, series=0)
        mask_threshold = get_otsu_threshold(slide)
        tiles_horizontal, tiles_vertical = get_output_dimensions(slide, 0, 512)

        predictions = get_predictions(
            slide, 0, tiles_horizontal, 512, tiles_vertical,
            mask_threshold, preprocessing_model, model,
            kmeans, label_mapping
        )

        np.save(file_name, predictions)
        plot_results(slide, predictions, case_name, k, output_dir)

if __name__ == "__main__":
    parser = ArgumentParser(description="Batch process clustering for CDPs")
    parser.add_argument("--ignore_existing", action="store_true", help="Skip cases with saved outputs")
    args = parser.parse_args()
    main(args)
