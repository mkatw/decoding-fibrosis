"""
cdp_processor.py
Core logic for Collagen Deposition Phenotype (CDP) inference.
"""

import os
from pathlib import Path
import numpy as np
import cv2
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms.functional import to_tensor
from sklearn.preprocessing import normalize
from PIL import Image
from tqdm import tqdm
from skimage import color
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


# -----------------------------
# Models
# -----------------------------

class ResNet(nn.Module):
    """ResNet18 feature extractor, final FC removed."""
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

    def forward(self, x):
        return self.resnet(x)


def normalize_input(x: torch.Tensor) -> torch.Tensor:
    """Per-channel z-score normalisation."""
    c = x.shape[0]
    mean = x.view(c, -1).mean(dim=-1)[:, None, None].expand_as(x)
    std = x.view(c, -1).std(dim=-1)[:, None, None].expand_as(x)
    return (x - mean) / (std + 1e-9)


# -----------------------------
# Tile handling
# -----------------------------

def get_output_dimensions(image, level: int, tile_size: int):
    width, height = image.level_dimensions[level]
    tiles_horizontal = int(np.floor(width / tile_size))
    tiles_vertical = int(np.floor(height / tile_size))
    return tiles_horizontal, tiles_vertical


def get_tile(slide, level: int, tile_size: int, t_h: int, t_v: int):
    x = t_h * 2 ** level * tile_size
    y = t_v * 2 ** level * tile_size
    tile_image, _ = slide.read_region((x, y), level, (tile_size, tile_size))
    return tile_image


def is_tissue(tile: np.ndarray, mask_threshold: float, cutoff: float) -> bool:
    """Decide if tile is tissue based on grayscale Otsu threshold."""
    grey_image = color.rgb2gray(tile)
    tissue_mask = (grey_image < mask_threshold / 255) & (grey_image > 0)
    tissue_mask_filled = ndi.binary_fill_holes(tissue_mask)
    tissue_ratio = tissue_mask_filled.sum() / tissue_mask.size
    return tissue_ratio > cutoff


# -----------------------------
# Feature extraction
# -----------------------------


def get_otsu_threshold(slide,bounds=False):

    ds = 4
    level = slide.get_best_level_for_downsample(ds)
    dims = slide.level_dimensions[level]
    slide_downsampled, alfa_mask = slide.get_downsampled_slide(dims, normalize=False)
    alfa = alfa_mask.astype(np.uint8).ravel()
    slide_downsampled = cv2.cvtColor(slide_downsampled, cv2.COLOR_RGB2GRAY).ravel()[alfa > 0]
    threshold, _ = cv2.threshold(slide_downsampled, 0, 255, cv2.THRESH_OTSU)
    if bounds:
        if threshold > 0.9 * 255:
            threshold = 0.9 * 255
        if threshold < 0.8 * 255:
            threshold = 0.8 * 255

    return threshold


def extract_features(tile: np.ndarray, preprocessing_model, model: nn.Module, device="cuda"):
    """Extract ResNet18 features from a collagen probability tile."""
    tile = (tile / 255).astype(np.float32)
    tile = np.expand_dims(tile, axis=0)
    y = (np.squeeze(preprocessing_model.predict(tile, verbose=0), 3) * 255).astype(np.uint8)
    tile = np.squeeze(y)

    img = Image.fromarray(tile).convert("RGB")
    img_data = to_tensor(img).to(device)
    img_data = normalize_input(img_data)

    with torch.no_grad():
        features = model(img_data.unsqueeze(0)).squeeze(0).cpu().numpy()

    features = features.flatten()
    return normalize([features], axis=1)


def get_predictions(slide, level, tiles_horizontal, tile_size, tiles_vertical,
                    mask_threshold, preprocessing_model, model, kmeans, label_mapping,
                    sub_kmeans=None, sub_label_mapping=None):
    """Run inference for one slide and return prediction grid."""
    predictions = np.full((tiles_horizontal, tiles_vertical), -1)

    for t_h in tqdm(range(tiles_horizontal)):
        for t_v in range(tiles_vertical):
            tile = get_tile(slide, level, tile_size, t_h, t_v)
            tile = (tile * 255).astype(np.uint8)

            if is_tissue(tile, mask_threshold, 0.3):
                features = extract_features(tile, preprocessing_model, model)
                prediction = kmeans.predict(features)[0]
                prediction_corrected = label_mapping[prediction]

                if prediction_corrected == 4 and sub_kmeans is not None:
                    prediction = sub_kmeans.predict(features)[0]
                    prediction_corrected = sub_label_mapping[prediction]

                predictions[t_h, t_v] = prediction_corrected
            else:
                predictions[t_h, t_v] = -1

    return predictions


# -----------------------------
# Plotting
# -----------------------------

def plot_results(slide, predictions, case_name, k, output_dir: Path):
    """Save overlay and histogram plots."""
    clusters = np.arange(0, k)
    upscale_parameter = 10
    preds = np.transpose(predictions)
    preds_upsampled = preds.repeat(upscale_parameter, axis=0).repeat(upscale_parameter, axis=1)
    target_dims = preds_upsampled.shape
    thumbnail, _ = slide.get_downsampled_slide((target_dims[1], target_dims[0]))
    thumbnail = (thumbnail * 255).astype(np.uint8)

    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    preds_upsampled = np.ma.array(preds_upsampled, mask=(preds_upsampled == -1))

    fig = plt.figure(figsize=(9, 8))
    plt.imshow(preds_upsampled, cmap="hot_r", vmin=0, vmax=k - 1)
    plt.colorbar(ticks=clusters)
    plt.imshow(thumbnail, alpha=0.7)
    plt.axis("off")
    plt.title(case_name)
    plt.savefig(overlay_dir / f"{case_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
