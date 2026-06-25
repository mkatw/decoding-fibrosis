#!/usr/bin/env python3
"""
Train CDP k-means classifiers from collagen probability tiles.

Inputs are 512 x 512 collagen maps, normally produced by segment_tile_pool.py.
The script writes joblib k-means models and JSON label mappings compatible with
histology/CDPs/batch_process_clustering.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.functional import to_tensor
import torch
from torch import nn


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
TILE_SIZE = 512


@dataclass(frozen=True)
class FeatureSet:
    features: np.ndarray
    files: list[Path]
    collagen_fraction: np.ndarray


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = torch.nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, x):
        return self.resnet(x)


def normalize_input(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[0]
    mean = x.view(c, -1).mean(dim=-1)[:, None, None].expand_as(x)
    std = x.view(c, -1).std(dim=-1)[:, None, None].expand_as(x)
    return (x - mean) / (std + 1e-9)


def iter_tiles(tile_dir: Path) -> list[Path]:
    return sorted(
        path for path in tile_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_collagen_tile(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("L")
        if image.size != (TILE_SIZE, TILE_SIZE):
            raise ValueError(f"{path} is {image.size}, expected {(TILE_SIZE, TILE_SIZE)}")
        return np.asarray(image)


def select_files(files: list[Path], fraction: float, seed: int) -> list[Path]:
    if not 0 < fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")
    if fraction == 1:
        return files
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n_files = max(1, int(round(len(shuffled) * fraction)))
    return sorted(shuffled[:n_files])


def extract_features(tile_dir: Path, device: torch.device, fraction: float, seed: int) -> FeatureSet:
    files = select_files(iter_tiles(tile_dir), fraction, seed)
    if not files:
        raise SystemExit(f"No collagen tiles found in {tile_dir}")

    model = ResNet().to(device).eval()
    feature_rows = []
    collagen_fraction = []

    with torch.no_grad():
        for index, path in enumerate(files, start=1):
            tile = load_collagen_tile(path)
            collagen_fraction.append(float((tile > 127).sum() / tile.size))

            image = Image.fromarray(tile).convert("RGB")
            tensor = normalize_input(to_tensor(image).to(device))
            features = model(tensor.unsqueeze(0)).squeeze(0).cpu().numpy().flatten()
            feature_rows.append(features)

            if index % 100 == 0:
                print(f"Extracted features for {index}/{len(files)} tiles")

    features = normalize(np.asarray(feature_rows), axis=1)
    return FeatureSet(features=features, files=files, collagen_fraction=np.asarray(collagen_fraction))


def train_kmeans(features: np.ndarray, n_clusters: int, seed: int) -> KMeans:
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(features)


def collagen_sorted_label_mapping(raw_labels: np.ndarray, collagen_fraction: np.ndarray, offset: int = 0) -> dict[int, int]:
    mapping = {}
    unique_labels = sorted(np.unique(raw_labels))
    mean_collagen = {
        label: float(np.mean(collagen_fraction[raw_labels == label]))
        for label in unique_labels
    }
    for new_label, raw_label in enumerate(sorted(unique_labels, key=mean_collagen.get)):
        mapping[int(raw_label)] = int(new_label + offset)
    return mapping


def apply_mapping(raw_labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.asarray([mapping[int(label)] for label in raw_labels])


def save_mapping(path: Path, mapping: dict[int, int]) -> None:
    with path.open("w") as handle:
        json.dump({str(key): value for key, value in mapping.items()}, handle, indent=2)


def write_manifest(
    path: Path,
    feature_set: FeatureSet,
    primary_raw_labels: np.ndarray,
    primary_labels: np.ndarray,
    sub_indices: np.ndarray,
    sub_raw_labels: np.ndarray,
    sub_labels: np.ndarray,
) -> None:
    sub_raw_by_index = {
        int(index): int(label)
        for index, label in zip(sub_indices, sub_raw_labels)
    }
    sub_label_by_index = {
        int(index): int(label)
        for index, label in zip(sub_indices, sub_labels)
    }

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile",
                "collagen_fraction",
                "primary_raw_label",
                "primary_label",
                "sub_raw_label",
                "sub_label",
            ],
        )
        writer.writeheader()
        for index, file_path in enumerate(feature_set.files):
            writer.writerow(
                {
                    "tile": str(file_path),
                    "collagen_fraction": f"{feature_set.collagen_fraction[index]:.8f}",
                    "primary_raw_label": int(primary_raw_labels[index]),
                    "primary_label": int(primary_labels[index]),
                    "sub_raw_label": sub_raw_by_index.get(index, ""),
                    "sub_label": sub_label_by_index.get(index, ""),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CDP k-means classifiers.")
    parser.add_argument("--tile-dir", type=Path, required=True, help="Collagen probability tile directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--subcluster-label", type=int, default=4)
    parser.add_argument("--n-subclusters", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=1.0, help="Optional fraction of tiles to use")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-features", action="store_true")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    return torch.device(device_arg)


def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    feature_set = extract_features(args.tile_dir, device, args.fraction, args.seed)
    if args.save_features:
        np.savez_compressed(
            args.out_dir / "cdp_training_features.npz",
            features=feature_set.features,
            files=np.asarray([str(path) for path in feature_set.files]),
            collagen_fraction=feature_set.collagen_fraction,
        )

    primary = train_kmeans(feature_set.features, args.n_clusters, args.seed)
    primary_raw_labels = primary.labels_
    primary_mapping = collagen_sorted_label_mapping(
        primary_raw_labels,
        feature_set.collagen_fraction,
        offset=0,
    )
    primary_labels = apply_mapping(primary_raw_labels, primary_mapping)

    primary_model_path = args.out_dir / f"cdps_k{args.n_clusters}.joblib"
    primary_mapping_path = args.out_dir / f"cdps_k{args.n_clusters}.labels.json"
    dump(primary, primary_model_path)
    save_mapping(primary_mapping_path, primary_mapping)

    sub_indices = np.where(primary_labels == args.subcluster_label)[0]
    if len(sub_indices) < args.n_subclusters:
        raise SystemExit(
            f"Only {len(sub_indices)} tiles have primary label {args.subcluster_label}; "
            f"cannot train {args.n_subclusters} subclusters"
        )

    sub = train_kmeans(feature_set.features[sub_indices], args.n_subclusters, args.seed)
    sub_raw_labels = sub.labels_
    sub_mapping = collagen_sorted_label_mapping(
        sub_raw_labels,
        feature_set.collagen_fraction[sub_indices],
        offset=args.subcluster_label,
    )
    sub_labels = apply_mapping(sub_raw_labels, sub_mapping)

    sub_model_path = args.out_dir / f"cdp{args.subcluster_label}_k{args.n_subclusters}.joblib"
    sub_mapping_path = args.out_dir / f"cdp{args.subcluster_label}_k{args.n_subclusters}.labels.json"
    dump(sub, sub_model_path)
    save_mapping(sub_mapping_path, sub_mapping)

    write_manifest(
        args.out_dir / "cdp_cluster_training_manifest.csv",
        feature_set,
        primary_raw_labels,
        primary_labels,
        sub_indices,
        sub_raw_labels,
        sub_labels,
    )

    print(f"Wrote {primary_model_path}")
    print(f"Wrote {primary_mapping_path}")
    print(f"Wrote {sub_model_path}")
    print(f"Wrote {sub_mapping_path}")


if __name__ == "__main__":
    main(parse_args())
