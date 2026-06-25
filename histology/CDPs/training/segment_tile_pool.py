#!/usr/bin/env python3
"""
Apply the collagen segmentation U-Net to a tile pool.

The CDP clustering workflow uses ResNet18 embeddings of collagen probability
tiles. This helper converts raw PSR tile images into 8-bit collagen probability
maps compatible with the training script and the inference pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def iter_tiles(tile_dir: Path) -> list[Path]:
    return sorted(
        path for path in tile_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_tile(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def save_prediction(prediction: np.ndarray, path: Path, binary: bool) -> None:
    prediction = np.squeeze(prediction)
    if binary:
        output = (prediction > 0.5).astype(np.uint8) * 255
    else:
        output = np.clip(prediction * 255, 0, 255).astype(np.uint8)
    Image.fromarray(output, mode="L").save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment a PSR tile pool into collagen maps.")
    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Collagen U-Net .h5 model")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--binary", action="store_true", help="Write thresholded masks instead of probability maps")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    import tensorflow as tf

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = iter_tiles(args.tile_dir)
    if not files:
        raise SystemExit(f"No image tiles found in {args.tile_dir}")

    model = tf.keras.models.load_model(args.model)

    for start in range(0, len(files), args.batch_size):
        batch_files = files[start : start + args.batch_size]
        batch = np.stack([load_tile(path) for path in batch_files], axis=0)
        predictions = model.predict(batch, verbose=0)

        for tile_path, prediction in zip(batch_files, predictions):
            out_path = args.out_dir / f"{tile_path.stem}.png"
            save_prediction(prediction, out_path, args.binary)

        print(f"Segmented {min(start + args.batch_size, len(files))}/{len(files)} tiles")


if __name__ == "__main__":
    main(parse_args())
