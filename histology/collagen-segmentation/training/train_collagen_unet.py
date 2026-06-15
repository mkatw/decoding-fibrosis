#!/usr/bin/env python3
"""
Train or fine-tune the PSR collagen segmentation U-Net mini.

Each dataset root should contain paired image and mask folders:

  dataset_root/
    images/
      tile_001.jpg
    masks/
      tile_001.png

Image and mask files are matched by filename stem. Masks are converted to
binary targets with mask > 0 treated as collagen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
TILE_SIZE = 512


@dataclass(frozen=True)
class TilePair:
    image: Path
    mask: Path
    source: str


def dice_coef_keras(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def files_by_stem(directory: Path, extensions: Iterable[str]) -> dict[str, Path]:
    files = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in files:
            raise ValueError(f"Duplicate file stem in {directory}: {path.stem}")
        files[path.stem] = path
    return files


def discover_pairs(data_root: Path) -> list[TilePair]:
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"{data_root} must contain images/ and masks/ subdirectories"
        )

    images = files_by_stem(image_dir, IMAGE_EXTENSIONS)
    masks = files_by_stem(mask_dir, MASK_EXTENSIONS)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"Unpaired files in {data_root}: "
            f"missing masks for {missing_masks[:5]}; "
            f"missing images for {missing_images[:5]}"
        )

    return [
        TilePair(image=images[stem], mask=masks[stem], source=str(data_root))
        for stem in sorted(images)
    ]


def split_pairs(
    pairs: list[TilePair],
    val_split: float,
    seed: int,
) -> tuple[list[TilePair], list[TilePair]]:
    if not 0 < val_split < 1:
        raise ValueError("--val-split must be between 0 and 1")
    if len(pairs) < 2:
        raise ValueError("At least two paired tiles are needed")

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    val_n = max(1, int(round(len(shuffled) * val_split)))
    val_n = min(val_n, len(shuffled) - 1)
    return shuffled[val_n:], shuffled[:val_n]


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if image.size != (TILE_SIZE, TILE_SIZE):
            raise ValueError(f"{path} is {image.size}, expected {(TILE_SIZE, TILE_SIZE)}")
        return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        mask = handle.convert("L")
        if mask.size != (TILE_SIZE, TILE_SIZE):
            raise ValueError(f"{path} is {mask.size}, expected {(TILE_SIZE, TILE_SIZE)}")
        return (np.asarray(mask) > 0).astype(np.float32)[..., np.newaxis]


def load_arrays(pairs: list[TilePair]) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(pairs), TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
    y = np.zeros((len(pairs), TILE_SIZE, TILE_SIZE, 1), dtype=np.float32)

    for index, pair in enumerate(pairs):
        x[index] = load_image(pair.image)
        y[index] = load_mask(pair.mask)
        if (index + 1) % 100 == 0:
            print(f"Loaded {index + 1}/{len(pairs)} tile pairs")

    return x, y


def batch_generator(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    augment: bool,
    shuffle: bool,
):
    indices = np.arange(x.shape[0])
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, x.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x[batch_idx].copy()
            y_batch = y[batch_idx].copy()
            if augment:
                augment_batch(x_batch, y_batch)
            yield x_batch, y_batch


def augment_batch(x_batch: np.ndarray, y_batch: np.ndarray) -> None:
    for index in range(x_batch.shape[0]):
        if random.random() < 0.5:
            x_batch[index] = np.fliplr(x_batch[index])
            y_batch[index] = np.fliplr(y_batch[index])
        if random.random() < 0.5:
            x_batch[index] = np.flipud(x_batch[index])
            y_batch[index] = np.flipud(y_batch[index])
        if random.random() < 0.25:
            k = random.choice([1, 2, 3])
            x_batch[index] = np.rot90(x_batch[index], k)
            y_batch[index] = np.rot90(y_batch[index], k)


def build_unet_mini():
    try:
        from keras_unet_collection import models
    except ImportError as exc:
        raise ImportError(
            "Training from scratch requires keras-unet-collection. "
            "Install the provided environment, or use --pretrained to fine-tune "
            "an existing .h5 model."
        ) from exc

    return models.unet_2d(
        input_size=(TILE_SIZE, TILE_SIZE, 3),
        filter_num=[32, 64, 128],
        n_labels=1,
        activation="ReLU",
        output_activation="Sigmoid",
        stack_num_down=2,
        stack_num_up=2,
        pool="max",
        unpool="bilinear",
        name="unet_mini",
    )


def compile_model(model, learning_rate: float) -> None:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            dice_coef_keras,
        ],
    )


def fit_model(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args: argparse.Namespace,
) -> tf.keras.callbacks.History:
    for layer in model.layers:
        layer.trainable = True
    compile_model(model, args.learning_rate)

    train_gen = batch_generator(
        train_x, train_y, args.batch_size, augment=args.augment, shuffle=True
    )
    val_gen = batch_generator(
        val_x, val_y, args.batch_size, augment=False, shuffle=False
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            str(args.out),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        CSVLogger(str(args.out.with_suffix(".training_log.csv")), append=True),
    ]

    return model.fit(
        train_gen,
        steps_per_epoch=max(1, math.ceil(train_x.shape[0] / args.batch_size)),
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=max(1, math.ceil(val_x.shape[0] / args.batch_size)),
        callbacks=callbacks,
        verbose=1,
    )


def write_pairs_csv(path: Path, train_pairs: list[TilePair], val_pairs: list[TilePair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "source", "image", "mask"],
        )
        writer.writeheader()
        for split, pairs in (("train", train_pairs), ("validation", val_pairs)):
            for pair in pairs:
                writer.writerow(
                    {
                        "split": split,
                        "source": pair.source,
                        "image": str(pair.image),
                        "mask": str(pair.mask),
                    }
                )


def save_history(history: tf.keras.callbacks.History, path: Path) -> None:
    serializable = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    with path.open("w") as handle:
        json.dump(serializable, handle, indent=2)


def save_metrics_plot(history: tf.keras.callbacks.History, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("dice_coef_keras", []), label="train_dice")
    plt.plot(history.history.get("val_dice_coef_keras", []), label="val_dice")
    plt.legend()
    plt.title("Dice")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune the PSR collagen U-Net mini."
    )
    parser.add_argument(
        "--data-root",
        action="append",
        required=True,
        type=Path,
        help="Paired tile root with images/ and masks/. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        help="Existing .h5 model to fine-tune. Omit to train from scratch.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the best model checkpoint (.h5).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    args.learning_rate = 5e-5 if args.pretrained else 1e-4

    all_pairs = []
    for root in args.data_root:
        pairs = discover_pairs(root)
        print(f"Found {len(pairs)} pairs in {root}")
        all_pairs.extend(pairs)

    train_pairs, val_pairs = split_pairs(all_pairs, args.val_split, args.seed)
    print(f"Split: train={len(train_pairs)} validation={len(val_pairs)}")
    write_pairs_csv(args.out.with_suffix(".pairs.csv"), train_pairs, val_pairs)

    print("Loading training tiles")
    train_x, train_y = load_arrays(train_pairs)
    print("Loading validation tiles")
    val_x, val_y = load_arrays(val_pairs)

    if args.pretrained:
        print(f"Loading pretrained model: {args.pretrained}")
        model = load_model(
            str(args.pretrained),
            compile=False,
            custom_objects={
                "dice_coef_keras": dice_coef_keras,
                "dice_keras": dice_coef_keras,
            },
        )
    else:
        print("Building U-Net mini")
        model = build_unet_mini()

    print(f"Training for {args.epochs} epochs")
    history = fit_model(model, train_x, train_y, val_x, val_y, args)

    save_history(history, args.out.with_suffix(".history.json"))
    save_metrics_plot(history, args.out.with_suffix(".metrics.png"))
    print(f"Best model checkpoint written to: {args.out}")


if __name__ == "__main__":
    main(parse_args())
