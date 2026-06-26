#!/usr/bin/env python3
"""
Search candidate CDP k-means cluster numbers and generate sanity-check plots.

This is the exploratory companion to train_cdp_clusters.py. It uses the same
ResNet18 feature extraction and collagen-fraction label sorting, then loops
over a user-specified K range and writes per-K metrics plus example tile panels.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from train_cdp_clusters import (
    FeatureSet,
    apply_mapping,
    choose_device,
    collagen_sorted_label_mapping,
    extract_features,
)


@dataclass(frozen=True)
class ClusterRun:
    k: int
    model: object
    raw_labels: np.ndarray
    sorted_labels: np.ndarray
    mapping: dict[int, int]
    silhouette: float
    silhouette_n: int
    inertia: float | None


def load_feature_cache(path: Path) -> FeatureSet:
    data = np.load(path, allow_pickle=False)
    return FeatureSet(
        features=data["features"],
        files=[Path(file_path) for file_path in data["files"]],
        collagen_fraction=data["collagen_fraction"],
    )


def save_feature_cache(path: Path, feature_set: FeatureSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=feature_set.features,
        files=np.asarray([str(path) for path in feature_set.files]),
        collagen_fraction=feature_set.collagen_fraction,
    )


def get_feature_set(args: argparse.Namespace) -> FeatureSet:
    if args.feature_cache and args.feature_cache.exists():
        print(f"Loading feature cache: {args.feature_cache}")
        return load_feature_cache(args.feature_cache)

    if args.tile_dir is None:
        raise SystemExit("--tile-dir is required when --feature-cache does not already exist")

    device = choose_device(args.device)
    print(f"Using device: {device}")
    feature_set = extract_features(args.tile_dir, device, args.fraction, args.seed)

    if args.feature_cache:
        print(f"Writing feature cache: {args.feature_cache}")
        save_feature_cache(args.feature_cache, feature_set)

    return feature_set


def make_clusters(
    features: np.ndarray,
    k: int,
    algorithm: str,
    seed: int,
    batch_size: int,
) -> object:
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(features)
    if algorithm == "minibatch_kmeans":
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            n_init=10,
            batch_size=batch_size,
        ).fit(features)
    if algorithm == "agglomerative_ward":
        return AgglomerativeClustering(
            linkage="ward",
            n_clusters=k,
            compute_distances=True,
        ).fit(features)
    raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def sample_indices(n_items: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size == 0 or n_items <= sample_size:
        return np.arange(n_items)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_items, size=sample_size, replace=False))


def compute_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> tuple[float, int]:
    indices = sample_indices(len(labels), sample_size, seed)
    sample_labels = labels[indices]
    if len(np.unique(sample_labels)) < 2:
        return math.nan, len(indices)
    return float(silhouette_score(features[indices], sample_labels)), len(indices)


def cluster_stats_rows(
    k: int,
    raw_labels: np.ndarray,
    sorted_labels: np.ndarray,
    mapping: dict[int, int],
    collagen_fraction: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    total = len(sorted_labels)
    reverse_mapping = {sorted_label: raw_label for raw_label, sorted_label in mapping.items()}

    for sorted_label in sorted(np.unique(sorted_labels)):
        indices = np.where(sorted_labels == sorted_label)[0]
        cluster_collagen = collagen_fraction[indices]
        rows.append(
            {
                "k": k,
                "cluster_label": int(sorted_label),
                "raw_cluster_label": int(reverse_mapping[int(sorted_label)]),
                "n_tiles": int(len(indices)),
                "fraction_tiles": float(len(indices) / total),
                "mean_collagen_fraction": float(np.mean(cluster_collagen)),
                "std_collagen_fraction": float(np.std(cluster_collagen)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_tile(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("L")
        return np.asarray(image)


def show_tile(ax, path: Path) -> None:
    image = read_tile(path)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_axis_off()


def plot_random_examples(
    out_path: Path,
    files: list[Path],
    sorted_labels: np.ndarray,
    stats_rows: list[dict[str, object]],
    n_examples: int,
    seed: int,
) -> None:
    cluster_labels = [int(row["cluster_label"]) for row in stats_rows]
    fig, axes = plt.subplots(
        nrows=len(cluster_labels),
        ncols=n_examples,
        figsize=(3 * n_examples, 2.8 * len(cluster_labels)),
        squeeze=False,
    )
    rng = random.Random(seed)

    for row_index, label in enumerate(cluster_labels):
        indices = np.where(sorted_labels == label)[0].tolist()
        chosen = rng.sample(indices, min(n_examples, len(indices)))
        stats = stats_rows[row_index]

        for col_index in range(n_examples):
            ax = axes[row_index, col_index]
            if col_index < len(chosen):
                show_tile(ax, files[chosen[col_index]])
            else:
                ax.set_axis_off()

        axes[row_index, 0].set_title(
            f"Cluster {label}\n"
            f"n={stats['n_tiles']}, collagen={stats['mean_collagen_fraction']:.3f}",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_centroid_examples(
    out_path: Path,
    files: list[Path],
    features: np.ndarray,
    raw_labels: np.ndarray,
    stats_rows: list[dict[str, object]],
    model: object,
    n_examples: int,
) -> bool:
    if not hasattr(model, "transform"):
        return False

    cluster_labels = [int(row["cluster_label"]) for row in stats_rows]
    fig, axes = plt.subplots(
        nrows=len(cluster_labels),
        ncols=n_examples,
        figsize=(3 * n_examples, 2.8 * len(cluster_labels)),
        squeeze=False,
    )

    for row_index, stats in enumerate(stats_rows):
        raw_label = int(stats["raw_cluster_label"])
        member_indices = np.where(raw_labels == raw_label)[0]
        distances = model.transform(features[member_indices])[:, raw_label]
        chosen = member_indices[np.argsort(distances)[:n_examples]]

        for col_index in range(n_examples):
            ax = axes[row_index, col_index]
            if col_index < len(chosen):
                show_tile(ax, files[int(chosen[col_index])])
            else:
                ax.set_axis_off()

        axes[row_index, 0].set_title(
            f"Cluster {cluster_labels[row_index]}\nnearest centroid",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return True


def copy_clustered_images(
    out_dir: Path,
    files: list[Path],
    sorted_labels: np.ndarray,
    k: int,
) -> None:
    for label in sorted(np.unique(sorted_labels)):
        label_dir = out_dir / f"k{k:02d}_cluster_{int(label)}"
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in np.where(sorted_labels == label)[0]:
            shutil.copy2(files[index], label_dir / files[index].name)


def plot_k_summary(out_path: Path, runs: list[ClusterRun]) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), squeeze=False)
    k_values = [run.k for run in runs]
    silhouettes = [run.silhouette for run in runs]
    inertias = [run.inertia if run.inertia is not None else math.nan for run in runs]

    axes[0, 0].plot(k_values, silhouettes, marker="o", linestyle=":")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Silhouette score")
    axes[0, 0].set_title("Cluster separation")

    axes[0, 1].plot(k_values, inertias, marker="o", linestyle=":")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Inertia")
    axes[0, 1].set_title("K-means inertia")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_property(
    out_path: Path,
    all_stats_rows: list[dict[str, object]],
    property_name: str,
    ylabel: str,
) -> None:
    k_values = sorted({int(row["k"]) for row in all_stats_rows})
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(k_values),
        figsize=(4 * len(k_values), 4),
        sharey=True,
        squeeze=False,
    )

    for axis, k in zip(axes[0], k_values):
        rows = [row for row in all_stats_rows if int(row["k"]) == k]
        x = [int(row["cluster_label"]) for row in rows]
        y = [float(row[property_name]) for row in rows]
        axis.bar(x, y)
        axis.set_title(f"k={k}")
        axis.set_xlabel("Cluster")
        axis.set_xticks(x)

    axes[0, 0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def reduce_for_embedding(
    features: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(features)
    if method == "umap":
        try:
            from umap import UMAP
        except ImportError as exc:
            raise SystemExit("UMAP plotting requires the optional umap-learn package") from exc
        return UMAP(n_components=2, random_state=seed).fit_transform(features)
    raise ValueError(f"Unknown embedding method: {method}")


def plot_embedding_and_silhouette(
    out_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
    method: str,
    sample_size: int,
    seed: int,
) -> None:
    indices = sample_indices(len(labels), sample_size, seed)
    sample_features = features[indices]
    sample_labels = labels[indices]
    if len(np.unique(sample_labels)) < 2:
        return

    reduced = reduce_for_embedding(sample_features, method, seed)
    sample_silhouettes = silhouette_samples(sample_features, sample_labels)
    silhouette_avg = float(np.mean(sample_silhouettes))
    cluster_labels = sorted(np.unique(sample_labels))
    cmap = plt.get_cmap("Spectral_r")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), squeeze=False)
    scatter_ax = axes[0, 0]
    silhouette_ax = axes[0, 1]

    for position, label in enumerate(cluster_labels):
        cluster_mask = sample_labels == label
        color = cmap(position / max(1, len(cluster_labels) - 1))
        scatter_ax.scatter(
            reduced[cluster_mask, 0],
            reduced[cluster_mask, 1],
            s=8,
            color=color,
            label=f"Cluster {int(label)}",
            alpha=0.75,
        )

    scatter_ax.set_title(f"{method.upper()} feature embedding, k={k}")
    scatter_ax.set_xlabel("Component 1")
    scatter_ax.set_ylabel("Component 2")
    scatter_ax.legend(markerscale=2, fontsize=8)

    y_lower = 10
    for position, label in enumerate(cluster_labels):
        values = np.sort(sample_silhouettes[sample_labels == label])
        y_upper = y_lower + len(values)
        color = cmap(position / max(1, len(cluster_labels) - 1))
        silhouette_ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        silhouette_ax.text(-0.05, y_lower + 0.5 * len(values), str(int(label)))
        y_lower = y_upper + 10

    silhouette_ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    silhouette_ax.set_title("Silhouette plot")
    silhouette_ax.set_xlabel("Silhouette coefficient")
    silhouette_ax.set_ylabel("Cluster label")
    silhouette_ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search candidate K values for CDP collagen-map clustering."
    )
    parser.add_argument("--tile-dir", type=Path, help="Collagen probability tile directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Optional .npz feature cache. Existing cache is loaded; missing cache is written.",
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10, help="Inclusive upper bound")
    parser.add_argument(
        "--algorithm",
        choices=["kmeans", "minibatch_kmeans", "agglomerative_ward"],
        default="kmeans",
    )
    parser.add_argument("--fraction", type=float, default=1.0, help="Optional fraction of tiles to use")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--examples-per-cluster", type=int, default=5)
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=5000,
        help="Number of tiles used for silhouette diagnostics. Use 0 for all tiles.",
    )
    parser.add_argument(
        "--embedding-method",
        choices=["none", "pca", "umap"],
        default="none",
        help="Optional per-K embedding/silhouette diagnostic plot.",
    )
    parser.add_argument("--no-random-examples", action="store_true")
    parser.add_argument("--no-centroid-examples", action="store_true")
    parser.add_argument("--copy-clustered-images", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.k_min > args.k_max:
        raise SystemExit("--k-min must be less than or equal to --k-max")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_set = get_feature_set(args)
    runs = []
    metrics_rows = []
    all_stats_rows = []

    for k in range(args.k_min, args.k_max + 1):
        print(f"Searching k={k}")
        model = make_clusters(
            feature_set.features,
            k,
            args.algorithm,
            args.seed,
            args.minibatch_size,
        )
        raw_labels = np.asarray(model.labels_)
        mapping = collagen_sorted_label_mapping(
            raw_labels,
            feature_set.collagen_fraction,
            offset=0,
        )
        sorted_labels = apply_mapping(raw_labels, mapping)
        silhouette, silhouette_n = compute_silhouette(
            feature_set.features,
            sorted_labels,
            args.silhouette_sample_size,
            args.seed + k,
        )
        inertia = float(model.inertia_) if hasattr(model, "inertia_") else None
        stats_rows = cluster_stats_rows(
            k,
            raw_labels,
            sorted_labels,
            mapping,
            feature_set.collagen_fraction,
        )

        k_dir = args.out_dir / f"k{k:02d}"
        k_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            k_dir / "cluster_stats.csv",
            stats_rows,
            [
                "k",
                "cluster_label",
                "raw_cluster_label",
                "n_tiles",
                "fraction_tiles",
                "mean_collagen_fraction",
                "std_collagen_fraction",
            ],
        )

        if not args.no_random_examples:
            plot_random_examples(
                k_dir / "random_examples.png",
                feature_set.files,
                sorted_labels,
                stats_rows,
                args.examples_per_cluster,
                args.seed + k,
            )

        if not args.no_centroid_examples:
            wrote_centroid_plot = plot_centroid_examples(
                k_dir / "centroid_examples.png",
                feature_set.files,
                feature_set.features,
                raw_labels,
                stats_rows,
                model,
                args.examples_per_cluster,
            )
            if not wrote_centroid_plot:
                print(f"Skipping centroid examples for {args.algorithm}; transform() is unavailable")

        if args.embedding_method != "none":
            plot_embedding_and_silhouette(
                k_dir / f"{args.embedding_method}_silhouette.png",
                feature_set.features,
                sorted_labels,
                k,
                args.embedding_method,
                args.silhouette_sample_size,
                args.seed + k,
            )

        if args.copy_clustered_images:
            copy_clustered_images(
                k_dir / "clustered_tiles",
                feature_set.files,
                sorted_labels,
                k,
            )

        metrics_rows.append(
            {
                "k": k,
                "algorithm": args.algorithm,
                "n_tiles": len(feature_set.files),
                "silhouette_score": silhouette,
                "silhouette_sample_size": silhouette_n,
                "inertia": "" if inertia is None else inertia,
            }
        )
        all_stats_rows.extend(stats_rows)
        runs.append(
            ClusterRun(
                k=k,
                model=model,
                raw_labels=raw_labels,
                sorted_labels=sorted_labels,
                mapping=mapping,
                silhouette=silhouette,
                silhouette_n=silhouette_n,
                inertia=inertia,
            )
        )

    write_csv(
        args.out_dir / "k_search_metrics.csv",
        metrics_rows,
        [
            "k",
            "algorithm",
            "n_tiles",
            "silhouette_score",
            "silhouette_sample_size",
            "inertia",
        ],
    )
    write_csv(
        args.out_dir / "k_search_cluster_stats.csv",
        all_stats_rows,
        [
            "k",
            "cluster_label",
            "raw_cluster_label",
            "n_tiles",
            "fraction_tiles",
            "mean_collagen_fraction",
            "std_collagen_fraction",
        ],
    )
    plot_k_summary(args.out_dir / "k_search_metrics.png", runs)
    plot_cluster_property(
        args.out_dir / "k_search_cluster_sizes.png",
        all_stats_rows,
        "fraction_tiles",
        "Fraction of tiles",
    )
    plot_cluster_property(
        args.out_dir / "k_search_mean_collagen_fraction.png",
        all_stats_rows,
        "mean_collagen_fraction",
        "Mean collagen fraction",
    )

    print(f"Wrote K-search outputs to: {args.out_dir}")


if __name__ == "__main__":
    main(parse_args())
