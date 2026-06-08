"""
Train a checklist-batch all-species graph bridge baseline.

Unlike the sampled-edge link baseline, this objective scores all modeled species
for each checklist in a mini-batch and trains against the full checklist x
species label matrix. That makes the training target match the all-pairs
evaluation target used by the tabular baselines.

Run from the project root:

    python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ebird_graph_link_baseline import LinkBaseline
from ebird_joint_tabular_baseline import SEED, auc_roc, average_precision
from evaluate_ebird_graph_all_pairs import calibration_table, build_summary


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an all-species-per-checklist graph bridge baseline."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to graph-dir/all_species_link_baselines.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional suffix for output files, for example hybrid_h128_l2_z128. "
            "Defaults to the architecture name."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=["pair-mlp", "factorized", "hybrid"],
        default="hybrid",
        help=(
            "Bridge architecture. pair-mlp matches the earlier pairwise "
            "species-embedding MLP; factorized uses checklist/species dot "
            "products; hybrid adds a direct multi-species checklist head. "
            "Defaults to hybrid."
        ),
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Latent dimension for factorized/hybrid architectures. Defaults to 64.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Species embedding dimension for pair-mlp. Defaults to 32.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden layer width. Defaults to 64.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=1,
        help="Hidden layers after checklist/species concatenation. Defaults to 1.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Dropout probability. Defaults to 0.10.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs. Defaults to 10.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Checklist mini-batch size. Defaults to 2,048.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate. Defaults to 1e-3.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. Defaults to 1e-4.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Predicted-probability bins for calibration. Defaults to 10.",
    )
    parser.add_argument(
        "--feature-augmentation",
        choices=[
            "none",
            "locality-spatial",
            "locality-spatial-scalars",
            "spatial-neighbor",
            "spatial-neighbor-scalars",
        ],
        default="none",
        help=(
            "Optional train-only relational feature augmentation. "
            "locality-spatial adds locality/grid counts and species prior logits; "
            "locality-spatial-scalars adds only the scalar locality/grid features; "
            "spatial-neighbor variants use distance-weighted train grid cells. "
            "Defaults to none."
        ),
    )
    parser.add_argument(
        "--spatial-grid-size-m",
        type=float,
        default=25_000.0,
        help="Grid cell size for spatial aggregate features. Defaults to 25,000 m.",
    )
    parser.add_argument(
        "--prior-smoothing",
        type=float,
        default=20.0,
        help="Empirical-Bayes smoothing strength for species priors. Defaults to 20.",
    )
    parser.add_argument(
        "--prior-logit-weight",
        type=float,
        default=0.0,
        help=(
            "Initial learnable weight for locality/spatial prior logits. "
            "Defaults to 0 so the model must learn whether the prior is useful."
        ),
    )
    parser.add_argument(
        "--spatial-neighbor-radius-m",
        type=float,
        default=75_000.0,
        help="Maximum train grid-cell centroid distance for spatial-neighbor features.",
    )
    parser.add_argument(
        "--spatial-neighbor-decay-m",
        type=float,
        default=50_000.0,
        help="Exponential distance-decay scale for spatial-neighbor features.",
    )
    parser.add_argument(
        "--spatial-neighbor-min-cells",
        type=int,
        default=3,
        help=(
            "Minimum nearby train grid cells required before using spatial-neighbor "
            "rates. Rows below this threshold shrink fully to global prevalence."
        ),
    )
    parser.add_argument(
        "--spatial-neighbor-batch-size",
        type=int,
        default=8192,
        help="Rows per batch when building spatial-neighbor features.",
    )
    parser.add_argument(
        "--spatial-residual",
        choices=["none", "rbf"],
        default="none",
        help=(
            "Optional additive smooth spatial residual head. rbf adds species-specific "
            "linear weights on fixed radial-basis spatial features. Defaults to none."
        ),
    )
    parser.add_argument(
        "--spatial-residual-grid-per-dim",
        type=int,
        default=12,
        help="RBF center grid size per spatial dimension. Defaults to 12.",
    )
    parser.add_argument(
        "--spatial-residual-length-scale-m",
        type=float,
        default=100_000.0,
        help="RBF spatial length scale in analysis CRS units. Defaults to 100,000 m.",
    )
    parser.add_argument(
        "--max-train-checklists",
        type=int,
        default=None,
        help="Optional train checklist cap for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-checklists",
        type=int,
        default=None,
        help="Optional test checklist cap for smoke tests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed. Defaults to 19.",
    )
    return parser.parse_args()


def load_split_checklists(
    graph_dir: Path,
    split: str,
    max_checklists: int | None,
    seed: int,
) -> np.ndarray:
    mask_column = f"{split}_mask"
    checklists = pd.read_parquet(
        graph_dir / "checklist_nodes.parquet",
        columns=["checklist_index", mask_column],
    )
    checklist_indices = (
        checklists.loc[checklists[mask_column], "checklist_index"]
        .to_numpy(dtype=np.int64)
    )
    if max_checklists is not None and len(checklist_indices) > max_checklists:
        rng = np.random.default_rng(seed)
        checklist_indices = np.sort(
            rng.choice(checklist_indices, size=max_checklists, replace=False)
        )
    return checklist_indices


def build_label_matrix(
    graph_dir: Path,
    checklist_indices: np.ndarray,
    species_count: int,
    split: str,
) -> np.ndarray:
    labels = np.zeros((len(checklist_indices), species_count), dtype=np.float32)
    positive_edges = pd.read_parquet(
        graph_dir / "positive_edges.parquet",
        columns=["checklist_index", "species_index", "split"],
    )
    positive_edges = positive_edges[positive_edges["split"] == split]
    checklist_lookup = np.full(int(checklist_indices.max()) + 1, -1, dtype=np.int64)
    checklist_lookup[checklist_indices] = np.arange(len(checklist_indices), dtype=np.int64)
    edge_checklists = positive_edges["checklist_index"].to_numpy(dtype=np.int64)
    edge_species = positive_edges["species_index"].to_numpy(dtype=np.int64)
    in_range = edge_checklists < len(checklist_lookup)
    edge_checklists = edge_checklists[in_range]
    edge_species = edge_species[in_range]
    positions = checklist_lookup[edge_checklists]
    keep = positions >= 0
    labels[positions[keep], edge_species[keep]] = 1.0
    return labels


def logit_array(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def group_codes_from_train(train_keys: pd.Series, all_keys: pd.Series) -> np.ndarray:
    train_codes, uniques = pd.factorize(train_keys.astype(str), sort=False)
    mapping = pd.Series(np.arange(len(uniques), dtype=np.int64), index=uniques)
    all_codes = all_keys.astype(str).map(mapping).fillna(-1).to_numpy(dtype=np.int64)
    return all_codes


def build_group_stats(
    train_checklists: np.ndarray,
    train_labels: np.ndarray,
    all_group_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_group_codes = all_group_codes[train_checklists]
    group_count = int(train_group_codes.max()) + 1 if len(train_group_codes) else 0
    counts = np.bincount(train_group_codes, minlength=group_count).astype(np.float32)
    positives = np.zeros((group_count, train_labels.shape[1]), dtype=np.float32)
    np.add.at(positives, train_group_codes, train_labels)
    return counts, positives


def smoothed_group_rates(
    checklist_indices: np.ndarray,
    labels: np.ndarray | None,
    all_group_codes: np.ndarray,
    counts: np.ndarray,
    positives: np.ndarray,
    global_prevalence: np.ndarray,
    smoothing: float,
    leave_one_out: bool,
) -> tuple[np.ndarray, np.ndarray]:
    species_count = len(global_prevalence)
    rates = np.repeat(global_prevalence[None, :], len(checklist_indices), axis=0)
    log_counts = np.zeros(len(checklist_indices), dtype=np.float32)
    group_codes = all_group_codes[checklist_indices]
    valid = (group_codes >= 0) & (group_codes < len(counts))
    if valid.any():
        valid_codes = group_codes[valid]
        valid_counts = counts[valid_codes].copy()
        valid_positives = positives[valid_codes].copy()
        if leave_one_out:
            if labels is None:
                raise ValueError("labels are required for leave-one-out group rates.")
            valid_counts = np.maximum(valid_counts - 1.0, 0.0)
            valid_positives = np.maximum(
                valid_positives - labels[valid],
                0.0,
            )
        rates[valid] = (
            valid_positives + smoothing * global_prevalence[None, :]
        ) / (valid_counts[:, None] + smoothing)
        log_counts[valid] = np.log1p(valid_counts).astype(np.float32)
    return rates.astype(np.float32), log_counts


def build_locality_spatial_augmentation(
    graph_dir: Path,
    base_features: np.ndarray,
    train_checklists: np.ndarray,
    test_checklists: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    nodes = pd.read_parquet(
        graph_dir / "checklist_nodes.parquet",
        columns=["checklist_index", "locality_id", "x", "y"],
    ).sort_values("checklist_index")
    if not np.array_equal(
        nodes["checklist_index"].to_numpy(dtype=np.int64),
        np.arange(len(nodes), dtype=np.int64),
    ):
        raise ValueError("checklist_nodes.parquet is not ordered by checklist_index.")

    locality_codes = group_codes_from_train(
        nodes.loc[train_checklists, "locality_id"],
        nodes["locality_id"],
    )
    cell_x = np.floor(nodes["x"].to_numpy() / args.spatial_grid_size_m).astype(np.int64)
    cell_y = np.floor(nodes["y"].to_numpy() / args.spatial_grid_size_m).astype(np.int64)
    cell_keys = pd.Series(cell_x.astype(str) + "_" + cell_y.astype(str))
    cell_codes = group_codes_from_train(cell_keys.iloc[train_checklists], cell_keys)

    global_prevalence = train_labels.mean(axis=0).astype(np.float32)
    locality_counts, locality_positives = build_group_stats(
        train_checklists, train_labels, locality_codes
    )
    cell_counts, cell_positives = build_group_stats(
        train_checklists, train_labels, cell_codes
    )

    train_locality_rates, train_locality_log_count = smoothed_group_rates(
        train_checklists,
        train_labels,
        locality_codes,
        locality_counts,
        locality_positives,
        global_prevalence,
        args.prior_smoothing,
        leave_one_out=True,
    )
    test_locality_rates, test_locality_log_count = smoothed_group_rates(
        test_checklists,
        test_labels,
        locality_codes,
        locality_counts,
        locality_positives,
        global_prevalence,
        args.prior_smoothing,
        leave_one_out=False,
    )
    train_cell_rates, train_cell_log_count = smoothed_group_rates(
        train_checklists,
        train_labels,
        cell_codes,
        cell_counts,
        cell_positives,
        global_prevalence,
        args.prior_smoothing,
        leave_one_out=True,
    )
    test_cell_rates, test_cell_log_count = smoothed_group_rates(
        test_checklists,
        test_labels,
        cell_codes,
        cell_counts,
        cell_positives,
        global_prevalence,
        args.prior_smoothing,
        leave_one_out=False,
    )

    train_prior_rates = 0.5 * (train_locality_rates + train_cell_rates)
    test_prior_rates = 0.5 * (test_locality_rates + test_cell_rates)
    train_prior_logits = logit_array(train_prior_rates)
    test_prior_logits = logit_array(test_prior_rates)

    train_aug = np.column_stack(
        [
            train_locality_log_count,
            train_cell_log_count,
            train_locality_rates.mean(axis=1),
            train_cell_rates.mean(axis=1),
        ]
    ).astype(np.float32)
    test_aug = np.column_stack(
        [
            test_locality_log_count,
            test_cell_log_count,
            test_locality_rates.mean(axis=1),
            test_cell_rates.mean(axis=1),
        ]
    ).astype(np.float32)
    mean = train_aug.mean(axis=0)
    std = train_aug.std(axis=0)
    std[std == 0] = 1.0
    train_aug = (train_aug - mean) / std
    test_aug = (test_aug - mean) / std

    full_aug = np.zeros((len(base_features), train_aug.shape[1]), dtype=np.float32)
    full_aug[train_checklists] = train_aug
    full_aug[test_checklists] = test_aug
    augmented_features = np.hstack([base_features, full_aug]).astype(np.float32)
    feature_names = [
        "locality_train_checklists_log1p",
        "spatial_cell_train_checklists_log1p",
        "locality_train_species_rate_mean",
        "spatial_cell_train_species_rate_mean",
    ]
    return augmented_features, train_prior_logits, test_prior_logits, feature_names


def build_train_cell_lookup(
    nodes: pd.DataFrame,
    train_checklists: np.ndarray,
    grid_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    cell_x = np.floor(nodes["x"].to_numpy() / grid_size_m).astype(np.int64)
    cell_y = np.floor(nodes["y"].to_numpy() / grid_size_m).astype(np.int64)
    cell_keys = pd.Series(cell_x.astype(str) + "_" + cell_y.astype(str))
    train_codes, uniques = pd.factorize(cell_keys.iloc[train_checklists], sort=False)
    mapping = pd.Series(np.arange(len(uniques), dtype=np.int64), index=uniques)
    all_codes = cell_keys.map(mapping).fillna(-1).to_numpy(dtype=np.int64)
    return train_codes.astype(np.int64), all_codes


def build_spatial_neighbor_rates(
    nodes_xy: np.ndarray,
    checklist_indices: np.ndarray,
    labels: np.ndarray | None,
    all_cell_codes: np.ndarray,
    cell_centroids: np.ndarray,
    cell_counts: np.ndarray,
    cell_positives: np.ndarray,
    global_prevalence: np.ndarray,
    args: argparse.Namespace,
    leave_one_out: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rates = np.repeat(global_prevalence[None, :], len(checklist_indices), axis=0)
    scalar_rows = np.zeros((len(checklist_indices), 4), dtype=np.float32)
    radius = float(args.spatial_neighbor_radius_m)
    decay = max(float(args.spatial_neighbor_decay_m), 1.0)
    batch_size = int(args.spatial_neighbor_batch_size)

    for start in range(0, len(checklist_indices), batch_size):
        stop = min(start + batch_size, len(checklist_indices))
        batch_indices = checklist_indices[start:stop]
        batch_xy = nodes_xy[batch_indices]
        dx = batch_xy[:, None, 0] - cell_centroids[None, :, 0]
        dy = batch_xy[:, None, 1] - cell_centroids[None, :, 1]
        distances = np.sqrt(dx * dx + dy * dy)
        neighbor_mask = distances <= radius
        neighbor_cells = neighbor_mask.sum(axis=1).astype(np.float32)
        weights = np.exp(-distances / decay).astype(np.float32)
        weights[~neighbor_mask] = 0.0
        weights[neighbor_cells < args.spatial_neighbor_min_cells] = 0.0

        weighted_counts = weights @ cell_counts
        weighted_positives = weights @ cell_positives
        weighted_distances = (weights * distances.astype(np.float32)).sum(axis=1)
        weight_sum = weights.sum(axis=1)
        mean_distance = np.divide(
            weighted_distances,
            weight_sum,
            out=np.zeros_like(weighted_distances),
            where=weight_sum > 0,
        )

        if leave_one_out:
            if labels is None:
                raise ValueError("labels are required for leave-one-out neighbor rates.")
            own_codes = all_cell_codes[batch_indices]
            valid_own = (own_codes >= 0) & (own_codes < len(cell_counts))
            if valid_own.any():
                rows = np.flatnonzero(valid_own)
                own_weights = weights[rows, own_codes[valid_own]]
                weighted_counts[rows] = np.maximum(
                    weighted_counts[rows] - own_weights,
                    0.0,
                )
                weighted_positives[rows] = np.maximum(
                    weighted_positives[rows] - own_weights[:, None] * labels[start:stop][rows],
                    0.0,
                )

        batch_rates = (
            weighted_positives + args.prior_smoothing * global_prevalence[None, :]
        ) / (weighted_counts[:, None] + args.prior_smoothing)
        rates[start:stop] = batch_rates.astype(np.float32)
        scalar_rows[start:stop, 0] = np.log1p(weighted_counts).astype(np.float32)
        scalar_rows[start:stop, 1] = np.log1p(neighbor_cells).astype(np.float32)
        scalar_rows[start:stop, 2] = batch_rates.mean(axis=1).astype(np.float32)
        scalar_rows[start:stop, 3] = (mean_distance / max(radius, 1.0)).astype(np.float32)

    return rates.astype(np.float32), scalar_rows


def build_spatial_neighbor_augmentation(
    graph_dir: Path,
    base_features: np.ndarray,
    train_checklists: np.ndarray,
    test_checklists: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    nodes = pd.read_parquet(
        graph_dir / "checklist_nodes.parquet",
        columns=["checklist_index", "x", "y"],
    ).sort_values("checklist_index")
    if not np.array_equal(
        nodes["checklist_index"].to_numpy(dtype=np.int64),
        np.arange(len(nodes), dtype=np.int64),
    ):
        raise ValueError("checklist_nodes.parquet is not ordered by checklist_index.")

    nodes_xy = nodes[["x", "y"]].to_numpy(dtype=np.float32)
    train_cell_codes, all_cell_codes = build_train_cell_lookup(
        nodes, train_checklists, args.spatial_grid_size_m
    )
    cell_count = int(train_cell_codes.max()) + 1 if len(train_cell_codes) else 0
    cell_counts = np.bincount(train_cell_codes, minlength=cell_count).astype(np.float32)
    cell_positives = np.zeros((cell_count, train_labels.shape[1]), dtype=np.float32)
    np.add.at(cell_positives, train_cell_codes, train_labels)

    cell_x_sum = np.zeros(cell_count, dtype=np.float64)
    cell_y_sum = np.zeros(cell_count, dtype=np.float64)
    np.add.at(cell_x_sum, train_cell_codes, nodes_xy[train_checklists, 0])
    np.add.at(cell_y_sum, train_cell_codes, nodes_xy[train_checklists, 1])
    cell_centroids = np.column_stack(
        [
            cell_x_sum / np.maximum(cell_counts, 1.0),
            cell_y_sum / np.maximum(cell_counts, 1.0),
        ]
    ).astype(np.float32)

    global_prevalence = train_labels.mean(axis=0).astype(np.float32)
    train_rates, train_aug = build_spatial_neighbor_rates(
        nodes_xy,
        train_checklists,
        train_labels,
        all_cell_codes,
        cell_centroids,
        cell_counts,
        cell_positives,
        global_prevalence,
        args,
        leave_one_out=True,
    )
    test_rates, test_aug = build_spatial_neighbor_rates(
        nodes_xy,
        test_checklists,
        test_labels,
        all_cell_codes,
        cell_centroids,
        cell_counts,
        cell_positives,
        global_prevalence,
        args,
        leave_one_out=False,
    )

    mean = train_aug.mean(axis=0)
    std = train_aug.std(axis=0)
    std[std == 0] = 1.0
    train_aug = (train_aug - mean) / std
    test_aug = (test_aug - mean) / std
    full_aug = np.zeros((len(base_features), train_aug.shape[1]), dtype=np.float32)
    full_aug[train_checklists] = train_aug
    full_aug[test_checklists] = test_aug

    feature_names = [
        "spatial_neighbor_train_checklists_log1p",
        "spatial_neighbor_train_cells_log1p",
        "spatial_neighbor_species_rate_mean",
        "spatial_neighbor_mean_distance_ratio",
    ]
    return (
        np.hstack([base_features, full_aug]).astype(np.float32),
        logit_array(train_rates),
        logit_array(test_rates),
        feature_names,
    )


def build_spatial_residual_features(
    graph_dir: Path,
    train_checklists: np.ndarray,
    args: argparse.Namespace,
) -> tuple[torch.Tensor | None, list[str]]:
    if args.spatial_residual == "none":
        return None, []
    nodes = pd.read_parquet(
        graph_dir / "checklist_nodes.parquet",
        columns=["checklist_index", "x", "y"],
    ).sort_values("checklist_index")
    if not np.array_equal(
        nodes["checklist_index"].to_numpy(dtype=np.int64),
        np.arange(len(nodes), dtype=np.int64),
    ):
        raise ValueError("checklist_nodes.parquet is not ordered by checklist_index.")
    xy = nodes[["x", "y"]].to_numpy(dtype=np.float32)
    train_xy = xy[train_checklists]
    grid_count = int(args.spatial_residual_grid_per_dim)
    if grid_count < 2:
        raise ValueError("--spatial-residual-grid-per-dim must be at least 2.")
    x_centers = np.linspace(train_xy[:, 0].min(), train_xy[:, 0].max(), grid_count)
    y_centers = np.linspace(train_xy[:, 1].min(), train_xy[:, 1].max(), grid_count)
    centers = np.array(
        [(x, y) for x in x_centers for y in y_centers],
        dtype=np.float32,
    )
    length_scale = max(float(args.spatial_residual_length_scale_m), 1.0)
    dx = xy[:, None, 0] - centers[None, :, 0]
    dy = xy[:, None, 1] - centers[None, :, 1]
    features = np.exp(-0.5 * (dx * dx + dy * dy) / (length_scale * length_scale))
    features = features.astype(np.float32)
    mean = features[train_checklists].mean(axis=0)
    std = features[train_checklists].std(axis=0)
    std[std == 0] = 1.0
    features = ((features - mean) / std).astype(np.float32)
    feature_names = [
        f"spatial_residual_rbf_{i:03d}" for i in range(features.shape[1])
    ]
    return torch.from_numpy(features), feature_names


class FactorizedAllSpeciesModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        species_count: int,
        hidden_dim: int,
        hidden_layers: int,
        latent_dim: int,
        dropout: float,
        direct_head: bool,
    ):
        super().__init__()
        if hidden_layers <= 0:
            raise ValueError("--hidden-layers must be greater than zero.")
        layers: list[nn.Module] = []
        current_dim = feature_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, latent_dim))
        layers.append(nn.ReLU())
        self.checklist_encoder = nn.Sequential(*layers)
        self.species_embedding = nn.Embedding(species_count, latent_dim)
        self.species_bias = nn.Parameter(torch.zeros(species_count))
        self.checklist_bias = nn.Linear(latent_dim, 1)
        self.direct_head = nn.Linear(latent_dim, species_count) if direct_head else None
        self.scale = float(np.sqrt(latent_dim))

    def forward_all_species(self, checklist_features: torch.Tensor) -> torch.Tensor:
        latent = self.checklist_encoder(checklist_features)
        logits = latent @ self.species_embedding.weight.T / self.scale
        logits = logits + self.species_bias + self.checklist_bias(latent)
        if self.direct_head is not None:
            logits = logits + self.direct_head(latent)
        return logits


def build_model(
    feature_dim: int,
    species_count: int,
    args: argparse.Namespace,
) -> nn.Module:
    if args.architecture == "pair-mlp":
        return LinkBaseline(
            feature_dim=feature_dim,
            species_count=species_count,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
        )
    return FactorizedAllSpeciesModel(
        feature_dim=feature_dim,
        species_count=species_count,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        direct_head=args.architecture == "hybrid",
    )


def logits_all_species(
    model: nn.Module,
    checklist_features: torch.Tensor,
    species_count: int,
    prior_logits: torch.Tensor | None = None,
    spatial_residual_features: torch.Tensor | None = None,
) -> torch.Tensor:
    if hasattr(model, "forward_all_species"):
        logits = model.forward_all_species(checklist_features)
    else:
        checklist_count = checklist_features.shape[0]
        species_index = torch.arange(species_count, dtype=torch.int64)
        species_index = species_index.repeat(checklist_count)
        checklist_expanded = (
            checklist_features[:, None, :]
            .expand(checklist_count, species_count, checklist_features.shape[1])
            .reshape(checklist_count * species_count, checklist_features.shape[1])
        )
        logits = model(checklist_expanded, species_index)
        logits = logits.reshape(checklist_count, species_count)
    if prior_logits is not None:
        logits = logits + model.prior_logit_weight * prior_logits
    if spatial_residual_features is not None:
        logits = logits + model.spatial_residual_head(spatial_residual_features)
    return logits


def evaluate_all_pairs(
    model: LinkBaseline,
    features: torch.Tensor,
    checklist_indices: np.ndarray,
    labels: np.ndarray,
    prior_logits: np.ndarray | None,
    spatial_residual_features: torch.Tensor | None,
    species: pd.DataFrame,
    batch_size: int,
    calibration_bins: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    model.eval()
    score_parts = []
    with torch.no_grad():
        for start in range(0, len(checklist_indices), batch_size):
            batch = checklist_indices[start : start + batch_size]
            checklist_tensor = torch.from_numpy(batch.astype(np.int64))
            logits = logits_all_species(
                model,
                features[checklist_tensor],
                species_count=len(species),
                prior_logits=(
                    torch.from_numpy(prior_logits[start : start + len(batch)])
                    if prior_logits is not None
                    else None
                ),
                spatial_residual_features=(
                    spatial_residual_features[checklist_tensor]
                    if spatial_residual_features is not None
                    else None
                ),
            )
            score_parts.append(torch.sigmoid(logits).cpu().numpy())
    scores = np.vstack(score_parts).astype(np.float32)

    species_rows = []
    for row in species.itertuples(index=False):
        species_index = int(row.species_index)
        species_labels = labels[:, species_index]
        species_scores = scores[:, species_index]
        species_rows.append(
            {
                "species_index": species_index,
                "species_key": row.species_key,
                "common_name": row.common_name,
                "scientific_name": row.scientific_name,
                "pairs": int(len(species_labels)),
                "positives": int(species_labels.sum()),
                "negatives": int(len(species_labels) - species_labels.sum()),
                "observed_rate": float(species_labels.mean()),
                "mean_predicted": float(species_scores.mean()),
                "calibration_error": float(
                    abs(species_scores.mean() - species_labels.mean())
                ),
                "auroc": auc_roc(species_labels, species_scores),
                "auprc": average_precision(species_labels, species_scores),
            }
        )

    score_np = scores.T.reshape(-1)
    label_np = labels.T.reshape(-1)
    species_metrics = pd.DataFrame(species_rows)
    calibration = calibration_table(score_np, label_np, calibration_bins)
    summary = build_summary(
        split="test",
        checklist_count=len(checklist_indices),
        species_count=len(species),
        labels=label_np,
        scores=score_np,
        species_metrics=species_metrics,
        calibration=calibration,
    )
    return summary, species_metrics, calibration


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else graph_dir / "all_species_link_baselines"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    features_np = np.load(graph_dir / "checklist_features.npy").astype(np.float32)
    features = torch.from_numpy(features_np)
    species = pd.read_csv(graph_dir / "species_nodes.csv")
    species_count = int(metadata["counts"]["species"])

    train_checklists = load_split_checklists(
        graph_dir, "train", args.max_train_checklists, args.seed
    )
    test_checklists = load_split_checklists(
        graph_dir, "test", args.max_eval_checklists, args.seed + 1
    )
    train_labels = build_label_matrix(graph_dir, train_checklists, species_count, "train")
    test_labels = build_label_matrix(graph_dir, test_checklists, species_count, "test")
    train_prior_logits = None
    test_prior_logits = None
    augmented_feature_names: list[str] = []
    spatial_residual_features, spatial_residual_feature_names = (
        build_spatial_residual_features(graph_dir, train_checklists, args)
    )
    if args.feature_augmentation in {"locality-spatial", "locality-spatial-scalars"}:
        features_np, train_prior_logits, test_prior_logits, augmented_feature_names = (
            build_locality_spatial_augmentation(
                graph_dir,
                features_np,
                train_checklists,
                test_checklists,
                train_labels,
                test_labels,
                args,
            )
        )
        if args.feature_augmentation == "locality-spatial-scalars":
            train_prior_logits = None
            test_prior_logits = None
        features = torch.from_numpy(features_np)
    elif args.feature_augmentation in {"spatial-neighbor", "spatial-neighbor-scalars"}:
        features_np, train_prior_logits, test_prior_logits, augmented_feature_names = (
            build_spatial_neighbor_augmentation(
                graph_dir,
                features_np,
                train_checklists,
                test_checklists,
                train_labels,
                test_labels,
                args,
            )
        )
        if args.feature_augmentation == "spatial-neighbor-scalars":
            train_prior_logits = None
            test_prior_logits = None
        features = torch.from_numpy(features_np)

    model = build_model(
        feature_dim=features_np.shape[1],
        species_count=species_count,
        args=args,
    )
    if args.feature_augmentation in {"locality-spatial", "spatial-neighbor"}:
        model.prior_logit_weight = nn.Parameter(
            torch.tensor(float(args.prior_logit_weight), dtype=torch.float32)
        )
    if spatial_residual_features is not None:
        model.spatial_residual_head = nn.Linear(
            spatial_residual_features.shape[1],
            species_count,
            bias=False,
        )
        nn.init.zeros_(model.spatial_residual_head.weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    train_dataset = TensorDataset(
        torch.from_numpy(train_checklists.astype(np.int64)),
        torch.from_numpy(train_labels),
        *(
            [torch.from_numpy(train_prior_logits.astype(np.float32))]
            if train_prior_logits is not None
            else []
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            checklist_index = batch[0]
            y = batch[1]
            batch_prior_logits = batch[2] if len(batch) > 2 else None
            optimizer.zero_grad()
            logits = logits_all_species(
                model,
                features[checklist_index],
                species_count=species_count,
                prior_logits=batch_prior_logits,
                spatial_residual_features=(
                    spatial_residual_features[checklist_index]
                    if spatial_residual_features is not None
                    else None
                ),
            )
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_bce": train_loss})
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(f"epoch {epoch:>3}: train BCE={train_loss:.5f}")

    summary, species_metrics, calibration = evaluate_all_pairs(
        model,
        features,
        test_checklists,
        test_labels,
        test_prior_logits,
        spatial_residual_features,
        species,
        args.batch_size,
        args.calibration_bins,
    )
    summary["model"] = {
        "architecture": args.architecture,
        "embedding_dim": args.embedding_dim,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "feature_augmentation": args.feature_augmentation,
        "augmented_feature_names": augmented_feature_names,
        "spatial_grid_size_m": args.spatial_grid_size_m,
        "prior_smoothing": args.prior_smoothing,
        "spatial_neighbor_radius_m": args.spatial_neighbor_radius_m,
        "spatial_neighbor_decay_m": args.spatial_neighbor_decay_m,
        "spatial_neighbor_min_cells": args.spatial_neighbor_min_cells,
        "spatial_residual": args.spatial_residual,
        "spatial_residual_grid_per_dim": args.spatial_residual_grid_per_dim,
        "spatial_residual_length_scale_m": args.spatial_residual_length_scale_m,
        "spatial_residual_feature_count": len(spatial_residual_feature_names),
    }
    if args.feature_augmentation in {"locality-spatial", "spatial-neighbor"}:
        summary["model"]["prior_logit_weight"] = float(
            model.prior_logit_weight.detach()
        )
    summary["train"] = {
        "checklists": int(len(train_checklists)),
        "pairs": int(train_labels.size),
        "positives": int(train_labels.sum()),
        "observed_rate": float(train_labels.mean()),
    }

    run_name = args.run_name or args.architecture.replace("-", "_")
    prefix = f"all_species_link_{run_name}"
    pd.DataFrame(history).to_csv(output_dir / f"{prefix}_history.csv", index=False)
    species_metrics.to_csv(output_dir / f"{prefix}_test_species_metrics.csv", index=False)
    calibration.to_csv(output_dir / f"{prefix}_test_calibration.csv", index=False)
    (output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), output_dir / f"{prefix}_model.pt")

    print("\nAll-species checklist-batch graph metrics:")
    print(
        f"micro AUROC={summary['auroc']:.4f}, micro AUPRC={summary['auprc']:.4f}"
    )
    print(
        f"macro AUROC={summary['species_macro_auroc']:.4f}, "
        f"macro AUPRC={summary['species_macro_auprc']:.4f}"
    )
    print(
        f"ECE={summary['probability_bin_ece']:.4f}, "
        f"max bin error={summary['probability_bin_max_error']:.4f}, "
        f"species calibration MAE={summary['species_calibration_mae']:.4f}"
    )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
