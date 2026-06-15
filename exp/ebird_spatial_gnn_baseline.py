"""
Train a conservative spatial-cell GNN baseline for the eBird graph dataset.

This model keeps the all-species checklist-batch objective from the strongest
bridge baseline, but adds message passing over a spatial grid-cell graph. The
GNN context is intentionally narrow: spatial cells exchange information with
neighboring spatial cells, then each checklist receives the embedding for its
cell before scoring all species.

Run from the project root:

    python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ebird_graph_all_species_baseline import build_label_matrix, load_split_checklists
from ebird_joint_tabular_baseline import SEED, auc_roc, average_precision
from evaluate_ebird_graph_all_pairs import build_summary, calibration_table


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial"
torch.sparse.check_sparse_tensor_invariants.disable()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a spatial-cell message-passing GNN eBird baseline."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to graph-dir/spatial_gnn_baselines.",
    )
    parser.add_argument(
        "--run-name",
        default="spatial_gcn_hybrid_h128_l2_z128",
        help="Output file suffix. Defaults to spatial_gcn_hybrid_h128_l2_z128.",
    )
    parser.add_argument(
        "--gnn-mode",
        choices=["concat", "residual", "gated"],
        default="concat",
        help=(
            "How spatial-cell messages enter the detector. concat reproduces the "
            "first GCN baseline; residual adds a zero-initialized species residual "
            "to a checklist-only hybrid; gated learns a checklist/species gate on "
            "that residual. Defaults to concat."
        ),
    )
    parser.add_argument(
        "--component-mode",
        choices=["joint", "separated", "shared"],
        default="joint",
        help=(
            "Prediction decomposition. joint uses the original shared "
            "checklist encoder; separated uses an ecology/suitability path plus "
            "an effort/bias path; shared uses a full-covariate shared trunk plus "
            "separate suitability and effort/bias heads. Defaults to joint."
        ),
    )
    parser.add_argument(
        "--spatial-grid-size-m",
        type=float,
        default=25_000.0,
        help="Grid cell size for spatial-cell nodes. Defaults to 25,000 m.",
    )
    parser.add_argument(
        "--cell-edge-mode",
        choices=["spatial", "environmental", "hybrid"],
        default="spatial",
        help=(
            "Spatial-cell graph edges: queen spatial adjacency, environmental "
            "nearest neighbors, or the union of both. Defaults to spatial."
        ),
    )
    parser.add_argument(
        "--environmental-neighbors",
        type=int,
        default=6,
        help=(
            "Nearest environmental neighbors per cell when --cell-edge-mode is "
            "environmental or hybrid. Defaults to 6."
        ),
    )
    parser.add_argument(
        "--species-edge-mode",
        choices=["none", "codetection"],
        default="none",
        help=(
            "Optional species graph for message passing over species embeddings. "
            "Defaults to none."
        ),
    )
    parser.add_argument(
        "--species-neighbors",
        type=int,
        default=10,
        help=(
            "Nearest co-detection neighbors per species when --species-edge-mode "
            "is codetection. Defaults to 10."
        ),
    )
    parser.add_argument(
        "--species-gcn-layers",
        type=int,
        default=0,
        help="Species embedding GCN layers. Defaults to 0.",
    )
    parser.add_argument(
        "--species-gcn-dropout",
        type=float,
        default=0.0,
        help="Dropout in species embedding GCN layers. Defaults to 0.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Checklist encoder hidden width. Defaults to 128.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=2,
        help="Checklist encoder hidden layers. Defaults to 2.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Checklist/species latent dimension. Defaults to 128.",
    )
    parser.add_argument(
        "--cell-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for spatial-cell GCN. Defaults to 64.",
    )
    parser.add_argument(
        "--cell-layers",
        type=int,
        default=2,
        help="Spatial-cell message-passing layers. Defaults to 2.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Dropout probability. Defaults to 0.10.",
    )
    parser.add_argument(
        "--gate-init-bias",
        type=float,
        default=-2.0,
        help="Initial bias for gated residual probabilities. Defaults to -2.0.",
    )
    parser.add_argument(
        "--species-residual-scale",
        choices=["none", "sigmoid", "softplus"],
        default="none",
        help=(
            "Optional species-specific residual scale for residual/gated modes. "
            "none leaves residual logits unchanged; sigmoid constrains each "
            "species scale to (0, 1); softplus constrains it positive but "
            "unbounded. Defaults to none."
        ),
    )
    parser.add_argument(
        "--species-residual-scale-init",
        type=float,
        default=0.25,
        help=(
            "Initial effective species residual scale for sigmoid/softplus "
            "scaling. Defaults to 0.25."
        ),
    )
    parser.add_argument(
        "--species-residual-scale-l2",
        type=float,
        default=0.0,
        help=(
            "Optional L2 penalty on effective species residual scales. "
            "Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--spatial-residual-logit-l2",
        type=float,
        default=0.0,
        help=(
            "Optional L2 penalty on spatial residual logits in each training "
            "batch. Use to discourage large spatial corrections. Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--spatial-residual-dropout",
        type=float,
        default=0.0,
        help=(
            "Training-only dropout probability applied to spatial residual "
            "logits before they are added to the base logits. Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--spatial-residual-noise-std",
        type=float,
        default=0.0,
        help=(
            "Training-only Gaussian noise standard deviation added to spatial "
            "residual logits before they are added to the base logits. Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--spatial-channel-mode",
        choices=["single", "separated"],
        default="single",
        help=(
            "Spatial message-passing channel structure. single uses one spatial "
            "cell encoder for all spatial corrections. separated uses an "
            "ecological cell channel for species-specific residuals and an "
            "access/effort cell channel for a shared checklist-level spatial "
            "bias. Defaults to single."
        ),
    )
    parser.add_argument(
        "--spatial-access-bias-l2",
        type=float,
        default=0.0,
        help=(
            "Optional L2 penalty on the shared access-channel spatial bias "
            "logits in each training batch. Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--access-density-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary MSE loss for separated spatial channels that "
            "predicts standardized log train-checklist density from the access "
            "cell embedding. Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--frozen-access-embeddings",
        default=None,
        help=(
            "Optional .npy file of pretrained spatial-cell access embeddings. "
            "Requires --spatial-channel-mode separated and an embedding dimension "
            "matching --cell-hidden-dim. The embeddings replace the learned "
            "access-cell channel while the ecological channel remains trainable."
        ),
    )
    parser.add_argument(
        "--effort-bias-mode",
        choices=["none", "shared", "lowrank"],
        default="none",
        help=(
            "Optional constrained effort/access bias component. shared learns "
            "one checklist-level logit adjustment shared by all species; "
            "lowrank adds species deviations through a low-rank interaction. "
            "Defaults to none."
        ),
    )
    parser.add_argument(
        "--effort-bias-rank",
        type=int,
        default=8,
        help="Rank for --effort-bias-mode lowrank. Defaults to 8.",
    )
    parser.add_argument(
        "--effort-bias-l2",
        type=float,
        default=0.0,
        help=(
            "Optional L2 penalty on effort/access bias logits in each training "
            "batch. Defaults to 0.0."
        ),
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


def default_component_feature_indices(feature_names: list[str]) -> tuple[list[int], list[int]]:
    ecology_names = {
        "day_of_year_sin",
        "day_of_year_cos",
        "canopy_median",
        "nc_usgs30m_match_tcc",
        "distance_to_waterbody_m",
        "distance_to_coastline_m",
    }
    bias_names = {
        "x",
        "y",
        "day_of_week_sin",
        "day_of_week_cos",
        "duration_log1p",
        "effort_distance_log1p",
        "number_observers_log1p",
        "is_traveling",
    }
    ecology = [idx for idx, name in enumerate(feature_names) if name in ecology_names]
    bias = [idx for idx, name in enumerate(feature_names) if name in bias_names]
    if not ecology:
        raise ValueError("Could not infer ecology feature indices from metadata.")
    if not bias:
        raise ValueError("Could not infer bias feature indices from metadata.")
    return ecology, bias


def build_spatial_cell_graph(
    graph_dir: Path,
    checklist_features: np.ndarray,
    train_checklists: np.ndarray,
    grid_size_m: float,
    cell_edge_mode: str = "spatial",
    environmental_neighbors: int = 6,
    environmental_cell_feature_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    if cell_edge_mode not in {"spatial", "environmental", "hybrid"}:
        raise ValueError(
            "--cell-edge-mode must be one of: spatial, environmental, hybrid."
        )
    if environmental_neighbors < 1:
        raise ValueError("--environmental-neighbors must be at least 1.")

    nodes = pd.read_parquet(
        graph_dir / "checklist_nodes.parquet",
        columns=["checklist_index", "x", "y"],
    ).sort_values("checklist_index")
    if not np.array_equal(
        nodes["checklist_index"].to_numpy(dtype=np.int64),
        np.arange(len(nodes), dtype=np.int64),
    ):
        raise ValueError("checklist_nodes.parquet is not ordered by checklist_index.")

    x = nodes["x"].to_numpy(dtype=np.float64)
    y = nodes["y"].to_numpy(dtype=np.float64)
    cell_x = np.floor(x / grid_size_m).astype(np.int64)
    cell_y = np.floor(y / grid_size_m).astype(np.int64)
    cell_keys = pd.Series(cell_x.astype(str) + "_" + cell_y.astype(str))
    all_cell_codes, unique_keys = pd.factorize(cell_keys, sort=False)
    cell_count = len(unique_keys)
    checklist_cell = all_cell_codes.astype(np.int64)

    train_codes = checklist_cell[train_checklists]
    train_counts = np.bincount(train_codes, minlength=cell_count).astype(np.float32)
    feature_sums = np.zeros((cell_count, checklist_features.shape[1]), dtype=np.float32)
    np.add.at(feature_sums, train_codes, checklist_features[train_checklists])
    mean_features = np.divide(
        feature_sums,
        np.maximum(train_counts[:, None], 1.0),
        out=np.zeros_like(feature_sums),
        where=train_counts[:, None] > 0,
    )

    x_sums = np.zeros(cell_count, dtype=np.float64)
    y_sums = np.zeros(cell_count, dtype=np.float64)
    all_counts = np.bincount(checklist_cell, minlength=cell_count).astype(np.float64)
    np.add.at(x_sums, checklist_cell, x)
    np.add.at(y_sums, checklist_cell, y)
    centroids = np.column_stack(
        [
            x_sums / np.maximum(all_counts, 1.0),
            y_sums / np.maximum(all_counts, 1.0),
        ]
    ).astype(np.float32)
    centroid_mean = centroids[train_codes].mean(axis=0)
    centroid_std = centroids[train_codes].std(axis=0)
    centroid_std[centroid_std == 0] = 1.0
    centroids_std = (centroids - centroid_mean) / centroid_std

    log_train_counts = np.log1p(train_counts)[:, None]
    count_mean = log_train_counts[train_codes].mean(axis=0)
    count_std = log_train_counts[train_codes].std(axis=0)
    count_std[count_std == 0] = 1.0
    log_train_counts = (log_train_counts - count_mean) / count_std

    cell_features = np.hstack(
        [centroids_std.astype(np.float32), log_train_counts.astype(np.float32), mean_features]
    ).astype(np.float32)

    edges: set[tuple[int, int]] = set()

    if cell_edge_mode in {"spatial", "hybrid"}:
        key_to_index = {
            tuple(map(int, key.split("_"))): idx for idx, key in enumerate(unique_keys)
        }
        for (cx, cy), src in key_to_index.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    dst = key_to_index.get((cx + dx, cy + dy))
                    if dst is not None:
                        edges.add((src, dst))
    else:
        edges.update((idx, idx) for idx in range(cell_count))

    environmental_edge_count = 0
    if cell_edge_mode in {"environmental", "hybrid"}:
        if not environmental_cell_feature_indices:
            raise ValueError(
                "Environmental cell edges require environmental cell feature indices."
            )
        env = cell_features[:, environmental_cell_feature_indices].astype(np.float32)
        env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
        env_mean = env.mean(axis=0, keepdims=True)
        env_std = env.std(axis=0, keepdims=True)
        env_std[env_std == 0] = 1.0
        env = (env - env_mean) / env_std
        distance_sq = (
            np.sum(env**2, axis=1, keepdims=True)
            + np.sum(env**2, axis=1, keepdims=True).T
            - 2.0 * env @ env.T
        )
        np.fill_diagonal(distance_sq, np.inf)
        neighbor_count = min(environmental_neighbors, max(cell_count - 1, 1))
        if cell_count > 1:
            nearest = np.argpartition(distance_sq, neighbor_count - 1, axis=1)[
                :, :neighbor_count
            ]
            for src, neighbors in enumerate(nearest):
                for dst in neighbors:
                    dst = int(dst)
                    if dst == src:
                        continue
                    if (src, dst) not in edges:
                        environmental_edge_count += 1
                    edges.add((src, dst))
                    if (dst, src) not in edges:
                        environmental_edge_count += 1
                    edges.add((dst, src))

    edge_index_np = np.array(sorted(edges), dtype=np.int64).T

    degree = np.bincount(edge_index_np[0], minlength=cell_count).astype(np.float32)
    weights = 1.0 / np.sqrt(
        np.maximum(degree[edge_index_np[0]], 1.0)
        * np.maximum(degree[edge_index_np[1]], 1.0)
    )
    indices = torch.from_numpy(edge_index_np)
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.from_numpy(weights.astype(np.float32)),
        size=(cell_count, cell_count),
        check_invariants=False,
    ).coalesce()
    metadata = {
        "spatial_cell_count": int(cell_count),
        "spatial_cell_edge_count": int(edge_index_np.shape[1]),
        "spatial_cell_feature_count": int(cell_features.shape[1]),
        "spatial_cells_with_train_checklists": int((train_counts > 0).sum()),
        "cell_edge_mode": cell_edge_mode,
        "environmental_neighbors": int(environmental_neighbors),
        "environmental_edge_count": int(environmental_edge_count),
        "environmental_cell_feature_indices": environmental_cell_feature_indices or [],
    }
    return torch.from_numpy(cell_features), adjacency, checklist_cell, metadata


def build_spatial_cell_graph_for_run(
    graph_dir: Path,
    checklist_features: np.ndarray,
    train_checklists: np.ndarray,
    model_info: dict,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    environmental_indices = model_info.get("environmental_cell_feature_indices")
    if environmental_indices is None:
        environmental_indices = model_info.get("ecology_cell_feature_indices", [])
    return build_spatial_cell_graph(
        graph_dir,
        checklist_features,
        train_checklists,
        float(model_info["spatial_grid_size_m"]),
        cell_edge_mode=model_info.get("cell_edge_mode", "spatial"),
        environmental_neighbors=int(model_info.get("environmental_neighbors", 6)),
        environmental_cell_feature_indices=environmental_indices,
    )


def build_species_adjacency(
    train_labels: np.ndarray,
    species_edge_mode: str = "none",
    species_neighbors: int = 10,
) -> tuple[torch.Tensor | None, dict]:
    if species_edge_mode not in {"none", "codetection"}:
        raise ValueError("--species-edge-mode must be none or codetection.")
    species_count = int(train_labels.shape[1])
    if species_edge_mode == "none":
        return None, {
            "species_edge_mode": "none",
            "species_neighbors": 0,
            "species_graph_edge_count": 0,
        }
    if species_neighbors < 1:
        raise ValueError("--species-neighbors must be at least 1.")

    labels = train_labels.astype(np.float32, copy=False)
    counts = labels.sum(axis=0)
    cooccurrence = labels.T @ labels
    np.fill_diagonal(cooccurrence, 0.0)
    denominator = np.sqrt(np.maximum(counts[:, None], 1.0) * np.maximum(counts[None, :], 1.0))
    similarity = np.divide(
        cooccurrence,
        denominator,
        out=np.zeros_like(cooccurrence, dtype=np.float32),
        where=denominator > 0,
    )

    edges: set[tuple[int, int]] = {(idx, idx) for idx in range(species_count)}
    neighbor_count = min(species_neighbors, max(species_count - 1, 1))
    if species_count > 1:
        for src in range(species_count):
            row = similarity[src]
            if np.all(row <= 0):
                continue
            nearest = np.argpartition(-row, neighbor_count - 1)[:neighbor_count]
            for dst in nearest:
                dst = int(dst)
                if dst == src or row[dst] <= 0:
                    continue
                edges.add((src, dst))
                edges.add((dst, src))

    edge_index_np = np.array(sorted(edges), dtype=np.int64).T
    degree = np.bincount(edge_index_np[0], minlength=species_count).astype(np.float32)
    weights = 1.0 / np.sqrt(
        np.maximum(degree[edge_index_np[0]], 1.0)
        * np.maximum(degree[edge_index_np[1]], 1.0)
    )
    adjacency = torch.sparse_coo_tensor(
        torch.from_numpy(edge_index_np),
        torch.from_numpy(weights.astype(np.float32)),
        size=(species_count, species_count),
        check_invariants=False,
    ).coalesce()
    return adjacency, {
        "species_edge_mode": species_edge_mode,
        "species_neighbors": int(species_neighbors),
        "species_graph_edge_count": int(edge_index_np.shape[1]),
        "species_with_train_detections": int((counts > 0).sum()),
    }


def build_species_adjacency_for_run(
    train_labels: np.ndarray,
    model_info: dict,
) -> tuple[torch.Tensor | None, dict]:
    return build_species_adjacency(
        train_labels,
        species_edge_mode=model_info.get("species_edge_mode", "none"),
        species_neighbors=int(model_info.get("species_neighbors", 10)),
    )


def load_frozen_access_embeddings(path: str | None, cell_count: int) -> torch.Tensor | None:
    if not path:
        return None
    frozen_path = Path(path)
    if not frozen_path.exists():
        raise FileNotFoundError(f"Frozen access embeddings do not exist: {frozen_path}")
    embeddings = np.load(frozen_path).astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError("--frozen-access-embeddings must be a 2D .npy array.")
    if embeddings.shape[0] != cell_count:
        raise ValueError(
            "--frozen-access-embeddings cell count does not match the current "
            f"spatial graph: {embeddings.shape[0]} != {cell_count}."
        )
    return torch.from_numpy(embeddings)


def inject_frozen_access_embeddings(
    model: "SpatialGCNHybrid",
    cell_embeddings: torch.Tensor,
    frozen_access_embeddings: torch.Tensor | None,
) -> torch.Tensor:
    if frozen_access_embeddings is None:
        return cell_embeddings
    if model.spatial_channel_mode != "separated":
        raise ValueError("Frozen access embeddings require separated spatial channels.")
    ecology_context, _access_context = model.split_cell_context(cell_embeddings)
    if ecology_context.shape[1] != frozen_access_embeddings.shape[1]:
        raise ValueError(
            "Frozen access embedding dimension must match the ecological cell "
            f"context dimension: {frozen_access_embeddings.shape[1]} != "
            f"{ecology_context.shape[1]}."
        )
    return torch.cat([ecology_context, frozen_access_embeddings], dim=1)


def cell_channel_feature_indices(feature_names: list[str]) -> tuple[list[int], list[int]]:
    """Return cell-feature columns for ecological and access spatial channels."""
    feature_offset = 3  # standardized centroid x/y plus log train checklist count.
    ecology_names = {
        "day_of_year_sin",
        "day_of_year_cos",
        "canopy_median",
        "nc_usgs30m_match_tcc",
        "distance_to_waterbody_m",
        "distance_to_coastline_m",
    }
    access_names = {
        "day_of_week_sin",
        "day_of_week_cos",
        "duration_log1p",
        "effort_distance_log1p",
        "number_observers_log1p",
        "is_traveling",
    }
    ecology_indices = [
        feature_offset + idx
        for idx, name in enumerate(feature_names)
        if name in ecology_names
    ]
    access_indices = [0, 1, 2] + [
        feature_offset + idx
        for idx, name in enumerate(feature_names)
        if name in access_names
    ]
    if not ecology_indices:
        raise ValueError("No ecological cell-channel features were found.")
    if len(access_indices) <= 3:
        raise ValueError("No access/effort cell-channel features were found.")
    return ecology_indices, access_indices


class SpatialGCNHybrid(nn.Module):
    def __init__(
        self,
        checklist_feature_dim: int,
        cell_feature_dim: int,
        species_count: int,
        hidden_dim: int,
        hidden_layers: int,
        latent_dim: int,
        cell_hidden_dim: int,
        cell_layers: int,
        dropout: float,
        gnn_mode: str,
        gate_init_bias: float,
        species_residual_scale: str,
        species_residual_scale_init: float,
        component_mode: str = "joint",
        ecology_feature_indices: list[int] | None = None,
        bias_feature_indices: list[int] | None = None,
        effort_bias_mode: str = "none",
        effort_bias_rank: int = 8,
        spatial_residual_dropout: float = 0.0,
        spatial_residual_noise_std: float = 0.0,
        spatial_channel_mode: str = "single",
        ecology_cell_feature_indices: list[int] | None = None,
        access_cell_feature_indices: list[int] | None = None,
        access_density_auxiliary: bool = False,
        species_gcn_layers: int = 0,
        species_gcn_dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_layers <= 0:
            raise ValueError("--hidden-layers must be greater than zero.")
        if cell_layers <= 0:
            raise ValueError("--cell-layers must be greater than zero.")
        if gnn_mode not in {"concat", "residual", "gated"}:
            raise ValueError("--gnn-mode must be concat, residual, or gated.")
        if species_residual_scale not in {"none", "sigmoid", "softplus"}:
            raise ValueError(
                "--species-residual-scale must be none, sigmoid, or softplus."
            )
        if species_residual_scale != "none" and gnn_mode == "concat":
            raise ValueError("--species-residual-scale requires residual or gated mode.")
        if species_residual_scale_init <= 0:
            raise ValueError("--species-residual-scale-init must be greater than zero.")
        if species_residual_scale == "sigmoid" and species_residual_scale_init >= 1:
            raise ValueError(
                "--species-residual-scale-init must be less than 1 for sigmoid."
            )
        if component_mode not in {"joint", "separated", "shared"}:
            raise ValueError("--component-mode must be joint, separated, or shared.")
        if component_mode == "separated":
            if not ecology_feature_indices or not bias_feature_indices:
                raise ValueError(
                    "Separated component mode requires ecology and bias feature indices."
                )
        if component_mode == "shared" and not bias_feature_indices:
            raise ValueError("Shared component mode requires bias feature indices.")
        if effort_bias_mode not in {"none", "shared", "lowrank"}:
            raise ValueError("--effort-bias-mode must be none, shared, or lowrank.")
        if effort_bias_mode != "none" and not bias_feature_indices:
            raise ValueError("--effort-bias-mode requires bias feature indices.")
        if effort_bias_rank <= 0:
            raise ValueError("--effort-bias-rank must be greater than zero.")
        if not 0 <= spatial_residual_dropout < 1:
            raise ValueError("--spatial-residual-dropout must be in [0, 1).")
        if spatial_residual_noise_std < 0:
            raise ValueError("--spatial-residual-noise-std must be nonnegative.")
        if spatial_channel_mode not in {"single", "separated"}:
            raise ValueError("--spatial-channel-mode must be single or separated.")
        if spatial_channel_mode == "separated" and gnn_mode == "concat":
            raise ValueError("--spatial-channel-mode separated requires residual or gated mode.")
        if spatial_channel_mode == "separated":
            if not ecology_cell_feature_indices or not access_cell_feature_indices:
                raise ValueError(
                    "Separated spatial channel mode requires ecology and access "
                    "cell feature indices."
                )
        if species_gcn_layers < 0:
            raise ValueError("--species-gcn-layers must be nonnegative.")
        if not 0 <= species_gcn_dropout < 1:
            raise ValueError("--species-gcn-dropout must be in [0, 1).")
        self.gnn_mode = gnn_mode
        self.component_mode = component_mode
        self.effort_bias_mode = effort_bias_mode
        self.effort_bias_rank = effort_bias_rank
        self.spatial_residual_dropout = spatial_residual_dropout
        self.spatial_residual_noise_std = spatial_residual_noise_std
        self.spatial_channel_mode = spatial_channel_mode
        self.access_density_auxiliary = access_density_auxiliary
        self.species_gcn_layers = species_gcn_layers
        self.species_gcn_dropout = species_gcn_dropout
        self.ecology_cell_feature_indices = list(ecology_cell_feature_indices or [])
        self.access_cell_feature_indices = list(access_cell_feature_indices or [])
        self.last_spatial_residual_logits: torch.Tensor | None = None
        self.last_spatial_access_bias_logits: torch.Tensor | None = None
        self.species_residual_scale_mode = species_residual_scale
        self.ecology_feature_indices = ecology_feature_indices or []
        self.bias_feature_indices = bias_feature_indices or []
        self.cell_linears = nn.ModuleList()
        self.ecology_cell_linears = nn.ModuleList()
        self.access_cell_linears = nn.ModuleList()
        if spatial_channel_mode == "single":
            current_cell_dim = cell_feature_dim
            for _ in range(cell_layers):
                self.cell_linears.append(nn.Linear(current_cell_dim, cell_hidden_dim))
                current_cell_dim = cell_hidden_dim
        else:
            current_cell_dim = len(self.ecology_cell_feature_indices)
            for _ in range(cell_layers):
                self.ecology_cell_linears.append(
                    nn.Linear(current_cell_dim, cell_hidden_dim)
                )
                current_cell_dim = cell_hidden_dim
            current_cell_dim = len(self.access_cell_feature_indices)
            for _ in range(cell_layers):
                self.access_cell_linears.append(
                    nn.Linear(current_cell_dim, cell_hidden_dim)
                )
                current_cell_dim = cell_hidden_dim

        self.checklist_encoder = None
        self.ecology_encoder = None
        self.bias_encoder = None
        if component_mode in {"joint", "shared"}:
            layers: list[nn.Module] = []
            if gnn_mode == "concat":
                current_dim = checklist_feature_dim + cell_hidden_dim
            else:
                current_dim = checklist_feature_dim
            for _ in range(hidden_layers):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, latent_dim))
            layers.append(nn.ReLU())
            self.checklist_encoder = nn.Sequential(*layers)
            if component_mode == "shared":
                bias_layers: list[nn.Module] = []
                current_dim = latent_dim + len(self.bias_feature_indices) + cell_hidden_dim
                for _ in range(hidden_layers):
                    bias_layers.append(nn.Linear(current_dim, hidden_dim))
                    bias_layers.append(nn.ReLU())
                    if dropout > 0:
                        bias_layers.append(nn.Dropout(dropout))
                    current_dim = hidden_dim
                bias_layers.append(nn.Linear(current_dim, latent_dim))
                bias_layers.append(nn.ReLU())
                self.bias_encoder = nn.Sequential(*bias_layers)
        elif component_mode == "separated":
            ecology_layers: list[nn.Module] = []
            current_dim = len(self.ecology_feature_indices)
            for _ in range(hidden_layers):
                ecology_layers.append(nn.Linear(current_dim, hidden_dim))
                ecology_layers.append(nn.ReLU())
                if dropout > 0:
                    ecology_layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            ecology_layers.append(nn.Linear(current_dim, latent_dim))
            ecology_layers.append(nn.ReLU())
            self.ecology_encoder = nn.Sequential(*ecology_layers)

            bias_layers: list[nn.Module] = []
            current_dim = len(self.bias_feature_indices) + cell_hidden_dim
            for _ in range(hidden_layers):
                bias_layers.append(nn.Linear(current_dim, hidden_dim))
                bias_layers.append(nn.ReLU())
                if dropout > 0:
                    bias_layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            bias_layers.append(nn.Linear(current_dim, latent_dim))
            bias_layers.append(nn.ReLU())
            self.bias_encoder = nn.Sequential(*bias_layers)

        self.species_embedding = nn.Embedding(species_count, latent_dim)
        self.species_linears = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(species_gcn_layers)]
        )
        self.species_bias = nn.Parameter(torch.zeros(species_count))
        self.checklist_bias = nn.Linear(latent_dim, 1)
        self.direct_head = nn.Linear(latent_dim, species_count)
        self.effort_bias_encoder = None
        self.effort_bias_shared_head = None
        self.effort_bias_lowrank_head = None
        self.effort_bias_species_embedding = None
        self.spatial_access_bias_head = None
        self.access_density_head = None
        if effort_bias_mode != "none":
            effort_layers: list[nn.Module] = []
            current_dim = len(self.bias_feature_indices) + cell_hidden_dim
            for _ in range(hidden_layers):
                effort_layers.append(nn.Linear(current_dim, hidden_dim))
                effort_layers.append(nn.ReLU())
                if dropout > 0:
                    effort_layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            effort_layers.append(nn.Linear(current_dim, latent_dim))
            effort_layers.append(nn.ReLU())
            self.effort_bias_encoder = nn.Sequential(*effort_layers)
            self.effort_bias_shared_head = nn.Linear(latent_dim, 1)
            nn.init.zeros_(self.effort_bias_shared_head.weight)
            nn.init.zeros_(self.effort_bias_shared_head.bias)
            if effort_bias_mode == "lowrank":
                self.effort_bias_lowrank_head = nn.Linear(latent_dim, effort_bias_rank)
                self.effort_bias_species_embedding = nn.Embedding(
                    species_count, effort_bias_rank
                )
                nn.init.zeros_(self.effort_bias_lowrank_head.weight)
                nn.init.zeros_(self.effort_bias_lowrank_head.bias)
                nn.init.normal_(
                    self.effort_bias_species_embedding.weight, mean=0.0, std=0.02
                )
        if spatial_channel_mode == "separated":
            self.spatial_access_bias_head = nn.Linear(cell_hidden_dim, 1)
            nn.init.zeros_(self.spatial_access_bias_head.weight)
            nn.init.zeros_(self.spatial_access_bias_head.bias)
            if access_density_auxiliary:
                self.access_density_head = nn.Linear(cell_hidden_dim, 1)
        if component_mode in {"separated", "shared"}:
            self.bias_checklist_bias = nn.Linear(latent_dim, 1)
            self.bias_direct_head = nn.Linear(latent_dim, species_count)
        else:
            self.bias_checklist_bias = None
            self.bias_direct_head = None
        if gnn_mode in {"residual", "gated"}:
            self.cell_residual_head = nn.Linear(cell_hidden_dim, species_count)
            nn.init.zeros_(self.cell_residual_head.weight)
            nn.init.zeros_(self.cell_residual_head.bias)
        else:
            self.cell_residual_head = None
        if species_residual_scale == "sigmoid":
            init = float(species_residual_scale_init)
            init_logit = np.log(init / (1.0 - init))
            self.species_residual_scale_param = nn.Parameter(
                torch.full((species_count,), float(init_logit))
            )
        elif species_residual_scale == "softplus":
            init = float(species_residual_scale_init)
            init_inverse = np.log(np.expm1(init))
            self.species_residual_scale_param = nn.Parameter(
                torch.full((species_count,), float(init_inverse))
            )
        else:
            self.species_residual_scale_param = None
        if gnn_mode == "gated":
            self.gate_head = nn.Linear(latent_dim + cell_hidden_dim, species_count)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, gate_init_bias)
        else:
            self.gate_head = None
        self.dropout = nn.Dropout(dropout)
        self.scale = float(np.sqrt(latent_dim))

    def encode_species(
        self, species_adjacency: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.species_embedding.weight
        if species_adjacency is None or not self.species_linears:
            return h
        for layer in self.species_linears:
            h = torch.sparse.mm(species_adjacency, h)
            h = layer(h)
            h = torch.relu(h)
            if self.training and self.species_gcn_dropout > 0:
                h = F.dropout(h, p=self.species_gcn_dropout, training=True)
        return h

    def encode_joint_checklist(
        self,
        checklist_features: torch.Tensor,
        cell_context: torch.Tensor,
    ) -> torch.Tensor:
        if self.checklist_encoder is None:
            raise RuntimeError("Joint checklist encoder is not initialized.")
        if self.gnn_mode == "concat":
            model_input = torch.cat([checklist_features, cell_context], dim=1)
        else:
            model_input = checklist_features
        return self.checklist_encoder(model_input)

    def encode_components(
        self,
        checklist_features: torch.Tensor,
        cell_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ecology_encoder is None or self.bias_encoder is None:
            raise RuntimeError("Separated component encoders are not initialized.")
        ecology_input = checklist_features[:, self.ecology_feature_indices]
        bias_input = torch.cat(
            [checklist_features[:, self.bias_feature_indices], cell_context],
            dim=1,
        )
        return self.ecology_encoder(ecology_input), self.bias_encoder(bias_input)

    def encode_shared_bias(
        self,
        shared_latent: torch.Tensor,
        checklist_features: torch.Tensor,
        cell_context: torch.Tensor,
    ) -> torch.Tensor:
        if self.bias_encoder is None:
            raise RuntimeError("Shared bias encoder is not initialized.")
        bias_input = torch.cat(
            [
                shared_latent,
                checklist_features[:, self.bias_feature_indices],
                cell_context,
            ],
            dim=1,
        )
        return self.bias_encoder(bias_input)

    def effort_bias_logits(
        self,
        checklist_features: torch.Tensor,
        cell_context: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.effort_bias_mode == "none":
            return None
        if self.effort_bias_encoder is None or self.effort_bias_shared_head is None:
            raise RuntimeError("Effort/access bias encoder is not initialized.")
        effort_input = torch.cat(
            [checklist_features[:, self.bias_feature_indices], cell_context],
            dim=1,
        )
        effort_latent = self.effort_bias_encoder(effort_input)
        bias_logits = self.effort_bias_shared_head(effort_latent)
        if self.effort_bias_mode == "lowrank":
            if (
                self.effort_bias_lowrank_head is None
                or self.effort_bias_species_embedding is None
            ):
                raise RuntimeError("Low-rank effort/access bias head is not initialized.")
            checklist_bias = self.effort_bias_lowrank_head(effort_latent)
            bias_logits = (
                bias_logits
                + checklist_bias @ self.effort_bias_species_embedding.weight.T
                / float(np.sqrt(self.effort_bias_rank))
            )
        return bias_logits

    def species_residual_scales(self) -> torch.Tensor | None:
        if self.species_residual_scale_param is None:
            return None
        if self.species_residual_scale_mode == "sigmoid":
            return torch.sigmoid(self.species_residual_scale_param)
        if self.species_residual_scale_mode == "softplus":
            return torch.nn.functional.softplus(self.species_residual_scale_param)
        return None

    def encode_cells(
        self, cell_features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        if self.spatial_channel_mode == "single":
            h = cell_features
            for layer in self.cell_linears:
                h = torch.sparse.mm(adjacency, h)
                h = layer(h)
                h = torch.relu(h)
                h = self.dropout(h)
            return h
        ecology_h = cell_features[:, self.ecology_cell_feature_indices]
        for layer in self.ecology_cell_linears:
            ecology_h = torch.sparse.mm(adjacency, ecology_h)
            ecology_h = layer(ecology_h)
            ecology_h = torch.relu(ecology_h)
            ecology_h = self.dropout(ecology_h)
        access_h = cell_features[:, self.access_cell_feature_indices]
        for layer in self.access_cell_linears:
            access_h = torch.sparse.mm(adjacency, access_h)
            access_h = layer(access_h)
            access_h = torch.relu(access_h)
            access_h = self.dropout(access_h)
        return torch.cat([ecology_h, access_h], dim=1)

    def split_cell_context(
        self, cell_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spatial_channel_mode == "single":
            return cell_context, cell_context
        midpoint = cell_context.shape[1] // 2
        return cell_context[:, :midpoint], cell_context[:, midpoint:]

    def access_density_prediction(
        self, cell_embeddings: torch.Tensor
    ) -> torch.Tensor | None:
        if self.access_density_head is None:
            return None
        _ecology_context, access_context = self.split_cell_context(cell_embeddings)
        return self.access_density_head(access_context).squeeze(-1)

    def base_and_full_logits(
        self,
        checklist_features: torch.Tensor,
        cell_embeddings: torch.Tensor,
        checklist_cell: torch.Tensor,
        species_adjacency: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cell_context = cell_embeddings[checklist_cell]
        ecology_cell_context, access_cell_context = self.split_cell_context(cell_context)
        base_cell_context = access_cell_context
        species_embeddings = self.encode_species(species_adjacency)
        if self.component_mode == "joint":
            latent = self.encode_joint_checklist(checklist_features, base_cell_context)
            logits = latent @ species_embeddings.T / self.scale
            logits = logits + self.species_bias + self.checklist_bias(latent)
            logits = logits + self.direct_head(latent)
        elif self.component_mode == "shared":
            latent = self.encode_joint_checklist(checklist_features, base_cell_context)
            logits = latent @ species_embeddings.T / self.scale
            logits = logits + self.species_bias + self.checklist_bias(latent)
            logits = logits + self.direct_head(latent)
            bias_latent = self.encode_shared_bias(
                latent, checklist_features, access_cell_context
            )
            if self.bias_checklist_bias is None or self.bias_direct_head is None:
                raise RuntimeError("Shared bias heads are not initialized.")
            logits = logits + self.bias_checklist_bias(bias_latent)
            logits = logits + self.bias_direct_head(bias_latent)
        else:
            ecology_latent, bias_latent = self.encode_components(
                checklist_features, access_cell_context
            )
            logits = ecology_latent @ species_embeddings.T / self.scale
            logits = logits + self.species_bias + self.checklist_bias(ecology_latent)
            logits = logits + self.direct_head(ecology_latent)
            if self.bias_checklist_bias is None or self.bias_direct_head is None:
                raise RuntimeError("Separated bias heads are not initialized.")
            logits = logits + self.bias_checklist_bias(bias_latent)
            logits = logits + self.bias_direct_head(bias_latent)
        effort_bias_logits = self.effort_bias_logits(
            checklist_features, access_cell_context
        )
        if effort_bias_logits is not None:
            logits = logits + effort_bias_logits
        if self.spatial_channel_mode == "separated":
            if self.spatial_access_bias_head is None:
                raise RuntimeError("Spatial access bias head is not initialized.")
            access_bias_logits = self.spatial_access_bias_head(access_cell_context)
            self.last_spatial_access_bias_logits = access_bias_logits
            logits = logits + access_bias_logits
        else:
            self.last_spatial_access_bias_logits = None
        base_logits = logits
        if self.gnn_mode in {"residual", "gated"}:
            if self.cell_residual_head is None:
                raise RuntimeError("Residual head is not initialized.")
            residual_logits = self.cell_residual_head(ecology_cell_context)
            residual_scales = self.species_residual_scales()
            if residual_scales is not None:
                residual_logits = residual_logits * residual_scales
            if self.gnn_mode == "gated":
                if self.gate_head is None:
                    raise RuntimeError("Gate head is not initialized.")
                if self.component_mode == "joint":
                    gate_latent = latent
                elif self.component_mode == "shared":
                    gate_latent = bias_latent
                else:
                    gate_latent = bias_latent
                gate_input = torch.cat([gate_latent, ecology_cell_context], dim=1)
                residual_logits = torch.sigmoid(self.gate_head(gate_input)) * residual_logits
            self.last_spatial_residual_logits = residual_logits
            if self.training and self.spatial_residual_dropout > 0:
                residual_logits = F.dropout(
                    residual_logits,
                    p=self.spatial_residual_dropout,
                    training=True,
                )
            if self.training and self.spatial_residual_noise_std > 0:
                residual_logits = residual_logits + torch.randn_like(
                    residual_logits
                ) * self.spatial_residual_noise_std
            logits = logits + residual_logits
        return base_logits, logits

    def forward_all_species(
        self,
        checklist_features: torch.Tensor,
        cell_embeddings: torch.Tensor,
        checklist_cell: torch.Tensor,
        species_adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _base_logits, full_logits = self.base_and_full_logits(
            checklist_features, cell_embeddings, checklist_cell, species_adjacency
        )
        return full_logits


def evaluate_all_pairs(
    model: SpatialGCNHybrid,
    features: torch.Tensor,
    cell_features: torch.Tensor,
    adjacency: torch.Tensor,
    frozen_access_embeddings: torch.Tensor | None,
    species_adjacency: torch.Tensor | None,
    checklist_cells: torch.Tensor,
    checklist_indices: np.ndarray,
    labels: np.ndarray,
    species: pd.DataFrame,
    batch_size: int,
    calibration_bins: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    model.eval()
    score_parts = []
    with torch.no_grad():
        cell_embeddings = model.encode_cells(cell_features, adjacency)
        cell_embeddings = inject_frozen_access_embeddings(
            model, cell_embeddings, frozen_access_embeddings
        )
        for start in range(0, len(checklist_indices), batch_size):
            batch = checklist_indices[start : start + batch_size]
            checklist_tensor = torch.from_numpy(batch.astype(np.int64))
            logits = model.forward_all_species(
                features[checklist_tensor],
                cell_embeddings,
                checklist_cells[checklist_tensor],
                species_adjacency,
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
    if args.access_density_loss_weight < 0:
        raise ValueError("--access-density-loss-weight must be nonnegative.")
    if args.access_density_loss_weight > 0 and args.spatial_channel_mode != "separated":
        raise ValueError(
            "--access-density-loss-weight requires --spatial-channel-mode separated."
        )
    if args.frozen_access_embeddings and args.spatial_channel_mode != "separated":
        raise ValueError(
            "--frozen-access-embeddings requires --spatial-channel-mode separated."
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else graph_dir / "spatial_gnn_baselines"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    features_np = np.load(graph_dir / "checklist_features.npy").astype(np.float32)
    features = torch.from_numpy(features_np)
    species = pd.read_csv(graph_dir / "species_nodes.csv")
    species_count = int(metadata["counts"]["species"])
    feature_names = list(metadata.get("feature_names", []))
    if args.component_mode in {"separated", "shared"} or args.effort_bias_mode != "none":
        ecology_indices, bias_indices = default_component_feature_indices(feature_names)
    else:
        ecology_indices, bias_indices = [], []

    train_checklists = load_split_checklists(
        graph_dir, "train", args.max_train_checklists, args.seed
    )
    test_checklists = load_split_checklists(
        graph_dir, "test", args.max_eval_checklists, args.seed + 1
    )
    train_labels = build_label_matrix(graph_dir, train_checklists, species_count, "train")
    test_labels = build_label_matrix(graph_dir, test_checklists, species_count, "test")
    if args.spatial_channel_mode == "separated" or args.cell_edge_mode in {
        "environmental",
        "hybrid",
    }:
        ecology_cell_indices, access_cell_indices = cell_channel_feature_indices(
            feature_names
        )
    else:
        ecology_cell_indices, access_cell_indices = [], []
    cell_features, adjacency, checklist_cell_np, cell_metadata = build_spatial_cell_graph(
        graph_dir,
        features_np,
        train_checklists,
        args.spatial_grid_size_m,
        cell_edge_mode=args.cell_edge_mode,
        environmental_neighbors=args.environmental_neighbors,
        environmental_cell_feature_indices=ecology_cell_indices,
    )
    checklist_cells = torch.from_numpy(checklist_cell_np.astype(np.int64))
    frozen_access_embeddings = load_frozen_access_embeddings(
        args.frozen_access_embeddings,
        int(cell_metadata["spatial_cell_count"]),
    )
    if frozen_access_embeddings is not None and frozen_access_embeddings.shape[1] != args.cell_hidden_dim:
        raise ValueError(
            "--frozen-access-embeddings dimension must match --cell-hidden-dim: "
            f"{frozen_access_embeddings.shape[1]} != {args.cell_hidden_dim}."
        )
    species_adjacency, species_graph_metadata = build_species_adjacency(
        train_labels,
        species_edge_mode=args.species_edge_mode,
        species_neighbors=args.species_neighbors,
    )
    if args.species_gcn_layers > 0 and species_adjacency is None:
        raise ValueError("--species-gcn-layers requires --species-edge-mode codetection.")

    model = SpatialGCNHybrid(
        checklist_feature_dim=features_np.shape[1],
        cell_feature_dim=cell_features.shape[1],
        species_count=species_count,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        latent_dim=args.latent_dim,
        cell_hidden_dim=args.cell_hidden_dim,
        cell_layers=args.cell_layers,
        dropout=args.dropout,
        gnn_mode=args.gnn_mode,
        gate_init_bias=args.gate_init_bias,
        species_residual_scale=args.species_residual_scale,
        species_residual_scale_init=args.species_residual_scale_init,
        component_mode=args.component_mode,
        ecology_feature_indices=ecology_indices,
        bias_feature_indices=bias_indices,
        effort_bias_mode=args.effort_bias_mode,
        effort_bias_rank=args.effort_bias_rank,
        spatial_residual_dropout=args.spatial_residual_dropout,
        spatial_residual_noise_std=args.spatial_residual_noise_std,
        spatial_channel_mode=args.spatial_channel_mode,
        ecology_cell_feature_indices=ecology_cell_indices,
        access_cell_feature_indices=access_cell_indices,
        access_density_auxiliary=args.access_density_loss_weight > 0,
        species_gcn_layers=args.species_gcn_layers,
        species_gcn_dropout=args.species_gcn_dropout,
    )
    if model.species_residual_scale_param is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        scale_param_id = id(model.species_residual_scale_param)
        decay_params = [
            parameter
            for parameter in model.parameters()
            if id(parameter) != scale_param_id
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": [model.species_residual_scale_param], "weight_decay": 0.0},
            ],
            lr=args.learning_rate,
        )
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.from_numpy(train_checklists.astype(np.int64)),
        torch.from_numpy(train_labels),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for checklist_index, y in train_loader:
            optimizer.zero_grad()
            cell_embeddings = model.encode_cells(cell_features, adjacency)
            cell_embeddings = inject_frozen_access_embeddings(
                model, cell_embeddings, frozen_access_embeddings
            )
            logits = model.forward_all_species(
                features[checklist_index],
                cell_embeddings,
                checklist_cells[checklist_index],
                species_adjacency,
            )
            loss = criterion(logits, y)
            residual_scales = model.species_residual_scales()
            if residual_scales is not None and args.species_residual_scale_l2 > 0:
                loss = loss + args.species_residual_scale_l2 * torch.mean(
                    residual_scales**2
                )
            if (
                args.spatial_residual_logit_l2 > 0
                and model.last_spatial_residual_logits is not None
            ):
                loss = loss + args.spatial_residual_logit_l2 * torch.mean(
                    model.last_spatial_residual_logits**2
                )
            if (
                args.spatial_access_bias_l2 > 0
                and model.last_spatial_access_bias_logits is not None
            ):
                loss = loss + args.spatial_access_bias_l2 * torch.mean(
                    model.last_spatial_access_bias_logits**2
                )
            if args.access_density_loss_weight > 0:
                density_prediction = model.access_density_prediction(cell_embeddings)
                if density_prediction is None:
                    raise ValueError(
                        "--access-density-loss-weight requires separated spatial "
                        "channels."
                    )
                loss = loss + args.access_density_loss_weight * F.mse_loss(
                    density_prediction,
                    cell_features[:, 2],
                )
            if args.effort_bias_l2 > 0:
                with torch.no_grad():
                    batch_cell_context = cell_embeddings[checklist_cells[checklist_index]]
                    _ecology_context, batch_access_context = model.split_cell_context(
                        batch_cell_context
                    )
                effort_bias_logits = model.effort_bias_logits(
                    features[checklist_index], batch_access_context
                )
                if effort_bias_logits is not None:
                    loss = loss + args.effort_bias_l2 * torch.mean(
                        effort_bias_logits**2
                    )
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
        cell_features,
        adjacency,
        frozen_access_embeddings,
        species_adjacency,
        checklist_cells,
        test_checklists,
        test_labels,
        species,
        args.batch_size,
        args.calibration_bins,
    )
    summary["model"] = {
        "architecture": "spatial-cell-gcn-hybrid",
        "gnn_mode": args.gnn_mode,
        "component_mode": args.component_mode,
        "ecology_feature_indices": ecology_indices,
        "ecology_feature_names": [feature_names[idx] for idx in ecology_indices],
        "bias_feature_indices": bias_indices,
        "bias_feature_names": [feature_names[idx] for idx in bias_indices],
        "effort_bias_mode": args.effort_bias_mode,
        "effort_bias_rank": args.effort_bias_rank,
        "effort_bias_l2": args.effort_bias_l2,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "latent_dim": args.latent_dim,
        "cell_hidden_dim": args.cell_hidden_dim,
        "cell_layers": args.cell_layers,
        "dropout": args.dropout,
        "gate_init_bias": args.gate_init_bias,
        "species_residual_scale": args.species_residual_scale,
        "species_residual_scale_init": args.species_residual_scale_init,
        "species_residual_scale_l2": args.species_residual_scale_l2,
        "spatial_residual_logit_l2": args.spatial_residual_logit_l2,
        "spatial_residual_dropout": args.spatial_residual_dropout,
        "spatial_residual_noise_std": args.spatial_residual_noise_std,
        "spatial_channel_mode": args.spatial_channel_mode,
        "cell_edge_mode": args.cell_edge_mode,
        "environmental_neighbors": args.environmental_neighbors,
        "access_density_auxiliary": args.access_density_loss_weight > 0,
        "access_density_loss_weight": args.access_density_loss_weight,
        "frozen_access_embeddings": args.frozen_access_embeddings,
        "ecology_cell_feature_indices": ecology_cell_indices,
        "access_cell_feature_indices": access_cell_indices,
        "spatial_access_bias_l2": args.spatial_access_bias_l2,
        "species_edge_mode": args.species_edge_mode,
        "species_neighbors": args.species_neighbors,
        "species_gcn_layers": args.species_gcn_layers,
        "species_gcn_dropout": args.species_gcn_dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "spatial_grid_size_m": args.spatial_grid_size_m,
        **cell_metadata,
        **species_graph_metadata,
    }
    summary["train"] = {
        "checklists": int(len(train_checklists)),
        "pairs": int(train_labels.size),
        "positives": int(train_labels.sum()),
        "observed_rate": float(train_labels.mean()),
    }
    residual_scales = model.species_residual_scales()
    if residual_scales is not None:
        scale_np = residual_scales.detach().cpu().numpy()
        summary["species_residual_scales"] = {
            "mean": float(scale_np.mean()),
            "min": float(scale_np.min()),
            "max": float(scale_np.max()),
            "p10": float(np.quantile(scale_np, 0.10)),
            "p50": float(np.quantile(scale_np, 0.50)),
            "p90": float(np.quantile(scale_np, 0.90)),
        }
        species_metrics["residual_scale"] = scale_np[
            species_metrics["species_index"].to_numpy(dtype=np.int64)
        ]

    prefix = f"spatial_gnn_{args.run_name}"
    pd.DataFrame(history).to_csv(output_dir / f"{prefix}_history.csv", index=False)
    species_metrics.to_csv(output_dir / f"{prefix}_test_species_metrics.csv", index=False)
    calibration.to_csv(output_dir / f"{prefix}_test_calibration.csv", index=False)
    (output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), output_dir / f"{prefix}_model.pt")

    print("\nSpatial-cell GNN metrics:")
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
