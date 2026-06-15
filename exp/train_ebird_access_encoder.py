"""
Train a spatial-cell access/effort encoder for the eBird graph dataset.

This is the first stage of a two-stage observation-process path. It trains a
cell-level GCN to predict train-only effort/access summaries, then saves the
access embeddings for later use in species detection models.

Run from the project root:

    python exp/train_ebird_access_encoder.py --graph-dir data/ebird/graph_top100_spatial_10x10
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

from ebird_graph_all_species_baseline import load_split_checklists
from ebird_joint_tabular_baseline import SEED
from ebird_spatial_gnn_baseline import (
    build_spatial_cell_graph,
    cell_channel_feature_indices,
)


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial_10x10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a spatial-cell access/effort encoder."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to graph-dir/access_encoder.",
    )
    parser.add_argument(
        "--run-name",
        default="access_gcn_h64_l2_z64",
        help="Output file suffix. Defaults to access_gcn_h64_l2_z64.",
    )
    parser.add_argument(
        "--spatial-grid-size-m",
        type=float,
        default=25_000.0,
        help="Grid cell size matching the spatial GNN. Defaults to 25,000 m.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden width for the access GCN. Defaults to 64.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of graph-convolution layers. Defaults to 2.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Saved access embedding dimension. Defaults to 64.",
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
        default=500,
        help="Training epochs. Defaults to 500.",
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
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of train-observed cells held out for validation. Defaults to 0.20.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed. Defaults to 19.",
    )
    return parser.parse_args()


class AccessGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        embedding_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        if layers <= 0:
            raise ValueError("--layers must be greater than zero.")
        self.linears = nn.ModuleList()
        current_dim = input_dim
        for _ in range(layers):
            self.linears.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)
        self.output_head = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.linears:
            h = torch.sparse.mm(adjacency, h)
            h = layer(h)
            h = F.relu(h)
            h = self.dropout(h)
        return F.relu(self.embedding_head(h))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(x, adjacency)
        return self.output_head(embedding), embedding


def safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return std


def build_cell_targets(
    graph_dir: Path,
    checklist_cell: np.ndarray,
    train_checklists: np.ndarray,
    seed: int,
    validation_fraction: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    nodes = pd.read_parquet(graph_dir / "checklist_nodes.parquet").sort_values(
        "checklist_index"
    )
    train_nodes = nodes.loc[nodes["checklist_index"].isin(train_checklists)].copy()
    train_nodes["spatial_cell"] = checklist_cell[
        train_nodes["checklist_index"].to_numpy(dtype=np.int64)
    ]
    grouped = train_nodes.groupby("spatial_cell", sort=True)
    cell_count = int(checklist_cell.max()) + 1

    summary = pd.DataFrame(
        {"spatial_cell": np.arange(cell_count, dtype=np.int64)}
    ).set_index("spatial_cell", drop=False)
    count = grouped.size().rename("train_checklists")
    traveling = (
        grouped["protocol_name"]
        .apply(lambda values: float((values == "Traveling").mean()))
        .rename("traveling_rate")
    )
    stationary = (
        grouped["protocol_name"]
        .apply(lambda values: float((values == "Stationary").mean()))
        .rename("stationary_rate")
    )
    duration = grouped["duration_minutes"].mean().rename("duration_minutes_mean")
    distance = grouped["effort_distance_km"].mean().rename("effort_distance_km_mean")
    observers = grouped["number_observers"].mean().rename("number_observers_mean")
    unique_observers = grouped["observer_id"].nunique().rename("unique_observers")
    unique_localities = grouped["locality_id"].nunique().rename("unique_localities")
    summary = summary.join(
        pd.concat(
            [
                count,
                traveling,
                stationary,
                duration,
                distance,
                observers,
                unique_observers,
                unique_localities,
            ],
            axis=1,
        )
    )
    summary["train_checklists"] = summary["train_checklists"].fillna(0).astype(int)
    fill_zero_columns = [
        "traveling_rate",
        "stationary_rate",
        "duration_minutes_mean",
        "effort_distance_km_mean",
        "number_observers_mean",
        "unique_observers",
        "unique_localities",
    ]
    summary[fill_zero_columns] = summary[fill_zero_columns].fillna(0.0)
    summary["log_train_checklists"] = np.log1p(summary["train_checklists"])
    summary["log_unique_observers"] = np.log1p(summary["unique_observers"])
    summary["log_unique_localities"] = np.log1p(summary["unique_localities"])
    summary["duration_log1p_mean"] = np.log1p(summary["duration_minutes_mean"])
    summary["effort_distance_log1p_mean"] = np.log1p(summary["effort_distance_km_mean"])
    summary["number_observers_log1p_mean"] = np.log1p(summary["number_observers_mean"])
    summary["observer_per_checklist"] = np.divide(
        summary["unique_observers"],
        np.maximum(summary["train_checklists"], 1),
    )
    summary["locality_per_checklist"] = np.divide(
        summary["unique_localities"],
        np.maximum(summary["train_checklists"], 1),
    )

    target_names = [
        "log_train_checklists",
        "traveling_rate",
        "stationary_rate",
        "duration_log1p_mean",
        "effort_distance_log1p_mean",
        "number_observers_log1p_mean",
        "log_unique_observers",
        "log_unique_localities",
        "observer_per_checklist",
        "locality_per_checklist",
    ]
    observed_cells = summary.index[summary["train_checklists"].to_numpy() > 0].to_numpy()
    if len(observed_cells) < 3:
        raise ValueError("Access encoder requires at least three observed train cells.")
    rng = np.random.default_rng(seed)
    shuffled = observed_cells.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * validation_fraction)))
    val_cells = np.sort(shuffled[:val_count])
    train_cells = np.sort(shuffled[val_count:])
    if len(train_cells) == 0:
        raise ValueError("Validation fraction left no cells for training.")

    raw_targets = summary[target_names].to_numpy(dtype=np.float32)
    target_mean = raw_targets[train_cells].mean(axis=0)
    target_std = safe_std(raw_targets[train_cells])
    targets = (raw_targets - target_mean) / target_std
    summary[[f"{name}_z" for name in target_names]] = targets
    return summary, targets.astype(np.float32), target_mean, target_std, train_cells, val_cells, target_names


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cell_indices: np.ndarray,
    weights: torch.Tensor,
) -> torch.Tensor:
    idx = torch.from_numpy(cell_indices.astype(np.int64))
    errors = (prediction[idx] - target[idx]) ** 2
    return torch.mean(errors * weights[idx, None])


def target_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    cell_indices: np.ndarray,
    target_names: list[str],
    split: str,
) -> list[dict]:
    rows = []
    for target_idx, target_name in enumerate(target_names):
        y = target[cell_indices, target_idx]
        yhat = prediction[cell_indices, target_idx]
        if len(y) < 2 or np.std(y) == 0 or np.std(yhat) == 0:
            corr = np.nan
        else:
            corr = float(np.corrcoef(y, yhat)[0, 1])
        rows.append(
            {
                "split": split,
                "target": target_name,
                "cells": int(len(cell_indices)),
                "mse": float(np.mean((yhat - y) ** 2)),
                "mae": float(np.mean(np.abs(yhat - y))),
                "pearson": corr,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir) if args.output_dir else graph_dir / "access_encoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_names = list(metadata.get("feature_names", []))
    features_np = np.load(graph_dir / "checklist_features.npy").astype(np.float32)
    train_checklists = load_split_checklists(graph_dir, "train", None, args.seed)
    cell_features, adjacency, checklist_cell, cell_metadata = build_spatial_cell_graph(
        graph_dir,
        features_np,
        train_checklists,
        args.spatial_grid_size_m,
    )
    _ecology_indices, access_indices = cell_channel_feature_indices(feature_names)
    access_features = cell_features[:, access_indices]
    target_frame, targets_np, target_mean, target_std, train_cells, val_cells, target_names = build_cell_targets(
        graph_dir,
        checklist_cell,
        train_checklists,
        args.seed,
        args.validation_fraction,
    )
    target_frame.to_csv(output_dir / f"{args.run_name}_cell_access_targets.csv", index=False)

    model = AccessGCN(
        input_dim=access_features.shape[1],
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        embedding_dim=args.embedding_dim,
        output_dim=targets_np.shape[1],
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    targets = torch.from_numpy(targets_np)
    weights_np = np.sqrt(np.maximum(target_frame["train_checklists"].to_numpy(dtype=np.float32), 1.0))
    weights_np = weights_np / weights_np[train_cells].mean()
    weights = torch.from_numpy(weights_np.astype(np.float32))

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction, _embedding = model(access_features, adjacency)
        train_loss = weighted_mse(prediction, targets, train_cells, weights)
        train_loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == args.epochs or epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                eval_prediction, _ = model(access_features, adjacency)
                val_loss = weighted_mse(eval_prediction, targets, val_cells, weights)
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": float(train_loss.detach()),
                    "val_mse": float(val_loss.detach()),
                }
            )
            print(
                f"epoch {epoch:>4}: train MSE={float(train_loss.detach()):.5f}, "
                f"val MSE={float(val_loss.detach()):.5f}"
            )

    model.eval()
    with torch.no_grad():
        prediction, embedding = model(access_features, adjacency)
    prediction_np = prediction.cpu().numpy().astype(np.float32)
    embedding_np = embedding.cpu().numpy().astype(np.float32)
    np.save(output_dir / f"{args.run_name}_cell_embeddings.npy", embedding_np)
    np.save(output_dir / f"{args.run_name}_cell_predictions_z.npy", prediction_np)
    pd.DataFrame(history).to_csv(output_dir / f"{args.run_name}_history.csv", index=False)
    metric_rows = []
    metric_rows.extend(
        target_metrics(prediction_np, targets_np, train_cells, target_names, "train")
    )
    metric_rows.extend(
        target_metrics(prediction_np, targets_np, val_cells, target_names, "validation")
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / f"{args.run_name}_target_metrics.csv", index=False)
    torch.save(model.state_dict(), output_dir / f"{args.run_name}_model.pt")
    summary = {
        "graph_dir": str(graph_dir),
        "run_name": args.run_name,
        "model": {
            "architecture": "access-cell-gcn",
            "input_dim": int(access_features.shape[1]),
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "embedding_dim": args.embedding_dim,
            "output_dim": int(targets_np.shape[1]),
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "spatial_grid_size_m": args.spatial_grid_size_m,
            "access_cell_feature_indices": access_indices,
            "target_names": target_names,
            "target_mean": target_mean.tolist(),
            "target_std": target_std.tolist(),
            **cell_metadata,
        },
        "cells": {
            "total": int(len(target_frame)),
            "observed_train_cells": int((target_frame["train_checklists"] > 0).sum()),
            "train_cells": int(len(train_cells)),
            "validation_cells": int(len(val_cells)),
        },
        "outputs": {
            "model": f"{args.run_name}_model.pt",
            "cell_embeddings": f"{args.run_name}_cell_embeddings.npy",
            "cell_predictions_z": f"{args.run_name}_cell_predictions_z.npy",
            "cell_access_targets": f"{args.run_name}_cell_access_targets.csv",
            "target_metrics": f"{args.run_name}_target_metrics.csv",
            "history": f"{args.run_name}_history.csv",
        },
    }
    (output_dir / f"{args.run_name}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nWrote access encoder outputs to {output_dir}")
    print("\nValidation target metrics:")
    print(
        metrics.loc[metrics["split"] == "validation"]
        .sort_values("mse")
        .to_string(index=False, float_format="%.4f")
    )


if __name__ == "__main__":
    main()
