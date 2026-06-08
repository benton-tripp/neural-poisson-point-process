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
        "--spatial-grid-size-m",
        type=float,
        default=25_000.0,
        help="Grid cell size for spatial-cell nodes. Defaults to 25,000 m.",
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


def build_spatial_cell_graph(
    graph_dir: Path,
    checklist_features: np.ndarray,
    train_checklists: np.ndarray,
    grid_size_m: float,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
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

    key_to_index = {
        tuple(map(int, key.split("_"))): idx for idx, key in enumerate(unique_keys)
    }
    edges: set[tuple[int, int]] = set()
    for (cx, cy), src in key_to_index.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                dst = key_to_index.get((cx + dx, cy + dy))
                if dst is not None:
                    edges.add((src, dst))
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
    }
    return torch.from_numpy(cell_features), adjacency, checklist_cell, metadata


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
    ):
        super().__init__()
        if hidden_layers <= 0:
            raise ValueError("--hidden-layers must be greater than zero.")
        if cell_layers <= 0:
            raise ValueError("--cell-layers must be greater than zero.")
        if gnn_mode not in {"concat", "residual", "gated"}:
            raise ValueError("--gnn-mode must be concat, residual, or gated.")
        self.gnn_mode = gnn_mode
        self.cell_linears = nn.ModuleList()
        current_cell_dim = cell_feature_dim
        for _ in range(cell_layers):
            self.cell_linears.append(nn.Linear(current_cell_dim, cell_hidden_dim))
            current_cell_dim = cell_hidden_dim
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
        self.species_embedding = nn.Embedding(species_count, latent_dim)
        self.species_bias = nn.Parameter(torch.zeros(species_count))
        self.checklist_bias = nn.Linear(latent_dim, 1)
        self.direct_head = nn.Linear(latent_dim, species_count)
        if gnn_mode in {"residual", "gated"}:
            self.cell_residual_head = nn.Linear(cell_hidden_dim, species_count)
            nn.init.zeros_(self.cell_residual_head.weight)
            nn.init.zeros_(self.cell_residual_head.bias)
        else:
            self.cell_residual_head = None
        if gnn_mode == "gated":
            self.gate_head = nn.Linear(latent_dim + cell_hidden_dim, species_count)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, gate_init_bias)
        else:
            self.gate_head = None
        self.dropout = nn.Dropout(dropout)
        self.scale = float(np.sqrt(latent_dim))

    def encode_cells(
        self, cell_features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        h = cell_features
        for layer in self.cell_linears:
            h = torch.sparse.mm(adjacency, h)
            h = layer(h)
            h = torch.relu(h)
            h = self.dropout(h)
        return h

    def forward_all_species(
        self,
        checklist_features: torch.Tensor,
        cell_embeddings: torch.Tensor,
        checklist_cell: torch.Tensor,
    ) -> torch.Tensor:
        cell_context = cell_embeddings[checklist_cell]
        if self.gnn_mode == "concat":
            model_input = torch.cat([checklist_features, cell_context], dim=1)
        else:
            model_input = checklist_features
        latent = self.checklist_encoder(model_input)
        logits = latent @ self.species_embedding.weight.T / self.scale
        logits = logits + self.species_bias + self.checklist_bias(latent)
        logits = logits + self.direct_head(latent)
        if self.gnn_mode in {"residual", "gated"}:
            if self.cell_residual_head is None:
                raise RuntimeError("Residual head is not initialized.")
            residual_logits = self.cell_residual_head(cell_context)
            if self.gnn_mode == "gated":
                if self.gate_head is None:
                    raise RuntimeError("Gate head is not initialized.")
                gate_input = torch.cat([latent, cell_context], dim=1)
                residual_logits = torch.sigmoid(self.gate_head(gate_input)) * residual_logits
            logits = logits + residual_logits
        return logits


def evaluate_all_pairs(
    model: SpatialGCNHybrid,
    features: torch.Tensor,
    cell_features: torch.Tensor,
    adjacency: torch.Tensor,
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
        for start in range(0, len(checklist_indices), batch_size):
            batch = checklist_indices[start : start + batch_size]
            checklist_tensor = torch.from_numpy(batch.astype(np.int64))
            logits = model.forward_all_species(
                features[checklist_tensor],
                cell_embeddings,
                checklist_cells[checklist_tensor],
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
        Path(args.output_dir) if args.output_dir else graph_dir / "spatial_gnn_baselines"
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
    cell_features, adjacency, checklist_cell_np, cell_metadata = build_spatial_cell_graph(
        graph_dir,
        features_np,
        train_checklists,
        args.spatial_grid_size_m,
    )
    checklist_cells = torch.from_numpy(checklist_cell_np.astype(np.int64))

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
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
            logits = model.forward_all_species(
                features[checklist_index],
                cell_embeddings,
                checklist_cells[checklist_index],
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
        cell_features,
        adjacency,
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
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "latent_dim": args.latent_dim,
        "cell_hidden_dim": args.cell_hidden_dim,
        "cell_layers": args.cell_layers,
        "dropout": args.dropout,
        "gate_init_bias": args.gate_init_bias,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "spatial_grid_size_m": args.spatial_grid_size_m,
        **cell_metadata,
    }
    summary["train"] = {
        "checklists": int(len(train_checklists)),
        "pairs": int(train_labels.size),
        "positives": int(train_labels.sum()),
        "observed_rate": float(train_labels.mean()),
    }

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
