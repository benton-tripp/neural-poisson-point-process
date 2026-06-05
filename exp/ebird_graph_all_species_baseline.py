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
) -> torch.Tensor:
    if hasattr(model, "forward_all_species"):
        return model.forward_all_species(checklist_features)
    checklist_count = checklist_features.shape[0]
    species_index = torch.arange(species_count, dtype=torch.int64)
    species_index = species_index.repeat(checklist_count)
    checklist_expanded = (
        checklist_features[:, None, :]
        .expand(checklist_count, species_count, checklist_features.shape[1])
        .reshape(checklist_count * species_count, checklist_features.shape[1])
    )
    logits = model(checklist_expanded, species_index)
    return logits.reshape(checklist_count, species_count)


def evaluate_all_pairs(
    model: LinkBaseline,
    features: torch.Tensor,
    checklist_indices: np.ndarray,
    labels: np.ndarray,
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

    model = build_model(
        feature_dim=features_np.shape[1],
        species_count=species_count,
        args=args,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    train_dataset = TensorDataset(
        torch.from_numpy(train_checklists.astype(np.int64)),
        torch.from_numpy(train_labels),
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
        for checklist_index, y in train_loader:
            optimizer.zero_grad()
            logits = logits_all_species(
                model,
                features[checklist_index],
                species_count=species_count,
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
    }
    summary["train"] = {
        "checklists": int(len(train_checklists)),
        "pairs": int(train_labels.size),
        "positives": int(train_labels.sum()),
        "observed_rate": float(train_labels.mean()),
    }

    prefix = f"all_species_link_{args.architecture.replace('-', '_')}"
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
