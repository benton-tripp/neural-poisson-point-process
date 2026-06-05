"""
Train a non-message-passing graph link baseline on an eBird graph dataset.

This uses checklist node features plus learned species embeddings to classify
sampled species-checklist edges. It deliberately does not perform graph message
passing; it is a bridge baseline between the tabular MLP and a heterogeneous
GNN.

Run from the project root:

    python exp/ebird_graph_link_baseline.py --graph-dir data/ebird/graph_top100_spatial --epochs 10 --train-positive-edges 1000000 --train-negative-edges 1000000 --eval-positive-edges 500000 --eval-negative-edges 500000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ebird_joint_tabular_baseline import SEED, auc_roc, average_precision


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a species-embedding/checklist-feature link baseline."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to graph-dir/link_baselines.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Species embedding dimension. Defaults to 32.",
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
        default=8192,
        help="Mini-batch size. Defaults to 8192.",
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
        "--train-positive-edges",
        type=int,
        default=1_000_000,
        help="Positive train edges to sample. Defaults to 1,000,000.",
    )
    parser.add_argument(
        "--train-negative-edges",
        type=int,
        default=1_000_000,
        help="Negative train edges to sample. Defaults to 1,000,000.",
    )
    parser.add_argument(
        "--eval-positive-edges",
        type=int,
        default=500_000,
        help="Positive test edges to sample. Defaults to 500,000.",
    )
    parser.add_argument(
        "--eval-negative-edges",
        type=int,
        default=500_000,
        help="Negative test edges to sample. Defaults to 500,000.",
    )
    parser.add_argument(
        "--edge-seed",
        type=int,
        default=SEED,
        help="Random seed for edge sampling. Defaults to 19.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Predicted-probability bins for calibration. Defaults to 10.",
    )
    return parser.parse_args()


def trim_sample(
    checklist_parts: list[np.ndarray],
    species_parts: list[np.ndarray],
    priority_parts: list[np.ndarray],
    max_edges: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    checklist = np.concatenate(checklist_parts)
    species = np.concatenate(species_parts)
    priority = np.concatenate(priority_parts)
    if len(priority) > max_edges:
        keep = np.argpartition(priority, max_edges - 1)[:max_edges]
        checklist = checklist[keep]
        species = species[keep]
        priority = priority[keep]
    return [checklist], [species], [priority]


def sample_edges(
    path: Path,
    split: str,
    label: int,
    max_edges: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_edges <= 0:
        raise ValueError("Edge sample sizes must be positive.")

    rng = np.random.default_rng(seed)
    parquet = pq.ParquetFile(path)
    checklist_parts: list[np.ndarray] = []
    species_parts: list[np.ndarray] = []
    priority_parts: list[np.ndarray] = []
    seen = 0

    columns = ["checklist_index", "species_index", "split"]
    for batch in parquet.iter_batches(columns=columns, batch_size=500_000):
        frame = batch.to_pandas()
        frame = frame[frame["split"] == split]
        if frame.empty:
            continue
        count = len(frame)
        seen += count
        checklist_parts.append(frame["checklist_index"].to_numpy(dtype=np.int64))
        species_parts.append(frame["species_index"].to_numpy(dtype=np.int64))
        priority_parts.append(rng.random(count))
        if sum(len(part) for part in priority_parts) > max_edges * 3:
            checklist_parts, species_parts, priority_parts = trim_sample(
                checklist_parts, species_parts, priority_parts, max_edges
            )

    if not priority_parts:
        raise ValueError(f"No {split} edges found in {path}")
    checklist_parts, species_parts, priority_parts = trim_sample(
        checklist_parts, species_parts, priority_parts, max_edges
    )
    checklist = np.concatenate(checklist_parts)
    species = np.concatenate(species_parts)
    labels = np.full(len(checklist), label, dtype=np.float32)
    order = rng.permutation(len(checklist))
    print(f"Sampled {len(checklist):,} of {seen:,} {split} edges from {path.name}")
    return checklist[order], species[order], labels[order]


def load_edge_sample(
    graph_dir: Path,
    split: str,
    positive_count: int,
    negative_count: int,
    seed: int,
) -> TensorDataset:
    pos_checklist, pos_species, pos_labels = sample_edges(
        graph_dir / "positive_edges.parquet",
        split,
        1,
        positive_count,
        seed,
    )
    neg_checklist, neg_species, neg_labels = sample_edges(
        graph_dir / "negative_edges.parquet",
        split,
        0,
        negative_count,
        seed + 1,
    )
    checklist = np.concatenate([pos_checklist, neg_checklist])
    species = np.concatenate([pos_species, neg_species])
    labels = np.concatenate([pos_labels, neg_labels])
    rng = np.random.default_rng(seed + 2)
    order = rng.permutation(len(labels))
    return TensorDataset(
        torch.from_numpy(checklist[order].astype(np.int64)),
        torch.from_numpy(species[order].astype(np.int64)),
        torch.from_numpy(labels[order].astype(np.float32)),
    )


class LinkBaseline(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        species_count: int,
        embedding_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.species_embedding = nn.Embedding(species_count, embedding_dim)
        layers: list[nn.Module] = []
        current_dim = feature_dim + embedding_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, checklist_features: torch.Tensor, species_index: torch.Tensor):
        species = self.species_embedding(species_index)
        return self.net(torch.cat([checklist_features, species], dim=1)).squeeze(1)


def evaluate_model(
    model: nn.Module,
    features: torch.Tensor,
    dataset: TensorDataset,
    batch_size: int,
) -> tuple[dict, pd.DataFrame]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    labels = []
    checklist_indices = []
    species_indices = []
    model.eval()
    with torch.no_grad():
        for checklist_index, species_index, y in loader:
            logits = model(features[checklist_index], species_index)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())
            checklist_indices.append(checklist_index.cpu().numpy())
            species_indices.append(species_index.cpu().numpy())
    score_np = np.concatenate(scores)
    label_np = np.concatenate(labels)
    predictions = pd.DataFrame(
        {
            "checklist_index": np.concatenate(checklist_indices),
            "species_index": np.concatenate(species_indices),
            "label": label_np,
            "score": score_np,
        }
    )
    metrics = {
        "auroc": auc_roc(label_np, score_np),
        "auprc": average_precision(label_np, score_np),
        "mean_score": float(score_np.mean()),
        "observed_rate": float(label_np.mean()),
        "edges": int(len(label_np)),
    }
    return metrics, predictions


def species_metrics(predictions: pd.DataFrame, species: pd.DataFrame) -> pd.DataFrame:
    rows = []
    species_lookup = species.set_index("species_index")
    for species_index, group in predictions.groupby("species_index", sort=True):
        labels = group["label"].to_numpy(dtype=np.float32)
        scores = group["score"].to_numpy(dtype=np.float32)
        info = species_lookup.loc[int(species_index)]
        rows.append(
            {
                "species_index": int(species_index),
                "common_name": info["common_name"],
                "scientific_name": info["scientific_name"],
                "edges": int(len(group)),
                "positives": int(labels.sum()),
                "negatives": int(len(labels) - labels.sum()),
                "observed_rate": float(labels.mean()),
                "mean_predicted": float(scores.mean()),
                "calibration_error": float(abs(scores.mean() - labels.mean())),
                "auroc": auc_roc(labels, scores),
                "auprc": average_precision(labels, scores),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(
    predictions: pd.DataFrame,
    species: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    if bins <= 0:
        raise ValueError("--calibration-bins must be positive.")

    rows = []
    edges_total = len(predictions)
    cut_bins = np.linspace(0.0, 1.0, bins + 1)
    binned = predictions.copy()
    binned["stratum"] = pd.cut(
        binned["score"],
        bins=cut_bins,
        include_lowest=True,
        duplicates="drop",
    ).astype(str)

    for stratum, group in binned.groupby("stratum", sort=True, observed=False):
        mean_predicted = float(group["score"].mean())
        observed_rate = float(group["label"].mean())
        rows.append(
            {
                "calibration_type": "predicted_probability_bin",
                "stratum": stratum,
                "species_index": np.nan,
                "common_name": "",
                "edges": int(len(group)),
                "edge_fraction": float(len(group) / edges_total),
                "mean_predicted": mean_predicted,
                "observed_rate": observed_rate,
                "calibration_error": abs(mean_predicted - observed_rate),
            }
        )

    species_lookup = species.set_index("species_index")
    for species_index, group in predictions.groupby("species_index", sort=True):
        mean_predicted = float(group["score"].mean())
        observed_rate = float(group["label"].mean())
        info = species_lookup.loc[int(species_index)]
        rows.append(
            {
                "calibration_type": "species",
                "stratum": info["common_name"],
                "species_index": int(species_index),
                "common_name": info["common_name"],
                "edges": int(len(group)),
                "edge_fraction": float(len(group) / edges_total),
                "mean_predicted": mean_predicted,
                "observed_rate": observed_rate,
                "calibration_error": abs(mean_predicted - observed_rate),
            }
        )
    return pd.DataFrame(rows)


def summarize_calibration(calibration: pd.DataFrame) -> dict:
    probability_bins = calibration[
        calibration["calibration_type"] == "predicted_probability_bin"
    ].copy()
    species_rows = calibration[calibration["calibration_type"] == "species"].copy()
    probability_ece = float(
        (probability_bins["edge_fraction"] * probability_bins["calibration_error"]).sum()
    )
    return {
        "probability_bin_ece": probability_ece,
        "probability_bin_max_error": float(probability_bins["calibration_error"].max()),
        "species_mean_absolute_error": float(species_rows["calibration_error"].mean()),
        "species_max_absolute_error": float(species_rows["calibration_error"].max()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir) if args.output_dir else graph_dir / "link_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    features_np = np.load(graph_dir / "checklist_features.npy").astype(np.float32)
    features = torch.from_numpy(features_np)
    species_count = int(metadata["counts"]["species"])
    species = pd.read_csv(graph_dir / "species_nodes.csv")

    train_data = load_edge_sample(
        graph_dir,
        "train",
        args.train_positive_edges,
        args.train_negative_edges,
        args.edge_seed,
    )
    test_data = load_edge_sample(
        graph_dir,
        "test",
        args.eval_positive_edges,
        args.eval_negative_edges,
        args.edge_seed + 10,
    )

    model = LinkBaseline(
        feature_dim=features_np.shape[1],
        species_count=species_count,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for checklist_index, species_index, y in train_loader:
            optimizer.zero_grad()
            logits = model(features[checklist_index], species_index)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_loss = float(np.mean(losses))
        row = {"epoch": epoch, "train_bce": train_loss}
        history.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(f"epoch {epoch:>3}: train BCE={train_loss:.5f}")

    train_metrics, train_predictions = evaluate_model(
        model, features, train_data, args.batch_size
    )
    test_metrics, test_predictions = evaluate_model(
        model, features, test_data, args.batch_size
    )
    test_species_metrics = species_metrics(test_predictions, species)
    test_calibration = calibration_table(
        test_predictions,
        species,
        args.calibration_bins,
    )

    metrics = {
        "model": {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "edge_sampling": {
            "train_positive_edges": args.train_positive_edges,
            "train_negative_edges": args.train_negative_edges,
            "eval_positive_edges": args.eval_positive_edges,
            "eval_negative_edges": args.eval_negative_edges,
            "edge_seed": args.edge_seed,
            "calibration_bins": args.calibration_bins,
        },
        "train": train_metrics,
        "test": test_metrics,
        "test_species_macro": {
            "auroc": float(test_species_metrics["auroc"].mean()),
            "auprc": float(test_species_metrics["auprc"].mean()),
            "calibration_error": float(
                test_species_metrics["calibration_error"].mean()
            ),
        },
        "test_calibration": summarize_calibration(test_calibration),
    }

    prefix = "species_embedding_link"
    pd.DataFrame(history).to_csv(output_dir / f"{prefix}_history.csv", index=False)
    test_species_metrics.to_csv(
        output_dir / f"{prefix}_test_species_metrics.csv", index=False
    )
    test_calibration.to_csv(
        output_dir / f"{prefix}_test_calibration.csv", index=False
    )
    (output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), output_dir / f"{prefix}_model.pt")

    print("\nLink baseline metrics:")
    print(
        f"train AUROC={metrics['train']['auroc']:.4f}, "
        f"train AUPRC={metrics['train']['auprc']:.4f}"
    )
    print(
        f"test AUROC={metrics['test']['auroc']:.4f}, "
        f"test AUPRC={metrics['test']['auprc']:.4f}"
    )
    print(
        f"test species macro AUROC={metrics['test_species_macro']['auroc']:.4f}, "
        f"test species macro AUPRC={metrics['test_species_macro']['auprc']:.4f}"
    )
    print(
        "test calibration "
        f"ECE={metrics['test_calibration']['probability_bin_ece']:.4f}, "
        f"max bin error={metrics['test_calibration']['probability_bin_max_error']:.4f}, "
        f"species MAE={metrics['test_calibration']['species_mean_absolute_error']:.4f}"
    )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
