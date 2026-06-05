"""
Evaluate the graph link baseline on the all-pairs tabular target.

This scores every held-out checklist crossed with every modeled species, so the
resulting AUROC/AUPRC are comparable to the tabular all-pairs baselines.

Run from the project root after training the graph link baseline:

    python exp/evaluate_ebird_graph_all_pairs.py --graph-dir data/ebird/graph_top100_spatial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ebird_graph_link_baseline import LinkBaseline
from ebird_joint_tabular_baseline import auc_roc, average_precision


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial"
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate graph link baseline on all held-out checklist/species pairs."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--link-output-dir",
        default=None,
        help="Graph link output directory. Defaults to graph-dir/link_baselines.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Checklist split to evaluate. Defaults to test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Checklist batch size per species. Defaults to 65,536.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Predicted-probability bins for calibration. Defaults to 10.",
    )
    return parser.parse_args()


def load_model(
    graph_dir: Path,
    output_dir: Path,
    feature_dim: int,
    species_count: int,
) -> LinkBaseline:
    metrics_path = output_dir / "species_embedding_link_metrics.json"
    model_path = output_dir / "species_embedding_link_model.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing link metrics JSON: {metrics_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing link model state dict: {model_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_config = metrics["model"]
    model = LinkBaseline(
        feature_dim=feature_dim,
        species_count=species_count,
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        hidden_layers=int(model_config["hidden_layers"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def logit(probability: float) -> float:
    clipped = min(max(probability, EPS), 1.0 - EPS)
    return float(np.log(clipped / (1.0 - clipped)))


def score_species(
    model: LinkBaseline,
    features: torch.Tensor,
    checklist_indices: np.ndarray,
    species_index: int,
    batch_size: int,
) -> np.ndarray:
    logits_parts = []
    species_tensor = None
    with torch.no_grad():
        for start in range(0, len(checklist_indices), batch_size):
            batch = checklist_indices[start : start + batch_size]
            checklist_tensor = torch.from_numpy(batch.astype(np.int64))
            if species_tensor is None or len(species_tensor) != len(batch):
                species_tensor = torch.full(
                    (len(batch),), species_index, dtype=torch.int64
                )
            logits = model(features[checklist_tensor], species_tensor)
            logits_parts.append(logits.cpu().numpy())
    return np.concatenate(logits_parts)


def species_metric_row(
    species_row: pd.Series,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict:
    return {
        "species_index": int(species_row.species_index),
        "species_key": species_row.species_key,
        "common_name": species_row.common_name,
        "scientific_name": species_row.scientific_name,
        "pairs": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
        "observed_rate": float(labels.mean()),
        "mean_predicted": float(scores.mean()),
        "calibration_error": float(abs(scores.mean() - labels.mean())),
        "auroc": auc_roc(labels, scores),
        "auprc": average_precision(labels, scores),
    }


def calibration_table(scores: np.ndarray, labels: np.ndarray, bins: int) -> pd.DataFrame:
    if bins <= 0:
        raise ValueError("--calibration-bins must be positive.")
    frame = pd.DataFrame({"score": scores, "label": labels})
    frame["stratum"] = pd.cut(
        frame["score"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        duplicates="drop",
    ).astype(str)
    rows = []
    for stratum, group in frame.groupby("stratum", sort=True, observed=False):
        mean_predicted = float(group["score"].mean())
        observed_rate = float(group["label"].mean())
        rows.append(
            {
                "calibration_type": "predicted_probability_bin",
                "stratum": stratum,
                "pairs": int(len(group)),
                "pair_fraction": float(len(group) / len(frame)),
                "mean_predicted": mean_predicted,
                "observed_rate": observed_rate,
                "calibration_error": abs(mean_predicted - observed_rate),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    split: str,
    checklist_count: int,
    species_count: int,
    labels: np.ndarray,
    scores: np.ndarray,
    species_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    prior_correction: dict | None = None,
) -> dict:
    ece = float((calibration["pair_fraction"] * calibration["calibration_error"]).sum())
    summary = {
        "split": split,
        "checklists": int(checklist_count),
        "species": species_count,
        "pairs": int(len(labels)),
        "positives": int(labels.sum()),
        "observed_rate": float(labels.mean()),
        "mean_predicted": float(scores.mean()),
        "auroc": auc_roc(labels, scores),
        "auprc": average_precision(labels, scores),
        "species_macro_auroc": float(species_metrics["auroc"].mean()),
        "species_macro_auprc": float(species_metrics["auprc"].mean()),
        "probability_bin_ece": ece,
        "probability_bin_max_error": float(calibration["calibration_error"].max()),
        "species_calibration_mae": float(
            species_metrics["calibration_error"].mean()
        ),
    }
    if prior_correction is not None:
        summary["prior_correction"] = prior_correction
    return summary


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.link_output_dir) if args.link_output_dir else graph_dir / "link_baselines"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    link_metrics = json.loads(
        (output_dir / "species_embedding_link_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    features_np = np.load(graph_dir / "checklist_features.npy").astype(np.float32)
    features = torch.from_numpy(features_np)
    species = pd.read_csv(graph_dir / "species_nodes.csv")
    species_count = int(metadata["counts"]["species"])
    model = load_model(
        graph_dir,
        output_dir,
        feature_dim=features_np.shape[1],
        species_count=species_count,
    )

    mask_column = f"{args.split}_mask"
    checklists = pd.read_parquet(
        graph_dir / "checklist_nodes.parquet",
        columns=["checklist_index", mask_column],
    )
    checklist_indices = (
        checklists.loc[checklists[mask_column], "checklist_index"]
        .to_numpy(dtype=np.int64)
    )
    checklist_position = {
        int(checklist_index): position
        for position, checklist_index in enumerate(checklist_indices)
    }

    positive_edges = pd.read_parquet(
        graph_dir / "positive_edges.parquet",
        columns=["checklist_index", "species_index", "split"],
    )
    positive_edges = positive_edges[positive_edges["split"] == args.split]
    target_prevalence = float(len(positive_edges) / (len(checklist_indices) * species_count))
    train_sample_prevalence = float(link_metrics["train"]["observed_rate"])
    prior_logit_shift = logit(target_prevalence) - logit(train_sample_prevalence)
    prior_correction = {
        "method": "global_case_control_logit_shift",
        "train_sample_prevalence": train_sample_prevalence,
        "target_all_pairs_prevalence": target_prevalence,
        "logit_shift": prior_logit_shift,
    }

    all_raw_scores = []
    all_corrected_scores = []
    all_labels = []
    raw_species_rows = []
    corrected_species_rows = []
    for row in species.itertuples(index=False):
        species_index = int(row.species_index)
        logits = score_species(
            model=model,
            features=features,
            checklist_indices=checklist_indices,
            species_index=species_index,
            batch_size=args.batch_size,
        )
        raw_scores = sigmoid(logits)
        corrected_scores = sigmoid(logits + prior_logit_shift)
        labels = np.zeros(len(checklist_indices), dtype=np.float32)
        species_positives = positive_edges[
            positive_edges["species_index"] == species_index
        ]["checklist_index"]
        positive_positions = [
            checklist_position[int(checklist_index)]
            for checklist_index in species_positives
            if int(checklist_index) in checklist_position
        ]
        labels[np.asarray(positive_positions, dtype=np.int64)] = 1.0

        all_raw_scores.append(raw_scores.astype(np.float32))
        all_corrected_scores.append(corrected_scores.astype(np.float32))
        all_labels.append(labels)
        raw_species_rows.append(species_metric_row(row, labels, raw_scores))
        corrected_species_rows.append(
            species_metric_row(row, labels, corrected_scores)
        )
        if (species_index + 1) % 10 == 0 or species_index + 1 == species_count:
            print(f"scored {species_index + 1:,} of {species_count:,} species")

    raw_score_np = np.concatenate(all_raw_scores)
    corrected_score_np = np.concatenate(all_corrected_scores)
    label_np = np.concatenate(all_labels)
    raw_species_metrics = pd.DataFrame(raw_species_rows)
    corrected_species_metrics = pd.DataFrame(corrected_species_rows)
    raw_calibration = calibration_table(raw_score_np, label_np, args.calibration_bins)
    corrected_calibration = calibration_table(
        corrected_score_np, label_np, args.calibration_bins
    )
    raw_summary = build_summary(
        args.split,
        len(checklist_indices),
        species_count,
        label_np,
        raw_score_np,
        raw_species_metrics,
        raw_calibration,
    )
    corrected_summary = build_summary(
        args.split,
        len(checklist_indices),
        species_count,
        label_np,
        corrected_score_np,
        corrected_species_metrics,
        corrected_calibration,
        prior_correction=prior_correction,
    )

    prefix = f"species_embedding_link_{args.split}_all_pairs"
    raw_species_metrics.to_csv(output_dir / f"{prefix}_species_metrics.csv", index=False)
    raw_calibration.to_csv(output_dir / f"{prefix}_calibration.csv", index=False)
    corrected_species_metrics.to_csv(
        output_dir / f"{prefix}_prior_corrected_species_metrics.csv", index=False
    )
    corrected_calibration.to_csv(
        output_dir / f"{prefix}_prior_corrected_calibration.csv", index=False
    )
    (output_dir / f"{prefix}_summary.json").write_text(
        json.dumps(raw_summary, indent=2), encoding="utf-8"
    )
    (output_dir / f"{prefix}_prior_corrected_summary.json").write_text(
        json.dumps(corrected_summary, indent=2), encoding="utf-8"
    )

    print("\nAll-pairs graph link metrics:")
    print(
        f"raw micro AUROC={raw_summary['auroc']:.4f}, "
        f"raw micro AUPRC={raw_summary['auprc']:.4f}"
    )
    print(
        f"raw macro AUROC={raw_summary['species_macro_auroc']:.4f}, "
        f"raw macro AUPRC={raw_summary['species_macro_auprc']:.4f}"
    )
    print(
        f"raw ECE={raw_summary['probability_bin_ece']:.4f}, "
        f"raw max bin error={raw_summary['probability_bin_max_error']:.4f}, "
        f"raw species calibration MAE={raw_summary['species_calibration_mae']:.4f}"
    )
    print(
        f"prior-corrected ECE={corrected_summary['probability_bin_ece']:.4f}, "
        "prior-corrected max bin error="
        f"{corrected_summary['probability_bin_max_error']:.4f}, "
        "prior-corrected species calibration MAE="
        f"{corrected_summary['species_calibration_mae']:.4f}"
    )
    print(
        "prior correction logit shift="
        f"{prior_correction['logit_shift']:.4f} "
        f"(train prevalence={train_sample_prevalence:.4f}, "
        f"target prevalence={target_prevalence:.4f})"
    )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
