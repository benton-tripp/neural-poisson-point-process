"""
Diagnose spatial GNN behavior by held-out spatial block and species.

This is a framework diagnostic, not a species/location tuning tool. It compares
a saved spatial GNN against a retrained tabular MLP on the same all-pairs test
target within each held-out spatial block and species, then summarizes whether
losses are local block artifacts or broader species-level failures.

Run from the project root:

    python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_residual_scaled_sigmoid010_l2_0p01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ebird_graph_all_species_baseline import build_label_matrix, load_split_checklists
from ebird_joint_tabular_baseline import SEED, auc_roc, average_precision, fit_model
from ebird_spatial_gnn_baseline import (
    SpatialGCNHybrid,
    build_cell_species_support_features,
    build_species_adjacency_for_run,
    build_spatial_cell_graph_for_run,
    inject_frozen_access_embeddings,
    load_frozen_access_embeddings,
)


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial_10x10"
DEFAULT_SPATIAL_RUN = "spatial_gcn_residual_scaled_sigmoid010_l2_0p01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tabular and spatial GNN metrics by block and species."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--spatial-run-name",
        default=DEFAULT_SPATIAL_RUN,
        help=f"Spatial GNN run name without spatial_gnn_ prefix. Defaults to {DEFAULT_SPATIAL_RUN}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to graph-dir/spatial_gnn_baselines/diagnostics/block_species.",
    )
    parser.add_argument(
        "--min-positive-checklists",
        type=int,
        default=25,
        help="Minimum positives for per-block/species AUROC/AUPRC reporting. Defaults to 25.",
    )
    parser.add_argument(
        "--min-negative-checklists",
        type=int,
        default=25,
        help="Minimum negatives for per-block/species AUROC reporting. Defaults to 25.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Tabular MLP epochs. Defaults to 50.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Tabular MLP batch size. Defaults to 8192.")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Tabular MLP learning rate. Defaults to 1e-2.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Tabular MLP weight decay. Defaults to 1e-4.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Tabular MLP hidden width. Defaults to 64.")
    parser.add_argument("--hidden-layers", type=int, default=1, help="Tabular MLP hidden layers. Defaults to 1.")
    parser.add_argument("--dropout", type=float, default=0.10, help="Tabular MLP dropout. Defaults to 0.10.")
    parser.add_argument("--gnn-batch-size", type=int, default=2048, help="Spatial GNN evaluation batch size. Defaults to 2048.")
    parser.add_argument("--top-rows", type=int, default=15, help="Rows to print in gain/loss tables. Defaults to 15.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed. Defaults to 19.")
    return parser.parse_args()


def load_spatial_run(graph_dir: Path, run_name: str) -> tuple[dict, Path]:
    output_dir = graph_dir / "spatial_gnn_baselines"
    prefix = f"spatial_gnn_{run_name}"
    summary_path = output_dir / f"{prefix}_summary.json"
    model_path = output_dir / f"{prefix}_model.pt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing spatial GNN summary: {summary_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing spatial GNN model: {model_path}")
    return json.loads(summary_path.read_text(encoding="utf-8")), model_path


def assign_spatial_blocks(nodes: pd.DataFrame, blocks_per_dim: int) -> np.ndarray:
    x = nodes["x"].to_numpy(dtype=np.float64)
    y = nodes["y"].to_numpy(dtype=np.float64)
    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    if x_span <= 0 or y_span <= 0:
        raise ValueError("Spatial block assignment requires non-degenerate coordinates.")
    x_bin = np.floor((x - x.min()) / x_span * blocks_per_dim).astype(np.int64)
    y_bin = np.floor((y - y.min()) / y_span * blocks_per_dim).astype(np.int64)
    x_bin = np.clip(x_bin, 0, blocks_per_dim - 1)
    y_bin = np.clip(y_bin, 0, blocks_per_dim - 1)
    return y_bin * blocks_per_dim + x_bin


def load_spatial_probabilities(
    graph_dir: Path,
    run_name: str,
    features_np: np.ndarray,
    train_checklists: np.ndarray,
    test_checklists: np.ndarray,
    species_count: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    summary, model_path = load_spatial_run(graph_dir, run_name)
    model_info = summary["model"]
    cell_features, adjacency, checklist_cell, _cell_metadata = build_spatial_cell_graph_for_run(
        graph_dir,
        features_np,
        train_checklists,
        model_info,
    )
    frozen_access_embeddings = load_frozen_access_embeddings(
        model_info.get("frozen_access_embeddings"),
        cell_features.shape[0],
    )
    train_labels = build_label_matrix(
        graph_dir, train_checklists, species_count, "train"
    )
    species_adjacency, _species_graph_metadata = build_species_adjacency_for_run(
        train_labels, model_info
    )
    cell_species_support_features = (
        build_cell_species_support_features(
            checklist_cell,
            train_checklists,
            train_labels,
            cell_features.shape[0],
        )
        if model_info.get("support_aware_residual", "none") == "species-cell"
        else None
    )
    model = SpatialGCNHybrid(
        checklist_feature_dim=features_np.shape[1],
        cell_feature_dim=cell_features.shape[1],
        species_count=species_count,
        hidden_dim=int(model_info["hidden_dim"]),
        hidden_layers=int(model_info["hidden_layers"]),
        latent_dim=int(model_info["latent_dim"]),
        cell_hidden_dim=int(model_info["cell_hidden_dim"]),
        cell_layers=int(model_info["cell_layers"]),
        dropout=float(model_info["dropout"]),
        gnn_mode=model_info["gnn_mode"],
        gate_init_bias=float(model_info["gate_init_bias"]),
        species_residual_scale=model_info.get("species_residual_scale", "none"),
        species_residual_scale_init=float(
            model_info.get("species_residual_scale_init", 0.25)
        ),
        component_mode=model_info.get("component_mode", "joint"),
        ecology_feature_indices=model_info.get("ecology_feature_indices", []),
        bias_feature_indices=model_info.get("bias_feature_indices", []),
        effort_bias_mode=model_info.get("effort_bias_mode", "none"),
        effort_bias_rank=int(model_info.get("effort_bias_rank", 8)),
        spatial_channel_mode=model_info.get("spatial_channel_mode", "single"),
        ecology_cell_feature_indices=model_info.get("ecology_cell_feature_indices", []),
        access_cell_feature_indices=model_info.get("access_cell_feature_indices", []),
        access_density_auxiliary=bool(model_info.get("access_density_auxiliary", False)),
        support_aware_residual=model_info.get("support_aware_residual", "none"),
        support_cell_feature_indices=model_info.get("support_cell_feature_indices", []),
        support_species_feature_dim=len(
            model_info.get("support_species_feature_names", [])
        ),
        support_gate_init_bias=float(model_info.get("support_gate_init_bias", 2.0)),
        species_gcn_layers=int(model_info.get("species_gcn_layers", 0)),
        species_gcn_dropout=float(model_info.get("species_gcn_dropout", 0.0)),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    features = torch.from_numpy(features_np.astype(np.float32))
    checklist_cells = torch.from_numpy(checklist_cell.astype(np.int64))
    base_parts = []
    full_parts = []
    with torch.no_grad():
        cell_embeddings = model.encode_cells(cell_features, adjacency)
        cell_embeddings = inject_frozen_access_embeddings(
            model, cell_embeddings, frozen_access_embeddings
        )
        for start in range(0, len(test_checklists), batch_size):
            batch = test_checklists[start : start + batch_size]
            batch_tensor = torch.from_numpy(batch.astype(np.int64))
            if model.gnn_mode == "concat":
                raise ValueError("Block/species residual diagnostics require residual or gated mode.")
            base_logits, full_logits = model.base_and_full_logits(
                features[batch_tensor],
                cell_embeddings,
                checklist_cells[batch_tensor],
                species_adjacency,
                cell_features=cell_features,
                cell_species_support_features=cell_species_support_features,
            )
            base_parts.append(torch.sigmoid(base_logits).cpu().numpy())
            full_parts.append(torch.sigmoid(full_logits).cpu().numpy())
    return (
        np.vstack(base_parts).astype(np.float32),
        np.vstack(full_parts).astype(np.float32),
        summary,
    )


def model_species_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    return {
        "mean_predicted": float(scores.mean()),
        "calibration_error": float(abs(scores.mean() - labels.mean())),
        "auroc": auc_roc(labels, scores) if positives > 0 and negatives > 0 else np.nan,
        "auprc": average_precision(labels, scores) if positives > 0 else np.nan,
    }


def block_species_rows(
    labels: np.ndarray,
    tabular_scores: np.ndarray,
    base_scores: np.ndarray,
    spatial_scores: np.ndarray,
    blocks: np.ndarray,
    species: pd.DataFrame,
    min_positives: int,
    min_negatives: int,
) -> pd.DataFrame:
    rows = []
    for block_id in sorted(pd.unique(blocks)):
        block_mask = blocks == block_id
        for row in species.itertuples(index=False):
            species_index = int(row.species_index)
            y = labels[block_mask, species_index]
            positives = int(y.sum())
            negatives = int(len(y) - positives)
            if positives < min_positives:
                continue
            tabular = tabular_scores[block_mask, species_index]
            base = base_scores[block_mask, species_index]
            spatial = spatial_scores[block_mask, species_index]
            tabular_metrics = model_species_metrics(y, tabular)
            spatial_metrics = model_species_metrics(y, spatial)
            if negatives < min_negatives:
                tabular_metrics["auroc"] = np.nan
                spatial_metrics["auroc"] = np.nan
            delta = spatial - base
            rows.append(
                {
                    "spatial_block": int(block_id),
                    "species_index": species_index,
                    "species_key": row.species_key,
                    "common_name": row.common_name,
                    "scientific_name": row.scientific_name,
                    "checklists": int(len(y)),
                    "positives": positives,
                    "negatives": negatives,
                    "observed_rate": float(y.mean()),
                    "tabular_auroc": tabular_metrics["auroc"],
                    "spatial_auroc": spatial_metrics["auroc"],
                    "delta_auroc": spatial_metrics["auroc"] - tabular_metrics["auroc"],
                    "tabular_auprc": tabular_metrics["auprc"],
                    "spatial_auprc": spatial_metrics["auprc"],
                    "delta_auprc": spatial_metrics["auprc"] - tabular_metrics["auprc"],
                    "tabular_mean_predicted": tabular_metrics["mean_predicted"],
                    "spatial_mean_predicted": spatial_metrics["mean_predicted"],
                    "delta_mean_predicted": spatial_metrics["mean_predicted"]
                    - tabular_metrics["mean_predicted"],
                    "tabular_calibration_error": tabular_metrics["calibration_error"],
                    "spatial_calibration_error": spatial_metrics["calibration_error"],
                    "delta_calibration_error": spatial_metrics["calibration_error"]
                    - tabular_metrics["calibration_error"],
                    "mean_base_probability": float(base.mean()),
                    "mean_full_probability": float(spatial.mean()),
                    "mean_probability_delta": float(delta.mean()),
                    "mean_abs_probability_delta": float(np.abs(delta).mean()),
                    "positive_mean_delta": float(delta[y.astype(bool)].mean())
                    if positives > 0
                    else np.nan,
                    "negative_mean_delta": float(delta[~y.astype(bool)].mean())
                    if negatives > 0
                    else np.nan,
                    "positive_p90_abs_delta": float(np.quantile(np.abs(delta[y.astype(bool)]), 0.9))
                    if positives > 0
                    else np.nan,
                    "all_p90_abs_delta": float(np.quantile(np.abs(delta), 0.9)),
                }
            )
    return pd.DataFrame(rows)


def block_summary_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block_id, frame in metrics.groupby("spatial_block", sort=True):
        rows.append(
            {
                "spatial_block": int(block_id),
                "species_rows": int(len(frame)),
                "checklists": int(frame["checklists"].max()),
                "mean_delta_auprc": float(frame["delta_auprc"].mean()),
                "median_delta_auprc": float(frame["delta_auprc"].median()),
                "worst_delta_auprc": float(frame["delta_auprc"].min()),
                "best_delta_auprc": float(frame["delta_auprc"].max()),
                "species_with_auprc_loss": int((frame["delta_auprc"] < 0).sum()),
                "species_with_auprc_gain": int((frame["delta_auprc"] > 0).sum()),
                "mean_delta_calibration_error": float(
                    frame["delta_calibration_error"].mean()
                ),
                "mean_positive_delta": float(frame["positive_mean_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_block_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    frame = summary.sort_values("mean_delta_auprc")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(frame["spatial_block"].astype(str), frame["mean_delta_auprc"], color="#4C78A8")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Mean Species Delta AUPRC")
    axes[0].set_xlabel("Held-out spatial block")
    axes[0].set_ylabel("spatial GNN - tabular MLP")
    axes[1].bar(
        frame["spatial_block"].astype(str),
        frame["species_with_auprc_loss"],
        color="#E45756",
    )
    axes[1].set_title("Species With AUPRC Loss")
    axes[1].set_xlabel("Held-out spatial block")
    axes[1].set_ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "block_species_summary.png", dpi=180)
    plt.close(fig)


def safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return safe.strip("_") or "run"


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else graph_dir / "spatial_gnn_baselines" / "diagnostics" / "block_species"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = output_dir / safe_name(args.spatial_run_name)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    species = pd.read_csv(graph_dir / "species_nodes.csv")
    species_count = int(metadata["counts"]["species"])
    features = np.load(graph_dir / "checklist_features.npy").astype(np.float32)
    train_checklists = load_split_checklists(graph_dir, "train", None, args.seed)
    test_checklists = load_split_checklists(graph_dir, "test", None, args.seed)
    y_train = build_label_matrix(graph_dir, train_checklists, species_count, "train")
    y_test = build_label_matrix(graph_dir, test_checklists, species_count, "test")

    tabular_args = SimpleNamespace(
        model="mlp",
        feature_set="both",
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    tabular_scores = fit_model(
        features[train_checklists],
        y_train,
        features[test_checklists],
        tabular_args,
    ).astype(np.float32)
    base_scores, spatial_scores, spatial_summary = load_spatial_probabilities(
        graph_dir,
        args.spatial_run_name,
        features,
        train_checklists,
        test_checklists,
        species_count,
        args.gnn_batch_size,
    )

    node_columns = ["checklist_index", "x", "y"]
    all_nodes = pd.read_parquet(graph_dir / "checklist_nodes.parquet", columns=node_columns)
    blocks_per_dim = int(metadata.get("split", {}).get("spatial_blocks_per_dim", 8))
    all_nodes["spatial_block"] = assign_spatial_blocks(all_nodes, blocks_per_dim)
    test_nodes = all_nodes.set_index("checklist_index").loc[test_checklists].reset_index()
    blocks = test_nodes["spatial_block"].to_numpy(dtype=np.int64)

    metrics = block_species_rows(
        y_test,
        tabular_scores,
        base_scores,
        spatial_scores,
        blocks,
        species,
        args.min_positive_checklists,
        args.min_negative_checklists,
    )
    summary = block_summary_rows(metrics)
    metadata_payload = {
        "graph_dir": str(graph_dir),
        "spatial_run_name": args.spatial_run_name,
        "min_positive_checklists": args.min_positive_checklists,
        "min_negative_checklists": args.min_negative_checklists,
        "spatial_summary": spatial_summary,
        "tabular_mlp": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "run_output_dir": str(run_output_dir),
        "compatibility_output_dir": str(output_dir),
    }
    for target_dir in (output_dir, run_output_dir):
        metrics.to_csv(target_dir / "block_species_metrics.csv", index=False)
        summary.to_csv(target_dir / "block_summary.csv", index=False)
        plot_block_summary(summary, target_dir)
        (target_dir / "block_species_metadata.json").write_text(
            json.dumps(
                metadata_payload,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Wrote block/species diagnostics to {run_output_dir}")
    print(f"Updated compatibility copy at {output_dir}")
    if metrics.empty:
        print("No block/species rows met the minimum support thresholds.")
        return
    columns = [
        "spatial_block",
        "common_name",
        "checklists",
        "positives",
        "observed_rate",
        "delta_auprc",
        "delta_auroc",
        "delta_calibration_error",
        "positive_mean_delta",
        "mean_probability_delta",
    ]
    print("\nLargest block/species AUPRC gains:")
    print(
        metrics.sort_values("delta_auprc", ascending=False)
        [columns]
        .head(args.top_rows)
        .to_string(index=False, float_format="%.4f")
    )
    print("\nLargest block/species AUPRC losses:")
    print(
        metrics.sort_values("delta_auprc")
        [columns]
        .head(args.top_rows)
        .to_string(index=False, float_format="%.4f")
    )
    print("\nBlock summary:")
    print(
        summary.sort_values("mean_delta_auprc")
        .to_string(index=False, float_format="%.4f")
    )


if __name__ == "__main__":
    main()
